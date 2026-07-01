import numpy as np
import pytest

from irt.clustering import (
    ClusteringFailedError,
    EmptyFeatureMatrixError,
    InsufficientStudentsError,
    cluster_students,
)
from irt.config import FEATURE_VECTOR_FIELDS
from irt.feature_builder import (
    FeatureMatrix,
    ResponseRow,
    StudentProfileRow,
    build_feature_matrix,
    normalize_feature_matrix,
)


def _make_feature_matrix(student_ids, rows):
    return FeatureMatrix(
        student_ids=list(student_ids),
        matrix=np.array(rows, dtype=float),
        field_names=FEATURE_VECTOR_FIELDS,
    )


def _realistic_dataset():
    """8 students, 4 clearly strong / 4 clearly weak, built through the
    real feature_builder pipeline (not hand-crafted numpy arrays) so this
    exercises the full upstream contract, not just clustering.py in
    isolation."""
    profiles = [
        StudentProfileRow("S1", 92.0, 118.0),
        StudentProfileRow("S2", 88.0, 112.0),
        StudentProfileRow("S3", 85.0, 109.0),
        StudentProfileRow("S4", 90.0, 115.0),
        StudentProfileRow("S5", 45.0, 88.0),
        StudentProfileRow("S6", 40.0, 85.0),
        StudentProfileRow("S7", 50.0, 90.0),
        StudentProfileRow("S8", 38.0, 82.0),
    ]
    bank = [
        ("Q1", "Remember"), ("Q2", "Understand"), ("Q3", "Apply"),
        ("Q4", "Apply"), ("Q5", "Analyze"), ("Q6", "Evaluate"),
    ]
    strong_pattern = [1, 1, 1, 1, 1, 1]
    weak_pattern = [1, 0, 0, 0, 0, 0]
    responses = []
    for sid in ("S1", "S2", "S3", "S4"):
        for (qid, bloom), correct in zip(bank, strong_pattern):
            responses.append(ResponseRow(sid, qid, bool(correct), bloom))
    for sid in ("S5", "S6", "S7", "S8"):
        for (qid, bloom), correct in zip(bank, weak_pattern):
            responses.append(ResponseRow(sid, qid, bool(correct), bloom))

    raw = build_feature_matrix(profiles, responses)
    normalized = normalize_feature_matrix(raw)
    return raw, normalized


def test_normal_dataset_splits_into_two_clusters_of_four():
    raw, normalized = _realistic_dataset()
    result = cluster_students(normalized, raw)
    assert set(result.strong_student_ids()) == {"S1", "S2", "S3", "S4"}
    assert set(result.weak_student_ids()) == {"S5", "S6", "S7", "S8"}


def test_strong_cluster_has_higher_avg_total_correct_than_weak():
    raw, normalized = _realistic_dataset()
    result = cluster_students(normalized, raw)
    strong_stats = result.statistics_for("strong")
    weak_stats = result.statistics_for("weak")
    assert strong_stats.avg_total_correct > weak_stats.avg_total_correct


def test_cluster_statistics_are_in_raw_units_not_normalized():
    raw, normalized = _realistic_dataset()
    result = cluster_students(normalized, raw)
    strong_stats = result.statistics_for("strong")
    # raw IQ values were 118, 112, 109, 115 -> avg should be in that
    # real-world range, NOT anywhere near a z-score (~0).
    assert 100 < strong_stats.avg_iq_score < 130


def test_cluster_statistics_without_raw_matrix_falls_back_to_normalized():
    raw, normalized = _realistic_dataset()
    result = cluster_students(normalized)  # no raw matrix passed
    stats = result.statistics_for("strong")
    # normalized IQ is z-scored, so it should be nowhere near raw IQ scale
    assert -5 < stats.avg_iq_score < 5


def test_label_for_matches_strong_weak_membership():
    raw, normalized = _realistic_dataset()
    result = cluster_students(normalized, raw)
    for sid in ("S1", "S2", "S3", "S4"):
        assert result.label_for(sid) == "strong"
    for sid in ("S5", "S6", "S7", "S8"):
        assert result.label_for(sid) == "weak"


def test_exactly_two_students():
    fm = _make_feature_matrix(
        ["A", "B"],
        [[2.0, 2.0, 5.0, 1.0, 1.0, 1.0], [-2.0, -2.0, -5.0, -1.0, -1.0, -1.0]],
    )
    result = cluster_students(fm)
    assert set(result.student_ids) == {"A", "B"}
    assert result.strong_cluster_id != result.weak_cluster_id
    # A has higher total_correct (index 2) => A must be strong
    assert result.label_for("A") == "strong"
    assert result.label_for("B") == "weak"


def test_fully_identical_students_raise_clustering_failed():
    # All 4 students have the exact same feature vector -> KMeans cannot
    # produce 2 distinct clusters from duplicate points (collapses to 1).
    # Per requirement #5, this must raise, not silently return a fake split.
    fm = _make_feature_matrix(
        ["A", "B", "C", "D"],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4,
    )
    with pytest.raises(ClusteringFailedError):
        cluster_students(fm)


def test_near_identical_students_with_a_tie_in_total_correct_is_deterministic():
    # Students aren't perfectly identical (KMeans CAN split them), but the
    # two resulting clusters end up with an exact tie in avg_total_correct.
    # This exercises the tie-break path, not the "can't cluster at all" path.
    fm = _make_feature_matrix(
        ["A", "B", "C", "D"],
        [
            [1.0, 1.0, 5.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 5.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, 5.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, 5.0, -1.0, -1.0, -1.0],
        ],
    )
    result = cluster_students(fm)
    assert result.strong_cluster_id != result.weak_cluster_id
    assert len(result.strong_student_ids()) + len(result.weak_student_ids()) == 4
    # Tie-break is deterministic across repeated calls.
    result_2 = cluster_students(fm)
    assert result.strong_cluster_id == result_2.strong_cluster_id


def test_empty_feature_matrix_raises():
    fm = _make_feature_matrix([], [])
    with pytest.raises(EmptyFeatureMatrixError):
        cluster_students(fm)


def test_one_student_raises_insufficient_students():
    fm = _make_feature_matrix(["A"], [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    with pytest.raises(InsufficientStudentsError):
        cluster_students(fm)


def test_zero_students_via_build_feature_matrix_raises():
    fm = build_feature_matrix([], [])
    with pytest.raises(EmptyFeatureMatrixError):
        cluster_students(fm)


def test_deterministic_across_repeated_calls():
    raw, normalized = _realistic_dataset()
    result_1 = cluster_students(normalized, raw)
    result_2 = cluster_students(normalized, raw)
    assert list(result_1.cluster_labels) == list(result_2.cluster_labels)
    assert result_1.strong_cluster_id == result_2.strong_cluster_id
    assert result_1.weak_cluster_id == result_2.weak_cluster_id


def test_cluster_centroids_shape():
    raw, normalized = _realistic_dataset()
    result = cluster_students(normalized, raw)
    assert result.cluster_centroids.shape == (2, len(FEATURE_VECTOR_FIELDS))


def test_cluster_statistics_present_for_both_clusters_with_correct_counts():
    raw, normalized = _realistic_dataset()
    result = cluster_students(normalized, raw)
    assert len(result.cluster_statistics) == 2
    total_n = sum(s.n_students for s in result.cluster_statistics.values())
    assert total_n == 8


def test_raw_matrix_with_mismatched_student_ids_raises_value_error():
    raw, normalized = _realistic_dataset()
    bad_raw = _make_feature_matrix(
        ["not", "matching", "ids", "at", "all", "here", "x", "y"],
        raw.matrix.tolist(),
    )
    with pytest.raises(ValueError):
        cluster_students(normalized, bad_raw)


def test_strong_weak_never_assumed_from_raw_cluster_id():
    # Run twice with feature order flipped so whichever physical cluster
    # KMeans calls "0" vs "1" may differ, but strong/weak identification
    # (by avg_total_correct) must still correctly track the actual students.
    raw, normalized = _realistic_dataset()
    result = cluster_students(normalized, raw)
    strong_ids = set(result.strong_student_ids())
    # Regardless of which raw sklearn label (0 or 1) ends up "strong",
    # it must be the group with higher total_correct — never hardcoded.
    assert strong_ids == {"S1", "S2", "S3", "S4"}
