import numpy as np
import pytest

from irt.clustering import ClusterResult, ClusterStatistics
from irt.feature_builder import ResponseRow
from irt.segregation import (
    InsufficientAttemptsError,
    SkippedQuestion,
    classify_discrimination,
    compute_segregation_score,
    compute_segregation_scores,
)


def _make_cluster_result(strong_ids, weak_ids):
    """Hand-build a ClusterResult without running real KMeans — segregation.py
    only depends on ClusterResult's public shape/methods, so this keeps
    the tests fast and focused on segregation logic, not clustering."""
    student_ids = list(strong_ids) + list(weak_ids)
    labels = np.array([0] * len(strong_ids) + [1] * len(weak_ids))
    stats = {
        0: ClusterStatistics(0, len(strong_ids), 0, 0, 0, 0, 0, 0),
        1: ClusterStatistics(1, len(weak_ids), 0, 0, 0, 0, 0, 0),
    }
    return ClusterResult(
        student_ids=student_ids,
        cluster_labels=labels,
        strong_cluster_id=0,
        weak_cluster_id=1,
        cluster_centroids=np.zeros((2, 6)),
        cluster_statistics=stats,
    )


STRONG = ["S1", "S2", "S3", "S4"]
WEAK = ["W1", "W2", "W3", "W4"]


def test_classify_discrimination_boundaries():
    assert classify_discrimination(0.40) == "excellent"
    assert classify_discrimination(0.99) == "excellent"
    assert classify_discrimination(0.39) == "good"
    assert classify_discrimination(0.30) == "good"
    assert classify_discrimination(0.29) == "moderate"
    assert classify_discrimination(0.20) == "moderate"
    assert classify_discrimination(0.19) == "poor"
    assert classify_discrimination(0.0) == "poor"
    assert classify_discrimination(-0.01) == "negative"
    assert classify_discrimination(-1.0) == "negative"


def test_normal_data_produces_correct_segregation_score():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = (
        [ResponseRow(sid, "Q1", True, "Apply") for sid in ["S1", "S2", "S3"]]
        + [ResponseRow("S4", "Q1", False, "Apply")]
        + [ResponseRow(sid, "Q1", False, "Apply") for sid in ["W1", "W2", "W3"]]
        + [ResponseRow("W4", "Q1", True, "Apply")]
    )
    result = compute_segregation_score("Q1", responses, cr)
    assert result.strong_accuracy == pytest.approx(0.75)
    assert result.weak_accuracy == pytest.approx(0.25)
    assert result.segregation_score == pytest.approx(0.5)
    assert result.discriminator_quality == "excellent"
    assert result.is_flagged is False
    assert result.discrimination == result.segregation_score  # alias check


def test_perfect_discriminator():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = [ResponseRow(sid, "Q1", True, "Apply") for sid in STRONG] + [
        ResponseRow(sid, "Q1", False, "Apply") for sid in WEAK
    ]
    result = compute_segregation_score("Q1", responses, cr)
    assert result.strong_accuracy == 1.0
    assert result.weak_accuracy == 0.0
    assert result.segregation_score == 1.0
    assert result.discriminator_quality == "excellent"


def test_poor_discriminator():
    cr = _make_cluster_result(STRONG, WEAK)
    # strong: 2/4 correct = 0.5, weak: 2/4 correct = 0.5 -> score 0.0 -> poor, flagged
    responses = (
        [ResponseRow(sid, "Q1", i < 2, "Apply") for i, sid in enumerate(STRONG)]
        + [ResponseRow(sid, "Q1", i < 2, "Apply") for i, sid in enumerate(WEAK)]
    )
    result = compute_segregation_score("Q1", responses, cr)
    assert result.segregation_score == pytest.approx(0.0)
    assert result.discriminator_quality == "poor"
    assert result.is_flagged is True


def test_negative_discriminator():
    cr = _make_cluster_result(STRONG, WEAK)
    # strong all wrong, weak all correct -> score -1.0
    responses = [ResponseRow(sid, "Q1", False, "Apply") for sid in STRONG] + [
        ResponseRow(sid, "Q1", True, "Apply") for sid in WEAK
    ]
    result = compute_segregation_score("Q1", responses, cr)
    assert result.segregation_score == -1.0
    assert result.discriminator_quality == "negative"
    assert result.is_flagged is True


def test_no_attempts_raises_in_strict_api():
    cr = _make_cluster_result(STRONG, WEAK)
    with pytest.raises(InsufficientAttemptsError):
        compute_segregation_score("Q1", [], cr)


def test_only_strong_attempted_raises_in_strict_api():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = [ResponseRow(sid, "Q1", True, "Apply") for sid in STRONG]
    with pytest.raises(InsufficientAttemptsError):
        compute_segregation_score("Q1", responses, cr)


def test_only_weak_attempted_raises_in_strict_api():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = [ResponseRow(sid, "Q1", True, "Apply") for sid in WEAK]
    with pytest.raises(InsufficientAttemptsError):
        compute_segregation_score("Q1", responses, cr)


# ── Batch API ────────────────────────────────────────────────────────────


def test_batch_normal_dataset_multiple_questions_sorted():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = []
    # Q1: excellent discriminator (strong 1.0, weak 0.0)
    responses += [ResponseRow(sid, "Q1", True, "Apply") for sid in STRONG]
    responses += [ResponseRow(sid, "Q1", False, "Apply") for sid in WEAK]
    # Q2: moderate discriminator (strong 0.75, weak 0.5)
    responses += [ResponseRow(sid, "Q2", i < 3, "Analyze") for i, sid in enumerate(STRONG)]
    responses += [ResponseRow(sid, "Q2", i < 2, "Analyze") for i, sid in enumerate(WEAK)]
    # Q3: negative discriminator
    responses += [ResponseRow(sid, "Q3", False, "Remember") for sid in STRONG]
    responses += [ResponseRow(sid, "Q3", True, "Remember") for sid in WEAK]

    batch = compute_segregation_scores(cr, responses)
    assert len(batch.results) == 3
    ranked = batch.sorted_by_segregation_score()
    assert [r.question_id for r in ranked] == ["Q1", "Q2", "Q3"]
    assert ranked[0].segregation_score > ranked[1].segregation_score > ranked[2].segregation_score


def test_batch_no_attempts_question_is_skipped_not_dropped_silently():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = [ResponseRow(sid, "Q1", True, "Apply") for sid in STRONG] + [
        ResponseRow(sid, "Q1", False, "Apply") for sid in WEAK
    ]
    # Explicitly ask about Q2, which has zero responses -> must show as skipped.
    batch = compute_segregation_scores(cr, responses, question_ids=["Q1", "Q2"])
    assert len(batch.results) == 1
    assert batch.skipped == [SkippedQuestion(question_id="Q2", reason="no_attempts")]
    assert any("Q2" in w for w in batch.warnings())


def test_batch_only_strong_attempted_is_skipped():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = [ResponseRow(sid, "Q1", True, "Apply") for sid in STRONG]
    batch = compute_segregation_scores(cr, responses)
    assert batch.results == []
    assert batch.skipped == [SkippedQuestion(question_id="Q1", reason="only_strong_attempted")]


def test_batch_only_weak_attempted_is_skipped():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = [ResponseRow(sid, "Q1", True, "Apply") for sid in WEAK]
    batch = compute_segregation_scores(cr, responses)
    assert batch.results == []
    assert batch.skipped == [SkippedQuestion(question_id="Q1", reason="only_weak_attempted")]


def test_batch_partial_attempts_still_scores_with_correct_denominator():
    cr = _make_cluster_result(STRONG, WEAK)
    # Only 2 of 4 strong students, and 3 of 4 weak students, attempt Q1.
    responses = [
        ResponseRow("S1", "Q1", True, "Apply"),
        ResponseRow("S2", "Q1", True, "Apply"),
        ResponseRow("W1", "Q1", False, "Apply"),
        ResponseRow("W2", "Q1", False, "Apply"),
        ResponseRow("W3", "Q1", True, "Apply"),
    ]
    batch = compute_segregation_scores(cr, responses)
    result = batch.as_dict_by_question()["Q1"]
    assert result.n_strong_attempted == 2
    assert result.n_weak_attempted == 3
    assert result.strong_accuracy == pytest.approx(1.0)
    assert result.weak_accuracy == pytest.approx(1 / 3)


def test_batch_missing_students_excluded_and_reported():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = (
        [ResponseRow(sid, "Q1", True, "Apply") for sid in STRONG]
        + [ResponseRow(sid, "Q1", False, "Apply") for sid in WEAK]
        + [ResponseRow("GHOST", "Q1", True, "Apply")]  # not in cluster_result
    )
    batch = compute_segregation_scores(cr, responses)
    assert batch.unknown_student_response_count == 1
    assert batch.unknown_student_ids == ["GHOST"]
    # GHOST's response must not have polluted either accuracy.
    result = batch.as_dict_by_question()["Q1"]
    assert result.strong_accuracy == 1.0
    assert result.weak_accuracy == 0.0


def test_batch_empty_responses_and_no_question_ids_returns_empty():
    cr = _make_cluster_result(STRONG, WEAK)
    batch = compute_segregation_scores(cr, [])
    assert batch.results == []
    assert batch.skipped == []


def test_flagged_returns_only_poor_and_negative():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = []
    responses += [ResponseRow(sid, "GOOD", True, "Apply") for sid in STRONG]
    responses += [ResponseRow(sid, "GOOD", False, "Apply") for sid in WEAK]
    responses += [ResponseRow(sid, "BAD", False, "Apply") for sid in STRONG]
    responses += [ResponseRow(sid, "BAD", True, "Apply") for sid in WEAK]
    batch = compute_segregation_scores(cr, responses)
    flagged_ids = {r.question_id for r in batch.flagged()}
    assert flagged_ids == {"BAD"}


def test_as_dict_by_question_gives_o1_lookup_for_theta_py():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = [ResponseRow(sid, "Q1", True, "Apply") for sid in STRONG] + [
        ResponseRow(sid, "Q1", False, "Apply") for sid in WEAK
    ]
    batch = compute_segregation_scores(cr, responses)
    lookup = batch.as_dict_by_question()
    assert "Q1" in lookup
    assert lookup["Q1"].discrimination == 1.0


def test_deterministic_across_repeated_calls():
    cr = _make_cluster_result(STRONG, WEAK)
    responses = [ResponseRow(sid, "Q1", True, "Apply") for sid in STRONG] + [
        ResponseRow(sid, "Q1", False, "Apply") for sid in WEAK
    ]
    b1 = compute_segregation_scores(cr, responses)
    b2 = compute_segregation_scores(cr, responses)
    assert b1.results == b2.results
