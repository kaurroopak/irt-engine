import numpy as np
import pytest

from irt.config import FEATURE_VECTOR_FIELDS
from irt.feature_builder import (
    ResponseRow,
    StudentProfileRow,
    build_feature_matrix,
    normalize_feature_matrix,
)


def _profiles():
    return [
        StudentProfileRow(student_id="s1", previous_class_percentage=80.0, iq_score=110.0),
        StudentProfileRow(student_id="s2", previous_class_percentage=60.0, iq_score=None),  # missing IQ
        StudentProfileRow(student_id="s3", previous_class_percentage=None, iq_score=95.0),  # missing pct
    ]


def _responses():
    return [
        # s1: 2 easy (both correct), 1 medium (correct), 1 hard (wrong)
        ResponseRow("s1", "q1", True, "Remember"),
        ResponseRow("s1", "q2", True, "Understand"),
        ResponseRow("s1", "q3", True, "Apply"),
        ResponseRow("s1", "q4", False, "Analyze"),
        # s2: 1 easy (wrong)
        ResponseRow("s2", "q1", False, "Remember"),
        # s3: no responses at all
    ]


def test_feature_vector_field_order_matches_config():
    fm = build_feature_matrix(_profiles(), _responses())
    assert fm.field_names == FEATURE_VECTOR_FIELDS
    assert fm.matrix.shape == (3, len(FEATURE_VECTOR_FIELDS))


def test_previous_class_percentage_used_directly_no_normalization():
    fm = build_feature_matrix(_profiles(), _responses())
    rows = {r["student_id"]: r for r in fm.as_dict_rows()}
    assert rows["s1"]["previous_class_percentage"] == 80.0
    assert rows["s2"]["previous_class_percentage"] == 60.0


def test_missing_iq_score_is_imputed_with_cohort_mean_and_reported():
    fm = build_feature_matrix(_profiles(), _responses())
    rows = {r["student_id"]: r for r in fm.as_dict_rows()}
    cohort_mean = (110.0 + 95.0) / 2  # s1 and s3 have known iq
    assert rows["s2"]["iq_score"] == pytest.approx(cohort_mean)

    iq_reports = [r for r in fm.imputations if r.field_name == "iq_score"]
    assert len(iq_reports) == 1
    assert iq_reports[0].imputed_student_ids == ["s2"]
    assert "s2" in fm.warnings()[0] or any("s2" in w for w in fm.warnings())


def test_missing_previous_class_percentage_is_imputed_and_reported():
    fm = build_feature_matrix(_profiles(), _responses())
    rows = {r["student_id"]: r for r in fm.as_dict_rows()}
    cohort_mean = (80.0 + 60.0) / 2  # s1 and s2 known
    assert rows["s3"]["previous_class_percentage"] == pytest.approx(cohort_mean)

    pct_reports = [r for r in fm.imputations if r.field_name == "previous_class_percentage"]
    assert len(pct_reports) == 1
    assert pct_reports[0].imputed_student_ids == ["s3"]


def test_easy_medium_hard_accuracy_derived_from_bloom_bucket():
    fm = build_feature_matrix(_profiles(), _responses())
    rows = {r["student_id"]: r for r in fm.as_dict_rows()}
    s1 = rows["s1"]
    assert s1["easy_accuracy"] == 1.0     # Remember + Understand, both correct
    assert s1["medium_accuracy"] == 1.0   # Apply, correct
    assert s1["hard_accuracy"] == 0.0     # Analyze, wrong
    assert s1["total_correct"] == 3.0


def test_student_with_no_responses_in_a_bucket_gets_zero_not_error():
    fm = build_feature_matrix(_profiles(), _responses())
    rows = {r["student_id"]: r for r in fm.as_dict_rows()}
    s2 = rows["s2"]
    assert s2["medium_accuracy"] == 0.0
    assert s2["hard_accuracy"] == 0.0


def test_student_with_zero_responses_still_gets_a_row():
    fm = build_feature_matrix(_profiles(), _responses())
    rows = {r["student_id"]: r for r in fm.as_dict_rows()}
    s3 = rows["s3"]
    assert s3["total_correct"] == 0.0
    assert s3["easy_accuracy"] == 0.0
    assert s3["medium_accuracy"] == 0.0
    assert s3["hard_accuracy"] == 0.0


def test_response_for_unknown_student_is_skipped_not_fabricated():
    responses = _responses() + [ResponseRow("ghost", "q1", True, "Remember")]
    fm = build_feature_matrix(_profiles(), responses)
    assert "ghost" not in fm.student_ids
    assert len(fm.student_ids) == 3


def test_empty_profiles_returns_empty_matrix():
    fm = build_feature_matrix([], [])
    assert fm.student_ids == []
    assert fm.matrix.shape == (0, len(FEATURE_VECTOR_FIELDS))


def test_normalize_feature_matrix_zero_mean_unit_std():
    fm = build_feature_matrix(_profiles(), _responses())
    normalized = normalize_feature_matrix(fm)
    assert normalized.matrix.shape == fm.matrix.shape
    means = normalized.matrix.mean(axis=0)
    assert np.allclose(means, 0.0, atol=1e-8)


def test_normalize_handles_zero_variance_column_without_nan():
    profiles = [
        StudentProfileRow(student_id="a", previous_class_percentage=50.0, iq_score=100.0),
        StudentProfileRow(student_id="b", previous_class_percentage=50.0, iq_score=100.0),
    ]
    fm = build_feature_matrix(profiles, [])
    normalized = normalize_feature_matrix(fm)
    assert not np.isnan(normalized.matrix).any()
    assert np.allclose(normalized.matrix, 0.0)


def test_normalize_preserves_student_id_alignment():
    fm = build_feature_matrix(_profiles(), _responses())
    normalized = normalize_feature_matrix(fm)
    assert normalized.student_ids == fm.student_ids
