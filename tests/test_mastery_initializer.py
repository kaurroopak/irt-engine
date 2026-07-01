import pytest

from irt.mastery_initializer import (
    ConceptAttempt,
    DuplicateConceptAttemptError,
    EmptyConceptDataError,
    InvalidBloomLevelError,
    MissingThetaError,
    initialize_mastery,
)
from irt.theta import ThetaResult


def _theta(value, converged=True, se=0.5):
    return ThetaResult(
        theta=value, iterations=6, converged=converged, log_likelihood=-2.0,
        standard_error=se, n_responses=5,
    )


def test_normal_student_produces_mastery_between_zero_and_one():
    attempts = [
        ConceptAttempt("Ohms_Law", "Q1", True, "Apply"),
        ConceptAttempt("Ohms_Law", "Q2", False, "Analyze"),
        ConceptAttempt("Resistance", "Q3", True, "Remember"),
    ]
    result = initialize_mastery("S1", _theta(0.5), attempts)
    assert result.student_id == "S1"
    assert result.theta == 0.5
    for cm in result.concept_masteries.values():
        assert 0.0 < cm.initial_mastery < 1.0


def test_high_theta_yields_higher_mastery_than_low_theta_for_identical_performance():
    attempts = [
        ConceptAttempt("C1", "Q1", True, "Apply"),
        ConceptAttempt("C1", "Q2", False, "Apply"),
    ]
    high = initialize_mastery("S1", _theta(2.5), attempts)
    low = initialize_mastery("S2", _theta(-2.5), attempts)
    assert high.mastery_for("C1") > low.mastery_for("C1")


def test_higher_concept_accuracy_yields_higher_mastery_for_same_theta():
    strong_attempts = [
        ConceptAttempt("C1", "Q1", True, "Apply"),
        ConceptAttempt("C1", "Q2", True, "Apply"),
        ConceptAttempt("C1", "Q3", True, "Apply"),
        ConceptAttempt("C1", "Q4", True, "Apply"),
    ]
    weak_attempts = [
        ConceptAttempt("C1", "Q1", False, "Apply"),
        ConceptAttempt("C1", "Q2", False, "Apply"),
        ConceptAttempt("C1", "Q3", False, "Apply"),
        ConceptAttempt("C1", "Q4", True, "Apply"),
    ]
    strong = initialize_mastery("S1", _theta(0.0), strong_attempts)
    weak = initialize_mastery("S2", _theta(0.0), weak_attempts)
    assert strong.mastery_for("C1") > weak.mastery_for("C1")


def test_harder_bloom_level_yields_lower_mastery_for_same_theta_and_no_observed_accuracy_difference():
    # Same theta, same single "correct" observation, but different Bloom
    # difficulty -> the theta-implied component should differ, and with a
    # single observation the prior still meaningfully pulls the blend.
    easy_attempts = [ConceptAttempt("C1", "Q1", True, "Remember")]
    hard_attempts = [ConceptAttempt("C1", "Q1", True, "Create")]
    easy = initialize_mastery("S1", _theta(0.0), easy_attempts)
    hard = initialize_mastery("S2", _theta(0.0), hard_attempts)
    assert easy.mastery_for("C1") > hard.mastery_for("C1")


def test_missing_theta_raises():
    attempts = [ConceptAttempt("C1", "Q1", True, "Apply")]
    with pytest.raises(MissingThetaError):
        initialize_mastery("S1", None, attempts)


def test_empty_concept_attempts_raises():
    with pytest.raises(EmptyConceptDataError):
        initialize_mastery("S1", _theta(0.0), [])


def test_duplicate_question_id_raises_even_across_different_concepts():
    attempts = [
        ConceptAttempt("C1", "Q1", True, "Apply"),
        ConceptAttempt("C2", "Q1", False, "Analyze"),  # same question_id, different concept
    ]
    with pytest.raises(DuplicateConceptAttemptError):
        initialize_mastery("S1", _theta(0.0), attempts)


def test_unknown_bloom_level_raises_invalid_bloom_level_error():
    attempts = [ConceptAttempt("C1", "Q1", True, "not_a_real_bloom_level")]
    with pytest.raises(InvalidBloomLevelError):
        initialize_mastery("S1", _theta(0.0), attempts)


def test_deterministic_output_across_repeated_calls():
    attempts = [
        ConceptAttempt("C1", "Q1", True, "Apply"),
        ConceptAttempt("C1", "Q2", False, "Analyze"),
        ConceptAttempt("C2", "Q3", True, "Remember"),
    ]
    r1 = initialize_mastery("S1", _theta(0.7), attempts)
    r2 = initialize_mastery("S1", _theta(0.7), attempts)
    assert r1.concept_masteries == r2.concept_masteries
    assert r1.summary == r2.summary


def test_mastery_is_clamped_within_seed_prior_bounds():
    from irt.config import SEED_PRIOR_MAX, SEED_PRIOR_MIN

    # Extremely high theta + perfect accuracy should still not hit exactly 1.0.
    attempts = [ConceptAttempt("C1", f"Q{i}", True, "Remember") for i in range(10)]
    result = initialize_mastery("S1", _theta(4.0), attempts)
    assert SEED_PRIOR_MIN <= result.mastery_for("C1") <= SEED_PRIOR_MAX

    attempts_low = [ConceptAttempt("C1", f"Q{i}", False, "Create") for i in range(10)]
    result_low = initialize_mastery("S2", _theta(-4.0), attempts_low)
    assert SEED_PRIOR_MIN <= result_low.mastery_for("C1") <= SEED_PRIOR_MAX


def test_summary_reports_correct_concept_count_and_extremes():
    attempts = [
        ConceptAttempt("Easy_Concept", "Q1", True, "Remember"),
        ConceptAttempt("Hard_Concept", "Q2", False, "Create"),
    ]
    result = initialize_mastery("S1", _theta(0.0), attempts)
    assert result.summary.n_concepts == 2
    assert result.summary.highest_mastery_concept_id == "Easy_Concept"
    assert result.summary.lowest_mastery_concept_id == "Hard_Concept"


def test_more_attempts_shift_weight_toward_observed_accuracy():
    # A single correct attempt should be pulled more toward the theta prior
    # than many attempts with the same 100% accuracy, since MASTERY_PRIOR_STRENGTH
    # gives few observations less weight than many.
    one_attempt = [ConceptAttempt("C1", "Q1", True, "Apply")]
    many_attempts = [ConceptAttempt("C1", f"Q{i}", True, "Apply") for i in range(10)]
    theta_val = -1.0  # below-average theta, so the prior pulls DOWN from 1.0
    one = initialize_mastery("S1", _theta(theta_val), one_attempt)
    many = initialize_mastery("S2", _theta(theta_val), many_attempts)
    # many_attempts' 100% observed accuracy should dominate more (higher
    # mastery) than one_attempt's single success, which gets pulled toward
    # the (lower, since theta is below average) theta-implied prior.
    assert many.mastery_for("C1") > one.mastery_for("C1")


def test_concept_mastery_exposes_intermediate_values_for_debugging():
    attempts = [
        ConceptAttempt("C1", "Q1", True, "Apply"),
        ConceptAttempt("C1", "Q2", False, "Apply"),
    ]
    result = initialize_mastery("S1", _theta(0.3), attempts)
    cm = result.concept_masteries["C1"]
    assert cm.n_attempted == 2
    assert cm.n_correct == 1
    assert cm.observed_accuracy == pytest.approx(0.5)
    assert 0.0 < cm.theta_implied_accuracy < 1.0
    assert 0.0 < cm.weight_observed < 1.0
