import math

import pytest

from irt.item_parameters import QuestionIRTParameters
from irt.theta import (
    AnswerRecord,
    DuplicateParameterError,
    DuplicateResponseError,
    EmptyResponsesError,
    MissingParameterError,
    estimate_theta,
    probability_correct,
)


def _params(spec):
    """spec: list of (question_id, a, b)"""
    return [QuestionIRTParameters(qid, a, b) for qid, a, b in spec]


def _score_at(theta, joined):
    """Recompute the score function directly from a joined (a,b,correct)
    list, independent of theta.py's internals, to verify the returned
    theta is actually near a zero of the score function (the MLE
    condition) rather than just trusting the module's own math."""
    score = 0.0
    for a, b, correct in joined:
        p = probability_correct(a, b, theta)
        y = 1.0 if correct else 0.0
        score += a * (y - p)
    return score


# ── probability_correct ─────────────────────────────────────────────────


def test_probability_correct_at_theta_equals_b_is_one_half():
    assert probability_correct(a=1.0, b=0.5, theta=0.5) == pytest.approx(0.5)


def test_probability_correct_increases_with_theta_for_positive_a():
    low = probability_correct(a=1.0, b=0.0, theta=-2.0)
    high = probability_correct(a=1.0, b=0.0, theta=2.0)
    assert high > low


def test_probability_correct_decreases_with_theta_for_negative_a():
    low = probability_correct(a=-1.0, b=0.0, theta=-2.0)
    high = probability_correct(a=-1.0, b=0.0, theta=2.0)
    assert high < low


def test_probability_correct_never_overflows_on_extreme_inputs():
    # Would overflow a naive exp() without exponent clamping.
    p_hi = probability_correct(a=100.0, b=-100.0, theta=100.0)
    p_lo = probability_correct(a=100.0, b=100.0, theta=-100.0)
    assert 0.0 <= p_hi <= 1.0
    assert 0.0 <= p_lo <= 1.0
    assert not math.isnan(p_hi)
    assert not math.isnan(p_lo)


# ── normal / mixed-performance students ─────────────────────────────────


def test_mixed_performance_student_converges_near_score_zero():
    params = _params([("Q1", 1.0, -1.0), ("Q2", 1.0, 0.0), ("Q3", 1.0, 1.0), ("Q4", 1.0, 2.0)])
    responses = [
        AnswerRecord("Q1", True),
        AnswerRecord("Q2", True),
        AnswerRecord("Q3", False),
        AnswerRecord("Q4", False),
    ]
    result = estimate_theta(responses, params)
    assert result.converged is True
    joined = [(p.discrimination, p.difficulty, r.is_correct) for p, r in zip(params, responses)]
    assert abs(_score_at(result.theta, joined)) < 1e-3


def test_high_ability_student_gets_high_theta():
    params = _params([("Q1", 1.0, -2.0), ("Q2", 1.0, -1.0), ("Q3", 1.0, 1.0), ("Q4", 1.0, 2.0)])
    responses = [AnswerRecord(qid, True) for qid, _, _ in [("Q1", 0, 0), ("Q2", 0, 0)]]
    # 3 correct (incl. hard ones), 1 wrong (hardest)
    responses = [
        AnswerRecord("Q1", True),
        AnswerRecord("Q2", True),
        AnswerRecord("Q3", True),
        AnswerRecord("Q4", False),
    ]
    result = estimate_theta(responses, params)
    assert result.theta > 0.5


def test_low_ability_student_gets_low_theta():
    params = _params([("Q1", 1.0, -2.0), ("Q2", 1.0, -1.0), ("Q3", 1.0, 1.0), ("Q4", 1.0, 2.0)])
    responses = [
        AnswerRecord("Q1", False),
        AnswerRecord("Q2", False),
        AnswerRecord("Q3", False),
        AnswerRecord("Q4", True),
    ]
    result = estimate_theta(responses, params)
    assert result.theta < -0.5


# ── extreme patterns: all correct / all incorrect / single response ─────


def test_all_correct_clamps_to_upper_bound_and_reports_not_converged():
    params = _params([("Q1", 1.0, -1.0), ("Q2", 1.0, 0.0), ("Q3", 1.0, 1.0)])
    responses = [AnswerRecord(q.question_id, True) for q in params]
    result = estimate_theta(responses, params)
    assert result.converged is False
    assert result.theta > 0  # clamped at the positive boundary
    assert result.iterations == 0  # detected up front, no Newton loop needed


def test_all_incorrect_clamps_to_lower_bound_and_reports_not_converged():
    params = _params([("Q1", 1.0, -1.0), ("Q2", 1.0, 0.0), ("Q3", 1.0, 1.0)])
    responses = [AnswerRecord(q.question_id, False) for q in params]
    result = estimate_theta(responses, params)
    assert result.converged is False
    assert result.theta < 0  # clamped at the negative boundary


def test_single_correct_response_behaves_like_all_correct():
    params = _params([("Q1", 1.0, 0.0)])
    responses = [AnswerRecord("Q1", True)]
    result = estimate_theta(responses, params)
    assert result.converged is False
    assert result.theta > 0
    assert result.n_responses == 1


def test_single_incorrect_response_behaves_like_all_incorrect():
    params = _params([("Q1", 1.0, 0.0)])
    responses = [AnswerRecord("Q1", False)]
    result = estimate_theta(responses, params)
    assert result.converged is False
    assert result.theta < 0


# ── validation ───────────────────────────────────────────────────────────


def test_empty_responses_raises():
    params = _params([("Q1", 1.0, 0.0)])
    with pytest.raises(EmptyResponsesError):
        estimate_theta([], params)


def test_duplicate_question_id_in_responses_raises():
    params = _params([("Q1", 1.0, 0.0)])
    responses = [AnswerRecord("Q1", True), AnswerRecord("Q1", False)]
    with pytest.raises(DuplicateResponseError):
        estimate_theta(responses, params)


def test_duplicate_question_id_in_parameters_raises():
    params = _params([("Q1", 1.0, 0.0), ("Q1", 1.2, 0.1)])
    responses = [AnswerRecord("Q1", True)]
    with pytest.raises(DuplicateParameterError):
        estimate_theta(responses, params)


def test_missing_parameter_for_a_response_raises():
    params = _params([("Q1", 1.0, 0.0)])
    responses = [AnswerRecord("Q1", True), AnswerRecord("Q2", False)]  # Q2 has no params
    with pytest.raises(MissingParameterError):
        estimate_theta(responses, params)


# ── determinism ──────────────────────────────────────────────────────────


def test_deterministic_output_across_repeated_calls():
    params = _params([("Q1", 1.0, -1.0), ("Q2", 1.0, 0.0), ("Q3", 1.0, 1.0), ("Q4", 1.0, 2.0)])
    responses = [
        AnswerRecord("Q1", True),
        AnswerRecord("Q2", True),
        AnswerRecord("Q3", False),
        AnswerRecord("Q4", False),
    ]
    r1 = estimate_theta(responses, params)
    r2 = estimate_theta(responses, params)
    assert r1 == r2


# ── numerical stability ────────────────────────────────────────────────


def test_theta_never_exceeds_configured_bounds():
    from irt.config import THETA_MAX, THETA_MIN

    params = _params([("Q1", 5.0, -3.0), ("Q2", 5.0, 3.0)])
    responses = [AnswerRecord("Q1", True), AnswerRecord("Q2", False)]
    result = estimate_theta(responses, params)
    assert THETA_MIN <= result.theta <= THETA_MAX


def test_zero_discrimination_items_do_not_crash_or_produce_nan():
    params = _params([("Q1", 0.0, 0.0), ("Q2", 0.0, 1.0)])
    responses = [AnswerRecord("Q1", True), AnswerRecord("Q2", False)]
    result = estimate_theta(responses, params)
    assert not math.isnan(result.theta)
    assert result.standard_error is None  # zero information -> no defined SE


def test_standard_error_is_positive_when_defined():
    params = _params([("Q1", 1.0, -1.0), ("Q2", 1.0, 0.0), ("Q3", 1.0, 1.0), ("Q4", 1.0, 2.0)])
    responses = [
        AnswerRecord("Q1", True),
        AnswerRecord("Q2", True),
        AnswerRecord("Q3", False),
        AnswerRecord("Q4", False),
    ]
    result = estimate_theta(responses, params)
    assert result.standard_error is not None
    assert result.standard_error > 0


def test_log_likelihood_is_finite_number():
    params = _params([("Q1", 1.0, -1.0), ("Q2", 1.0, 1.0)])
    responses = [AnswerRecord("Q1", True), AnswerRecord("Q2", False)]
    result = estimate_theta(responses, params)
    assert math.isfinite(result.log_likelihood)


def test_negative_discrimination_item_does_not_break_concavity():
    # a negative discrimination question (flagged by segregation.py, but
    # theta.py never refuses to use it — it just does the math correctly).
    params = _params([("Q1", -1.0, 0.0), ("Q2", 1.0, 0.0), ("Q3", 1.0, -1.0)])
    responses = [AnswerRecord("Q1", False), AnswerRecord("Q2", True), AnswerRecord("Q3", True)]
    result = estimate_theta(responses, params)
    assert math.isfinite(result.theta)
    assert not math.isnan(result.theta)
