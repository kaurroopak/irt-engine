from irt.item_parameters import (
    QuestionIRTParameters,
    SkippedQuestionParameters,
    build_question_parameters,
)
from irt.segregation import SegregationBatchResult, SegregationResult


def _seg_result(qid, score, quality="good"):
    return SegregationResult(
        question_id=qid,
        strong_accuracy=0.8,
        weak_accuracy=0.8 - score,
        n_strong_attempted=4,
        n_weak_attempted=4,
        segregation_score=score,
        discriminator_quality=quality,
        is_flagged=False,
    )


def test_builds_parameters_for_fully_specified_questions():
    bloom_levels = {"Q1": "Apply", "Q2": "Analyze"}
    batch = SegregationBatchResult(results=[_seg_result("Q1", 0.5), _seg_result("Q2", 0.3)])
    params, skipped = build_question_parameters(bloom_levels, batch)
    by_id = {p.question_id: p for p in params}
    assert by_id["Q1"] == QuestionIRTParameters("Q1", discrimination=0.5, difficulty=0.0)
    assert by_id["Q2"] == QuestionIRTParameters("Q2", discrimination=0.3, difficulty=1.0)
    assert skipped == []


def test_question_with_no_segregation_score_is_skipped():
    bloom_levels = {"Q1": "Apply", "Q2": "Analyze"}
    batch = SegregationBatchResult(results=[_seg_result("Q1", 0.5)])  # Q2 missing
    params, skipped = build_question_parameters(bloom_levels, batch)
    assert [p.question_id for p in params] == ["Q1"]
    assert skipped == [SkippedQuestionParameters("Q2", reason="no_segregation_score")]


def test_question_with_missing_bloom_level_is_skipped():
    bloom_levels = {"Q1": "Apply"}  # Q2 has a segregation score but no bloom level
    batch = SegregationBatchResult(results=[_seg_result("Q1", 0.5), _seg_result("Q2", 0.3)])
    params, skipped = build_question_parameters(bloom_levels, batch)
    assert [p.question_id for p in params] == ["Q1"]
    assert skipped == [SkippedQuestionParameters("Q2", reason="missing_bloom_level")]


def test_question_with_unknown_bloom_level_is_skipped():
    bloom_levels = {"Q1": "Apply", "Q2": "not_a_real_level"}
    batch = SegregationBatchResult(results=[_seg_result("Q1", 0.5), _seg_result("Q2", 0.3)])
    params, skipped = build_question_parameters(bloom_levels, batch)
    assert [p.question_id for p in params] == ["Q1"]
    assert skipped == [SkippedQuestionParameters("Q2", reason="unknown_bloom_level")]


def test_every_question_id_ends_up_in_exactly_one_output_list():
    bloom_levels = {"Q1": "Apply", "Q2": "Analyze", "Q3": "Remember"}
    batch = SegregationBatchResult(results=[_seg_result("Q1", 0.5), _seg_result("Q3", 0.1)])
    # Q2 has a bloom level but no segregation score; Q3 has both.
    params, skipped = build_question_parameters(bloom_levels, batch)
    covered = {p.question_id for p in params} | {s.question_id for s in skipped}
    assert covered == {"Q1", "Q2", "Q3"}


def test_empty_inputs_produce_empty_outputs():
    params, skipped = build_question_parameters({}, SegregationBatchResult(results=[]))
    assert params == []
    assert skipped == []
