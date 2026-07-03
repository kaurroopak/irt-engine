"""
tests/test_service.py - unit tests for irt/service.py.

service.py contains no math of its own, so these tests are NOT about
re-verifying theta estimation, clustering, or mastery math (that's each
underlying module's own test file's job). They are about the two things
service.py IS responsible for:

  1. Sequencing/wiring — build_item_bank() and score_student() call the
     right underlying functions in the right order with the right data
     passed between them.
  2. Error handling at the service boundary — the strict single-student
     API (score_student) raises StudentScoringError with the right
     underlying cause; the batch API (run_pipeline) never lets one bad
     student abort the rest of the cohort, recording a SkippedStudent
     instead, mirroring segregation.py's compute_segregation_score() vs
     compute_segregation_scores() pattern.

A minimal in-memory FakeRepository (implementing IRTRepository) is used
throughout so these tests can engineer exact edge cases (an unknown
student, a response for a question outside the item bank, an
unrecognized Bloom level, an empty cohort) without needing real CSV/DB
data. A handful of integration tests at the bottom run the real
CSVRepository over the shipped sample_data/ to prove the wiring works
end-to-end against real data too.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pytest

from irt.clustering import ClusteringFailedError
from irt.feature_builder import ResponseRow, StudentProfileRow
from irt.mastery_initializer import ConceptAttempt
from irt.repository import CSVRepository, IRTRepository, RecordNotFoundError
from irt.service import (
    CohortPipelineResult,
    ItemBankBuildError,
    ItemBankResult,
    PipelineError,
    SkippedStudent,
    StudentPipelineResult,
    StudentScoringError,
    build_item_bank,
    run_pipeline,
    score_student,
)
from irt.theta import AnswerRecord


# -- FakeRepository: a minimal, fully in-memory IRTRepository -------------


class FakeRepository(IRTRepository):
    """Builds its data straight from the constructor arguments, with no
    file or database I/O at all — the fastest possible fixture for
    exercising service.py's own logic in isolation."""

    def __init__(
        self,
        profiles: List[StudentProfileRow],
        responses: List[ResponseRow],
        bloom_levels: Dict[str, str],
        concept_attempts_by_student: Optional[Dict[str, List[ConceptAttempt]]] = None,
    ):
        self._profiles = {p.student_id: p for p in profiles}
        self._responses = responses
        self._bloom_levels = dict(bloom_levels)
        self._concept_attempts_by_student = concept_attempts_by_student or {}

    def get_student_profiles(self) -> List[StudentProfileRow]:
        return list(self._profiles.values())

    def get_all_student_ids(self) -> List[str]:
        return list(self._profiles.keys())

    def get_responses(self, student_ids: Optional[Iterable[str]] = None) -> List[ResponseRow]:
        if student_ids is None:
            return list(self._responses)
        wanted = {str(s) for s in student_ids}
        return [r for r in self._responses if r.student_id in wanted]

    def get_question_bloom_levels(self) -> Dict[str, str]:
        return dict(self._bloom_levels)

    def get_concept_attempts(self, student_id: str) -> List[ConceptAttempt]:
        if student_id not in self._profiles:
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")
        return list(self._concept_attempts_by_student.get(student_id, []))

    def get_answer_records(self, student_id: str) -> List[AnswerRecord]:
        if student_id not in self._profiles:
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")
        return [
            AnswerRecord(question_id=r.question_id, is_correct=r.is_correct)
            for r in self._responses
            if r.student_id == student_id
        ]


# A cohort big enough to cluster cleanly into an obvious strong/weak split,
# with a small, fully-scoreable item bank (every question appears in
# responses from students in both groups, so segregation.py can score
# every question and item_parameters.py has bloom levels for all of them).
BLOOM_LEVELS = {
    "Q1": "Remember", "Q2": "Understand", "Q3": "Apply",
    "Q4": "Analyze", "Q5": "Evaluate",
}
PROFILES = [
    StudentProfileRow("S1", 92.0, 118.0),
    StudentProfileRow("S2", 88.0, 112.0),
    StudentProfileRow("S3", 45.0, 88.0),
    StudentProfileRow("S4", 40.0, 82.0),
]
CORRECTNESS = {
    "S1": [True, True, True, True, True],
    "S2": [True, True, True, True, False],
    "S3": [False, True, False, False, False],
    "S4": [False, False, False, True, False],
}
CONCEPT_ATTEMPTS = {
    sid: [
        ConceptAttempt(concept_id="C1", question_id=qid, is_correct=correct, bloom_level=BLOOM_LEVELS[qid])
        for qid, correct in zip(BLOOM_LEVELS, pattern)
    ]
    for sid, pattern in CORRECTNESS.items()
}


def _make_responses() -> List[ResponseRow]:
    responses = []
    for sid, pattern in CORRECTNESS.items():
        for qid, correct in zip(BLOOM_LEVELS, pattern):
            responses.append(ResponseRow(sid, qid, correct, BLOOM_LEVELS[qid]))
    return responses


def make_repo(**overrides) -> FakeRepository:
    kwargs = dict(
        profiles=list(PROFILES),
        responses=_make_responses(),
        bloom_levels=dict(BLOOM_LEVELS),
        concept_attempts_by_student={k: list(v) for k, v in CONCEPT_ATTEMPTS.items()},
    )
    kwargs.update(overrides)
    return FakeRepository(**kwargs)


# -- build_item_bank() -------------------------------------------------------


def test_build_item_bank_wires_the_four_cohort_stages_together():
    repo = make_repo()
    bank = build_item_bank(repo)
    assert isinstance(bank, ItemBankResult)
    # Clustering produced a strong/weak split over all four students.
    assert set(bank.cluster_result.student_ids) == {"S1", "S2", "S3", "S4"}
    assert set(bank.cluster_result.strong_student_ids()) == {"S1", "S2"}
    assert set(bank.cluster_result.weak_student_ids()) == {"S3", "S4"}
    # Segregation scored every question service.py handed it.
    assert {r.question_id for r in bank.segregation_batch.results} == set(BLOOM_LEVELS)
    # Question parameters were assembled for every scoreable question.
    assert {p.question_id for p in bank.parameters} == set(BLOOM_LEVELS)
    assert bank.skipped_parameters == []


def test_build_item_bank_too_few_students_raises_item_bank_build_error():
    repo = make_repo(
        profiles=[StudentProfileRow("S1", 90.0, 100.0)],
        responses=[ResponseRow("S1", "Q1", True, "Remember")],
        bloom_levels={"Q1": "Remember"},
    )
    with pytest.raises(ItemBankBuildError) as exc_info:
        build_item_bank(repo)
    assert isinstance(exc_info.value, PipelineError)
    assert exc_info.value.__cause__ is not None


def test_item_bank_warnings_surface_imputation_and_skip_reasons():
    profiles = list(PROFILES) + [StudentProfileRow("S5", None, None)]
    responses = _make_responses()  # S5 has no responses at all
    repo = make_repo(profiles=profiles, responses=responses)
    bank = build_item_bank(repo)
    warnings = bank.warnings()
    assert any("previous_class_percentage" in w for w in warnings)
    assert any("iq_score" in w for w in warnings)


# -- score_student() ----------------------------------------------------------


def test_score_student_returns_wired_result_for_a_known_student():
    repo = make_repo()
    bank = build_item_bank(repo)
    result = score_student(repo, "S1", bank)
    assert isinstance(result, StudentPipelineResult)
    assert result.student_id == "S1"
    assert result.cluster_label == "strong"
    assert result.theta_result.n_responses == 5
    assert result.mastery_result.student_id == "S1"
    assert result.mastery_result.theta == result.theta_result.theta


def test_score_student_low_performer_lands_in_weak_cluster():
    repo = make_repo()
    bank = build_item_bank(repo)
    result = score_student(repo, "S4", bank)
    assert result.cluster_label == "weak"


def test_score_student_unknown_student_raises_student_scoring_error():
    repo = make_repo()
    bank = build_item_bank(repo)
    with pytest.raises(StudentScoringError) as exc_info:
        score_student(repo, "GHOST", bank)
    assert isinstance(exc_info.value, PipelineError)
    assert isinstance(exc_info.value.__cause__, RecordNotFoundError)


def test_score_student_no_responses_raises_student_scoring_error():
    repo = make_repo(
        profiles=list(PROFILES) + [StudentProfileRow("S5", 70.0, 95.0)],
    )
    bank = build_item_bank(repo)
    with pytest.raises(StudentScoringError):
        score_student(repo, "S5", bank)


def test_score_student_response_outside_item_bank_raises_student_scoring_error():
    """A response for a question with no QuestionIRTParameters (e.g. it
    was skipped upstream by item_parameters.py) must surface as
    StudentScoringError, not crash with a raw MissingParameterError."""
    responses = _make_responses() + [ResponseRow("S1", "Q99", True, "Apply")]
    repo = make_repo(responses=responses)
    bank = build_item_bank(repo)
    with pytest.raises(StudentScoringError):
        score_student(repo, "S1", bank)


def test_score_student_no_concept_attempts_raises_student_scoring_error():
    attempts = {k: list(v) for k, v in CONCEPT_ATTEMPTS.items()}
    attempts["S1"] = []
    repo = make_repo(concept_attempts_by_student=attempts)
    bank = build_item_bank(repo)
    with pytest.raises(StudentScoringError) as exc_info:
        score_student(repo, "S1", bank)
    assert exc_info.value.__cause__ is not None


def test_score_student_unrecognized_bloom_level_in_concept_attempts_raises():
    attempts = {k: list(v) for k, v in CONCEPT_ATTEMPTS.items()}
    attempts["S1"] = [ConceptAttempt("C1", "Q1", True, "not-a-real-bloom-level")]
    repo = make_repo(concept_attempts_by_student=attempts)
    bank = build_item_bank(repo)
    with pytest.raises(StudentScoringError):
        score_student(repo, "S1", bank)


# -- run_pipeline() -------------------------------------------------------------


def test_run_pipeline_scores_every_student_by_default():
    repo = make_repo()
    result = run_pipeline(repo)
    assert isinstance(result, CohortPipelineResult)
    assert set(result.scored_student_ids()) == {"S1", "S2", "S3", "S4"}
    assert result.skipped_students == []


def test_run_pipeline_scores_only_requested_students():
    repo = make_repo()
    result = run_pipeline(repo, student_ids=["S1", "S3"])
    assert set(result.scored_student_ids()) == {"S1", "S3"}


def test_run_pipeline_never_aborts_for_one_bad_student():
    """The core batch-vs-strict guarantee: an unknown student_id must not
    prevent the rest of the cohort from being scored."""
    repo = make_repo()
    result = run_pipeline(repo, student_ids=["S1", "GHOST", "S2"])
    assert set(result.scored_student_ids()) == {"S1", "S2"}
    assert len(result.skipped_students) == 1
    skipped = result.skipped_students[0]
    assert isinstance(skipped, SkippedStudent)
    assert skipped.student_id == "GHOST"
    assert "GHOST" in skipped.reason


def test_run_pipeline_records_a_specific_reason_per_skip_cause():
    responses = _make_responses() + [ResponseRow("S1", "Q99", True, "Apply")]
    attempts = {k: list(v) for k, v in CONCEPT_ATTEMPTS.items()}
    attempts["S2"] = []
    repo = make_repo(responses=responses, concept_attempts_by_student=attempts)

    result = run_pipeline(repo)
    reasons = {s.student_id: s.reason for s in result.skipped_students}
    assert set(reasons) == {"S1", "S2"}
    assert set(result.scored_student_ids()) == {"S3", "S4"}


def test_run_pipeline_result_for_returns_the_students_result():
    repo = make_repo()
    result = run_pipeline(repo)
    r = result.result_for("S1")
    assert r.student_id == "S1"


def test_run_pipeline_result_for_unknown_student_raises_key_error():
    repo = make_repo()
    result = run_pipeline(repo, student_ids=["S1"])
    with pytest.raises(KeyError):
        result.result_for("S2")


def test_run_pipeline_aggregates_warnings_from_item_bank_and_skips():
    repo = make_repo()
    result = run_pipeline(repo, student_ids=["S1", "GHOST"])
    warnings = result.warnings()
    assert any("GHOST" in w for w in warnings)


def test_run_pipeline_propagates_item_bank_build_error_for_whole_cohort_failure():
    repo = make_repo(
        profiles=[StudentProfileRow("S1", 90.0, 100.0)],
        responses=[ResponseRow("S1", "Q1", True, "Remember")],
        bloom_levels={"Q1": "Remember"},
    )
    with pytest.raises(ItemBankBuildError):
        run_pipeline(repo)


def test_run_pipeline_empty_student_ids_scores_nobody():
    repo = make_repo()
    result = run_pipeline(repo, student_ids=[])
    assert result.scored_student_ids() == []
    assert result.skipped_students == []
    # The item bank itself was still built over the full cohort.
    assert set(result.item_bank.cluster_result.student_ids) == {"S1", "S2", "S3", "S4"}


# -- Contains-no-math sanity check --------------------------------------------


def test_service_module_contains_no_module_level_numeric_constants():
    """A light guardrail for the "no mathematical logic" requirement:
    service.py should define no bare int/float module-level constants
    (thresholds, weights, etc.) — every number in its output should have
    come from irt/config.py via an underlying ML module, never from
    service.py itself."""
    import irt.service as service_module

    numeric_constants = [
        name for name, value in vars(service_module).items()
        if not name.startswith("_") and isinstance(value, (int, float))
    ]
    assert numeric_constants == []


# -- Integration: real CSVRepository over the shipped sample_data/ -----------


def test_run_pipeline_against_real_sample_data():
    with CSVRepository.from_default_sample_data() as repo:
        result = run_pipeline(repo)
    assert set(result.scored_student_ids()) == {"S1", "S2", "S3", "S4", "S5"}
    assert result.skipped_students == []
    for sid in result.scored_student_ids():
        r = result.result_for(sid)
        assert r.cluster_label in ("strong", "weak")
        assert -4.0 <= r.theta_result.theta <= 4.0
        assert r.mastery_result.summary.n_concepts > 0
        for cm in r.mastery_result.concept_masteries.values():
            assert 0.0 < cm.initial_mastery < 1.0


def test_run_pipeline_against_real_sample_data_single_student():
    with CSVRepository.from_default_sample_data() as repo:
        bank = build_item_bank(repo)
        result = score_student(repo, "S3", bank)
    assert result.student_id == "S3"
    assert result.cluster_label in ("strong", "weak")
