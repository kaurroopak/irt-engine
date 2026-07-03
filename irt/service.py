"""
service.py — orchestrates the complete Hybrid IRT pipeline end-to-end:

    Feature Builder -> Clustering -> Segregation -> Question Parameters
    -> Theta -> Mastery Initializer

Responsibility
--------------
This module contains NO mathematical or statistical logic of its own —
every number anywhere in its output was computed by one of the modules
under irt/ (feature_builder.py, clustering.py, segregation.py,
item_parameters.py, theta.py, mastery_initializer.py). service.py's only
job is to:

  1. Pull data from an IRTRepository (repository.py — the only module
     allowed to know where that data actually comes from).
  2. Call the six pipeline stages in the documented order, wiring each
     stage's output into the next stage's input exactly as
     docs/ARCHITECTURE.md's pipeline diagram describes.
  3. Assemble the results into plain dataclasses a caller (a future
     CLI, or the Quiz Portal once it's wired up) can consume directly.
  4. Decide what to do when one student's data can't be scored — without
     ever computing a number itself.

Nowhere in this file is there a formula, a threshold, a probability, or a
statistical decision. If a bug ever produces a wrong theta or a wrong
mastery value, the fix belongs in theta.py or mastery_initializer.py, not
here — this module can only be wrong about *sequencing* or *data
plumbing*, never about the math.

Two-phase design: item bank vs. per-student scoring
-----------------------------------------------------
The first four pipeline stages (Feature Builder, Clustering, Segregation,
Question Parameters) are COHORT-level: they need every student's profile
and every response to produce one shared ClusterResult and one shared
list of QuestionIRTParameters for the whole item bank. The last two
stages (Theta, Mastery Initializer) are STUDENT-level: they run once per
student, reusing that same item bank.

Recomputing the cohort-level stages for every single student would be
both wasteful and *wrong* — clustering.py's "strong vs weak" split is
only meaningful when computed once over the whole cohort — so this
module splits the pipeline into two public entry points that mirror that
distinction:

    build_item_bank(repo)                 -> ItemBankResult   (run once)
    score_student(repo, student_id, bank)  -> StudentPipelineResult  (run per student)
    run_pipeline(repo, student_ids=None)   -> CohortPipelineResult   (both, together)

Never-silently-continue, mirrored from segregation.py's own convention
-------------------------------------------------------------------------
segregation.py already establishes this exact pattern for one pipeline
stage: compute_segregation_score() (singular) raises
InsufficientAttemptsError for one bad question, while
compute_segregation_scores() (batch) never raises for a single bad
question — it records a SkippedQuestion with a reason instead, so one
unscoreable item doesn't abort scoring for the rest of the item bank.

This module applies the identical pattern one level up, across students
instead of questions:

    score_student()   — strict, single-student API. Raises
                         StudentScoringError if that student's data can't
                         be scored (missing responses, an unrecognized
                         Bloom level, a question outside the item bank,
                         etc.), wrapping whichever underlying ML-module
                         exception was the actual cause.
    run_pipeline()     — batch API. Never raises for one bad student —
                         it calls score_student() per student, catches
                         StudentScoringError, and records a
                         SkippedStudent(student_id, reason) instead of
                         aborting the rest of the cohort.

How this fits into the Hybrid IRT architecture
---------------------------------------------------------------------
    repository.IRTRepository (CSVRepository / PostgresRepository)
        -> service.build_item_bank()
            -> feature_builder.build_feature_matrix()
            -> feature_builder.normalize_feature_matrix()
            -> clustering.cluster_students()
            -> segregation.compute_segregation_scores()
            -> item_parameters.build_question_parameters()
            -> ItemBankResult
        -> service.score_student(student_id, item_bank)
            -> theta.estimate_theta()
            -> mastery_initializer.initialize_mastery()
            -> StudentPipelineResult
        -> service.run_pipeline(student_ids=None)
            -> CohortPipelineResult (both of the above, for every student)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .clustering import (
    ClusteringFailedError,
    ClusterResult,
    EmptyFeatureMatrixError,
    InsufficientStudentsError,
    cluster_students,
)
from .feature_builder import (
    FeatureMatrix,
    build_feature_matrix,
    normalize_feature_matrix,
)
from .item_parameters import (
    QuestionIRTParameters,
    SkippedQuestionParameters,
    build_question_parameters,
)
from .mastery_initializer import (
    DuplicateConceptAttemptError,
    EmptyConceptDataError,
    InvalidBloomLevelError,
    MasteryInitializationResult,
    MissingThetaError,
    initialize_mastery,
)
from .repository import IRTRepository, RecordNotFoundError
from .segregation import SegregationBatchResult, compute_segregation_scores
from .theta import (
    DuplicateResponseError,
    EmptyResponsesError,
    MissingParameterError,
    ThetaResult,
    estimate_theta,
)


# ── Exceptions ───────────────────────────────────────────────────────────
# Mirrors this codebase's "never silently continue" convention (see
# docs/ARCHITECTURE.md, principle 1): every failure mode gets its own
# specific, documented exception type, at the service boundary just as
# each underlying module already does at its own boundary.


class PipelineError(Exception):
    """Base class for every exception raised by this module. Callers who
    don't care about the distinction between the specific error types
    below can catch this one type and know they've caught anything
    service.py can raise."""


class ItemBankBuildError(PipelineError):
    """Raised when build_item_bank() cannot produce a usable item bank at
    all — e.g. fewer students than clustering.py's required minimum, or
    scikit-learn itself failing. Wraps the underlying exception from
    feature_builder.py or clustering.py so callers get one exception type
    to catch regardless of which cohort-level stage failed."""


class StudentScoringError(PipelineError):
    """Raised by score_student() (the strict, single-student API) when
    one student's data can't be scored: no responses recorded, a
    response for a question outside the item bank, an unrecognized Bloom
    level in their concept attempts, or the student_id simply isn't
    known to the repository. Wraps the underlying exception from
    repository.py, theta.py, or mastery_initializer.py. Use
    run_pipeline() instead of catching this directly if one bad student
    should not abort scoring for the rest of the cohort."""


# ── Cohort-level result ──────────────────────────────────────────────────


@dataclass
class ItemBankResult:
    """Output of build_item_bank(): everything the STUDENT-level stages
    (theta, mastery_initializer) need, computed once for the whole
    cohort. Every field here is itself a dataclass already defined by an
    upstream ML module — this dataclass only bundles them together for a
    single return value.
    """

    feature_matrix: FeatureMatrix  # raw (non-normalized); kept for debugging/reporting
    cluster_result: ClusterResult
    segregation_batch: SegregationBatchResult
    parameters: List[QuestionIRTParameters]
    skipped_parameters: List[SkippedQuestionParameters]

    def warnings(self) -> List[str]:
        """Every non-fatal issue surfaced while building the item bank:
        cohort-mean imputations (feature_builder.py), skipped/flagged
        questions (segregation.py), and questions that couldn't get IRT
        parameters assembled (item_parameters.py) — collected here so a
        caller can log/display them in one place without re-querying
        three different sub-results."""
        msgs = list(self.feature_matrix.warnings())
        msgs.extend(self.segregation_batch.warnings())
        for s in self.skipped_parameters:
            msgs.append(f"Question {s.question_id} has no IRT parameters: {s.reason}")
        return msgs


# ── Student-level result ─────────────────────────────────────────────────


@dataclass(frozen=True)
class StudentPipelineResult:
    """Output of score_student(): one student's full pipeline result,
    from cluster membership through initial per-concept mastery."""

    student_id: str
    cluster_label: str  # "strong" or "weak" (clustering.ClusterResult.label_for)
    theta_result: ThetaResult
    mastery_result: MasteryInitializationResult


# ── Batch (cohort) result ────────────────────────────────────────────────


@dataclass(frozen=True)
class SkippedStudent:
    """A student who could not be scored, with why — mirrors
    segregation.SkippedQuestion's and item_parameters.SkippedQuestionParameters'
    never-silently-drop pattern, one level up (students instead of
    questions)."""

    student_id: str
    reason: str


@dataclass
class CohortPipelineResult:
    """Output of run_pipeline(): the complete Hybrid IRT pipeline result
    for a cohort of students. Every student requested ends up in either
    `.student_results` (scored) or `.skipped_students` (with a reason),
    never silently absent from both.
    """

    item_bank: ItemBankResult
    student_results: Dict[str, StudentPipelineResult] = field(default_factory=dict)
    skipped_students: List[SkippedStudent] = field(default_factory=list)

    def result_for(self, student_id: str) -> StudentPipelineResult:
        """Look up one student's result by id. Raises KeyError (via plain
        dict lookup) if that student was skipped or never requested —
        check `.skipped_students` first if a lookup might fail."""
        return self.student_results[student_id]

    def scored_student_ids(self) -> List[str]:
        return list(self.student_results.keys())

    def warnings(self) -> List[str]:
        """Every non-fatal issue from the whole pipeline run: item-bank
        warnings (imputations, flagged/skipped questions) plus one
        message per skipped student."""
        msgs = list(self.item_bank.warnings())
        for s in self.skipped_students:
            msgs.append(f"Student {s.student_id} skipped: {s.reason}")
        return msgs


# ── Orchestration ─────────────────────────────────────────────────────────


def build_item_bank(repo: IRTRepository) -> ItemBankResult:
    """Run the four COHORT-level pipeline stages once, over every student
    and response the repository has:

        Feature Builder -> Clustering -> Segregation -> Question Parameters

    This is pure orchestration: it fetches data via `repo` and passes it
    straight through to feature_builder.py, clustering.py, segregation.py,
    and item_parameters.py in the order docs/ARCHITECTURE.md documents,
    with no transformation of its own beyond wiring one call's output
    into the next call's input.

    Raises
    ------
    ItemBankBuildError
        if clustering.py cannot produce a strong/weak split for this
        cohort (e.g. fewer than config.N_CLUSTERS students, or a
        scikit-learn failure). Wraps the underlying
        EmptyFeatureMatrixError / InsufficientStudentsError /
        ClusteringFailedError from clustering.py.
    """
    profiles = repo.get_student_profiles()
    responses = repo.get_responses()

    raw = build_feature_matrix(profiles, responses)
    normalized = normalize_feature_matrix(raw)

    try:
        cluster_result = cluster_students(normalized, raw)
    except (EmptyFeatureMatrixError, InsufficientStudentsError, ClusteringFailedError) as exc:
        raise ItemBankBuildError(
            f"Could not build the item bank: clustering failed ({exc})"
        ) from exc

    segregation_batch = compute_segregation_scores(cluster_result, responses)

    bloom_levels = repo.get_question_bloom_levels()
    parameters, skipped_parameters = build_question_parameters(bloom_levels, segregation_batch)

    return ItemBankResult(
        feature_matrix=raw,
        cluster_result=cluster_result,
        segregation_batch=segregation_batch,
        parameters=parameters,
        skipped_parameters=skipped_parameters,
    )


def score_student(
    repo: IRTRepository,
    student_id: str,
    item_bank: ItemBankResult,
) -> StudentPipelineResult:
    """Run the two STUDENT-level pipeline stages for one student, reusing
    an already-built ItemBankResult:

        Theta -> Mastery Initializer

    This is the strict, single-student API — see the module docstring's
    "Never-silently-continue" section. Use run_pipeline() instead when
    scoring many students and one bad student's data should not abort
    the rest.

    Raises
    ------
    StudentScoringError
        if student_id is unknown to the repository, has no recorded
        responses, answered a question outside the item bank's
        parameters, has no concept-tagged attempts, or has a concept
        attempt with an unrecognized Bloom level. Wraps the underlying
        exception from repository.py, theta.py, or mastery_initializer.py.
    """
    try:
        answers = repo.get_answer_records(student_id)
    except RecordNotFoundError as exc:
        raise StudentScoringError(
            f"Cannot score student {student_id!r}: {exc}"
        ) from exc

    try:
        theta_result = estimate_theta(answers, item_bank.parameters)
    except (EmptyResponsesError, DuplicateResponseError, MissingParameterError) as exc:
        raise StudentScoringError(
            f"Cannot estimate theta for student {student_id!r}: {exc}"
        ) from exc

    try:
        concept_attempts = repo.get_concept_attempts(student_id)
    except RecordNotFoundError as exc:
        raise StudentScoringError(
            f"Cannot score student {student_id!r}: {exc}"
        ) from exc

    try:
        mastery_result = initialize_mastery(student_id, theta_result, concept_attempts)
    except (
        MissingThetaError,
        EmptyConceptDataError,
        DuplicateConceptAttemptError,
        InvalidBloomLevelError,
    ) as exc:
        raise StudentScoringError(
            f"Cannot initialize mastery for student {student_id!r}: {exc}"
        ) from exc

    try:
        cluster_label = item_bank.cluster_result.label_for(student_id)
    except ValueError as exc:
        raise StudentScoringError(
            f"Student {student_id!r} was not part of the cohort used to "
            f"build this item bank's cluster split: {exc}"
        ) from exc

    return StudentPipelineResult(
        student_id=student_id,
        cluster_label=cluster_label,
        theta_result=theta_result,
        mastery_result=mastery_result,
    )


def run_pipeline(
    repo: IRTRepository,
    student_ids: Optional[Iterable[str]] = None,
) -> CohortPipelineResult:
    """Run the complete Hybrid IRT pipeline for a cohort of students:

        Feature Builder -> Clustering -> Segregation -> Question Parameters
        -> Theta -> Mastery Initializer

    This is the one function most callers (a future CLI, or the Quiz
    Portal once it's wired up) need: it builds the item bank once via
    build_item_bank(), then scores every requested student via
    score_student(), never letting one unscoreable student abort the
    rest — see the module docstring's "Never-silently-continue" section.

    Parameters
    ----------
    repo:
        Any IRTRepository (CSVRepository, PostgresRepository, or a
        future implementation) — service.py never knows or cares which.
    student_ids:
        Which students to score. If omitted, every student the
        repository knows about (repo.get_all_student_ids()) is scored.

    Raises
    ------
    ItemBankBuildError
        if the cohort-level stages can't run at all (see
        build_item_bank()). Individual students failing the
        student-level stages do NOT raise — see `.skipped_students` on
        the returned CohortPipelineResult instead.
    """
    item_bank = build_item_bank(repo)

    ids = list(student_ids) if student_ids is not None else repo.get_all_student_ids()

    student_results: Dict[str, StudentPipelineResult] = {}
    skipped_students: List[SkippedStudent] = []
    for sid in ids:
        try:
            student_results[sid] = score_student(repo, sid, item_bank)
        except StudentScoringError as exc:
            skipped_students.append(SkippedStudent(student_id=sid, reason=str(exc)))

    return CohortPipelineResult(
        item_bank=item_bank,
        student_results=student_results,
        skipped_students=skipped_students,
    )


__all__ = [
    "PipelineError",
    "ItemBankBuildError",
    "StudentScoringError",
    "ItemBankResult",
    "StudentPipelineResult",
    "SkippedStudent",
    "CohortPipelineResult",
    "build_item_bank",
    "score_student",
    "run_pipeline",
]
