"""
scripts/demo_mastery_initializer.py

Full-chain integration demo: feature_builder -> clustering -> segregation
-> item_parameters -> theta -> mastery_initializer, the complete Hybrid
IRT pipeline as it stands after this module.

Run with:
    python -m scripts.demo_mastery_initializer
"""

from __future__ import annotations

from irt.clustering import cluster_students
from irt.feature_builder import (
    ResponseRow,
    StudentProfileRow,
    build_feature_matrix,
    normalize_feature_matrix,
)
from irt.item_parameters import build_question_parameters
from irt.mastery_initializer import ConceptAttempt, initialize_mastery
from irt.segregation import compute_segregation_scores
from irt.theta import AnswerRecord, estimate_theta

PROFILES = [
    StudentProfileRow(student_id="S1", previous_class_percentage=92.0, iq_score=118.0),
    StudentProfileRow(student_id="S2", previous_class_percentage=88.0, iq_score=112.0),
    StudentProfileRow(student_id="S3", previous_class_percentage=85.0, iq_score=109.0),
    StudentProfileRow(student_id="S4", previous_class_percentage=90.0, iq_score=115.0),
    StudentProfileRow(student_id="S5", previous_class_percentage=45.0, iq_score=88.0),
    StudentProfileRow(student_id="S6", previous_class_percentage=40.0, iq_score=None),
    StudentProfileRow(student_id="S7", previous_class_percentage=50.0, iq_score=90.0),
    StudentProfileRow(student_id="S8", previous_class_percentage=38.0, iq_score=82.0),
]

CLUSTERING_BANK = [
    ("F1", "Remember"), ("F2", "Understand"), ("F3", "Apply"),
    ("F4", "Apply"), ("F5", "Analyze"), ("F6", "Evaluate"),
]
CLUSTERING_CORRECTNESS = {
    "S1": [1, 1, 1, 1, 1, 1],
    "S2": [1, 1, 1, 1, 1, 0],
    "S3": [1, 1, 1, 1, 0, 1],
    "S4": [1, 1, 1, 0, 1, 1],
    "S5": [1, 0, 1, 0, 0, 0],
    "S6": [1, 1, 0, 0, 0, 0],
    "S7": [1, 0, 0, 0, 0, 0],
    "S8": [0, 1, 0, 0, 0, 0],
}

# Diagnostic quiz item bank -- now with a concept tag per question, since
# mastery is initialized per-concept, not per-quiz. Two concepts,
# deliberately probed at different Bloom levels.
ITEM_BANK = {
    # question_id: (concept_id, bloom_level)
    "Q1": ("Ohms_Law", "Remember"),
    "Q2": ("Ohms_Law", "Understand"),
    "Q3": ("Ohms_Law", "Apply"),
    "Q4": ("Resistance", "Analyze"),
    "Q5": ("Resistance", "Evaluate"),
}
QUALITY_CORRECTNESS = {
    "S1": [True, True, True, True, False],
    "S2": [True, True, True, False, True],
    "S3": [True, True, False, True, False],
    "S4": [True, True, True, True, True],
    "S5": [False, True, False, False, True],
    "S6": [False, False, False, True, True],
    "S7": [False, False, True, False, True],
    "S8": [False, True, False, True, True],
}

DEMO_STUDENTS = ["S4", "S3", "S7"]  # high / average / low, matching demo_theta.py


def build_clustering_responses() -> list[ResponseRow]:
    responses = []
    for sid, pattern in CLUSTERING_CORRECTNESS.items():
        for (qid, bloom), correct in zip(CLUSTERING_BANK, pattern):
            responses.append(ResponseRow(sid, qid, bool(correct), bloom))
    return responses


def build_quality_bank_responses() -> list[ResponseRow]:
    responses = []
    for sid, pattern in QUALITY_CORRECTNESS.items():
        for qid, correct in zip(ITEM_BANK, pattern):
            _, bloom = ITEM_BANK[qid]
            responses.append(ResponseRow(sid, qid, correct, bloom))
    return responses


def main() -> None:
    # 1-2. Cluster the cohort.
    clustering_responses = build_clustering_responses()
    raw = build_feature_matrix(PROFILES, clustering_responses)
    normalized = normalize_feature_matrix(raw)
    cluster_result = cluster_students(normalized, raw)

    # 3. Score discrimination via segregation.
    quality_responses = build_quality_bank_responses()
    bloom_by_question = {qid: bloom for qid, (_, bloom) in ITEM_BANK.items()}
    segregation_batch = compute_segregation_scores(
        cluster_result, quality_responses, question_ids=list(ITEM_BANK)
    )

    # 4. Assemble QuestionIRTParameters (a, b) per question.
    parameters, _skipped = build_question_parameters(bloom_by_question, segregation_batch)

    # 5. Estimate theta per student, then initialize concept mastery.
    for sid in DEMO_STUDENTS:
        pattern = QUALITY_CORRECTNESS[sid]
        answer_records = [AnswerRecord(qid, correct) for qid, correct in zip(ITEM_BANK, pattern)]
        theta_result = estimate_theta(answer_records, parameters)

        concept_attempts = [
            ConceptAttempt(
                concept_id=ITEM_BANK[qid][0],
                question_id=qid,
                is_correct=correct,
                bloom_level=ITEM_BANK[qid][1],
            )
            for qid, correct in zip(ITEM_BANK, pattern)
        ]
        mastery_result = initialize_mastery(sid, theta_result, concept_attempts)

        print("-" * 44)
        print(f"Student {sid}")
        print(f"Theta = {mastery_result.theta:.2f}")
        print()
        for concept_id, cm in mastery_result.concept_masteries.items():
            print(f"Concept: {concept_id}")
            print(f"Mastery = {cm.initial_mastery:.2f}")
            print(
                f"  (observed_accuracy={cm.observed_accuracy:.2f}, "
                f"theta_implied_accuracy={cm.theta_implied_accuracy:.2f}, "
                f"n_attempted={cm.n_attempted}, weight_observed={cm.weight_observed:.2f})"
            )
        print()
        s = mastery_result.summary
        print(
            f"Summary: {s.n_concepts} concept(s), avg mastery = "
            f"{s.average_initial_mastery:.2f}, lowest = {s.lowest_mastery_concept_id} "
            f"({s.lowest_mastery_value:.2f}), highest = {s.highest_mastery_concept_id} "
            f"({s.highest_mastery_value:.2f})"
        )
        print("-" * 44)
        print()


if __name__ == "__main__":
    main()
