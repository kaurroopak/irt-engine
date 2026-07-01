"""
scripts/demo_theta.py

Full-chain integration demo: bloom_mapper + segregation -> item_parameters
-> theta, exactly the order service.py (not built yet) will call them in.
Reuses the same 8-student cohort and 5-question quality item bank as
demo_segregation.py so the segregation numbers here match that demo.

Run with:
    python -m scripts.demo_theta
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
from irt.segregation import compute_segregation_scores
from irt.theta import AnswerRecord, ThetaResult, estimate_theta, probability_correct

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

# The item bank being turned into QuestionIRTParameters (Bloom -> b,
# segregation -> a) and then used to estimate theta for three example
# students spanning high / average / low ability.
QUALITY_BANK_BLOOM = {"Q1": "Remember", "Q2": "Understand", "Q3": "Apply", "Q4": "Analyze", "Q5": "Evaluate"}
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

# Three students to headline the theta report: expected high / average / low.
DEMO_STUDENTS = ["S4", "S3", "S7"]


def interpretation(theta: float) -> str:
    """Simple, documented banding for the printed report only — NOT used
    anywhere in theta.py itself, since theta.py's output is a number, not
    a label; labeling is a presentation concern that belongs to whoever
    prints/consumes the result (here, the demo; later, mastery_initializer.py
    might use a different banding for its own purposes)."""
    if theta >= 1.0:
        return "High ability"
    if theta >= -0.5:
        return "Average ability"
    return "Needs support"


def build_clustering_responses() -> list[ResponseRow]:
    responses = []
    for sid, pattern in CLUSTERING_CORRECTNESS.items():
        for (qid, bloom), correct in zip(CLUSTERING_BANK, pattern):
            responses.append(ResponseRow(sid, qid, bool(correct), bloom))
    return responses


def build_quality_bank_responses() -> list[ResponseRow]:
    responses = []
    for sid, pattern in QUALITY_CORRECTNESS.items():
        for qid, correct in zip(QUALITY_BANK_BLOOM, pattern):
            responses.append(ResponseRow(sid, qid, correct, QUALITY_BANK_BLOOM[qid]))
    return responses


def main() -> None:
    # 1. Cluster the cohort (needed to produce segregation-based discrimination).
    clustering_responses = build_clustering_responses()
    raw = build_feature_matrix(PROFILES, clustering_responses)
    normalized = normalize_feature_matrix(raw)
    cluster_result = cluster_students(normalized, raw)

    # 2. Score every question's discrimination via segregation.
    quality_responses = build_quality_bank_responses()
    segregation_batch = compute_segregation_scores(
        cluster_result, quality_responses, question_ids=list(QUALITY_BANK_BLOOM)
    )

    # 3. Assemble QuestionIRTParameters (Bloom -> b, segregation -> a).
    #    This is the ONLY place theta.py's inputs get built from Bloom/
    #    segregation — theta.py itself never imports either module.
    parameters, skipped = build_question_parameters(QUALITY_BANK_BLOOM, segregation_batch)
    if skipped:
        print("Questions skipped while building IRT parameters:")
        for s in skipped:
            print(f"  {s.question_id}: {s.reason}")
        print()

    print("Question IRT Parameters (a = discrimination, b = difficulty)")
    print("-" * 50)
    for p in sorted(parameters, key=lambda p: p.question_id):
        print(f"{p.question_id:<6} a = {p.discrimination:+.2f}   b = {p.difficulty:+.2f}")
    print("-" * 50)
    print()

    # 4. Estimate theta per student.
    responses_by_student: dict[str, list[AnswerRecord]] = {}
    for sid, pattern in QUALITY_CORRECTNESS.items():
        responses_by_student[sid] = [
            AnswerRecord(qid, correct) for qid, correct in zip(QUALITY_BANK_BLOOM, pattern)
        ]

    results: dict[str, ThetaResult] = {
        sid: estimate_theta(responses_by_student[sid], parameters) for sid in DEMO_STUDENTS
    }

    print("-" * 48)
    for sid in DEMO_STUDENTS:
        result = results[sid]
        print(f"Student {sid}")
        print(f"Theta = {result.theta:.2f}")
        print(f"Interpretation: {interpretation(result.theta)}")
        se_str = f"{result.standard_error:.3f}" if result.standard_error is not None else "n/a"
        print(f"(converged={result.converged}, iterations={result.iterations}, SE={se_str})")
        print("-" * 48)
    print()

    # 5. Per-question P(correct) at each demo student's estimated theta —
    #    lets a reader sanity-check the math by eye (e.g. a high-theta
    #    student should show high P(correct) on easy/moderate items).
    print("Per-question P(correct) at each student's estimated theta")
    header = f"{'Question':<10}{'a':<8}{'b':<8}" + "".join(f"P({sid})".ljust(10) for sid in DEMO_STUDENTS)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for p in sorted(parameters, key=lambda p: p.question_id):
        row = f"{p.question_id:<10}{p.discrimination:<8.2f}{p.difficulty:<8.2f}"
        for sid in DEMO_STUDENTS:
            prob = probability_correct(p.discrimination, p.difficulty, results[sid].theta)
            row += f"{prob:<10.2f}"
        print(row)
    print("-" * len(header))


if __name__ == "__main__":
    main()
