"""
scripts/demo_segregation.py

Full-chain integration demo: feature_builder -> clustering -> segregation,
exactly the order service.py (not built yet) will call them in.

Run with:
    python -m scripts.demo_segregation
"""

from __future__ import annotations

from irt.clustering import cluster_students
from irt.feature_builder import (
    ResponseRow,
    StudentProfileRow,
    build_feature_matrix,
    normalize_feature_matrix,
)
from irt.segregation import compute_segregation_scores

# ── Same 8-student cohort as demo_clustering.py, plus a richer 5-question
# item bank designed to showcase every discriminator quality band. ──────
PROFILES = [
    StudentProfileRow(student_id="S1", previous_class_percentage=92.0, iq_score=118.0),
    StudentProfileRow(student_id="S2", previous_class_percentage=88.0, iq_score=112.0),
    StudentProfileRow(student_id="S3", previous_class_percentage=85.0, iq_score=109.0),
    StudentProfileRow(student_id="S4", previous_class_percentage=90.0, iq_score=115.0),
    StudentProfileRow(student_id="S5", previous_class_percentage=45.0, iq_score=88.0),
    StudentProfileRow(student_id="S6", previous_class_percentage=40.0, iq_score=None),  # missing IQ
    StudentProfileRow(student_id="S7", previous_class_percentage=50.0, iq_score=90.0),
    StudentProfileRow(student_id="S8", previous_class_percentage=38.0, iq_score=82.0),
]

# Feature-building responses (drives clustering) — kept identical in shape
# to demo_clustering.py so the strong/weak split is the same and easy to
# cross-check against that demo's output.
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

# Separate diagnostic item bank being *evaluated* for discrimination
# quality — deliberately engineered to hit excellent/good/moderate/poor/
# negative, plus one item ("Q6") only the strong cluster attempted, to
# demonstrate the skip-and-report path in the same demo run.
QUALITY_BANK = ["Q1", "Q2", "Q3", "Q4", "Q5"]
QUALITY_CORRECTNESS = {
    # question:      Q1     Q2     Q3     Q4     Q5
    "S1": [True, True, True, True, False],
    "S2": [True, True, True, False, True],
    "S3": [True, True, False, True, False],
    "S4": [True, True, True, True, True],
    "S5": [False, True, False, False, True],
    "S6": [False, False, False, True, True],
    "S7": [False, False, True, False, True],
    "S8": [False, True, False, True, True],
}
# Q6: attempted by strong-cluster students only -> will be skipped, not scored.
STRONG_ONLY_QUESTION = "Q6"


def build_clustering_responses() -> list[ResponseRow]:
    responses = []
    for sid, pattern in CLUSTERING_CORRECTNESS.items():
        for (qid, bloom), correct in zip(CLUSTERING_BANK, pattern):
            responses.append(ResponseRow(sid, qid, bool(correct), bloom))
    return responses


def build_quality_bank_responses() -> list[ResponseRow]:
    bloom_by_question = {
        "Q1": "Remember", "Q2": "Understand", "Q3": "Apply",
        "Q4": "Analyze", "Q5": "Evaluate",
    }
    responses = []
    for sid, pattern in QUALITY_CORRECTNESS.items():
        for qid, correct in zip(QUALITY_BANK, pattern):
            responses.append(ResponseRow(sid, qid, correct, bloom_by_question[qid]))
    # Q6 attempted only by the (expected) strong cluster.
    for sid in ("S1", "S2", "S3", "S4"):
        responses.append(ResponseRow(sid, STRONG_ONLY_QUESTION, True, "Apply"))
    return responses


def main() -> None:
    clustering_responses = build_clustering_responses()
    raw = build_feature_matrix(PROFILES, clustering_responses)
    normalized = normalize_feature_matrix(raw)
    cluster_result = cluster_students(normalized, raw)

    print("Strong cluster:", cluster_result.strong_student_ids())
    print("Weak cluster:  ", cluster_result.weak_student_ids())
    print()

    quality_responses = build_quality_bank_responses()
    batch = compute_segregation_scores(
        cluster_result,
        quality_responses,
        question_ids=QUALITY_BANK + [STRONG_ONLY_QUESTION],
    )

    header = f"{'Question':<10}{'Strong Acc':<12}{'Weak Acc':<12}{'Segregation':<14}{'Quality':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in batch.sorted_by_segregation_score():
        print(
            f"{r.question_id:<10}{r.strong_accuracy:<12.2f}{r.weak_accuracy:<12.2f}"
            f"{r.segregation_score:<14.2f}{r.discriminator_quality.capitalize():<10}"
        )
    print("-" * len(header))
    print()

    if batch.skipped:
        print("Skipped questions:")
        for s in batch.skipped:
            print(f"  {s.question_id}: {s.reason}")
        print()

    if batch.warnings():
        print("Warnings:")
        for w in batch.warnings():
            print(f"  - {w}")
        print()

    flagged = batch.flagged()
    if flagged:
        print(f"Flagged for review ({len(flagged)}):", [r.question_id for r in flagged])


if __name__ == "__main__":
    main()
