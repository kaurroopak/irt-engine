"""
scripts/demo_feature_builder.py

Integration demo for feature_builder.py — CHANGE 2 verification step.

This script does NOT test internals; it calls the public API
(build_feature_matrix, normalize_feature_matrix) exactly the way
clustering.py will, using synthetic-but-realistic Synapse diagnostic-quiz
data: 8 students, a 10-question Bloom-tagged item bank spanning Remember
through Evaluate, and one student with a missing IQ score to demonstrate
cohort-mean imputation.

Run with:
    python -m scripts.demo_feature_builder
"""

from __future__ import annotations

from irt.feature_builder import (
    ResponseRow,
    StudentProfileRow,
    build_feature_matrix,
    normalize_feature_matrix,
)

# ── Synthetic student profiles ───────────────────────────────────────────
# previous_class_percentage mirrors StudentProfile.class9_marks (already
# 0-100, per the earlier decision). iq_score is intentionally missing for
# S6 to exercise the imputation path.
PROFILES = [
    StudentProfileRow(student_id="S1", previous_class_percentage=92.0, iq_score=118.0),
    StudentProfileRow(student_id="S2", previous_class_percentage=85.0, iq_score=105.0),
    StudentProfileRow(student_id="S3", previous_class_percentage=78.0, iq_score=101.0),
    StudentProfileRow(student_id="S4", previous_class_percentage=64.0, iq_score=97.0),
    StudentProfileRow(student_id="S5", previous_class_percentage=55.0, iq_score=90.0),
    StudentProfileRow(student_id="S6", previous_class_percentage=70.0, iq_score=None),  # missing IQ
    StudentProfileRow(student_id="S7", previous_class_percentage=48.0, iq_score=85.0),
    StudentProfileRow(student_id="S8", previous_class_percentage=40.0, iq_score=82.0),
]

# ── Synthetic 10-question diagnostic bank, Bloom-tagged ─────────────────
# (question_id, bloom_level) — mirrors Synapse_Quiz_30Q.xlsx's bloom_level
# column. Evaluate is included even though it wasn't present in the 30Q
# sample, since the pipeline must support the full configured Bloom map.
QUESTION_BANK = [
    ("Q1", "Remember"),
    ("Q2", "Remember"),
    ("Q3", "Understand"),
    ("Q4", "Understand"),
    ("Q5", "Apply"),
    ("Q6", "Apply"),
    ("Q7", "Apply"),
    ("Q8", "Analyze"),
    ("Q9", "Analyze"),
    ("Q10", "Evaluate"),
]

# ── Per-student correctness pattern ──────────────────────────────────────
# Hand-authored so accuracy differences are legible in the printed matrix:
# S1-S3 are stronger performers, S4-S6 are mid, S7-S8 are weaker, with a
# clear easy > medium > hard accuracy gradient for most students (as real
# diagnostic data tends to show), plus a couple of realistic exceptions
# (e.g. S4 misses an "easy" Remember question — students do that) so the
# demo doesn't look artificially clean.
CORRECTNESS = {
    #       Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10
    "S1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    "S2": [1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    "S3": [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    "S4": [1, 0, 1, 1, 1, 0, 1, 0, 0, 0],
    "S5": [1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    "S6": [1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    "S7": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "S8": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
}


def build_responses() -> list[ResponseRow]:
    responses = []
    for student_id, pattern in CORRECTNESS.items():
        for (question_id, bloom_level), correct in zip(QUESTION_BANK, pattern):
            responses.append(
                ResponseRow(
                    student_id=student_id,
                    question_id=question_id,
                    is_correct=bool(correct),
                    bloom_level=bloom_level,
                )
            )
    return responses


def _print_matrix(label: str, student_ids: list[str], matrix) -> None:
    print(label)
    print("-" * len(label))
    for sid, row in zip(student_ids, matrix):
        formatted = ", ".join(f"{v:.2f}" for v in row)
        print(f"{sid}: [{formatted}]")
    print()


def main() -> None:
    responses = build_responses()

    fm = build_feature_matrix(PROFILES, responses)

    print("=" * 60)
    print("Student IDs")
    print("-" * 11)
    for sid in fm.student_ids:
        print(sid)
    print()

    print(f"Feature order: {fm.field_names}")
    print()

    _print_matrix("Raw Feature Matrix", fm.student_ids, fm.matrix)

    print("Imputations")
    print("-" * 11)
    if not fm.imputations:
        print("(none)")
    else:
        for report in fm.imputations:
            label = "IQ" if report.field_name == "iq_score" else report.field_name
            print(f"{label}:")
            for sid in report.imputed_student_ids:
                print(f"  Student {sid} -> cohort mean {report.fill_value:.1f}")
    print()

    normalized = normalize_feature_matrix(fm)
    _print_matrix("Normalized Feature Matrix", normalized.student_ids, normalized.matrix)

    print("=" * 60)


if __name__ == "__main__":
    main()