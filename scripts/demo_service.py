"""
scripts/demo_service.py

Demonstrates the Service Layer: irt.service.run_pipeline() driving the
ENTIRE Hybrid IRT pipeline — Feature Builder -> Clustering -> Segregation
-> Question Parameters -> Theta -> Mastery Initializer — from one
function call, given nothing but a repository.

Contrast this with scripts/demo_repository.py, which called every ML
module directly, one stage at a time, to prove the Repository Layer's
output was compatible with each of them. This script shows what a real
caller (a future CLI, or the Quiz Portal once it's wired up) actually
does: it doesn't know or care about feature_builder.py, clustering.py,
segregation.py, item_parameters.py, theta.py, or mastery_initializer.py
individually — it calls service.run_pipeline(repo) once and reads the
result.

Run with:
    python -m scripts.demo_service
"""

from __future__ import annotations

from irt.repository import CSVRepository
from irt.service import build_item_bank, run_pipeline, score_student


def print_cohort_report(result) -> None:
    bank = result.item_bank
    print("Item bank")
    print("-" * 60)
    print(f"Strong cluster: {bank.cluster_result.strong_student_ids()}")
    print(f"Weak cluster:   {bank.cluster_result.weak_student_ids()}")
    print(f"Questions scored by segregation: {len(bank.segregation_batch.results)}")
    flagged = bank.segregation_batch.flagged()
    if flagged:
        print(f"Flagged (poor/negative) discriminators: {[r.question_id for r in flagged]}")
    print(f"Question IRT parameters built: {len(bank.parameters)}")
    print()

    print("Per-student results")
    print("-" * 60)
    for sid in result.scored_student_ids():
        r = result.result_for(sid)
        se_str = (
            f"{r.theta_result.standard_error:.3f}"
            if r.theta_result.standard_error is not None
            else "n/a"
        )
        s = r.mastery_result.summary
        print(f"Student {sid}  [{r.cluster_label}]")
        print(
            f"  theta = {r.theta_result.theta:+.2f}  "
            f"(converged={r.theta_result.converged}, SE={se_str})"
        )
        print(
            f"  mastery: {s.n_concepts} concept(s), avg = "
            f"{s.average_initial_mastery:.2f}, lowest = "
            f"{s.lowest_mastery_concept_id} ({s.lowest_mastery_value:.2f}), "
            f"highest = {s.highest_mastery_concept_id} "
            f"({s.highest_mastery_value:.2f})"
        )
    print("-" * 60)

    if result.skipped_students:
        print()
        print("Skipped students")
        print("-" * 60)
        for s in result.skipped_students:
            print(f"  {s.student_id}: {s.reason}")
        print("-" * 60)

    warnings = result.warnings()
    if warnings:
        print()
        print("Warnings")
        print("-" * 60)
        for w in warnings:
            print(f"  {w}")
        print("-" * 60)


def main() -> None:
    with CSVRepository.from_default_sample_data() as repo:
        # ── Whole-cohort run: one call runs every pipeline stage for
        #    every student the repository knows about. ────────────────
        print("=" * 60)
        print("Service Layer demo — run_pipeline() over the whole cohort")
        print("=" * 60)
        result = run_pipeline(repo)
        print_cohort_report(result)
        print()

        # ── Scoping to specific students, including one that doesn't
        #    exist — proves run_pipeline() never aborts the rest of the
        #    cohort over one bad student_id. ──────────────────────────
        print("=" * 60)
        print("Service Layer demo — run_pipeline() with a bad student_id mixed in")
        print("=" * 60)
        partial = run_pipeline(repo, student_ids=["S1", "S3", "DOES_NOT_EXIST"])
        print_cohort_report(partial)
        print()

        # ── Single-student strict API: build the item bank once, then
        #    score one student directly. This is what a caller wants
        #    when scoring one student right after they submit a quiz,
        #    without re-running clustering/segregation for everyone. ──
        print("=" * 60)
        print("Service Layer demo — score_student() for a single student")
        print("=" * 60)
        item_bank = build_item_bank(repo)
        single = score_student(repo, "S2", item_bank)
        print(f"Student {single.student_id}  [{single.cluster_label}]")
        print(f"  theta = {single.theta_result.theta:+.2f}")
        print(f"  mastery concepts = {single.mastery_result.summary.n_concepts}")


if __name__ == "__main__":
    main()
