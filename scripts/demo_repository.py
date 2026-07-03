"""
scripts/demo_repository.py

Full-chain integration demo, but with a twist relative to every other
demo in scripts/: instead of hand-building StudentProfileRow/ResponseRow
lists inline, every input in this demo comes from CSVRepository reading
sample_data/*.csv. This is the proof that the Repository Layer's output
plugs into feature_builder -> clustering -> segregation ->
item_parameters -> theta -> mastery_initializer with ZERO adaptation —
exactly the guarantee irt/repository.py's module docstring describes.

Run with:
    python -m scripts.demo_repository

Swapping in PostgresRepository later
-------------------------------------
Every ML module call below reads from `repo`, an IRTRepository. Once the
Quiz Portal's Postgres database is reachable, the ONLY line that needs to
change in a script like this one is the repository construction itself:

    # CSVRepository (this demo):
    repo = CSVRepository.from_default_sample_data()

    # PostgresRepository (production):
    repo = PostgresRepository(database_url="postgresql://...")
    # or, letting config.load_database_url() resolve it from .env / the
    # DATABASE_URL environment variable:
    repo = PostgresRepository()

Nothing else in this file — not one call to feature_builder, clustering,
segregation, item_parameters, theta, or mastery_initializer — would need
to change, because both repository classes satisfy the same
IRTRepository contract and return the same dataclasses either way.
"""

from __future__ import annotations

from irt.clustering import cluster_students
from irt.feature_builder import build_feature_matrix, normalize_feature_matrix
from irt.item_parameters import build_question_parameters
from irt.mastery_initializer import initialize_mastery
from irt.repository import CSVRepository
from irt.segregation import compute_segregation_scores
from irt.theta import estimate_theta

# Students to headline the per-student report.
DEMO_STUDENT_COUNT = 3


def main() -> None:
    # 0. The Repository Layer: every downstream call below reads from
    #    `repo`, never from a CSV path or a SQL connection directly.
    with CSVRepository.from_default_sample_data() as repo:
        print("=" * 60)
        print("Repository Layer demo — CSVRepository over sample_data/")
        print("=" * 60)
        student_ids = repo.get_all_student_ids()
        print(f"Students loaded:  {len(student_ids)}  {student_ids}")

        profiles = repo.get_student_profiles()
        responses = repo.get_responses()
        bloom_levels = repo.get_question_bloom_levels()
        print(f"Responses loaded: {len(responses)}")
        print(f"Questions loaded: {len(bloom_levels)}")
        print()

        # 1-2. Feature Builder + Normalization (Change 2).
        raw = build_feature_matrix(profiles, responses)
        for warning in raw.warnings():
            print(f"[imputation] {warning}")
        normalized = normalize_feature_matrix(raw)

        # 3. KMeans strong/weak clustering (Change 2).
        cluster_result = cluster_students(normalized, raw)
        print(f"Strong cluster: {cluster_result.strong_student_ids()}")
        print(f"Weak cluster:   {cluster_result.weak_student_ids()}")
        print()

        # 4. Segregation -> per-question discrimination (Change 3).
        segregation_batch = compute_segregation_scores(cluster_result, responses)
        flagged = segregation_batch.flagged()
        print(f"Questions scored: {len(segregation_batch.results)}")
        print(f"Questions skipped: {len(segregation_batch.skipped)}")
        if flagged:
            flagged_ids = [r.question_id for r in flagged]
            print(f"Flagged (poor/negative) discriminators: {flagged_ids}")
        print()

        # 5. QuestionIRTParameters — Bloom (b) + segregation (a) assembled
        #    from repository output, same as every other demo script.
        parameters, skipped_params = build_question_parameters(bloom_levels, segregation_batch)
        print(f"QuestionIRTParameters built for {len(parameters)} question(s).")
        if skipped_params:
            print(f"Skipped: {[(s.question_id, s.reason) for s in skipped_params]}")
        print()

        # 6-7. Theta estimation + mastery initialization, per student.
        print("-" * 60)
        for sid in student_ids[:DEMO_STUDENT_COUNT]:
            answers = repo.get_answer_records(sid)
            theta_result = estimate_theta(answers, parameters)

            concept_attempts = repo.get_concept_attempts(sid)
            mastery_result = initialize_mastery(sid, theta_result, concept_attempts)

            se_str = (
                f"{theta_result.standard_error:.3f}"
                if theta_result.standard_error is not None
                else "n/a"
            )
            print(f"Student {sid}")
            print(
                f"  theta = {theta_result.theta:+.2f}  "
                f"(converged={theta_result.converged}, SE={se_str}, "
                f"n_responses={theta_result.n_responses})"
            )
            s = mastery_result.summary
            print(
                f"  mastery: {s.n_concepts} concept(s), avg = "
                f"{s.average_initial_mastery:.2f}, lowest = "
                f"{s.lowest_mastery_concept_id} ({s.lowest_mastery_value:.2f}), "
                f"highest = {s.highest_mastery_concept_id} "
                f"({s.highest_mastery_value:.2f})"
            )
            print("-" * 60)


if __name__ == "__main__":
    main()
