"""
scripts/demo_clustering.py

Integration demo chaining feature_builder.py -> clustering.py exactly as
the real pipeline will (service.py, not built yet, will do the same
chaining later): build profiles/responses -> build_feature_matrix() ->
normalize_feature_matrix() -> cluster_students(normalized, raw).

Run with:
    python -m scripts.demo_clustering
"""

from __future__ import annotations

from irt.clustering import ClusterStatistics, cluster_students
from irt.feature_builder import (
    ResponseRow,
    StudentProfileRow,
    build_feature_matrix,
    normalize_feature_matrix,
)

# ── Synthetic cohort: 8 students, a clear strong/weak split by design so
# the demo output is easy to sanity-check by eye. ──────────────────────
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

QUESTION_BANK = [
    ("Q1", "Remember"),
    ("Q2", "Understand"),
    ("Q3", "Apply"),
    ("Q4", "Apply"),
    ("Q5", "Analyze"),
    ("Q6", "Evaluate"),
]

CORRECTNESS = {
    #       Q1 Q2 Q3 Q4 Q5 Q6
    "S1": [1, 1, 1, 1, 1, 1],
    "S2": [1, 1, 1, 1, 1, 0],
    "S3": [1, 1, 1, 1, 0, 1],
    "S4": [1, 1, 1, 0, 1, 1],
    "S5": [1, 0, 1, 0, 0, 0],
    "S6": [1, 1, 0, 0, 0, 0],
    "S7": [1, 0, 0, 0, 0, 0],
    "S8": [0, 1, 0, 0, 0, 0],
}


def build_responses() -> list[ResponseRow]:
    responses = []
    for student_id, pattern in CORRECTNESS.items():
        for (question_id, bloom_level), correct in zip(QUESTION_BANK, pattern):
            responses.append(
                ResponseRow(student_id, question_id, bool(correct), bloom_level)
            )
    return responses


def _print_stats(label: str, stats: ClusterStatistics) -> None:
    print(label)
    print("-" * len(label))
    print(f"Students                 : {stats.n_students}")
    print(f"Average IQ               : {stats.avg_iq_score:.2f}")
    print(f"Average Previous Percentage : {stats.avg_previous_class_percentage:.2f}")
    print(f"Average Total Correct    : {stats.avg_total_correct:.2f}")
    print(f"Average Easy Accuracy    : {stats.avg_easy_accuracy:.2f}")
    print(f"Average Medium Accuracy  : {stats.avg_medium_accuracy:.2f}")
    print(f"Average Hard Accuracy    : {stats.avg_hard_accuracy:.2f}")
    print()


def main() -> None:
    responses = build_responses()
    raw = build_feature_matrix(PROFILES, responses)
    normalized = normalize_feature_matrix(raw)

    if raw.warnings():
        print("Imputation warnings:")
        for w in raw.warnings():
            print(f"  - {w}")
        print()

    result = cluster_students(normalized, raw)

    print("-" * 36)
    for sid in result.student_ids:
        print(f"{sid} -> {result.label_for(sid).capitalize()}")
    print("-" * 36)
    print()

    print("Cluster Statistics")
    print("-" * 36)
    _print_stats("Strong Cluster", result.statistics_for("strong"))
    print("-" * 36)
    _print_stats("Weak Cluster", result.statistics_for("weak"))
    print("-" * 36)
    print()

    print("Cluster Centroids (normalized feature space)")
    print("-" * 36)
    print(f"Feature order: {normalized.field_names}")
    for cluster_id, centroid in enumerate(result.cluster_centroids):
        role = "strong" if cluster_id == result.strong_cluster_id else "weak"
        formatted = ", ".join(f"{v:.3f}" for v in centroid)
        print(f"Cluster {cluster_id} ({role}): [{formatted}]")
    print("-" * 36)


if __name__ == "__main__":
    main()
