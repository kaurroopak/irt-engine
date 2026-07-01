"""
config.py — single source of truth for the hybrid IRT engine's tunable
constants and database connection.

Nothing downstream hard-codes a threshold, a Bloom weight, or a column
default. If a supervisor asks "what does 'Apply' map to?" or "what counts
as a poor discriminator?", the answer lives here and nowhere else.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# ── CHANGE 1: Bloom -> difficulty (b) mapping ───────────────────────────────
# Configurable per your supervisor's instruction. Keys are matched
# case-insensitively against the question/concept dataset's `bloom_level`
# column (e.g. "remember", "Understand", "APPLY").
BLOOM_DIFFICULTY_MAP: dict[str, float] = {
    "remember": -2.0,
    "understand": -1.0,
    "apply": 0.0,
    "analyze": 1.0,
    "evaluate": 2.0,
    "create": 2.5,
}

# ── CHANGE 2: easy/medium/hard bucketing ────────────────────────────────────
# Per-decision: buckets are derived from the SAME Bloom signal as difficulty
# (b), not from a separate raw difficulty column, so there's exactly one
# source of truth for "how hard is this question" in the whole system.
# Every Bloom level named in BLOOM_DIFFICULTY_MAP must appear in exactly one
# bucket below — this is validated at import time (see _validate_buckets).
BLOOM_DIFFICULTY_BUCKETS: dict[str, str] = {
    "remember": "easy",
    "understand": "easy",
    "apply": "medium",
    "analyze": "hard",
    "evaluate": "hard",
    "create": "hard",
}
ACCURACY_BUCKETS = ("easy", "medium", "hard")

# ── CHANGE 2: feature vector construction ───────────────────────────────────
# Decision: class9_marks is already a 0-100 percentage; used directly, no
# max-marks normalization step.
PREVIOUS_CLASS_PERCENTAGE_IS_NORMALIZED = True

# Decision: iq_score is optional (psychometric test ships separately).
# Missing values are imputed with the cohort mean at feature-build time,
# and every imputation is logged/recorded so callers can see how much of
# the clustering input was real vs imputed.
IQ_SCORE_COLUMN_OPTIONAL = True

# The exact feature vector order from the spec. Every module that builds or
# consumes a feature matrix must agree on this order — defined once here.
FEATURE_VECTOR_FIELDS: tuple[str, ...] = (
    "previous_class_percentage",
    "iq_score",
    "total_correct",
    "easy_accuracy",
    "medium_accuracy",
    "hard_accuracy",
)

# ── CHANGE 2/3: clustering + segregation ────────────────────────────────────
N_CLUSTERS = 2
RANDOM_STATE = 42  # KMeans seed, for reproducible strong/weak assignment
SEGREGATION_POOR_DISCRIMINATOR_THRESHOLD = 0.15  # |strong_acc - weak_acc| below this => flagged

# ── Guess-detection parameters (unchanged from the prior IRT module; mirror
# the constants in the TypeScript submit path — see docs/ for the sync note) ─
RT_GUESS_THRESHOLD_MS = 1500
SURPRISE_THRESHOLD = 0.30
MIN_STUDENTS_WARN = 50

# Bounds used when seeding cold-start mastery from theta.
SEED_PRIOR_MIN = 0.05
SEED_PRIOR_MAX = 0.95


def _validate_buckets() -> None:
    missing = set(BLOOM_DIFFICULTY_MAP) - set(BLOOM_DIFFICULTY_BUCKETS)
    if missing:
        raise ValueError(
            f"BLOOM_DIFFICULTY_BUCKETS is missing an entry for: {sorted(missing)}. "
            "Every Bloom level in BLOOM_DIFFICULTY_MAP must map to an accuracy bucket."
        )
    bad = set(BLOOM_DIFFICULTY_BUCKETS.values()) - set(ACCURACY_BUCKETS)
    if bad:
        raise ValueError(f"BLOOM_DIFFICULTY_BUCKETS has unknown bucket name(s): {sorted(bad)}")


_validate_buckets()


def load_database_url(override: Optional[str] = None) -> str:
    """Resolve DATABASE_URL: explicit override > backend/.env > environment.
    Only used by the Postgres repository path; CSV-mode callers never need
    this."""
    if override:
        return override
    try:
        from dotenv import dotenv_values

        here = os.path.dirname(os.path.abspath(__file__))
        env = dotenv_values(os.path.join(here, "..", ".env"))
        if env.get("DATABASE_URL"):
            return env["DATABASE_URL"]
    except ImportError:
        pass
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("No DATABASE_URL found. Pass --database-url or set it in .env")
    return url
