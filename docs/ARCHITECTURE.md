# ARCHITECTURE.md

## System Overview

`irt-engine` is a pure-Python, database-free package (`irt/`) that turns raw diagnostic-quiz responses into two outputs consumed by the rest of Synapse:

1. **θ (theta)** — a single latent ability estimate per student, returned as a `ThetaResult` (`irt/theta.py`).
2. **Initial per-concept mastery** — a `MasteryInitializationResult` (`irt/mastery_initializer.py`) that seeds the Student Knowledge Graph before Bayesian Knowledge Tracing (BKT) starts updating it from live activity.

Every module between those two endpoints exists because classical Item Response Theory's usual joint-estimation approach needs response volume Synapse doesn't have yet (see `docs/RESEARCH_CONTRIBUTION.md`). Each module replaces one piece of that joint estimation with either a deterministic rule (Bloom mapping), an unsupervised split (KMeans), or keeps the original statistical method where it's still viable with limited data (Newton-Raphson MLE for θ alone).

## Design Philosophy

Four principles recur across every module in this repository, enforced by convention rather than a shared base class:

1. **Never silently continue.** Every module defines its own specific exception types (e.g. `EmptyFeatureMatrixError`, `InsufficientAttemptsError`, `MissingParameterError`) and raises them for missing data, duplicate records, or degenerate inputs, rather than defaulting to a plausible-looking but wrong value.
2. **No duplicated logic.** Shared math lives in exactly one place and is imported, not re-derived — e.g. `theta.probability_correct()` (the 2PL curve) is reused directly by `mastery_initializer.py` rather than reimplemented there.
3. **Configuration lives in one file.** `irt/config.py` is the only place a threshold, mapping, or numerical-stability bound is defined. No algorithmic module hardcodes a magic number.
4. **Plain data across every boundary.** Every public function takes and returns dataclasses (or NumPy arrays for numeric matrices) — never a database row, an ORM object, or a framework-specific type. This is what makes the whole package usable standalone, independent of Postgres, an API layer, or a CLI (none of which exist yet — see `docs/FUTURE_WORK.md`).

## Why the Architecture Is Modular

Each pipeline stage is a separate module with a narrow, single responsibility specifically so that any one stage can be replaced independently as the project matures — most importantly, so that Bloom-derived difficulty and KMeans-derived discrimination (both explicitly flagged throughout this documentation as engineering approximations, not established psychometric methods) can later be swapped for statistically calibrated parameters **without changing `theta.py` at all**, because `theta.py` only ever consumes the `QuestionIRTParameters` dataclass, never `bloom_mapper.py` or `segregation.py` directly (see §4, `item_parameters.py`, below).

## Pipeline

```
Student
   │
   ▼
Feature Builder            (feature_builder.py)
   │  StudentProfileRow + ResponseRow  ->  FeatureMatrix
   ▼
Normalization               (feature_builder.normalize_feature_matrix)
   │  FeatureMatrix  ->  FeatureMatrix (z-scored)
   ▼
KMeans                      (clustering.py)
   │  FeatureMatrix  ->  ClusterResult (strong/weak split)
   ▼
Segregation                 (segregation.py)
   │  ClusterResult + ResponseRow  ->  SegregationBatchResult
   ▼
QuestionIRTParameters       (item_parameters.py)
   │  Bloom levels + SegregationBatchResult  ->  list[QuestionIRTParameters]
   ▼
Theta Estimation             (theta.py)
   │  AnswerRecord + QuestionIRTParameters  ->  ThetaResult
   ▼
Mastery Initialization        (mastery_initializer.py)
   │  ThetaResult + ConceptAttempt  ->  MasteryInitializationResult
   ▼
Student Knowledge Graph        (quiz portal — already implemented, not part of this repo)
   ▼
Bayesian Knowledge Tracing      (quiz portal — already implemented, not part of this repo)
   ▼
Adaptive Learning
```

## Module Dependency Graph

```
config.py  ◄──────────────────────────────────────────────────┐
   ▲                                                             │
   │ (constants)                                                 │ (constants)
   │                                                              │
bloom_mapper.py                                                   │
   ▲            ▲                                                 │
   │            │                                                 │
   │            └───────────────┐                                 │
   │                             │                                 │
feature_builder.py               │                                 │
   ▲                             │                                 │
   │                             │                                 │
clustering.py                    │                                 │
   ▲                             │                                 │
   │                             │                                 │
segregation.py ──────────────────┤                                 │
   ▲                             │                                 │
   │                             │                                 │
item_parameters.py ◄─────────────┘  (imports bloom_mapper + segregation)
   ▲
   │  (QuestionIRTParameters ONLY — no bloom_mapper or segregation import)
   │
theta.py ─────────────────────────────────────────────────────────┘
   ▲
   │  (ThetaResult + probability_correct ONLY — no bloom_mapper, segregation,
   │   clustering, or feature_builder import)
   │
mastery_initializer.py
```

**Read literally from `import` statements in the current codebase:**

- `config.py` — no internal dependencies (this is the root of the graph).
- `bloom_mapper.py` — imports `config` only.
- `feature_builder.py` — imports `bloom_mapper` (for `bucket_for()`) and `config`.
- `clustering.py` — imports `feature_builder` (for `FeatureMatrix`) and `config`.
- `segregation.py` — imports `clustering` (for `ClusterResult`), `feature_builder` (for `ResponseRow`), and `config`.
- `item_parameters.py` — imports `bloom_mapper` and `segregation`. **This is the only module that imports both.**
- `theta.py` — imports `item_parameters` (for the `QuestionIRTParameters` dataclass only) and `config`. Does **not** import `bloom_mapper`, `segregation`, `clustering`, or `feature_builder`.
- `mastery_initializer.py` — imports `bloom_mapper` (for difficulty lookups in the theta-implied-accuracy calculation) and `theta` (for `ThetaResult` and `probability_correct`), plus `config`. Does **not** import `clustering`, `segregation`, or `feature_builder`.

The decoupling that matters most architecturally is **`theta.py`'s isolation from `bloom_mapper.py` and `segregation.py`**: it was introduced specifically (via `item_parameters.py`) so that a future change to how difficulty or discrimination are computed never requires touching, or re-testing, θ estimation.

## Module-by-Module Description

### `config.py`

**Why it exists:** a single source of truth for every tunable value in the system — the Bloom-to-difficulty mapping, clustering parameters, discrimination-quality thresholds, θ-estimation numerical-stability bounds, and mastery-initialization blend weights. **Responsibility:** hold constants and one validation function (`_validate_buckets()`, run at import time) that checks the Bloom map and Bloom bucket map stay in sync. It has no algorithmic logic of its own.

### `bloom_mapper.py`

**Why it exists:** implements the project's difficulty substitute — reading a question's Bloom's Taxonomy level directly as its 2PL difficulty parameter (b), instead of statistically calibrating it. **Responsibility:** `difficulty_for(bloom_level) -> float` and `bucket_for(bloom_level) -> str` (easy/medium/hard), both failing loudly (`UnknownBloomLevelError`) on an unrecognized level rather than silently defaulting.

### `feature_builder.py`

**Why it exists:** produces the per-student feature vector that clustering.py needs — previous class percentage, IQ score, total correct, and accuracy in each Bloom-derived difficulty bucket. **Responsibility:** `build_feature_matrix()` assembles and imputes (cohort-mean, for missing IQ/percentage) the raw vector; `normalize_feature_matrix()` z-scores it for KMeans.

### `clustering.py`

**Why it exists:** implements the project's discrimination substitute's first half — splitting the student population into a strong and a weak group via KMeans(k=2), which `segregation.py` then uses to score individual questions. **Responsibility:** `cluster_students()` runs KMeans, then labels the resulting clusters "strong"/"weak" by comparing average `total_correct` (never assumed from the raw sklearn cluster id), and computes per-cluster descriptive statistics.

### `segregation.py`

**Why it exists:** implements the project's discrimination substitute's second half — for every question, computes strong-cluster accuracy minus weak-cluster accuracy as a KMeans-based stand-in for the classical IRT discrimination parameter (a). **Responsibility:** `compute_segregation_scores()` scores every question, classifies each into a quality band, and flags poor/negative discriminators, without ever raising for one bad question (it reports a skip reason instead, per the never-silently-continue principle).

### `item_parameters.py`

**Why it exists:** the decoupling seam described above. **Responsibility:** defines `QuestionIRTParameters` (the entire contract `theta.py` depends on) and `build_question_parameters()`, the one function that actually knows about both `bloom_mapper.py` and `segregation.py` and assembles their outputs into that plain dataclass.

### `theta.py`

**Why it exists:** the one part of this pipeline that *is* classical IRT statistical estimation, unmodified in its mathematics — Newton-Raphson maximum-likelihood estimation of θ under the 2PL model, with a and b held fixed as known inputs. **Responsibility:** `estimate_theta()` runs the Newton-Raphson loop (with numerical-stability safeguards documented in `docs/THETA_VALIDATION.md`) and returns a `ThetaResult`. Also exposes `probability_correct()` as a public, reusable implementation of the 2PL curve itself.

### `mastery_initializer.py`

**Why it exists:** bridges the gap between θ (a single, whole-quiz ability number) and the Student Knowledge Graph's need for a mastery probability *per concept*. **Responsibility:** `initialize_mastery()` blends each concept's observed accuracy with a "theta-implied accuracy" (computed by reusing `theta.probability_correct()` at a fixed reference discrimination, since per-item discrimination is outside this module's declared inputs), weighted by how many items probed that concept — a Beta-Bernoulli-style shrinkage design documented fully in the module and in `docs/RESEARCH_CONTRIBUTION.md`.

## How Modules Communicate

Every inter-module boundary in this list is a **plain Python dataclass or NumPy array**, never a database row, dict-of-dicts, or framework object:

| Producer | Dataclass | Consumer |
|---|---|---|
| `feature_builder.build_feature_matrix` | `FeatureMatrix` | `feature_builder.normalize_feature_matrix`, `clustering.cluster_students` |
| `clustering.cluster_students` | `ClusterResult` | `segregation.compute_segregation_scores` |
| `segregation.compute_segregation_scores` | `SegregationBatchResult` | `item_parameters.build_question_parameters` |
| `item_parameters.build_question_parameters` | `list[QuestionIRTParameters]` | `theta.estimate_theta` |
| `theta.estimate_theta` | `ThetaResult` | `mastery_initializer.initialize_mastery` |
| `mastery_initializer.initialize_mastery` | `MasteryInitializationResult` | (future) `repository.py`, to persist into `student_masteries` |

This table is also the map of what a future `service.py` orchestration layer needs to call, in order — see `docs/INTEGRATION_GUIDE.md`.
