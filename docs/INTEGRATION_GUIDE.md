# INTEGRATION_GUIDE.md

**Status: this repository (`irt-engine`) is standalone today. `repository.py`, `service.py`, a CLI, and a REST wrapper are all planned but not yet implemented (see `docs/FUTURE_WORK.md`). This document describes the integration plan, not the current state.**

## Why the engine is standalone

Per the original architecture requirement, `irt-engine` must not be tightly coupled to the quiz portal, and must remain callable as an independent Python project. Every module under `irt/` takes and returns plain dataclasses/NumPy arrays — none of them import a database driver, an HTTP framework, or a portal-specific type. This is what makes the integration plan below possible without changing any of the eight modules already built.

## Current Quiz Flow (as implemented in the quiz portal today)

The quiz portal (`backend/src`, Express + Prisma) already implements:

```
Student submits a quiz session
        │
        ▼
POST /sessions/:id/submit
        │
        ▼
submitSession()                      (controllers/session.controller.ts)
        │  grades each SessionAnswer, computes is_correct
        ▼
KnowledgeService.updateMastery()      (services/knowledge.service.ts)
        │  Bayesian Knowledge Tracing update, per (student_id, concept_id)
        ▼
student_masteries table               (StudentMastery Prisma model)
        mastery_prob, attempt_count, correct_count, last_seen
```

This flow already runs BKT updates on every submission. **It has no ability (θ) estimation and no mastery-initialization step today** — every student's `student_masteries` rows start from BKT's own default prior (`mastery_prob` defaults to `0.2` per the Prisma schema) rather than a diagnostic-informed initial value. Closing that gap is exactly what `irt-engine` is for.

## Planned Integration Flow

```
Diagnostic Quiz Session Submitted
        │
        ▼
submitSession()                       (existing, quiz portal)
        │
        ▼
Repository Layer                       (NEW — repository.py, not yet implemented)
        │  reads: StudentProfile (class9_marks, iq_score, theta),
        │         Question (bloom_level, irt_difficulty, irt_discrimination),
        │         SessionAnswer (is_correct, concept_id)
        │  writes: theta, theta_se back onto StudentProfile;
        │          irt_difficulty, irt_discrimination back onto Question;
        │          initial mastery_prob into student_masteries
        ▼
IRT Engine                             (irt/ — THIS repository, already implemented)
        │  feature_builder -> clustering -> segregation -> item_parameters
        │       -> theta.estimate_theta()          => ThetaResult
        │       -> mastery_initializer.initialize_mastery()
        │                                            => MasteryInitializationResult
        ▼
Service Layer                          (NEW — service.py, not yet implemented)
        │  orchestrates the pipeline call above, and calls the repository
        │  to persist ThetaResult.theta and each ConceptMastery.initial_mastery
        ▼
student_masteries                      (existing table, now seeded with an
        │                                informed initial mastery_prob instead
        │                                of the BKT default of 0.2)
        ▼
KnowledgeService.updateMastery()        (existing — continues to run on every
                                          subsequent submission, exactly as today,
                                          now starting from a better prior)
```

## Where `repository.py` Fits

Per the original architecture requirement, `repository.py` is the **only** place in `irt-engine` allowed to talk to a database or CSV file — no module in `irt/` (the algorithmic core already built) may depend on Postgres directly. Planned shape:

- A `Repository` protocol/interface, with two implementations:
  - `CSVRepository` — reads `sample_data/*.csv` (development/testing, no live database needed).
  - `PostgresRepository` — reads/writes via the same `DATABASE_URL` the quiz portal's Prisma client uses (see `config.load_database_url()`, already implemented and ready for this).
- Reads needed: `StudentProfile.class9_marks`, `StudentProfile.iq_score` (if the psychometric test has populated it), `Question.bloom_level`, `SessionAnswer.is_correct` + `SessionAnswer.concept_id`.
- Writes needed: `StudentProfile.theta` (already a column in the Prisma schema — `theta Float? @map("theta")`), `Question.irt_difficulty` / `Question.irt_discrimination` (already columns — `irt_difficulty Float? @map("irt_difficulty")`, `irt_discrimination Float? @map("irt_discrimination")`), and `StudentMastery.mastery_prob` for each `(student_id, concept_id)` pair produced by `mastery_initializer.initialize_mastery()`.

**No schema change is required for any of this** — the Prisma schema already has every column this integration needs (added when the current `backend/irt/` Python module was first scaffolded), which is exactly the compatibility requirement `mastery_initializer.py`'s docstring commits to.

## Where `service.py` Fits

`service.py` is the planned orchestration layer that calls the pipeline modules in dependency order (the same order shown in `docs/ARCHITECTURE.md`'s pipeline diagram) and hands the repository the final results to persist. It is the only module, besides `repository.py`, that is allowed to know about *both* the database *and* the algorithmic core — none of the eight modules already implemented in `irt/` will need to change to support this.

Sketch of its responsibility (not yet implemented):

```python
# service.py (planned, not yet implemented)
def run_diagnostic_pipeline(repo: Repository, cohort_id: str) -> None:
    profiles, responses = repo.fetch_cohort(cohort_id)
    raw = feature_builder.build_feature_matrix(profiles, responses)
    normalized = feature_builder.normalize_feature_matrix(raw)
    cluster_result = clustering.cluster_students(normalized, raw)
    segregation_batch = segregation.compute_segregation_scores(cluster_result, responses)
    bloom_levels = repo.fetch_bloom_levels()
    parameters, _skipped = item_parameters.build_question_parameters(bloom_levels, segregation_batch)
    repo.save_question_parameters(parameters)  # -> irt_difficulty, irt_discrimination

    for student_id in cluster_result.student_ids:
        answer_records = repo.fetch_answer_records(student_id)
        theta_result = theta.estimate_theta(answer_records, parameters)
        repo.save_theta(student_id, theta_result)  # -> StudentProfile.theta

        concept_attempts = repo.fetch_concept_attempts(student_id)
        mastery_result = mastery_initializer.initialize_mastery(student_id, theta_result, concept_attempts)
        repo.save_initial_mastery(mastery_result)  # -> student_masteries.mastery_prob
```

## How PostgreSQL Will Be Used

The quiz portal already runs on PostgreSQL via Prisma (`DATABASE_URL` in its `.env`, same as `config.load_database_url()` already expects). `PostgresRepository` (planned) will connect to the **same database**, not a separate one — per the original architecture requirement that both the quiz portal and `irt-engine` eventually share one database rather than each maintaining its own.

## How Prisma Interacts

Prisma is a TypeScript/JavaScript ORM and is **not** used from the Python side. `PostgresRepository` will use a Python database driver (e.g. `psycopg2`, already listed as a planned dependency in `requirements.txt`) to read/write the same tables Prisma's schema defines — Prisma remains the schema's source of truth (its migrations define the columns), while `irt-engine`'s repository layer reads/writes to those same tables via raw SQL or a lightweight query layer, not through Prisma itself.

## How APIs Will Call the Engine

Two integration shapes are possible and are not mutually exclusive:

1. **Batch/offline invocation** — a scheduled job or admin-triggered script calls `service.run_diagnostic_pipeline()` for a cohort after a diagnostic quiz window closes, writing results directly to Postgres. This is the simplest integration and requires no new HTTP endpoint.
2. **On-demand invocation** — the Express backend's `submitSession()` controller, after grading a diagnostic-quiz submission, calls out to a small internal endpoint (or subprocess) that runs the pipeline for that one student and returns `ThetaResult` + `MasteryInitializationResult` synchronously. This requires a REST wrapper (see `docs/FUTURE_WORK.md`) that does not exist yet.

Both shapes use exactly the same `service.py` orchestration function and exactly the same `irt/` modules — the difference is only in what triggers the call.

## Expected Directory Layout After Integration

```
irt-engine/
├── README.md
├── requirements.txt
├── .env.example                    # DATABASE_URL, matching the quiz portal's own
├── irt/                             # unchanged — the algorithmic core
│   ├── config.py
│   ├── bloom_mapper.py
│   ├── feature_builder.py
│   ├── clustering.py
│   ├── segregation.py
│   ├── item_parameters.py
│   ├── theta.py
│   └── mastery_initializer.py
│   ├── repository.py                # NEW — CSVRepository + PostgresRepository
│   └── service.py                   # NEW — orchestration
├── cli.py                           # NEW — command-line entry point (planned)
├── tests/
├── scripts/                         # existing demos remain useful for local dev
├── sample_data/                     # CSVs for CSVRepository (dev/testing)
└── docs/
```

No module currently in `irt/` needs to move, be renamed, or change its public API for this integration — `repository.py` and `service.py` are additive.
