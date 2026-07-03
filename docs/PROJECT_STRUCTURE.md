# PROJECT_STRUCTURE.md

## Tree

```
irt-engine/
├── README.md
├── requirements.txt
├── irt/
│   ├── __init__.py
│   ├── config.py
│   ├── bloom_mapper.py
│   ├── feature_builder.py
│   ├── clustering.py
│   ├── segregation.py
│   ├── item_parameters.py
│   ├── theta.py
│   └── mastery_initializer.py
├── tests/
│   ├── __init__.py
│   ├── test_bloom_mapper.py
│   ├── test_feature_builder.py
│   ├── test_clustering.py
│   ├── test_segregation.py
│   ├── test_item_parameters.py
│   ├── test_theta.py
│   └── test_mastery_initializer.py
├── scripts/
│   ├── __init__.py
│   ├── demo_feature_builder.py
│   ├── demo_clustering.py
│   ├── demo_segregation.py
│   ├── demo_theta.py
│   └── demo_mastery_initializer.py
├── sample_data/          (reserved, currently empty — see below)
└── docs/
    ├── THETA_VALIDATION.md
    ├── ARCHITECTURE.md
    ├── LITERATURE_REVIEW.md
    ├── INTEGRATION_GUIDE.md
    ├── API_REFERENCE.md
    ├── PROJECT_STRUCTURE.md   (this file)
    ├── DEVELOPER_GUIDE.md
    ├── RESEARCH_CONTRIBUTION.md
    ├── FUTURE_WORK.md
    └── GLOSSARY.md
```

## Top-Level Files

### `README.md`
Project entry point: motivation, architecture summary, installation, and links to every document in `docs/`.

### `requirements.txt`
Pinned dependencies: `numpy`, `pytest`, `scikit-learn` (currently in active use). Commented placeholders for `pandas`, `python-dotenv`, `psycopg2-binary` — needed once `repository.py` (not yet built) adds CSV/Postgres support.

## `irt/` — The Algorithmic Core

Every file here follows the same rule: **plain dataclasses and NumPy arrays in and out, no database or framework dependency.** This is why the whole package works standalone (see `docs/ARCHITECTURE.md`'s "Design Philosophy").

| File | Exists because... |
|---|---|
| `__init__.py` | Marks `irt/` as a Python package. Intentionally empty — no re-export shortcuts that could hide which module actually owns a symbol. |
| `config.py` | Single source of truth for every tunable constant across the whole pipeline — Bloom mapping, clustering seed, discrimination thresholds, θ numerical-stability bounds, mastery-blend weight. Nothing downstream hardcodes a magic number. |
| `bloom_mapper.py` | Implements the project's difficulty substitute (Bloom level → b). Small and dependency-free (only imports `config`) because every other module that needs difficulty imports *this*, not `config` directly — one place to retune the mapping. |
| `feature_builder.py` | Builds the per-student feature vector `clustering.py` needs, and its normalization. Imports `bloom_mapper` (for bucketing) and `config`. |
| `clustering.py` | Splits students into strong/weak via KMeans — the first half of the discrimination substitute. Imports `feature_builder` (for the `FeatureMatrix` type) and `config`. |
| `segregation.py` | Scores each question's discrimination from the cluster split — the second half of the discrimination substitute. Imports `clustering` and `feature_builder`. |
| `item_parameters.py` | The decoupling seam: the only file that imports **both** `bloom_mapper` and `segregation`, assembling their outputs into the plain `QuestionIRTParameters` dataclass that `theta.py` depends on instead. |
| `theta.py` | Newton-Raphson MLE ability estimation — the one stage that is unmodified classical IRT math. Deliberately does **not** import `bloom_mapper`, `segregation`, `clustering`, or `feature_builder`. |
| `mastery_initializer.py` | Bridges whole-quiz θ to per-concept initial mastery. Imports `bloom_mapper` (difficulty lookups) and `theta` (reuses `probability_correct` and consumes `ThetaResult`) — not `clustering`, `segregation`, or `feature_builder`. |

## `tests/`

One test file per `irt/` module, same name prefixed `test_`. 94 tests total at time of writing. Every test file exercises: the normal case, at least one boundary/edge case per validation requirement in the corresponding module's docstring, and a determinism check (same input twice → identical output). See `docs/DEVELOPER_GUIDE.md` for the testing conventions these files follow.

## `scripts/`

One runnable demo per module (`demo_<module>.py`), each chaining the real pipeline from `feature_builder` up through whichever module it's demonstrating — not synthetic/mocked calls. Each demo is also how the numerical example in `docs/THETA_VALIDATION.md` §5 was actually produced (by instrumenting, not modifying, the real functions). Running these is the fastest way for a new developer to see the pipeline work end-to-end without writing any code.

## `sample_data/`

Currently empty. Reserved for CSV fixtures once `repository.py`'s planned `CSVRepository` is implemented (see `docs/INTEGRATION_GUIDE.md` and `docs/FUTURE_WORK.md`) — the original architecture requirement was that the ML modules in `irt/` never depend on a database directly, and that CSV-mode loading be supported during development. That repository layer does not exist yet, so this folder has no contents to describe yet.

## `docs/`

This documentation set. `THETA_VALIDATION.md` predates the rest (written as a deep, research-grade validation of `theta.py` specifically, before the other nine files existed); the other nine cover the whole repository. Cross-references between files use relative links (e.g. `docs/ARCHITECTURE.md` links to `docs/RESEARCH_CONTRIBUTION.md`) so the set can be browsed either top-down from `README.md` or read standalone per file.

## What Is Deliberately Not Here Yet

Per explicit instruction at every module-implementation stage of this project, the following are **not implemented** in this repository and should not be assumed present when reading the code: `repository.py`, `service.py`, a CLI entry point, a REST API wrapper, and a `.env`/`.env.example` file. All are discussed as planned work in `docs/INTEGRATION_GUIDE.md` and `docs/FUTURE_WORK.md`.
