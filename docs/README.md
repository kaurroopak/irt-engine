# irt-engine

**A Hybrid Item Response Theory engine for Synapse — an AI-driven personalized learning platform.**

`irt-engine` turns a student's diagnostic-quiz responses into a single, comparable ability estimate (θ) and a set of initial per-concept mastery probabilities, using a data-light architecture designed for a platform that does not yet have the thousands of responses classical IRT calibration requires.

> **Note on terminology:** earlier planning documents for this project used the placeholder name `StudentAbility` for the ability-estimation output. The class actually implemented is **`ThetaResult`** (`irt/theta.py`). This README and all files under `docs/` use the real class names throughout — `ThetaResult`, `QuestionIRTParameters`, `MasteryInitializationResult`, etc. — not placeholder names from earlier planning notes.

---

## Motivation: the educational problem

Synapse's long-term goal is personalized, adaptive learning: recommend the right content to the right student at the right time. That requires knowing two things about every student: **how able they are overall**, and **which specific concepts they've mastered versus not**. A raw quiz percentage answers neither question well — it doesn't account for which questions were actually hard, and it says nothing at the level of individual concepts.

Item Response Theory (IRT) is the standard statistical framework for solving the first problem: it estimates a latent ability parameter (θ) from a response pattern, on a scale that accounts for how difficult and how discriminating each item was (Lord, 1980; Hambleton, Swaminathan, & Rogers, 1991). The problem for Synapse specifically is that classical IRT jointly estimates item difficulty, item discrimination, *and* student ability from a large response matrix — and a new platform's first cohort of students doesn't have that data yet.

## Why Hybrid IRT was developed

Under supervisor guidance, this project replaces the two data-hungry parts of classical 2PL calibration with deterministic substitutes that work from day one, and estimates **only** ability (θ) statistically:

| 2PL parameter | Classical IRT | This project's Hybrid IRT |
|---|---|---|
| Difficulty (b) | Estimated from response data | Read directly from each question's **Bloom's Taxonomy** level (`bloom_mapper.py`) |
| Discrimination (a) | Estimated from response data | Computed as the accuracy gap between a **KMeans-derived** strong/weak student cluster (`clustering.py`, `segregation.py`) |
| Ability (θ) | Estimated from response data | Estimated via **Newton-Raphson MLE** on the standard 2PL log-likelihood — this part *is* classical IRT math, just with a and b already known |

**This Hybrid IRT architecture is this project's own engineering design, not an established IRT model from the literature.** See [`docs/RESEARCH_CONTRIBUTION.md`](docs/RESEARCH_CONTRIBUTION.md) and [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) for a full breakdown of which parts are standard theory and which are this project's contribution.

---

## High-level architecture

```
Student takes diagnostic quiz
        │
        ▼
┌───────────────────┐
│ Feature Builder     │  per-student feature vector (accuracy, IQ, prior %)
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ Normalization       │  z-score, so KMeans isn't scale-biased
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ KMeans (k=2)         │  strong / weak student clusters
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ Segregation          │  per-question discrimination (a) from the cluster split
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ QuestionIRTParameters│  (a, b) per question — b from Bloom, a from segregation
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ Theta Estimation     │  Newton-Raphson MLE  ->  ThetaResult
└─────────┬──────────┘
          ▼
┌───────────────────┐
│ Mastery Initialization│  θ + concept accuracy + Bloom difficulty -> per-concept mastery
└─────────┬──────────┘
          ▼
   Student Knowledge Graph  (quiz portal, already implemented)
          ▼
   Bayesian Knowledge Tracing (quiz portal, already implemented)
          ▼
   Adaptive Learning
```

Full module-by-module detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Features

- **Fully deterministic** given a fixed random seed (`config.RANDOM_STATE`) — same input always produces the same θ, clusters, and mastery values.
- **Never silently continues.** Every module raises specific, documented exceptions for missing data, duplicate records, or degenerate inputs (empty cohorts, all-correct/all-incorrect response patterns, unscoreable questions) rather than guessing.
- **Fully decoupled item parameters.** `theta.py` never imports `bloom_mapper.py` or `segregation.py` — it only depends on the plain `QuestionIRTParameters` dataclass (`item_parameters.py`), so either upstream source can be replaced later without touching θ estimation.
- **No database dependency in the ML modules.** Every module in `irt/` takes and returns plain Python dataclasses / NumPy arrays; no module imports a database driver.
- **Configuration-driven.** Every threshold, mapping, and numerical-stability bound lives in `irt/config.py` — nothing is a hardcoded magic number inside the algorithmic modules.

## Folder structure

```
irt-engine/
├── README.md                    (this file)
├── requirements.txt
├── irt/                         core package
│   ├── config.py                  all tunable constants
│   ├── bloom_mapper.py             Bloom level -> difficulty (b) + accuracy bucket
│   ├── feature_builder.py          per-student feature vector + normalization
│   ├── clustering.py               KMeans(k=2) strong/weak clustering
│   ├── segregation.py              per-question discrimination (a)
│   ├── item_parameters.py          QuestionIRTParameters assembly (decoupling seam)
│   ├── theta.py                    Newton-Raphson MLE ability (θ) estimation
│   └── mastery_initializer.py      per-concept initial mastery
├── tests/                        pytest suite, one file per module (94 tests)
├── scripts/                      runnable integration demos, one per module
├── docs/                         this documentation set
└── sample_data/                  (reserved for future CSV-mode repository layer)
```

Full breakdown: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

## Requirements

See [`requirements.txt`](requirements.txt). Currently:

```
numpy>=1.24
pytest>=7.4
scikit-learn>=1.3
```

`theta.py`'s Newton-Raphson implementation uses only the Python standard library (`math`) — no `scipy`/`girth` dependency was needed, since a and b are fixed inputs here rather than something to jointly calibrate.

## Installation

```bash
git clone <this-repo>
cd irt-engine
pip install -r requirements.txt
```

## Running the demos

Each module has a corresponding runnable demo under `scripts/`, chaining the real pipeline up to that point:

```bash
python -m scripts.demo_feature_builder
python -m scripts.demo_clustering
python -m scripts.demo_segregation
python -m scripts.demo_theta
python -m scripts.demo_mastery_initializer
```

## Running the tests

```bash
python -m pytest tests/ -q
```

At time of writing: **94/94 tests passing.**

## Pipeline overview

See the architecture diagram above and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module-dependency graph and design rationale.

## Future integration with Synapse

`irt-engine` is intentionally standalone — no module in `irt/` imports a database driver or web framework. Integration into the quiz portal is planned via a `repository.py` (CSV in development, PostgreSQL in production) and a `service.py` orchestration layer, neither of which has been implemented yet. See [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md) for the full integration plan.

## Future work

Replacing the Bloom-proxy difficulty and KMeans-segregation discrimination with statistically calibrated parameters once enough response data exists, a REST API, a CLI, real classroom validation, and more. Full list: [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md).

## Documentation index

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, module responsibilities, dependency graph |
| [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) | Every technique's literature grounding, standard vs. project-original |
| [`docs/INTEGRATION_GUIDE.md`](docs/INTEGRATION_GUIDE.md) | How this engine will plug into the Synapse quiz portal |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Every public function/dataclass, with examples |
| [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | Repository layout, file-by-file |
| [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) | Conventions, testing, how to extend the engine |
| [`docs/RESEARCH_CONTRIBUTION.md`](docs/RESEARCH_CONTRIBUTION.md) | Established theory vs. engineering decisions vs. this project's contribution |
| [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md) | Roadmap |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Term definitions |
| [`docs/THETA_VALIDATION.md`](docs/THETA_VALIDATION.md) | Deep-dive validation of the θ estimation module specifically |
