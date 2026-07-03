# FUTURE_WORK.md

Organized roughly in the order a team would likely tackle it, though not all items are strictly sequential.

## Near-Term: Making the Engine Usable in Production

- **`repository.py`** — a `Repository` protocol with `CSVRepository` (development, reads `sample_data/`) and `PostgresRepository` (production, shares the quiz portal's database) implementations. Currently `sample_data/` exists but is empty and no repository code exists. See `docs/INTEGRATION_GUIDE.md`.
- **`service.py`** — orchestration layer calling the eight `irt/` modules in pipeline order and handing results to the repository to persist. Sketch already in `docs/INTEGRATION_GUIDE.md`; not implemented.
- **Database integration** — writing `ThetaResult.theta` back to `StudentProfile.theta`, `QuestionIRTParameters` back to `Question.irt_difficulty`/`irt_discrimination`, and `MasteryInitializationResult` into `student_masteries.mastery_prob` — all target columns already exist in the Prisma schema; no migration is needed, only the write path.
- **CLI** — a command-line entry point (the earlier, pre-refactor IRT module had one; this repository's rebuild has not reintroduced it yet) for running the pipeline against a CSV cohort without needing the quiz portal running.
- **REST API** — an HTTP wrapper so the quiz portal's Node/Express backend can call the pipeline synchronously after `submitSession()`, rather than only via an offline batch job.
- **GUI** — a minimal internal dashboard (could be the "Teacher Dashboard" item below, or a smaller debug-only view) for inspecting θ, cluster assignments, and segregation scores without querying the database directly.

## Validation & Calibration

- **Real classroom deployment** — everything in this document's earlier sections is aimed at making deployment possible; actual use with real students is the real validation step, discussed at length in `docs/RESEARCH_CONTRIBUTION.md` §8.
- **Calibration with thousands of students** — once response volume is large enough, run classical joint 2PL calibration (MML or Bayesian) on the same item bank and compare its a/b estimates against this project's Bloom/segregation-derived values, to check the Hybrid IRT assumptions directly rather than theoretically.
- **Replacing the Bloom proxy with learned difficulty** — per `docs/DEVELOPER_GUIDE.md`'s "How to Integrate Future Psychometric Models" section, this can be done by writing a new parameter-producing module that outputs the same `QuestionIRTParameters` dataclass `item_parameters.py` produces today — `theta.py` and `mastery_initializer.py` require no changes.
- **Replacing KMeans segregation with true IRT discrimination** — same integration path as above, once per-item discrimination can be statistically estimated with confidence.
- **Empirically validating the mastery-initialization blend weight** (`config.MASTERY_PRIOR_STRENGTH`) — currently a hand-chosen constant (see `docs/RESEARCH_CONTRIBUTION.md` §5); could be tuned against how well BKT's early-session mastery tracking improves when seeded from this pipeline vs. a naive uniform prior.

## Adaptive Learning & Beyond

- **CAT (Computerized Adaptive Testing)** — using θ (and its standard error) to select the next-best diagnostic item in real time, rather than administering a fixed quiz — the standard extension of IRT ability estimation into item selection (Weiss, 1982, is the standard early reference; see `docs/LITERATURE_REVIEW.md`). Would consume `ThetaResult.standard_error` directly, which is already computed and exposed for this reason.
- **LLM-assisted feedback** — once Root Cause Analysis (per the original Synapse architecture: Gap Detection → Root Cause Analysis → RAG-based recommendations) identifies a specific weak concept, an LLM could generate targeted explanations — out of scope for this repository, which stops at mastery initialization.
- **Knowledge Graph expansion** — richer prerequisite/misconception relationships in the Student Knowledge Graph (already partially described in the Synapse Knowledge Graph dataset used to seed concepts) than this repository currently interacts with (this repository only produces flat per-concept mastery values, not graph-aware ones).
- **RAG integration** — retrieval-augmented generation for personalized learning content recommendations, the final stage of the original Synapse architecture diagram; entirely outside this repository's scope.
- **Teacher dashboard** — a view surfacing `SegregationBatchResult.flagged()` (poor/negative discriminating questions needing review) and cohort-level `ClusterStatistics`, which are already computed by this engine but have no UI consumer yet.
- **Student dashboard** — a student-facing view of their own θ trajectory and concept mastery over time, which would require persisting a *history* of `ThetaResult`s (currently only the latest is written to `StudentProfile.theta`) rather than a single snapshot.
- **Analytics** — cohort-level trends in θ distribution, segregation quality over time (are questions getting better or worse discriminators as more data comes in?), and Bloom-level performance gaps — all derivable from data this engine already produces but not yet aggregated anywhere.
- **Research publication** — writing up the Hybrid IRT architecture, its assumptions, and (once available) its validation results against classical calibration, per the discussion in `docs/RESEARCH_CONTRIBUTION.md` §7.
