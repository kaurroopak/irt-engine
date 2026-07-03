# GLOSSARY.md

Alphabetical. Terms specific to this project's Hybrid IRT design (as opposed to standard psychometric terms) are marked **(project term)**.

---

**2PL (Two-Parameter Logistic Model)** — An IRT model giving the probability of a correct response as `P(correct) = 1/(1+exp(-a(θ-b)))`, parameterized by item discrimination (a) and difficulty (b). See `docs/THETA_VALIDATION.md` §2.

**Adaptive Learning** — Instruction or item selection that adjusts to a student's current estimated ability/mastery in real time, rather than following a fixed sequence. The final stage of the overall Synapse architecture; not implemented in this repository (see `docs/FUTURE_WORK.md`).

**Ability** — See **Theta**.

**Bloom Bucket (project term)** — This project's mapping of a Bloom level to one of three accuracy categories (`easy`, `medium`, `hard`) used in the clustering feature vector, defined in `config.BLOOM_DIFFICULTY_BUCKETS` and computed by `bloom_mapper.bucket_for()`.

**Bloom's Taxonomy** — A hierarchical classification of cognitive-process complexity (Remember, Understand, Apply, Analyze, Evaluate, Create — the revised, verb-based version from Anderson & Krathwohl, 2001), originally designed for curriculum and assessment planning. Repurposed by this project (`bloom_mapper.py`) as a difficulty proxy — see `docs/LITERATURE_REVIEW.md` for the distinction between the taxonomy itself (established) and this use of it (project contribution).

**ClusterResult** — The dataclass (`irt/clustering.py`) returned by `cluster_students()`, holding per-student cluster labels, which cluster id is "strong"/"weak", centroids, and per-cluster statistics.

**Concept Mastery** — A probability, in `[0,1]`, representing how likely a student is to correctly apply a specific concept right now. Distinct from **Theta** — see `docs/THETA_VALIDATION.md` §9.2. Tracked over time by BKT (in the quiz portal); initialized once, per concept, by this repository's `mastery_initializer.py`.

**Convergence (Newton-Raphson)** — The condition under which iterative theta estimation stops: `|score| < config.THETA_CONVERGENCE_TOLERANCE`. A `ThetaResult` with `converged=False` means either the iteration limit was hit, or (more commonly in this codebase) the response pattern was all-correct/all-incorrect, for which no finite maximum-likelihood θ exists (see **Extreme Response Pattern**).

**Difficulty (b)** — The 2PL parameter marking the point on the θ scale where `P(correct) = 0.5`. In this project, read directly from a question's Bloom level (`bloom_mapper.difficulty_for()`) rather than statistically estimated — see `docs/RESEARCH_CONTRIBUTION.md`.

**Discrimination (a)** — The 2PL parameter controlling how steeply the Item Characteristic Curve rises around its difficulty point — how well a question separates high- from low-ability examinees. In this project, computed as the accuracy gap between a KMeans strong/weak cluster split (`segregation.py`) rather than statistically estimated — see `docs/RESEARCH_CONTRIBUTION.md`.

**Extreme Response Pattern** — A response set that is entirely correct or entirely incorrect (including the trivial single-response case). Has no finite maximum-likelihood θ (proof in `docs/THETA_VALIDATION.md` §6.1); handled by `theta._extreme_pattern_theta()`, which reports a clamp-boundary θ with `converged=False`.

**Feature Matrix (`FeatureMatrix`)** — The dataclass (`irt/feature_builder.py`) holding the per-student, 6-field numeric vector (`previous_class_percentage, iq_score, total_correct, easy_accuracy, medium_accuracy, hard_accuracy`) used as KMeans clustering input.

**Fisher Information** — The negative expected second derivative of the log-likelihood with respect to θ; `I(θ) = Σ a_i² P_i(1-P_i)` in the 2PL case. Used both to drive the Newton-Raphson update and to compute the standard error of θ. See `docs/THETA_VALIDATION.md` §3.3.

**Hybrid IRT (project term)** — This project's overall proposed architecture: Bloom-derived difficulty + KMeans-derived discrimination + statistically-estimated ability. Explicitly **not** an established IRT model from the literature — see `docs/RESEARCH_CONTRIBUTION.md`.

**Imputation** — Filling a missing value (in this project, `iq_score` or `previous_class_percentage`) with a substitute — here, the cohort mean, always recorded in an `ImputationReport` rather than done silently.

**IRT (Item Response Theory)** — A family of statistical models relating an examinee's probability of a correct response to their latent ability and one or more item parameters. See `docs/THETA_VALIDATION.md` §1.

**Item Characteristic Curve (ICC)** — The S-shaped (logistic) curve of `P(correct)` plotted against θ for a fixed item (a, b). See `docs/THETA_VALIDATION.md` §2.3.

**Knowledge Graph (Student Knowledge Graph)** — The quiz portal's representation of a student's per-concept mastery, plus prerequisite/misconception relationships between concepts. Seeded by this repository's `MasteryInitializationResult`; implemented and updated outside this repository.

**Latent Ability** — See **Theta**. "Latent" because it is never observed directly, only inferred from a response pattern.

**Likelihood** — The probability of an observed response pattern, as a function of θ (with item parameters fixed): `L(θ) = Π P_i(θ)^{y_i}(1-P_i(θ))^{1-y_i}`. See `docs/THETA_VALIDATION.md` §3.1.

**Log-Likelihood** — The natural logarithm of the likelihood, `ℓ(θ)`, maximized in practice instead of the raw likelihood for numerical stability and because sums are easier to differentiate than products.

**MasteryInitializationResult** — The dataclass (`irt/mastery_initializer.py`) returned by `initialize_mastery()`, holding a student's θ and a per-concept `ConceptMastery` breakdown plus a `MasteryInitializationSummary`.

**Maximum Likelihood Estimation (MLE)** — Choosing the parameter value (here, θ) that maximizes the likelihood of the observed data. The general estimation framework `theta.estimate_theta()` implements.

**Newton-Raphson** — An iterative root-finding method used here to find the θ where the log-likelihood's derivative (the score) is zero, i.e. the MLE. Converges quadratically on concave, twice-differentiable objectives — see `docs/THETA_VALIDATION.md` §3.6 for why that applies here.

**Posterior** — In Bayesian terms, an updated probability distribution after combining a prior with observed data. Referenced conceptually in `mastery_initializer.py`'s docstring: the observed/theta-implied accuracy blend is structured like a Beta-Bernoulli posterior mean, though this project does not implement full Bayesian inference (no distribution is tracked, only a point-estimate blend).

**Prior** — In Bayesian terms, a belief held before observing data. In this project (informally, not full Bayesian inference), the "theta-implied accuracy" term in `mastery_initializer.py` functions as a prior that observed concept accuracy is blended against, weighted by how many items were attempted.

**Psychometrics** — The field of measurement theory concerned with the design and analysis of tests measuring latent psychological/educational constructs (ability, aptitude, personality, etc.). IRT is one of psychometrics' central frameworks.

**QuestionIRTParameters** — The dataclass (`irt/item_parameters.py`) holding one question's `question_id`, `discrimination` (a), and `difficulty` (b) — the entire input contract `theta.py` depends on, deliberately decoupled from how a and b were produced.

**Response Pattern** — The full set of a student's (question, correctness) pairs for a given assessment — the raw data θ is estimated from.

**Segregation Score (project term)** — This project's discrimination substitute: `strong_cluster_accuracy - weak_cluster_accuracy` for a given question, computed by `segregation.py`. Structurally related to, but not identical to, the classical item-discrimination index (Ebel & Frisbie, 1991) — see `docs/LITERATURE_REVIEW.md`.

**Standard Error (of θ)** — `SE(θ) = 1/√I(θ)`, the asymptotic uncertainty of the MLE estimate. Larger with fewer responses or lower-discrimination items; `None` (not computed) when Fisher information is exactly zero.

**Student Model** — A system's representation of what it believes about a learner's knowledge/ability state. This entire repository is, collectively, one component of Synapse's student model.

**StudentProfileRow** — The dataclass (`irt/feature_builder.py`) representing one student's profile fields relevant to feature-building: `student_id`, `previous_class_percentage`, `iq_score`.

**Theta (θ)** — The latent ability parameter this engine estimates. Represented by the `ThetaResult` dataclass (`irt/theta.py`) — **not** by a class named `StudentAbility`; see the note in `docs/API_REFERENCE.md`. Estimated by `theta.estimate_theta()` via Newton-Raphson MLE under the 2PL model.

**ThetaResult** — The dataclass holding the output of θ estimation: `theta`, `iterations`, `converged`, `log_likelihood`, `standard_error`, `n_responses`.
