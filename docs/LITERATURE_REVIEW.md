# LITERATURE_REVIEW.md

## Literature Review — Synapse Hybrid IRT Engine

This document grounds every technique used in `irt-engine` in its source literature, and states plainly — per topic — whether the technique is used **as standard theory**, **adapted as an engineering decision**, or **is this project's original contribution**. It is written to support the capstone report, thesis, and viva; every table row is traceable to a specific module and, where applicable, a specific function or constant in the codebase.

Only real, verifiable references are used below. No citation in this document is fabricated.

---

## 1. Item Response Theory (General)

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Latent-trait measurement framework | Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems.* Lawrence Erlbaum Associates. | Establishes IRT as the framework for placing examinee ability and item properties on a shared latent scale. | Motivates estimating θ (`irt/theta.py`) instead of using raw percentage-correct. | Standard Theory |
| Foundational statistical test theory linking classical and latent-trait models | Lord, F. M., & Novick, M. R. (1968). *Statistical Theories of Mental Test Scores.* Addison-Wesley. | Historical/statistical foundation for latent-trait modeling. | Background justification for choosing IRT over classical test theory in `docs/README.md` / `docs/THETA_VALIDATION.md`. | Standard Theory |
| Practical IRT reference text | Hambleton, R. K., Swaminathan, H., & Rogers, H. J. (1991). *Fundamentals of Item Response Theory.* Sage Publications. | Standard applied reference for the 2PL model, ICC interpretation, and MLE ability estimation. | Cited throughout `docs/THETA_VALIDATION.md` §1–§3 for the 2PL equation, ICC shape, and score/information derivations. | Standard Theory |
| Concise IRT primer | Baker, F. B. (2001). *The Basics of Item Response Theory* (2nd ed.). ERIC Clearinghouse on Assessment and Evaluation. | Accessible treatment of the 2PL model and parameter interpretation. | Cited for parameter interpretation (`a`, `b`, `θ`) in `docs/THETA_VALIDATION.md` §2.2. | Standard Theory |
| IRT parameter estimation techniques | Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker. | Standard reference for Newton-Raphson MLE ability estimation under logistic IRT models, including score/information formulation and boundary-estimate handling. | Directly grounds `theta.py`'s `_score_and_information()` and the Newton-Raphson loop in `estimate_theta()`; also grounds the extreme-response-pattern handling in `_extreme_pattern_theta()`. | Standard Theory |
| Modern IRT for psychology/education | Embretson, S. E., & Reise, S. P. (2000). *Item Response Theory for Psychologists.* Lawrence Erlbaum Associates. | Explains why θ is preferable to a raw score (test-dependence, non-interval scaling of raw scores). | Cited in `docs/README.md` / `docs/THETA_VALIDATION.md` §1.3 to motivate the whole project's approach over a simple percentage score. | Standard Theory |

---

## 2. The Two-Parameter Logistic (2PL) Model

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| 2PL logistic response function `P(correct\|θ) = 1/(1+exp(-a(θ-b)))` | Hambleton, Swaminathan, & Rogers (1991); Baker (2001) | Defines the probability of a correct response as a function of ability, difficulty, and discrimination. | Implemented verbatim as `irt/theta.py::probability_correct(a, b, theta)`, the single shared implementation of the 2PL curve used by both `estimate_theta()` and `mastery_initializer.py`. | Standard Theory (equation); numerically-stable clamped implementation is an engineering decision (see `docs/THETA_VALIDATION.md` §7.1). |
| Item Characteristic Curve (ICC) interpretation | Baker (2001); Hambleton et al. (1991) | Describes the S-shaped curve's inflection point (at θ=b) and slope (proportional to a). | Used to justify why `b` is read from Bloom level (inflection point = "50% chance point") and why `a` is derived from cluster-accuracy gap (curve steepness ≈ separation power). | Standard Theory |
| Local independence assumption | Lord (1980); Hambleton et al. (1991) | Assumes responses to different items are conditionally independent given θ, which allows the joint likelihood to factor into a simple product. | Underlies the log-likelihood formulation in `theta.py::_score_and_information()`, which sums per-item log-likelihood contributions. | Standard Theory |

---

## 3. Maximum Likelihood Estimation (MLE) for Ability

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| MLE ability estimation under IRT | Baker & Kim (2004); Hambleton et al. (1991) | Standard method for estimating θ given known item parameters, by maximizing the log-likelihood of the observed response pattern. | This is exactly the problem `theta.py::estimate_theta()` solves; `a` and `b` are supplied as known inputs (via `QuestionIRTParameters`) rather than jointly estimated. | Standard Theory (estimation target); fixing `a`,`b` as known (rather than jointly calibrated) is this project's engineering decision — see §4 of `docs/THETA_VALIDATION.md` and `docs/RESEARCH_CONTRIBUTION.md`. |
| Non-existence of a finite MLE for perfect/zero response patterns | Baker & Kim (2004) (general discussion of estimation difficulty at ability-distribution extremes); result follows algebraically from the logistic score function | Explains why all-correct/all-incorrect response patterns have a monotonically increasing/decreasing log-likelihood with no finite maximizer. | Detected explicitly by `theta.py::_extreme_pattern_theta()`, which reports the configured clamp boundary (`THETA_EXTREME_PATTERN_CLAMP`) with `converged=False` instead of iterating toward an unreachable optimum. | Standard Theory (the mathematical fact); the specific clamp-and-flag reporting convention is this project's engineering decision. |
| Asymptotic standard error of the MLE | Baker & Kim (2004); Hambleton et al. (1991) | `SE(θ) = 1/√I(θ)`, where `I(θ)` is Fisher information. | Implemented as `theta.py::_standard_error()`. | Standard Theory |

---

## 4. Newton-Raphson Optimization

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Newton-Raphson method for maximizing a concave log-likelihood | Baker & Kim (2004) (applied to IRT ability estimation); standard numerical-optimization theory | Iteratively updates `θ ← θ + score/information` to find the MLE, exploiting quadratic convergence near the optimum for a smooth, concave objective. | Implemented directly as the main loop in `theta.py::estimate_theta()`. | Standard Theory |
| Concavity of the 2PL log-likelihood in θ (fixed a, b) | Follows algebraically from `I(θ) = Σaᵢ²Pᵢ(1-Pᵢ) ≥ 0` for all θ, so `ℓ''(θ) = -I(θ) ≤ 0` everywhere | Guarantees Newton-Raphson is globally well-behaved for this specific objective, even when some `aᵢ < 0` (poorly-discriminating/flagged items). | Documented and empirically verified in `docs/THETA_VALIDATION.md` §3.5; motivates why Newton-Raphson (not gradient descent or grid search) was chosen. | Standard Theory (the mathematical property); the explicit note that this still holds even with negative segregation-derived `a` values is this project's applied observation. |
| Numerical-stability safeguards (exponent clamping, probability clamping, step capping, iteration ceiling) | General numerical-methods practice; no single external source claims these exact bound values | Prevent floating-point overflow/underflow and runaway steps in a from-scratch Newton-Raphson implementation. | `THETA_EXPONENT_CLAMP`, per-response `eps=1e-9` clamp, `THETA_MAX_STEP`, `THETA_MAX_ITERATIONS` in `irt/config.py` and `irt/theta.py`. | Our Contribution (engineering decision) |

---

## 5. Bloom's Taxonomy

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Original cognitive-domain taxonomy | Bloom, B. S. (Ed.), Engelhart, M. D., Furst, E. J., Hill, W. H., & Krathwohl, D. R. (1956). *Taxonomy of Educational Objectives: The Classification of Educational Goals, Handbook I: Cognitive Domain.* David McKay Company. | Establishes a hierarchy of cognitive-process complexity for educational objectives. | Conceptual basis for `bloom_level` as a question metadata field throughout the dataset (`data/Synapse_Quiz_30Q.xlsx`, `sample_data/questions.csv`). | Standard Theory (established educational literature) |
| Revised, verb-based taxonomy | Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives.* Longman. | Provides the six ordered cognitive levels actually used as labels: Remember, Understand, Apply, Analyze, Evaluate, Create. | These exact six levels are the keys of `irt/config.py::BLOOM_DIFFICULTY_MAP` and `BLOOM_DIFFICULTY_BUCKETS`, consumed by `irt/bloom_mapper.py`. | Standard Theory (the taxonomy itself) |
| Using Bloom level as a *statistical difficulty proxy* (b parameter) | No published source establishes this mapping; it is this project's design | Substitutes an expert-assigned, theory-grounded ordinal difficulty signal for a data-estimated IRT difficulty parameter, given the absence of large-scale response data. | `bloom_mapper.py::difficulty_for()` returns a fixed configured float per Bloom level (e.g. Remember → −2.0, Create → 2.5); `bucket_for()` returns the parallel easy/medium/hard bucket consumed by `feature_builder.py`. | **Our Original Contribution** — explicitly *not* an established IRT calibration technique; see `docs/RESEARCH_CONTRIBUTION.md` for assumptions and limitations. |

---

## 6. K-Means Clustering

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| K-means clustering algorithm | MacQueen, J. B. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297. University of California Press. | Foundational unsupervised partitioning algorithm minimizing within-cluster variance. | `irt/clustering.py::cluster_students()` calls `sklearn.cluster.KMeans(n_clusters=2, ...)` directly on the normalized feature matrix. | Standard Theory (algorithm); scikit-learn is the standard, widely-used implementation. |
| K-means++ centroid initialization | Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035. | Improves K-means convergence and initialization quality over naive random seeding. | Used implicitly — this is scikit-learn's `KMeans` default initialization strategy, invoked with `n_init=10` in `clustering.py` for stability across random restarts. | Standard Theory (library default) |
| Using a 2-cluster split of students (by feature vector) as a proxy for "ability grouping," which then drives item discrimination scoring | No published source establishes this exact pipeline; it is this project's design | Provides a data-driven strong/weak split usable with a small cohort, replacing the need for a large-sample statistical calibration of item discrimination. | `clustering.py::cluster_students()` produces `ClusterResult`; `strong_cluster_id`/`weak_cluster_id` are then assigned by comparing average `total_correct` (never assumed from the raw sklearn label). | **Our Original Contribution** |

---

## 7. Classical Item-Discrimination Index (Ebel's D-index)

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Upper-group/lower-group item discrimination index | Ebel, R. L., & Frisbie, D. A. (1991). *Essentials of Educational Measurement* (5th ed.). Prentice-Hall. | Classical classroom-testing method: `D = (accuracy of top-scoring group) − (accuracy of bottom-scoring group)`, historically computed on a fixed top/bottom 27% split by total score, with published quality bands (excellent/good/moderate/poor/negative). | `irt/config.py::DISCRIMINATION_QUALITY_THRESHOLDS` is explicitly aligned to Ebel's classification bands (0.40/0.30/0.20/0.00 cut points); `segregation.py::classify_discrimination()` applies them. | Standard Theory (the index formula and quality bands) |
| Substituting a KMeans-derived, multi-feature strong/weak split for Ebel's fixed top/bottom 27%-by-score split | No published source; this project's adaptation | Computes the *same* accuracy-gap formula as Ebel's D-index, but over a cluster membership derived from `clustering.py` (percentage, IQ, and per-difficulty-band accuracy jointly) rather than a single-variable score cut, and over the whole cluster rather than a fixed percentile. | `irt/segregation.py::compute_segregation_scores()` computes `strong_accuracy − weak_accuracy` per question using `ClusterResult` membership. | **Our Original Contribution** (shares Ebel's underlying logic; the population-split method is this project's engineering adaptation) |

---

## 8. Bayesian Knowledge Tracing (BKT)

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Bayesian Knowledge Tracing model | Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278. | Sequential probabilistic model that updates a per-skill mastery estimate after each observed response, using learn/guess/slip parameters. | Referenced as the *downstream consumer* of this repository's output — BKT is already implemented in the Synapse quiz portal (`knowledge.service.ts`) and is **not part of this repository**. `mastery_initializer.py`'s output (`MasteryInitializationResult`) is designed to seed BKT's starting mastery prior per concept. | Standard Theory (BKT itself); this repository does not implement or modify BKT — see `docs/INTEGRATION_GUIDE.md`. |
| Using a data-informed prior (rather than a flat 0.5) to seed a sequential Bayesian tracker | General Bayesian-inference practice (informative vs. uninformative priors); no single source claims this specific application | Motivates *why* `mastery_initializer.py` exists at all — to give BKT a better starting point than an uninformative uniform prior. | `mastery_initializer.py::initialize_mastery()` blends observed concept accuracy with θ-implied accuracy as a Beta-Bernoulli-style shrinkage prior (§9 below). | **Our Original Contribution** |

---

## 9. Bayesian Shrinkage / Beta-Bernoulli Blending

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Beta-Bernoulli conjugate posterior mean as a precision-weighted blend of prior and observed data | Standard Bayesian statistics (e.g. Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press — general reference for conjugate-prior shrinkage estimators) | A Beta-Bernoulli posterior mean has the form `weight_observed × observed + (1 − weight_observed) × prior`, where `weight_observed = n/(n+K)` for effective prior strength `K`. | `mastery_initializer.py::initialize_mastery()` implements exactly this structure: `weight_observed = n_attempted / (n_attempted + MASTERY_PRIOR_STRENGTH)`, blending `observed_accuracy` with `theta_implied_accuracy`. | Standard Theory (the shrinkage structure); the specific choice to use *2PL-implied accuracy* as the prior mean, and `MASTERY_PRIOR_STRENGTH=3.0` as the effective sample size, are this project's engineering decisions. |

---

## 10. Knowledge Graphs in Education

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Knowledge graph representations for student modeling | Sun, K., Liu, Y., Guo, Z., & Wang, C. (2020). Application of knowledge graph in education. In *2020 International Conference on Big Data & Artificial Intelligence & Software Engineering (ICBASE)*, 344–347. IEEE. — representative of the general literature on knowledge-graph-based student/concept modeling in intelligent tutoring systems | Frames per-concept mastery tracking as nodes/edges in a graph connecting students, concepts, and prerequisite relationships. | Referenced as the downstream structure `mastery_initializer.py`'s output seeds (the "Student Knowledge Graph," implemented elsewhere in Synapse, not in this repository). This repository produces per-concept scalar mastery values (`ConceptMastery.initial_mastery`) that populate graph node attributes; it does not implement graph storage or traversal itself. | Standard Theory (concept); graph implementation is outside this repository's scope — see `docs/INTEGRATION_GUIDE.md`. |

---

## 11. Adaptive Learning / Computerized Adaptive Testing (CAT)

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Computerized Adaptive Testing | Wainer, H. (Ed.). (2000). *Computerized Adaptive Testing: A Primer* (2nd ed.). Lawrence Erlbaum Associates. | Standard framework for selecting each subsequent test item based on the current θ estimate, to maximize measurement precision per item administered. | Cited as a **future-work direction** (`docs/FUTURE_WORK.md`) — this repository currently estimates θ from a fixed diagnostic set; it does not perform item selection or adaptive item sequencing. | Standard Theory (not yet implemented in this repository) |
| Adaptive learning systems generally | VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist*, 46(4), 197–221. | Establishes the pedagogical motivation for adaptive, individualized instruction over fixed-sequence content delivery. | Motivates the overall Synapse platform goal stated in `docs/README.md`'s Motivation section; this repository is one input (θ, initial mastery) to that larger adaptive system. | Standard Theory (motivation only) |

---

## 12. Student Modelling and Educational Measurement / Psychometrics (General)

| Component | Reference | Purpose | How Used In This Project | Standard Theory / Our Contribution |
|---|---|---|---|---|
| Educational measurement foundations | Ebel & Frisbie (1991) (also listed in §7) | General classroom-assessment measurement theory: reliability, item analysis, discrimination, difficulty. | Grounds the overall framing of "item difficulty" and "item discrimination" as concepts throughout the codebase's docstrings and this documentation set. | Standard Theory |
| Psychometric test theory | Baker (2001); Hambleton et al. (1991) (also listed in §1) | General psychometric measurement principles underlying latent-trait estimation. | Grounds the entire Hybrid IRT design rationale. | Standard Theory |
| Student modeling in intelligent tutoring systems | VanLehn (2011) (also listed in §11); Corbett & Anderson (1995) (also listed in §8) | Establishes the broader field this project's output feeds into — computational models of what a student knows. | Frames θ and per-concept mastery as *student model* outputs consumed by Synapse's downstream adaptive components. | Standard Theory (framing) |

---

## Summary Table: Standard Theory vs. Our Contribution, By Module

| Module | Primary technique | Classification |
|---|---|---|
| `bloom_mapper.py` | Bloom's Taxonomy (levels) → used as a difficulty proxy | Bloom's Taxonomy = Standard Theory; using it as `b` = **Our Contribution** |
| `feature_builder.py` | Feature engineering, z-score normalization, cohort-mean imputation | Standard statistical technique; specific feature choices = **Our Contribution** |
| `clustering.py` | K-Means (MacQueen, 1967; scikit-learn) | K-Means = Standard Theory; using it to define strong/weak ability groups = **Our Contribution** |
| `segregation.py` | Ebel's D-index formula, computed over KMeans clusters | D-index formula = Standard Theory; KMeans-based group definition = **Our Contribution** |
| `item_parameters.py` | Software architecture (decoupling seam) | **Our Contribution** (engineering design, not a psychometric technique) |
| `theta.py` | 2PL model + Newton-Raphson MLE | Standard Theory (unmodified classical estimation, given fixed a/b) |
| `mastery_initializer.py` | 2PL curve reuse + Beta-Bernoulli-style shrinkage blending | 2PL and Bayesian shrinkage structure = Standard Theory; the specific blend formula and its role bridging θ→per-concept mastery = **Our Contribution** |

See [`docs/RESEARCH_CONTRIBUTION.md`](RESEARCH_CONTRIBUTION.md) for the full discussion of advantages, limitations, assumptions, and validation status of the contributions marked above.

## Reference List (APA 7th)

Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives.* Longman.

Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms* (pp. 1027–1035). Society for Industrial and Applied Mathematics.

Baker, F. B. (2001). *The basics of item response theory* (2nd ed.). ERIC Clearinghouse on Assessment and Evaluation, University of Maryland.

Baker, F. B., & Kim, S.-H. (2004). *Item response theory: Parameter estimation techniques* (2nd ed.). Marcel Dekker.

Bloom, B. S. (Ed.), Engelhart, M. D., Furst, E. J., Hill, W. H., & Krathwohl, D. R. (1956). *Taxonomy of educational objectives: The classification of educational goals, handbook I: Cognitive domain.* David McKay Company.

Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction, 4*(4), 253–278.

Ebel, R. L., & Frisbie, D. A. (1991). *Essentials of educational measurement* (5th ed.). Prentice-Hall.

Embretson, S. E., & Reise, S. P. (2000). *Item response theory for psychologists.* Lawrence Erlbaum Associates.

Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian data analysis* (3rd ed.). CRC Press.

Hambleton, R. K., Swaminathan, H., & Rogers, H. J. (1991). *Fundamentals of item response theory.* Sage Publications.

Lord, F. M. (1980). *Applications of item response theory to practical testing problems.* Lawrence Erlbaum Associates.

Lord, F. M., & Novick, M. R. (1968). *Statistical theories of mental test scores.* Addison-Wesley.

MacQueen, J. B. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281–297). University of California Press.

Sun, K., Liu, Y., Guo, Z., & Wang, C. (2020). Application of knowledge graph in education. In *2020 International Conference on Big Data & Artificial Intelligence & Software Engineering (ICBASE)* (pp. 344–347). IEEE.

VanLehn, K. (2011). The relative effectiveness of human tutoring, intelligent tutoring systems, and other tutoring systems. *Educational Psychologist, 46*(4), 197–221.

Wainer, H. (Ed.). (2000). *Computerized adaptive testing: A primer* (2nd ed.). Lawrence Erlbaum Associates.
