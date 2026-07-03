# RESEARCH_CONTRIBUTION.md

**This document exists to draw one line clearly and repeatedly: everything under "Our Hybrid IRT" below is this project's own proposed engineering framework. It has not been published, peer-reviewed, or validated against real classroom data at the time of writing. Where classical IRT is described, it is described as the established baseline this project departs from — never as a description of what this project's Hybrid IRT architecture already is in the literature.**

## 1. Established Theory

The following are unmodified applications of published psychometric and statistical methods, fully cited in `docs/LITERATURE_REVIEW.md`:

- The **two-parameter logistic (2PL) IRT model**: `P(correct) = 1/(1+exp(-a(θ-b)))` (Hambleton, Swaminathan, & Rogers, 1991; Baker, 2001).
- **Maximum likelihood estimation** of θ from a response pattern given known item parameters (Baker & Kim, 2004).
- **Newton-Raphson** as the numerical method for that estimation, including the score function, Fisher information, and standard-error formulas (Baker & Kim, 2004) — all implemented exactly as derived in `docs/THETA_VALIDATION.md` §3.
- **Bloom's Taxonomy** as a classification of cognitive-process complexity (Bloom et al., 1956; Anderson & Krathwohl, 2001) — the taxonomy itself, independent of how this project repurposes it.
- **KMeans clustering** as an unsupervised partitioning algorithm (MacQueen, 1967; Lloyd, 1982; implemented here via scikit-learn) — the algorithm itself, independent of how this project repurposes its output.
- **Bayesian Knowledge Tracing** (Corbett & Anderson, 1994) — implemented in the Synapse quiz portal, outside this repository; this project's `mastery_initializer.py` produces its *starting* values only, and does not reimplement or modify BKT itself.

## 2. Engineering Decisions

These are implementation-level choices made while building this project, distinct from the Hybrid IRT architecture's headline design (§3) — smaller, more local decisions that could be changed without altering the overall approach:

| Decision | Where | Rationale |
|---|---|---|
| `previous_class_percentage` treated as already 0–100, no max-marks normalization | `feature_builder.py` | Confirmed data-source decision; `StudentProfile.class9_marks` is stored as a percentage. |
| `iq_score` optional, cohort-mean-imputed when missing | `feature_builder.py` | The psychometric test is a separate, in-progress workstream; the feature vector shouldn't be blocked on it. |
| Easy/medium/hard accuracy buckets derived from Bloom level, not a separate raw difficulty column | `feature_builder.py` | Keeps exactly one source of truth for "how hard is this question," consistent with the difficulty (b) parameter itself. |
| Discrimination-quality thresholds aligned to Ebel & Frisbie (1991) rather than the originally suggested 0.50/0.30/0.10/0.00 | `config.py`, `segregation.py` | The Ebel-index formula (upper-group minus lower-group accuracy) is structurally identical to this project's segregation score, making it the closest available literature anchor for quality thresholds. |
| θ clamped to [-4, 4]; per-iteration Newton step capped at 1.0 | `theta.py` | Numerical safety bounds — standard practice in operational IRT software, not a change to the estimation method itself. |
| All-correct/all-incorrect patterns detected and clamped before the Newton-Raphson loop runs | `theta.py` | An MLE non-existence result (§6, `docs/THETA_VALIDATION.md`) handled explicitly rather than left to an iteration-count timeout. |
| Fixed reference discrimination (a = 1.0) used when computing theta-implied accuracy | `mastery_initializer.py` | Per-item segregation-based discrimination is outside this module's declared input set (θ, concept accuracy, Bloom difficulty only); a=1.0 is a documented simplification, not a claim about any question's true discrimination. |

## 3. Our Hybrid IRT Contribution

### 3.1 Classical IRT (established baseline)

```
Difficulty (b)     ─┐
Discrimination (a)  ├─►  jointly estimated from a large response matrix
Ability (θ)         ─┘   (thousands of responses, typically via MML or Bayesian estimation)
```

### 3.2 Our Hybrid IRT (this project's proposed architecture)

```
Difficulty (b)      ─►  read from Bloom's Taxonomy level (bloom_mapper.py)
Discrimination (a)  ─►  computed from KMeans strong/weak cluster accuracy gap (clustering.py + segregation.py)
Ability (θ)         ─►  estimated statistically via 2PL MLE / Newton-Raphson (theta.py) — same math as classical IRT
Mastery              ─►  initialized per-concept from θ + concept accuracy + Bloom difficulty (mastery_initializer.py),
                          then updated over time by Bayesian Knowledge Tracing (quiz portal, outside this repo)
```

**Only θ is estimated statistically from response data in this architecture. Difficulty and discrimination are deterministic, rule-based substitutes designed specifically for the cold-start problem of a new educational platform with limited response history.**

This is stated as plainly as possible because it is the central claim of this document: **the overall five-stage pipeline above — Bloom→difficulty, KMeans/segregation→discrimination, Newton-Raphson→ability, blended-shrinkage→mastery — is not an established IRT model. No citation in `docs/LITERATURE_REVIEW.md` describes this combination.** It is this project's synthesis of established statistical machinery (2PL, MLE, Newton-Raphson, KMeans) applied to substitute inputs (Bloom labels, cluster splits) that the literature does not describe using for this purpose.

## 4. Advantages

- **Usable from the first cohort of students**, without waiting for the response volume classical calibration needs.
- **Fully deterministic and auditable** given a fixed configuration (`config.RANDOM_STATE` and the Bloom map) — a teacher or supervisor can trace exactly why a question got a given difficulty or discrimination value, unlike a black-box MML fit.
- **Modular by construction** (see `docs/ARCHITECTURE.md`): the Bloom and KMeans substitutes can each be replaced independently, without touching θ estimation, once better data or methods are available (see §7 below and `docs/FUTURE_WORK.md`).
- **θ estimation itself is not a novel or risky component** — it reuses the exact, well-studied classical MLE/Newton-Raphson machinery, isolating this project's actual novelty (the substitute parameter sources) from its statistical estimation step.

## 5. Limitations

- **Bloom-derived difficulty is an unvalidated assumption.** There is no statistical evidence in this project, at time of writing, that "Analyze" questions are reliably harder than "Apply" questions *for the same content* in this specific student population — the assumption is pedagogically motivated but not empirically checked.
- **KMeans-derived discrimination depends on cluster quality.** If the strong/weak split itself is weak (e.g. a genuinely unimodal, non-bimodal student population, or a feature vector that doesn't actually separate ability well), every downstream segregation score inherits that weakness. `clustering.py`'s `ClusteringFailedError` catches the most extreme failure (identical feature vectors), but a *poor-but-technically-valid* split is not detected or flagged anywhere in the current pipeline.
- **Small-sample instability.** The numerical example in `docs/THETA_VALIDATION.md` §5 shows a standard error of ≈2.5 from only 5 items — entirely expected behavior for MLE with few observations, but a reminder that θ estimates from a short diagnostic quiz carry real uncertainty that downstream consumers (e.g. `mastery_initializer.py`) should be aware of (it is — `theta_converged` and each concept's `n_attempted` are both exposed for exactly this reason).
- **The mastery-initialization blend (`MASTERY_PRIOR_STRENGTH`) is a hand-chosen constant**, not fit to any data — it encodes a reasonable-sounding but unvalidated belief about how much to trust few versus many diagnostic observations per concept.
- **No comparison, in this project, against classical IRT's actual item parameters.** Because classical calibration is exactly what this project is trying to avoid needing (due to lack of data), there is currently no dataset large enough within this project to check how far Bloom-derived b and KMeans-derived a diverge from what statistical calibration would have produced on the same items.

## 6. Assumptions

Stated explicitly, since they are load-bearing for every result this engine produces:

1. Bloom's Taxonomy level is monotonically related to empirical item difficulty, for the questions in this system.
2. The student features used for clustering (previous percentage, IQ, per-bucket accuracy) meaningfully separate "generally stronger" from "generally weaker" students, and that separation is relevant to *item-level* discrimination, not just overall ability.
3. Local independence between items given θ (the standard IRT assumption underlying the log-likelihood in `docs/THETA_VALIDATION.md` §3.1) holds for this item bank.
4. A single unidimensional θ per student is an adequate summary of "overall ability" for the purpose of seeding, not fully determining, concept-level mastery (mitigated by `mastery_initializer.py` blending in concept-specific evidence, but not eliminated).

## 7. Potential Publication Value

If validated against real classroom data (see §8), this architecture could plausibly contribute to the applied educational-data-mining literature on **cold-start ability estimation** — a recognized practical problem (new platforms, new item banks, or new domains lack the response volume classical IRT calibration needs) that is less thoroughly addressed in the literature than mature-platform IRT calibration itself. The specific, potentially novel angle is the **combination** of an expert-taxonomy-based difficulty proxy with an unsupervised, multi-feature discrimination proxy, feeding an otherwise-unmodified classical MLE ability estimator — as opposed to, e.g., pure expert-judgment item parameters (no statistical estimation at all) or pure small-sample Bayesian calibration (statistical but still data-hungry, just with an informative prior). Whether this combination outperforms either of those simpler alternatives, on this project's actual data, is an open empirical question this document does not claim to have answered.

## 8. Future Validation Using Real Classroom Data

Concretely, as more response data accumulates through Synapse's use:

1. **Check the Bloom-difficulty assumption directly** — once enough responses exist per question, compute empirical difficulty (e.g. proportion correct, or a properly calibrated b) and compare it against the Bloom-mapped value. Systematic disagreement would be evidence the fixed Bloom map needs per-subject or per-chapter recalibration, not just a global constant table.
2. **Check the KMeans-segregation assumption directly** — compare segregation-derived a against a statistically calibrated a on the same items once feasible, to establish how well the KMeans proxy actually approximates classical discrimination.
3. **Validate the mastery-initialization blend** — compare BKT's mastery trajectory for students whose initial mastery came from this pipeline versus a naive uniform prior (0.5 or the BKT default of 0.2), to check whether the informed initialization actually improves early-session mastery-tracking accuracy, which is the entire practical justification for `mastery_initializer.py` existing.
4. **Re-run θ estimation in parallel with classical joint calibration** once volume allows, as a direct empirical comparison rather than a purely theoretical one.

None of the above validation has been performed as part of this project to date; all of it requires real classroom deployment and data collection first, which is out of scope for this repository (see `docs/FUTURE_WORK.md`).
