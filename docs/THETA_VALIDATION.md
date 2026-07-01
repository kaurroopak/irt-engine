# THETA_VALIDATION.md

## Technical Validation of the θ (Ability) Estimation Module — Synapse Hybrid IRT Engine

**Module under validation:** `irt/theta.py`
**Supporting modules:** `irt/bloom_mapper.py`, `irt/clustering.py`, `irt/segregation.py`, `irt/item_parameters.py`
**Status:** 81/81 automated tests passing at time of writing; all example values in this document were produced by running the actual codebase, not hand-calculated or fabricated (see §5 and §6 for the scripts used).

---

## 1. Introduction

### 1.1 What Item Response Theory is

Item Response Theory (IRT) is a family of statistical models that relate an examinee's probability of responding correctly to a test item to (a) a latent, unobservable trait of the examinee — conventionally called *ability* and denoted θ — and (b) one or more parameters describing the item itself, such as its difficulty and how well it discriminates between high- and low-ability examinees. The foundational formalization of this idea, and the term "item response theory" itself, are most closely associated with Lord's work relating classical test theory to a latent-trait framework (Lord, 1980; building on Lord & Novick, 1968). IRT models place items and examinees on the same latent scale, which is the property that makes the rest of this document possible: a question's difficulty (b) and a student's ability (θ) are directly comparable numbers.

### 1.2 Why latent ability is estimated rather than assumed

θ is *latent* — it is never observed directly. What is observed is a binary (or polytomous) response pattern: which items a student got right and which they got wrong. IRT estimates the value of θ that makes the observed response pattern most probable under the assumed response model (Baker, 2001; Hambleton, Swaminathan, & Rogers, 1991). This is a maximum-likelihood estimation problem, and it is the exact problem `theta.py` solves — see §3.

### 1.3 Why θ is preferable to a raw score

A raw score (number or percentage correct) treats every item as interchangeable: getting a hard item right counts the same as getting an easy item right. This creates two well-documented problems that IRT was specifically designed to correct (Hambleton et al., 1991; Embretson & Reise, 2000):

1. **Raw scores are test-dependent.** Two students who took different subsets of items — or the same items in a diagnostic that adapts to prior answers — cannot be compared on raw score alone, because the difficulty of the items each of them saw differs. θ, by contrast, is estimated *relative to the known difficulty of the specific items each student answered*, so two students who saw different item subsets remain comparable on the same θ scale.
2. **Raw scores are not interval-scaled.** The difference between 8/10 and 9/10 is not psychometrically equivalent to the difference between 3/10 and 4/10 — it depends heavily on which items were involved. θ, being estimated on a continuous latent scale via a probabilistic model, does not have this problem in the same way; Embretson & Reise (2000) discuss this at length as one of IRT's central motivations for adoption in psychological and educational measurement over classical test theory.

For Synapse specifically, this matters because the platform's downstream stages (Student Knowledge Graph initialization, Bayesian Knowledge Tracing, adaptive item selection) all need a *single, comparable* ability signal per student, independent of which particular diagnostic items that student happened to see — which is exactly the property raw percentage-correct does not reliably provide.

---

## 2. The Classical 2PL Model

### 2.1 The model

Synapse's θ estimation is based on the two-parameter logistic (2PL) model, one of the standard unidimensional dichotomous IRT models described in Hambleton et al. (1991) and Baker (2001):

```
P(correct | θ) = 1 / (1 + exp(-a(θ - b)))
```

### 2.2 Parameter interpretation

- **θ (theta) — ability.** A real-valued latent trait, typically scaled so that 0 represents average ability in the reference population and the scale is unbounded in principle (Baker, 2001). Higher θ means a higher probability of answering any given item correctly.
- **b — difficulty.** The point on the θ scale at which `P(correct) = 0.5`. This is directly visible by substituting `θ = b` into the equation: the exponent becomes 0, `exp(0) = 1`, and `P = 1/(1+1) = 0.5`. A question with a higher b requires a higher θ to have even odds of being answered correctly — i.e., it is harder. This is the standard IRT interpretation of b (Hambleton et al., 1991).
- **a — discrimination.** Controls the *steepness* of the curve at its inflection point (θ = b). A larger `|a|` means the probability of a correct response changes more sharply as θ crosses b — the item does a better job of separating examinees just below b from examinees just above it. A small `|a|` means the item's correctness is only weakly related to θ at all. Classical treatment of a's role in the ICC's slope is standard across the IRT literature (Baker, 2001; Hambleton et al., 1991).

### 2.3 The Item Characteristic Curve (ICC)

The function `P(correct | θ)` plotted against θ for fixed a, b is called the Item Characteristic Curve. It is a monotonic, S-shaped (logistic) curve: it approaches 0 as θ → −∞ and 1 as θ → +∞, is symmetric about the point (b, 0.5), and its slope at that inflection point is proportional to a (specifically, the slope at θ = b is a/4 for the logistic form above). This is the standard graphical device used throughout the IRT literature to communicate an item's behavior (Baker, 2001; Hambleton et al., 1991) and is exactly what `scripts/demo_theta.py`'s per-question `P(correct)` table (§5) samples at three points.

---

## 3. Newton–Raphson θ Estimation

### 3.1 The likelihood function

Given a student's responses to items with known (a, b), let `y_i ∈ {0, 1}` be the correctness of the i-th response, and `P_i(θ) = P(correct | θ, a_i, b_i)` as in §2.1. Assuming local independence between items given θ — the standard IRT assumption that, conditional on the examinee's ability, responses to different items are statistically independent (Lord, 1980; Hambleton et al., 1991) — the likelihood of the full response pattern is:

```
L(θ) = Π_i  P_i(θ)^{y_i} · (1 - P_i(θ))^{1 - y_i}
```

and the log-likelihood, which is what is actually maximized numerically:

```
ℓ(θ) = Σ_i [ y_i·ln P_i(θ) + (1 - y_i)·ln(1 - P_i(θ)) ]
```

### 3.2 Gradient (score function)

Differentiating ℓ(θ) with respect to θ (a standard derivation reproduced in, e.g., Baker & Kim, 2004, for MLE ability estimation under IRT models) gives:

```
dℓ/dθ = Σ_i  a_i · (y_i - P_i(θ))
```

This is implemented in `theta.py`'s `_score_and_information()` as `score`.

### 3.3 Information (negative Hessian)

The second derivative:

```
d²ℓ/dθ² = - Σ_i  a_i² · P_i(θ) · (1 - P_i(θ))
```

Its negation, `I(θ) = Σ_i a_i² P_i(θ)(1-P_i(θ))`, is the **Fisher information** at θ — the standard quantity used both to drive the Newton-Raphson update and to compute the asymptotic standard error of the estimate (Baker & Kim, 2004; Hambleton et al., 1991):

```
SE(θ) = 1 / √I(θ)
```

This is implemented in `theta.py` as `_standard_error()`.

### 3.4 Newton–Raphson update

The standard univariate Newton-Raphson update for maximizing ℓ(θ) is:

```
θ_{t+1} = θ_t - ℓ'(θ_t) / ℓ''(θ_t) = θ_t + score(θ_t) / I(θ_t)
```

(the sign flips because `ℓ'' = -I`). This is exactly what `estimate_theta()`'s main loop computes as `step = score / information` before updating θ. Newton-Raphson MLE for ability estimation under logistic IRT models, using this exact score/information formulation, is the estimation approach described in Baker & Kim (2004) and is one of the standard ability-estimation procedures discussed across the IRT literature (Hambleton et al., 1991; Embretson & Reise, 2000).

### 3.5 Why the log-likelihood is concave in θ (fixed a, b)

`I(θ) = Σ a_i² P_i(θ)(1-P_i(θ)) ≥ 0` for **every** real θ, because it is a sum of terms that are each a square (`a_i²`) times a probability-times-its-complement (`P(1-P)`), which is non-negative on `[0,1]` by construction. Since `d²ℓ/dθ² = -I(θ) ≤ 0` everywhere, ℓ(θ) is concave for all θ, **regardless of the sign of any individual a_i.** This is a specific property of the codebase's implementation being validated here — noted explicitly because segregation-derived discrimination values (§4) can be negative for a poorly-behaved question, and it is important to establish that Newton-Raphson remains well-behaved (globally convergent from any starting point, single interior maximum if one exists) even then.

### 3.6 Why Newton-Raphson specifically

A concave, twice-differentiable, one-dimensional objective is the textbook case for Newton-Raphson: it converges quadratically near the optimum (each iteration roughly squares the number of correct digits), versus the linear convergence of gradient ascent, and it does not require the step-size tuning that gradient-based methods do. The empirical trace in §5 converges to 1e-5 tolerance in 6 iterations, which is consistent with the rapid convergence expected of Newton-Raphson on a well-conditioned concave function. This module-specific application (Newton-Raphson with a, b held fixed rather than jointly estimated) is discussed further in §4 as an engineering decision distinct from classical joint calibration.

---

## 4. Hybrid IRT Architecture — Synapse's Contribution

> **This section describes original engineering design work for the Synapse capstone project. It is not an established IRT model from the literature. Where classical IRT is referenced below, it is referenced explicitly as the baseline this architecture departs from — not as a description of what Synapse implements.**

### 4.1 Classical IRT: joint calibration

In classical 2PL practice, **all three parameters — a, b, and θ, for every item and every examinee — are estimated jointly from a single large response matrix**, typically via marginal maximum likelihood or Bayesian estimation across hundreds or thousands of examinees (Baker & Kim, 2004; Hambleton et al., 1991). This joint estimation is what makes classical IRT calibration data-hungry: item parameters are only estimated reliably with large, representative response samples.

### 4.2 Synapse's Hybrid IRT (this project's contribution)

Synapse's quiz portal is early-stage and does not yet have the response volume classical calibration requires. To make ability estimation usable from the first cohort of students onward, this project — under supervisor guidance — replaces statistical calibration of item parameters with two deterministic, rule-based substitutes, and estimates **only θ** statistically:

| Parameter | Classical IRT | Synapse Hybrid IRT (this project) |
|---|---|---|
| Difficulty (b) | Estimated from response data (e.g. via MML) | **Read directly from each question's Bloom's Taxonomy level**, via a configurable mapping (`bloom_mapper.py`) |
| Discrimination (a) | Estimated from response data (e.g. via MML) | **Computed as the accuracy gap between a KMeans-derived strong/weak student cluster** (`clustering.py`, `segregation.py`) |
| Ability (θ) | Estimated from response data | **Estimated via Newton-Raphson MLE**, exactly as in classical IRT (§3), but with a and b held fixed as known inputs rather than jointly estimated |

Bloom's Taxonomy itself (Bloom et al., 1956; revised by Anderson & Krathwohl, 2001) is established educational literature describing a hierarchy of cognitive-process complexity (Remember < Understand < Apply < Analyze < Evaluate < Create). **Using it as a proxy for statistical item difficulty is this project's engineering decision, not a technique drawn from the IRT literature** — it substitutes an expert-assigned, theory-grounded ordinal difficulty signal for a data-estimated one, on the assumption (not statistically verified at this stage of the project) that a question requiring "Analyze"-level cognition is harder, on average, than one requiring "Remember"-level cognition for the same content.

Likewise, using the accuracy gap between two KMeans-identified sub-populations as a stand-in for discrimination is **an engineering approximation introduced by this project**. It shares its underlying logic with Ebel's classical item-discrimination index — upper-group accuracy minus lower-group accuracy, historically computed on a fixed top/bottom 27% split of examinees by total score (see `segregation.py`'s docstring and `config.py`'s threshold citation to Ebel & Frisbie, 1991) — but substitutes a multi-feature KMeans split (percentage, IQ, and per-difficulty-band accuracy jointly) for the single-variable top/bottom split Ebel's method uses, and computes it over the *whole* cluster population per question, at any cohort size, rather than requiring a large enough sample to trust a percentile cut.

### 4.3 Why this is useful for a data-limited educational platform

The practical benefit is that θ becomes estimable **from the very first quiz a student takes**, without waiting for a large enough response corpus to calibrate a and b statistically. The tradeoff, made explicit rather than hidden, is that a and b are only as good as the Bloom mapping and the cluster separation quality — they are not empirically validated against actual item performance the way classically-calibrated parameters are. As response volume grows, replacing this Hybrid IRT stage with classical joint calibration (or a Bayesian empirical-Bayes blend of the two) is a natural evolution path for the project, but is out of scope for the current implementation.

---

## 5. Numerical Example

Taken directly from `scripts/demo_theta.py`'s cohort (student **S3**), using the actual item parameters the pipeline produced for that run (Bloom-derived b, segregation-derived a):

| Question | a | b | S3's response |
|---|---|---|---|
| Q1 | 1.00 | -2.00 | Correct |
| Q2 | 0.50 | -1.00 | Correct |
| Q3 | 0.50 | 0.00 | Incorrect |
| Q4 | 0.25 | 1.00 | Correct |
| Q5 | -0.50 | 2.00 | Incorrect |

Newton-Raphson iteration trace (θ₀ = 0.0, produced by directly instrumenting `theta.py`'s actual `probability_correct()` and score/information formulas — not hand-computed):

| Iteration | score (dℓ/dθ) | information I(θ) | step | θ after step | ℓ(θ) |
|---|---|---|---|---|---|
| 1 | 0.564047 | 0.290781 | +1.000000 (capped) | 1.000000 | -3.433353 |
| 2 | 0.306897 | 0.227456 | +1.000000 (capped) | 2.000000 | -3.003150 |
| 3 | 0.103126 | 0.181986 | +0.566669 | 2.566669 | -2.801911 |
| 4 | 0.006370 | 0.159755 | +0.039876 | 2.606545 | -2.771482 |
| 5 | 0.000030 | 0.158243 | +0.000191 | 2.606736 | -2.771355 |
| 6 | 0.000000 | 0.158236 | +0.000000 | 2.606736 | -2.771355 |

**Final θ = 2.607**, converged = True, iterations = 6, SE(θ) = 1/√0.158236 ≈ **2.514**.

**Interpretation:** Iterations 1–2 hit the configured `THETA_MAX_STEP = 1.0` cap (§7.3) — the raw Newton step would have overshot further, and the cap keeps early iterations conservative on a function that is still far from its optimum. From iteration 3 onward the step size shrinks rapidly (0.567 → 0.040 → 0.0002), which is the expected signature of quadratic convergence near a maximum. θ ≈ 2.61 reflects that S3 answered all items up to and including the hardest positively-discriminating item (Q4, b = +1.00) correctly, and only "missed" Q3 (a moderate item) and Q5 — but Q5 has *negative* discrimination (a = −0.50), meaning weak-cluster students answered it correctly more often than strong-cluster students in this cohort; getting Q5 "wrong" is therefore not weighted as strong evidence against high ability, which is precisely why final θ lands well above the difficulty of every item S3 answered correctly rather than being pulled down further by the Q5 miss. The large standard error (≈2.5) reflects the small item count (5) available for this synthetic demo cohort, not a defect in the estimator — SE shrinks with more responses, as `I(θ)` accumulates more per-item information terms.

---

## 6. Extreme Response Patterns

### 6.1 Why all-correct and all-incorrect patterns have no finite MLE

If a student answers **every** item correctly, then `y_i = 1` for all i, and the score function becomes:

```
dℓ/dθ = Σ_i a_i (1 - P_i(θ))
```

For any item with `a_i > 0`, `(1 - P_i(θ)) → 0⁺` as `θ → +∞` but is **never exactly zero** for any finite θ (the logistic function is asymptotic, not bounded). This means the score is **strictly positive for every finite θ** — the log-likelihood is monotonically increasing in θ, with a supremum only as `θ → +∞`. There is no finite θ at which the derivative is zero, so **no finite maximum-likelihood estimate exists.** The symmetric argument holds for an all-incorrect pattern as `θ → -∞`. This is a well-known, expected degeneracy of maximum likelihood estimation for perfect and zero response patterns under logistic IRT models (Baker & Kim, 2004, discuss estimation difficulties at the extremes of the ability distribution generally; the non-existence result itself follows directly from the algebra of the logistic score function above).

`theta.py`'s `_extreme_pattern_theta()` detects this condition directly — checking whether the response set contains only one outcome value, which also correctly subsumes the single-response case, since one response is trivially "all correct" or "all incorrect" — rather than letting Newton-Raphson iterate toward a boundary it will never reach within a finite iteration budget.

### 6.2 Why `θ = ±4, converged = False` is the mathematically correct output

Given that no finite MLE exists, the module reports θ at a **configured clamp boundary** (`THETA_EXTREME_PATTERN_CLAMP = 4.0`, matching `THETA_MAX`/`THETA_MIN`) with `converged = False`. Both parts of this are deliberate:

- **θ = ±4 is not a claim that 4.0 is the maximum-likelihood ability.** It is a practical reporting boundary, chosen (per `config.py`) because ±4 logits covers effectively the entire practically relevant range of a logistic ability distribution — beyond it, `P(correct)` for any reasonably-discriminating item is already indistinguishable from 0 or 1 to more precision than any real assessment can resolve. This mirrors common practice in operational IRT software, which typically imposes a finite ability range for exactly this reason (Baker & Kim, 2004, note the general need for bounding numerical ability estimates in practice).
- **`converged = False` is factually accurate, not a failure state to be alarmed about.** Newton-Raphson did not converge to an interior stationary point — because none exists — and the module reports that honestly rather than reporting `converged = True` at an arbitrary boundary. Downstream consumers (`mastery_initializer.py`, §8–9) can use `converged` to distinguish "a confidently interior-estimated ability" from "a boundary estimate driven by a perfect or zero score," which is meaningfully different information for, e.g., how much weight to place on this θ when initializing concept mastery.

This confirms empirically the demo's S4 result in the previous module's validation run (`scripts/demo_theta.py`): S4 answered all 5 quality-bank items correctly and received `θ = 4.00, converged = False, iterations = 0` — the "0 iterations" reflects that the extreme-pattern check runs *before* the Newton-Raphson loop, so no iterations are spent approaching an unreachable interior optimum.

---

## 7. Numerical Stability

Each safeguard below addresses a specific, concrete failure mode of a naive Newton-Raphson-on-logistic-likelihood implementation:

### 7.1 Exponent clamping (`THETA_EXPONENT_CLAMP = 35.0`)

`probability_correct()` computes `exp()` of `∓a(θ-b)`. Without a bound, a large `|a(θ-b)|` (e.g. a highly discriminating item far from a student's ability) causes `exp()` to either overflow (a positive exponent beyond ~709 for float64) or silently underflow to exactly `0.0` (a very negative exponent) in a way that is numerically correct for the *sigmoid* but can propagate incorrectly if the raw `exp()` value itself is used elsewhere. Clamping the exponent to ±35 keeps `exp(35) ≈ 1.6×10^15` — already representing a probability of `1` or `0` to far more precision than any real response can resolve — while staying safely inside float64 range. The implementation additionally evaluates the sigmoid via two algebraically equivalent forms depending on the sign of the (clamped) exponent, so `exp()` is only ever called on a non-positive argument, eliminating the overflow direction entirely rather than merely bounding it.

### 7.2 Probability clamping (`eps = 1e-9` in `_score_and_information`)

`ln(P)` and `ln(1-P)` are undefined (−∞) at `P = 0` or `P = 1` exactly. A response pattern combined with extreme (a, θ-b) values can drive the sigmoid arbitrarily close to those bounds. Clamping `P` to `[1e-9, 1 - 1e-9]` before taking a logarithm keeps `ℓ(θ)` finite and prevents a single well-fit response from producing a `NaN` log-likelihood that would corrupt the sum across all of a student's responses.

### 7.3 θ bounds and per-iteration step cap (`THETA_MIN/MAX = ±4.0`, `THETA_MAX_STEP = 1.0`)

Even for response patterns with an interior maximum, early Newton-Raphson iterations starting from `θ₀ = 0` can produce large steps when the current θ estimate is far from the optimum and local curvature is shallow (see §5, iterations 1–2, where the cap actually engaged in a normal, converging example). Without a per-step cap, a single early iteration on a data-sparse or poorly-conditioned response set could jump to an extreme θ from which recovery is slow or numerically unstable. Capping both the per-step move and the absolute θ range keeps every intermediate value in a well-behaved numeric region, without materially affecting the converged answer for well-conditioned inputs — as shown in §5, capped early steps still converge to the same final θ as an uncapped trajectory would.

### 7.4 Convergence tolerance (`THETA_CONVERGENCE_TOLERANCE = 1e-5`)

Iteration stops once `|score| < 1e-5`. This is a standard MLE stopping criterion (the gradient is negligibly close to zero, i.e. at a stationary point) tight enough that further iteration would not meaningfully change θ (as seen in §5, iteration 6 already shows a step of `0.000000` at 6-decimal precision), while loose enough to terminate in a small, bounded number of iterations rather than chasing floating-point noise.

### 7.5 Maximum iterations (`THETA_MAX_ITERATIONS = 50`)

A hard iteration ceiling guarantees `estimate_theta()` always terminates and returns a result (with `converged = False` if the ceiling is hit without meeting tolerance) rather than looping indefinitely on a pathological or misconfigured input that the extreme-pattern check (§6) does not catch. 50 iterations is far beyond what quadratically-convergent Newton-Raphson needs for any mixed response pattern in practice (§5 converges in 6), so this ceiling is a safety bound, not an expected operating point.

---

## 8. Educational Interpretation

The following bands are used **only** for human-readable reporting (e.g. `scripts/demo_theta.py`'s printed interpretation column) and are **not** a psychometric classification with any statistical meaning attached — they are a project-specific convenience for presenting θ to non-technical stakeholders (students, teachers, supervisors) who are not expected to interpret a raw logit value.

| θ range | Educational category |
|---|---|
| θ > 2 | Very High Ability |
| 1 ≤ θ ≤ 2 | High Ability |
| 0 ≤ θ < 1 | Average |
| -1 ≤ θ < 0 | Needs Improvement |
| θ < -1 | Needs Significant Support |

**These are educational categories chosen by this project for presentation purposes, not standardized psychometric cutoffs from the IRT literature.** No published source is cited for these specific boundary values because none is being claimed; they are a project design decision, analogous to how a school might band percentage grades into letter grades. (Note: the current codebase's `scripts/demo_theta.py` uses a simpler two-cutoff banding for its own demo output; the five-band table above is the more granular version requested for this document and is recommended as the version to carry forward into any future reporting UI.)

---

## 9. Relationship with the Student Knowledge Graph

### 9.1 How θ initializes the Student Knowledge Graph

θ is a **single, unidimensional, whole-quiz** ability estimate. The Student Knowledge Graph, by contrast, tracks **per-concept mastery** — a separate probability for every concept a student may be assessed on (e.g. "Ohm's Law", "Series Circuits"). `mastery_initializer.py` (§ next module) is the bridge between the two: it combines the single θ value with **per-concept performance** (how the student did specifically on questions tagged with that concept) to produce an initial, concept-level mastery estimate for every concept touched by the diagnostic — rather than assigning every concept the same initial value derived from θ alone. This is discussed further, with the exact combination formula, in `mastery_initializer.py`'s own documentation.

### 9.2 Why θ is not the same as concept mastery

θ answers "how able is this student, overall, on this assessment" — it is a property of the *student*, estimated from *all* their responses jointly, on a continuous, unbounded logit scale. Concept mastery answers "how likely is this student to correctly apply *this specific concept* right now" — it is a property of the *(student, concept) pair*, is bounded to `[0, 1]` as a probability (matching the Bayesian Knowledge Tracing framework already implemented in the quiz portal's `knowledge.service.ts`), and is expected to *change* over time as the student engages with that specific concept — which θ, as a single point-in-time ability snapshot from one diagnostic, does not do on its own. Conflating the two would mean, for example, a generally high-ability student is assumed equally strong on every concept regardless of whether they specifically demonstrated that on the diagnostic — exactly the granularity gap `mastery_initializer.py` exists to close by blending θ with per-concept evidence.

### 9.3 Why BKT updates concept mastery after initialization

θ is estimated once, from the diagnostic quiz. Bayesian Knowledge Tracing, as already implemented in the quiz portal, is a *sequential* model — it updates a per-concept mastery probability after every subsequent relevant response, using the classical BKT update equations (learn/guess/slip parameters) to incorporate new evidence over time (Corbett & Anderson's original BKT formulation is the standard reference for this class of model, though the portal's `knowledge.service.ts` implementation is this project's own code, not being validated in this document). θ and the diagnostic-derived initial mastery values exist specifically to give BKT a well-informed **starting point** rather than an uninformative uniform prior (e.g. 0.5 for every concept, for every student) — after which BKT, not θ or `mastery_initializer.py`, is responsible for keeping mastery current as the student continues to interact with the platform.

### 9.4 Complete pipeline

```
Student
   ↓
Diagnostic Quiz
   ↓
Bloom Mapping                (bloom_mapper.py    — CHANGE 1, difficulty b)
   ↓
Feature Builder               (feature_builder.py — per-student feature vector)
   ↓
KMeans                        (clustering.py      — strong/weak split)
   ↓
Segregation                   (segregation.py     — CHANGE 3, discrimination a)
   ↓
Question Parameters           (item_parameters.py — assembles (a, b) per question)
   ↓
Theta                         (theta.py            — CHANGE 4, this document)
   ↓
Mastery Initialization        (mastery_initializer.py — next module)
   ↓
Student Knowledge Graph       (quiz portal, already implemented)
   ↓
Bayesian Knowledge Tracing    (quiz portal, already implemented)
   ↓
Adaptive Learning
```

---

## 10. References

- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems.* Hillsdale, NJ: Lawrence Erlbaum Associates.
- Lord, F. M., & Novick, M. R. (1968). *Statistical Theories of Mental Test Scores.* Reading, MA: Addison-Wesley.
- Hambleton, R. K., Swaminathan, H., & Rogers, H. J. (1991). *Fundamentals of Item Response Theory.* Newbury Park, CA: Sage Publications.
- Baker, F. B. (2001). *The Basics of Item Response Theory* (2nd ed.). College Park, MD: ERIC Clearinghouse on Assessment and Evaluation, University of Maryland.
- Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). New York, NY: Marcel Dekker.
- Embretson, S. E., & Reise, S. P. (2000). *Item Response Theory for Psychologists.* Mahwah, NJ: Lawrence Erlbaum Associates.
- Ebel, R. L., & Frisbie, D. A. (1991). *Essentials of Educational Measurement* (5th ed.). Englewood Cliffs, NJ: Prentice-Hall. — *cited for the classical item-discrimination index framework (upper-group minus lower-group accuracy) that `segregation.py`'s quality thresholds are aligned to; see `config.py`'s inline citation.*
- Bloom, B. S. (Ed.), Engelhart, M. D., Furst, E. J., Hill, W. H., & Krathwohl, D. R. (1956). *Taxonomy of Educational Objectives: The Classification of Educational Goals, Handbook I: Cognitive Domain.* New York, NY: David McKay Company.
- Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A Taxonomy for Learning, Teaching, and Assessing: A Revision of Bloom's Taxonomy of Educational Objectives.* New York, NY: Longman. — *the revised, verb-based taxonomy (Remember, Understand, Apply, Analyze, Evaluate, Create) that `bloom_mapper.py`'s configured levels follow.*
- Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4), 253–278. — *the standard reference for the Bayesian Knowledge Tracing framework mentioned in §9.3; the quiz portal's BKT implementation itself is this project's own code and is not the subject of this validation document.*

**A note on citation scope:** every equation in §2 and §3 (the 2PL model, its log-likelihood, gradient, information, and the Newton-Raphson update) is standard IRT theory, attributable to the general body of literature represented by the references above rather than to any single source claiming original authorship of the algebra. §4 (Hybrid IRT architecture), §5's specific numerical trace, §6.2's clamp-boundary reporting convention, and §8's educational bands are **this project's own design decisions**, explicitly *not* attributed to the literature above, per the distinction requested for this document.
