# API_REFERENCE.md

Complete public API for every module in `irt/`. Every signature, field name, and exception below was extracted directly from the current source (`inspect`-verified, not transcribed from memory), so this document should let another developer use the engine without reading the implementation.

---

## `bloom_mapper.py`

### `difficulty_for(bloom_level: str) -> float`
Returns the configured difficulty parameter (b) for a Bloom level, per `config.BLOOM_DIFFICULTY_MAP`.

- **Raises:** `UnknownBloomLevelError` if `bloom_level` is `None`, empty, or not a recognized level.
- **Example:**
  ```python
  from irt.bloom_mapper import difficulty_for
  difficulty_for("Understand")  # -> -1.0
  difficulty_for("Create")      # -> 2.5
  ```

### `bucket_for(bloom_level: str) -> str`
Returns the easy/medium/hard accuracy bucket for a Bloom level, per `config.BLOOM_DIFFICULTY_BUCKETS`.

- **Raises:** `UnknownBloomLevelError`, same conditions as above.
- **Example:** `bucket_for("Analyze")  # -> "hard"`

### `describe(bloom_level: str) -> BloomInfo`
Convenience: both difficulty and bucket in one lookup.

### `BloomInfo` (dataclass, frozen)
| Field | Type |
|---|---|
| `bloom_level` | `str` |
| `difficulty` | `float` |
| `bucket` | `str` |

### `UnknownBloomLevelError(ValueError)`
Raised by any function above on an unrecognized/missing Bloom level. Never silently defaults.

---

## `feature_builder.py`

### `StudentProfileRow` (dataclass, frozen)
| Field | Type | Notes |
|---|---|---|
| `student_id` | `str` | |
| `previous_class_percentage` | `Optional[float]` | Maps to `StudentProfile.class9_marks`; treated as already 0–100 |
| `iq_score` | `Optional[float]` | Optional by design — psychometric test is a separate workstream |

### `ResponseRow` (dataclass, frozen)
| Field | Type |
|---|---|
| `student_id` | `str` |
| `question_id` | `str` |
| `is_correct` | `bool` |
| `bloom_level` | `str` |

### `ImputationReport` (dataclass)
| Field | Type |
|---|---|
| `field_name` | `str` |
| `fill_value` | `float` |
| `imputed_student_ids` | `list[str]` |

Methods: `__bool__()` (truthy iff any imputation occurred), `as_warning() -> Optional[str]`.

### `FeatureMatrix` (dataclass)
| Field | Type |
|---|---|
| `student_ids` | `list[str]` |
| `matrix` | `np.ndarray`, shape `(n_students, 6)` |
| `field_names` | `tuple[str, ...]`, equals `config.FEATURE_VECTOR_FIELDS` |
| `imputations` | `list[ImputationReport]` |

Methods: `warnings() -> list[str]`, `as_dict_rows() -> list[dict[str, float]]`.

### `build_feature_matrix(profiles: Iterable[StudentProfileRow], responses: Iterable[ResponseRow]) -> FeatureMatrix`
Builds the 6-field feature vector (`previous_class_percentage, iq_score, total_correct, easy_accuracy, medium_accuracy, hard_accuracy`) for every student in `profiles`. Missing `iq_score`/`previous_class_percentage` are cohort-mean-imputed (recorded in `.imputations`). Responses for a `student_id` not in `profiles` are silently skipped (not fabricated into a new profile). Students with zero responses still get a row (all-zero accuracy fields).

- **Example:**
  ```python
  from irt.feature_builder import StudentProfileRow, ResponseRow, build_feature_matrix

  profiles = [StudentProfileRow("S1", 80.0, 110.0)]
  responses = [ResponseRow("S1", "Q1", True, "Apply")]
  fm = build_feature_matrix(profiles, responses)
  ```

### `normalize_feature_matrix(fm: FeatureMatrix) -> FeatureMatrix`
Z-score normalizes each column. Zero-variance columns become all-zeros (not `NaN`).

---

## `clustering.py`

### `cluster_students(feature_matrix: FeatureMatrix, raw_feature_matrix: Optional[FeatureMatrix] = None) -> ClusterResult`
Runs `KMeans(n_clusters=2, random_state=42)` on `feature_matrix` (must already be normalized — this function normalizes nothing). Labels the two clusters "strong"/"weak" by comparing average `total_correct`, computed from `raw_feature_matrix` if provided (real units), otherwise from `feature_matrix` itself (normalized units — documented, still internally consistent, less human-readable).

- **Raises:**
  - `EmptyFeatureMatrixError` — zero students.
  - `InsufficientStudentsError` — fewer than 2 students.
  - `ClusteringFailedError` — scikit-learn fails to produce 2 distinct clusters (e.g. all students have identical feature vectors).
  - `ValueError` — `raw_feature_matrix` given but its `student_ids` don't match `feature_matrix`'s.
- **Example:**
  ```python
  from irt.clustering import cluster_students
  result = cluster_students(normalized_fm, raw_fm)
  result.label_for("S1")           # -> "strong" or "weak"
  result.strong_student_ids()      # -> [...]
  result.statistics_for("strong")  # -> ClusterStatistics
  ```

### `ClusterStatistics` (dataclass, frozen)
| Field | Type |
|---|---|
| `cluster_id` | `int` |
| `n_students` | `int` |
| `avg_previous_class_percentage` | `float` |
| `avg_iq_score` | `float` |
| `avg_total_correct` | `float` |
| `avg_easy_accuracy` | `float` |
| `avg_medium_accuracy` | `float` |
| `avg_hard_accuracy` | `float` |

### `ClusterResult` (dataclass)
| Field | Type |
|---|---|
| `student_ids` | `List[str]` |
| `cluster_labels` | `np.ndarray`, raw sklearn ids (0/1), aligned to `student_ids` |
| `strong_cluster_id` | `int` |
| `weak_cluster_id` | `int` |
| `cluster_centroids` | `np.ndarray`, shape `(2, 6)` |
| `cluster_statistics` | `Dict[int, ClusterStatistics]`, keyed by raw cluster id |

Methods: `label_for(student_id) -> str`, `strong_student_ids() -> List[str]`, `weak_student_ids() -> List[str]`, `statistics_for("strong"|"weak") -> ClusterStatistics`.

### Exceptions
`EmptyFeatureMatrixError(ValueError)`, `InsufficientStudentsError(ValueError)`, `ClusteringFailedError(RuntimeError)`.

---

## `segregation.py`

### `classify_discrimination(segregation_score: float) -> str`
Maps a score to a quality label via `config.DISCRIMINATION_QUALITY_THRESHOLDS` — `"excellent"` (≥0.40), `"good"` (≥0.30), `"moderate"` (≥0.20), `"poor"` (≥0.00), `"negative"` (<0.00).

### `compute_segregation_score(question_id: str, responses_for_question: Iterable[ResponseRow], cluster_result: ClusterResult) -> SegregationResult`
Strict single-question API.

- **Raises:** `InsufficientAttemptsError` if either cluster has zero attempts for this question.

### `compute_segregation_scores(cluster_result: ClusterResult, responses: Iterable[ResponseRow], question_ids: Optional[Iterable[str]] = None) -> SegregationBatchResult`
Batch API. Never raises for a single bad question — every question ends up in `.results` or `.skipped`. Pass `question_ids` explicitly to also flag questions with **zero** responses at all (otherwise they're simply absent from `responses` and would go unnoticed).

- **Example:**
  ```python
  from irt.segregation import compute_segregation_scores
  batch = compute_segregation_scores(cluster_result, responses)
  batch.sorted_by_segregation_score()  # highest -> lowest
  batch.flagged()                       # poor/negative questions
  batch.as_dict_by_question()           # {question_id: SegregationResult}
  ```

### `SegregationResult` (dataclass, frozen)
| Field | Type |
|---|---|
| `question_id` | `str` |
| `strong_accuracy` | `float` |
| `weak_accuracy` | `float` |
| `n_strong_attempted` | `int` |
| `n_weak_attempted` | `int` |
| `segregation_score` | `float` |
| `discriminator_quality` | `str` |
| `is_flagged` | `bool` |

Property: `.discrimination` — alias for `segregation_score`, matching IRT (a) notation.

### `SkippedQuestion` (dataclass, frozen): `question_id: str`, `reason: str` (`"no_attempts"` / `"only_strong_attempted"` / `"only_weak_attempted"`)

### `SegregationBatchResult` (dataclass)
| Field | Type |
|---|---|
| `results` | `List[SegregationResult]` |
| `skipped` | `List[SkippedQuestion]` |
| `unknown_student_response_count` | `int` |
| `unknown_student_ids` | `List[str]` |

Methods: `sorted_by_segregation_score(descending=True)`, `flagged()`, `as_dict_by_question()`, `warnings() -> List[str]`.

### Exceptions
`InsufficientAttemptsError(ValueError)`.

---

## `item_parameters.py`

**This is the decoupling seam** — the only module importing both `bloom_mapper` and `segregation`.

### `QuestionIRTParameters` (dataclass, frozen)
| Field | Type |
|---|---|
| `question_id` | `str` |
| `discrimination` | `float` (a) |
| `difficulty` | `float` (b) |

### `SkippedQuestionParameters` (dataclass, frozen): `question_id: str`, `reason: str` (`"no_segregation_score"` / `"missing_bloom_level"` / `"unknown_bloom_level"`)

### `build_question_parameters(question_bloom_levels: Mapping[str, str], segregation_result: SegregationBatchResult) -> Tuple[List[QuestionIRTParameters], List[SkippedQuestionParameters]]`
Assembles `QuestionIRTParameters` for every question with both a recognized Bloom level and a scored segregation result. Every `question_id` from either input ends up in exactly one of the two returned lists.

- **Example:**
  ```python
  from irt.item_parameters import build_question_parameters
  params, skipped = build_question_parameters({"Q1": "Apply"}, segregation_batch)
  ```

---

## `theta.py`

### `probability_correct(a: float, b: float, theta: float) -> float`
The 2PL curve: `1 / (1 + exp(-a(theta - b)))`. Numerically stable (exponent-clamped, no overflow direction). Public — reused by `mastery_initializer.py` and safe to call directly for reporting/plotting.

### `estimate_theta(responses: Sequence[AnswerRecord], parameters: Sequence[QuestionIRTParameters]) -> ThetaResult`
Newton-Raphson MLE for one student's ability, given known (a, b) per question.

- **Raises:**
  - `EmptyResponsesError` — zero responses.
  - `DuplicateResponseError` — a `question_id` appears twice in `responses`.
  - `DuplicateParameterError` — a `question_id` appears twice in `parameters`.
  - `MissingParameterError` — a response references a `question_id` with no matching parameters.
- **Example:**
  ```python
  from irt.theta import AnswerRecord, estimate_theta
  responses = [AnswerRecord("Q1", True), AnswerRecord("Q2", False)]
  result = estimate_theta(responses, params)
  result.theta, result.converged, result.standard_error
  ```

### `AnswerRecord` (dataclass, frozen): `question_id: str`, `is_correct: bool`

### `ThetaResult` (dataclass, frozen)
| Field | Type |
|---|---|
| `theta` | `float` |
| `iterations` | `int` |
| `converged` | `bool` |
| `log_likelihood` | `float` |
| `standard_error` | `Optional[float]` — `None` when Fisher information is 0 |
| `n_responses` | `int` |

**Note:** this is the class that fulfills the "ability estimate" role. It is named `ThetaResult`, not `StudentAbility` (an earlier planning-document placeholder name that was never implemented as a class).

### Exceptions
`EmptyResponsesError(ValueError)`, `DuplicateResponseError(ValueError)`, `DuplicateParameterError(ValueError)`, `MissingParameterError(ValueError)`.

---

## `mastery_initializer.py`

### `initialize_mastery(student_id: str, theta_result: Optional[ThetaResult], concept_attempts: Sequence[ConceptAttempt]) -> MasteryInitializationResult`
Computes per-concept initial mastery by blending observed concept accuracy with a theta-implied accuracy (via `theta.probability_correct()`, reused directly).

- **Raises:**
  - `MissingThetaError` — `theta_result` is `None`.
  - `EmptyConceptDataError` — `concept_attempts` is empty.
  - `DuplicateConceptAttemptError` — a `question_id` appears more than once (even across different concepts).
  - `InvalidBloomLevelError` — an attempt's `bloom_level` isn't recognized by `bloom_mapper`.
- **Example:**
  ```python
  from irt.mastery_initializer import ConceptAttempt, initialize_mastery
  attempts = [ConceptAttempt("Ohms_Law", "Q1", True, "Apply")]
  result = initialize_mastery("S1", theta_result, attempts)
  result.mastery_for("Ohms_Law")  # -> float in (0.05, 0.95)
  ```

### `ConceptAttempt` (dataclass, frozen)
| Field | Type |
|---|---|
| `concept_id` | `str` |
| `question_id` | `str` |
| `is_correct` | `bool` |
| `bloom_level` | `str` |

### `ConceptMastery` (dataclass, frozen)
| Field | Type |
|---|---|
| `concept_id` | `str` |
| `initial_mastery` | `float` |
| `observed_accuracy` | `float` |
| `theta_implied_accuracy` | `float` |
| `n_attempted` | `int` |
| `n_correct` | `int` |
| `weight_observed` | `float` |

### `MasteryInitializationSummary` (dataclass, frozen)
| Field | Type |
|---|---|
| `n_concepts` | `int` |
| `average_initial_mastery` | `float` |
| `lowest_mastery_concept_id` | `str` |
| `lowest_mastery_value` | `float` |
| `highest_mastery_concept_id` | `str` |
| `highest_mastery_value` | `float` |

### `MasteryInitializationResult` (dataclass)
| Field | Type |
|---|---|
| `student_id` | `str` |
| `theta` | `float` |
| `theta_converged` | `bool` |
| `concept_masteries` | `Dict[str, ConceptMastery]` |
| `summary` | `MasteryInitializationSummary` |

Method: `mastery_for(concept_id) -> float`.

### Exceptions
`MissingThetaError(ValueError)`, `EmptyConceptDataError(ValueError)`, `DuplicateConceptAttemptError(ValueError)`, `InvalidBloomLevelError(ValueError)`.

---

## `config.py`

Not an API surface in the function-call sense, but every constant referenced above is defined here and nowhere else: `BLOOM_DIFFICULTY_MAP`, `BLOOM_DIFFICULTY_BUCKETS`, `FEATURE_VECTOR_FIELDS`, `N_CLUSTERS`, `RANDOM_STATE`, `DISCRIMINATION_QUALITY_THRESHOLDS`, `FLAGGED_DISCRIMINATOR_QUALITIES`, `THETA_INITIAL`, `THETA_MIN`/`THETA_MAX`, `THETA_MAX_STEP`, `THETA_MAX_ITERATIONS`, `THETA_CONVERGENCE_TOLERANCE`, `THETA_EXTREME_PATTERN_CLAMP`, `THETA_EXPONENT_CLAMP`, `MASTERY_REFERENCE_DISCRIMINATION`, `MASTERY_PRIOR_STRENGTH`, `SEED_PRIOR_MIN`/`SEED_PRIOR_MAX`. Also exposes `load_database_url(override=None) -> str`, used only by the (not-yet-implemented) Postgres repository path.

---

## End-to-End Example

Chaining every module, exactly as `scripts/demo_mastery_initializer.py` does:

```python
from irt.feature_builder import build_feature_matrix, normalize_feature_matrix
from irt.clustering import cluster_students
from irt.segregation import compute_segregation_scores
from irt.item_parameters import build_question_parameters
from irt.theta import AnswerRecord, estimate_theta
from irt.mastery_initializer import ConceptAttempt, initialize_mastery

raw = build_feature_matrix(profiles, clustering_responses)
normalized = normalize_feature_matrix(raw)
cluster_result = cluster_students(normalized, raw)

segregation_batch = compute_segregation_scores(cluster_result, quiz_responses)
parameters, _skipped = build_question_parameters(bloom_levels_by_question, segregation_batch)

theta_result = estimate_theta(answer_records_for_one_student, parameters)

mastery_result = initialize_mastery("S1", theta_result, concept_attempts_for_one_student)
print(mastery_result.mastery_for("Ohms_Law"))
```
