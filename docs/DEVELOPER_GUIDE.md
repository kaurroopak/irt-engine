# DEVELOPER_GUIDE.md

## Coding Conventions

### Type hints
Every public function and dataclass field is fully type-hinted. `from __future__ import annotations` is used at the top of every module so hints can reference types defined later in the same file, and built-in generics (`list[str]`, `dict[str, float]`) are used directly rather than importing `List`/`Dict` from `typing` where Python's version allows it — though some modules do still import `List`/`Dict`/`Tuple` from `typing` for consistency with earlier modules; either style is acceptable in this codebase, but check the surrounding file before mixing both within one module.

### Dataclasses
Every structured input/output crossing a module boundary is a `@dataclass`, preferring `@dataclass(frozen=True)` for anything that represents a fact that shouldn't change after creation (e.g. `QuestionIRTParameters`, `ThetaResult`, `AnswerRecord`). Mutable dataclasses (plain `@dataclass`) are used only where a result object legitimately aggregates a list that grows during construction (e.g. `ImputationReport`, `FeatureMatrix`). When adding a new dataclass, default to frozen; only drop `frozen=True` if there's a specific reason.

### Configuration philosophy
**No module outside `config.py` may define a magic number that affects behavior.** If you're about to write a literal threshold, weight, bound, or mapping inline in an algorithmic module, it belongs in `config.py` instead, with a comment explaining *why* that value was chosen (see any existing constant in `config.py` for the expected comment depth — every one cites either a decision rationale or a literature anchor). This is not a style preference; it's what makes the "what does 'Apply' map to?" question in `config.py`'s own docstring answerable by one file.

### Logging
No module currently uses Python's `logging` module. Instead, functions that need to surface a non-fatal issue (e.g. an imputed value, a skipped question) return it as structured data — `ImputationReport`, `SkippedQuestion`, `SkippedQuestionParameters` — attached to the result object, with a `.warnings()` method that formats them as strings for a caller who wants to print or log them. This is a deliberate choice: a caller (e.g. the future `service.py`) can decide whether to log, raise, or silently accept a warning, rather than that decision being made inside the algorithmic module. If you add a new non-fatal condition, follow this pattern rather than calling `logging.warning()` directly from inside `irt/`.

### Error handling
Every module defines its own specific exception subclasses (always inheriting from a standard built-in like `ValueError` or `RuntimeError`, never a bare `Exception`), one per distinct failure condition, rather than reusing a generic exception with a different message. See any module's top-of-file exception class definitions for the pattern: a one-paragraph docstring on each explaining *why* it's raised and *why that's the right behavior* (not just what triggers it). When adding a new failure mode:

1. Define a new exception class, named for the condition (`VerbNounError` or `NounError` — see existing names for the convention).
2. Write a docstring explaining why silently continuing would be wrong.
3. Raise it as early as possible — validate inputs before doing any computation, not partway through.
4. Add a test that asserts the exception is raised (see Testing below).

**Never add a bare `except Exception: pass` or a silent default value for a condition that indicates missing/ambiguous data.** The one deliberate exception-swallowing pattern in this codebase is `_extreme_pattern_theta()` in `theta.py`, which detects (rather than catches) an all-correct/all-incorrect response pattern up front and returns a clamped result with `converged=False` — that is a documented, tested, mathematically-justified special case, not a generic catch-all.

## How to Add a New Module

1. **Write the responsibility docstring first.** Every existing module's file-level docstring follows the same shape: what it does, why it exists as a separate module (not folded into an adjacent one), and how it fits into the pipeline (with an ASCII diagram of what calls it and what it calls). Write this before any code — it's what keeps modules from creeping into overlapping responsibilities.
2. **Decide what it imports.** Check `docs/ARCHITECTURE.md`'s dependency graph before adding a new import — if your new module needs something from an "upstream" concept (e.g. Bloom levels) but is meant to be consumable by something "downstream" that shouldn't know about Bloom levels (the way `theta.py` shouldn't know about `bloom_mapper.py`), consider whether you need a decoupling seam like `item_parameters.py`.
3. **Add any new constants to `config.py`**, not inline.
4. **Define dataclasses for every input and output.** Don't pass primitives or dicts across a module boundary if a dataclass would document the shape better.
5. **Write the test file** (`tests/test_<module>.py`) covering: the normal case, every documented exception, at least one degenerate/boundary input, and a determinism check.
6. **Write the demo script** (`scripts/demo_<module>.py`), chaining the real upstream modules rather than hand-building fake intermediate objects, so it doubles as an integration check.
7. **Run the full suite** (`python -m pytest tests/ -q`) to confirm no regression in earlier modules before considering the module done.

## Writing Tests

Conventions observed across every existing test file:

- **One test function per behavior**, named `test_<condition>_<expected_outcome>` (e.g. `test_missing_iq_score_is_imputed_with_cohort_mean_and_reported`) — the name alone should describe what would have to break for the test to fail, without reading the body.
- **Build real objects through the real upstream API where practical**, rather than hand-constructing every field of a downstream dataclass — e.g. `test_segregation.py` hand-builds a `ClusterResult` directly (since `segregation.py`'s tests shouldn't depend on `clustering.py`'s KMeans behavior being correct), but `test_clustering.py`'s realistic-dataset tests go through the actual `feature_builder.build_feature_matrix()` pipeline. Choose based on whether the module under test should be isolated from, or should validate integration with, its upstream dependency.
- **Every documented exception gets its own test**, using `pytest.raises(SpecificExceptionType)` — never a bare `pytest.raises(Exception)`.
- **Determinism is tested explicitly** (call the function twice with identical input, assert identical output) for every module with any source of nondeterminism risk (KMeans seeding, floating-point iteration) — not because it's ever failed, but because it's a property worth protecting from an accidental regression (e.g. someone removing `random_state=42`).
- **A finding that contradicts a test's assumption is a signal to fix the test, not the code** — see `test_clustering.py`'s `test_fully_identical_students_raise_clustering_failed`, which replaced an earlier, incorrect assumption after the real scikit-learn behavior (a `ConvergenceWarning` and a raised exception, both correct) was observed.

## Running Tests

```bash
python -m pytest tests/ -q          # full suite, quiet output
python -m pytest tests/test_theta.py -q   # one module's tests only
python -m pytest tests/ -q -k "extreme"   # tests matching a keyword
```

## How to Contribute

1. Read `docs/ARCHITECTURE.md` first — understand which module owns the responsibility you're touching before writing code.
2. Follow the "How to Add a New Module" steps above for new functionality; for changes to existing modules, keep the same dataclass shapes where possible (see `docs/API_REFERENCE.md` for the current public contract) since downstream modules and the future `service.py` depend on them.
3. Run the full test suite before and after your change — 94/94 must remain passing, and any new behavior needs new tests, not just updated docstrings.
4. If your change affects a documented design decision (e.g. a threshold in `config.py`), update the relevant `docs/` file in the same change — documentation drift is treated as a bug in this project, not a follow-up task.

## How to Integrate Future Psychometric Models

This codebase was deliberately structured so that its two biggest simplifications — Bloom-as-difficulty and KMeans-segregation-as-discrimination — can be replaced without touching `theta.py` or `mastery_initializer.py`, because both consume only the plain `QuestionIRTParameters` dataclass (via `item_parameters.py`) and `ThetaResult`, never the modules that produce difficulty/discrimination directly. To integrate a real statistical calibration method later (e.g. once enough response data exists for classical MML/Bayesian 2PL calibration, per `docs/FUTURE_WORK.md`):

1. Write a new module (e.g. `statistical_calibration.py`) that takes a response matrix and produces `QuestionIRTParameters` objects — the exact same dataclass `item_parameters.build_question_parameters()` produces today.
2. `theta.py` and `mastery_initializer.py` require **zero changes** — they already only depend on that dataclass.
3. `service.py` (once built) chooses which parameter-production path to call — this could even be made a runtime decision (e.g. use statistical calibration once a question has enough responses, fall back to the Bloom/segregation hybrid otherwise), without either path needing to know about the other.

This is the concrete payoff of the "Future Compatibility" requirement stated at every module-implementation stage of this project, and is the reason `item_parameters.py` exists as a separate file rather than being folded into `theta.py` or `segregation.py`.
