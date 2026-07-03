"""
repository.py — the ONLY module in this package allowed to know where data
comes from.

Responsibility
--------------
Every module upstream of this one (bloom_mapper.py, feature_builder.py,
clustering.py, segregation.py, item_parameters.py, theta.py,
mastery_initializer.py) is pure: it takes and returns plain dataclasses
and never imports a database driver, a CSV parser, or an HTTP client (see
docs/ARCHITECTURE.md, principle 4: "Plain data across every boundary").
This module is the seam that produces those dataclasses from something
real — a CSV folder today, a Postgres database in production, and (per
the project's stated direction) potentially a REST API later — so that no
ML module ever has to know or care which one it was.

Design
------
    IRTRepository (ABC)
        - defines the read contract every data source must satisfy
        - returns ONLY dataclasses already defined elsewhere in this
          package (StudentProfileRow, ResponseRow from feature_builder.py;
          ConceptAttempt from mastery_initializer.py; AnswerRecord from
          theta.py) plus plain dict/list/str for the few shapes that
          don't already have a dataclass (question_id -> bloom_level is a
          Mapping[str, str], exactly what item_parameters.build_question_parameters
          already expects — introducing a dataclass for a two-field
          mapping would be the "duplicate model" the brief says not to
          invent).

    CSVRepository(IRTRepository)
        - reads sample_data/students.csv, questions.csv, responses.csv
        - loads everything into memory at construction time (the sample
          data is tiny; there is no lazy-loading benefit here and eager
          loading means every "file missing/malformed" error surfaces
          immediately at construction, not on some later, harder-to-debug
          call)

    PostgresRepository(IRTRepository)
        - same contract, backed by a live Postgres connection
        - never imports psycopg2 at module import time (see "Why psycopg2
          is imported lazily" below) — CSV-only callers, and anyone just
          running the unit test suite for CSVRepository, never need
          psycopg2 installed at all
        - accepts an already-open DB-API connection via dependency
          injection (the `connection=` constructor argument), which is
          what makes this class unit-testable without a real Postgres
          server or network access — see tests/test_repository.py

Why psycopg2 is imported lazily
--------------------------------
irt/config.py already documents this repository as "the ONLY module
allowed to communicate with external data sources." If `import psycopg2`
sat at the top of this file, then simply `import irt.repository` (which
CSVRepository users have no choice but to do, since both classes live in
one module) would require psycopg2 to be installed even for someone who
only ever touches CSVs. The import is deferred to the moment a
PostgresRepository actually needs to open a connection — see
PostgresRepository._connect().

A note on question -> concept mapping
--------------------------------------
sample_data/questions.csv tags some questions with more than one
concept_id (e.g. question 3: "E07,E10"). mastery_initializer.py's
initialize_mastery() deliberately raises DuplicateConceptAttemptError if
the same question_id appears more than once in a student's
concept_attempts — "under the same or different concept_id" (see its
docstring) — because a single answered question contributing observed-
accuracy evidence to two different concepts' mastery scores at once was
never part of that module's declared algorithm (see config.py's
MASTERY_REFERENCE_DISCRIMINATION note: mastery_initializer's inputs are
explicitly theta + concept accuracy + Bloom difficulty, one row per
question). Rather than changing that finalized module, this repository
resolves a multi-concept question to its FIRST listed concept_id ("the
primary concept") when building ConceptAttempt rows — see
_primary_concept_id() below. This is a repository-layer data-mapping
decision, not a change to any ML module's algorithm, and is called out
here so it's easy to find if/when Synapse's schema gains a proper
question<->concept many-to-many table and multi-concept credit becomes
worth doing differently upstream (e.g. splitting attempt weight across
concepts) — that would still be a repository.py change, not a
mastery_initializer.py change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, TYPE_CHECKING

from .config import load_database_url
from .feature_builder import ResponseRow, StudentProfileRow
from .mastery_initializer import ConceptAttempt
from .theta import AnswerRecord

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import psycopg2  # noqa: F401


# ── Exceptions ───────────────────────────────────────────────────────────
# Mirrors the rest of this codebase's "never silently continue" convention
# (see docs/ARCHITECTURE.md, principle 1): every failure mode a caller
# could hit gets its own specific, documented exception type.


class RepositoryError(Exception):
    """Base class for every exception raised by this module. Callers who
    don't care about the distinction between the specific error types
    below can catch this one type and know they've caught anything
    repository.py can raise."""


class DataSourceUnavailableError(RepositoryError):
    """Raised when the underlying data source itself can't be reached at
    all: a missing sample_data directory/CSV file for CSVRepository, or a
    failed connection for PostgresRepository. Distinguished from
    RecordNotFoundError because this means the repository can't answer
    ANY query, not just one for a specific id."""


class RecordNotFoundError(RepositoryError):
    """Raised when a specific requested record (most commonly a
    student_id) does not exist in an otherwise-reachable data source."""


class MissingDependencyError(RepositoryError):
    """Raised when PostgresRepository needs psycopg2 to open its own
    connection (i.e. no `connection=` was injected) but psycopg2 isn't
    installed."""


# ── Shared helpers (used by both CSVRepository and PostgresRepository, so
#    a CSV row and a Postgres row get mapped into dataclasses identically) ─


def _split_concept_ids(raw: Any) -> List[str]:
    """questions.csv's concept_id column is sometimes a single id
    ("E02") and sometimes several, comma-separated ("E07,E10"). Returns
    the ordered, whitespace-trimmed list of ids either way. Assumed
    identical in Postgres mode unless/until the schema normalizes this
    into a proper join table (see module docstring)."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _primary_concept_id(raw: Any) -> Optional[str]:
    """The first concept_id in a (possibly multi-valued) concept_id field.
    See the module docstring's "A note on question -> concept mapping"
    section for why only one concept is used per question here."""
    ids = _split_concept_ids(raw)
    return ids[0] if ids else None


def _to_bool(value: Any) -> bool:
    """Normalizes is_correct across sources: CSV gives ints/strings
    ('1'/'0'), Postgres boolean columns give real Python bools via the
    DB-API driver. Never treats a missing value as False-by-default —
    callers are expected to have already filtered out null rows."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y"):
        return True
    if text in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean is_correct value.")


def _to_optional_float(value: Any) -> Optional[float]:
    """NaN (pandas' representation of a blank CSV cell) and None (a SQL
    NULL, via psycopg2) both mean 'missing' — collapsed to a single
    Optional[float] contract, matching StudentProfileRow.iq_score's own
    Optional[float] type."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check without importing math/numpy just for this
        return None
    return f


# ── Abstract interface ──────────────────────────────────────────────────


class IRTRepository(ABC):
    """The read contract every data source implementation must satisfy.

    Every method returns dataclasses already defined by the ML modules
    (StudentProfileRow / ResponseRow from feature_builder.py,
    ConceptAttempt from mastery_initializer.py, AnswerRecord from
    theta.py) or plain built-in containers — never a row object, an ORM
    model, or a new bespoke type. This is what lets feature_builder.py,
    segregation.py, item_parameters.py, and mastery_initializer.py be
    handed the results of these calls with zero adaptation.

    Supports use as a context manager (`with SomeRepository(...) as
    repo:`); the default no-op close() below is correct for CSVRepository
    (nothing external stays open between calls) and is overridden by
    PostgresRepository to close its connection.
    """

    @abstractmethod
    def get_student_profiles(self) -> List[StudentProfileRow]:
        """Every student's profile fields relevant to feature-building.
        Feeds directly into feature_builder.build_feature_matrix()."""

    @abstractmethod
    def get_all_student_ids(self) -> List[str]:
        """Every known student_id, in a stable order. Convenience for
        callers (e.g. a service layer, or the demo script) that need to
        iterate every student without re-deriving the id list from
        get_student_profiles()."""

    @abstractmethod
    def get_responses(
        self, student_ids: Optional[Iterable[str]] = None
    ) -> List[ResponseRow]:
        """Responses, already joined with each question's bloom_level
        (feature_builder.ResponseRow carries bloom_level directly — see
        its docstring: "feature_builder never touches a question
        table/CSV directly, only this flattened row"). Only this module
        is allowed to do that join.

        Parameters
        ----------
        student_ids:
            If given, restricts results to these students. If omitted,
            returns every response for every student.
        """

    @abstractmethod
    def get_question_bloom_levels(self) -> Dict[str, str]:
        """{question_id: bloom_level} for every question. This IS the
        `question_bloom_levels` argument
        item_parameters.build_question_parameters() expects — a plain
        Mapping[str, str], not a new dataclass (per the requirement not
        to invent duplicate models for a shape that's already exactly
        what the consumer wants)."""

    @abstractmethod
    def get_concept_attempts(self, student_id: str) -> List[ConceptAttempt]:
        """One student's answered questions as ConceptAttempt rows, ready
        for mastery_initializer.initialize_mastery(). See the module
        docstring's note on multi-concept questions for how concept_id is
        resolved when a question is tagged with more than one concept.

        Raises
        ------
        RecordNotFoundError
            if student_id isn't a known student.
        """

    @abstractmethod
    def get_answer_records(self, student_id: str) -> List[AnswerRecord]:
        """One student's answered questions as AnswerRecord rows, ready
        for theta.estimate_theta().

        Raises
        ------
        RecordNotFoundError
            if student_id isn't a known student.
        """

    def close(self) -> None:
        """Release any held resources. No-op by default; overridden by
        implementations (e.g. PostgresRepository) that hold a live
        connection."""
        return None

    def __enter__(self) -> "IRTRepository":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ── CSVRepository ───────────────────────────────────────────────────────


class CSVRepository(IRTRepository):
    """Development/testing data source: reads the three sample_data CSVs
    into memory once, at construction time.

    Expects a directory containing exactly:
        students.csv   — student_id, previous_percentage, iq_score
        questions.csv  — question_id, chapter, concept_id, bloom_level,
                          difficulty, question_type, correct_answer,
                          correct_reasoning (only question_id, concept_id,
                          and bloom_level are used by this class)
        responses.csv  — student_id, question_id, is_correct

    matching sample_data/ exactly as shipped in this repository.
    """

    STUDENTS_FILENAME = "students.csv"
    QUESTIONS_FILENAME = "questions.csv"
    RESPONSES_FILENAME = "responses.csv"

    def __init__(self, data_dir: "str | Path") -> None:
        self._data_dir = Path(data_dir)
        if not self._data_dir.is_dir():
            raise DataSourceUnavailableError(
                f"CSVRepository data directory not found: {self._data_dir}"
            )

        self._students_by_id: Dict[str, StudentProfileRow] = self._load_students()
        self._bloom_by_question: Dict[str, str] = {}
        self._concept_ids_by_question: Dict[str, List[str]] = {}
        self._load_questions()
        self._responses: List[ResponseRow] = self._load_responses()
        self._responses_by_student: Dict[str, List[ResponseRow]] = {}
        for r in self._responses:
            self._responses_by_student.setdefault(r.student_id, []).append(r)

    @classmethod
    def from_default_sample_data(cls) -> "CSVRepository":
        """Convenience constructor pointing at this repository's own
        sample_data/ folder (../sample_data relative to irt/), so callers
        (and the demo script) don't need to know the on-disk layout."""
        here = Path(__file__).resolve().parent
        return cls(here.parent / "sample_data")

    # -- internal loading -------------------------------------------------

    def _csv_path(self, filename: str) -> Path:
        path = self._data_dir / filename
        if not path.is_file():
            raise DataSourceUnavailableError(
                f"Required CSV file not found: {path}"
            )
        return path

    def _load_students(self) -> Dict[str, StudentProfileRow]:
        import pandas as pd

        path = self._csv_path(self.STUDENTS_FILENAME)
        df = pd.read_csv(path)
        required = {"student_id", "previous_percentage"}
        missing = required - set(df.columns)
        if missing:
            raise DataSourceUnavailableError(
                f"{path} is missing required column(s): {sorted(missing)}"
            )

        students: Dict[str, StudentProfileRow] = {}
        for row in df.itertuples(index=False):
            sid = str(row.student_id)
            iq = _to_optional_float(getattr(row, "iq_score", None))
            pct = _to_optional_float(row.previous_class_percentage) if hasattr(
                row, "previous_class_percentage"
            ) else _to_optional_float(row.previous_percentage)
            students[sid] = StudentProfileRow(
                student_id=sid,
                previous_class_percentage=pct,
                iq_score=iq,
            )
        return students

    def _load_questions(self) -> None:
        import pandas as pd

        path = self._csv_path(self.QUESTIONS_FILENAME)
        df = pd.read_csv(path)
        required = {"question_id", "bloom_level"}
        missing = required - set(df.columns)
        if missing:
            raise DataSourceUnavailableError(
                f"{path} is missing required column(s): {sorted(missing)}"
            )

        for row in df.itertuples(index=False):
            qid = str(row.question_id)
            self._bloom_by_question[qid] = str(row.bloom_level).strip()
            concept_raw = getattr(row, "concept_id", None)
            self._concept_ids_by_question[qid] = _split_concept_ids(concept_raw)

    def _load_responses(self) -> List[ResponseRow]:
        import pandas as pd

        path = self._csv_path(self.RESPONSES_FILENAME)
        df = pd.read_csv(path)
        required = {"student_id", "question_id", "is_correct"}
        missing = required - set(df.columns)
        if missing:
            raise DataSourceUnavailableError(
                f"{path} is missing required column(s): {sorted(missing)}"
            )

        rows: List[ResponseRow] = []
        for row in df.itertuples(index=False):
            qid = str(row.question_id)
            bloom = self._bloom_by_question.get(qid)
            if bloom is None:
                raise DataSourceUnavailableError(
                    f"{path} references question_id {qid!r} which does not "
                    f"appear in {self.QUESTIONS_FILENAME}."
                )
            rows.append(
                ResponseRow(
                    student_id=str(row.student_id),
                    question_id=qid,
                    is_correct=_to_bool(row.is_correct),
                    bloom_level=bloom,
                )
            )
        return rows

    # -- IRTRepository interface ------------------------------------------

    def get_student_profiles(self) -> List[StudentProfileRow]:
        return list(self._students_by_id.values())

    def get_all_student_ids(self) -> List[str]:
        return list(self._students_by_id.keys())

    def get_responses(
        self, student_ids: Optional[Iterable[str]] = None
    ) -> List[ResponseRow]:
        if student_ids is None:
            return list(self._responses)
        wanted = {str(s) for s in student_ids}
        return [r for r in self._responses if r.student_id in wanted]

    def get_question_bloom_levels(self) -> Dict[str, str]:
        return dict(self._bloom_by_question)

    def get_concept_attempts(self, student_id: str) -> List[ConceptAttempt]:
        student_id = str(student_id)
        if student_id not in self._students_by_id:
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")

        attempts: List[ConceptAttempt] = []
        for r in self._responses_by_student.get(student_id, []):
            concept_id = _primary_concept_id(
                ",".join(self._concept_ids_by_question.get(r.question_id, []))
            )
            if concept_id is None:
                # A question with no concept tag at all can't seed any
                # concept's mastery; skip it rather than fabricating one.
                continue
            attempts.append(
                ConceptAttempt(
                    concept_id=concept_id,
                    question_id=r.question_id,
                    is_correct=r.is_correct,
                    bloom_level=r.bloom_level,
                )
            )
        return attempts

    def get_answer_records(self, student_id: str) -> List[AnswerRecord]:
        student_id = str(student_id)
        if student_id not in self._students_by_id:
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")
        return [
            AnswerRecord(question_id=r.question_id, is_correct=r.is_correct)
            for r in self._responses_by_student.get(student_id, [])
        ]


# ── PostgresRepository ──────────────────────────────────────────────────


class PostgresRepository(IRTRepository):
    """Production data source: same contract as CSVRepository, backed by
    a live Postgres connection.

    Assumed schema (see module docstring's note on question -> concept
    mapping; adjust the SQL below, not any ML module, if/when Synapse's
    actual Prisma schema differs):

        students(student_id, previous_percentage, iq_score)
        questions(question_id, concept_id, bloom_level)
        responses(student_id, question_id, is_correct)

    mirroring sample_data/*.csv exactly, so CSVRepository and
    PostgresRepository are interchangeable in every test and every
    downstream call.

    Connection handling
    --------------------
    Pass an already-open DB-API connection via `connection=` (this is how
    tests inject a fake connection without a real Postgres server — see
    tests/test_repository.py) or omit it and let the repository open its
    own connection lazily, resolved via config.load_database_url()
    (override > backend/.env > DATABASE_URL environment variable).
    psycopg2 itself is only imported at that point (see module docstring,
    "Why psycopg2 is imported lazily").
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        connection: Optional[Any] = None,
    ) -> None:
        self._database_url = database_url
        self._connection = connection
        self._owns_connection = connection is None

    # -- connection management --------------------------------------------

    def _connect(self) -> Any:
        if self._connection is not None:
            return self._connection

        try:
            import psycopg2
        except ImportError as exc:
            raise MissingDependencyError(
                "PostgresRepository needs psycopg2 to open its own connection. "
                "Install it with `pip install psycopg2-binary` (see "
                "requirements.txt), or pass an already-open connection via "
                "PostgresRepository(connection=...)."
            ) from exc

        url = load_database_url(self._database_url)
        try:
            self._connection = psycopg2.connect(url)
        except Exception as exc:  # psycopg2's own error hierarchy varies by
            # failure mode (auth, host, etc.); normalized to one type here,
            # same pattern clustering.py uses for scikit-learn failures.
            raise DataSourceUnavailableError(
                f"Could not connect to Postgres: {exc}"
            ) from exc
        return self._connection

    def close(self) -> None:
        if self._connection is not None and self._owns_connection:
            self._connection.close()
        self._connection = None

    # -- query helpers ------------------------------------------------------

    def _query(self, sql: str, params: tuple = ()) -> List[tuple]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())
        except DataSourceUnavailableError:
            raise
        except Exception as exc:
            raise DataSourceUnavailableError(f"Postgres query failed: {exc}") from exc

    # -- IRTRepository interface ------------------------------------------

    def get_student_profiles(self) -> List[StudentProfileRow]:
        rows = self._query(
            "SELECT student_id, previous_percentage, iq_score "
            "FROM students ORDER BY student_id"
        )
        return [
            StudentProfileRow(
                student_id=str(sid),
                previous_class_percentage=_to_optional_float(pct),
                iq_score=_to_optional_float(iq),
            )
            for sid, pct, iq in rows
        ]

    def get_all_student_ids(self) -> List[str]:
        rows = self._query("SELECT student_id FROM students ORDER BY student_id")
        return [str(r[0]) for r in rows]

    def get_responses(
        self, student_ids: Optional[Iterable[str]] = None
    ) -> List[ResponseRow]:
        base_sql = (
            "SELECT r.student_id, r.question_id, r.is_correct, q.bloom_level "
            "FROM responses r JOIN questions q ON q.question_id = r.question_id"
        )
        if student_ids is not None:
            wanted = [str(s) for s in student_ids]
            sql = base_sql + " WHERE r.student_id = ANY(%s) ORDER BY r.student_id, r.question_id"
            rows = self._query(sql, (wanted,))
        else:
            sql = base_sql + " ORDER BY r.student_id, r.question_id"
            rows = self._query(sql)

        return [
            ResponseRow(
                student_id=str(sid),
                question_id=str(qid),
                is_correct=_to_bool(is_correct),
                bloom_level=str(bloom).strip(),
            )
            for sid, qid, is_correct, bloom in rows
        ]

    def get_question_bloom_levels(self) -> Dict[str, str]:
        rows = self._query("SELECT question_id, bloom_level FROM questions")
        return {str(qid): str(bloom).strip() for qid, bloom in rows}

    def _ensure_known_student(self, student_id: str) -> None:
        rows = self._query(
            "SELECT 1 FROM students WHERE student_id = %s", (student_id,)
        )
        if not rows:
            raise RecordNotFoundError(f"Unknown student_id: {student_id!r}")

    def get_concept_attempts(self, student_id: str) -> List[ConceptAttempt]:
        student_id = str(student_id)
        self._ensure_known_student(student_id)

        rows = self._query(
            "SELECT r.question_id, r.is_correct, q.bloom_level, q.concept_id "
            "FROM responses r JOIN questions q ON q.question_id = r.question_id "
            "WHERE r.student_id = %s ORDER BY r.question_id",
            (student_id,),
        )

        attempts: List[ConceptAttempt] = []
        for qid, is_correct, bloom, concept_raw in rows:
            concept_id = _primary_concept_id(concept_raw)
            if concept_id is None:
                continue
            attempts.append(
                ConceptAttempt(
                    concept_id=concept_id,
                    question_id=str(qid),
                    is_correct=_to_bool(is_correct),
                    bloom_level=str(bloom).strip(),
                )
            )
        return attempts

    def get_answer_records(self, student_id: str) -> List[AnswerRecord]:
        student_id = str(student_id)
        self._ensure_known_student(student_id)

        rows = self._query(
            "SELECT question_id, is_correct FROM responses "
            "WHERE student_id = %s ORDER BY question_id",
            (student_id,),
        )
        return [
            AnswerRecord(question_id=str(qid), is_correct=_to_bool(is_correct))
            for qid, is_correct in rows
        ]


__all__ = [
    "IRTRepository",
    "CSVRepository",
    "PostgresRepository",
    "RepositoryError",
    "DataSourceUnavailableError",
    "RecordNotFoundError",
    "MissingDependencyError",
]
