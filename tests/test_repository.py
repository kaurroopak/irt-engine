"""
tests/test_repository.py - unit tests for irt/repository.py.

Three groups:
  1. IRTRepository - the abstract contract itself (can't be instantiated
     directly; every implementation must satisfy every abstract method).
  2. CSVRepository - built against small, controlled CSVs written to a
     pytest tmp_path fixture (so each test's data is exactly what that
     test needs, not shared/mutated sample_data state), plus a handful
     of tests against the repository's real sample_data/ folder to prove
     from_default_sample_data() and the shipped CSVs actually work
     together, including a full-pipeline integration smoke test that
     feeds CSVRepository's output straight into every ML module.
  3. PostgresRepository - built against a tiny fake DB-API connection
     (FakeConnection/FakeCursor below), so these tests need neither a
     real Postgres server nor network access, plus a couple of tests
     that monkeypatch sys.modules to exercise the real lazy
     `import psycopg2` + `psycopg2.connect()` code path with a fake
     psycopg2 module.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from irt.clustering import cluster_students
from irt.feature_builder import build_feature_matrix, normalize_feature_matrix
from irt.item_parameters import build_question_parameters
from irt.mastery_initializer import ConceptAttempt, initialize_mastery
from irt.repository import (
    CSVRepository,
    DataSourceUnavailableError,
    IRTRepository,
    MissingDependencyError,
    PostgresRepository,
    RecordNotFoundError,
)
from irt.segregation import compute_segregation_scores
from irt.theta import AnswerRecord, estimate_theta


# -- Group 1: IRTRepository (abstract contract) --------------------------


def test_irt_repository_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        IRTRepository()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated():
    class Incomplete(IRTRepository):
        def get_student_profiles(self):
            return []

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# -- Group 2: CSVRepository ------------------------------------------------


STUDENTS_CSV = """student_id,previous_percentage,iq_score
S1,90,120
S2,60,
S3,75,100
"""

QUESTIONS_CSV = """question_id,chapter,concept_id,bloom_level,difficulty,question_type,correct_answer,correct_reasoning
1,Electricity,E02,understand,1,Conceptual,B,reasoning one
2,Electricity,"E07,E10",apply,2,Numerical,C,reasoning two
3,Magnetism,M01,analyze,2,Conceptual,A,reasoning three
"""

RESPONSES_CSV = """student_id,question_id,is_correct
S1,1,1
S1,2,0
S1,3,1
S2,1,0
S2,2,1
S3,1,1
"""


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    (tmp_path / "students.csv").write_text(STUDENTS_CSV)
    (tmp_path / "questions.csv").write_text(QUESTIONS_CSV)
    (tmp_path / "responses.csv").write_text(RESPONSES_CSV)
    return tmp_path


def test_csv_repository_loads_student_profiles(csv_dir):
    repo = CSVRepository(csv_dir)
    profiles = {p.student_id: p for p in repo.get_student_profiles()}
    assert set(profiles) == {"S1", "S2", "S3"}
    assert profiles["S1"].previous_class_percentage == 90.0
    assert profiles["S1"].iq_score == 120.0
    # S2's iq_score cell is blank in the CSV -> None, not NaN, not 0.0.
    assert profiles["S2"].iq_score is None


def test_csv_repository_get_all_student_ids(csv_dir):
    repo = CSVRepository(csv_dir)
    assert repo.get_all_student_ids() == ["S1", "S2", "S3"]


def test_csv_repository_get_responses_joins_bloom_level(csv_dir):
    repo = CSVRepository(csv_dir)
    responses = repo.get_responses()
    assert len(responses) == 6
    by_qid = {(r.student_id, r.question_id): r for r in responses}
    r = by_qid[("S1", "2")]
    assert r.is_correct is False
    assert r.bloom_level == "apply"


def test_csv_repository_get_responses_filters_by_student_ids(csv_dir):
    repo = CSVRepository(csv_dir)
    responses = repo.get_responses(["S1"])
    assert len(responses) == 3
    assert all(r.student_id == "S1" for r in responses)


def test_csv_repository_get_question_bloom_levels(csv_dir):
    repo = CSVRepository(csv_dir)
    levels = repo.get_question_bloom_levels()
    assert levels == {"1": "understand", "2": "apply", "3": "analyze"}


def test_csv_repository_concept_attempts_uses_primary_concept_for_multi_tagged_question(csv_dir):
    repo = CSVRepository(csv_dir)
    attempts = repo.get_concept_attempts("S1")
    # question 2 is tagged "E07,E10" in the CSV; only ONE ConceptAttempt
    # should be produced for it (the primary concept), never two, or
    # mastery_initializer.initialize_mastery() would raise
    # DuplicateConceptAttemptError downstream.
    question_ids = [a.question_id for a in attempts]
    assert question_ids.count("2") == 1
    by_qid = {a.question_id: a for a in attempts}
    assert by_qid["2"].concept_id == "E07"
    assert by_qid["1"].concept_id == "E02"


def test_csv_repository_concept_attempts_feed_initialize_mastery_without_error(csv_dir):
    """Direct proof that the primary-concept resolution avoids
    DuplicateConceptAttemptError: run the real mastery_initializer
    against the repository's output."""
    from irt.theta import ThetaResult

    repo = CSVRepository(csv_dir)
    attempts = repo.get_concept_attempts("S1")
    theta_result = ThetaResult(
        theta=0.5, iterations=3, converged=True, log_likelihood=-1.0,
        standard_error=0.4, n_responses=3,
    )
    result = initialize_mastery("S1", theta_result, attempts)
    assert result.student_id == "S1"
    assert result.summary.n_concepts == len({a.concept_id for a in attempts})


def test_csv_repository_get_answer_records(csv_dir):
    repo = CSVRepository(csv_dir)
    records = repo.get_answer_records("S2")
    assert set((r.question_id, r.is_correct) for r in records) == {
        ("1", False), ("2", True),
    }
    assert all(isinstance(r, AnswerRecord) for r in records)


def test_csv_repository_unknown_student_raises_for_concept_attempts(csv_dir):
    repo = CSVRepository(csv_dir)
    with pytest.raises(RecordNotFoundError):
        repo.get_concept_attempts("NOPE")


def test_csv_repository_unknown_student_raises_for_answer_records(csv_dir):
    repo = CSVRepository(csv_dir)
    with pytest.raises(RecordNotFoundError):
        repo.get_answer_records("NOPE")


def test_csv_repository_student_with_no_responses_returns_empty_lists(tmp_path):
    (tmp_path / "students.csv").write_text("student_id,previous_percentage,iq_score\nS9,80,100\n")
    (tmp_path / "questions.csv").write_text(QUESTIONS_CSV)
    (tmp_path / "responses.csv").write_text("student_id,question_id,is_correct\n")
    repo = CSVRepository(tmp_path)
    assert repo.get_answer_records("S9") == []
    assert repo.get_concept_attempts("S9") == []


def test_csv_repository_missing_directory_raises():
    with pytest.raises(DataSourceUnavailableError):
        CSVRepository("/no/such/directory/at/all")


def test_csv_repository_missing_file_raises(tmp_path):
    (tmp_path / "students.csv").write_text(STUDENTS_CSV)
    # questions.csv and responses.csv deliberately absent
    with pytest.raises(DataSourceUnavailableError):
        CSVRepository(tmp_path)


def test_csv_repository_missing_required_column_raises(tmp_path):
    (tmp_path / "students.csv").write_text("student_id,iq_score\nS1,100\n")  # no previous_percentage
    (tmp_path / "questions.csv").write_text(QUESTIONS_CSV)
    (tmp_path / "responses.csv").write_text(RESPONSES_CSV)
    with pytest.raises(DataSourceUnavailableError):
        CSVRepository(tmp_path)


def test_csv_repository_response_referencing_unknown_question_raises(tmp_path):
    (tmp_path / "students.csv").write_text(STUDENTS_CSV)
    (tmp_path / "questions.csv").write_text(QUESTIONS_CSV)
    (tmp_path / "responses.csv").write_text("student_id,question_id,is_correct\nS1,999,1\n")
    with pytest.raises(DataSourceUnavailableError):
        CSVRepository(tmp_path)


def test_csv_repository_context_manager_returns_self_and_closes_cleanly(csv_dir):
    with CSVRepository(csv_dir) as repo:
        assert isinstance(repo, CSVRepository)
        assert repo.get_all_student_ids() == ["S1", "S2", "S3"]
    # close() is a documented no-op for CSVRepository; simply must not raise.


# -- against the real, shipped sample_data/ folder --------------------------


def test_csv_repository_from_default_sample_data_loads_real_files():
    repo = CSVRepository.from_default_sample_data()
    ids = repo.get_all_student_ids()
    assert ids == ["S1", "S2", "S3", "S4", "S5"]
    assert len(repo.get_responses()) == 155
    levels = repo.get_question_bloom_levels()
    assert levels["1"] == "understand"
    assert len(levels) == 31


def test_csv_repository_real_sample_data_feeds_full_pipeline_without_error():
    """The strongest guarantee this repository layer can offer: its
    output, completely unmodified, is accepted by every downstream ML
    module with zero adaptation."""
    repo = CSVRepository.from_default_sample_data()
    profiles = repo.get_student_profiles()
    responses = repo.get_responses()

    raw = build_feature_matrix(profiles, responses)
    normalized = normalize_feature_matrix(raw)
    cluster_result = cluster_students(normalized, raw)

    segregation_batch = compute_segregation_scores(cluster_result, responses)
    bloom_levels = repo.get_question_bloom_levels()
    parameters, _skipped = build_question_parameters(bloom_levels, segregation_batch)
    assert parameters  # at least some questions were scoreable

    sid = repo.get_all_student_ids()[0]
    answers = repo.get_answer_records(sid)
    theta_result = estimate_theta(answers, parameters)
    assert -4.0 <= theta_result.theta <= 4.0

    concept_attempts = repo.get_concept_attempts(sid)
    mastery_result = initialize_mastery(sid, theta_result, concept_attempts)
    # n_concepts counts UNIQUE concepts probed, which can be fewer than
    # len(concept_attempts) when several questions tag the same concept.
    unique_concepts = {a.concept_id for a in concept_attempts}
    assert mastery_result.summary.n_concepts == len(unique_concepts)
    for cm in mastery_result.concept_masteries.values():
        assert 0.0 < cm.initial_mastery < 1.0


# -- Group 3: PostgresRepository -------------------------------------------


class FakeCursor:
    """Minimal DB-API cursor stub. `script` maps a SQL substring to the
    rows that should be returned when a query containing that substring
    is executed, so each test only has to describe the queries it cares
    about."""

    def __init__(self, script, calls):
        self._script = script
        self._calls = calls
        self._result = []

    def execute(self, sql, params=()):
        self._calls.append((sql, params))
        for key, rows in self._script.items():
            if key in sql:
                self._result = rows
                return
        self._result = []

    def fetchall(self):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, script):
        self._script = script
        self.calls = []
        self.closed = False

    def cursor(self):
        return FakeCursor(self._script, self.calls)

    def close(self):
        self.closed = True


PG_SCRIPT = {
    "SELECT student_id, previous_percentage, iq_score": [
        ("S1", 92.0, 118.0),
        ("S2", 87.0, None),
    ],
    "SELECT student_id FROM students": [("S1",), ("S2",)],
    "SELECT r.student_id, r.question_id, r.is_correct, q.bloom_level": [
        ("S1", "1", True, "understand"),
        ("S2", "1", False, "understand"),
    ],
    "SELECT question_id, bloom_level FROM questions": [
        ("1", "understand"), ("2", "apply"),
    ],
    "SELECT 1 FROM students WHERE student_id = %s": [(1,)],
    "SELECT r.question_id, r.is_correct, q.bloom_level, q.concept_id": [
        ("2", True, "apply", "E07,E10"),
    ],
    "SELECT question_id, is_correct FROM responses": [
        ("1", True), ("2", False),
    ],
}


def test_postgres_repository_get_student_profiles():
    repo = PostgresRepository(connection=FakeConnection(PG_SCRIPT))
    profiles = repo.get_student_profiles()
    assert profiles[0].student_id == "S1"
    assert profiles[0].previous_class_percentage == 92.0
    assert profiles[1].iq_score is None


def test_postgres_repository_get_all_student_ids():
    repo = PostgresRepository(connection=FakeConnection(PG_SCRIPT))
    assert repo.get_all_student_ids() == ["S1", "S2"]


def test_postgres_repository_get_responses_no_filter():
    repo = PostgresRepository(connection=FakeConnection(PG_SCRIPT))
    responses = repo.get_responses()
    assert len(responses) == 2
    assert responses[0].bloom_level == "understand"
    assert responses[0].is_correct is True


def test_postgres_repository_get_responses_with_filter_passes_list_param():
    conn = FakeConnection(PG_SCRIPT)
    repo = PostgresRepository(connection=conn)
    repo.get_responses(["S1", "S2"])
    sql, params = conn.calls[-1]
    assert "ANY(%s)" in sql
    assert params == (["S1", "S2"],)


def test_postgres_repository_get_question_bloom_levels():
    repo = PostgresRepository(connection=FakeConnection(PG_SCRIPT))
    assert repo.get_question_bloom_levels() == {"1": "understand", "2": "apply"}


def test_postgres_repository_concept_attempts_uses_primary_concept():
    repo = PostgresRepository(connection=FakeConnection(PG_SCRIPT))
    attempts = repo.get_concept_attempts("S1")
    assert len(attempts) == 1
    assert attempts[0].concept_id == "E07"
    assert attempts[0].question_id == "2"


def test_postgres_repository_concept_attempts_unknown_student_raises():
    script = dict(PG_SCRIPT)
    script["SELECT 1 FROM students WHERE student_id = %s"] = []
    repo = PostgresRepository(connection=FakeConnection(script))
    with pytest.raises(RecordNotFoundError):
        repo.get_concept_attempts("GHOST")


def test_postgres_repository_get_answer_records():
    repo = PostgresRepository(connection=FakeConnection(PG_SCRIPT))
    records = repo.get_answer_records("S1")
    assert records == [
        AnswerRecord(question_id="1", is_correct=True),
        AnswerRecord(question_id="2", is_correct=False),
    ]


def test_postgres_repository_does_not_close_injected_connection():
    conn = FakeConnection(PG_SCRIPT)
    repo = PostgresRepository(connection=conn)
    repo.get_all_student_ids()
    repo.close()
    assert conn.closed is False


def test_postgres_repository_context_manager_does_not_close_injected_connection():
    conn = FakeConnection(PG_SCRIPT)
    with PostgresRepository(connection=conn) as repo:
        repo.get_all_student_ids()
    assert conn.closed is False


def test_postgres_repository_query_failure_wrapped_as_data_source_unavailable():
    class BoomCursor:
        def execute(self, sql, params=()):
            raise RuntimeError("connection reset by peer")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class BoomConnection:
        def cursor(self):
            return BoomCursor()

    repo = PostgresRepository(connection=BoomConnection())
    with pytest.raises(DataSourceUnavailableError):
        repo.get_all_student_ids()


# -- exercising the real lazy `import psycopg2` code path -------------------


def test_postgres_repository_lazy_imports_and_connects_via_psycopg2(monkeypatch):
    created = {}

    class FakeConn:
        def __init__(self):
            self.closed = False

        def cursor(self):
            return FakeCursor(PG_SCRIPT, created.setdefault("calls", []))

        def close(self):
            self.closed = True

    def fake_connect(dsn):
        created["dsn"] = dsn
        created["conn"] = FakeConn()
        return created["conn"]

    fake_psycopg2 = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    repo = PostgresRepository(database_url="postgresql://example/db")
    ids = repo.get_all_student_ids()
    assert ids == ["S1", "S2"]
    assert created["dsn"] == "postgresql://example/db"

    repo.close()
    assert created["conn"].closed is True


def test_postgres_repository_missing_psycopg2_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    repo = PostgresRepository(database_url="postgresql://example/db")
    with pytest.raises(MissingDependencyError):
        repo.get_all_student_ids()


def test_postgres_repository_connection_failure_wrapped(monkeypatch):
    def fake_connect(dsn):
        raise RuntimeError("could not connect to server")

    fake_psycopg2 = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    repo = PostgresRepository(database_url="postgresql://bad-host/db")
    with pytest.raises(DataSourceUnavailableError):
        repo.get_all_student_ids()


def test_postgres_repository_resolves_database_url_via_config_override(monkeypatch):
    """No database_url passed at all -> falls back to
    config.load_database_url(), which resolves from the DATABASE_URL
    environment variable. Confirms repository.py reuses config.py's
    existing resolution function rather than re-implementing it."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")
    created = {}

    def fake_connect(dsn):
        created["dsn"] = dsn
        return FakeConnection(PG_SCRIPT)

    fake_psycopg2 = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    repo = PostgresRepository()
    repo.get_all_student_ids()
    assert created["dsn"] == "postgresql://from-env/db"
