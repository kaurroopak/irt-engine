"""
tests/test_cli.py - unit tests for irt/cli.py.

cli.py contains no pipeline math of its own (see its module docstring),
so these tests are NOT about re-verifying theta/mastery/clustering
numbers — that is each underlying module's own test file's job. They are
about the things cli.py IS responsible for:

  1. Argument parsing — the three data-source flags are mutually
     exclusive and one is required; --student is repeatable; --help
     exits 0 and documents every flag.
  2. Wiring the parsed flags to the right IRTRepository (--csv ->
     CSVRepository, --demo -> CSVRepository.from_default_sample_data(),
     --postgres -> PostgresRepository), without adapting any data along
     the way.
  3. Exit-code behavior: 0 on a successful run (even with some students
     skipped — mirrors service.run_pipeline()'s never-abort-the-cohort
     design), 1 when the data source itself can't be reached, 2 when the
     data source is reachable but the item bank can't be built.
  4. Report content: per-student theta/mastery lines, skipped-student
     reporting, and --verbose gating of the item-bank/warnings sections.

A handful of tests run the real CSVRepository against small CSVs written
to a pytest tmp_path fixture (mirrors tests/test_repository.py's own
`csv_dir` fixture pattern) or against this repository's bundled
sample_data/ via --demo. The --postgres tests monkeypatch sys.modules
with a fake psycopg2 module (mirrors
tests/test_repository.py's own lazy-import tests) so they need neither a
real Postgres server nor network access.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from irt.cli import (
    EXIT_DATA_SOURCE_ERROR,
    EXIT_OK,
    EXIT_PIPELINE_ERROR,
    build_parser,
    main,
)

# ── Shared CSV fixture data ──────────────────────────────────────────────
# Small enough to read at a glance, large enough to satisfy
# clustering.py's N_CLUSTERS=2 minimum and to exercise a mixed (not
# all-correct/all-incorrect) response pattern for every student.

STUDENTS_CSV = (
    "student_id,previous_percentage,iq_score\n"
    "S1,92,118\n"
    "S2,85,105\n"
    "S3,55,90\n"
    "S4,40,82\n"
)

QUESTIONS_CSV = (
    "question_id,concept_id,bloom_level\n"
    "Q1,C1,remember\n"
    "Q2,C1,understand\n"
    "Q3,C2,apply\n"
    "Q4,C2,analyze\n"
)

RESPONSES_CSV = (
    "student_id,question_id,is_correct\n"
    "S1,Q1,1\nS1,Q2,1\nS1,Q3,1\nS1,Q4,0\n"
    "S2,Q1,1\nS2,Q2,1\nS2,Q3,0\nS2,Q4,0\n"
    "S3,Q1,1\nS3,Q2,0\nS3,Q3,0\nS3,Q4,0\n"
    "S4,Q1,0\nS4,Q2,0\nS4,Q3,0\nS4,Q4,0\n"
)


@pytest.fixture()
def csv_dir(tmp_path: Path) -> Path:
    (tmp_path / "students.csv").write_text(STUDENTS_CSV)
    (tmp_path / "questions.csv").write_text(QUESTIONS_CSV)
    (tmp_path / "responses.csv").write_text(RESPONSES_CSV)
    return tmp_path


# ── Group 1: argument parsing ───────────────────────────────────────────


def test_help_exits_zero_and_documents_every_flag(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--csv", "--postgres", "--demo", "--student", "--verbose"):
        assert flag in out


def test_no_data_source_flag_is_a_parse_error():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])
    assert excinfo.value.code == 2


def test_two_data_source_flags_together_is_a_parse_error():
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--demo", "--csv", "sample_data"])
    assert excinfo.value.code == 2


def test_demo_flag_parses_true():
    args = build_parser().parse_args(["--demo"])
    assert args.demo is True
    assert args.csv is None
    assert args.postgres is None


def test_csv_flag_captures_data_dir():
    args = build_parser().parse_args(["--csv", "some/dir"])
    assert args.csv == "some/dir"
    assert args.demo is False


def test_postgres_flag_bare_defaults_to_empty_string():
    args = build_parser().parse_args(["--postgres"])
    assert args.postgres == ""


def test_postgres_flag_with_url_captures_it():
    args = build_parser().parse_args(["--postgres", "postgresql://host/db"])
    assert args.postgres == "postgresql://host/db"


def test_student_flag_is_repeatable():
    args = build_parser().parse_args(
        ["--demo", "--student", "S1", "--student", "S2"]
    )
    assert args.student == ["S1", "S2"]


def test_student_flag_short_form():
    args = build_parser().parse_args(["--demo", "-s", "S1"])
    assert args.student == ["S1"]


def test_student_flag_omitted_defaults_to_none():
    args = build_parser().parse_args(["--demo"])
    assert args.student is None


def test_verbose_flag_short_form():
    args = build_parser().parse_args(["--demo", "-v"])
    assert args.verbose is True


def test_verbose_defaults_to_false():
    args = build_parser().parse_args(["--demo"])
    assert args.verbose is False


# ── Group 2: --csv end-to-end ────────────────────────────────────────────


def test_csv_run_exits_ok_and_reports_every_student(csv_dir, capsys):
    code = main(["--csv", str(csv_dir)])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Students scored:               4" in out
    assert "Students skipped:              0" in out
    for sid in ("S1", "S2", "S3", "S4"):
        assert sid in out


def test_csv_run_reports_theta_and_mastery_lines(csv_dir, capsys):
    main(["--csv", str(csv_dir)])
    out = capsys.readouterr().out
    assert "theta=" in out
    assert "converged=" in out
    assert "mastery:" in out


def test_csv_run_verbose_shows_item_bank_and_clusters(csv_dir, capsys):
    main(["--csv", str(csv_dir), "--verbose"])
    out = capsys.readouterr().out
    assert "Item bank" in out
    assert "Strong cluster:" in out
    assert "Weak cluster:" in out


def test_csv_run_not_verbose_hides_item_bank_section(csv_dir, capsys):
    main(["--csv", str(csv_dir)])
    out = capsys.readouterr().out
    assert "Item bank" not in out


def test_csv_run_missing_directory_exits_data_source_error(capsys):
    code = main(["--csv", "/this/path/does/not/exist"])
    err = capsys.readouterr().err
    assert code == EXIT_DATA_SOURCE_ERROR
    assert "error:" in err


def test_csv_run_student_filter_scores_only_requested_students(csv_dir, capsys):
    code = main(["--csv", str(csv_dir), "--student", "S1"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "S1" in out
    assert "S2" not in out
    assert "Students scored:               1" in out


def test_csv_run_unknown_student_id_is_skipped_not_fatal(csv_dir, capsys):
    code = main(["--csv", str(csv_dir), "--student", "S1", "--student", "GHOST"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Students skipped:              1" in out
    assert "Skipped students" in out
    assert "GHOST" in out


def test_csv_run_too_few_students_exits_pipeline_error(tmp_path, capsys):
    (tmp_path / "students.csv").write_text(
        "student_id,previous_percentage,iq_score\nS1,90,110\n"
    )
    (tmp_path / "questions.csv").write_text(
        "question_id,concept_id,bloom_level\nQ1,C1,remember\n"
    )
    (tmp_path / "responses.csv").write_text(
        "student_id,question_id,is_correct\nS1,Q1,1\n"
    )
    code = main(["--csv", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == EXIT_PIPELINE_ERROR
    assert "error:" in err


# ── Group 3: --demo end-to-end (bundled sample_data/) ───────────────────


def test_demo_run_exits_ok(capsys):
    code = main(["--demo"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Synapse Hybrid IRT Engine" in out
    assert "Students scored:" in out


def test_demo_run_matches_csv_run_against_the_same_bundled_directory(capsys):
    """--demo is documented as a shortcut for
    CSVRepository.from_default_sample_data() — this proves the two flags
    actually produce the same scored-student count against the same
    on-disk sample_data/, not just similar-looking output."""
    demo_code = main(["--demo"])
    demo_out = capsys.readouterr().out

    here = Path(__file__).resolve().parent.parent / "sample_data"
    csv_code = main(["--csv", str(here)])
    csv_out = capsys.readouterr().out

    assert demo_code == csv_code == EXIT_OK
    demo_line = next(l for l in demo_out.splitlines() if l.startswith("Students scored:"))
    csv_line = next(l for l in csv_out.splitlines() if l.startswith("Students scored:"))
    assert demo_line == csv_line


# ── Group 4: --postgres end-to-end (fake psycopg2, no real server) ──────


class _FakeCursor:
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


class _FakeConnection:
    def __init__(self, script):
        self._script = script
        self.calls = []
        self.closed = False

    def cursor(self):
        return _FakeCursor(self._script, self.calls)

    def close(self):
        self.closed = True


_PG_SCRIPT = {
    "SELECT student_id, previous_percentage, iq_score": [
        ("S1", 92.0, 118.0),
        ("S2", 40.0, 80.0),
    ],
    "SELECT student_id FROM students": [("S1",), ("S2",)],
    "SELECT r.student_id, r.question_id, r.is_correct, q.bloom_level": [
        ("S1", "1", True, "understand"),
        ("S1", "2", True, "apply"),
        ("S2", "1", False, "understand"),
        ("S2", "2", False, "apply"),
    ],
    "SELECT question_id, bloom_level FROM questions": [
        ("1", "understand"),
        ("2", "apply"),
    ],
    "SELECT 1 FROM students WHERE student_id = %s": [(1,)],
    "SELECT r.question_id, r.is_correct, q.bloom_level, q.concept_id": [
        ("1", True, "understand", "C1"),
        ("2", True, "apply", "C1"),
    ],
    "SELECT question_id, is_correct FROM responses": [
        ("1", True),
        ("2", True),
    ],
}


def test_postgres_run_uses_lazy_psycopg2_import_and_reports_students(monkeypatch, capsys):
    created = {}

    def fake_connect(dsn):
        created["dsn"] = dsn
        return _FakeConnection(_PG_SCRIPT)

    fake_psycopg2 = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    code = main(["--postgres", "postgresql://fake-host/db"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert created["dsn"] == "postgresql://fake-host/db"
    assert "Students scored:               2" in out


def test_postgres_missing_psycopg2_exits_data_source_error(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "psycopg2", None)

    code = main(["--postgres", "postgresql://fake-host/db"])
    err = capsys.readouterr().err

    assert code == EXIT_DATA_SOURCE_ERROR
    assert "error:" in err


def test_postgres_bare_flag_resolves_url_from_environment(monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")

    created = {}

    def fake_connect(dsn):
        created["dsn"] = dsn
        return _FakeConnection(_PG_SCRIPT)

    fake_psycopg2 = types.SimpleNamespace(connect=fake_connect)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)

    code = main(["--postgres"])
    capsys.readouterr()

    assert code == EXIT_OK
    assert created["dsn"] == "postgresql://from-env/db"


# ── Group 5: main() signature / argv handling ────────────────────────────


def test_main_returns_int_not_none(csv_dir):
    result = main(["--csv", str(csv_dir)])
    assert isinstance(result, int)


def test_main_help_raises_system_exit_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
