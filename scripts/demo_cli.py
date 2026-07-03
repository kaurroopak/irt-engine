"""
scripts/demo_cli.py

Demonstrates irt/cli.py — the `python -m irt` command-line entry point —
by calling irt.cli.main() directly with a few representative argv lists,
exactly as if they had been typed at a shell. This is the CLI's own
integration demo, in the same spirit as scripts/demo_service.py: it
proves an end user can drive the ENTIRE Hybrid IRT pipeline (Feature
Builder -> Clustering -> Segregation -> Question Parameters -> Theta ->
Mastery Initializer) from one shell command, with zero Python code of
their own.

Run with:
    python -m scripts.demo_cli

Everything below can equally be run directly from a shell (from the
repository root, so the relative --csv path resolves):
    python -m irt --demo
    python -m irt --demo --student S1 --student S2
    python -m irt --csv sample_data --verbose
    python -m irt --csv /this/path/does/not/exist
    python -m irt --help
"""

from __future__ import annotations

from typing import List

from irt.cli import main


def _run(label: str, argv: List[str]) -> None:
    print("=" * 70)
    print(f"$ python -m irt {' '.join(argv)}")
    print(f"  ({label})")
    print("=" * 70)
    try:
        exit_code = main(argv)
    except SystemExit as exc:
        # argparse itself raises SystemExit for --help and for invalid/
        # missing/conflicting flags (see cli.main()'s docstring: this is
        # intentionally left to propagate out of main() rather than being
        # caught there). Catching it only here, in the demo driver, lets
        # this single script showcase --help without exiting the whole
        # demo early.
        exit_code = exc.code
    print()
    print(f"[exit code: {exit_code}]")
    print()


def main_demo() -> None:
    # 1. The fastest way to see the whole pipeline run: bundled sample
    #    data, every student the repository knows about, concise output.
    _run("whole cohort, bundled sample_data/", ["--demo"])

    # 2. Restricting to specific students with repeated --student flags.
    _run(
        "two specific students only",
        ["--demo", "--student", "S1", "--student", "S2"],
    )

    # 3. --csv pointed explicitly at a data directory, with --verbose to
    #    show the item-bank section (cluster membership, flagged
    #    discriminators) that --demo's concise output above hides.
    _run("explicit --csv DATA_DIR, --verbose", ["--csv", "sample_data", "--verbose"])

    # 4. A student_id that does not exist: proves the CLI reports it as a
    #    skipped student rather than crashing the whole run (mirrors
    #    service.run_pipeline()'s never-let-one-bad-student-abort-the-
    #    cohort design).
    _run(
        "one bad student_id mixed in",
        ["--demo", "--student", "S1", "--student", "DOES_NOT_EXIST"],
    )

    # 5. A missing --csv directory: proves the CLI reports a clear error
    #    and a non-zero exit code instead of a raw traceback.
    _run("missing --csv directory", ["--csv", "/this/path/does/not/exist"])

    # 6. Help text, exactly as `python -m irt --help` would print it.
    _run("--help", ["--help"])


if __name__ == "__main__":
    main_demo()
