from pathlib import Path

import pytest

from tests.support import numba_gate

# numba is in the opt-in `train` dependency group (no Intel-macOS wheels). Without it, test
# modules that import the solvers are not collected; `make verify-strategy` runs them, and
# ARENA_REQUIRE_NUMBA=1 turns a missing numba into a usage error instead.
_ignored: list[Path] = []


def pytest_configure(config):
    if numba_gate.numba_required() and not numba_gate.HAVE_NUMBA:
        raise pytest.UsageError(
            f"{numba_gate.REQUIRE_ENV}=1 but numba is not installed; run under `uv run --group train` "
            "(make verify-strategy) so the solver tests are collected")


def pytest_ignore_collect(collection_path: Path, config):
    if numba_gate.HAVE_NUMBA or collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    if numba_gate.needs_numba(collection_path):
        _ignored.append(collection_path)
        return True
    return None


def pytest_report_header(config):
    if not numba_gate.HAVE_NUMBA:
        return "numba not installed: solver test modules are not collected (make verify-strategy runs them)"


def pytest_terminal_summary(terminalreporter):
    if _ignored:
        terminalreporter.write_line(
            f"numba not installed: {len(_ignored)} solver test modules not collected (make verify-strategy)")
