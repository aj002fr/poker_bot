"""The numba collection gate in tests/conftest.py and tests/support/numba_gate.py."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.support.numba_gate import ROOT, needs_numba, solver_test_modules

KNOWN_SOLVER_TESTS = ["test_averaging.py", "test_bench.py", "test_best_response.py", "test_compile.py",
                      "test_fixture_gates.py", "test_mutations.py", "test_openspiel_oracle.py",
                      "test_openspiel_reference.py", "test_reach_and_card_removal.py", "test_regret_matching.py",
                      "test_sampling.py"]


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))
    return root


def test_repository_solver_tests_are_detected_and_others_are_not():
    found = {p.relative_to(ROOT).as_posix() for p in solver_test_modules()}
    assert {f"tests/strategy/{name}" for name in KNOWN_SOLVER_TESTS} <= found
    for clean in ("tests/strategy/test_numba_gate.py", "tests/strategy/test_isolation.py",
                  "tests/strategy/test_eval.py", "tests/strategy/population/test_registry.py",
                  "tests/golden/test_core_fixtures.py"):
        assert clean not in found


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    pkgs = ["strategy", "strategy/solve", "strategy/games", "tests", "tests/strategy", "tests/strategy/deep",
            "tests/strategy/deep/nested", "tests/strategy/pkg"]
    return _tree(tmp_path, {
        **{f"{p}/__init__.py": "" for p in pkgs},
        "strategy/plain.py": "import numpy\n",
        "tests/strategy/helpers.py": "from strategy.solve.game import compile_game\n",
        "tests/strategy/lazy.py": "def f():\n    from strategy.solve import cfr\n",
        "tests/strategy/typed.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numba\n",
        "tests/strategy/pkg/__init__.py": "from strategy.games import kuhn\n",
        "tests/strategy/pkg/sub.py": "X = 1\n",
        "tests/strategy/cycle_a.py": "from tests.strategy import cycle_b\n",
        "tests/strategy/cycle_b.py": "from tests.strategy import cycle_a\n",
    })


@pytest.mark.parametrize(("body", "expected"), [
    ("import numba\n", True),
    ("from strategy.games import kuhn\n", True),
    ("from strategy.solve.cfr import CFRSolver\n", True),
    ("from tests.strategy.helpers import compile_game\n", True),
    ("from ...helpers import compile_game\n", True),
    ("from ... import helpers\n", True),
    ("from tests.strategy.pkg.sub import X\n", True),
    ("def test_x():\n    from strategy.solve import cfr\n", True),
    ("from tests.strategy.lazy import f\n", False),
    ("from tests.strategy import typed\n", False),
    ("from tests.strategy import cycle_a\n", False),
    ("import numpy\nfrom strategy import plain\n", False),
])
def test_new_nested_module_is_classified(repo: Path, body: str, expected: bool):
    path = _tree(repo, {"tests/strategy/deep/nested/test_new.py": body}) / "tests/strategy/deep/nested/test_new.py"
    assert needs_numba(path, repo) is expected


def _pytest_without_numba(tmp_path: Path, *args: str, require: bool = False) -> subprocess.CompletedProcess[str]:
    """Run pytest with tests/conftest.py as a plugin while numba is hidden from find_spec."""
    script = ("import sys; sys.modules['numba'] = None; import pytest; "
              "sys.exit(pytest.main(['-p', 'tests.conftest', '-p', 'no:cacheprovider', *sys.argv[1:]]))")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), "ARENA_REQUIRE_NUMBA": "1" if require else ""}
    return subprocess.run([sys.executable, "-c", script, "--rootdir", str(tmp_path), "-c", str(tmp_path / "pytest.ini"),
                           *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


@pytest.fixture
def nested_suite(tmp_path: Path) -> Path:
    _tree(tmp_path, {
        "pytest.ini": "[pytest]\n",
        "suite/deep/nested/test_new_solver.py": """
            from strategy.solve.cfr import CFRSolver

            def test_solver():
                assert CFRSolver
        """,
        "suite/deep/nested/test_new_helper_user.py": """
            from tests.strategy.helpers import *

            def test_helper():
                pass
        """,
        "suite/deep/nested/test_new_plain.py": "def test_plain():\n    assert True\n",
    })
    return tmp_path


def test_new_nested_solver_module_is_not_collected_without_numba(nested_suite: Path):
    proc = _pytest_without_numba(nested_suite, "-q", str(nested_suite / "suite"))
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "1 passed" in out and "error" not in out.lower(), out
    assert "2 solver test modules not collected" in out, out


def test_required_numba_missing_is_a_usage_error(nested_suite: Path):
    proc = _pytest_without_numba(nested_suite, "-q", str(nested_suite / "suite"), require=True)
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR, proc.stdout + proc.stderr
    assert "ARENA_REQUIRE_NUMBA=1 but numba is not installed" in proc.stderr
