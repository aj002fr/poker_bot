"""The oracles must stay independent of the code they check."""

import ast
import sys
from pathlib import Path

ORACLE_DIR = Path(__file__).parent
FORBIDDEN = {"pokerkit", "arena", "phevaluator"}


def imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_engine_or_production_imports():
    for path in sorted(ORACLE_DIR.glob("*.py")):
        assert not imported_roots(path) & FORBIDDEN, path.name


def test_oracle_modules_use_only_stdlib():
    for path in sorted(ORACLE_DIR.glob("*.py")):
        if path.name.startswith("test_") or path.name == "helpers.py":
            allowed = sys.stdlib_module_names | {"tests", "pytest"}
        else:
            allowed = sys.stdlib_module_names | {"tests"}
        assert imported_roots(path) <= allowed, (path.name, imported_roots(path) - allowed)
