"""Find test modules that cannot import without numba (the opt-in `train` group).

Detection is static: a test module needs numba when it imports numba, llvmlite,
`strategy.solve` or `strategy.games`, directly or through any repository module it imports
(`tests.strategy.helpers`, relative imports, package `__init__` files). Every import in the
test module counts, including ones inside test functions. In imported repository modules,
only imports that run at import time count, so a lazy import inside a function is not
followed.
"""

from __future__ import annotations

import ast
import importlib.util
import os
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEEDS_NUMBA_MODULES = ("numba", "llvmlite", "strategy.solve", "strategy.games")
REQUIRE_ENV = "ARENA_REQUIRE_NUMBA"
HAVE_NUMBA = importlib.util.find_spec("numba") is not None


def numba_required() -> bool:
    return os.environ.get(REQUIRE_ENV, "") not in ("", "0")


def _is_root(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in NEEDS_NUMBA_MODULES)


def _module_name(path: Path, root: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None


def _module_file(name: str, root: Path) -> Path | None:
    base = root.joinpath(*name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _import_time_nodes(tree: ast.AST) -> list[ast.AST]:
    """Import statements that execute when the module is imported."""
    out: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _imported_names(path: Path, root: Path, *, everywhere: bool) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    nodes = list(ast.walk(tree)) if everywhere else _import_time_nodes(tree)
    package = _module_name(path, root) or ""
    if path.name != "__init__.py":
        package = package.rpartition(".")[0]
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                if node.level - 1 > len(parts):
                    continue
                base = ".".join(parts[: len(parts) - (node.level - 1)] + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                names.add(base)
            names.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names if alias.name != "*")
    return names


def _with_parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


@cache
def _module_imports(name: str, root: Path) -> frozenset[str]:
    path = _module_file(name, root)
    if path is None:
        return frozenset()
    return frozenset(d for imported in _imported_names(path, root, everywhere=False) for d in _with_parents(imported))


def needs_numba(path: Path, root: Path = ROOT) -> bool:
    """Whether importing or running the test module at `path` needs numba."""
    root = root.resolve()
    pending = {d for imported in _imported_names(path, root, everywhere=True) for d in _with_parents(imported)}
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if _is_root(name):
            return True
        if name not in seen:
            seen.add(name)
            pending |= _module_imports(name, root) - seen
    return False


def solver_test_modules(root: Path = ROOT, tests_dir: str = "tests") -> list[Path]:
    """Every `test_*.py` under `tests_dir` that needs numba, sorted."""
    return sorted(p for p in (root / tests_dir).rglob("test_*.py") if needs_numba(p, root))
