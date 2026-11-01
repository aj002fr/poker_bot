"""Bundle a parameter vector into a self-contained, sandbox-ready bot directory.

The sandboxed worker hides the repository before it imports bot code, so a bundle
carries its own copy of the decision-time closure of :mod:`strategy.policy.runtime`
(:data:`RUNTIME_FILES` plus the equity tables) next to ``bot.py`` and ``params.json``.
The worker puts the bundle directory first on ``sys.path``, so ``import strategy``
resolves to the staged copy; in-process the repository package (already imported)
is used, which is byte-identical. From the arena the closure needs only
``arena.agents.sdk``, which the worker loads before the sandbox applies.

``manifest.json`` lists every file's sha256, the parameter vector hash and the source
commit. Bundles are deterministic: the same vector at the same commit gives the same
bytes.

    python -m strategy.policy.package --params artifacts/search/nlhe-cem-v1/incumbent.json --out artifacts/bots/x
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from strategy.policy.baseline import SPACE
from strategy.policy.params import vector_hash

REPO = Path(__file__).resolve().parents[2]
RUNTIME_FILES = tuple(f"strategy/{rel}" for rel in (
    "__init__.py",
    "rules/__init__.py", "rules/adapter.py", "rules/canonical.py", "rules/variants.py",
    "features/__init__.py", "features/cards.py", "features/equity.py", "features/state.py",
    "tables/__init__.py", "tables/store.py", "tables/preflop.py", "tables/postflop.py",
    "policy/__init__.py", "policy/baseline.py", "policy/fallback.py", "policy/params.py", "policy/runtime.py",
    "policy/spot.py",
))
"""The decision-time import closure of :mod:`strategy.policy.runtime` (checked by the tests)."""
TABLE_PATTERNS = ("preflop-*.json.gz", "postflop-*.json.gz")

BOT_TEMPLATE = '''"""Packaged baseline policy {vector_hash_short} (strategy.policy.package). Do not edit."""

import json
from pathlib import Path

from strategy.policy.runtime import RuntimeBot

_P = json.loads((Path(__file__).resolve().parent / "params.json").read_text(encoding="ascii"))


def make_bot():
    return RuntimeBot(_P["params"], name={name!r}, gc_mode={gc_mode!r})
'''


def table_files() -> list[str]:
    data = REPO / "strategy" / "tables" / "data"
    return sorted(str(p.relative_to(REPO)) for pat in TABLE_PATTERNS for p in data.glob(pat))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bundle(out: Path | str, params: Mapping[str, float] | None = None, *, name: str = "strategy_baseline",
                 gc_mode: str = "freeze", commit: str | None = None) -> Path:
    """Write the bundle to ``out`` (replacing it) and return the path."""
    out = Path(out)
    values = SPACE.complete(params)
    vh = vector_hash(values)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    files = [*RUNTIME_FILES, *table_files()]
    for rel in files:
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)
    (tmp / "params.json").write_text(json.dumps({"vector_hash": vh, "space_hash": SPACE.space_hash(),
                                                 "params": values}, sort_keys=True, indent=1) + "\n",
                                     encoding="ascii")
    (tmp / "bot.py").write_text(BOT_TEMPLATE.format(vector_hash_short=vh[:12], name=name, gc_mode=gc_mode),
                                encoding="ascii")
    manifest: dict[str, Any] = {"vector_hash": vh, "space_hash": SPACE.space_hash(), "name": name,
                                "gc_mode": gc_mode, "commit": commit,
                                "files": {rel: _sha(tmp / rel) for rel in sorted([*files, "bot.py", "params.json"])}}
    manifest["bundle_hash"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    (tmp / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="ascii")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def load_params(path: Path | str) -> dict[str, float]:
    """A vector from ``incumbent.json``/``params.json`` (``{"params": {...}}``) or a bare mapping."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(raw.get("params", raw))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.policy.package")
    p.add_argument("--params", help="JSON vector (default: the hand-written control)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="strategy_baseline")
    p.add_argument("--gc-mode", default="freeze", choices=("none", "freeze"))
    args = p.parse_args(argv)
    params = load_params(args.params) if args.params else None
    from strategy.search.cem import _git_commit

    out = build_bundle(args.out, params, name=args.name, gc_mode=args.gc_mode, commit=_git_commit())
    print(json.dumps(json.loads((out / "manifest.json").read_text())["bundle_hash"]), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
