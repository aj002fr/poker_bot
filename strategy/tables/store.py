"""Write and load generated tables with provenance.

A table file is gzip-compressed canonical JSON ``{"provenance": {...}, "data": {...}}``.
``provenance.data_sha256`` hashes the canonical ``data`` JSON, and
``provenance.inputs_sha256`` hashes the generator and key-definition sources the data
depends on. :func:`load` verifies the data hash; :func:`stale` reports tables whose
inputs changed since they were built.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = Path(__file__).resolve().parent / "data"
ENV_DIR = "STRATEGY_TABLES_DIR"


def table_dir() -> Path:
    return Path(os.environ.get(ENV_DIR) or DEFAULT_DIR)


def table_path(name: str, directory: Path | None = None) -> Path:
    return (directory or table_dir()) / f"{name}.json.gz"


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sources_sha256(modules: list[str]) -> str:
    """Hash of the source files (package-relative paths) a table's content depends on."""
    h = hashlib.sha256()
    for rel in sorted(modules):
        h.update(rel.encode())
        h.update((PACKAGE_ROOT / rel).read_bytes())
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE_ROOT.parent, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write(name: str, data: dict[str, Any], *, generator: str, inputs: list[str], params: dict[str, Any],
          seconds: float, directory: Path | None = None) -> Path:
    body = canonical(data)
    provenance = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "generator": generator,
        "params": params,
        "inputs": sorted(inputs),
        "inputs_sha256": sources_sha256(inputs),
        "data_sha256": hashlib.sha256(body).hexdigest(),
        "git_commit": _git_commit(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "wall_seconds": round(seconds, 1),
        "command": " ".join([os.path.basename(sys.executable), "-m", generator, *sys.argv[1:]]),
    }
    path = table_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical({"provenance": provenance, "data": data})
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(raw)
    load.cache_clear()
    return path


class TableError(ValueError):
    pass


@lru_cache(maxsize=None)
def load(name: str, directory: str | None = None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(provenance, data)`` or None when the table has not been generated."""
    path = table_path(name, Path(directory) if directory else None)
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        obj = json.loads(f.read())
    prov, data = obj["provenance"], obj["data"]
    if prov.get("format_version") != FORMAT_VERSION:
        raise TableError(f"{path}: unsupported table format {prov.get('format_version')!r}")
    if hashlib.sha256(canonical(data)).hexdigest() != prov["data_sha256"]:
        raise TableError(f"{path}: data hash mismatch")
    return prov, data


def stale(name: str) -> bool | None:
    """True when the inputs changed since the table was built; None when it is missing."""
    loaded = load(name)
    if loaded is None:
        return None
    prov, _ = loaded
    return sources_sha256(prov["inputs"]) != prov["inputs_sha256"]
