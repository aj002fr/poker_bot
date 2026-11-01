"""Canonical JSON and domain-tagged SHA-256, byte-identical to ``arena.domain.canonical``.

Kept here so decision-time code can hash without importing arena internals.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"not canonical-JSON serializable: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    return json.dumps(plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_hash(obj: Any, *, domain: str) -> str:
    return hashlib.sha256(f"{domain}\n{canonical_json(obj)}".encode("ascii")).hexdigest()
