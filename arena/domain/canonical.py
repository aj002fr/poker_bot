"""Canonical JSON: the single byte representation used for hashes and stored histories.

Keys are sorted, separators are fixed, output is ASCII, and NaN/infinity are
rejected. Integer map keys are written as decimal strings, as JSON requires.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"not canonical-JSON serializable: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def canonical_hash(obj: Any, *, domain: str) -> str:
    """SHA-256 of ``domain`` + canonical JSON; the domain tag versions the hashed layout."""
    return sha256_hex(f"{domain}\n{canonical_json(obj)}")
