"""Study packs: preprocessed hands for the browser's pause-and-predict mode.

Packs are plain JSON built offline (``python -m strategy.data.study``); this module
only reads them and never imports ``strategy``. A pack is a directory under
``ARENA_STUDY_DIR`` (default ``artifacts/study``) holding ``pack.json``, ``index.json``
and ``hands/<id>.json``. A pack with another schema version or a malformed manifest
is skipped, not served.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from arena.api.manager import ApiError

STUDY_SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_DIR = REPO / "artifacts" / "study"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")
FILTERS = ("all", "postflop", "showdown", "big_pot")
BIG_POT_BB = 20.0
MAX_PAGE = 200


def study_dir_from_env() -> Path:
    return Path(os.environ.get("ARENA_STUDY_DIR") or DEFAULT_STUDY_DIR)


def _keep(entry: dict[str, Any], name: str) -> bool:
    if name == "postflop":
        return entry["last_street"] != "preflop"
    if name == "showdown":
        return bool(entry["showdown"])
    if name == "big_pot":
        return float(entry["pot_bb"]) >= BIG_POT_BB
    return True


class StudyLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._index: dict[str, tuple[float, dict[str, Any], list[dict[str, Any]]]] = {}

    def _pack_dir(self, pack_id: str) -> Path:
        if not _ID.match(pack_id):
            raise ApiError(404, "pack_not_found", "no such study pack")
        return self.root / pack_id

    def _load(self, pack_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        d = self._pack_dir(pack_id)
        manifest = d / "pack.json"
        try:
            mtime = manifest.stat().st_mtime
        except OSError:
            raise ApiError(404, "pack_not_found", "no such study pack") from None
        with self._lock:
            hit = self._index.get(pack_id)
            if hit is not None and hit[0] == mtime:
                return hit[1], hit[2]
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            hands = json.loads((d / "index.json").read_text(encoding="utf-8"))["hands"]
        except (OSError, ValueError, KeyError, TypeError):
            raise ApiError(503, "pack_unreadable", f"study pack {pack_id} is incomplete or malformed") from None
        if meta.get("schema_version") != STUDY_SCHEMA_VERSION or meta.get("id") != pack_id:
            raise ApiError(503, "pack_unreadable", f"study pack {pack_id} has another schema version; rebuild it")
        with self._lock:
            self._index[pack_id] = (mtime, meta, hands)
        return meta, hands

    def packs(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or not _ID.match(d.name) or not (d / "pack.json").is_file():
                continue
            try:
                meta, hands = self._load(d.name)
            except ApiError:
                continue
            out.append({k: meta.get(k) for k in ("id", "title", "hero", "ruleset_id", "hands", "decisions",
                                                  "transform", "dataset", "caveat")}
                       | {"filters": {f: sum(1 for h in hands if _keep(h, f)) for f in FILTERS}})
        out.sort(key=lambda p: (-int(p["hands"] or 0), p["id"]))
        return out

    def hands(self, pack_id: str, *, filter: str = "all", offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if filter not in FILTERS:
            raise ApiError(422, "invalid_filter", f"filter must be one of {', '.join(FILTERS)}")
        _, hands = self._load(pack_id)
        chosen = [h for h in hands if _keep(h, filter)]
        offset = max(0, offset)
        limit = max(1, min(MAX_PAGE, limit))
        return {"pack": pack_id, "filter": filter, "total": len(chosen), "offset": offset,
                "hands": chosen[offset:offset + limit]}

    def hand(self, pack_id: str, hand_id: str) -> dict[str, Any]:
        _, hands = self._load(pack_id)
        if not _ID.match(hand_id) or not any(h["id"] == hand_id for h in hands):
            raise ApiError(404, "hand_not_found", "no such hand in this study pack")
        try:
            return json.loads((self._pack_dir(pack_id) / "hands" / f"{hand_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ApiError(503, "pack_unreadable", f"hand {hand_id} is missing from study pack {pack_id}") from None
