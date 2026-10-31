"""Canonical private table history: one JSON document per table.

The document is the table's complete private event log plus a summary. It holds
the explicit deck of every hand, every private card, the RNG seed and decision
state hashes, so it is an operator/reviewer artifact. Viewer-safe histories come
from :func:`arena.table.projections.history_view`.

The format is versioned (``format`` and ``schema_version``); loading rejects
unknown versions, unknown event types and malformed payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr

from arena.domain.canonical import canonical_json
from arena.domain.events import EVENT_SCHEMA_VERSION, EventEnvelope

if TYPE_CHECKING:
    from arena.table.service import Table

HISTORY_FORMAT = "arena.table-history"
HISTORY_SCHEMA_VERSION = 1
DEFAULT_HISTORY_DIR = Path("artifacts/tables")


class HistorySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    complete: StrictBool
    hands_played: StrictInt
    decisions: StrictInt
    final_stacks: dict[int, StrictInt]
    points: dict[int, float] | None


class TableHistory(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, arbitrary_types_allowed=True)

    format: Literal["arena.table-history"] = HISTORY_FORMAT
    schema_version: Literal[1] = HISTORY_SCHEMA_VERSION
    event_schema_version: Literal[1] = EVENT_SCHEMA_VERSION
    privacy: Literal["private"] = "private"
    table_id: StrictStr
    events: list[EventEnvelope]
    summary: HistorySummary

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "event_schema_version": self.event_schema_version,
            "privacy": self.privacy,
            "table_id": self.table_id,
            "events": [e.to_json_obj() for e in self.events],
            "summary": self.summary.model_dump(mode="json"),
        }

    def dumps(self) -> str:
        return canonical_json(self.to_json_obj()) + "\n"

    @classmethod
    def loads(cls, text: str) -> TableHistory:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("history must be a JSON object")
        if obj.get("format") != HISTORY_FORMAT:
            raise ValueError(f"not an arena table history (format {obj.get('format')!r})")
        if obj.get("schema_version") != HISTORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported history schema_version {obj.get('schema_version')!r}")
        events = [EventEnvelope.from_json_obj(e) for e in obj.get("events", [])]
        for index, event in enumerate(events):
            if event.seq != index:
                raise ValueError(f"event sequence gap at position {index}")
            if event.table_id != obj.get("table_id"):
                raise ValueError(f"event {index} belongs to another table")
        summary = HistorySummary.model_validate_json(json.dumps(obj.get("summary")))
        rest = {k: v for k, v in obj.items() if k not in ("events", "summary")}
        return cls(**rest, events=events, summary=summary)

    def hand_events(self, hand_number: int) -> list[EventEnvelope]:
        hand_id = self.hand_id(hand_number)
        return [e for e in self.events if e.hand_id == hand_id]

    def hand_id(self, hand_number: int) -> str:
        for e in self.events:
            if e.type == "HandStarted" and e.data["hand_number"] == hand_number:
                assert e.hand_id is not None
                return e.hand_id
        raise KeyError(f"hand {hand_number} not in history")

    @property
    def table_started(self) -> EventEnvelope:
        first = self.events[0]
        if first.type != "TableStarted":
            raise ValueError("history does not begin with TableStarted")
        return first


def export_history(table: Table) -> TableHistory:
    decisions = sum(1 for e in table.log if e.type == "DecisionRequested")
    return TableHistory(
        table_id=table.table_id,
        events=list(table.log),
        summary=HistorySummary(
            complete=table.complete,
            hands_played=len(table.hands),
            decisions=decisions,
            final_stacks=dict(table.stacks),
            points={s: float(v) for s, v in table.points().items()} if table.complete else None,
        ),
    )


def history_path(table_id: str, history_dir: Path = DEFAULT_HISTORY_DIR) -> Path:
    if not table_id or any(c in table_id for c in "/\\") or table_id.startswith("."):
        raise ValueError(f"invalid table id {table_id!r}")
    return history_dir / f"{table_id}.json"


def write_history(table: Table, history_dir: Path = DEFAULT_HISTORY_DIR) -> Path:
    path = history_path(table.table_id, history_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(export_history(table).dumps(), encoding="ascii")
    tmp.replace(path)
    return path


def load_history(path: Path) -> TableHistory:
    return TableHistory.loads(path.read_text(encoding="ascii"))
