"""JSON Schemas for the stable public contracts (written to ``docs/schemas/``).

``tests/property/test_schemas_stable.py`` fails when a model drifts from the
committed schema, so a contract change is always a deliberate, versioned edit.
"""

from __future__ import annotations

import json
from pathlib import Path

from arena.benchmark.manifest import RunManifest
from arena.domain.events import EVENT_MODELS, EventEnvelope
from arena.domain.models import Action, ActionEnvelope, LegalActions
from arena.domain.observations import HistoryView, Observation, PublicEvent, TableView

CONTRACTS = {
    "observation.v1.json": Observation,
    "table-view.v1.json": TableView,
    "history-view.v1.json": HistoryView,
    "public-event.v1.json": PublicEvent,
    "action.v1.json": Action,
    "action-envelope.v1.json": ActionEnvelope,
    "legal-actions.v1.json": LegalActions,
    "event-envelope.v1.json": EventEnvelope,
    "run-manifest.v1.json": RunManifest,
}


def schema_documents() -> dict[str, dict]:
    docs = {name: model.model_json_schema() for name, model in CONTRACTS.items()}
    docs["event-data.v1.json"] = {
        "title": "Event data by type (event schema version 1)",
        "type": "object",
        "properties": {name: model.model_json_schema() for name, model in sorted(EVENT_MODELS.items())},
    }
    return docs


def render(doc: dict) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def write_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, doc in schema_documents().items():
        path = out_dir / name
        path.write_text(render(doc), encoding="utf-8")
        paths.append(path)
    return paths
