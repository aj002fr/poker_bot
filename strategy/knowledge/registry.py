"""Load the knowledge base from disk and resolve parameter defaults safely."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]


class UnsupportedDefaultError(LookupError):
    """A caller asked for a default that no measurement or derivation supports."""


@dataclass
class Document:
    path: Path
    data: Any
    error: str | None = None

    @property
    def label(self) -> str:
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


@dataclass
class KnowledgeBase:
    root: Path
    vocabulary: dict[str, Any]
    sources: dict[str, Any]
    digest_index: dict[str, Any]
    records: list[Document] = field(default_factory=list)
    experiments: list[Document] = field(default_factory=list)
    test_specs: list[Document] = field(default_factory=list)

    def by_id(self, docs: list[Document]) -> dict[str, Document]:
        out = {}
        for doc in docs:
            if isinstance(doc.data, dict) and isinstance(doc.data.get("id"), str):
                out.setdefault(doc.data["id"], doc)
        return out

    def record(self, record_id: str) -> dict[str, Any]:
        doc = self.by_id(self.records).get(record_id)
        if doc is None:
            raise KeyError(record_id)
        return doc.data


def _read(path: Path) -> Document:
    try:
        return Document(path, yaml.safe_load(path.read_text(encoding="utf-8")))
    except yaml.YAMLError as exc:
        return Document(path, None, f"invalid YAML: {exc}")


def _mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return data if isinstance(data, dict) else {}


def load(root: Path | str = ROOT) -> KnowledgeBase:
    root = Path(root)
    return KnowledgeBase(
        root=root,
        vocabulary=_mapping(root / "vocabulary.yaml"),
        sources=_mapping(root / "sources.yaml"),
        digest_index=_mapping(root / "digest_index.yaml"),
        records=[_read(p) for p in sorted((root / "records").glob("*.yaml"))],
        experiments=[_read(p) for p in sorted((root / "experiment_specs").glob("*.yaml"))],
        test_specs=[_read(p) for p in sorted((root / "test_specs").glob("*.yaml"))],
    )


def parameter_default(record: dict[str, Any], name: str) -> Any:
    """Return a parameter default only when the record's evidence supports one.

    Measured defaults need a supported record with a result artifact; analytic and
    neutral defaults are always usable; sweep-only parameters have no default and
    must come from an experiment manifest.
    """
    for param in record.get("parameters", []):
        if param.get("name") != name:
            continue
        basis = param.get("basis")
        if basis in ("analytic", "neutral"):
            return param.get("default")
        if basis == "measured":
            artifact = (record.get("experiment") or {}).get("result_artifact")
            if record.get("status") == "supported" and artifact:
                return param.get("default")
            raise UnsupportedDefaultError(
                f"{record.get('id')}.{name}: measured default without a supported result"
            )
        raise UnsupportedDefaultError(
            f"{record.get('id')}.{name}: basis {basis!r} has no default; take it from a manifest"
        )
    raise KeyError(f"{record.get('id')} has no parameter {name!r}")
