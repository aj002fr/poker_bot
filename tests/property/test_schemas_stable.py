"""The committed JSON Schemas in docs/schemas match the models.

If this fails after an intentional contract change, bump the schema version,
regenerate with ``python -m arena.cli schemas`` and update
docs/observations-and-projections.md. Downstream workers rely on these files.
"""

from pathlib import Path

import pytest

from arena.schemas import render, schema_documents

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schemas"


@pytest.mark.parametrize("name", sorted(schema_documents()))
def test_schema_matches_committed_file(name):
    committed = (SCHEMA_DIR / name).read_text(encoding="utf-8")
    assert render(schema_documents()[name]) == committed


def test_no_stray_schema_files():
    assert {p.name for p in SCHEMA_DIR.glob("*.json")} == set(schema_documents())
