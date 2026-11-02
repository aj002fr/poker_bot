"""Study packs over HTTP: listing, filters and paging, one hand's JSON, and 404/422/503
for unknown ids, bad filters and broken packs. The packs are built offline from the
committed Pluribus fixture; the arena only reads them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arena.api.app import create_app
from arena.api.study import BIG_POT_BB, StudyLibrary
from strategy.data.phh_import import FIXTURE_DIR, iter_source
from strategy.data.study import write_pack


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("packs")
    write_pack(root / "pluribus-sample", iter_source(FIXTURE_DIR), pack_id="pluribus-sample",
               title="Pluribus sample hands", source_label="fixture")
    return root


@pytest.fixture
def root(built, tmp_path) -> Path:
    shutil.copytree(built, tmp_path / "packs")
    return tmp_path / "packs"


def client(root: Path) -> TestClient:
    return TestClient(create_app(background_pump=False, study=StudyLibrary(root)))


def test_lists_packs_with_filter_counts(root):
    body = client(root).get("/study/packs").json()
    assert body["compatibility_status"] == "practice_assumptions"
    [pack] = body["packs"]
    assert (pack["id"], pack["hands"], pack["decisions"], pack["hero"]) == ("pluribus-sample", 24, 81, "Pluribus")
    f = pack["filters"]
    assert f["all"] == 24 and 0 < f["showdown"] <= f["postflop"] < f["all"] and 0 < f["big_pot"] < f["all"]
    assert pack["dataset"]["licence"].startswith("MIT")


def test_pages_and_filters_hands(root):
    c = client(root)
    first = c.get("/study/packs/pluribus-sample/hands?limit=10").json()
    rest = c.get("/study/packs/pluribus-sample/hands?offset=10&limit=200").json()
    assert first["total"] == rest["total"] == 24 and len(first["hands"]) == 10 and len(rest["hands"]) == 14
    ids = [h["id"] for h in first["hands"] + rest["hands"]]
    assert len(set(ids)) == 24 and ids[0] == "pluribus-100-0"
    for name, keep in (("postflop", lambda h: h["last_street"] != "preflop"), ("showdown", lambda h: h["showdown"]),
                       ("big_pot", lambda h: h["pot_bb"] >= BIG_POT_BB)):
        page = c.get(f"/study/packs/pluribus-sample/hands?filter={name}&limit=200").json()
        assert page["hands"] and all(keep(h) for h in page["hands"])
    r = c.get("/study/packs/pluribus-sample/hands?filter=everything")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_filter"


def test_serves_one_hand(root):
    doc = client(root).get("/study/packs/pluribus-sample/hands/pluribus-100-11").json()
    assert doc["id"] == "pluribus-100-11" and doc["hero_name"] == "Pluribus"
    assert doc["decisions"] and doc["frames"][doc["decisions"][0]["frame"]]["actor_seat"] == doc["hero_seat"]


@pytest.mark.parametrize("path", [
    "/study/packs/nope/hands",
    "/study/packs/nope/hands/pluribus-100-0",
    "/study/packs/pluribus-sample/hands/pluribus-999-0",
    "/study/packs/pluribus-sample/hands/..%2Fpack",
    "/study/packs/..%2F..%2Fetc/hands",
    "/study/packs/.hidden/hands",
])
def test_unknown_ids_are_404(root, path):
    r = client(root).get(path)
    assert r.status_code == 404
    if "%2F" not in path:
        assert r.json()["error"]["code"] in ("pack_not_found", "hand_not_found")


def test_missing_or_empty_study_dir_lists_nothing(tmp_path):
    assert client(tmp_path / "absent").get("/study/packs").json()["packs"] == []
    assert client(tmp_path).get("/study/packs").json()["packs"] == []


def test_broken_packs_are_skipped_and_reported(root):
    meta_path = root / "pluribus-sample" / "pack.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps(meta | {"schema_version": 99}))
    c = client(root)
    assert c.get("/study/packs").json()["packs"] == []
    r = c.get("/study/packs/pluribus-sample/hands")
    assert r.status_code == 503 and r.json()["error"]["code"] == "pack_unreadable"
    meta_path.write_text(json.dumps(meta))
    (root / "pluribus-sample" / "hands" / "pluribus-100-0.json").unlink()
    r = client(root).get("/study/packs/pluribus-sample/hands/pluribus-100-0")
    assert r.status_code == 503 and r.json()["error"]["code"] == "pack_unreadable"


def test_pack_is_reloaded_when_rebuilt(root):
    lib = StudyLibrary(root)
    c = TestClient(create_app(background_pump=False, study=lib))
    assert c.get("/study/packs/pluribus-sample/hands").json()["total"] == 24
    write_pack(root / "pluribus-sample", list(iter_source(FIXTURE_DIR))[:3], pack_id="pluribus-sample",
               title="smaller", source_label="fixture")
    assert c.get("/study/packs/pluribus-sample/hands").json()["total"] == 3
