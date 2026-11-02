"""Pluribus PHH import: the committed fixture is byte-identical to the pinned upstream files,
every fixture hand replays through the arena engine, malformed records are rejected with
a reason, study packs are deterministic and show the hero nothing it could not see, and
the agreement diagnostic runs on the fixture."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest

from strategy.data import agreement, fetch, study
from strategy.data.phh_import import (FIXTURE_DIR, ImportRejected, PhhHand, import_all, import_hand, iter_source,
                                      map_size, parse_phh, provenance_for, scaled)

FIXTURES = FIXTURE_DIR.parent
HAND = (FIXTURE_DIR / "100" / "0.phh").read_text()


def _import(text: str, rel: str = "100/0.phh"):
    return import_hand(text, provenance_for(rel, text.encode()))


def _rejected(text: str) -> str:
    with pytest.raises(ImportRejected) as exc:
        _import(text)
    return exc.value.code


def test_fixture_files_match_the_pinned_manifest():
    source = json.loads((FIXTURES / "SOURCE.json").read_text())
    listed = {f["path"]: f["sha256"] for f in source["files"]}
    on_disk = {p.relative_to(FIXTURE_DIR).as_posix(): p for p in FIXTURE_DIR.rglob("*.phh")}
    assert set(listed) == set(on_disk)
    for rel, path in on_disk.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == listed[rel], rel
    pins = fetch.PLURIBUS
    assert (source["commit"], source["tree"], source["content_sha256"]) == (pins.commit, pins.tree,
                                                                            pins.content_sha256)
    assert re.fullmatch(r"[0-9a-f]{40}", pins.commit) and re.fullmatch(r"[0-9a-f]{64}", pins.content_sha256)
    assert (FIXTURES / source["licence_file"]).read_text().startswith("MIT License")


def test_every_fixture_hand_imports():
    report = import_all(FIXTURE_DIR, label="fixture")
    assert (report.hands, report.imported, dict(report.rejected)) == (24, 24, {})
    assert report.pluribus_decisions == 81
    assert set(report.by_category) == {"fold", "check", "call", "raise"}
    assert all(report.by_street[s] for s in ("preflop", "flop", "turn", "river"))
    assert report.raises > 0 and report.size_log_error_max < 0.5


def test_imported_hand_matches_the_record():
    hand = _import(HAND)
    assert hand.seat_names == dict(enumerate(["MrBlue", "MrBlonde", "MrWhite", "MrPink", "MrBrown", "Pluribus"]))
    assert hand.positions[0] == "SB" and hand.positions[5] == "BTN"
    assert hand.finishing_stacks == {0: 10310, 1: 9900, 2: 10000, 3: 9790, 4: 10000, 5: 10000}
    assert hand.board == ["7d", "5h", "9d", "7c", "Qh"]
    assert [(d.player, d.category) for d in hand.decisions][:4] == [
        ("MrWhite", "fold"), ("MrPink", "raise"), ("MrBrown", "fold"), ("Pluribus", "fold")]
    river_bet = hand.decisions[-2]
    assert (river_bet.player, river_bet.street, river_bet.action.amount) == ("MrBlue", "river", 230)


def test_half_chip_split_is_validated_at_double_stakes():
    text = (FIXTURE_DIR / "102" / "0.phh").read_text()
    hand = import_hand(text, provenance_for("102/0.phh", text.encode()))
    assert any(n.startswith("half-chip split") for n in hand.notes)


def test_scaled_multiplies_every_amount():
    phh = parse_phh(HAND)
    double = scaled(phh, 2)
    assert isinstance(double, PhhHand)
    assert double.starting_stacks == tuple(2 * s for s in phh.starting_stacks)
    assert double.blinds == tuple(2 * b for b in phh.blinds) and double.min_bet == 2 * phh.min_bet
    assert "p4 cbr 420" in double.actions


def test_sizes_map_to_the_nearest_canonical_option():
    hand = _import((FIXTURE_DIR / "100" / "11.phh").read_text(), "100/11.phh")
    raises = [d for d in hand.decisions if d.size is not None]
    assert raises
    for d in raises:
        assert map_size(d.action.amount, d.observation, d.menu) == d.size
        for option in d.menu:
            if option.amount is not None:
                exact = map_size(option.amount, d.observation, d.menu)
                assert exact.log_error == 0.0 and exact.option.amount == option.amount


@pytest.mark.parametrize(("old", "new", "code"), [
    ("finishing_stacks = [10310,", "finishing_stacks = [10300,", "finishing_stacks"),
    ("'p4 cbr 210'", "'p5 cbr 210'", "actor"),
    ("'p3 f'", "'p3 cbr 150'", "illegal_action"),
    ("variant = 'NT'", "variant = 'FT'", "variant"),
    ("antes = [0, 0, 0, 0, 0, 0]", "antes = [5, 5, 5, 5, 5, 5]", "antes"),
    ("blinds_or_straddles = [50, 100, 0, 0, 0, 0]", "blinds_or_straddles = [50, 100, 200, 0, 0, 0]", "blinds"),
    ("min_bet = 100", "min_bet = 200", "blinds"),
    ("starting_stacks = [10000, 10000,", "starting_stacks = [9000, 10000,", "stacks"),
    ("'d dh p2 8s4c'", "'d dh p2 TcQc'", "cards"),
    (", 'd db Qh', 'p1 cbr 230', 'p4 f']", "]", "incomplete"),
    ("'d db 7c'", "'d db 7c8c'", "board"),
    ("'p1 cbr 230', 'p4 f'", "'p1 cc', 'p4 cc', 'p1 sm 2c3c'", "showdown"),
    ("variant = 'NT'", "variant = ", "toml"),
])
def test_malformed_records_are_rejected_with_a_reason(old, new, code):
    assert old in HAND
    assert _rejected(HAND.replace(old, new, 1)) == code


def test_six_max_only():
    five = (HAND.replace("[0, 0, 0, 0, 0, 0]", "[0, 0, 0, 0, 0]")
            .replace("[50, 100, 0, 0, 0, 0]", "[50, 100, 0, 0, 0]")
            .replace("[10000, 10000, 10000, 10000, 10000, 10000]", "[10000, 10000, 10000, 10000, 10000]"))
    assert _rejected(five) == "not_six_max"


def test_fetch_verify_reports_a_missing_or_altered_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DATA_CACHE", str(tmp_path))
    assert fetch.cache_root() == tmp_path
    with pytest.raises(fetch.FetchError):
        fetch.verify()
    root = fetch.data_dir()
    (root / "100").mkdir(parents=True)
    (root / "100" / "0.phh").write_text(HAND)
    lines = fetch.manifest_lines(root)
    assert lines == [f"100/0.phh {hashlib.sha256(HAND.encode()).hexdigest()}"]
    with pytest.raises(fetch.FetchError, match="holds 1 hands"):
        fetch.verify()
    with pytest.raises(fetch.FetchError, match="content hash"):
        fetch.verify(dataclasses.replace(fetch.PLURIBUS, files=1))
    one = dataclasses.replace(fetch.PLURIBUS, files=1, content_sha256=fetch.manifest_hash(lines))
    assert fetch.verify(one) == root


@pytest.fixture(scope="module")
def pack(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("study") / "pluribus-sample"
    study.write_pack(out, iter_source(FIXTURE_DIR), pack_id="pluribus-sample", title="t", source_label="fixture")
    return out


def _hands(pack: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((pack / "hands").glob("*.json"))]


def test_study_pack_is_deterministic(pack, tmp_path):
    again = tmp_path / "pluribus-sample"
    study.write_pack(again, iter_source(FIXTURE_DIR), pack_id="pluribus-sample", title="t", source_label="fixture")
    for name in ("pack.json", "index.json"):
        assert (pack / name).read_bytes() == (again / name).read_bytes()
    for p in (pack / "hands").glob("*.json"):
        assert p.read_bytes() == (again / "hands" / p.name).read_bytes()


def test_study_pack_schema(pack):
    meta = json.loads((pack / "pack.json").read_text())
    index = json.loads((pack / "index.json").read_text())["hands"]
    assert (meta["format"], meta["schema_version"], meta["hands"], meta["decisions"]) == (
        "arena.study-pack", study.STUDY_SCHEMA_VERSION, 24, 81)
    assert meta["rejected"] == {} and meta["dataset"]["commit"] == fetch.PLURIBUS.commit
    assert [h["id"] for h in index] == sorted((h["id"] for h in index), key=lambda i: (i.split("-")[1],
                                                                                      int(i.split("-")[2])))
    for doc, entry in zip(_hands(pack), sorted(index, key=lambda h: h["id"]), strict=True):
        assert doc["format"] == study.STUDY_FORMAT and doc["id"] == entry["id"]
        assert doc["hero_name"] == "Pluribus" and len(doc["seats"]) == 6
        assert entry["decisions"] == len(doc["decisions"]) >= 1
        assert doc["frames"][0]["kind"] == "deal" and doc["frames"][-1]["kind"] == "hand_end"
        assert sum(doc["result"]["net"].values()) == 0
        for d in doc["decisions"]:
            frame = doc["frames"][d["frame"]]
            assert frame["actor_seat"] == doc["hero_seat"]
            keys = {o["key"] for o in d["options"]}
            assert d["pluribus"]["option"] in keys
            assert 0.0 <= d["spot"]["equity"] <= 1.0 and 0.0 <= d["spot"]["pot_odds"] < 1.0


def test_study_frames_show_only_what_pluribus_saw(pack):
    for doc in _hands(pack):
        hero = doc["hero_seat"]
        others = {c for s, cards in doc["result"]["hole_cards"].items() if int(s) != hero for c in cards}
        shown = {c for s in doc["result"]["showdown"] if s != hero for c in doc["result"]["hole_cards"][str(s)]}
        assert doc["hole_cards"] == doc["result"]["hole_cards"][str(hero)]
        public = json.dumps([doc["frames"], doc["events"], doc["decisions"]])
        for card in others - shown:
            assert f'"{card}"' not in public or card in doc["result"]["board"], (doc["id"], card)
        for d in doc["decisions"]:
            seq = doc["frames"][d["frame"]]["seq"]
            seen = [e for e in doc["events"] if e["seq"] <= seq]
            assert not any(e["type"] == "CardsRevealed" for e in seen)
            assert not any(e["type"] == "ActionApplied" and e["data"]["seat"] == hero and e["seq"] > seq
                           for e in seen)


def test_write_pack_refuses_to_replace_other_directories(tmp_path):
    (tmp_path / "keep.txt").write_text("not a pack")
    with pytest.raises(FileExistsError):
        study.write_pack(tmp_path, iter([]), pack_id="x", title="t", source_label="fixture")
    assert (tmp_path / "keep.txt").exists()


def test_agreement_diagnostic_on_the_fixture():
    bots = ("baseline:nlhe-v2", "arena:calling_station", "arena:check_fold", "style:rock")
    report = agreement.run(FIXTURE_DIR, bots, replicates=2, label="fixture")
    assert report["hands"] == 24 and report["rejected"] == 0
    rows = {r["bot"]: r for r in report["bots"]}
    assert all(r["decisions"] == 81 and r["illegal"] == 0.0 and r["errors"] == 0.0 for r in rows.values())
    station, folder = rows["arena:calling_station"], rows["arena:check_fold"]
    assert station["by_pluribus_kind"]["fold"] == 0.0 and station["by_pluribus_kind"]["call"] == 1.0
    assert folder["by_pluribus_kind"]["fold"] == 1.0 and folder["by_pluribus_kind"]["raise"] == 0.0
    assert folder["balanced_agreement"] == 0.5
    assert agreement.run(FIXTURE_DIR, bots, replicates=2, label="fixture") == report
    assert "baseline:nlhe-v2" in agreement.markdown(report)
