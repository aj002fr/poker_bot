"""Spec 12.1 invariant 10: replaying the accepted record reproduces hashes, chips and points."""

import json

import pytest

from arena.cli import main as cli
from arena.domain.canonical import canonical_json
from arena.history.canonical import TableHistory, export_history, load_history, write_history
from arena.history.replay import ReplayIncompatible, ReplayMismatch, replay
from tests.support.tables import RULESETS, play_table


def roundtrip(table) -> TableHistory:
    return TableHistory.loads(export_history(table).dumps())


@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("ruleset", RULESETS)
def test_replay_reproduces_every_hash_chip_and_point(ruleset, n):
    table = play_table(ruleset, n, f"replay-{n}", fallback_every=11)
    history = roundtrip(table)
    report = replay(history)
    assert report.complete and report.final_stacks == table.stacks
    assert report.points == {s: float(v) for s, v in table.points().items()}
    assert report.decisions_verified == sum(1 for e in table.log if e.type == "DecisionRequested") > 0
    assert report.hand_starts_verified == len(table.hands)
    assert report.notes == []


def test_deterministic_repeat_gives_identical_history():
    a = export_history(play_table("uncertainty_plo4", 6, "repeat")).dumps()
    b = export_history(play_table("uncertainty_plo4", 6, "repeat")).dumps()
    assert a == b


def test_state_hashes_exclude_table_and_decision_ids():
    def hashes(table):
        return [(e.type, e.data.get("state_hash") or e.data.get("start_hash"))
                for e in table.log if e.type in ("DecisionRequested", "HandStarted")]

    a = play_table("practice_nlhe", 5, "ids", table_id="t_one")
    b = play_table("practice_nlhe", 5, "ids", table_id="t_two")
    assert hashes(a) == hashes(b)


def test_incomplete_table_replays_to_its_last_committed_event():
    table = play_table("practice_nlhe", 4, "partial", hand_cap=5, hands_per_level=1)
    full = export_history(table)
    # Rebuild the same table but stop mid-way: a history exported from a live table.
    from arena.domain.models import TournamentConfig
    from arena.table.driver import legal_random, policy_rngs
    from arena.table.service import seeded_table

    live = seeded_table(TournamentConfig(initial_player_count=4, starting_stack=400, hand_cap=5, hands_per_level=1),
                        "partial", table_id=table.table_id)
    rngs = policy_rngs("partial", live.config.resolved_seats)
    for _ in range(6):
        if live.complete:
            break
        live.submit(live.actor_seat, legal_random(live.actor_seat, live.legal_actions(), rngs[live.actor_seat]))
    partial = roundtrip(live)
    assert not partial.summary.complete
    report = replay(partial)
    assert not report.complete and "incomplete" in report.notes[0]
    assert partial.events == full.events[: len(partial.events)]


# ------------------------------------------------------------------ tampering


@pytest.fixture(scope="module")
def stored():
    return json.loads(export_history(play_table("practice_nlhe", 6, "tamper", fallback_every=9)).dumps())


def _load(obj) -> TableHistory:
    return TableHistory.loads(canonical_json(obj))


def _first(obj, event_type, pred=lambda e: True):
    return next(e for e in obj["events"] if e["type"] == event_type and pred(e))


def test_untampered_fixture_replays(stored):
    assert replay(_load(stored)).complete


def test_swapped_undealt_cards_change_the_hand_start_hash(stored):
    obj = json.loads(json.dumps(stored))
    deck = _first(obj, "HandStarted")["data"]["deck"]
    deck[-1], deck[-2] = deck[-2], deck[-1]
    with pytest.raises(ReplayMismatch, match="hand start hash differs"):
        replay(_load(obj))


def test_swapped_hole_card_is_detected(stored):
    obj = json.loads(json.dumps(stored))
    deck = _first(obj, "HandStarted")["data"]["deck"]
    deck[0], deck[-1] = deck[-1], deck[0]
    with pytest.raises(ReplayMismatch):
        replay(_load(obj))


def test_forged_decision_hash_is_detected(stored):
    obj = json.loads(json.dumps(stored))
    _first(obj, "DecisionRequested")["data"]["state_hash"] = "0" * 64
    with pytest.raises(ReplayMismatch, match="decision state hash differs") as exc:
        replay(_load(obj))
    assert exc.value.decision_id == "d_00001"


def test_changed_action_is_detected(stored):
    obj = json.loads(json.dumps(stored))
    applied = _first(obj, "ActionApplied", lambda e: e["data"]["action"]["type"] == "raise_to")
    applied["data"]["action"]["amount"] += 1
    with pytest.raises(ReplayMismatch):
        replay(_load(obj))


def test_changed_payout_is_detected(stored):
    obj = json.loads(json.dumps(stored))
    done = _first(obj, "HandCompleted")["data"]["final_stacks"]
    a, b = sorted(done)[:2]
    done[a] += 1
    done[b] -= 1
    with pytest.raises(ReplayMismatch, match="HandCompleted differs"):
        replay(_load(obj))


def test_changed_summary_points_are_detected(stored):
    obj = json.loads(json.dumps(stored))
    seat = next(iter(obj["summary"]["points"]))
    obj["summary"]["points"][seat] += 1.0
    with pytest.raises(ReplayMismatch, match="summary"):
        replay(_load(obj))


def test_engine_version_mismatch_halts(stored):
    obj = json.loads(json.dumps(stored))
    obj["events"][0]["data"]["engine"]["pokerkit_version"] = "0.7.6"
    with pytest.raises(ReplayIncompatible):
        replay(_load(obj))
    assert replay(_load(obj), allow_engine_mismatch=True).notes


# ------------------------------------------------------------------------ CLI


def test_cli_replay_verifies_file_history(tmp_path, capsys):
    table = play_table("uncertainty_shortdeck", 5, "cli", table_id="t_cli")
    write_history(table, tmp_path)
    assert load_history(tmp_path / "t_cli.json") == export_history(table)
    assert cli(["replay", "--table-id", "t_cli", "--history-dir", str(tmp_path), "--verify-hashes"]) == 0
    out = capsys.readouterr().out
    assert "decision-boundary state hashes verified" in out and "replay matches" in out

    obj = json.loads((tmp_path / "t_cli.json").read_text())
    _first(obj, "DecisionRequested")["data"]["state_hash"] = "f" * 64
    (tmp_path / "t_cli.json").write_text(canonical_json(obj))
    assert cli(["replay", "--table-id", "t_cli", "--history-dir", str(tmp_path), "--verify-hashes"]) == 2
    assert "HALT" in capsys.readouterr().err


def test_cli_rejects_path_like_table_ids(tmp_path):
    with pytest.raises(ValueError):
        cli(["replay", "--table-id", "../etc/passwd", "--history-dir", str(tmp_path)])
