"""Snapshot + event replay reproduces a table exactly, at every boundary, or refuses."""

from __future__ import annotations

import json

import pytest

from arena.domain.events import EventEnvelope
from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng
from arena.history.replay import ReplayIncompatible, ReplayMismatch
from arena.persistence.recovery import rebuild_table
from arena.table.driver import legal_random, policy_rngs
from arena.table.service import Table, seeded_table
from tests.support.tables import RULESETS


def stored(events):
    return [EventEnvelope.from_json_obj(json.loads(json.dumps(e.to_json_obj()))) for e in events]


def play(ruleset: str, n: int, seed: str | None, *, fallback_every: int = 7):
    config = TournamentConfig(ruleset_id=ruleset, initial_player_count=n, starting_stack=300)
    points: list[dict] = []
    capture = lambda t: points.append(json.loads(json.dumps(t.resume_point())))  # noqa: E731
    table = (seeded_table(config, seed, table_id="t_resume", on_hand_start=capture) if seed is not None
             else Table(config, table_id="t_resume", on_hand_start=capture))
    rngs = policy_rngs(seed or "unseeded-policy", config.resolved_seats)
    boundaries = []
    decisions = 0
    while not table.complete:
        boundaries.append((len(table.log), len(points)))
        decisions += 1
        seat = table.actor_seat
        if decisions % fallback_every == 0:
            table.apply_fallback(seat, "timeout", None)
        else:
            table.submit(seat, legal_random(seat, table.legal_actions(), rngs[seat]))
    boundaries.append((len(table.log), len(points)))
    return config, table, points, boundaries


def test_seeded_rng_checkpoint_continues_the_same_stream():
    a = SeededRng("x", "env")
    [a.randbelow(97) for _ in range(5)]
    b = SeededRng.from_state(json.loads(json.dumps(a.getstate())))
    assert [a.randbelow(1000) for _ in range(50)] == [b.randbelow(1000) for _ in range(50)]


@pytest.mark.parametrize("ruleset", RULESETS)
@pytest.mark.parametrize("seed", ["resume-a", None])
def test_resume_from_every_snapshot_and_every_commit_boundary(ruleset, seed):
    config, table, points, boundaries = play(ruleset, 4, seed)
    events = stored(table.log)
    assert len(points) == len(table.hands) > 1
    for cut, n_points in boundaries:
        prefix = events[:cut]
        point = points[n_points - 1]
        rebuilt = rebuild_table(config, prefix, point, seed=seed)
        assert rebuilt.log == prefix
        if cut == len(events):
            assert rebuilt.complete and rebuilt.stacks == table.stacks
            assert rebuilt.points() == table.points()
        else:
            assert rebuilt.decision_id is not None and rebuilt.actor_seat is not None
    # Full replay without any snapshot gives the same result.
    assert rebuild_table(config, events, None, seed=seed).log == events


def test_seeded_resume_continues_dealing_the_original_decks():
    config, table, points, boundaries = play("practice_nlhe", 3, "resume-continue")
    events = stored(table.log)
    second = next(cut for cut, n in boundaries if n == 2)
    # Resume from the older hand-1 snapshot; recovery deals hand 2 from the dealer stream itself.
    rebuilt = rebuild_table(config, events[:second], points[0], seed="resume-continue")
    # Play the rest of the table with the original commands; dealing must follow the original stream.
    from arena.history.replay import commands_from_events

    for c in commands_from_events(events[second:]):
        if c.fallback_reason:
            rebuilt.apply_fallback(c.seat, c.fallback_reason, c.rejected)
        else:
            rebuilt.submit(c.seat, c.action)
    assert rebuilt.log == events


def test_tampered_snapshot_halts():
    config, table, points, _ = play("practice_nlhe", 4, "resume-tamper")
    events = stored(table.log)
    point = dict(points[2])
    stacks = dict(point["stacks"])
    a, b = list(stacks)[:2]
    stacks[a], stacks[b] = stacks[a] + 10, stacks[b] - 10
    point["stacks"] = stacks
    with pytest.raises(ReplayMismatch, match="hand start hash differs|HandStarted differs"):
        rebuild_table(config, events, point, seed="resume-tamper")


def test_tampered_event_halts():
    config, table, points, _ = play("practice_nlhe", 4, "resume-event")
    events = stored(table.log)
    i = next(i for i, e in enumerate(events) if e.type == "PotAwarded" and i > points[-1]["seq"])
    bad = events[i].model_copy(update={"data": {**events[i].data, "amount": events[i].data["amount"] + 1}})
    with pytest.raises(ReplayMismatch):
        rebuild_table(config, events[:i] + [bad] + events[i + 1:], points[-1], seed="resume-event")


def test_wrong_dealer_state_halts_instead_of_reshuffling():
    config, table, points, _ = play("practice_nlhe", 4, "resume-dealer")
    events = stored(table.log)
    point = dict(points[3])
    point["env_rng"] = SeededRng("someone-else", "env").getstate()
    with pytest.raises(ReplayMismatch, match="hand start hash differs"):
        rebuild_table(config, events, point, seed="resume-dealer")


def test_engine_version_change_is_incompatible():
    config, table, points, _ = play("practice_nlhe", 2, "resume-engine")
    events = stored(table.log)
    first = events[0]
    engine = {**first.data["engine"], "pokerkit_version": "0.0.1"}
    events[0] = first.model_copy(update={"data": {**first.data, "engine": engine}})
    with pytest.raises(ReplayIncompatible):
        rebuild_table(config, events, points[-1], seed="resume-engine")
