"""Spec 12.3: process crash before/during/after commit and before broadcast; restart restores state."""

from __future__ import annotations

import pytest

from arena.api.clock import FakeClock
from arena.api.manager import SimulatedCrash
from arena.domain.models import TournamentConfig
from arena.persistence.recovery import rebuild_table
from tests.api.helpers import HUMAN, bot
from tests.persistence.helpers import (
    boot,
    chips_conserved,
    crash_on,
    mixed_table,
    play_humans,
    principal,
    raw_action,
    stored_events,
)


def finish(h, limit: int = 2000) -> None:
    for _ in range(limit):
        if h.snap()["status"] == "complete":
            return
        play_humans(h, 1)
    raise AssertionError("table did not finish")


def verify_record(store, table_id: str) -> None:
    """The durable record replays from scratch: no double debit, reshuffle, duplicate payout or decision."""
    stored = store.load(table_id)
    events = stored_events(store, table_id)
    table = rebuild_table(TournamentConfig.model_validate(stored.table.config), events, None,
                          seed=stored.table.deck_seed)
    assert table.log == events
    assert chips_conserved(events)
    decisions = [e.data["decision_id"] for e in events if e.type == "DecisionRequested"]
    assert len(decisions) == len(set(decisions))
    hands = [e.data["hand_number"] for e in events if e.type == "HandStarted"]
    assert hands == list(range(1, len(hands) + 1))
    if table.complete:
        awards = {a.seat: a.points for a in stored.awards}
        assert awards == {s: float(v) for s, v in table.points().items()}


def test_restart_resumes_the_same_table(new_store):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock)
    mixed_table(a, seed="restart", hand_cap=6)
    play_humans(a, 5)
    before = {s: a.snap(s) for s in a.tokens}
    log_before = stored_events(store, a.table_id)

    b = boot(new_store(), clock, carry=a)
    after = {s: b.snap(s) for s in b.tokens}
    for s in b.tokens:
        assert after[s]["view"] == before[s]["view"]
        assert after[s]["turn"]["decision_id"] == before[s]["turn"]["decision_id"]
        assert after[s]["version"] == before[s]["version"] + 1  # the recorded restart pause
    assert stored_events(store, b.table_id) == log_before
    finish(b)
    assert b.snap()["results"]["hands_played"] <= 6
    verify_record(store, b.table_id)


@pytest.mark.parametrize("point", ["before_tx", "in_tx", "after_commit", "before_broadcast"])
def test_crash_during_a_human_action(new_store, point):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock)
    mixed_table(a, seed=f"crash-{point}", hand_cap=5)
    play_humans(a, 3)
    seat = a.snap()["turn"]["seat"]
    raw = raw_action(a, seat, action_id="the-action")
    before = store.load(a.table_id)
    deck_before = a.manager.tables[a.table_id].table.hands[-1].deck

    a.manager.faults[point] = crash_on("publish" if point == "before_broadcast" else "action")
    with pytest.raises(SimulatedCrash):
        a.manager.submit_action(a.table_id, principal(a, seat), raw)

    durable = point in ("after_commit", "before_broadcast")
    mid = store.load(a.table_id)
    assert (mid.table.version > before.table.version) is durable
    assert any(r.action_id == "the-action" for r in mid.receipts) is durable
    if not durable:
        assert mid.events == before.events
    else:
        assert mid.table.version in mid.unpublished, "a crash before broadcast leaves the outbox row pending"

    b = boot(new_store(), clock, carry=a)
    live = b.manager.tables[b.table_id]
    assert live.table.hands[-1].deck == deck_before or durable, "an interrupted hand is never reshuffled"
    out = b.manager.submit_action(b.table_id, principal(b, seat), raw)
    assert out["replayed"] is durable
    events = stored_events(store, b.table_id)
    applied = [e for e in events if e.type == "ActionApplied" and e.data["seat"] == seat and e.data["auto"] is None]
    assert len([e for e in applied if e.seq > len(before.events) - 1]) >= 1
    receipts = [r for r in store.load(b.table_id).receipts if r.action_id == "the-action"]
    assert len(receipts) == 1
    b.manager.pump()
    assert store.load(b.table_id).unpublished == []
    finish(b)
    verify_record(store, b.table_id)


@pytest.mark.parametrize("point", ["before_tx", "in_tx", "after_commit"])
@pytest.mark.parametrize("kind", ["bot", "timeout"])
def test_crash_during_bot_and_timeout_commits(new_store, point, kind):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock, bot_delay_ms=100)
    a.setup([HUMAN, bot("legal_random"), bot("check_or_call")], seed=f"crash-{kind}-{point}", human_turn_ms=1000,
            hand_cap=5)
    a.manager.faults[point] = crash_on(kind)
    with pytest.raises(SimulatedCrash):
        for _ in range(200):
            clock.advance_ms(1500)
            a.manager.pump()
    b = boot(new_store(), clock, carry=a, bot_delay_ms=100)
    for _ in range(3000):
        if b.snap()["status"] == "complete":
            break
        clock.advance_ms(1500)
        b.manager.pump()
    assert b.snap()["status"] == "complete"
    verify_record(store, b.table_id)


def test_seeded_bot_table_interrupted_twice_matches_an_uninterrupted_run(new_store):
    """Dealer and bot RNG checkpoints restore exactly: the restarted run equals the clean run."""
    seats = [bot("legal_random"), bot("legal_random"), bot("check_or_call")]

    def run(crashes: list[int]) -> list[tuple]:
        clock = FakeClock()
        store = new_store()
        h = boot(store, clock, bot_delay_ms=10)
        h.setup(seats, seed="bots-checkpoint", hand_cap=12)
        steps = 0
        while h.snap()["status"] != "complete":
            clock.advance_ms(10)
            steps += 1
            if crashes and steps == crashes[0]:
                crashes.pop(0)
                h.manager.faults["after_commit"] = crash_on("bot")
                with pytest.raises(SimulatedCrash):
                    h.manager.pump()
                h = boot(new_store(), clock, carry=h, bot_delay_ms=10)
            else:
                h.manager.pump()
        return [(e.type, e.hand_id, e.data) for e in stored_events(store, h.table_id)]

    clean = run([])
    assert run([7, 40]) == clean


def test_corrupt_stored_event_halts_the_table(new_store):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock)
    mixed_table(a, seed="tamper", hand_cap=5)
    play_humans(a, 4)
    tamper_last_payout(store, a.table_id)
    b = boot(new_store(), clock, carry=a)
    r = b.get(b.tokens[0])
    assert r.status_code == 503 and r.json()["error"]["code"] == "table_halted"
    assert "hole" not in r.text and "deck" not in r.text
    r = b.post_action(0, {"action_id": "x"})
    assert r.status_code == 503
    assert store.load(b.table_id).table.status == "halted"
    # Still halted after another restart: nothing repairs the record.
    c = boot(new_store(), clock, carry=b)
    assert c.get(c.tokens[0]).status_code == 503


def tamper_last_payout(store, table_id: str) -> None:
    inner = store.inner
    if inner.kind == "memory":
        events = inner._tables[table_id].events
        i = max(i for i, e in enumerate(events) if e["type"] == "HoleDealt")
        events[i]["data"]["card"] = "2c" if events[i]["data"]["card"] != "2c" else "3c"
    else:
        with inner._conn() as conn:
            conn.execute(
                "UPDATE events SET data = jsonb_set(data, '{card}', (CASE WHEN data->>'card' = '2c' THEN '\"3c\"' "
                "ELSE '\"2c\"' END)::jsonb) WHERE table_id = %s AND seq = (SELECT max(seq) FROM events WHERE table_id = %s "
                "AND type = 'HoleDealt')", (table_id, table_id))
