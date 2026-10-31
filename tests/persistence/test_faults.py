"""Spec 12.3 action-delivery faults against a durable store: duplicates, races, fencing, outages, reconnects."""

from __future__ import annotations

import json
import threading

import pytest

from arena.api.clock import FakeClock
from arena.api.manager import ApiError, SimulatedCrash
from tests.api.helpers import HUMAN, ORIGIN, Harness, auth
from tests.persistence.helpers import (
    TickingClock,
    boot,
    crash_on,
    mixed_table,
    play_humans,
    principal,
    raw_action,
    safe_action,
    stored_events,
)


def decision_events(events, decision_id: str):
    """Events from a DecisionRequested up to (not including) the next one."""
    start = next(i for i, e in enumerate(events) if e.type == "DecisionRequested" and e.data["decision_id"] == decision_id)
    out = []
    for e in events[start + 1:]:
        if e.type == "DecisionRequested":
            break
        out.append(e)
    return out


def consumed_once(events, decision_id: str) -> tuple[int, int]:
    """(manual ActionApplied count, ActionFallback count) that consumed this decision."""
    window = decision_events(events, decision_id)
    first = next(e for e in window if e.type in ("ActionApplied", "ActionFallback"))
    applied = sum(1 for e in window if e.type == "ActionApplied" and e.data["auto"] is None)
    fallbacks = sum(1 for e in window if e.type == "ActionFallback")
    assert first is not None
    return applied, fallbacks


def run_threads(targets) -> None:
    threads = [threading.Thread(target=t) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
        assert not t.is_alive()


# ------------------------------------------------------- duplicates and races


def test_duplicate_identical_actions_racing_commit_once(new_store):
    store = new_store()
    h = boot(store, FakeClock())
    h.setup([HUMAN, HUMAN], hand_cap=3)
    seat = h.actor()
    raw = raw_action(h, seat, action_id="dup")
    who = principal(h, seat)
    results: list[dict] = []
    barrier = threading.Barrier(8)

    def send() -> None:
        barrier.wait()
        results.append(h.manager.submit_action(h.table_id, who, raw))

    run_threads([send] * 8)
    assert sorted(r["replayed"] for r in results) == [False] + [True] * 7
    assert len({json.dumps(r["receipt"], sort_keys=True) for r in results}) == 1
    assert [r.action_id for r in store.load(h.table_id).receipts] == ["dup"]
    decision = results[0]["receipt"]["decision_id"]
    assert consumed_once(stored_events(store, h.table_id), decision) == (1, 0)


def test_same_action_id_with_changed_payload_conflicts_even_after_restart(new_store):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock)
    a.setup([HUMAN, HUMAN], hand_cap=3)
    seat = a.actor()
    view = a.snap(seat)["view"]
    assert a.post_action(seat, a.envelope(view, {"type": "call"}, "same")).status_code == 200
    b = boot(new_store(), clock, carry=a)
    r = b.post_action(seat, b.envelope(view, {"type": "fold"}, "same"))
    assert r.status_code == 409 and r.json()["error"]["code"] == "action_id_conflict"
    again = b.post_action(seat, b.envelope(view, {"type": "call"}, "same"))
    assert again.status_code == 200 and again.json()["replayed"] is True
    # A late response to the previous decision is stale, not applied.
    late = b.post_action(seat, b.envelope(view, {"type": "call"}, "late"))
    assert late.status_code == 409 and late.json()["error"]["code"] in {"stale_decision", "out_of_turn"}


def test_two_tabs_racing_different_action_ids_for_one_decision(new_store):
    store = new_store()
    h = boot(store, FakeClock())
    h.setup([HUMAN, HUMAN], hand_cap=3)
    seat = h.actor()
    view = h.snap(seat)["view"]
    decision = view["decision_id"]
    who = principal(h, seat)
    outcomes: list[tuple[int, str | None]] = []
    barrier = threading.Barrier(4)

    def tab(i: int):
        def send() -> None:
            raw = json.dumps(h.envelope(view, safe_action(view), f"tab-{i}")).encode()
            barrier.wait()
            try:
                h.manager.submit_action(h.table_id, who, raw)
                outcomes.append((200, None))
            except ApiError as exc:
                outcomes.append((exc.status, exc.code))
        return send

    run_threads([tab(i) for i in range(4)])
    assert sorted(s for s, _ in outcomes) == [200, 409, 409, 409]
    assert {c for _, c in outcomes if c} <= {"stale_decision", "stale_version", "out_of_turn", "wrong_hand"}
    assert len(store.load(h.table_id).receipts) == 1
    assert consumed_once(stored_events(store, h.table_id), decision) == (1, 0)


def test_action_racing_timeout_100_times(new_store):
    """Exactly one of the action and the timeout consumes each decision, decided by arrival time alone."""
    clock = TickingClock()
    store = new_store()
    h = boot(store, clock)
    counts = {"action": 0, "timeout": 0}
    for i in range(100):
        h.setup([HUMAN, HUMAN], human_turn_ms=1000, hand_cap=2)
        seat = h.actor()
        raw = raw_action(h, seat, action_id=f"race-{i}")
        who = principal(h, seat)
        live = h.manager.tables[h.table_id]
        deadline, decision = live.deadline_ns, live.table.decision_id
        clock.set_ns(deadline + (-8, -5, -3, -2, -1, 0, 1)[i % 7])
        barrier = threading.Barrier(2)
        seen: dict = {}

        def act() -> None:
            barrier.wait()
            seen["received"] = h.manager.register_action(h.table_id)
            try:
                h.manager.submit_action(h.table_id, who, raw, seen["received"], registered=True)
                seen["status"] = 200
            except ApiError as exc:
                seen["status"], seen["code"] = exc.status, exc.code

        def tick() -> None:
            barrier.wait()
            for _ in range(3):
                h.manager.pump()

        run_threads([act, tick])
        applied, fallbacks = consumed_once(stored_events(store, h.table_id), decision)
        on_time = seen["received"] <= deadline
        if on_time:
            assert seen["status"] == 200 and (applied, fallbacks) == (1, 0), (i, seen)
            counts["action"] += 1
        else:
            assert (seen["status"], seen.get("code")) == (409, "deadline_passed"), (i, seen)
            assert (applied, fallbacks) == (1, 1), (i, seen)
            counts["timeout"] += 1
        # Keep each pump to the table under test, so clock reads stay near the deadline.
        with h.manager._registry:
            del h.manager.tables[h.table_id]
    print(f"action-vs-timeout outcomes over 100 races: {counts}")
    assert counts["action"] > 0 and counts["timeout"] > 0, counts


# ------------------------------------------------------------- ownership


def test_obsolete_writer_is_fenced_and_cannot_fork_the_table(new_store):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock, writer_id="writer-a")
    a.setup([HUMAN, HUMAN], hand_cap=4)
    play_humans(a, 2)
    seat = a.actor()
    stale_raw = raw_action(a, seat, action_id="from-a")

    b = boot(new_store(), clock, carry=a, writer_id="writer-b", force=True)
    version = store.load(a.table_id).table.version
    with pytest.raises(ApiError) as err:
        a.manager.submit_action(a.table_id, principal(a, seat), stale_raw)
    assert (err.value.status, err.value.code) == (409, "writer_fenced")
    assert store.load(a.table_id).table.version == version
    assert all(r.action_id != "from-a" for r in store.load(a.table_id).receipts)
    assert a.get(a.tokens[seat]).status_code == 409
    a.manager.pump()
    assert store.load(a.table_id).table.version == version

    assert b.act(seat, safe_action(b.snap(seat)["view"])).status_code == 200
    assert store.load(b.table_id).owner_id == "writer-b"


def test_live_lease_blocks_a_second_writer(new_store):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock, writer_id="writer-a")
    a.setup([HUMAN, HUMAN], hand_cap=3)
    b = boot(new_store(), clock, carry=a, writer_id="writer-b")
    assert b.get(b.tokens[0]).json()["error"]["code"] == "writer_fenced"
    seat = a.actor()
    assert a.act(seat, safe_action(a.snap(seat)["view"])).status_code == 200
    assert store.load(a.table_id).owner_id == "writer-a"


# ---------------------------------------------- infrastructure pauses


def heads_up(h, turn_ms: int = 5000):
    h.setup([HUMAN, HUMAN], human_turn_ms=turn_ms, hand_cap=3)
    return h.actor()


def test_storage_outage_pauses_and_restores_human_think_time(new_store):
    clock = FakeClock()
    store = new_store()
    h = boot(store, clock)
    seat = heads_up(h)
    raw = raw_action(h, seat, action_id="during-outage")
    clock.advance_ms(2000)
    h.manager.heartbeat()
    store.available = False
    clock.advance_ms(400)
    r = h.client.post(f"/tables/{h.table_id}/actions", headers=auth(h.tokens[seat]), content=raw)
    assert r.status_code == 503 and r.json()["error"]["code"] == "storage_unavailable"
    assert h.get(h.tokens[seat]).status_code == 503
    clock.advance_ms(120_000)
    assert h.manager.pump() == 0, "no timeout fires while storage is down"
    store.available = True
    h.manager.pump()
    snap = h.snap(seat)
    assert snap["turn"]["seat"] == seat and snap["turn"]["remaining_ms"] == 3000
    pauses = store.pauses(h.table_id)
    assert [(p.reason, p.remaining_ms, p.resumed_at is not None) for p in pauses] == [("storage_outage", 3000, True)]
    r = h.client.post(f"/tables/{h.table_id}/actions", headers=auth(h.tokens[seat]), content=raw)
    assert r.status_code == 200 and r.json()["replayed"] is False


def test_outage_seen_only_by_heartbeat_pauses_every_table(new_store):
    clock = FakeClock()
    store = new_store()
    h = boot(store, clock)
    first = heads_up(h)
    first_table = h.table_id
    other = Harness(h.client, h.manager, clock)
    second = heads_up(other)
    clock.advance_ms(1000)
    h.manager.heartbeat()
    store.available = False
    clock.advance_ms(1000)
    h.manager.heartbeat()
    assert h.manager.outage_since is not None
    clock.advance_ms(30_000)
    assert h.manager.pump() == 0
    store.available = True
    h.manager.pump()
    h.table_id = first_table
    assert h.snap(first)["turn"]["remaining_ms"] == 4000
    assert other.snap(second)["turn"]["remaining_ms"] == 4000
    for table_id in (first_table, other.table_id):
        assert [p.reason for p in store.pauses(table_id)] == ["storage_outage"]


def test_ambiguous_commit_is_resolved_from_the_store(new_store, store_kind):
    if store_kind != "memory":
        pytest.skip("lost-acknowledgement injection exists only in the memory store")
    clock = FakeClock()
    store = new_store()
    h = boot(store, clock)
    seat = heads_up(h)
    raw = raw_action(h, seat, action_id="ambiguous")
    store.inner.lose_next_ack = True
    r = h.client.post(f"/tables/{h.table_id}/actions", headers=auth(h.tokens[seat]), content=raw)
    assert r.status_code == 503
    h.manager.pump()
    r = h.client.post(f"/tables/{h.table_id}/actions", headers=auth(h.tokens[seat]), content=raw)
    assert r.status_code == 200 and r.json()["replayed"] is True


def test_process_crash_restores_think_time_from_the_last_heartbeat(new_store):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock)
    seat = heads_up(a)
    clock.advance_ms(2000)
    a.manager.heartbeat()
    clock.advance_ms(300)  # lost with the process: it was never made durable
    clock.advance_ms(45_000)  # downtime
    b = boot(new_store(), clock, carry=a)
    assert b.snap(seat)["turn"]["remaining_ms"] == 3000
    [pause] = store.pauses(b.table_id)
    assert (pause.reason, pause.remaining_ms) == ("process_restart", 3000) and pause.resumed_at is not None
    clock.advance_ms(3000)
    assert b.manager.pump() == 0
    clock.advance_ms(1)
    assert b.manager.pump() == 1


def test_graceful_shutdown_records_the_pause_and_restart_closes_it(new_store):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock)
    seat = heads_up(a)
    clock.advance_ms(1500)
    a.manager.close()
    [pause] = store.pauses(a.table_id)
    assert (pause.reason, pause.remaining_ms, pause.resumed_at) == ("shutdown", 3500, None)
    clock.advance_ms(10_000)
    b = boot(new_store(), clock, carry=a)
    assert b.snap(seat)["turn"]["remaining_ms"] == 3500
    [pause] = store.pauses(b.table_id)
    assert pause.reason == "shutdown" and pause.resumed_at is not None


# ----------------------------------------------------------- reconnects


def stream(h, token: str):
    ws = h.client.websocket_connect(f"/tables/{h.table_id}/stream", headers={"origin": ORIGIN})
    sock = ws.__enter__()
    sock.send_json({"type": "auth", "token": token})
    return ws, sock


def test_reconnect_after_crash_before_broadcast_gets_the_committed_state_first(new_store):
    clock = FakeClock()
    store = new_store()
    a = boot(store, clock)
    mixed_table(a, hand_cap=4)
    seat = a.actor()
    ws, sock = stream(a, a.tokens[seat])
    versions = [sock.receive_json()["version"]]
    play_humans(a, 2)
    target = a.snap()["version"]
    while versions[-1] < target:
        frame = sock.receive_json()
        assert frame["version"] > versions[-1]
        versions.append(frame["version"])
    actor = a.actor()
    decision = a.snap()["turn"]["decision_id"]
    a.manager.faults["before_broadcast"] = crash_on("publish")
    with pytest.raises(SimulatedCrash):
        a.manager.submit_action(a.table_id, principal(a, actor), raw_action(a, actor))
    ws.__exit__(None, None, None)
    committed = store.load(a.table_id).table.version
    assert committed > versions[-1]

    b = boot(new_store(), clock, carry=a)
    ws2, sock2 = stream(b, b.tokens[seat])
    first = sock2.receive_json()
    assert first["version"] > committed > versions[-1]
    assert first["turn"] is None or first["turn"]["decision_id"] != decision
    assert first["view"]["hole_cards"]
    seen = [first["version"]]
    play_humans(b, 2)
    target = b.snap()["version"]
    while seen[-1] < target:
        frame = sock2.receive_json()
        assert frame["version"] > seen[-1]
        seen.append(frame["version"])
    ws2.__exit__(None, None, None)
    assert store.load(b.table_id).unpublished == [] or b.manager.pump() >= 0
    b.manager.pump()
    assert store.load(b.table_id).unpublished == []
