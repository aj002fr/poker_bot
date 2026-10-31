"""Spec 12.3 agent faults through real sandboxed workers: one correct fallback per bad
decision, the table completes, and the other seats are unaffected."""

from __future__ import annotations

import pytest

from arena.agents.sdk import UnsupportedRuleset
from arena.agents.worker import AgentLoadError, WorkerSeat
from arena.benchmark.runner import Entrant, run_benchmark
from tests.agents.conftest import bot, info_for, needs_sandbox, play_sandboxed, small_config

pytestmark = needs_sandbox

GOOD = ["calling_station", "calling_station"]
# Healthy seats may still miss the 1 ms deadline through VM jitter; that is the only
# fallback they may record.
JITTER_OK = {None, "timeout"}


def _number(decision_id: str) -> int:
    return int(decision_id[2:])


def _assert_others_clean(played, bad_seat: int = 0) -> None:
    for seat, agent in played.seats.items():
        if seat != bad_seat:
            assert {f for _, f in played.seat_records(seat)} <= JITTER_OK
            assert agent.stats.restarts == []


def _fallback_events(table, seat: int) -> list[dict]:
    return [e.data for e in table.log if e.type == "ActionFallback" and e.data["seat"] == seat]


def test_illegal_action_falls_back_every_time(pool):
    played = play_sandboxed(pool, ["illegal_bot", *GOOD])
    assert played.table.complete
    recs = played.seat_records(0)
    assert recs and all(f == "illegal_action" for _, f in recs)
    events = _fallback_events(played.table, 0)
    assert len(events) == len(recs)
    assert all(e["reason"] == "illegal_action" and e["rejected"] == {"action": {"type": "raise_to", "amount": 1}}
               for e in events)
    # State never commits without an accepted action.
    assert played.seats[0].committed == b'{"n":0}'
    _assert_others_clean(played)


def test_invalid_schema_falls_back(pool):
    played = play_sandboxed(pool, ["invalid_schema_bot", *GOOD])
    assert {f for _, f in played.seat_records(0)} == {"invalid_schema"}
    _assert_others_clean(played)


def test_bot_exception_consumes_only_that_decision(pool):
    played = play_sandboxed(pool, ["raising_bot", *GOOD])
    for decision_id, fallback in played.seat_records(0):
        if _number(decision_id) % 3 == 0:
            assert fallback == "bot_exception"
        else:
            assert fallback in JITTER_OK
    assert played.seats[0].stats.restarts == []
    assert any(e["rejected"]["detail"].startswith("ValueError: deliberate") for e in _fallback_events(played.table, 0))
    _assert_others_clean(played)


def test_crashing_worker_is_restored_from_committed_state(pool):
    played = play_sandboxed(pool, ["crash_bot", *GOOD], config=small_config(3, hand_cap=10))
    crashes = [d for d, _ in played.seat_records(0) if _number(d) % 7 == 0]
    assert crashes, "seed should give the crash bot at least one crash decision"
    for decision_id, fallback in played.seat_records(0):
        if _number(decision_id) % 7 == 0:
            # EOF from a dying process can arrive after the deadline (address-space
            # teardown), in which case the parent correctly saw a timeout first.
            assert fallback in ("worker_crash", "timeout")
        else:
            assert fallback in JITTER_OK
    seat = played.seats[0]
    # A crash on the seat's last decision of the table is never followed by a restore.
    assert len(crashes) - 1 <= len(seat.stats.restarts) <= len(crashes)
    assert all(r.restored and r.cause in ("worker_crash", "unresponsive") for r in seat.stats.restarts)
    accepted = sum(1 for _, f in played.seat_records(0) if f is None)
    assert seat.committed == f'{{"accepted":{accepted}}}'.encode()
    _assert_others_clean(played)


def test_looping_worker_times_out_and_is_replaced(pool):
    played = play_sandboxed(pool, ["loop_bot", *GOOD])
    loops = [d for d, _ in played.seat_records(0) if _number(d) % 5 == 0]
    assert loops
    for decision_id, fallback in played.seat_records(0):
        if _number(decision_id) % 5 == 0:
            assert fallback == "timeout"
    seat = played.seats[0]
    unresponsive = [r for r in seat.stats.restarts if r.cause == "unresponsive"]
    # Every loop is killed when the seat is next asked, except possibly the last one.
    assert len(loops) - 1 <= len(unresponsive) <= len(loops)
    assert all(r.restored for r in seat.stats.restarts)
    assert played.table.complete
    _assert_others_clean(played)


def test_late_reply_is_discarded_and_state_not_committed(pool):
    played = play_sandboxed(pool, ["slow_bot", *GOOD])
    slow = [d for d, _ in played.seat_records(0) if _number(d) % 4 == 0]
    assert slow
    for decision_id, fallback in played.seat_records(0):
        if _number(decision_id) % 4 == 0:
            assert fallback == "timeout"
    seat = played.seats[0]
    assert seat.stats.late_discarded >= len(slow) - 1
    assert seat.stats.restarts == []
    accepted = sum(1 for _, f in played.seat_records(0) if f is None)
    assert seat.committed == f'{{"accepted":{accepted}}}'.encode()
    _assert_others_clean(played)


def test_oversized_state_is_rejected_at_the_cap(pool):
    played = play_sandboxed(pool, ["oversized_state_bot", *GOOD])
    for decision_id, fallback in played.seat_records(0):
        if _number(decision_id) % 2 == 0:
            assert fallback == "state_too_large"
        else:
            assert fallback in JITTER_OK
    seat = played.seats[0]
    if any(f is None for _, f in played.seat_records(0)):
        assert seat.stats.max_state_bytes == 65536
    _assert_others_clean(played)


def test_oversized_output_is_rejected_and_worker_replaced(pool):
    played = play_sandboxed(pool, ["oversized_output_bot", *GOOD])
    recs = played.seat_records(0)
    assert recs and {f for _, f in recs} == {"oversized_output"}
    assert len(played.seats[0].stats.restarts) >= len(recs) - 1
    _assert_others_clean(played)


def test_forged_frames_are_rejected(pool):
    played = play_sandboxed(pool, ["forged_frame_bot", *GOOD])
    for decision_id, fallback in played.seat_records(0):
        assert fallback == ("invalid_output" if _number(decision_id) % 2 else "oversized_output")
    _assert_others_clean(played)


def test_memory_limit_turns_into_a_fallback(pool):
    played = play_sandboxed(pool, ["memory_bot", *GOOD])
    recs = played.seat_records(0)
    assert recs and {f for _, f in recs} == {"bot_memory_error"}
    assert played.seats[0].stats.restarts == []
    _assert_others_clean(played)


def test_output_spam_is_capped_and_does_not_block(pool):
    played = play_sandboxed(pool, ["spam_bot", *GOOD], budget_ns=50_000_000)
    assert {f for _, f in played.seat_records(0)} <= JITTER_OK
    assert played.seats[0].stats.restarts == []


def test_unsupported_ruleset_is_rejected_before_the_table_starts(pool, tmp_path):
    config = small_config(3)
    seat = WorkerSeat(0, pool.stage(bot("plo_only_bot")), info_for(config, 0), pool.acquire)
    with pytest.raises(UnsupportedRuleset):
        seat.start()
    assert seat.channel is None
    entrants = [Entrant("plo_only", bot("plo_only_bot")), Entrant("cs", bot("calling_station"), True),
                Entrant("cs2", bot("calling_station"), True)]
    with pytest.raises(UnsupportedRuleset):
        run_benchmark(config=config, entrants=entrants, blocks=1, base_seed="x", out_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_failing_bot_load_is_reported(pool):
    config = small_config(3)
    seat = WorkerSeat(0, pool.stage(bot("load_fail_bot")), info_for(config, 0), pool.acquire)
    with pytest.raises(AgentLoadError) as exc:
        seat.start()
    assert exc.value.code == "bot_exception" and "cannot build bot" in exc.value.detail


def test_instance_state_bot_is_flagged_not_restorable_after_a_kill(pool):
    def kill(index, seat, seats):
        if seat == 0 and seats[0].stats.decisions == 2:
            seats[0].kill_worker()

    played = play_sandboxed(pool, ["instance_bot", *GOOD], before_decision=kill)
    seat = played.seats[0]
    assert [(r.cause, r.restored) for r in seat.stats.restarts] == [("infrastructure_kill", False)]
    assert not seat.stats.restorable
    assert played.table.complete
