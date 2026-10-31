"""Exact replay for explicit-state bots: restored execution equals uninterrupted execution.

These use a large diagnostic budget so that VM jitter cannot turn a decision into a
timeout fallback, which would legitimately change the game. The deadline itself is
covered with a fake clock in ``test_deadline.py``.
"""

from __future__ import annotations

from arena.agents.baselines import load_baseline
from arena.agents.sdk import load_bot
from arena.agents.worker import InProcessSeat, play_table
from arena.table.service import seeded_table
from tests.agents.conftest import bot, info_for, needs_sandbox, play_sandboxed, small_config

pytestmark = needs_sandbox

BOTS = ["template_bot", "legal_random", "nlhe_heuristic"]
BIG_BUDGET = 250_000_000
CONFIG = small_config(3, hand_cap=24, starting_stack=600)


def _template_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "examples" / "template_bot.py"


def _bot(name: str):
    return _template_path() if name == "template_bot" else bot(name)


def _log(table) -> list[dict]:
    return [e.to_json_obj() for e in table.log]


def _play(pool, kills: set[int] = frozenset(), seed: str = "restore"):
    from arena.agents.worker import WorkerSeat

    seats = {}
    for s, name in zip(CONFIG.resolved_seats, BOTS):
        seats[s] = WorkerSeat(s, pool.stage(_bot(name)), info_for(CONFIG, s, seed=seed), pool.acquire,
                              budget_ns=BIG_BUDGET)
        seats[s].start()
    table = seeded_table(CONFIG, seed, table_id="t_test")

    def before(index: int, seat: int) -> None:
        if index in kills:
            seats[seat].kill_worker()

    try:
        play_table(table, seats, before_decision=before)
    finally:
        for a in seats.values():
            a.close()
    return table, seats


def test_restored_run_matches_uninterrupted_run(pool):
    plain, plain_seats = _play(pool)
    killed, killed_seats = _play(pool, kills={2, 5, 9, 17, 30, 31})
    assert plain.complete and killed.complete
    assert all(sum(a.stats.fallbacks.values()) == 0 for a in plain_seats.values())
    assert all(sum(a.stats.fallbacks.values()) == 0 for a in killed_seats.values())
    restarts = [r for a in killed_seats.values() for r in a.stats.restarts]
    assert len(restarts) >= 4 and all(r.restored and r.cause == "infrastructure_kill" for r in restarts)
    assert _log(killed) == _log(plain)
    for s in CONFIG.resolved_seats:
        assert killed_seats[s].committed_history == plain_seats[s].committed_history


def test_sandboxed_run_matches_in_process_run(pool):
    sandboxed, sandboxed_seats = _play(pool, seed="inproc")
    seats = {}
    for s, name in zip(CONFIG.resolved_seats, BOTS):
        b, info = load_bot(_template_path(), "arena_restore_template") if name == "template_bot" else load_baseline(name)
        seats[s] = InProcessSeat(s, b, info, info_for(CONFIG, s, seed="inproc"), "0" * 64)
        seats[s].start()
    table = seeded_table(CONFIG, "inproc", table_id="t_test")
    play_table(table, seats)
    assert _log(table) == _log(sandboxed)
    for s in CONFIG.resolved_seats:
        assert seats[s].committed_history == sandboxed_seats[s].committed_history


def test_policy_rng_lives_in_state(pool):
    played = play_sandboxed(pool, ["legal_random", "calling_station", "calling_station"], budget_ns=BIG_BUDGET,
                            config=CONFIG)
    import json

    states = [json.loads(s) for s in played.seats[0].committed_history]
    assert states[0] == {"seed": "faults-0", "n": 0}
    assert [s["n"] for s in states] == sorted(s["n"] for s in states) and states[-1]["n"] > 0
