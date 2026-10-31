"""Tournament gate: complete tables played by the SDK baselines in-process.

Each tournament seats a seeded mix of the built-in baseline bots
(``arena/agents/baselines``) behind :class:`~arena.agents.worker.InProcessSeat`
and drives the table with :func:`~arena.agents.worker.play_table`: the same
observation, submit and commit path the sandboxed benchmark uses, without
worker processes. Per tournament:

- invariants after every accepted transition, and every finished hand
  re-settled by the independent pot ledger and evaluators;
- blind level and posted amounts for every hand, recomputed from the config;
- button, small-blind and big-blind markers recomputed by an independent
  implementation of the section 4.1 dead-button rules;
- termination by a sole survivor or exactly at the hand cap;
- placements and points against the independent ``table_points`` oracle
  (ICM by permutation), 1e-9 absolute tolerance;
- the exported history replayed and hash-verified event for event;
- no fallback: an in-process baseline never needs one, so any fallback is an
  unexplained invalid table.

A tenth of the tables each use the slow and fast blind schedules, winner-take-all
and equal-top-two payouts, and ``short_bet_reopens_checker=true`` (spec 3.3).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from arena.agents.baselines import baseline_info, baseline_path, load_baseline
from arena.agents.worker import InProcessSeat, artifact_hash, play_table
from arena.benchmark.runner import Assignment, table_info
from arena.domain.models import LocalRules, TournamentConfig
from arena.domain.rng import SeededRng
from arena.history.canonical import TableHistory, export_history
from arena.history.replay import replay
from arena.table.service import Table
from arena.tournament.scoring import SCORE_TOLERANCE
from tests.oracles.icm import Bust, table_points
from tests.property.generated import RULESETS, check_invariants
from tests.release.workloads.common import CaseRunner, digest
from tests.support.oracle_bridge import assert_matches_oracles

SHARD_TABLES = 250
REQUIRED = {("practice_nlhe", 8): 10_000, **{("practice_nlhe", n): 2_000 for n in range(2, 8)},
            **{(r, n): 1_000 for r in RULESETS[1:] for n in range(2, 9)}}
REQUIRED_TOTAL = 36_000

SLOW_BLINDS = [[10, 20], [15, 30], [20, 40], [30, 60], [40, 80], [60, 120], [80, 160], [100, 200]]
FAST_BLINDS = [[10, 20], [20, 40], [40, 80], [80, 160], [160, 320], [320, 640], [640, 1280], [1280, 2560]]
VARIANTS = ("fast_blinds", "slow_blinds", "winner_take_all", "equal_top_two", "short_bet_reopens_checker",
            "default", "default", "default", "default", "default")


def baselines_for(ruleset: str) -> list[str]:
    return [n for n in ("check_fold", "calling_station", "legal_random", "nlhe_heuristic")
            if ruleset in baseline_info(n).rulesets]


def tournament_config(ruleset: str, n: int, variant: str) -> TournamentConfig:
    kw: dict[str, Any] = {}
    if variant == "fast_blinds":
        kw["blind_levels"] = FAST_BLINDS
    elif variant == "slow_blinds":
        kw["blind_levels"] = SLOW_BLINDS
    elif variant == "winner_take_all":
        kw["payout_points"] = [100] + [0] * (n - 1)
    elif variant == "equal_top_two":
        kw["payout_points"] = [50, 50] + [0] * (n - 2)
    elif variant == "short_bet_reopens_checker":
        kw["local_rules"] = LocalRules(short_bet_reopens_checker=True)
    return TournamentConfig(ruleset_id=ruleset, initial_player_count=n, **kw)


class CheckedTable(Table):
    """The authoritative table, with every finished hand checked against the oracles before it is recorded."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.hand_counts: Counter = Counter()
        super().__init__(*args, **kwargs)

    def _record_result(self) -> None:
        hand = self.hand
        assert hand is not None and hand.is_complete
        check_invariants(hand)
        assert_matches_oracles(hand)
        result = hand.result()
        self.hand_counts["hands_oracle_checked"] += 1
        self.hand_counts["showdowns" if not result.ended_by_fold else "fold_wins"] += 1
        self.hand_counts["hands_with_side_pots"] += len(result.pots) > 1
        self.hand_counts["auto_runouts"] += any(e.type == "ActionApplied" and "auto" in e.data for e in hand.events)
        super()._record_result()


# ----------------------------------------------------- independent checks


def _first_live_after(seat: int, ring: list[int], live: set[int]) -> int:
    i = ring.index(seat)
    for s in ring[i + 1:] + ring[: i + 1]:
        if s in live:
            return s
    raise AssertionError("no live seat")


def check_positions_and_blinds(table: Table) -> None:
    ring = list(table.seat_ring)
    cfg = table.config
    prev = None
    for k, rec in enumerate(table.hands):
        assert rec.hand_number == k + 1 and rec.hand_id == f"h_{k + 1:03d}"
        level = (rec.hand_number - 1) // cfg.hands_per_level
        assert rec.level == level, f"hand {rec.hand_number}: level {rec.level} != {level}"
        sb, bb = cfg.blind_levels[level]
        assert rec.blinds == (sb, bb), f"hand {rec.hand_number}: blinds {rec.blinds} != {(sb, bb)}"
        live = {s for s, v in rec.start_stacks.items() if v > 0}
        assert live == set(rec.start_stacks)
        pos = rec.positions
        if prev is None:
            assert pos.button in live
            if len(live) == 2:
                want = (pos.button, pos.button, _first_live_after(pos.button, ring, live))
            else:
                first = _first_live_after(pos.button, ring, live)
                want = (pos.button, first, _first_live_after(first, ring, live))
        else:
            nbb = _first_live_after(prev.big_blind, ring, live)
            if len(live) == 2:
                other = next(s for s in live if s != nbb)
                want = (other, other, nbb)
            else:
                want = (prev.small_blind, prev.big_blind, nbb)
                between = ring[ring.index(want[1]) + 1:] + ring[: ring.index(want[1]) + 1]
                for s in between:
                    if s == nbb:
                        break
                    assert s not in live, f"hand {rec.hand_number}: live seat {s} skipped by the blinds"
        got = (pos.button, pos.small_blind, pos.big_blind)
        assert got == want, f"hand {rec.hand_number}: markers {got}, expected {want}"
        assert pos.heads_up == (len(live) == 2)
        posted = {e.data["seat"]: e.data["amount"] for e in rec.events if e.type == "BlindPosted"}
        expected = {pos.big_blind: min(bb, rec.start_stacks[pos.big_blind])}
        if pos.small_blind in live:
            expected[pos.small_blind] = min(sb, rec.start_stacks[pos.small_blind])
        assert posted == expected, f"hand {rec.hand_number}: posted {posted}, expected {expected}"
        prev = pos


def check_termination_and_points(table: Table) -> dict[str, int]:
    cfg = table.config
    n = len(table.seat_ring)
    assert table.complete
    assert sum(table.stacks.values()) == n * cfg.starting_stack, "table chips not conserved"
    assert len(table.hands) <= cfg.hand_cap
    survivors = {s: v for s, v in table.stacks.items() if v > 0}
    methods = {p.seat: p.method for p in table.standings.placements.values()}
    assert sorted(methods) == sorted(table.seat_ring), "every initial player gets exactly one placement"
    if len(survivors) > 1:
        assert len(table.hands) == cfg.hand_cap, "table with several survivors ended before the cap"
        assert all(methods[s] == "icm" for s in survivors)
    else:
        assert len(survivors) == 1 and methods[next(iter(survivors))] == "winner"
    busts = []
    for rec in table.hands:
        if rec.eliminations:
            for p in rec.eliminations:
                assert rec.start_stacks[p.seat] > 0 and rec.result.final_stacks[p.seat] == 0
            busts.append([Bust(p.seat, rec.start_stacks[p.seat]) for p in rec.eliminations])
    points = table.points()
    assert abs(sum(points.values()) - sum(cfg.resolved_payouts)) <= SCORE_TOLERANCE
    oracle = table_points(cfg.resolved_payouts, bust_hands=busts, survivors=survivors, exact=False)
    for seat, value in oracle.items():
        assert abs(points[seat] - float(value)) <= SCORE_TOLERANCE, f"seat {seat}: {points[seat]} != oracle {value}"
    return {
        "cap_finishes": int(len(survivors) > 1),
        "outright_wins": int(len(survivors) == 1),
        "eliminations": sum(len(b) for b in busts),
        "simultaneous_bust_hands": sum(1 for b in busts if len(b) > 1),
        "shared_position_placements": sum(1 for p in table.standings.placements.values() if len(p.positions) > 1),
    }


# ------------------------------------------------------------------ cases


def case_seed(case: dict[str, Any]) -> str:
    return f"tg|{case['ruleset']}|{case['players']}|{case['index']}"


def play_case(case: dict[str, Any], counts: Counter) -> str:
    ruleset, n, index = case["ruleset"], case["players"], case["index"]
    variant = VARIANTS[index % len(VARIANTS)]
    config = tournament_config(ruleset, n, variant)
    seed = case_seed(case)
    names = baselines_for(ruleset)
    mix = random.Random(f"{seed}|mix")
    seats = config.resolved_seats
    lineup = {s: mix.choice(names) for s in seats}
    assignment = Assignment(index, 0, 0, seed, {s: i for i, s in enumerate(seats)})
    agents = {}
    for s in seats:
        bot, info = load_baseline(lineup[s])
        agent = InProcessSeat(s, bot, info, table_info(config, assignment, s, config.bot_budget_ns),
                              artifact_hash(baseline_path(lineup[s])))
        agent.start()
        agents[s] = agent
    table = CheckedTable(config, env_rng=SeededRng(seed, "env"), table_id=assignment.table_id)

    def after_transition(seat, decision, t) -> None:
        if decision.fallback is not None:
            raise AssertionError(f"in-process baseline {lineup[seat]} fell back: {decision.fallback} "
                                 f"{decision.rejected}")
        if t.hand is not None:
            check_invariants(t.hand)

    play_table(table, agents, on_decision=after_transition)
    check_positions_and_blinds(table)
    outcome = check_termination_and_points(table)
    text = export_history(table).dumps()
    report = replay(TableHistory.loads(text))
    assert report.complete and report.final_stacks == table.stacks and report.points == table.points()
    counts["tournaments"] += 1
    counts[f"variant:{variant}"] += 1
    counts["hands"] += len(table.hands)
    counts["decisions"] += sum(a.stats.decisions for a in agents.values())
    counts["replays_verified"] += 1
    counts["decision_hashes_verified"] += report.decisions_verified
    counts.update(table.hand_counts)
    counts.update(outcome)
    for name in set(lineup.values()):
        counts[f"seats:{name}"] += sum(1 for v in lineup.values() if v == name)
    return digest(text)


def run_case(case: dict[str, Any]) -> str:
    return play_case(case, Counter())


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    runner = CaseRunner("tournaments")
    for index in range(params["start"], params["start"] + params["count"]):
        case = {"ruleset": params["ruleset"], "players": params["players"], "index": index}
        runner.run(case, lambda c: play_case(c, runner.counts))
    return runner.result(index_range=[params["start"], params["start"] + params["count"] - 1],
                         seed_format="tg|<ruleset>|<players>|<index> (env stream: decks and first button)")


def shards() -> list[dict[str, Any]]:
    out = []
    for (ruleset, n), need in REQUIRED.items():
        for k in range(need // SHARD_TABLES):
            out.append({"gate": "tournaments", "ruleset": ruleset, "players": n,
                        "start": k * SHARD_TABLES, "count": SHARD_TABLES})
    return out


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict) -> list[dict[str, Any]]:
    per_cell: Counter = Counter()
    totals: Counter = Counter()
    for r in results:
        per_cell[(r["params"]["ruleset"], r["params"]["players"])] += r["counts"].get("tournaments", 0)
        totals.update(r["counts"])
    checks = []
    short = {f"{r} N={n}": f"{per_cell[(r, n)]:,} of {need:,}" for (r, n), need in REQUIRED.items()
             if per_cell[(r, n)] < need}
    checks.append({"check": "per-scenario minimums (NLHE 10,000 at N=8 and 2,000 at N=2..7; others 1,000 per N)",
                   "passed": not short, "detail": short or "all met"})
    checks.append({"check": f"complete tournaments >= {REQUIRED_TOTAL:,}", "passed": totals["tournaments"] >= REQUIRED_TOTAL,
                   "detail": f"{totals['tournaments']:,}"})
    checks.append({"check": "every tournament replayed and hash-verified",
                   "passed": totals["replays_verified"] == totals["tournaments"],
                   "detail": f"{totals['replays_verified']:,} replays, {totals['decision_hashes_verified']:,} decision hashes"})
    checks.append({"check": "hand-cap (ICM) and outright finishes both exercised",
                   "passed": totals["cap_finishes"] > 0 and totals["outright_wins"] > 0,
                   "detail": f"{totals['cap_finishes']:,} cap finishes, {totals['outright_wins']:,} outright"})
    return checks
