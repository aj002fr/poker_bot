"""Isolated-worker gate: eight-bot tournaments through real sandboxed workers.

Each shard is one :func:`~arena.benchmark.runner.run_benchmark` call in
``sandbox`` isolation under the configured 1 ms deadline (two tables in
parallel), with histories saved. The scheduler's paired-seed rotations give
eight tables per seed block. Afterwards every table must:

- be valid (completed, no error, no infrastructure invalidation);
- replay from its saved history with every decision and hand-start hash
  verified (:func:`arena.history.replay.replay`);
- replay a second time through the oracle-checked table, so every hand is
  re-settled by the independent pot ledger and evaluators, the markers and
  blinds are recomputed, and placements and points match the ``table_points``
  oracle;
- report the same points as the benchmark result.

Agent fallbacks (a baseline missing the 1 ms deadline) are legal outcomes and
are counted, not hidden; the performance gate measures them in detail.
"""

from __future__ import annotations

import tempfile
from array import array
from collections import Counter
from pathlib import Path
from typing import Any

from arena.agents.baselines import baseline_path
from arena.benchmark.runner import Entrant, run_benchmark
from arena.domain.models import TournamentConfig
from arena.history.canonical import load_history
from arena.history.replay import commands_from_events, replay
from tests.release.workloads.common import CaseRunner, digest
from tests.release.workloads.tournaments import (
    CheckedTable,
    check_positions_and_blinds,
    check_termination_and_points,
)

BLOCKS_PER_SHARD = 4
PLAN = {"practice_nlhe": 40, "uncertainty_shortdeck": 12, "uncertainty_plo4": 12}
REQUIRED_TABLES = 500
JOBS = 2

LINEUPS = {
    "practice_nlhe": ["nlhe_heuristic", "legal_random", "calling_station", "check_fold"] * 2,
    "uncertainty_shortdeck": ["legal_random", "calling_station", "check_fold", "legal_random",
                              "calling_station", "check_fold", "legal_random", "calling_station"],
    "uncertainty_plo4": ["legal_random", "calling_station", "check_fold", "legal_random",
                         "calling_station", "check_fold", "legal_random", "calling_station"],
}


def entrants(ruleset: str) -> list[Entrant]:
    return [Entrant(f"{name}#{k}", baseline_path(name), True) for k, name in enumerate(LINEUPS[ruleset])]


def oracle_replay(history) -> CheckedTable:
    """Rebuild the table from its history through :class:`CheckedTable` (every hand oracle-checked)."""
    started = history.table_started.data
    config = TournamentConfig.model_validate(started["config"], strict=True)
    hand_starts = [e for e in history.events if e.type == "HandStarted"]
    decks = {e.data["hand_number"]: tuple(e.data["deck"]) for e in hand_starts}
    table = CheckedTable(config, table_id=history.table_id, deck_source=lambda n, _p: decks[n],
                         initial_button=hand_starts[0].data["positions"]["button"],
                         start_stacks=started["start_stacks"], rng_record=started["rng"])
    for command in commands_from_events(history.events):
        assert not table.complete and table.actor_seat == command.seat, "command out of turn on oracle replay"
        if command.fallback_reason is not None:
            receipt = table.apply_fallback(command.seat, command.fallback_reason, command.rejected)
            assert receipt.action == command.action, "fallback resolved differently on oracle replay"
        else:
            table.submit(command.seat, command.action)
    return table


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    ruleset = params["ruleset"]
    config = TournamentConfig(ruleset_id=ruleset, initial_player_count=8)
    runner = CaseRunner("isolated_tournaments")
    counts = runner.counts
    fallbacks: Counter = Counter()
    e2e: list[int] = []
    with tempfile.TemporaryDirectory(prefix="arena-isolated-") as tmp:
        run = run_benchmark(config=config, entrants=entrants(ruleset), blocks=params["blocks"],
                            base_seed=params["base_seed"], isolation="sandbox", jobs=JOBS, out_dir=Path(tmp),
                            save_histories=True)
        manifest = run.manifest
        counts["attempts"] += len(run.attempts)
        counts["reruns"] += len(run.attempts) - len(run.results)
        for r in run.results:
            case = {"ruleset": ruleset, "base_seed": params["base_seed"], "blocks": params["blocks"],
                    "table_index": r.index, "table_id": r.table_id}

            def check(_case: dict[str, Any], r=r) -> str:
                assert r.valid, f"table invalid: {r.error or r.invalid_reason}"
                history = load_history(Path(tmp) / "tables" / f"{r.table_id}.json")
                report = replay(history)
                assert report.complete, "replayed table incomplete"
                table = oracle_replay(history)
                check_positions_and_blinds(table)
                outcome = check_termination_and_points(table)
                points = {r.seats[s]: p for s, p in table.points().items()}
                assert points == r.points, f"benchmark points {r.points} != replayed {points}"
                assert report.points == table.points()
                counts["tables"] += 1
                counts["hands"] += len(table.hands)
                counts["replays_verified"] += 1
                counts["decision_hashes_verified"] += report.decisions_verified
                counts.update(table.hand_counts)
                counts.update(outcome)
                return digest([r.table_id, history.summary.model_dump(mode="json")])

            runner.run(case, check)
            counts["decisions"] += sum(r.decisions.values())
            counts["restarts"] += sum(len(v) for v in r.restarts.values())
            counts["late_replies_discarded"] += sum(r.late_discarded.values())
            for per in r.fallbacks.values():
                fallbacks.update(per)
            for raw in r.e2e_ns.values():
                e2e.extend(array("q", raw))
    counts["fallbacks"] = sum(fallbacks.values())
    for reason, k in fallbacks.items():
        counts[f"fallback:{reason}"] = k
    e2e.sort()
    q = (lambda p: e2e[min(len(e2e) - 1, int(p * len(e2e)))] / 1000 if e2e else None)
    return runner.result(
        base_seed=params["base_seed"],
        summary={"ruleset": ruleset, "tables": len(run.results), "seed_list_sha256": manifest.seed_list_sha256,
                 "manifest_content_hash": manifest.content_hash(), "competitive": manifest.competitive,
                 "budget_ns": manifest.budget_ns, "jobs": JOBS,
                 "e2e_us": {"p50": q(0.50), "p99": q(0.99), "max": e2e[-1] / 1000 if e2e else None}},
    )


def run_case(case: dict[str, Any]) -> str:
    shard = {"ruleset": case["ruleset"], "blocks": case["blocks"], "base_seed": case["base_seed"]}
    result = run_shard(shard)
    if result["failed_cases"]:
        raise AssertionError(result["failures"][0]["error"])
    return result["result_hash"]


def shards() -> list[dict[str, Any]]:
    out = []
    for ruleset, blocks in PLAN.items():
        for k in range(blocks // BLOCKS_PER_SHARD):
            out.append({"gate": "isolated_tournaments", "ruleset": ruleset, "blocks": BLOCKS_PER_SHARD,
                        "base_seed": f"iso|{ruleset}|{k}"})
    return out


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict) -> list[dict[str, Any]]:
    totals: Counter = Counter()
    for r in results:
        totals.update(r["counts"])
    decisions = totals["decisions"] or 1
    return [
        {"check": f"sandboxed eight-bot tournaments >= {REQUIRED_TABLES}", "passed": totals["tables"] >= REQUIRED_TABLES,
         "detail": f"{totals['tables']:,} valid tables, {totals['hands']:,} hands, {totals['decisions']:,} decisions"},
        {"check": "every table valid, replayed, hash-verified and oracle re-settled",
         "passed": totals["replays_verified"] == totals["tables"] == sum(r["cases"] for r in results),
         "detail": f"{totals['replays_verified']:,} of {sum(r['cases'] for r in results):,}"},
        {"check": "competitive 1 ms deadline in force (sandbox, configured budget)",
         "passed": all(r["summary"]["competitive"] and r["summary"]["budget_ns"] == 1_000_000 for r in results),
         "detail": "all shards" if all(r["summary"]["competitive"] for r in results) else "some shard not competitive"},
        {"check": "fallbacks reported (informational; the performance gate bounds the rate)", "passed": True,
         "detail": f"{totals['fallbacks']:,} of {totals['decisions']:,} decisions "
                   f"({totals['fallbacks'] / decisions:.4%}); infrastructure reruns {totals['reruns']}"},
    ]
