"""Deployed-path gate: the packaged bot through the real SDK sandbox worker.

The bundle from :mod:`strategy.policy.package` is staged and played by
:class:`arena.agents.worker.WorkerSeat` exactly as in a sandboxed tournament
(Landlock, seccomp, rlimits, the 1 ms deadline, explicit state round-trips through
the protocol). Only the candidate's seat is sandboxed; opponents (train-partition
lineups) run in-process so they do not compete with the worker for CPU, and tables run
one at a time by default (``--jobs 1``): parallel sandboxed play on a small VM
produces infrastructure timeouts that say nothing about the bot.

Pass criteria (all must hold):

* at least ``min_decisions`` decisions (default 50,000);
* worker decision wall time p99 below ``p99_limit_us`` (default 500 us) - this is the
  bot's own time (state decode + decide) as the worker measures it;
* zero arena fallbacks of any kind (illegal action, timeout, crash, invalid output,
  state errors), zero worker restarts and zero late replies;
* zero internal fallbacks (``state["fb"]`` in the committed state).

The report also gives p50/p95/p99.9/max of worker wall time and end-to-end time
(including serialization and IPC), and the counts above 0.5 ms and 1 ms. GC pauses
are attributed by :mod:`strategy.policy.bench`, which replays the same decision path.

    python -m strategy.policy.gate --bundle artifacts/bots/baseline-final --decisions 50000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from strategy.search.evaluate import _init, make_tasks, seat_agents
from strategy.search.population import load_population


def quantiles(xs: list[int]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> float:
        return s[min(n - 1, int(p * n))] / 1000.0

    return {"n": n, "p50_us": q(0.5), "p95_us": q(0.95), "p99_us": q(0.99), "p999_us": q(0.999),
            "max_us": s[-1] / 1000.0, "over_500us": sum(1 for x in s if x > 500_000),
            "over_1ms": sum(1 for x in s if x > 1_000_000)}


def run_gate(bundle: Path, *, ruleset_id: str = "practice_nlhe", min_decisions: int = 50_000,
             p99_limit_us: float = 500.0, seed: str = "deployed-gate-v1", partition: str = "train",
             max_tables: int = 2000, players: int = 8, hand_cap: int = 200, log: Any = print) -> dict[str, Any]:
    from arena.agents.sdk import decode_state
    from arena.agents.worker import WorkerPool, WorkerSeat, play_table
    from arena.domain.models import TournamentConfig

    config = TournamentConfig(ruleset_id=ruleset_id, initial_player_count=players, hand_cap=hand_cap)
    _init(config.model_dump_json())
    source = load_population()
    wall: list[int] = []
    e2e: list[int] = []
    fallbacks: Counter[str] = Counter()
    restarts = late = internal = tables = decisions = 0
    max_state = 0
    t0 = time.time()
    with WorkerPool(size=2) as pool:
        artifact = pool.stage(bundle)
        batch = 0
        while decisions < min_decisions and tables < max_tables:
            tasks = make_tasks(source, partition, ruleset_id=ruleset_id, players=config.initial_player_count,
                               count=50, namespace=f"{seed}|{ruleset_id}|{batch}")
            batch += 1
            for task in tasks:
                agents, cs, table = seat_agents(
                    config, task, lambda seat, info: WorkerSeat(seat, artifact, info, pool.acquire,
                                                                budget_ns=config.bot_budget_ns))
                try:
                    play_table(table, agents)
                finally:
                    cand = agents[cs]
                    st = cand.stats
                    cand.close()
                wall += st.worker_wall_ns
                e2e += st.e2e_ns
                fallbacks.update(st.fallbacks)
                restarts += len(st.restarts)
                late += st.late_discarded
                max_state = max(max_state, st.max_state_bytes)
                state = decode_state(cand.committed)
                internal += int(state.get("fb", 0)) if isinstance(state, dict) else 0
                decisions += st.decisions
                tables += 1
                if tables % 25 == 0:
                    log(f"{tables} tables, {decisions} decisions, fallbacks {sum(fallbacks.values())}, "
                        f"internal {internal}, {time.time() - t0:.0f}s")
                if decisions >= min_decisions:
                    break
    w, e = quantiles(wall), quantiles(e2e)
    checks = {
        "decisions": decisions >= min_decisions,
        "p99": bool(w) and w["p99_us"] < p99_limit_us,
        "arena_fallbacks": sum(fallbacks.values()) == 0,
        "restarts": restarts == 0,
        "late_replies": late == 0,
        "internal_fallbacks": internal == 0,
    }
    return {"bundle": str(bundle), "manifest": json.loads((Path(bundle) / "manifest.json").read_text()),
            "ruleset_id": ruleset_id, "partition": partition, "seed": seed, "tables": tables,
            "decisions": decisions, "worker_wall": w, "e2e": e, "fallbacks": dict(fallbacks),
            "restarts": restarts, "late_replies": late, "internal_fallbacks": internal,
            "max_state_bytes": max_state, "p99_limit_us": p99_limit_us, "checks": checks,
            "passed": all(checks.values()), "wall_s": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.policy.gate")
    p.add_argument("--bundle", required=True)
    p.add_argument("--ruleset", default="practice_nlhe")
    p.add_argument("--decisions", type=int, default=50_000)
    p.add_argument("--p99-us", type=float, default=500.0)
    p.add_argument("--seed", default="deployed-gate-v1")
    p.add_argument("--out")
    args = p.parse_args(argv)
    rep = run_gate(Path(args.bundle), ruleset_id=args.ruleset, min_decisions=args.decisions,
                   p99_limit_us=args.p99_us, seed=args.seed, log=lambda s: print(s, file=sys.stderr, flush=True))
    rep.pop("manifest")
    text = json.dumps(rep, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
