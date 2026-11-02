"""Deployed-path gate: the packaged bot through the real SDK sandbox worker.

The bundle from :mod:`strategy.policy.package` is staged and played by
:class:`arena.agents.worker.WorkerSeat` exactly as in a sandboxed tournament
(Landlock, seccomp, rlimits, the 1 ms deadline, explicit state round-trips through
the protocol). Only the candidate's seat is sandboxed; opponents (train-partition
lineups) run in-process so they do not compete with the worker for CPU, and tables run
one at a time: parallel sandboxed play on a small VM produces infrastructure timeouts
that say nothing about the bot.

Pass criteria (all must hold):

* at least ``min_decisions`` decisions (default 50,000);
* worker decision wall time p99 below ``p99_limit_us`` (default 500 us) - this is the
  bot's own time (state decode + decide) as the worker measures it;
* zero arena fallbacks of any kind (illegal action, timeout, crash, invalid output,
  state errors), zero worker restarts and zero late replies;
* zero internal fallbacks (``state["fb"]`` in the committed state).

The report also gives p50/p95/p99.9/max of worker wall time and end-to-end time
(including serialization and IPC), and the counts above 0.5 ms and 1 ms. The seat
records worker time only for replies that arrive in time, so every miss is listed
with the timing the worker reported in its late reply (``late_worker``), any overlap
with a GC pause in this harness process, and a cause (:func:`_cause`). Run the same
gate with a trivial bot (``--bundle arena/agents/baselines/calling_station.py``) to
measure the machine's own miss floor. GC inside the bot is attributed by
:mod:`strategy.policy.bench`, which replays the same decision path.

    python -m strategy.policy.gate --bundle artifacts/bots/baseline-candidate --decisions 50000
"""

from __future__ import annotations

import argparse
import gc
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


class _Probe:
    """A worker channel that keeps every reply's timing metadata, including late replies the seat discards."""

    def __init__(self, inner: Any, frames: dict[int, dict[str, Any]]) -> None:
        self.inner = inner
        self.frames = frames

    def send(self, frame: bytes) -> None:
        self.inner.send(frame)

    def recv(self, deadline_ns: int | None, spin_ns: int = 0) -> Any:
        f = self.inner.recv(deadline_ns, spin_ns)
        if f is not None and isinstance(f.meta.get("seq"), int):
            self.frames[f.meta["seq"]] = {k: f.meta.get(k) for k in ("wall_ns", "parse_ns", "cpu_ns")}
        return f

    def kill(self) -> None:
        self.inner.kill()

    def log_excerpt(self) -> str:
        return self.inner.log_excerpt()


def _cause(miss: dict[str, Any], wall: dict[str, float]) -> str:
    """``bot_time`` when the bot's own (late-reported) time alone was far above its p99.9, else ``transport``
    (parse, IPC or scheduling outside the bot); ``unknown`` when no late reply arrived."""
    late = miss.get("late_worker") or {}
    t = late.get("wall_ns")
    if t is None:
        return "unknown"
    return "bot_time" if wall and t > max(500.0, 2 * wall.get("p999_us", 0.0)) else "transport"


def run_gate(bundle: Path, *, ruleset_id: str = "practice_nlhe", min_decisions: int = 50_000,
             p99_limit_us: float = 500.0, seed: str = "deployed-gate-v1", partition: str = "train",
             max_tables: int = 2000, players: int = 8, hand_cap: int = 200, parent_gc: str = "default",
             log: Any = print) -> dict[str, Any]:
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
    pauses: list[tuple[int, int, int]] = []
    gc_start = [0]

    def on_gc(phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            gc_start[0] = time.perf_counter_ns()
        else:
            pauses.append((gc_start[0], time.perf_counter_ns(), int(info["generation"])))

    misses: list[dict[str, Any]] = []
    gc.callbacks.append(on_gc)
    with WorkerPool(size=2) as pool:
        artifact = pool.stage(bundle)
        batch = 0
        while decisions < min_decisions and tables < max_tables:
            tasks = make_tasks(source, partition, ruleset_id=ruleset_id, players=config.initial_player_count,
                               count=50, namespace=f"{seed}|{ruleset_id}|{batch}")
            batch += 1
            for task in tasks:
                frames: dict[int, dict[str, Any]] = {}
                agents, cs, table = seat_agents(
                    config, task, lambda seat, info, _f=frames: WorkerSeat(
                        seat, artifact, info, lambda: _Probe(pool.acquire(), _f), budget_ns=config.bot_budget_ns))
                seen = [0]

                def on_decision(seat: int, d: Any, _t: Any, _cs: int = cs, _seen: list[int] = seen) -> None:
                    if seat != _cs:
                        return
                    _seen[0] += 1
                    if d.fallback is not None:
                        now = time.perf_counter_ns()
                        lo = now - (d.e2e_ns or 0) - 50_000
                        hit = [(b - a) // 1000 for a, b, _ in pauses[-50:] if b >= lo and a <= now]
                        misses.append({"table": tables, "decision": _seen[0], "fallback": d.fallback,
                                       "seq": agents[_cs].seq,
                                       "e2e_us": (d.e2e_ns or 0) / 1000, "worker_wall_us": (d.worker_wall_ns or 0) / 1000
                                       if d.worker_wall_ns else None, "parent_gc_overlap_us": hit})

                if parent_gc == "freeze":
                    gc.collect()
                    gc.freeze()
                try:
                    play_table(table, agents, on_decision=on_decision)
                finally:
                    if parent_gc == "freeze":
                        gc.unfreeze()
                    cand = agents[cs]
                    st = cand.stats
                    cand.close()
                for m in misses:
                    if m["table"] == tables and "seq" in m:
                        f = frames.get(m.pop("seq")) or {}
                        m["late_worker"] = {k: (v / 1000 if isinstance(v, int) else None) for k, v in f.items()}
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
    gc.callbacks.remove(on_gc)
    w, e = quantiles(wall), quantiles(e2e)
    checks = {
        "decisions": decisions >= min_decisions,
        "p99": bool(w) and w["p99_us"] < p99_limit_us,
        "arena_fallbacks": sum(fallbacks.values()) == 0,
        "restarts": restarts == 0,
        "late_replies": late == 0,
        "internal_fallbacks": internal == 0,
    }
    mf = Path(bundle) / "manifest.json"
    return {"bundle": str(bundle), "manifest": json.loads(mf.read_text()) if mf.exists() else None,
            "ruleset_id": ruleset_id, "partition": partition, "seed": seed, "tables": tables,
            "decisions": decisions, "worker_wall": w, "e2e": e, "fallbacks": dict(fallbacks),
            "restarts": restarts, "late_replies": late, "internal_fallbacks": internal,
            "max_state_bytes": max_state, "p99_limit_us": p99_limit_us, "checks": checks,
            "parent_gc": parent_gc, "misses": misses,
            "misses_with_parent_gc": sum(1 for m in misses if m["parent_gc_overlap_us"]),
            "miss_causes": dict(Counter(_cause(m, w) for m in misses)),
            "parent_gc_pauses_over_500us": sum(1 for a, b, _ in pauses if b - a > 500_000),
            "passed": all(checks.values()), "wall_s": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.policy.gate")
    p.add_argument("--bundle", required=True)
    p.add_argument("--ruleset", default="practice_nlhe")
    p.add_argument("--decisions", type=int, default=50_000)
    p.add_argument("--p99-us", type=float, default=500.0)
    p.add_argument("--seed", default="deployed-gate-v1")
    p.add_argument("--parent-gc", default="default", choices=("default", "freeze"),
                   help="freeze the harness process's heap per table (isolates the bot from harness GC pauses)")
    p.add_argument("--out")
    args = p.parse_args(argv)
    rep = run_gate(Path(args.bundle), ruleset_id=args.ruleset, min_decisions=args.decisions,
                   p99_limit_us=args.p99_us, seed=args.seed, parent_gc=args.parent_gc, log=lambda s: print(s, file=sys.stderr, flush=True))
    rep.pop("manifest")
    text = json.dumps(rep, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
