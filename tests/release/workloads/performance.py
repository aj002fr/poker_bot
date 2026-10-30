"""Performance gate: baseline decisions in sandboxed workers under the 1 ms deadline.

Each shard plays eight-seat NLHE tables with the production path (pooled
sandboxed :class:`~arena.agents.worker.WorkerSeat` workers driven by
:func:`~arena.agents.worker.play_table`), two tables at a time (the default
``match`` layout on this VM), and records every decision as the parent saw it:

- ``e2e``: dispatch to complete reply, the only clock the deadline uses;
- ``parse``: the worker validating the observation (worker-reported);
- ``bot``: the bot's own ``decide()`` wall time (worker-reported);
- ``ipc``: ``e2e - parse - bot`` - serialization, the pipe hop and scheduling.

Worker-reported times are informational. The gate checks that no accepted
decision exceeded the budget (a late reply is always a ``timeout`` fallback,
never silently accepted) and that the baselines are *suitable* for the
competitive deadline on this machine, a threshold declared here and never
loosened to make a run pass: 1 ms fallback rate <= 1% and p99 e2e <= 0.5 ms.
The budget itself is the configured 1,000,000 ns and is not changed.
"""

from __future__ import annotations

import multiprocessing
import resource
import time
from array import array
from collections import Counter
from typing import Any

from arena.benchmark.runner import Assignment, policy_seed
from arena.domain.models import TournamentConfig
from arena.history.canonical import export_history
from arena.table.service import seeded_table
from tests.release.workloads.common import digest
from tests.release.workloads.isolated import LINEUPS

TABLES_PER_SHARD = 32
SHARDS = 4
JOBS = 2
REQUIRED_DECISIONS = 100_000
SUITABLE_FALLBACK_RATE = 0.01
SUITABLE_P99_NS = 500_000


def _play(args: tuple[str, list[int]]) -> dict[str, Any]:
    from arena.agents.baselines import baseline_path
    from arena.agents.sdk import TableInfo
    from arena.agents.worker import WorkerPool, WorkerSeat, play_table

    base, indices = args
    config = TournamentConfig(ruleset_id="practice_nlhe", initial_player_count=8)
    names = LINEUPS["practice_nlhe"]
    seats = config.resolved_seats
    cols = {k: array("q") for k in ("e2e", "parse", "bot", "cpu", "fallback", "bot_id")}
    reasons: Counter = Counter()
    counts: Counter = Counter()
    hashes = []
    distinct = sorted(set(names))
    t0 = time.perf_counter()
    with WorkerPool(size=len(seats) + 2) as pool:
        artifacts = {n: pool.stage(baseline_path(n)) for n in distinct}
        for index in indices:
            seed = f"{base}|{index}"
            a = Assignment(index, index, 0, seed, {s: i for i, s in enumerate(seats)})
            agents = {}
            for s in seats:
                info = TableInfo(table_id=a.table_id, hero_seat=s, ruleset_id=config.ruleset_id,
                                 config=config.resolved(), config_hash=config.config_hash(), seats=seats,
                                 decision_budget_ns=config.bot_budget_ns, policy_seed=policy_seed(seed, s))
                agent = WorkerSeat(s, artifacts[names[s % len(names)]], info, pool.acquire,
                                   budget_ns=config.bot_budget_ns)
                agent.start()
                agents[s] = agent
            table = seeded_table(config, seed, table_id=a.table_id)

            def record(seat: int, d, _t, seat_names=names) -> None:
                if d.fallback is None and d.e2e_ns is not None and d.e2e_ns > config.bot_budget_ns:
                    counts["accepted_over_budget"] += 1
                cols["e2e"].append(d.e2e_ns if d.e2e_ns is not None else -1)
                cols["parse"].append(d.worker_parse_ns if d.worker_parse_ns is not None else -1)
                cols["bot"].append(d.worker_wall_ns if d.worker_wall_ns is not None else -1)
                cols["cpu"].append(d.worker_cpu_ns if d.worker_cpu_ns is not None else -1)
                cols["fallback"].append(0 if d.fallback is None else 1)
                cols["bot_id"].append(distinct.index(seat_names[seat % len(seat_names)]))
                if d.fallback is not None:
                    reasons[d.fallback] += 1

            try:
                play_table(table, agents, on_decision=record)
            finally:
                for agent in agents.values():
                    agent.close()
            counts["tables"] += 1
            counts["tables_complete"] += int(table.complete)
            counts["hands"] += len(table.hands)
            counts["restarts"] += sum(len(ag.stats.restarts) for ag in agents.values())
            counts["late_replies_discarded"] += sum(ag.stats.late_discarded for ag in agents.values())
            hashes.append(digest(export_history(table).summary.model_dump(mode="json")))
    wall = time.perf_counter() - t0
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {"cols": {k: v.tobytes() for k, v in cols.items()}, "reasons": dict(reasons), "counts": dict(counts),
            "wall_s": wall, "parent_max_rss_kib": self_rss, "worker_max_rss_kib": child_rss, "hashes": hashes}


def _q(xs: list[int], p: float) -> int | None:
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def _stats(xs: list[int]) -> dict[str, Any]:
    xs = sorted(x for x in xs if x >= 0)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50_us": _q(xs, 0.5) / 1000, "p95_us": _q(xs, 0.95) / 1000,
            "p99_us": _q(xs, 0.99) / 1000, "max_us": xs[-1] / 1000, "mean_us": round(sum(xs) / len(xs) / 1000, 3)}


def _hist(xs: list[int]) -> dict[str, int]:
    """Sparse histogram in whole microseconds, so shards pool into exact 1 us quantiles."""
    h = Counter(x // 1000 for x in xs if x >= 0)
    return {str(k): v for k, v in sorted(h.items())}


def _pooled(hists: list[dict[str, int]]) -> dict[str, Any]:
    total: Counter = Counter()
    for h in hists:
        total.update({int(k): v for k, v in h.items()})
    n = sum(total.values())
    if not n:
        return {"n": 0}
    out: dict[str, Any] = {"n": n}
    keys = sorted(total)
    targets = {"p50_us": 0.5, "p95_us": 0.95, "p99_us": 0.99}
    seen = 0
    for k in keys:
        seen += total[k]
        for name, p in list(targets.items()):
            if seen > p * n - 1e-9 or seen == n:
                out[name] = k
                del targets[name]
    out["max_us"] = keys[-1]
    return out


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    indices = list(range(params["start"], params["start"] + params["count"]))
    parts = [(params["base_seed"], indices[j::JOBS]) for j in range(JOBS)]
    t0 = time.perf_counter()
    with multiprocessing.get_context("fork").Pool(JOBS) as mp:
        outs = mp.map(_play, parts)
    wall = time.perf_counter() - t0
    cols: dict[str, array] = {k: array("q") for k in outs[0]["cols"]}
    counts: Counter = Counter()
    reasons: Counter = Counter()
    for o in outs:
        for k, raw in o["cols"].items():
            cols[k].frombytes(raw)
        counts.update(o["counts"])
        reasons.update(o["reasons"])
    e2e, parse, bot, cpu, fb = (list(cols[k]) for k in ("e2e", "parse", "bot", "cpu", "fallback"))
    accepted = [i for i, f in enumerate(fb) if f == 0]
    ipc = [e2e[i] - parse[i] - bot[i] for i in accepted if parse[i] >= 0 and bot[i] >= 0]
    decisions = len(e2e)
    counts["decisions"] = decisions
    counts["fallbacks"] = sum(fb)
    counts["timeouts"] = reasons.get("timeout", 0)
    counts.setdefault("accepted_over_budget", 0)
    for r, k in reasons.items():
        counts[f"fallback:{r}"] = k
    counts["max_e2e_ns"] = max(e2e) if e2e else 0
    counts["max_parent_rss_kib"] = max(o["parent_max_rss_kib"] for o in outs)
    counts["max_worker_rss_kib"] = max(o["worker_max_rss_kib"] for o in outs)
    failures = []
    if counts["accepted_over_budget"]:
        failures.append({"case": {"gate": "performance", **params}, "error": f"{counts['accepted_over_budget']} "
                         "decisions accepted after the deadline", "traceback": "", "repro": ""})
    if counts["tables_complete"] != len(indices):
        failures.append({"case": {"gate": "performance", **params}, "error": "a table did not complete",
                         "traceback": "", "repro": ""})
    per_bot = {}
    names = sorted(set(LINEUPS["practice_nlhe"]))
    for b, name in enumerate(names):
        idx = [i for i, x in enumerate(cols["bot_id"]) if x == b]
        per_bot[name] = {"decisions": len(idx), "fallbacks": sum(fb[i] for i in idx),
                         "e2e": _stats([e2e[i] for i in idx]), "bot": _stats([bot[i] for i in idx])}
    return {
        "cases": len(indices),
        "failed_cases": len(failures),
        "counts": dict(sorted(counts.items())),
        "exclusions": {},
        "failures": failures,
        "result_hash": digest(sorted(h for o in outs for h in o["hashes"])),
        "timing": {
            "decisions": decisions, "wall_s": round(wall, 2), "jobs": JOBS, "budget_ns": 1_000_000,
            "throughput_decisions_per_s": round(decisions / wall, 1), "tables_per_s": round(len(indices) / wall, 3),
            "fallback_rate": counts["fallbacks"] / decisions if decisions else None,
            "e2e_all": _stats(e2e), "e2e_accepted": _stats([e2e[i] for i in accepted]),
            "worker_parse": _stats([parse[i] for i in accepted]), "bot_decide_wall": _stats([bot[i] for i in accepted]),
            "bot_decide_cpu": _stats([cpu[i] for i in accepted]), "ipc_and_scheduling": _stats(ipc),
            "per_bot": per_bot,
            "peak_rss_mib": {"parent_job": round(counts["max_parent_rss_kib"] / 1024, 1),
                             "largest_worker": round(counts["max_worker_rss_kib"] / 1024, 1)},
            "hist_us": {"e2e": _hist(e2e), "ipc": _hist(ipc), "bot": _hist([bot[i] for i in accepted]),
                        "parse": _hist([parse[i] for i in accepted])},
        },
    }


def shards() -> list[dict[str, Any]]:
    return [{"gate": "performance", "base_seed": "perf", "start": k * TABLES_PER_SHARD, "count": TABLES_PER_SHARD}
            for k in range(SHARDS)]


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict) -> list[dict[str, Any]]:
    decisions = sum(r["timing"]["decisions"] for r in results)
    fallbacks = sum(r["counts"]["fallbacks"] for r in results)
    over = sum(r["counts"].get("accepted_over_budget", 0) for r in results)
    rate = fallbacks / decisions if decisions else 1.0
    pooled = {k: _pooled([r["timing"]["hist_us"][k] for r in results]) for k in ("e2e", "parse", "bot", "ipc")}
    p99_ns = pooled["e2e"].get("p99_us", 1e9) * 1000
    wall = sum(r["timing"]["wall_s"] for r in results)
    rss = (max(r["counts"]["max_parent_rss_kib"] for r in results) / 1024,
           max(r["counts"]["max_worker_rss_kib"] for r in results) / 1024)
    return [
        {"check": "pooled end-to-end latency (1 us resolution, all decisions incl. fallbacks)", "passed": True,
         "detail": pooled["e2e"]},
        {"check": "pooled split for accepted decisions: worker parse / bot decide / IPC and scheduling",
         "passed": True, "detail": {"parse": pooled["parse"], "bot": pooled["bot"], "ipc": pooled["ipc"]}},
        {"check": "peak memory (max RSS)", "passed": True,
         "detail": f"job process {rss[0]:.1f} MiB, largest worker {rss[1]:.1f} MiB"},
        {"check": f"sandboxed baseline decisions >= {REQUIRED_DECISIONS:,}", "passed": decisions >= REQUIRED_DECISIONS,
         "detail": f"{decisions:,} decisions in {wall:,.0f} s ({decisions / wall:,.0f}/s, {JOBS} jobs)"},
        {"check": "no accepted decision exceeded the 1 ms budget", "passed": over == 0, "detail": f"{over} over budget"},
        {"check": f"baseline suitability: 1 ms fallback rate <= {SUITABLE_FALLBACK_RATE:.0%}",
         "passed": rate <= SUITABLE_FALLBACK_RATE, "detail": f"{fallbacks:,} of {decisions:,} ({rate:.4%})"},
        {"check": f"baseline suitability: pooled p99 end-to-end <= {SUITABLE_P99_NS / 1e6:g} ms",
         "passed": p99_ns <= SUITABLE_P99_NS, "detail": f"{p99_ns / 1000:.0f} us"},
    ]
