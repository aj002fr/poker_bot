"""Decision latency and GC pauses of a packaged bot, replayed the way the sandbox worker runs it.

1. **Capture**: play in-process tables (train-partition lineups) and record the
   candidate's observations as the JSON the worker would receive.
2. **Replay** in a clean subprocess that holds only the bundle, like a worker: load the
   bot, then per table ``initial_state``, ``gc.collect()``, ``gc.freeze()``; per decision
   parse the observation, decode the state, decide (the worker's ``wall_ns``), encode the
   state. ``gc.callbacks`` attributes every collection (generation, pause) to the stage it
   interrupted.

GC strategies compared (``--gc``):

* ``worker``: what the sandbox does (collect and freeze after load);
* ``nofreeze``: no freeze (the heap holding the tables is rescanned by every full
  collection - the source of the ~1.5 ms spikes seen in the feature bench);
* ``disabled``: as ``worker`` plus ``gc.disable()`` (safe only if the path makes no
  reference cycles; memory growth is reported).

    python -m strategy.policy.bench --bundle artifacts/bots/control --tables 40
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def capture(bundle: Path, out: Path, *, tables: int, ruleset_id: str = "practice_nlhe",
            seed: str = "policy-bench-v1") -> int:
    from arena.agents.sdk import bot_info, load_bot
    from arena.agents.worker import InProcessSeat, play_table
    from arena.domain.models import TournamentConfig
    from strategy.search.evaluate import _init, make_tasks, seat_agents
    from strategy.search.population import load_population

    config = TournamentConfig(ruleset_id=ruleset_id)
    _init(config.model_dump_json())
    tasks = make_tasks(load_population(), "train", ruleset_id=ruleset_id, players=config.initial_player_count,
                       count=tables, namespace=f"{seed}|{ruleset_id}")
    load_bot(bundle / "bot.py", module_name="bench_capture_bot")
    mod = sys.modules["bench_capture_bot"]
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for task in tasks:
            seen: list[str] = []
            inner = mod.make_bot()
            original = inner.decide

            def decide(obs: Any, state: Any, _o: Any = original, _s: list[str] = seen) -> Any:
                _s.append(obs.model_dump_json())
                return _o(obs, state)

            inner.decide = decide

            def candidate(seat: int, info: Any, _b: Any = inner) -> Any:
                candidate.info = info
                return InProcessSeat(seat, _b, bot_info(_b), info, "bench")

            agents, _, table = seat_agents(config, task, candidate)
            play_table(table, agents)
            f.write(json.dumps({"table": candidate.info.model_dump(mode="json"), "obs": seen}) + "\n")
            n += len(seen)
    return n


def replay(bundle: Path, capture_path: Path, mode: str, only: int | None = None) -> dict[str, Any]:
    """Runs in a fresh interpreter (see :func:`main`); returns timing and GC records."""
    import resource

    from arena.agents.sdk import TableInfo, decode_state, encode_state, load_bot
    from arena.domain.observations import Observation

    sys.path.insert(0, str(bundle))
    bot, _ = load_bot(bundle / "bot.py", module_name="arena_bot")
    stage = ["idle"]
    pauses: list[tuple[str, int, int]] = []
    started = [0]

    def on_gc(phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            started[0] = time.perf_counter_ns()
        else:
            pauses.append((stage[0], int(info["generation"]), time.perf_counter_ns() - started[0]))

    gc.callbacks.append(on_gc)
    rows = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines()]
    if only is not None:
        rows = [rows[only]]
    wall: list[int] = []
    parse: list[int] = []
    gc_in_decide: list[int] = []
    where: list[tuple[int, int]] = []
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for ti, row in enumerate(rows):
        table = TableInfo.model_validate(row["table"])
        blob = encode_state(bot.initial_state(table))
        if mode != "nofreeze":
            gc.collect()
            gc.freeze()
        if mode == "disabled":
            gc.disable()
        for di, raw in enumerate(row["obs"]):
            where.append((ti, di))
            raw_b = raw.encode()
            stage[0] = "parse"
            t0 = time.perf_counter_ns()
            obs = Observation.model_validate_json(raw_b)
            t1 = time.perf_counter_ns()
            stage[0] = "decide"
            k = len(pauses)
            _, new_state = bot.decide(obs, decode_state(blob))
            t2 = time.perf_counter_ns()
            stage[0] = "encode"
            blob = encode_state(new_state)
            stage[0] = "idle"
            wall.append(t2 - t1)
            parse.append(t1 - t0)
            gc_in_decide.append(sum(p[2] for p in pauses[k:] if p[0] == "decide"))
        if mode == "disabled":
            gc.enable()
    gc.callbacks.remove(on_gc)
    return {"mode": mode, "wall": wall, "parse": parse, "gc_in_decide": gc_in_decide, "pauses": pauses, "where": where,
            "rss_growth_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0}


def summarize(r: dict[str, Any]) -> dict[str, Any]:
    from strategy.policy.gate import quantiles

    wall, gcd = r["wall"], r["gc_in_decide"]
    by: dict[str, dict[str, Any]] = {}
    for st, gen, ns in r["pauses"]:
        b = by.setdefault(f"{st}/gen{gen}", {"count": 0, "max_us": 0.0, "total_us": 0.0})
        b["count"] += 1
        b["max_us"] = max(b["max_us"], ns / 1000)
        b["total_us"] += ns / 1000
    slow = [i for i, x in enumerate(wall) if x > 500_000]
    return {"mode": r["mode"], "decisions": len(wall), "decide_wall": quantiles(wall),
            "decide_wall_excluding_gc": quantiles([w - g for w, g in zip(wall, gcd)]),
            "parse": quantiles(r["parse"]), "gc": by,
            "max_gc_pause_us": max((ns for _, _, ns in r["pauses"]), default=0) / 1000,
            "slow_decisions": len(slow), "slow_with_gc": sum(1 for i in slow if gcd[i] > 0),
            "slowest": [[*r["where"][i], wall[i] / 1000] for i in sorted(range(len(wall)), key=wall.__getitem__)[-8:]],
            "rss_growth_kb": r["rss_growth_kb"]}


def _fresh(bundle: Path, cap: Path, tables: int) -> dict[str, Any]:
    """``worker`` mode with one new interpreter per table, as the sandbox runs it."""
    from strategy.policy.gate import quantiles

    wall: list[int] = []
    gcd: list[int] = []
    where: list[tuple[int, int]] = []
    for t in range(tables):
        res = subprocess.run([sys.executable, "-m", "strategy.policy.bench", "--bundle", str(bundle), "--replay",
                              str(bundle), str(cap), f"worker:{t}"], capture_output=True, text=True,
                             cwd=tempfile.gettempdir(), check=True)
        r = json.loads(res.stdout.strip().splitlines()[-1])
        wall += r["wall"]
        gcd += r["gc_in_decide"]
        where += [(t, i) for i in range(len(r["wall"]))]
    order = sorted(range(len(wall)), key=wall.__getitem__)[-8:]
    return {"mode": "fresh", "decisions": len(wall), "decide_wall": quantiles(wall),
            "slow_decisions": sum(1 for x in wall if x > 500_000),
            "slow_with_gc": sum(1 for x, g in zip(wall, gcd) if x > 500_000 and g > 0),
            "slowest": [[*where[i], wall[i] / 1000, gcd[i] / 1000] for i in order]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.policy.bench")
    p.add_argument("--bundle", required=True)
    p.add_argument("--tables", type=int, default=40)
    p.add_argument("--ruleset", default="practice_nlhe")
    p.add_argument("--gc", nargs="*", default=["worker", "nofreeze", "disabled"])
    p.add_argument("--out")
    p.add_argument("--replay", nargs=3, metavar=("BUNDLE", "CAPTURE", "MODE"), help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.replay:
        mode, _, only = args.replay[2].partition(":")
        r = replay(Path(args.replay[0]), Path(args.replay[1]), mode, int(only) if only else None)
        print(json.dumps(summarize(r) if not only else {"wall": r["wall"], "gc_in_decide": r["gc_in_decide"]}))
        return 0
    bundle = Path(args.bundle).resolve()
    out: dict[str, Any] = {"bundle": str(bundle), "ruleset_id": args.ruleset, "tables": args.tables, "modes": {}}
    with tempfile.TemporaryDirectory() as tmp:
        cap = Path(tmp) / "capture.jsonl"
        out["captured_decisions"] = capture(bundle, cap, tables=args.tables, ruleset_id=args.ruleset)
        for mode in args.gc:
            if mode == "fresh":
                out["modes"][mode] = _fresh(bundle, cap, out["tables"])
                continue
            res = subprocess.run([sys.executable, "-m", "strategy.policy.bench", "--bundle", str(bundle),
                                  "--replay", str(bundle), str(cap), mode], capture_output=True, text=True,
                                 cwd=tempfile.gettempdir(), check=True)
            out["modes"][mode] = json.loads(res.stdout.strip().splitlines()[-1])
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
