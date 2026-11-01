"""Microbenchmark: full feature and belief update per decision, in-process, per variant.

Plays seeded in-process tables of the variant's bots, records every seat's
observations, then replays each seat's observations through a fresh
:class:`~strategy.features.engine.FeatureEngine` and times each ``update`` call
(events since the last decision, belief updates, state, card, equity and range
features). Engine construction (table loading) is timed separately.

    python -m strategy.features.bench                     # all three variants
    python -m strategy.features.bench --ruleset practice_nlhe --tables 4 --json
    python -m strategy.features.bench --check             # exit 1 if a mean exceeds the target
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from typing import Any

TARGET_US = 200.0
RULESETS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")
LINEUPS = {
    "practice_nlhe": ("rock", "aggressive", "station", "nlhe_heuristic", "super_aggressive", "weak_tight",
                      "switcher", "legal_random"),
    "uncertainty_shortdeck": ("legal_random", "calling_station", "legal_random", "check_fold", "legal_random",
                              "calling_station", "legal_random", "legal_random"),
    "uncertainty_plo4": ("legal_random", "calling_station", "legal_random", "check_fold", "legal_random",
                         "calling_station", "legal_random", "legal_random"),
}


def record(ruleset_id: str, *, tables: int = 2, hand_cap: int = 120, seed: int = 7) -> list[tuple[int, list, list]]:
    """``(seat, seats, observations)`` for every seat of every table."""
    from arena.agents.baselines import BASELINES
    from arena.agents.sdk import TableInfo, load_bot
    from arena.agents.styles import STYLES
    from arena.agents.worker import InProcessSeat, play_table
    from arena.benchmark.runner import policy_seed
    from arena.domain.models import TournamentConfig
    from arena.table.service import seeded_table

    class Recording(InProcessSeat):
        def __init__(self, *a: Any) -> None:
            super().__init__(*a)
            self.observations: list = []

        def decide(self, observation: Any) -> Any:
            self.observations.append(observation)
            return super().decide(observation)

    out = []
    for t in range(tables):
        config = TournamentConfig(ruleset_id=ruleset_id, initial_player_count=8, hand_cap=hand_cap)
        table_seed = f"bench-{seed}-{t}"
        lineup = LINEUPS[ruleset_id]
        seats = config.resolved_seats
        agents = {}
        for i, s in enumerate(seats):
            name = lineup[(i + t) % len(lineup)]
            bot, info = load_bot(STYLES.get(name) or BASELINES[name], module_name=f"strategy_bench_{name}")
            ti = TableInfo(table_id=f"t_bench_{t}", hero_seat=s, ruleset_id=ruleset_id, config=config.resolved(),
                           config_hash=config.config_hash(), seats=seats, decision_budget_ns=config.bot_budget_ns,
                           policy_seed=policy_seed(table_seed, s))
            agent = Recording(s, bot, info, ti, "bench")
            agent.start()
            agents[s] = agent
        play_table(seeded_table(config, table_seed, table_id=f"t_bench_{t}"), agents)
        out.extend((s, list(seats), a.observations) for s, a in agents.items())
    return out


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


def run(ruleset_id: str, *, tables: int = 2, hand_cap: int = 120, seed: int = 7, repeat: int = 3) -> dict[str, Any]:
    from strategy.features.engine import FeatureEngine

    recorded = record(ruleset_id, tables=tables, hand_cap=hand_cap, seed=seed)
    times: list[float] = []
    by_street: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
    build: list[float] = []
    state_bytes = 0
    sources: dict[str, int] = {}
    for rep in range(repeat):
        for seat, seats, observations in recorded:
            t0 = time.perf_counter_ns()
            eng = FeatureEngine(ruleset_id, seat, seats)
            build.append((time.perf_counter_ns() - t0) / 1000)
            gc.collect()
            for obs in observations:
                t0 = time.perf_counter_ns()
                f = eng.update(obs)
                dt = (time.perf_counter_ns() - t0) / 1000
                if rep == repeat - 1:
                    times.append(dt)
                    by_street[f.state.street].append(dt)
                    sources[f.cards.equity_source] = sources.get(f.cards.equity_source, 0) + 1
            if rep == repeat - 1:
                state_bytes = max(state_bytes, len(json.dumps(eng.to_state(), separators=(",", ":"))))
    times.sort()
    summary = {
        "ruleset_id": ruleset_id,
        "decisions": len(times),
        "mean_us": round(statistics.fmean(times), 1) if times else 0.0,
        "p50_us": round(_pct(times, 0.5), 1),
        "p95_us": round(_pct(times, 0.95), 1),
        "p99_us": round(_pct(times, 0.99), 1),
        "max_us": round(times[-1], 1) if times else 0.0,
        "by_street_mean_us": {("preflop", "flop", "turn", "river")[k]: round(statistics.fmean(v), 1)
                              for k, v in by_street.items() if v},
        "engine_build_ms": round(statistics.median(build) / 1000, 2) if build else 0.0,
        "max_state_bytes": state_bytes,
        "equity_sources": sources,
        "target_us": TARGET_US,
    }
    summary["meets_target"] = summary["mean_us"] <= TARGET_US
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruleset", action="append", choices=RULESETS)
    ap.add_argument("--tables", type=int, default=2)
    ap.add_argument("--hand-cap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 1 when a variant's mean exceeds the target")
    a = ap.parse_args(argv)
    results = [run(r, tables=a.tables, hand_cap=a.hand_cap, seed=a.seed, repeat=a.repeat)
               for r in a.ruleset or RULESETS]
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'ruleset':24s} {'n':>6s} {'mean':>7s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>8s}  us "
              f"(target mean <= {TARGET_US:.0f})")
        for r in results:
            print(f"{r['ruleset_id']:24s} {r['decisions']:6d} {r['mean_us']:7.1f} {r['p50_us']:7.1f} "
                  f"{r['p95_us']:7.1f} {r['p99_us']:7.1f} {r['max_us']:8.1f}  streets {r['by_street_mean_us']} "
                  f"build {r['engine_build_ms']} ms, state {r['max_state_bytes']} B, equity {r['equity_sources']}")
    return 1 if a.check and not all(r["meets_target"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
