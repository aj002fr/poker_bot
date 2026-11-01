"""Opponent-family likelihood table: measured action frequencies of the arena's bots.

Plays seeded in-process tables of mixed lineups drawn from the variant's available
styles and baselines, reads only the spectator (public) event stream, and counts each
eligible action by ``(family, opportunity x depth cell, category)`` using the same
context definitions as :mod:`strategy.beliefs.posterior`. Probabilities are smoothed
toward the population prior with ``SMOOTH`` pseudo-counts.

    python -m strategy.tables.families --ruleset practice_nlhe --jobs 8
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from strategy.beliefs.posterior import DEFAULT_PRIOR, OPPORTUNITIES, from_record
from strategy.beliefs.types import DEPTHS, FAMILY_CELLS, family_cell
from strategy.features.state import HandContext
from strategy.tables import store

GENERATOR = "strategy.tables.families"
INPUTS = ["beliefs/posterior.py", "beliefs/types.py", "features/state.py", "tables/families.py"]
SMOOTH = 4.0

BASELINE_FAMILIES = ("calling_station", "check_fold", "legal_random")
FAMILIES = {
    "practice_nlhe": ("rock", "aggressive", "super_aggressive", "station", "weak_tight", "switcher",
                      "nlhe_heuristic", *BASELINE_FAMILIES),
    "uncertainty_shortdeck": BASELINE_FAMILIES,
    "uncertainty_plo4": BASELINE_FAMILIES,
}
DEFAULTS = {"practice_nlhe": 240, "uncertainty_shortdeck": 64, "uncertainty_plo4": 64}


def table_name(ruleset_id: str) -> str:
    return f"families-{ruleset_id}"


def _bot_path(name: str) -> Path:
    from arena.agents.baselines import BASELINES
    from arena.agents.styles import STYLES

    return STYLES.get(name) or BASELINES[name]


def _play(args: tuple[str, int, int, int]) -> dict[str, list[list[float]]]:
    from arena.agents.sdk import TableInfo, load_bot
    from arena.agents.worker import InProcessSeat, play_table
    from arena.benchmark.runner import policy_seed
    from arena.domain.models import TournamentConfig
    from arena.table.projections import Viewer, project_events
    from arena.table.service import seeded_table

    ruleset_id, index, seed, hand_cap = args
    fams = FAMILIES[ruleset_id]
    h = hashlib.sha256(f"{GENERATOR}|{ruleset_id}|{seed}|{index}".encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    lineup = [fams[rng.randrange(len(fams))] for _ in range(8)]
    config = TournamentConfig(ruleset_id=ruleset_id, initial_player_count=8, hand_cap=hand_cap)
    table_seed = f"families-{seed}-{index}"
    agents = {}
    names = dict(zip(config.resolved_seats, lineup, strict=True))
    for s, name in names.items():
        bot, info = load_bot(_bot_path(name), module_name=f"strategy_family_{name}")
        ti = TableInfo(table_id=f"t_fam_{index}", hero_seat=s, ruleset_id=ruleset_id, config=config.resolved(),
                       config_hash=config.config_hash(), seats=config.resolved_seats,
                       decision_budget_ns=config.bot_budget_ns, policy_seed=policy_seed(table_seed, s))
        agent = InProcessSeat(s, bot, info, ti, "family-table")
        agent.start()
        agents[s] = agent
    table = seeded_table(config, table_seed, table_id=f"t_fam_{index}")
    play_table(table, agents)
    ctx = HandContext(config.resolved_seats)
    counts = {f: [[0.0, 0.0, 0.0] for _ in range(FAMILY_CELLS)] for f in fams}
    ctx.feed(project_events(table.log, Viewer.spectator()))
    for rec in ctx.drain():
        ob = from_record(rec)
        if ob is not None:
            counts[names[rec.seat]][family_cell(ob.opportunity, rec.eff_bb)][ob.category] += 1
    return counts


def build(ruleset_id: str, *, tables: int, seed: int = 1, jobs: int = 1, hand_cap: int = 200) -> dict:
    started = time.perf_counter()
    fams = FAMILIES[ruleset_id]
    work = [(ruleset_id, i, seed, hand_cap) for i in range(tables)]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(_play, work))
    else:
        results = [_play(w) for w in work]
    total = {f: [[0.0, 0.0, 0.0] for _ in range(FAMILY_CELLS)] for f in fams}
    for res in results:
        for f, cells in res.items():
            for c, row in enumerate(cells):
                for k in range(3):
                    total[f][c][k] += row[k]
    probs = {}
    for f in fams:
        rows = []
        for c, row in enumerate(total[f]):
            prior = DEFAULT_PRIOR[c // DEPTHS]
            n = sum(row)
            rows.append([round((row[k] + SMOOTH * prior[k]) / (n + SMOOTH), 4) for k in range(3)])
        probs[f] = rows
    return {"ruleset_id": ruleset_id, "families": list(fams), "opportunities": list(OPPORTUNITIES), "depths": DEPTHS,
            "smooth": SMOOTH, "counts": {f: [[int(x) for x in r] for r in total[f]] for f in fams}, "probs": probs,
            "_seconds": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruleset", required=True)
    ap.add_argument("--tables", type=int)
    ap.add_argument("--hand-cap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    params = {"tables": a.tables or DEFAULTS[a.ruleset], "seed": a.seed, "hand_cap": a.hand_cap}
    data = build(a.ruleset, jobs=a.jobs, **params)
    seconds = data.pop("_seconds")
    path = store.write(table_name(a.ruleset), data, generator=GENERATOR, inputs=INPUTS,
                       params={"ruleset_id": a.ruleset, **params}, seconds=seconds,
                       directory=Path(a.out) if a.out else None)
    print(f"wrote {path} ({path.stat().st_size} bytes, {seconds:.1f}s)")


if __name__ == "__main__":
    main()
