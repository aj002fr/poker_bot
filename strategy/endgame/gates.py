"""Stage 3 endgame gates: exact regrets, multiway deviation gains, pruning gaps and table loss.

``normalised point`` = points / first prize (the largest single payout; 50 in the
default 8-player structure, 65 heads-up). Chip-EV results are also given in big blinds.

1. Two-player states, exact: heads-up grids (depth x stack ratio x horizon, both
   variants) and three-handed states with the big blind all-in from posting (two acting
   players, non-affine ICM). Gate: largest conditional action regret < 0.01.
2. Multiway (3-8 players): every player's exact unilateral best-response gain against
   the averaged CFR+ profile, by iteration count. No equilibrium guarantee; reported.
3. Pruning: the small blind's equilibrium gain from limp / min-raise branches
   (:mod:`strategy.endgame.limp`), heads-up and blind-vs-blind with bystanders.
4. Late states: one hand before the cap, solved by backward induction from the
   official cap score (the cap hand itself solved for every survivor vector) and
   compared with the ICM continuation.
5. Table audit: on held-out states (a seed the generator never used), each player's
   loss when it plays the table instead of the exact solution, the others fixed.

    python -m strategy.endgame.gates --jobs 8      # writes artifacts/endgame/gates.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from strategy.endgame.keys import Snapshot

OUT = Path("artifacts/endgame")
HU_DEPTHS = (1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)
HU_RATIOS = (0.33, 0.5, 1.0, 2.0, 3.0)
GATE = 0.01


def _hu_snapshot(depth: float, ratio: float, final: bool, payouts) -> Snapshot:
    bb = 100
    a = int(depth * bb)
    b = max(1, int(a * ratio))
    return Snapshot((a, b), (min(50, a), min(100, b)), bb, payouts, 200 if final else 100, 200)


def _report_state(args):
    from strategy.endgame.harvest import solve_snapshot
    from strategy.endgame.showdown import load_showdown

    ruleset, snap, players, iterations = args
    model = load_showdown(ruleset)
    solved = solve_snapshot(snap, model, iterations=iterations, with_gains=False)
    reps = [solved.solver.report(solved.sigma, q) for q in players]
    scale = snap.payouts[0]
    return {"stacks": list(snap.stacks), "final": snap.hand_number >= snap.hand_cap,
            "max_action_regret": max(r.max_action_regret for r in reps) / scale,
            "mean_action_regret": max(r.mean_action_regret for r in reps) / scale,
            "gain": max(r.gain for r in reps) / scale}


def two_player(ruleset: str, jobs: int, iterations: int = 400) -> dict:
    tasks = []
    for payouts in ((65.0, 35.0), (50.0, 30.0, 20.0, 0, 0, 0, 0, 0)):
        for d in HU_DEPTHS:
            for r in HU_RATIOS:
                for final in (False, True):
                    tasks.append((ruleset, _hu_snapshot(d, r, final, tuple(payouts)), (0, 1), iterations))
    rng = np.random.default_rng(11)
    for _ in range(40):
        bb = 100
        a, b = (int(x) for x in rng.integers(150, 3000, 2))
        c = int(rng.integers(10, 100))
        tasks.append((ruleset, Snapshot((a, b, c), (0, 50, c), bb, (50.0, 30.0, 20.0, 0, 0, 0, 0, 0),
                                        int(rng.choice([100, 200])), 200), (0, 1), iterations))
    rows = _map(_report_state, tasks, jobs)
    worst = max(r["max_action_regret"] for r in rows)
    return {"states": len(rows), "iterations": iterations, "max_action_regret": worst,
            "max_gain": max(r["gain"] for r in rows), "pass": worst < GATE,
            "worst_state": max(rows, key=lambda r: r["max_action_regret"])}


def _gains_state(args):
    from strategy.endgame.harvest import subgame_for
    from strategy.endgame.showdown import load_showdown
    from strategy.endgame.solver import PushFoldSolver
    from strategy.endgame.subgame import SubgameTree

    ruleset, snap, odd, checkpoints = args
    model = load_showdown(ruleset)
    solver = PushFoldSolver(SubgameTree(subgame_for(snap, odd_order=odd)), model)
    out = {}
    done = 0
    t0 = time.perf_counter()
    for it in checkpoints:
        solver.run(it - done)
        done = it
        out[it] = float(solver.deviation_gains(solver.average()).max()) / snap.payouts[0]
    return snap.n, out, time.perf_counter() - t0


def multiway(ruleset: str, jobs: int, per_n: int = 12, checkpoints=(50, 100, 250, 500)) -> dict:
    from strategy.tables.endgame import default_config, synthetic_state

    config = default_config(ruleset)
    rng = np.random.default_rng(23)
    by_n: dict[int, list] = {}
    while min((len(by_n.get(n, [])) for n in range(3, 9)), default=0) < per_n:
        s = synthetic_state(rng, config)
        if s is None or s.snapshot.n < 3 or len(by_n.get(s.snapshot.n, [])) >= per_n:
            continue
        by_n.setdefault(s.snapshot.n, []).append((ruleset, s.snapshot, s.odd_order, checkpoints))
    rows = _map(_gains_state, [t for n in sorted(by_n) for t in by_n[n]], jobs)
    out = {}
    for n in range(3, 9):
        rs = [r for r in rows if r[0] == n]
        out[str(n)] = {str(it): {"median": float(np.median([r[1][it] for r in rs])),
                                 "p90": float(np.quantile([r[1][it] for r in rs], 0.9)),
                                 "max": float(max(r[1][it] for r in rs))} for it in checkpoints}
        out[str(n)]["seconds_per_state"] = float(np.mean([r[2] for r in rs]))
        out[str(n)]["states"] = len(rs)
    return out


def _gap_task(args):
    from strategy.endgame.limp import restriction_gap
    from strategy.endgame.showdown import load_showdown

    ruleset, spot, scale, bb_scale = args
    g = restriction_gap(spot, load_showdown(ruleset))
    return {"sb": spot.sb_stack / spot.bb, "bb": spot.bb_stack / spot.bb,
            "bystanders_bb": [s / spot.bb for s in spot.bystanders], "utility": spot.utility,
            "gap": g.gap / scale, "gap_bb": g.gap / bb_scale if bb_scale else None,
            "solution_gain": max(max(g.push_fold_gains), max(g.extended_gains)) / scale,
            "root_mix": g.root_mix}


def pruning(ruleset: str, jobs: int) -> dict:
    from strategy.endgame.limp import BlindSpot

    tasks = []
    for d in (2, 3, 5, 6, 8, 10, 12, 15, 20, 25, 30):
        tasks.append((ruleset, BlindSpot(d * 100, d * 100, 50, 100, 100, utility="chips"), 100.0, 100.0))
        # Heads-up points: 16,000 chips in play at the 200/400 level, prizes 50/30.
        bb = 400
        tasks.append((ruleset, BlindSpot(d * bb, 16000 - d * bb, bb // 2, bb, bb, payouts=(50.0, 30.0)), 50.0,
                      None))
        # Blind vs blind on the bubble: four left, three paid, two bystanders.
        tasks.append((ruleset, BlindSpot(d * bb, d * bb, bb // 2, bb, bb, bystanders=(4000, 4000),
                                         payouts=(50.0, 30.0, 20.0, 0.0)), 50.0, None))
    rows = _map(_gap_task, tasks, jobs)
    chips = [r for r in rows if r["utility"] == "chips"]
    prune_to = max((r["sb"] for r in chips if all(x["gap_bb"] < 0.02 for x in chips if x["sb"] <= r["sb"])),
                   default=0)
    return {"rows": rows, "push_fold_only_up_to_bb": prune_to, "tolerance_bb": 0.02,
            "max_gap_normalised": max(r["gap"] for r in rows if r["utility"] == "points")}


def _late_state(args):
    from strategy.endgame.harvest import solve_snapshot
    from strategy.endgame.points import icm
    from strategy.endgame.showdown import load_showdown
    from strategy.value.model import cap_value_fn, late_values

    ruleset, snap = args
    model = load_showdown(ruleset)
    scale = snap.payouts[0]
    cap = Snapshot(snap.stacks, snap.posts, snap.bb, snap.payouts, snap.hand_cap, snap.hand_cap)
    exact_cap = late_values(cap, model)
    icm_now = np.array(icm(list(snap.stacks), snap.prizes))
    t0 = time.perf_counter()
    by_icm = solve_snapshot(snap, model, iterations=300, with_gains=False)
    exact = solve_snapshot(snap, model, iterations=300, value_fn=cap_value_fn(snap, model), with_gains=True)
    v_icm, v_exact = by_icm.solver.values(by_icm.sigma), exact.solver.values(exact.sigma)
    prior = model.prior
    shift = max(float(np.abs(by_icm.sigma[d][1] - exact.sigma[d][1]) @ prior) for d in exact.sigma)
    return {"players": snap.n, "cap_hand_vs_icm": float(np.abs(exact_cap - icm_now).max()) / scale,
            "value_change": float(np.abs(v_exact - v_icm).max()) / scale, "max_strategy_shift": shift,
            "exact_deviation_gain": float(exact.gains.max()) / scale, "seconds": time.perf_counter() - t0}


def late(ruleset: str, jobs: int, states: int = 16) -> dict:
    """One hand before the cap, solved by backward induction from the official cap score."""
    rng = np.random.default_rng(31)
    tasks = []
    payouts = (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(states):
        n = 3 if i % 2 == 0 else 4
        bb = 400
        stacks = rng.dirichlet(np.full(n, 1.5)) * 16000
        stacks = np.maximum(bb, np.floor(stacks)).astype(int)
        posts = [0] * n
        posts[-2], posts[-1] = bb // 2, bb
        tasks.append((ruleset, Snapshot(tuple(int(s) for s in stacks), tuple(posts), bb, payouts, 199, 200)))
    rows = _map(_late_state, tasks, jobs)
    return {"states": len(rows), "rows": rows,
            "max_cap_hand_vs_icm": max(r["cap_hand_vs_icm"] for r in rows),
            "max_value_change": max(r["value_change"] for r in rows),
            "max_strategy_shift": max(r["max_strategy_shift"] for r in rows)}


def _audit_state(args):
    from strategy.endgame.harvest import decision_nodes, solve_snapshot
    from strategy.endgame.keys import keys_for
    from strategy.endgame.lookup import load_table
    from strategy.endgame.showdown import load_showdown

    ruleset, mode, snap, odd = args
    model = load_showdown(ruleset)
    utility = "points" if mode == "points" else "chips"
    solved = solve_snapshot(snap, model, utility=utility, odd_order=odd, with_gains=False)
    table = load_table(mode, ruleset)
    solver, sigma = solved.solver, solved.sigma
    base = solver.values(sigma)
    swapped: dict[int, dict[int, np.ndarray]] = {}
    covered = total = 0
    for kind, node, hero, shover, caller in decision_nodes(solved.tree):
        total += 1
        keys = keys_for(kind, snap, hero, shover, caller)
        rows = [table.lookup(kind, keys, c) for c in range(model.h)]
        if any(r is None for r in rows):
            continue
        covered += 1
        p = np.array([r[0] for r in rows])
        swapped.setdefault(hero, {})[node] = np.stack([1 - p, p])
    scale = snap.payouts[0] if utility == "points" else snap.bb
    loss = []
    for q in range(snap.n):
        if q not in swapped:
            continue
        s2 = {**sigma, **swapped[q]}
        loss.append(float(base[q] - solver.values(s2)[q]) / scale)
    return snap.n, covered, total, loss


def audit(ruleset: str, mode: str, jobs: int, states: int = 160) -> dict:
    from strategy.tables.endgame import default_config, synthetic_state

    config = default_config(ruleset)
    rng = np.random.default_rng(97)
    tasks = []
    while len(tasks) < states:
        s = synthetic_state(rng, config)
        if s is not None:
            tasks.append((ruleset, mode, s.snapshot, s.odd_order))
    rows = _map(_audit_state, tasks, jobs)
    losses = [x for r in rows for x in r[3]]
    by_n = {}
    for n in sorted({r[0] for r in rows}):
        ls = [x for r in rows if r[0] == n for x in r[3]]
        by_n[str(n)] = {"players": len(ls), "mean_loss": float(np.mean(ls)) if ls else None,
                        "max_loss": float(max(ls)) if ls else None}
    return {"states": len(rows), "unit": "normalised point" if mode == "points" else "big blinds",
            "coverage": sum(r[1] for r in rows) / max(1, sum(r[2] for r in rows)),
            "mean_loss": float(np.mean(losses)), "p90_loss": float(np.quantile(losses, 0.9)),
            "max_loss": float(max(losses)), "by_players": by_n}


def _map(fn, tasks, jobs):
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            return list(ex.map(fn, tasks, chunksize=1))
    return [fn(t) for t in tasks]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rulesets", nargs="+", default=["practice_nlhe", "uncertainty_shortdeck"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", nargs="+", choices=["two_player", "multiway", "pruning", "late", "audit"])
    ap.add_argument("--out", default=str(OUT / "gates.json"))
    a = ap.parse_args(argv)
    sections = a.only or ["two_player", "multiway", "pruning", "late", "audit"]
    out: dict = {}
    for ruleset in a.rulesets:
        res: dict = {}
        t0 = time.perf_counter()
        if "two_player" in sections:
            res["two_player"] = two_player(ruleset, a.jobs)
            print(ruleset, "two-player", json.dumps({k: v for k, v in res["two_player"].items()
                                                      if k != "worst_state"}), flush=True)
        if "multiway" in sections:
            res["multiway"] = multiway(ruleset, a.jobs)
            print(ruleset, "multiway", json.dumps(res["multiway"]), flush=True)
        if "pruning" in sections:
            res["pruning"] = pruning(ruleset, a.jobs)
            print(ruleset, "pruning up to", res["pruning"]["push_fold_only_up_to_bb"], "bb; max points gap",
                  res["pruning"]["max_gap_normalised"], flush=True)
        if "late" in sections:
            res["late"] = late(ruleset, a.jobs)
            print(ruleset, "late", json.dumps({k: v for k, v in res["late"].items() if k != "rows"}), flush=True)
        if "audit" in sections:
            res["audit"] = {m: audit(ruleset, m, a.jobs) for m in ("points", "chips")}
            print(ruleset, "audit", json.dumps(res["audit"]), flush=True)
        res["seconds"] = round(time.perf_counter() - t0, 1)
        out[ruleset] = res
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
