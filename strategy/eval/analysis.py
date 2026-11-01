"""Primary estimate, section-5 slices, robustness diagnostics and the promotion decision.

Everything is computed from block pairs (:class:`~strategy.eval.paired.BlockPair`)
and per-table records; the only inferential unit is the block. Slices that condition
on an outcome (survivor count, finish method, bust phase) are post-treatment and are
reported descriptively per arm, never as paired effects. Slice intervals are
exploratory; family, seat and variant slices also carry Holm-adjusted p-values.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from strategy.eval.execution import evaluate_gates, merge_execution
from strategy.eval.outcomes import ACTION_CLASSES
from strategy.eval.paired import CANDIDATE, BlockPair, CellResult, pair_blocks
from strategy.eval.stats import (
    bootstrap_mean,
    holm,
    mean,
    minimum_detectable_effect,
    newcombe_difference,
    paired_t,
    quantile,
    sample_sd,
    wilson,
)

REPORTED_RATE_CLASSES = ("raise", "all_in", "fold")


@dataclass(frozen=True)
class Declared:
    """What was fixed before the run (copied from the experiment spec)."""

    min_effect: float
    alpha: float = 0.05
    power: float = 0.80
    interval: str = "paired_t"
    bootstrap_resamples: int = 2000
    seed: str = "analysis"
    stage: str = "discovery"
    gates: Mapping[str, Any] | None = None

    @property
    def level(self) -> float:
        return 1 - self.alpha


def _summary(values: Sequence[float], level: float) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    t = paired_t(values, level)
    return {"n": t.n, "mean_difference": t.mean, "sd": t.sd, "lo": t.lo, "hi": t.hi, "p_value": t.p_value}


def _interval(pairs: Sequence[BlockPair], d: Declared) -> dict[str, Any]:
    diffs = [p.d for p in pairs]
    t = paired_t(diffs, d.level)
    strata = [p.cell_id for p in pairs]
    boot = bootstrap_mean(diffs, seed=f"{d.seed}|primary", strata=strata if len(set(strata)) > 1 else None,
                          resamples=d.bootstrap_resamples, level=d.level)
    return {"paired_t": t.as_dict(), "cluster_bootstrap": boot.as_dict()}


def decide(lo: float | None, hi: float | None, min_effect: float, gates_ok: bool, excluded: int) -> dict[str, Any]:
    """``pass``: B beats A by more than the declared minimum effect, with every gate green."""
    if lo is None or hi is None:
        verdict, why = "inconclusive", "fewer than two blocks"
    elif excluded:
        verdict, why = "invalid", f"{excluded} blocks excluded; promotion needs complete blocks"
    elif not gates_ok:
        verdict, why = "fail", "an execution gate failed"
    elif lo > min_effect:
        verdict, why = "pass", f"interval lower bound {lo:.4g} exceeds the minimum effect {min_effect:g}"
    elif hi < 0:
        verdict, why = "fail", f"B is worse: interval upper bound {hi:.4g} is below zero"
    else:
        verdict, why = "inconclusive", f"interval [{lo:.4g}, {hi:.4g}] does not clear {min_effect:g}"
    return {"verdict": verdict, "reason": why}


def _paired_slice(groups: Mapping[str, list[float]], d: Declared) -> dict[str, Any]:
    rows = {k: _summary(v, d.level) for k, v in sorted(groups.items())}
    ps = {k: r["p_value"] for k, r in rows.items() if r.get("p_value") is not None}
    for k, adj in holm(ps).items():
        rows[k]["p_holm"] = adj
    return rows


def _records(cells: Iterable[CellResult], arm: str) -> list[Any]:
    return [r for c in cells for r in c.runs[arm].records if r.valid]


def _arm_descriptives(records: Sequence[Any], payouts: Sequence[int]) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"tables": 0}
    ents = [r.entrants[CANDIDATE] for r in records]
    positions: Counter = Counter()
    methods: Counter = Counter()
    bust: Counter = Counter()
    survivors: dict[int, list[float]] = defaultdict(list)
    for r, e in zip(records, ents):
        methods[e.finish.method] += 1
        if e.finish.method == "elimination":
            for p in e.finish.positions:
                positions[str(p)] += 1 / len(e.finish.positions)
            bust[f"L{e.bust_level + 1}"] += 1
        else:
            positions["survived_icm" if e.finish.method == "icm" else "1"] += 1
        survivors[r.survivors_at_end].append(e.points)
    pts = [e.points for e in ents]
    icm_pts = [e.cap_icm_points for e in ents]
    worst = sorted(pts)[: max(1, math.ceil(0.1 * n))]
    return {
        "tables": n,
        "mean_points": mean(pts),
        "sd_points": sample_sd(pts),
        "finish_position_distribution": {k: v / n for k, v in sorted(positions.items())},
        "finish_method": {k: v / n for k, v in sorted(methods.items())},
        "tables_ended_at_cap": sum(1 for r in records if r.ended_at_cap) / n,
        "cap_icm_points_mean": mean(icm_pts),
        "cap_icm_share_of_points": (sum(icm_pts) / sum(pts)) if sum(pts) else None,
        "bust_blind_level": {k: v / n for k, v in sorted(bust.items(), key=lambda kv: int(kv[0][1:]))},
        "points_by_survivors_at_end": {str(k): {"tables": len(v), "mean_points": mean(v)}
                                       for k, v in sorted(survivors.items())},
        "lower_tail_cvar10_points": mean(worst),
        "payout_total": sum(payouts),
        "chips_diagnostic": {
            "diagnostic_only": True,
            "mean_final_chips": mean([e.chips["final_chips"] for e in ents]),
            "mean_chip_delta": mean([e.chips["chip_delta"] for e in ents]),
        },
    }


def _opportunities(records: Sequence[Any]) -> Counter:
    total: Counter = Counter()
    for r in records:
        total.update(r.entrants[CANDIDATE].opportunities)
    return total


def opportunity_rates(opp_a: Counter, opp_b: Counter, level: float) -> list[dict[str, Any]]:
    keys = sorted({k.rsplit("|", 1)[0] for k in (*opp_a, *opp_b)})
    rows = []
    for key in keys:
        na, nb = opp_a.get(f"{key}|*", 0), opp_b.get(f"{key}|*", 0)
        row: dict[str, Any] = {"slice": key, "opportunities_A": na, "opportunities_B": nb, "rates": {}}
        for cls in REPORTED_RATE_CLASSES:
            ka, kb = opp_a.get(f"{key}|{cls}", 0), opp_b.get(f"{key}|{cls}", 0)
            row["rates"][cls] = {
                "A": {"k": ka, "n": na, "rate": ka / na if na else None, "wilson": wilson(ka, na, level)},
                "B": {"k": kb, "n": nb, "rate": kb / nb if nb else None, "wilson": wilson(kb, nb, level)},
                "difference": (kb / nb - ka / na) if na and nb else None,
                "newcombe": newcombe_difference(ka, na, kb, nb, level) if na and nb else None,
            }
        row["action_mix_B"] = {c: opp_b.get(f"{key}|{c}", 0) for c in ACTION_CLASSES}
        rows.append(row)
    return rows


def analyze(cells: Sequence[CellResult], declared: Declared) -> dict[str, Any]:
    pairs: list[BlockPair] = []
    excluded: list[dict[str, Any]] = []
    for c in cells:
        p, x = pair_blocks(c)
        pairs += p
        excluded += x
    diffs = [p.d for p in pairs]
    level = declared.level
    primary: dict[str, Any] = {
        "estimand": "mean official terminal points per tournament, arm B minus arm A, candidate seat",
        "unit": "paired seed block (all seat rotations, both arms)",
        "blocks": len(pairs),
        "blocks_excluded": excluded,
        "tables_per_arm": sum(len(c.runs["A"].records) for c in cells),
        "min_effect": declared.min_effect,
        "level": level,
        "interval_method": declared.interval,
    }
    if pairs:
        primary.update({
            "mean_points_A": mean([p.u_a for p in pairs]),
            "mean_points_B": mean([p.u_b for p in pairs]),
            "mean_difference": mean(diffs),
            "sd_difference": sample_sd(diffs),
            "blocks_B_better": sum(1 for x in diffs if x > 0),
            "blocks_tied": sum(1 for x in diffs if x == 0),
            "blocks_A_better": sum(1 for x in diffs if x < 0),
            "max_abs_block_difference": max(abs(x) for x in diffs),
            **_interval(pairs, declared),
        })
        chosen = primary["paired_t"] if declared.interval == "paired_t" else primary["cluster_bootstrap"]
        primary["chosen_interval"] = [chosen["lo"], chosen["hi"]]
        primary["minimum_detectable_effect"] = minimum_detectable_effect(
            primary["sd_difference"], len(pairs), alpha=declared.alpha, power=declared.power)
    payouts = cells[0].cell.config.resolved_payouts if cells else []

    by_seat: dict[str, list[float]] = defaultdict(list)
    by_variant: dict[str, list[float]] = defaultdict(list)
    by_lineup: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        for seat, x in p.seat_diffs.items():
            by_seat[f"seat {seat}"].append(x)
        by_variant[p.ruleset_id].append(p.d)
        by_lineup[p.cell_id].append(p.d)
        for fam in p.families:
            by_family[fam].append(p.d)
    recs_a, recs_b = _records(cells, "A"), _records(cells, "B")
    slices = {
        "seat": _paired_slice(by_seat, declared),
        "variant": _paired_slice(by_variant, declared),
        "lineup": _paired_slice(by_lineup, declared),
        "family": _paired_slice(by_family, declared),
        "descriptive_by_arm": {"A": _arm_descriptives(recs_a, payouts), "B": _arm_descriptives(recs_b, payouts)},
        "opportunity_rates": opportunity_rates(_opportunities(recs_a), _opportunities(recs_b), level),
    }
    lineup_means = [mean(v) for v in by_lineup.values()]
    family_means = {k: mean(v) for k, v in by_family.items()}
    worst_family = min(family_means, key=lambda k: family_means[k]) if family_means else None
    robustness = {
        "worst_family": {"family": worst_family, **slices["family"][worst_family]} if worst_family else None,
        "lineup_p5_mean_difference": quantile(lineup_means, 0.05) if len(lineup_means) >= 2 else None,
        "lineups": len(lineup_means),
        "block_difference_p5": quantile(diffs, 0.05) if diffs else None,
        "lower_tail_cvar10_points": {arm: slices["descriptive_by_arm"][arm].get("lower_tail_cvar10_points")
                                     for arm in ("A", "B")},
    }
    executions = {}
    for arm in ("A", "B"):
        parts = [c.runs[arm].execution for c in cells]
        raw: dict[str, list[int]] = defaultdict(list)
        for c in cells:
            for k, v in c.runs[arm].raw_latency.items():
                raw[k].extend(v)
        executions[arm] = merge_execution(parts, raw)
    gate_checks = evaluate_gates(executions["B"], declared.gates or {}) if executions.get("B") else []
    gates_ok = all(g["passed"] for g in gate_checks)
    lo, hi = primary.get("chosen_interval", (None, None))
    return {
        "primary": primary,
        "decision": {**decide(lo, hi, declared.min_effect, gates_ok, len(excluded)), "stage": declared.stage,
                     "note": "discovery results must be confirmed on fresh locked data" if declared.stage ==
                     "discovery" else None},
        "slices": slices,
        "robustness": robustness,
        "execution": {"A": executions.get("A"), "B": executions.get("B"), "gates": gate_checks},
        "not_measured": {
            "approximate_response_loss": "needs response training (stage 5+)",
            "small_game_regret": "needs the toy-game solver ladder (stage 2)",
            "resident_memory": "the arena does not sample RSS; the sandbox enforces RLIMIT_AS",
        },
        "pairs": [{"cluster": p.cluster, "cell_id": p.cell_id, "block": p.block, "u_a": p.u_a, "u_b": p.u_b,
                   "d": p.d} for p in pairs],
    }
