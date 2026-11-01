"""Point-aware push/fold tables per variant, from solved full-stack-vector subgames.

States are start-of-hand stack vectors drawn from complete tournament rollouts of the
training partition (so the validation opponents never shape the tables) plus synthetic
stack vectors that cover the rest of the key space. Each state is solved as a push/fold
subgame (:mod:`strategy.endgame.subgame`) and every covered decision is harvested
(:mod:`strategy.endgame.harvest`) into the key ladder of :mod:`strategy.endgame.keys`.
A table row is the reach-weighted mean, over the states that hit that key, of the
probability of the aggressive action (shove or call) per hand class.

Modes (one table per mode and variant, ``endgame-<mode>-<ruleset>.npz``):

``points``  survivors valued by the continuation value (ICM by default, or a fitted
            value model with ``--value``), eliminations by finishing points, the
            hand-cap hand by the official cap ICM: the point-aware tables.
``chips``   the same states with chip-EV utility: the chip-EV baseline.
``chart``   symmetric stacks only, chip-EV, coarsest keys only: a fixed push/fold chart.

Arrays: ``<kind>_keys`` (sorted int64), ``<kind>_probs`` (uint8, ``rows x classes``,
probability x 255), ``<kind>_weight`` (float32 total reach weight), ``<kind>_count``
(int32 states with weight above :data:`COUNT_FLOOR`); ``class_keys`` (the variant's hand
classes, which the runtime checks); ``diag_*`` per solved state (players, the largest
unilateral deviation gain normalised by the first prize or by the big blind for chips,
and the iterations).

    python -m strategy.tables.endgame --ruleset practice_nlhe --mode points --jobs 8
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from strategy.endgame.keys import LEVELS, MAX_DEPTH_BB, Snapshot
from strategy.endgame.table_state import SeatedSnapshot, seated_snapshot
from strategy.tables import store

GENERATOR = "strategy.tables.endgame"
INPUTS = ["endgame/classes.py", "endgame/harvest.py", "endgame/keys.py", "endgame/points.py", "endgame/showdown.py",
          "endgame/solver.py", "endgame/subgame.py", "endgame/table_state.py", "tables/endgame.py",
          "value/rollouts.py", "rules/variants.py"]
MODES = ("points", "chips", "chart")
DEFAULTS = {"rollout_blocks": 6, "synthetic": 3000, "max_states": 6000, "seed": 1}
COUNT_FLOOR = 1e-3
"""A state counts towards a key's support when its reach weight there is at least this."""
CHART_DEPTHS = (1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 16.0,
                18.5, 21.5, 24.5, 28.0)
"""Big blinds per player of the chart's symmetric states: one inside each depth bucket."""


def table_name(mode: str, ruleset_id: str) -> str:
    return f"endgame-{mode}-{ruleset_id}"


def default_config(ruleset_id: str):
    from arena.domain.models import TournamentConfig

    return TournamentConfig(ruleset_id=ruleset_id)


def training_policies(ruleset_id: str, seats: int) -> list[str]:
    """The training partition's members that play the variant, repeated to fill the table."""
    from strategy.eval.population import StubPartitionSource

    members = sorted(StubPartitionSource().partition("train").eligible(ruleset_id), key=lambda m: m.member_id)
    if not members:
        raise ValueError(f"no training opponents play {ruleset_id}")
    return [members[i % len(members)].policy for i in range(seats)]


# ------------------------------------------------------------------- states


def in_scope(snap: Snapshot) -> bool:
    return snap.n >= 2 and min(snap.stacks) / snap.bb <= MAX_DEPTH_BB


def rollout_states(ruleset_id: str, *, blocks: int, seed: int, jobs: int) -> list[SeatedSnapshot]:
    from strategy.value.rollouts import run_rollouts

    config = default_config(ruleset_id)
    policies = training_policies(ruleset_id, config.initial_player_count)
    rollouts = run_rollouts(config, policies, blocks=blocks, base_seed=f"endgame-states-{seed}", jobs=jobs)
    out = []
    for ro in rollouts:
        for h in ro.hands:
            s = seated_snapshot(ring=ro.ring, start_stacks=h.start_stacks, button=h.button,
                                small_blind_seat=h.small_blind_seat, big_blind=h.big_blind, heads_up=h.heads_up,
                                blinds=h.blinds, payouts=ro.payouts, hand_number=h.hand_number,
                                hand_cap=ro.hand_cap)
            if in_scope(s.snapshot):
                out.append(s)
    return out


def _blinds_at(config, hand_number: int) -> tuple[int, int]:
    level = min((hand_number - 1) // config.hands_per_level, len(config.blind_levels) - 1)
    sb, bb = config.blind_levels[level]
    return sb, bb


def synthetic_state(rng: np.random.Generator, config) -> SeatedSnapshot | None:
    """A random live-player count, stack vector and hand number of the default tournament."""
    seats = config.initial_player_count
    n = int(rng.choice(np.arange(2, seats + 1), p=_n_weights(seats)))
    u = rng.random()
    if u < 0.1:
        hand = config.hand_cap
    elif u < 0.2:
        hand = int(config.hand_cap - rng.integers(1, n + 1))
    else:
        hand = int(rng.integers(1, config.hand_cap + 1))
    sb, bb = _blinds_at(config, hand)
    total = seats * config.starting_stack
    shares = rng.dirichlet(np.full(n, float(rng.choice([0.7, 1.5, 4.0]))))
    stacks = np.maximum(1, np.floor(shares * total).astype(np.int64))
    stacks[int(np.argmax(stacks))] += total - int(stacks.sum())
    if stacks.min() <= 0:
        return None
    ring = tuple(range(seats))
    live = sorted(rng.choice(seats, n, replace=False).tolist())
    start = {s: int(c) for s, c in zip(live, stacks)}
    i = int(rng.integers(n))
    big = live[i]
    if n == 2:
        button = live[1 - i]
        small = button
    else:
        small = live[i - 1]
        button = live[i - 2]
    s = seated_snapshot(ring=ring, start_stacks=start, button=button, small_blind_seat=small, big_blind=big,
                        heads_up=n == 2, blinds=(sb, bb), payouts=config.resolved_payouts, hand_number=hand,
                        hand_cap=config.hand_cap)
    return s if in_scope(s.snapshot) else None


def _n_weights(seats: int) -> np.ndarray:
    w = np.array([1.0 / math.sqrt(k - 1) for k in range(2, seats + 1)])
    return w / w.sum()


def chart_states(config, *, bb: int = 100) -> list[SeatedSnapshot]:
    out = []
    for n in range(2, config.initial_player_count + 1):
        ring = tuple(range(n))
        for depth in CHART_DEPTHS:
            start = {s: int(round(depth * bb)) for s in ring}
            big = n - 1
            button = 0 if n == 2 else n - 3
            small = 0 if n == 2 else n - 2
            out.append(seated_snapshot(ring=ring, start_stacks=start, button=button, small_blind_seat=small,
                                       big_blind=big, heads_up=n == 2, blinds=(bb // 2, bb),
                                       payouts=config.resolved_payouts, hand_number=1,
                                       hand_cap=config.hand_cap))
    return out


def sample_states(ruleset_id: str, mode: str, *, rollout_blocks: int, synthetic: int, max_states: int, seed: int,
                  jobs: int) -> list[SeatedSnapshot]:
    config = default_config(ruleset_id)
    if mode == "chart":
        return chart_states(config)
    states = rollout_states(ruleset_id, blocks=rollout_blocks, seed=seed, jobs=jobs) if rollout_blocks else []
    rng = np.random.default_rng(seed)
    if len(states) > max_states - synthetic:
        keep = rng.choice(len(states), max_states - synthetic, replace=False)
        states = [states[i] for i in sorted(keep)]
    made = 0
    while made < synthetic:
        s = synthetic_state(rng, config)
        if s is not None:
            states.append(s)
            made += 1
    return states


# ------------------------------------------------------------------ solving

_MODEL = None
_VALUE = None


def _init(ruleset_id: str, value_name: str | None, threads: int) -> None:
    global _MODEL, _VALUE
    from strategy.endgame.showdown import load_showdown

    _MODEL = load_showdown(ruleset_id)
    if value_name:
        from strategy.value.model import load_value_fn

        _VALUE = load_value_fn(value_name)


def _solve_chunk(args) -> tuple[dict, list[tuple[int, float, int]]]:
    from strategy.endgame.harvest import harvest, solve_snapshot

    items, mode, levels = args
    acc: dict[tuple[str, int], list] = {}
    diags = []
    utility = "points" if mode == "points" else "chips"
    for snap, odd in items:
        value_fn = _VALUE.for_snapshot(snap) if _VALUE is not None and utility == "points" else None
        solved = solve_snapshot(snap, _MODEL, utility=utility, odd_order=odd, value_fn=value_fn)
        scale = snap.payouts[0] if utility == "points" else snap.bb
        diags.append((snap.n, float(np.max(solved.gains)) / scale, solved.solver.iterations))
        for rec in harvest(solved, max_depth_bb=MAX_DEPTH_BB):
            for level, key in enumerate(rec.keys):
                if level not in levels:
                    continue
                slot = acc.get((rec.kind, key))
                if slot is None:
                    slot = acc[(rec.kind, key)] = [np.zeros(len(rec.row)), 0.0, 0]
                slot[0] += rec.weight * rec.row
                slot[1] += rec.weight
                slot[2] += int(rec.weight >= COUNT_FLOOR)
    return acc, diags


def _chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def solve_states(ruleset_id: str, mode: str, states: Sequence[SeatedSnapshot], *, jobs: int,
                 value_name: str | None = None, levels: Iterable[int] | None = None):
    levels = frozenset(levels if levels is not None else (range(LEVELS) if mode != "chart" else (LEVELS - 1,)))
    items = [(s.snapshot, s.odd_order) for s in states]
    # Big states first so the pool does not finish on one slow chunk.
    items.sort(key=lambda it: -it[0].n)
    tasks = [(list(c), mode, levels) for c in _chunks(items, 8)]
    acc: dict[tuple[str, int], list] = {}
    diags: list[tuple[int, float, int]] = []
    if jobs > 1:
        with ProcessPoolExecutor(jobs, initializer=_init, initargs=(ruleset_id, value_name, 1)) as ex:
            results = ex.map(_solve_chunk, tasks)
            for part, d in results:
                _merge(acc, part)
                diags.extend(d)
    else:
        _init(ruleset_id, value_name, 1)
        for t in tasks:
            part, d = _solve_chunk(t)
            _merge(acc, part)
            diags.extend(d)
    return acc, diags


def _merge(acc: dict, part: dict) -> None:
    for k, (s, w, c) in part.items():
        slot = acc.get(k)
        if slot is None:
            acc[k] = [s, w, c]
        else:
            slot[0] += s
            slot[1] += w
            slot[2] += c


def to_arrays(acc: dict, class_keys: np.ndarray, diags: Sequence[tuple[int, float, int]]) -> dict[str, np.ndarray]:
    from strategy.endgame.keys import KINDS

    h = len(class_keys)
    out: dict[str, np.ndarray] = {"class_keys": np.asarray(class_keys, np.int64)}
    for kind in KINDS:
        keys = sorted(k for kd, k in acc if kd == kind)
        probs = np.zeros((len(keys), h), np.uint8)
        weight = np.zeros(len(keys), np.float32)
        count = np.zeros(len(keys), np.int32)
        for r, k in enumerate(keys):
            s, w, c = acc[(kind, k)]
            if w > 0:
                probs[r] = np.clip(np.rint(255 * s / w), 0, 255).astype(np.uint8)
            weight[r] = w
            count[r] = c
        out[f"{kind}_keys"] = np.array(keys, np.int64)
        out[f"{kind}_probs"] = probs
        out[f"{kind}_weight"] = weight
        out[f"{kind}_count"] = count
    d = np.array(diags, np.float64).reshape(-1, 3)
    out["diag_players"] = d[:, 0].astype(np.int8)
    out["diag_gain"] = d[:, 1].astype(np.float32)
    out["diag_iterations"] = d[:, 2].astype(np.int16)
    return out


def build(ruleset_id: str, mode: str, *, rollout_blocks: int, synthetic: int, max_states: int, seed: int, jobs: int,
          value_name: str | None = None) -> tuple[dict[str, np.ndarray], dict]:
    from strategy.endgame.classes import PUSH_FOLD_CAPABILITIES, hand_classes
    from strategy.endgame.showdown import load_showdown
    from strategy.rules.variants import get_variant

    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}")
    get_variant(ruleset_id).require(*PUSH_FOLD_CAPABILITIES, asset="push/fold tables")
    hc = hand_classes(ruleset_id)
    model = load_showdown(ruleset_id)
    started = time.perf_counter()
    states = sample_states(ruleset_id, mode, rollout_blocks=rollout_blocks, synthetic=synthetic,
                           max_states=max_states, seed=seed, jobs=jobs)
    sampled = time.perf_counter() - started
    acc, diags = solve_states(ruleset_id, mode, states, jobs=jobs, value_name=value_name)
    arrays = to_arrays(acc, np.array(hc.keys, np.int64), diags)
    showdown_prov = store.load_arrays(f"showdown-{ruleset_id}")[0]
    extra = {"ruleset_id": ruleset_id, "variant_hash": hc.variant_hash, "mode": mode,
             "utility": "points" if mode == "points" else "chips", "value": value_name or "icm",
             "showdown_sha256": showdown_prov["data_sha256"], "states": len(states),
             "sample_seconds": round(sampled, 1), "payouts": list(states[0].snapshot.payouts),
             "max_depth_bb": MAX_DEPTH_BB, "levels": LEVELS, "model_classes": model.h}
    return arrays, extra


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruleset", required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    ap.add_argument("--value", help="continuation-value table for --mode points (default: ICM)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", help="output directory (default: strategy/tables/data)")
    a = ap.parse_args(argv)
    params = {k: getattr(a, k) for k in DEFAULTS}
    started = time.perf_counter()
    arrays, extra = build(a.ruleset, a.mode, jobs=a.jobs, value_name=a.value, **params)
    seconds = time.perf_counter() - started
    name = table_name(a.mode, a.ruleset)
    path = store.write_arrays(name, arrays, generator=GENERATOR, inputs=INPUTS,
                              params={"ruleset_id": a.ruleset, "mode": a.mode, "value": a.value, **params},
                              seconds=seconds, extra=extra, directory=Path(a.out) if a.out else None)
    g = arrays["diag_gain"]
    rows = sum(len(arrays[f"{k}_keys"]) for k in ("open", "call", "overcall"))
    print(f"wrote {path} ({path.stat().st_size} bytes, {extra['states']} states, {rows} rows, {seconds:.0f}s); "
          f"deviation gain median {np.median(g):.2e} p99 {np.quantile(g, 0.99):.2e} max {g.max():.2e}")


if __name__ == "__main__":
    main()
