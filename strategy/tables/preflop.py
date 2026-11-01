"""Preflop equity table per variant: one row per canonical hole class.

For each suit-isomorphism class (169 in hold'em, 81 in short deck, 16,432 in PLO4) it
stores, by seeded Monte Carlo from a representative combo:

* ``eq[k-1]``: equity against ``k`` = 1..7 random hands (ties as shares);
* ``tie``: heads-up tie share;
* ``flop_mean``/``flop_sq``/``hist``: the distribution, across random flops, of the
  heads-up equity on that flop (10 bins), i.e. the future-equity distribution;
* ``se``: standard error of the heads-up equity.

Derived per variant: combo-weighted ``percentile`` (strength by heads-up equity), the
strength decile ``bucket`` used as the latent range bucket, and ``bucket_card`` (combos
per bucket containing each card) for card-removal corrections.

    python -m strategy.tables.preflop --ruleset practice_nlhe
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

from strategy.features.cards import canonical_hole_key, capabilities, evaluate, hole_label
from strategy.tables import store

GENERATOR = "strategy.tables.preflop"
INPUTS = ["features/cards.py", "tables/preflop.py"]
SCALE = 10000
MAX_OPPONENTS = 7
HIST_BINS = 10
BUCKETS = 10

DEFAULTS = {
    "practice_nlhe": {"samples": 20000, "flops": 100, "inner": 100},
    "uncertainty_shortdeck": {"samples": 20000, "flops": 100, "inner": 100},
    "uncertainty_plo4": {"samples": 600, "flops": 24, "inner": 24},
}


def table_name(ruleset_id: str) -> str:
    return f"preflop-{ruleset_id}"


def classes(ruleset_id: str) -> tuple[dict[int, tuple[int, ...]], dict[int, int]]:
    """Representative combo and combo count for every canonical class."""
    caps = capabilities(ruleset_id)
    rep: dict[int, tuple[int, ...]] = {}
    count: dict[int, int] = {}
    for combo in combinations(caps.deck, caps.hole_count):
        k = canonical_hole_key(combo)
        if k not in rep:
            rep[k] = combo
            count[k] = 0
        count[k] += 1
    return rep, count


def _rng(ruleset_id: str, key: int, seed: int) -> random.Random:
    h = hashlib.sha256(f"{GENERATOR}|{ruleset_id}|{key}|{seed}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _class_row(args: tuple[str, int, tuple[int, ...], int, int, int, int]) -> tuple[int, list[float]]:
    ruleset_id, key, hole, samples, flops, inner, seed = args
    caps = capabilities(ruleset_id)
    rng = _rng(ruleset_id, key, seed)
    deck = [c for c in caps.deck if c not in hole]
    hc = caps.hole_count
    wins = [0.0] * MAX_OPPONENTS
    tie = 0.0
    take = 5 + hc * MAX_OPPONENTS
    for _ in range(samples):
        draw = rng.sample(deck, take)
        board = draw[:5]
        hero = evaluate(hole, board, caps)
        ties = 0
        for k in range(MAX_OPPONENTS):
            v = evaluate(draw[5 + hc * k:5 + hc * (k + 1)], board, caps)
            if v > hero:
                break
            if v == hero:
                ties += 1
            wins[k] += 1.0 / (ties + 1)
            if k == 0 and ties:
                tie += 1.0
    eq = [w / samples for w in wins]
    hist = [0] * HIST_BINS
    fsum = fsq = 0.0
    inner_total = 0.0
    for _ in range(flops):
        draw = rng.sample(deck, 3)
        rest = [c for c in deck if c not in draw]
        e = 0.0
        for _ in range(inner):
            d2 = rng.sample(rest, 2 + hc)
            board = [*draw, d2[0], d2[1]]
            hv = evaluate(hole, board, caps)
            ov = evaluate(d2[2:], board, caps)
            e += 1.0 if hv > ov else 0.5 if hv == ov else 0.0
        e /= inner
        inner_total += e
        fsum += e
        fsq += e * e
        hist[min(HIST_BINS - 1, int(e * HIST_BINS))] += 1
    n1 = samples + flops * inner
    eq1 = (eq[0] * samples + inner_total * inner) / n1
    eq[0] = eq1
    se = (max(eq1 * (1 - eq1), 1e-6) / n1) ** 0.5
    row = [*eq, tie / samples, fsum / flops, fsq / flops, se, *(h / flops for h in hist)]
    return key, row


def build(ruleset_id: str, *, samples: int, flops: int, inner: int, seed: int = 1, jobs: int = 1) -> dict:
    started = time.perf_counter()
    rep, count = classes(ruleset_id)
    keys = sorted(rep)
    work = [(ruleset_id, k, rep[k], samples, flops, inner, seed) for k in keys]
    rows: dict[int, list[float]] = {}
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            for key, row in pool.map(_class_row, work, chunksize=max(1, len(work) // (jobs * 8))):
                rows[key] = row
    else:
        for item in work:
            key, row = _class_row(item)
            rows[key] = row
    total = sum(count.values())
    order = sorted(keys, key=lambda k: (rows[k][0], k))
    cum = 0
    percentile: dict[int, float] = {}
    bucket: dict[int, int] = {}
    for k in order:
        mid = (cum + count[k] / 2) / total
        percentile[k] = mid
        bucket[k] = min(BUCKETS - 1, int(mid * BUCKETS))
        cum += count[k]
    caps = capabilities(ruleset_id)
    bucket_combos = [0] * BUCKETS
    bucket_card = [[0] * 52 for _ in range(BUCKETS)]
    for combo in combinations(caps.deck, caps.hole_count):
        b = bucket[canonical_hole_key(combo)]
        bucket_combos[b] += 1
        for c in combo:
            bucket_card[b][c] += 1
    q = lambda x: int(round(x * SCALE))  # noqa: E731
    data = {
        "ruleset_id": ruleset_id,
        "scale": SCALE,
        "max_opponents": MAX_OPPONENTS,
        "hist_bins": HIST_BINS,
        "keys": keys,
        "labels": [hole_label(rep[k]) for k in keys] if caps.hole_count == 2 else [],
        "combos": [count[k] for k in keys],
        "eq": [[q(rows[k][i]) for k in keys] for i in range(MAX_OPPONENTS)],
        "tie": [q(rows[k][MAX_OPPONENTS]) for k in keys],
        "flop_mean": [q(rows[k][MAX_OPPONENTS + 1]) for k in keys],
        "flop_sq": [q(rows[k][MAX_OPPONENTS + 2]) for k in keys],
        "se": [q(rows[k][MAX_OPPONENTS + 3]) for k in keys],
        "hist": [[q(x) for x in rows[k][MAX_OPPONENTS + 4:]] for k in keys],
        "percentile": [q(percentile[k]) for k in keys],
        "bucket": [bucket[k] for k in keys],
        "bucket_combos": bucket_combos,
        "bucket_card": bucket_card,
    }
    data["_seconds"] = time.perf_counter() - started
    return data


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruleset", required=True)
    ap.add_argument("--samples", type=int)
    ap.add_argument("--flops", type=int)
    ap.add_argument("--inner", type=int)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", help="output directory (default: strategy/tables/data)")
    a = ap.parse_args(argv)
    d = DEFAULTS[a.ruleset]
    params = {"samples": a.samples or d["samples"], "flops": a.flops or d["flops"], "inner": a.inner or d["inner"],
              "seed": a.seed}
    data = build(a.ruleset, jobs=a.jobs, **params)
    seconds = data.pop("_seconds")
    from pathlib import Path

    path = store.write(table_name(a.ruleset), data, generator=GENERATOR, inputs=INPUTS,
                       params={"ruleset_id": a.ruleset, **params}, seconds=seconds,
                       directory=Path(a.out) if a.out else None)
    print(f"wrote {path} ({path.stat().st_size} bytes, {len(data['keys'])} classes, {seconds:.1f}s)")


if __name__ == "__main__":
    main()
