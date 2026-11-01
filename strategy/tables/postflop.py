"""Postflop bucket-equity table per variant.

Cells are ``(street, made level, draw class, texture key)`` from
:func:`strategy.features.cards.postflop_key`, the same functions the decision path
uses. For random hero hands and boards it records, by seeded Monte Carlo, the equity
against 1, 2 and 3 random hands (ties as shares), the heads-up tie share, and an
unbiased estimate of the mean squared hand equity (product of two independent
runouts), which gives the spread of hand equity inside the bucket.

Sparse cells fall back to the coarser ``(street, made, draw)``, ``(street, made)`` and
``(street)`` aggregates; ``src`` records which level filled each cell.

    python -m strategy.tables.postflop --ruleset practice_nlhe --jobs 8
"""

from __future__ import annotations

import argparse
import hashlib
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from strategy.features.cards import (DRAW_CLASSES, MADE_LEVELS, POSTFLOP_CELLS, TEXTURE_KEYS, capabilities, draws,
                                     evaluate, made_hand, postflop_key, texture)
from strategy.tables import store

GENERATOR = "strategy.tables.postflop"
INPUTS = ["features/cards.py", "tables/postflop.py"]
SCALE = 10000
OPPONENTS = 3
MIN_N = 40
STREET_BOARD = {1: 3, 2: 4, 3: 5}
DEFAULTS = {"practice_nlhe": 300000, "uncertainty_shortdeck": 300000, "uncertainty_plo4": 150000}
CHUNK = 5000
STATS = 6  # n, eq1, eq2, eq3, eq1 product, tie1


def table_name(ruleset_id: str) -> str:
    return f"postflop-{ruleset_id}"


def _share(hero: int, others: list[int]) -> float:
    ties = 0
    for v in others:
        if v > hero:
            return 0.0
        if v == hero:
            ties += 1
    return 1.0 / (ties + 1)


def _chunk(args: tuple[str, int, int, int, int]) -> dict[int, list[float]]:
    ruleset_id, street, n, seed, chunk = args
    caps = capabilities(ruleset_id)
    h = hashlib.sha256(f"{GENERATOR}|{ruleset_id}|{street}|{seed}|{chunk}".encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    hc = caps.hole_count
    nb = STREET_BOARD[street]
    runout = 5 - nb
    need = runout + hc * OPPONENTS
    deck = caps.deck
    acc: dict[int, list[float]] = {}
    for _ in range(n):
        drawn = rng.sample(deck, hc + nb)
        hole, board = drawn[:hc], drawn[hc:]
        made = made_hand(hole, board, caps)
        key = postflop_key(street, made.level, draws(hole, board, caps, made).draw_class, texture(board, caps).key)
        taken = set(drawn)
        rest = [c for c in deck if c not in taken]
        shares = []
        for _rep in range(2):
            d = rng.sample(rest, need)
            full = [*board, *d[:runout]]
            hero = evaluate(hole, full, caps)
            opp = [evaluate(d[runout + hc * i:runout + hc * (i + 1)], full, caps) for i in range(OPPONENTS)]
            shares.append((hero, opp))
        (hero, opp), (hero_b, opp_b) = shares
        e1 = _share(hero, opp[:1])
        row = acc.get(key)
        if row is None:
            row = acc[key] = [0.0] * STATS
        row[0] += 1
        row[1] += e1
        row[2] += _share(hero, opp[:2])
        row[3] += _share(hero, opp)
        row[4] += e1 * _share(hero_b, opp_b[:1])
        row[5] += 1.0 if hero == opp[0] else 0.0
    return acc


def _parents(key: int) -> tuple[int, int, int]:
    tex = key % TEXTURE_KEYS
    rest = key // TEXTURE_KEYS
    draw = rest % DRAW_CLASSES
    rest //= DRAW_CLASSES
    made = rest % MADE_LEVELS
    street = rest // MADE_LEVELS
    del tex
    return (street * MADE_LEVELS + made) * DRAW_CLASSES + draw, street * MADE_LEVELS + made, street


def build(ruleset_id: str, *, samples: int, seed: int = 1, jobs: int = 1) -> dict:
    started = time.perf_counter()
    work = []
    for street in (1, 2, 3):
        left, chunk = samples, 0
        while left > 0:
            work.append((ruleset_id, street, min(CHUNK, left), seed, chunk))
            left -= CHUNK
            chunk += 1
    cells = [[0.0] * STATS for _ in range(POSTFLOP_CELLS)]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(_chunk, work))
    else:
        results = [_chunk(w) for w in work]
    for acc in results:
        for key, row in acc.items():
            cell = cells[key]
            for i in range(STATS):
                cell[i] += row[i]
    levels: list[dict[int, list[float]]] = [{}, {}, {}]
    for key, cell in enumerate(cells):
        if not cell[0]:
            continue
        for lvl, pk in enumerate(_parents(key)):
            agg = levels[lvl].setdefault(pk, [0.0] * STATS)
            for i in range(STATS):
                agg[i] += cell[i]
    q = lambda x: int(round(x * SCALE))  # noqa: E731
    out = {"n": [], "src": [], "eq": [[], [], []], "sd": [], "tie": []}
    for key, cell in enumerate(cells):
        src, row = 0, cell
        if row[0] < MIN_N:
            for lvl, pk in enumerate(_parents(key)):
                cand = levels[lvl].get(pk)
                if cand is not None and cand[0] >= MIN_N:
                    src, row = lvl + 1, cand
                    break
            else:
                src, row = 4, None
        if row is None or not row[0]:
            out["n"].append(int(cell[0]))
            out["src"].append(4)
            for k in range(OPPONENTS):
                out["eq"][k].append(-1)
            out["sd"].append(-1)
            out["tie"].append(-1)
            continue
        n = row[0]
        mean = row[1] / n
        var = max(0.0, row[4] / n - mean * mean)
        out["n"].append(int(cell[0]))
        out["src"].append(src)
        for k in range(OPPONENTS):
            out["eq"][k].append(q(row[1 + k] / n))
        out["sd"].append(q(var ** 0.5))
        out["tie"].append(q(row[5] / n))
    return {"ruleset_id": ruleset_id, "scale": SCALE, "opponents": OPPONENTS, "min_n": MIN_N, "cells": POSTFLOP_CELLS,
            **out, "_seconds": time.perf_counter() - started}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruleset", required=True)
    ap.add_argument("--samples", type=int, help="random hands per street")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    params = {"samples": a.samples or DEFAULTS[a.ruleset], "seed": a.seed}
    data = build(a.ruleset, jobs=a.jobs, **params)
    seconds = data.pop("_seconds")
    path = store.write(table_name(a.ruleset), data, generator=GENERATOR, inputs=INPUTS,
                       params={"ruleset_id": a.ruleset, **params}, seconds=seconds,
                       directory=Path(a.out) if a.out else None)
    print(f"wrote {path} ({path.stat().st_size} bytes, {seconds:.1f}s)")


if __name__ == "__main__":
    main()
