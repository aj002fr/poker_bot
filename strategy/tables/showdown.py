"""All-in showdown tables per variant: heads-up class matrices and three-way bucket tensors.

Built through the rules adapter: the variant's deck, hole-card count and hand ranking
come from :mod:`strategy.rules.variants` (via :func:`strategy.endgame.classes.hand_classes`),
so every variant gets its own tables and a Hold'em table can never stand in for another
game (the loader checks the variant hash).

Arrays (``showdown-<ruleset>.npz``):

* ``w2[i, j]``: exact number of card-disjoint combo pairs with hero in class ``i`` and
  villain in class ``j`` (so ``w2[i] / w2[i].sum()`` is the villain class distribution
  given hero's class, with card removal).
* ``p_win[i, j]``, ``p_tie[i, j]``: heads-up all-in win and tie probability, by seeded
  Monte Carlo over uniformly random compatible combo pairs and boards (``hu_samples`` per
  unordered cell; ``p_win[j, i] = 1 - p_win[i, j] - p_tie[i, j]``).
* ``eq_random[i]``: equity against one random hand; ``bucket[i]``: equal-combo-mass
  strength bucket of class ``i`` (``buckets`` of them).
* ``w3[a, b, c]``: joint probability that three dealt hands fall in buckets ``a, b, c``
  (card-removal aware); ``o3[a, b, c, k]``: probability of weak ordering ``codes3[k]``.

    python -m strategy.tables.showdown --ruleset practice_nlhe --jobs 8
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

from strategy.endgame.classes import hand_classes
from strategy.tables import store

GENERATOR = "strategy.tables.showdown"
INPUTS = ["features/cards.py", "endgame/classes.py", "endgame/kernels.py", "tables/showdown.py",
          "rules/variants.py"]
DEFAULTS = {"hu_samples": 20000, "tri_samples": 4000, "buckets": 20, "seed": 1}


def table_name(ruleset_id: str) -> str:
    return f"showdown-{ruleset_id}"


def outcome_code(values) -> int:
    """Python twin of ``kernels._outcome_code``: base-n digits of 'players strictly better'."""
    n = len(values)
    code = 0
    for v in values:
        code = code * n + sum(1 for w in values if w > v)
    return code


def weak_ordering_codes(n: int) -> list[int]:
    return sorted({outcome_code(v) for v in itertools.product(range(n), repeat=n)})


def _combo_arrays(groups):
    offsets = [0]
    flat = []
    for g in groups:
        flat.extend(g)
        offsets.append(len(flat))
    return np.array(flat, np.int64).reshape(-1, 2), np.array(offsets, np.int64)


def exact_pair_weights(combos) -> np.ndarray:
    flat, offsets = _combo_arrays(combos)
    masks = (np.uint64(1) << flat[:, 0].astype(np.uint64)) | (np.uint64(1) << flat[:, 1].astype(np.uint64))
    ok = (masks[:, None] & masks[None, :]) == 0
    owner = np.repeat(np.arange(len(combos)), np.diff(offsets))
    onehot = np.zeros((len(flat), len(combos)), np.int64)
    onehot[np.arange(len(flat)), owner] = 1
    return (onehot.T @ ok.astype(np.int64) @ onehot).astype(np.int64)


def build(ruleset_id: str, *, hu_samples: int, tri_samples: int, buckets: int, seed: int) -> dict:
    from strategy.endgame import kernels

    started = time.perf_counter()
    hc = hand_classes(ruleset_id)
    caps = hc.caps
    h = hc.count
    sh = np.array(caps.straight_high, np.int64)
    st = np.array(caps.strength, np.int64)
    deck = np.array(caps.deck, np.int64)
    flat, offsets = _combo_arrays(hc.combos)
    w2 = exact_pair_weights(hc.combos)

    iu, ju = np.triu_indices(h)
    wins, ties, acc, _ = kernels.heads_up_cells(iu.astype(np.int64), ju.astype(np.int64), flat, offsets, deck, sh,
                                                st, hu_samples, seed)
    if (acc < hu_samples).any():
        raise RuntimeError("a heads-up cell could not draw compatible combos")
    pw = np.zeros((h, h))
    pt = np.zeros((h, h))
    pw[iu, ju] = wins / acc
    pt[iu, ju] = ties / acc
    lower = iu != ju
    pw[ju[lower], iu[lower]] = 1.0 - pw[iu[lower], ju[lower]] - pt[iu[lower], ju[lower]]
    pt[ju[lower], iu[lower]] = pt[iu[lower], ju[lower]]
    diag = iu[~lower]
    # Same class: the two seats are exchangeable, so win = lose exactly.
    pw[diag, diag] = (1.0 - pt[diag, diag]) / 2
    eq = (w2 * (pw + pt / 2)).sum(1) / w2.sum(1)

    counts = np.array(hc.combo_counts, np.float64)
    order = np.lexsort((np.arange(h), eq))
    cum = 0.0
    bucket = np.zeros(h, np.int64)
    for i in order:
        mid = (cum + counts[i] / 2) / counts.sum()
        bucket[i] = min(buckets - 1, int(mid * buckets))
        cum += counts[i]
    b_groups = [[c for i in range(h) if bucket[i] == b for c in hc.combos[i]] for b in range(buckets)]
    if any(not g for g in b_groups):
        raise ValueError(f"{buckets} buckets leave an empty bucket for {ruleset_id}")
    b_flat, b_offsets = _combo_arrays(b_groups)
    cells = np.array(list(itertools.product(range(buckets), repeat=3)), np.int64)
    c3, acc3, tried3 = kernels.three_way_cells(cells, b_flat, b_offsets, deck, sh, st, tri_samples, seed + 1)
    if (acc3 < tri_samples).any():
        raise RuntimeError("a three-way cell could not draw compatible combos")
    codes = weak_ordering_codes(3)
    o3 = (c3[:, codes] / acc3[:, None]).reshape(buckets, buckets, buckets, len(codes))
    mass = np.array([len(g) for g in b_groups], np.float64)
    w3 = (mass[cells[:, 0]] * mass[cells[:, 1]] * mass[cells[:, 2]] * (acc3 / tried3)).reshape((buckets,) * 3)
    w3 /= w3.sum()
    return {
        "_seconds": time.perf_counter() - started,
        "class_keys": np.array(hc.keys, np.int64),
        "class_labels": np.array(hc.labels),
        "class_combos": counts.astype(np.int32),
        "w2": w2.astype(np.int32),
        "p_win": pw.astype(np.float32),
        "p_tie": pt.astype(np.float32),
        "eq_random": eq.astype(np.float32),
        "bucket": bucket.astype(np.int16),
        "w3": w3,
        "o3": o3.astype(np.float32),
        "codes3": np.array(codes, np.int16),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruleset", required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=v)
    ap.add_argument("--jobs", type=int, default=0, help="numba threads (0: all cores)")
    ap.add_argument("--out", help="output directory (default: strategy/tables/data)")
    a = ap.parse_args(argv)
    if a.jobs:
        import numba

        numba.set_num_threads(a.jobs)
    params = {k: getattr(a, k) for k in DEFAULTS}
    data = build(a.ruleset, **params)
    seconds = data.pop("_seconds")
    hc = hand_classes(a.ruleset)
    path = store.write_arrays(table_name(a.ruleset), data, generator=GENERATOR, inputs=INPUTS,
                              params={"ruleset_id": a.ruleset, **params}, seconds=seconds,
                              extra={"ruleset_id": a.ruleset, "variant_hash": hc.variant_hash},
                              directory=Path(a.out) if a.out else None)
    print(f"wrote {path} ({path.stat().st_size} bytes, {hc.count} classes, {seconds:.1f}s)")


if __name__ == "__main__":
    main()
