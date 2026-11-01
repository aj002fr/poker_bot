"""Point utilities for endgame subgames: exact ICM, eliminations, all-in settlement.

Offline twins of :mod:`arena.tournament.scoring` and the arena's pot rules, written
for speed (vectorised ICM over many stack vectors) and checked against the arena
implementations by ``tests/strategy/test_endgame.py``:

* :func:`icm_batch` equals ``arena.tournament.scoring.icm`` (Malmuth-Harville).
* :func:`eliminations` follows ``Standings.record_eliminations``: busted players are
  ranked by their stack at the start of the hand, equal stacks share their places.
* :func:`settle` follows spec settlement: one pot per contribution level, eligible
  showdown players only, odd chips one each in ``odd_order`` (clockwise from the first
  seat after the button), uncalled excess returned.

The decision-time twin of ICM is :func:`strategy.endgame.keys.icm_player`.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _masks(n: int) -> tuple[np.ndarray, ...]:
    masks = np.arange(1 << n)
    pop = np.array([bin(m).count("1") for m in masks])
    members = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)
    order = np.argsort(pop, kind="stable")
    return masks, pop, members, order


def icm_batch(stacks: np.ndarray, prizes: Sequence[float]) -> np.ndarray:
    """Exact Malmuth-Harville ICM for each row of ``stacks`` (shape ``(m, n)``, positive).

    ``prizes[k]`` pays finishing place ``k + 1`` among the ``n`` players.
    """
    s = np.asarray(stacks, dtype=np.float64)
    if s.ndim == 1:
        return icm_batch(s[None, :], prizes)[0]
    m, n = s.shape
    if n == 0:
        return np.zeros((m, 0))
    if n > 12:
        raise ValueError("icm_batch supports up to 12 players")
    if (s <= 0).any():
        raise ValueError("ICM stacks must be positive")
    prize = np.zeros(n)
    k = min(n, len(prizes))
    prize[:k] = np.asarray(prizes, dtype=np.float64)[:k]
    paid = int(np.max(np.nonzero(prize)[0]) + 1) if prize.any() else 0
    if paid == 0:
        return np.zeros((m, n))
    _, pop, members, order = _masks(n)
    full = (1 << n) - 1
    # e[mask] is the expected prize of every player given the players in ``mask`` are
    # still competing for places n - |mask| + 1 onward. Masks whose next place pays
    # nothing (and all smaller masks) contribute zero.
    e = np.zeros((1 << n, m, n))
    totals = s @ members.T.astype(np.float64)  # (m, masks)
    for mask in order:
        size = pop[mask]
        if size == 0:
            continue
        place = n - size
        if place >= paid:
            continue
        idx = np.nonzero(members[mask])[0]
        tot = totals[:, mask]
        acc = np.zeros((m, n))
        for w in idx:
            p = s[:, w] / tot
            acc[:, w] += p * prize[place]
            acc += p[:, None] * e[mask & ~(1 << w)]
        e[mask] = acc
    return e[full]


def icm(stacks: Sequence[int], prizes: Sequence[float]) -> list[float]:
    return [float(x) for x in icm_batch(np.asarray(stacks, dtype=np.float64), prizes)]


def eliminations(start: Sequence[int], end: Sequence[int], payouts: Sequence[float],
                 n_alive: int) -> dict[int, float]:
    """Points for players who end the hand with zero chips (indices into ``start``).

    ``n_alive`` players held finishing places ``1..n_alive`` before the hand.
    """
    busted = [i for i in range(len(start)) if end[i] == 0]
    if not busted:
        return {}
    busted.sort(key=lambda i: -start[i])
    out: dict[int, float] = {}
    position = n_alive - len(busted) + 1
    i = 0
    while i < len(busted):
        j = i
        while j < len(busted) and start[busted[j]] == start[busted[i]]:
            j += 1
        group = busted[i:j]
        places = range(position, position + len(group))
        share = sum(float(payouts[p - 1]) for p in places) / len(group)
        for b in group:
            out[b] = share
        position += len(group)
        i = j
    return out


def settle(contrib: Sequence[int], folded: Sequence[bool], strength: Sequence[int],
           odd_order: Sequence[int]) -> list[int]:
    """Chips each player receives from the pot.

    ``strength`` is compared only among non-folded players (higher wins). Each pot
    layer goes to the best eligible hands (non-folded players who contributed to that
    level); a layer with no eligible player (dead money above every live contribution)
    goes to the best hands of the highest live layer.
    """
    n = len(contrib)
    awards = [0] * n
    levels = sorted({c for c in contrib if c > 0})
    rank = {p: i for i, p in enumerate(odd_order)}
    prev = 0
    last_winners: list[int] | None = None
    carry = 0
    for level in levels:
        amount = sum(min(max(c - prev, 0), level - prev) for c in contrib)
        eligible = [p for p in range(n) if not folded[p] and contrib[p] >= level]
        prev = level
        if not eligible:
            carry += amount
            continue
        best = max(strength[p] for p in eligible)
        winners = sorted((p for p in eligible if strength[p] == best), key=lambda p: rank[p])
        share, rem = divmod(amount, len(winners))
        for k, w in enumerate(winners):
            awards[w] += share + (1 if k < rem else 0)
        last_winners = winners
    if carry:
        if last_winners is None:
            live = [p for p in range(n) if not folded[p]]
            best = max(strength[p] for p in live)
            last_winners = sorted((p for p in live if strength[p] == best), key=lambda p: rank[p])
        share, rem = divmod(carry, len(last_winners))
        for k, w in enumerate(last_winners):
            awards[w] += share + (1 if k < rem else 0)
    return awards
