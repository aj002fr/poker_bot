"""Numba kernels for the endgame generators (offline only).

The evaluator is a line-by-line port of :func:`strategy.features.cards.evaluate_any`
and returns the same integers; it is parameterised by the variant's straight table and
category strengths, so short deck and other best-five variants share one code path.
Monte Carlo estimators draw from a splitmix64 stream seeded per cell, so results do not
depend on thread scheduling.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Category ids as in strategy.features.cards.
HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)


@njit(cache=True, inline="always")
def _pack(strength, t0, t1, t2, t3, t4):
    return (strength << 20) | (t0 << 16) | (t1 << 12) | (t2 << 8) | (t3 << 4) | t4


@njit(cache=True)
def evaluate7(cards, n, straight_high, strength):
    """Best five of ``cards[:n]`` (hold'em family); identical to ``evaluate_any``."""
    counts = np.zeros(13, np.int64)
    sm = np.zeros(4, np.int64)
    sc = np.zeros(4, np.int64)
    mask = 0
    for i in range(n):
        c = cards[i]
        r = c >> 2
        s = c & 3
        counts[r] += 1
        sm[s] |= 1 << r
        sc[s] += 1
        mask |= 1 << r
    quads = -1
    t0 = -1
    t1 = -1
    p0 = -1
    p1 = -1
    p2 = -1
    singles = np.full(5, 0, np.int64)
    ns = 0
    for r in range(12, -1, -1):
        k = counts[r]
        if k == 1:
            if ns < 5:
                singles[ns] = r
            ns += 1
        elif k == 2:
            if p0 < 0:
                p0 = r
            elif p1 < 0:
                p1 = r
            elif p2 < 0:
                p2 = r
        elif k == 3:
            if t0 < 0:
                t0 = r
            elif t1 < 0:
                t1 = r
        elif k >= 4:
            if quads < 0:
                quads = r
    if quads >= 0:
        kick = 0
        for r in range(12, -1, -1):
            if r != quads and counts[r] > 0:
                kick = r
                break
        best = _pack(strength[QUADS], quads, kick, 0, 0, 0)
    elif t0 >= 0 and (t1 >= 0 or p0 >= 0):
        p = max(t1, p0)
        best = _pack(strength[FULL_HOUSE], t0, p, 0, 0, 0)
    elif straight_high[mask] >= 0:
        best = _pack(strength[STRAIGHT], straight_high[mask], 0, 0, 0, 0)
    elif t0 >= 0:
        best = _pack(strength[TRIPS], t0, singles[0] if ns > 0 else 0, singles[1] if ns > 1 else 0, 0, 0)
    elif p1 >= 0:
        kick = max(p2, singles[0] if ns > 0 else -1)
        best = _pack(strength[TWO_PAIR], p0, p1, max(kick, 0), 0, 0)
    elif p0 >= 0:
        best = _pack(strength[PAIR], p0, singles[0] if ns > 0 else 0, singles[1] if ns > 1 else 0,
                     singles[2] if ns > 2 else 0, 0)
    else:
        best = _pack(strength[HIGH_CARD], singles[0] if ns > 0 else 0, singles[1] if ns > 1 else 0,
                     singles[2] if ns > 2 else 0, singles[3] if ns > 3 else 0, singles[4] if ns > 4 else 0)
    for s in range(4):
        if sc[s] >= 5:
            m = sm[s]
            if straight_high[m] >= 0:
                fv = _pack(strength[STRAIGHT_FLUSH], straight_high[m], 0, 0, 0, 0)
            else:
                top = np.zeros(5, np.int64)
                k = 0
                for r in range(12, -1, -1):
                    if (m >> r) & 1 and k < 5:
                        top[k] = r
                        k += 1
                fv = _pack(strength[FLUSH], top[0], top[1], top[2], top[3], top[4])
            if fv > best:
                best = fv
    return best


@njit(cache=True)
def evaluate_many(hands, straight_high, strength):
    out = np.empty(hands.shape[0], np.int64)
    for i in range(hands.shape[0]):
        out[i] = evaluate7(hands[i], hands.shape[1], straight_high, strength)
    return out


@njit(cache=True, inline="always")
def _splitmix(state):
    state = (state + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = state
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return state, z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def _below(state, n):
    state, z = _splitmix(state)
    return state, np.int64(z % np.uint64(n))


@njit(cache=True)
def _draw_board(state, deck, used, board):
    """Five distinct cards from ``deck`` avoiding ``used`` (a 52-flag array); flags are restored."""
    k = 0
    nd = deck.shape[0]
    while k < 5:
        state, j = _below(state, nd)
        c = deck[j]
        if used[c] == 0:
            used[c] = 1
            board[k] = c
            k += 1
    for i in range(5):
        used[board[i]] = 0
    return state


@njit(cache=True, parallel=True)
def heads_up_cells(pairs_i, pairs_j, class_combos, combo_offsets, deck, straight_high, strength, samples, seed):
    """Win and tie counts for each ordered class pair ``(pairs_i[k], pairs_j[k])``.

    Every sample draws a uniformly random compatible combo pair (rejection on shared
    cards) and a uniformly random board from the remaining deck.
    Returns ``(wins, ties, accepted, tried)`` per cell.
    """
    m = pairs_i.shape[0]
    wins = np.zeros(m, np.int64)
    ties = np.zeros(m, np.int64)
    accepted = np.zeros(m, np.int64)
    tried = np.zeros(m, np.int64)
    for k in prange(m):
        state = np.uint64(seed) * np.uint64(1000003) + np.uint64(k) * np.uint64(7919) + np.uint64(1)
        ci = pairs_i[k]
        cj = pairs_j[k]
        ni = combo_offsets[ci + 1] - combo_offsets[ci]
        nj = combo_offsets[cj + 1] - combo_offsets[cj]
        used = np.zeros(52, np.int64)
        board = np.zeros(5, np.int64)
        h = np.zeros(7, np.int64)
        w = 0
        t = 0
        acc = 0
        tr = 0
        while acc < samples:
            state, a = _below(state, ni)
            state, b = _below(state, nj)
            a0 = class_combos[combo_offsets[ci] + a, 0]
            a1 = class_combos[combo_offsets[ci] + a, 1]
            b0 = class_combos[combo_offsets[cj] + b, 0]
            b1 = class_combos[combo_offsets[cj] + b, 1]
            tr += 1
            if a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1:
                if tr > 1000 * samples:
                    break
                continue
            acc += 1
            used[a0] = 1
            used[a1] = 1
            used[b0] = 1
            used[b1] = 1
            state = _draw_board(state, deck, used, board)
            used[a0] = 0
            used[a1] = 0
            used[b0] = 0
            used[b1] = 0
            for q in range(5):
                h[q] = board[q]
            h[5] = a0
            h[6] = a1
            va = evaluate7(h, 7, straight_high, strength)
            h[5] = b0
            h[6] = b1
            vb = evaluate7(h, 7, straight_high, strength)
            if va > vb:
                w += 1
            elif va == vb:
                t += 1
        wins[k] = w
        ties[k] = t
        accepted[k] = acc
        tried[k] = tr
    return wins, ties, accepted, tried


@njit(cache=True)
def _outcome_code(v, n):
    """Weak ordering of ``n`` hand values as base-``n`` digits of 'players strictly better'."""
    code = 0
    for i in range(n):
        better = 0
        for j in range(n):
            if v[j] > v[i]:
                better += 1
        code = code * n + better
    return code


@njit(cache=True, parallel=True)
def three_way_cells(cells, bucket_combos, bucket_offsets, deck, straight_high, strength, samples, seed):
    """Outcome counts (27 base-3 weak-ordering codes) for ordered bucket triples ``cells[k]``.

    Returns ``(counts[m, 27], accepted, tried)``; ``accepted / tried`` estimates the
    no-shared-card probability that makes the joint bucket weight card-removal aware.
    """
    m = cells.shape[0]
    counts = np.zeros((m, 27), np.int64)
    accepted = np.zeros(m, np.int64)
    tried = np.zeros(m, np.int64)
    for k in prange(m):
        state = np.uint64(seed) * np.uint64(1000033) + np.uint64(k) * np.uint64(104729) + np.uint64(3)
        used = np.zeros(52, np.int64)
        board = np.zeros(5, np.int64)
        h = np.zeros(7, np.int64)
        v = np.zeros(3, np.int64)
        hole = np.zeros(6, np.int64)
        acc = 0
        tr = 0
        while acc < samples:
            tr += 1
            ok = True
            for p in range(3):
                b = cells[k, p]
                n = bucket_offsets[b + 1] - bucket_offsets[b]
                state, a = _below(state, n)
                c0 = bucket_combos[bucket_offsets[b] + a, 0]
                c1 = bucket_combos[bucket_offsets[b] + a, 1]
                if used[c0] or used[c1]:
                    ok = False
                used[c0] = 1
                used[c1] = 1
                hole[2 * p] = c0
                hole[2 * p + 1] = c1
            if not ok:
                for q in range(52):
                    used[q] = 0
                if tr > 1000 * samples:
                    break
                continue
            acc += 1
            state = _draw_board(state, deck, used, board)
            for q in range(6):
                used[hole[q]] = 0
            for q in range(5):
                h[q] = board[q]
            for p in range(3):
                h[5] = hole[2 * p]
                h[6] = hole[2 * p + 1]
                v[p] = evaluate7(h, 7, straight_high, strength)
            counts[k, _outcome_code(v, 3)] += 1
        accepted[k] = acc
        tried[k] = tr
    return counts, accepted, tried
