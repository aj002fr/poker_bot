"""Numba kernels over `GameTree` arrays. No Python objects on these paths.

The fault codes mirror `strategy.solve.faults.Fault`; they are compiled into
the kernels so mutation tests exercise the same code that production runs.
"""

from __future__ import annotations

import numpy as np
from numba import njit

TERMINAL = -2
CHANCE = -1

NONE = 0
LAST_ITERATE = 1
REGRET_OWN_REACH = 2
AVERAGE_UNWEIGHTED = 3
PAYOFF_ORIENTATION = 4
IGNORE_CHANCE_REACH = 5
RESET_REACH_AT_CHANCE = 6
ES_UNIFORM_OPPONENT = 7
ES_UNIFORM_CHANCE = 8
ES_NO_EXPLORATION = 9


@njit(cache=True)
def edge_weights(node_slot, node_chance_prob, sigma, out):
    out[0] = 1.0
    for i in range(1, node_slot.shape[0]):
        s = node_slot[i]
        if s >= 0:
            out[i] = sigma[s]
        else:
            out[i] = node_chance_prob[i]


@njit(cache=True)
def forward_reach(node_player, node_parent, edge_w, fault, reach):
    """``reach[i, p]`` for players p < P and ``reach[i, P]`` for chance."""
    n = reach.shape[0]
    k = reach.shape[1]
    chance_col = k - 1
    for j in range(k):
        reach[0, j] = 1.0
    for i in range(1, n):
        par = node_parent[i]
        owner = node_player[par]
        for j in range(k):
            reach[i, j] = reach[par, j]
        if owner >= 0:
            reach[i, owner] *= edge_w[i]
        else:
            if fault == RESET_REACH_AT_CHANCE and par != 0:
                for j in range(chance_col):
                    reach[i, j] = 1.0
            reach[i, chance_col] *= edge_w[i]


@njit(cache=True)
def backward_values(node_parent, node_terminal, terminal_utility, edge_w, values):
    """Expected utility of every node under the edge weights."""
    n = values.shape[0]
    num_players = values.shape[1]
    for i in range(n):
        for j in range(num_players):
            values[i, j] = 0.0
    for i in range(n - 1, -1, -1):
        t = node_terminal[i]
        if t >= 0:
            for j in range(num_players):
                values[i, j] = terminal_utility[t, j]
        if i > 0:
            w = edge_w[i]
            if w != 0.0:
                par = node_parent[i]
                for j in range(num_players):
                    values[par, j] += w * values[i, j]


@njit(cache=True)
def accumulate_regrets(
    node_player,
    node_infoset,
    first_child,
    num_children,
    infoset_offset,
    reach,
    values,
    sigma,
    player,
    avg_weight,
    fault,
    regret,
    strat_sum,
):
    """Add counterfactual regret and reach-weighted strategy for ``player`` (-1: all)."""
    k = reach.shape[1]
    chance_col = k - 1
    for h in range(node_player.shape[0]):
        p = node_player[h]
        if p < 0:
            continue
        if player >= 0 and p != player:
            continue
        own = reach[h, p]
        cf = 1.0
        for j in range(k):
            if j != p:
                if j == chance_col and fault == IGNORE_CHANCE_REACH:
                    continue
                cf *= reach[h, j]
        if fault == REGRET_OWN_REACH:
            cf = own
        if cf == 0.0 and own == 0.0:
            continue
        col = 0 if fault == PAYOFF_ORIENTATION else p
        off = infoset_offset[node_infoset[h]]
        fc = first_child[h]
        vh = values[h, col]
        for a in range(num_children[h]):
            regret[off + a] += cf * (values[fc + a, col] - vh)
            if avg_weight != 0.0:
                if fault == AVERAGE_UNWEIGHTED:
                    strat_sum[off + a] += avg_weight * sigma[off + a]
                else:
                    strat_sum[off + a] += avg_weight * own * sigma[off + a]


@njit(cache=True)
def regret_matching(regret, infoset_offset, infoset_player, player, sigma):
    """Positive-regret normalisation into ``sigma``; uniform when none is positive."""
    for i in range(infoset_player.shape[0]):
        if player >= 0 and infoset_player[i] != player:
            continue
        a0 = infoset_offset[i]
        a1 = infoset_offset[i + 1]
        total = 0.0
        for s in range(a0, a1):
            if regret[s] > 0.0:
                total += regret[s]
        if total > 0.0:
            for s in range(a0, a1):
                sigma[s] = regret[s] / total if regret[s] > 0.0 else 0.0
        else:
            u = 1.0 / (a1 - a0)
            for s in range(a0, a1):
                sigma[s] = u


@njit(cache=True)
def update_cumulative_regret(regret, slot_player, player, floor, pos_factor, neg_factor):
    """Discount (DCFR/linear) and optionally floor (CFR+) ``player``'s regrets."""
    for s in range(regret.shape[0]):
        if player >= 0 and slot_player[s] != player:
            continue
        r = regret[s]
        if r > 0.0:
            r *= pos_factor
        elif r < 0.0:
            r *= neg_factor
            if floor:
                r = 0.0
        regret[s] = r


@njit(cache=True)
def best_response_level(
    node_player,
    node_infoset,
    first_child,
    num_children,
    infoset_offset,
    infoset_player,
    infoset_level,
    reach,
    values,
    br_player,
    level,
    q,
    sigma,
):
    """Choose pure best responses for ``br_player``'s infosets at ``level``.

    ``values`` must already reflect best responses at deeper levels.
    """
    k = reach.shape[1]
    for h in range(node_player.shape[0]):
        if node_player[h] != br_player:
            continue
        iset = node_infoset[h]
        if infoset_level[iset] != level:
            continue
        cf = 1.0
        for j in range(k):
            if j != br_player:
                cf *= reach[h, j]
        if cf == 0.0:
            continue
        off = infoset_offset[iset]
        fc = first_child[h]
        for a in range(num_children[h]):
            q[off + a] += cf * values[fc + a, br_player]
    for i in range(infoset_level.shape[0]):
        if infoset_player[i] != br_player or infoset_level[i] != level:
            continue
        a0 = infoset_offset[i]
        a1 = infoset_offset[i + 1]
        best = a0
        for s in range(a0, a1):
            if q[s] > q[best] + 1e-12:
                best = s
        for s in range(a0, a1):
            sigma[s] = 1.0 if s == best else 0.0


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def _sample(probs, k):
    u = np.random.random()
    acc = 0.0
    for a in range(k):
        acc += probs[a]
        if u < acc:
            return a
    for a in range(k - 1, -1, -1):
        if probs[a] > 0.0:
            return a
    return k - 1


# Not disk-cached: numba's cache of recursive functions segfaults on reload.
@njit
def _es_walk(
    h,
    traverser,
    depth,
    node_player,
    node_infoset,
    first_child,
    num_children,
    node_chance_cdf,
    node_terminal,
    terminal_utility,
    infoset_offset,
    regret,
    strat_sum,
    fixed_sigma,
    use_fixed,
    regret_weight,
    avg_weight,
    fault,
    scratch_sigma,
    scratch_vals,
    counter,
):
    counter[0] += 1
    p = node_player[h]
    if p == TERMINAL:
        return terminal_utility[node_terminal[h], traverser]
    fc = first_child[h]
    k = num_children[h]
    if p == CHANCE:
        u = np.random.random()
        if fault == ES_UNIFORM_CHANCE:
            c = int(u * k)
            if c >= k:
                c = k - 1
        else:
            lo = 0
            hi = k - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if node_chance_cdf[fc + mid] > u:
                    hi = mid
                else:
                    lo = mid + 1
            c = lo
        return _es_walk(
            fc + c, traverser, depth + 1, node_player, node_infoset, first_child, num_children,
            node_chance_cdf, node_terminal, terminal_utility, infoset_offset, regret, strat_sum,
            fixed_sigma, use_fixed, regret_weight, avg_weight, fault, scratch_sigma, scratch_vals, counter,
        )
    off = infoset_offset[node_infoset[h]]
    sig = scratch_sigma[depth]
    if use_fixed:
        for a in range(k):
            sig[a] = fixed_sigma[off + a]
    else:
        total = 0.0
        for a in range(k):
            r = regret[off + a]
            if r > 0.0:
                total += r
        for a in range(k):
            r = regret[off + a]
            if total > 0.0:
                sig[a] = r / total if r > 0.0 else 0.0
            else:
                sig[a] = 1.0 / k
    if p != traverser:
        if avg_weight != 0.0:
            for a in range(k):
                strat_sum[off + a] += avg_weight * sig[a]
        if fault == ES_UNIFORM_OPPONENT:
            a = int(np.random.random() * k)
            if a >= k:
                a = k - 1
        else:
            a = _sample(sig, k)
        return _es_walk(
            fc + a, traverser, depth + 1, node_player, node_infoset, first_child, num_children,
            node_chance_cdf, node_terminal, terminal_utility, infoset_offset, regret, strat_sum,
            fixed_sigma, use_fixed, regret_weight, avg_weight, fault, scratch_sigma, scratch_vals, counter,
        )
    if fault == ES_NO_EXPLORATION:
        a = _sample(sig, k)
        v = _es_walk(
            fc + a, traverser, depth + 1, node_player, node_infoset, first_child, num_children,
            node_chance_cdf, node_terminal, terminal_utility, infoset_offset, regret, strat_sum,
            fixed_sigma, use_fixed, regret_weight, avg_weight, fault, scratch_sigma, scratch_vals, counter,
        )
        for b in range(k):
            indicator = 1.0 if b == a else 0.0
            regret[off + b] += regret_weight * (indicator - sig[b]) * v
        return v
    vals = scratch_vals[depth]
    v = 0.0
    for a in range(k):
        va = _es_walk(
            fc + a, traverser, depth + 1, node_player, node_infoset, first_child, num_children,
            node_chance_cdf, node_terminal, terminal_utility, infoset_offset, regret, strat_sum,
            fixed_sigma, use_fixed, regret_weight, avg_weight, fault, scratch_sigma, scratch_vals, counter,
        )
        vals[a] = va
        v += sig[a] * va
    for a in range(k):
        regret[off + a] += regret_weight * (vals[a] - v)
    return v


@njit
def es_iterations(
    start_iteration,
    num_iterations,
    players,
    linear,
    node_player,
    node_infoset,
    first_child,
    num_children,
    node_chance_cdf,
    node_terminal,
    terminal_utility,
    infoset_offset,
    regret,
    strat_sum,
    fault,
    scratch_sigma,
    scratch_vals,
    counter,
):
    """Run external-sampling MCCFR iterations; one traversal per active player each."""
    dummy = np.zeros(1)
    for it in range(num_iterations):
        t = start_iteration + it + 1
        w = float(t) if linear else 1.0
        for i in range(players.shape[0]):
            _es_walk(
                0, players[i], 0, node_player, node_infoset, first_child, num_children,
                node_chance_cdf, node_terminal, terminal_utility, infoset_offset, regret, strat_sum,
                dummy, False, w, w, fault, scratch_sigma, scratch_vals, counter,
            )


@njit
def es_regret_samples(
    num_samples,
    traverser,
    node_player,
    node_infoset,
    first_child,
    num_children,
    node_chance_cdf,
    node_terminal,
    terminal_utility,
    infoset_offset,
    fixed_sigma,
    fault,
    scratch_sigma,
    scratch_vals,
    total,
    total_sq,
    counter,
):
    """Sum and sum of squares of single-traversal sampled regrets under a fixed policy."""
    inc = np.zeros(total.shape[0])
    dummy = np.zeros(1)
    for _ in range(num_samples):
        inc[:] = 0.0
        _es_walk(
            0, traverser, 0, node_player, node_infoset, first_child, num_children,
            node_chance_cdf, node_terminal, terminal_utility, infoset_offset, inc, dummy,
            fixed_sigma, True, 1.0, 0.0, fault, scratch_sigma, scratch_vals, counter,
        )
        for s in range(inc.shape[0]):
            total[s] += inc[s]
            total_sq[s] += inc[s] * inc[s]
