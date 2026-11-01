"""Solve a sampled hand state and harvest its decisions into table keys (offline).

For one start-of-hand :class:`~strategy.endgame.keys.Snapshot` this builds the full
push/fold subgame, solves it with CFR+, and walks the tree for the decisions the
tables cover: every open shove along the all-fold path, every call facing a single
shove, every overcall facing a shove and a call. Each record carries the hero's
aggressive-action probability per hand class and the node's reach (the probability,
under the solved profile, that the other players' actions lead there), which is the
weight the table averages with.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from strategy.endgame.keys import Snapshot, keys_for
from strategy.endgame.showdown import ShowdownModel
from strategy.endgame.solver import PushFoldSolver
from strategy.endgame.subgame import Subgame, SubgameTree, icm_value


@dataclass(frozen=True)
class Record:
    kind: str
    keys: tuple[int, ...]
    hero: int
    weight: float
    row: np.ndarray


@dataclass(frozen=True)
class Solved:
    snap: Snapshot
    tree: SubgameTree
    solver: PushFoldSolver
    sigma: dict[int, np.ndarray]
    gains: np.ndarray
    """Exact unilateral best-response gain of each player against the averaged profile (points)."""


def subgame_for(snap: Snapshot, *, utility: str = "points", odd_order: Sequence[int] | None = None,
                value_fn: Callable | None = None, max_in_pot: int = 3) -> Subgame:
    return Subgame(stacks=snap.stacks, posts=snap.posts, payouts=snap.payouts, bb=snap.bb,
                   odd_order=tuple(odd_order) if odd_order is not None else None,
                   final_hand=snap.hand_number >= snap.hand_cap, max_in_pot=max_in_pot, utility=utility,
                   value_fn=value_fn or icm_value)


def default_iterations(n: int) -> int:
    return 400 if n <= 3 else 250


def solve_snapshot(snap: Snapshot, model: ShowdownModel, *, utility: str = "points", iterations: int | None = None,
                   odd_order: Sequence[int] | None = None, value_fn: Callable | None = None,
                   with_gains: bool = True) -> Solved:
    tree = SubgameTree(subgame_for(snap, utility=utility, odd_order=odd_order, value_fn=value_fn))
    solver = PushFoldSolver(tree, model).run(iterations or default_iterations(snap.n))
    sigma = solver.average()
    gains = solver.deviation_gains(sigma) if with_gains else np.full(snap.n, np.nan)
    return Solved(snap, tree, solver, sigma, gains)


def decision_nodes(tree: SubgameTree):
    """``(kind, node, hero, shover, caller)`` for every covered decision."""
    out = []
    nodes = tree.nodes
    node = 0
    while nodes[node].player >= 0:
        hero = nodes[node].player
        out.append(("open", node, hero, None, None))
        c = nodes[node].children["s"]
        while nodes[c].player >= 0:
            p = nodes[c].player
            out.append(("call", c, p, hero, None))
            oc = nodes[c].children["c"]
            while nodes[oc].player >= 0:
                out.append(("overcall", oc, nodes[oc].player, hero, p))
                oc = nodes[oc].children["f"]
            c = nodes[c].children["f"]
        node = nodes[node].children["f"]
    return out


def harvest(solved: Solved, *, max_depth_bb: float) -> list[Record]:
    snap = solved.snap
    r = solved.solver.reach(solved.sigma)
    prior = solved.solver.prior
    out = []
    for kind, node, hero, shover, caller in decision_nodes(solved.tree):
        if kind == "open":
            behind = range(hero + 1, snap.n)
            risk = min(snap.stacks[hero], max(snap.stacks[p] for p in behind))
        elif kind == "call":
            risk = min(snap.stacks[hero], snap.stacks[shover])
        else:
            risk = min(snap.stacks[hero], max(snap.stacks[shover], snap.stacks[caller]))
        if risk / snap.bb > max_depth_bb:
            continue
        rbar = r[node] @ prior
        weight = float(np.prod(np.delete(rbar, hero)))
        out.append(Record(kind, keys_for(kind, snap, hero, shover, caller), hero, weight,
                          solved.sigma[node][1].copy()))
    return out
