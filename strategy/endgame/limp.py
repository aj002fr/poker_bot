"""Limp and min-raise branches for blind-vs-blind spots, and the measured cost of pruning them.

The push/fold tables give the small blind two options when everyone folds to it. This
module solves the richer game and measures what the restriction costs:

* small blind: fold, shove, limp (complete to one big blind) or min-raise (to two);
* after a limp the big blind checks or shoves; after a min-raise it folds, calls or
  shoves; facing a shove the small blind folds or calls;
* a checked or called pot is played out as a check-down to showdown (both players
  realise their full equity), the standard postflop model of limp/shove analyses.

Everyone else has folded: their stacks stay in the point vector (ICM) as bystanders, so
blind-vs-blind spots at a bigger table are point-aware, and heads-up states reduce to
chip EV (points are affine in chips with two players).

:func:`restriction_gap` solves both games with CFR+ over hand classes and reports the
small blind's equilibrium gain from the extra branches, and each game's exact
best-response gains (the solutions' own accuracy). The gap is the regret of pruning
the branches at that depth: the tables keep push/fold only where it is below the
tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strategy.endgame.points import eliminations, icm_batch
from strategy.endgame.showdown import ShowdownModel


@dataclass
class _Node:
    player: int
    """0 small blind, 1 big blind, -1 terminal."""
    labels: list[str] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    contrib: tuple[int, int] = (0, 0)
    folded: int | None = None
    """Terminal: the player who folded, or None for a showdown."""


@dataclass(frozen=True)
class BlindSpot:
    sb_stack: int
    bb_stack: int
    sb_post: int
    bb_post: int
    bb: int
    bystanders: tuple[int, ...] = ()
    payouts: tuple[float, ...] = (65.0, 35.0)
    utility: str = "points"
    """``points`` (ICM over every stack, eliminations by finishing place) or ``chips``."""


def build_tree(spot: BlindSpot, *, extended: bool) -> list[_Node]:
    nodes: list[_Node] = []
    stacks = (spot.sb_stack, spot.bb_stack)

    def new(player: int, contrib, folded=None) -> int:
        nodes.append(_Node(player, [], [], tuple(contrib), folded))
        return len(nodes) - 1

    def add(parent: int, label: str, child: int) -> None:
        nodes[parent].labels.append(label)
        nodes[parent].children.append(child)

    def facing_shove(contrib, shover: int) -> int:
        other = 1 - shover
        n = new(other, contrib)
        add(n, "f", new(-1, contrib, folded=other))
        call = list(contrib)
        call[other] = stacks[other]
        add(n, "c", new(-1, call))
        return n

    root = new(0, (spot.sb_post, spot.bb_post))
    c0 = (spot.sb_post, spot.bb_post)
    add(root, "f", new(-1, c0, folded=0))
    add(root, "s", facing_shove((stacks[0], spot.bb_post), 0))
    if extended and spot.sb_stack > spot.bb_post and spot.bb_stack > spot.bb_post:
        limp = (min(spot.bb_post, stacks[0]), spot.bb_post)
        n = new(1, limp)
        add(n, "x", new(-1, limp))
        add(n, "s", facing_shove((limp[0], stacks[1]), 1))
        add(root, "l", n)
        raise_to = 2 * spot.bb
        if stacks[0] > raise_to and stacks[1] > raise_to:
            mr = (raise_to, spot.bb_post)
            m = new(1, mr)
            add(m, "f", new(-1, mr, folded=1))
            add(m, "c", new(-1, (raise_to, raise_to)))
            add(m, "s", facing_shove((raise_to, stacks[1]), 1))
            add(root, "r", m)
    return nodes


def _settle_two(contrib, winner: int | None, stacks) -> tuple[int, int]:
    """End stacks after a two-player pot; ``winner`` None splits (odd chip to the small blind)."""
    a, b = contrib
    matched = min(a, b)
    pot = 2 * matched
    back = (a - matched, b - matched)
    if winner is None:
        half, odd = divmod(pot, 2)
        win = (half + odd, half)
    else:
        win = (pot, 0) if winner == 0 else (0, pot)
    return (stacks[0] - a + back[0] + win[0], stacks[1] - b + back[1] + win[1])


def _points(spot: BlindSpot, ends: list[tuple[int, int]]) -> np.ndarray:
    start = [spot.sb_stack, spot.bb_stack, *spot.bystanders]
    n = len(start)
    if spot.utility == "chips":
        return np.array([[e[0] - spot.sb_stack, e[1] - spot.bb_stack] for e in ends], np.float64)
    out = np.zeros((len(ends), 2))
    prizes = spot.payouts[:n]
    for r, (e0, e1) in enumerate(ends):
        end = [e0, e1, *spot.bystanders]
        elim = eliminations(start, end, spot.payouts, n)
        alive = [i for i in range(n) if end[i] > 0]
        vals = np.zeros(n)
        for i, pts in elim.items():
            vals[i] = pts
        if len(alive) == 1:
            vals[alive[0]] = spot.payouts[0]
        else:
            vals[alive] = icm_batch(np.array([[end[i] for i in alive]], np.float64), prizes[:len(alive)])[0]
        out[r] = vals[:2]
    return out


class BlindSolver:
    """CFR+ over hand classes for a two-player tree; exact best responses."""

    def __init__(self, spot: BlindSpot, model: ShowdownModel, *, extended: bool) -> None:
        self.spot = spot
        self.nodes = build_tree(spot, extended=extended)
        self.model = model
        self.h = model.h
        self.prior = model.prior
        self.mw, self.mt, self.ml, *_ = model.matrices()
        self.cond = model.cond
        stacks = (spot.sb_stack, spot.bb_stack)
        self.terms = [i for i, nd in enumerate(self.nodes) if nd.player < 0]
        ends = []
        for i in self.terms:
            nd = self.nodes[i]
            if nd.folded is not None:
                ends.append(_settle_two(nd.contrib, 1 - nd.folded, stacks))
            else:
                ends += [_settle_two(nd.contrib, 0, stacks), _settle_two(nd.contrib, None, stacks),
                         _settle_two(nd.contrib, 1, stacks)]
        u = _points(spot, ends)
        self.u: dict[int, np.ndarray] = {}
        k = 0
        for i in self.terms:
            if self.nodes[i].folded is not None:
                self.u[i] = u[k:k + 1]
                k += 1
            else:
                self.u[i] = u[k:k + 3]
                k += 3
        self.decisions = [i for i, nd in enumerate(self.nodes) if nd.player >= 0]
        self.regret = {d: np.zeros((len(self.nodes[d].children), self.h)) for d in self.decisions}
        self.strat_sum = {d: np.zeros_like(self.regret[d]) for d in self.decisions}
        self.sigma = {d: np.full_like(self.regret[d], 1.0 / len(self.nodes[d].children)) for d in self.decisions}
        self.parent = {c: (d, a) for d in self.decisions for a, c in enumerate(self.nodes[d].children)}
        self.iterations = 0

    def reach(self, sigma) -> np.ndarray:
        r = np.ones((len(self.nodes), 2, self.h))
        for idx in range(1, len(self.nodes)):
            d, a = self.parent[idx]
            r[idx] = r[d]
            r[idx, self.nodes[d].player] *= sigma[d][a]
        return r

    def _terminal(self, r: np.ndarray) -> dict[int, np.ndarray]:
        out = {}
        for i in self.terms:
            u = self.u[i]
            v = np.zeros((2, self.h))
            if self.nodes[i].folded is not None:
                v[0] = u[0, 0] * (self.cond @ r[i, 1])
                v[1] = u[0, 1] * (self.cond @ r[i, 0])
            else:
                rb, ra = r[i, 1], r[i, 0]
                v[0] = u[0, 0] * (self.mw @ rb) + u[1, 0] * (self.mt @ rb) + u[2, 0] * (self.ml @ rb)
                v[1] = u[2, 1] * (self.mw @ ra) + u[1, 1] * (self.mt @ ra) + u[0, 1] * (self.ml @ ra)
            out[i] = v
        return out

    def _backward(self, sigma, term) -> np.ndarray:
        v = np.zeros((len(self.nodes), 2, self.h))
        for idx in range(len(self.nodes) - 1, -1, -1):
            nd = self.nodes[idx]
            if nd.player < 0:
                v[idx] = term[idx]
                continue
            kids = v[nd.children]
            v[idx] = kids.sum(0)
            v[idx, nd.player] = (sigma[idx] * kids[:, nd.player]).sum(0)
        return v

    def run(self, iterations: int) -> BlindSolver:
        for _ in range(iterations):
            self.iterations += 1
            r = self.reach(self.sigma)
            v = self._backward(self.sigma, self._terminal(r))
            for d in self.decisions:
                p = self.nodes[d].player
                kids = v[self.nodes[d].children, p]
                reg = self.regret[d]
                reg += kids - v[d, p]
                np.maximum(reg, 0.0, out=reg)
                self.strat_sum[d] += self.iterations * r[d, p] * self.sigma[d]
            for d in self.decisions:
                reg = self.regret[d]
                tot = reg.sum(0)
                s = np.full_like(reg, 1.0 / len(reg))
                pos = tot > 0
                s[:, pos] = reg[:, pos] / tot[pos]
                self.sigma[d] = s
        return self

    def average(self):
        out = {}
        for d in self.decisions:
            s = self.strat_sum[d]
            tot = s.sum(0)
            a = np.full_like(s, 1.0 / len(s))
            pos = tot > 0
            a[:, pos] = s[:, pos] / tot[pos]
            out[d] = a
        return out

    def values(self, sigma) -> np.ndarray:
        v = self._backward(sigma, self._terminal(self.reach(sigma)))
        return v[0] @ self.prior

    def best_response_value(self, sigma, q: int) -> float:
        term = self._terminal(self.reach(sigma))
        v = np.zeros((len(self.nodes), self.h))
        for idx in range(len(self.nodes) - 1, -1, -1):
            nd = self.nodes[idx]
            if nd.player < 0:
                v[idx] = term[idx][q]
            elif nd.player == q:
                v[idx] = v[nd.children].max(0)
            else:
                v[idx] = v[nd.children].sum(0)
        return float(v[0] @ self.prior)

    def gains(self, sigma) -> np.ndarray:
        vals = self.values(sigma)
        return np.array([self.best_response_value(sigma, q) - vals[q] for q in range(2)])

    def root_mix(self, sigma) -> dict[str, float]:
        """Share of small-blind combos taking each root action."""
        root = self.nodes[0]
        return {lab: float(sigma[0][a] @ self.prior) for a, lab in enumerate(root.labels)}


@dataclass(frozen=True)
class Gap:
    spot: BlindSpot
    push_fold_value: float
    extended_value: float
    gap: float
    """Small blind's equilibrium gain from the limp and min-raise branches (utility units)."""
    push_fold_gains: tuple[float, float]
    extended_gains: tuple[float, float]
    root_mix: dict[str, float]


def restriction_gap(spot: BlindSpot, model: ShowdownModel, *, iterations: int = 1500) -> Gap:
    pf = BlindSolver(spot, model, extended=False).run(iterations)
    ext = BlindSolver(spot, model, extended=True).run(iterations)
    sp, se = pf.average(), ext.average()
    vp, ve = float(pf.values(sp)[0]), float(ext.values(se)[0])
    return Gap(spot, vp, ve, ve - vp, tuple(pf.gains(sp).tolist()), tuple(ext.gains(se).tolist()),
               ext.root_mix(se))
