"""Hierarchical Dirichlet posteriors over opponents' action choices, by eligible context.

Every public action is one observation of exactly one *opportunity* (what the actor
could do): first in preflop, facing one raise, facing two or more, postflop with no bet,
postflop facing a bet. Each opportunity has three action categories, so a context's
counts are a Dirichlet (a Beta when one category is impossible). An action updates only
the opportunity it was taken in, and only the context cell that matches the actor's
street, position group, depth and field size: counts are never pooled across contexts
that offered different choices.

Predictions shrink hierarchically::

    population prior (fixed, or a family-mixture prior)
      -> table pool (all opponents, by opportunity)
        -> seat (by opportunity)
          -> seat x context cell

with pseudo-count weights ``kappa`` at each step, then a probability floor so no
category ever has zero probability. Counts decay geometrically per hand (recency) and
the posterior reports its effective sample sizes and per-category standard deviations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PRE_UNOPENED, PRE_FACING_RAISE, PRE_FACING_RERAISE, POST_UNBET, POST_FACING_BET = range(5)
OPPORTUNITIES = ("pre_unopened", "pre_facing_raise", "pre_facing_reraise", "post_unbet", "post_facing_bet")
CATEGORIES = {
    PRE_UNOPENED: ("fold", "limp", "raise"),
    PRE_FACING_RAISE: ("fold", "call", "raise"),
    PRE_FACING_RERAISE: ("fold", "call", "raise"),
    POST_UNBET: ("check", "bet_small", "bet_large"),
    POST_FACING_BET: ("fold", "call", "raise"),
}
N_CATEGORIES = 3
BET_LARGE_FRAC = 0.6
"""A postflop first bet above this fraction of the pot is ``bet_large``."""

# Provisional population priors (category probabilities); replaced by a family mixture
# when a family table exists for the variant. Order follows CATEGORIES.
DEFAULT_PRIOR: dict[int, tuple[float, float, float]] = {
    PRE_UNOPENED: (0.70, 0.10, 0.20),
    PRE_FACING_RAISE: (0.70, 0.22, 0.08),
    PRE_FACING_RERAISE: (0.55, 0.30, 0.15),
    POST_UNBET: (0.60, 0.25, 0.15),
    POST_FACING_BET: (0.40, 0.45, 0.15),
}


def opportunity(street: int, raises_before: int, level_before: int) -> int:
    if street == 0:
        return PRE_UNOPENED if raises_before == 0 else PRE_FACING_RAISE if raises_before == 1 else PRE_FACING_RERAISE
    return POST_UNBET if level_before == 0 else POST_FACING_BET


def category(opp: int, kind: str, size_frac: float, to_amount: int, pot_before: int) -> int:
    if opp == POST_UNBET:
        if kind != "raise_to":
            return 0
        return 2 if to_amount > BET_LARGE_FRAC * max(1, pot_before) else 1
    if kind == "fold":
        return 0
    if kind == "raise_to":
        return 2
    return 1


def depth_bucket(eff_bb: float) -> int:
    return 0 if eff_bb < 12 else 1 if eff_bb < 30 else 2


def context_cell(street: int, pos_group: int, eff_bb: float, live_before: int, initiative: bool) -> int:
    """Refinement cell within an opportunity: street x position x depth x multiway x initiative."""
    s = 0 if street == 0 else street - 1
    return (((s * 4 + pos_group) * 3 + depth_bucket(eff_bb)) * 2 + (1 if live_before >= 3 else 0)) * 2 + int(initiative)


@dataclass(frozen=True, slots=True)
class Observation:
    seat: int
    hand_number: int
    opportunity: int
    cell: int
    category: int


def from_record(rec: Any) -> Observation | None:
    """The eligible observation for an :class:`~strategy.features.state.ActionRecord`, or None (automatic)."""
    if rec.auto:
        return None
    opp = opportunity(rec.street, rec.raises_before, rec.level_before)
    cat = category(opp, rec.kind, rec.size_frac, rec.to_amount, rec.pot_before)
    return Observation(rec.seat, rec.hand_number, opp, context_cell(rec.street, rec.pos_group, rec.eff_bb,
                                                                    rec.live_before, rec.initiative), cat)


@dataclass(frozen=True, slots=True)
class Prediction:
    probs: tuple[float, float, float]
    ess_cell: float
    ess_seat: float
    sd: tuple[float, float, float]


class _Counts:
    __slots__ = ("c", "last")

    def __init__(self, c: list[float] | None = None, last: int = 0) -> None:
        self.c = c or [0.0, 0.0, 0.0]
        self.last = last

    def decayed(self, hand: int, decay: float) -> list[float]:
        if decay < 1.0 and hand > self.last:
            f = decay ** (hand - self.last)
            self.c = [x * f for x in self.c]
            self.last = hand
        return self.c


def _shrink(counts: Sequence[float], parent: Sequence[float], kappa: float) -> tuple[float, float, float]:
    n = counts[0] + counts[1] + counts[2]
    d = n + kappa
    return ((counts[0] + kappa * parent[0]) / d, (counts[1] + kappa * parent[1]) / d,
            (counts[2] + kappa * parent[2]) / d)


class ActionPosterior:
    """Per-table hierarchical action model. Serializable within the bot state budget."""

    def __init__(self, *, kappa: tuple[float, float, float] = (10.0, 8.0, 5.0), floor: float = 0.02,
                 decay: float = 1.0, prior: Mapping[int, Sequence[float]] | None = None,
                 exclude_from_pool: Sequence[int] = ()) -> None:
        self.kappa = kappa
        self.floor = floor
        self.decay = decay
        self.prior = {o: tuple(p) for o, p in (prior or DEFAULT_PRIOR).items()}
        self.exclude = set(exclude_from_pool)
        self.pool: dict[int, _Counts] = {}
        self.seat: dict[tuple[int, int], _Counts] = {}
        self.cell: dict[tuple[int, int, int], _Counts] = {}
        self.hand = 0

    def advance(self, hand_number: int) -> None:
        """Set the clock that recency decay is measured against."""
        self.hand = max(self.hand, hand_number)

    def observe(self, ob: Observation | None) -> bool:
        if ob is None:
            return False
        self.hand = max(self.hand, ob.hand_number)
        h, dec = ob.hand_number, self.decay
        targets = [self.seat.setdefault((ob.seat, ob.opportunity), _Counts(last=h)),
                   self.cell.setdefault((ob.seat, ob.opportunity, ob.cell), _Counts(last=h))]
        if ob.seat not in self.exclude:
            targets.append(self.pool.setdefault(ob.opportunity, _Counts(last=h)))
        for node in targets:
            node.decayed(h, dec)[ob.category] += 1.0
        return True

    def observe_record(self, rec: Any) -> bool:
        return self.observe(from_record(rec))

    def predict(self, seat: int, opp: int, cell: int, *, prior: Sequence[float] | None = None) -> Prediction:
        k1, k2, k3 = self.kappa
        h, dec = self.hand, self.decay
        p = tuple(prior) if prior is not None else self.prior[opp]
        node = self.pool.get(opp)
        if node is not None:
            p = _shrink(node.decayed(h, dec), p, k1)
        node = self.seat.get((seat, opp))
        n_seat = 0.0
        if node is not None:
            c = node.decayed(h, dec)
            n_seat = c[0] + c[1] + c[2]
            p = _shrink(c, p, k2)
        node = self.cell.get((seat, opp, cell))
        n_cell = 0.0
        if node is not None:
            c = node.decayed(h, dec)
            n_cell = c[0] + c[1] + c[2]
            p = _shrink(c, p, k3)
        f = self.floor
        scale = 1.0 - N_CATEGORIES * f
        probs = (p[0] * scale + f, p[1] * scale + f, p[2] * scale + f)
        a = n_cell + k3
        sd = tuple((q * (1 - q) / (a + 1)) ** 0.5 for q in probs)
        return Prediction(probs, n_cell, n_seat, sd)  # type: ignore[arg-type]

    def predict_record(self, rec: Any, *, prior: Sequence[float] | None = None) -> Prediction:
        opp = opportunity(rec.street, rec.raises_before, rec.level_before)
        return self.predict(rec.seat, opp, context_cell(rec.street, rec.pos_group, rec.eff_bb, rec.live_before,
                                                        rec.initiative), prior=prior)

    def prune(self, keep: int) -> int:
        """Keep the ``keep`` context cells with the most evidence; returns how many were dropped."""
        if len(self.cell) <= keep:
            return 0
        ranked = sorted(self.cell.items(), key=lambda kv: -sum(kv[1].c))
        self.cell = dict(ranked[:max(0, keep)])
        return len(ranked) - len(self.cell)

    # --------------------------------------------------------------- state

    def to_state(self) -> dict[str, Any]:
        r = lambda c: [round(x, 3) for x in c.c] + [c.last]  # noqa: E731
        return {"v": 1, "hand": self.hand,
                "pool": {str(o): r(c) for o, c in self.pool.items()},
                "seat": {f"{s}.{o}": r(c) for (s, o), c in self.seat.items()},
                "cell": {f"{s}.{o}.{x}": r(c) for (s, o, x), c in self.cell.items()}}

    def load_state(self, state: Mapping[str, Any]) -> ActionPosterior:
        self.hand = state["hand"]
        mk = lambda v: _Counts(list(v[:3]), v[3])  # noqa: E731
        self.pool = {int(k): mk(v) for k, v in state["pool"].items()}
        self.seat = {tuple(map(int, k.split("."))): mk(v) for k, v in state["seat"].items()}  # type: ignore[misc]
        self.cell = {tuple(map(int, k.split("."))): mk(v) for k, v in state["cell"].items()}  # type: ignore[misc]
        return self
