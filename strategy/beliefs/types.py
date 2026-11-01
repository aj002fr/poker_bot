"""Posterior over coarse opponent families (styles), per seat.

Each family has measured category probabilities per ``(opportunity, depth)`` cell,
generated offline by :mod:`strategy.tables.families` from the arena's own style and
baseline bots. An ``unknown`` family predicts with the population prior and absorbs
behaviour no named family explains. The posterior is updated in log space from every
eligible public action::

    log P(f | data) = log P(f) + sum_t lambda^(T - t) log max(floor, P_f(a_t | cell_t))

``lambda`` < 1 forgets old evidence so a mid-table style switch is tracked, and the
likelihood floor keeps one off-model action from eliminating a family.

The table is variant-specific (the styles exist only for No-limit Hold'em). Without a
table the posterior is disabled and reports the unknown family with probability one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, log
from typing import Any

from strategy.beliefs.posterior import DEFAULT_PRIOR, OPPORTUNITIES, Observation, depth_bucket

UNKNOWN = "unknown"
DEPTHS = 3


def family_cell(opportunity: int, eff_bb: float | None = None, depth: int | None = None) -> int:
    d = depth if depth is not None else depth_bucket(eff_bb or 0.0)
    return opportunity * DEPTHS + d


FAMILY_CELLS = len(OPPORTUNITIES) * DEPTHS


@dataclass(frozen=True)
class FamilyTable:
    ruleset_id: str
    families: tuple[str, ...]
    probs: Mapping[str, Sequence[Sequence[float]]]
    """``probs[family][cell]`` = category probabilities."""

    @classmethod
    def load(cls, ruleset_id: str, directory: str | None = None) -> FamilyTable | None:
        from strategy.tables import store
        from strategy.tables.families import table_name

        loaded = store.load(table_name(ruleset_id), directory)
        if loaded is None:
            return None
        _, data = loaded
        return cls(ruleset_id, tuple(data["families"]), {f: data["probs"][f] for f in data["families"]})


@dataclass(frozen=True, slots=True)
class FamilySummary:
    probs: tuple[tuple[str, float], ...]
    top: str
    top_prob: float
    entropy: float
    observations: int


class FamilyPosterior:
    def __init__(self, table: FamilyTable | None, *, unknown_prior: float = 0.25, forget: float = 0.995,
                 floor: float = 0.005) -> None:
        self.table = table
        self.names: tuple[str, ...] = (*(table.families if table else ()), UNKNOWN)
        n = len(self.names)
        named = (1.0 - unknown_prior) / (n - 1) if n > 1 else 0.0
        self.log_prior = [log(named) if i < n - 1 else 0.0 for i in range(n)] if n > 1 else [0.0]
        if n > 1:
            self.log_prior[-1] = log(unknown_prior)
        self.forget = forget
        self.floor = floor
        self.acc: dict[int, list[float]] = {}
        self.count: dict[int, int] = {}
        self._probs = [table.probs[f] for f in table.families] if table else []
        fl = floor
        # Per cell: every family's category row (unknown last) and the floored log-likelihoods.
        self._rows = [tuple((*(t[c] for t in self._probs), DEFAULT_PRIOR[c // DEPTHS])) for c in range(FAMILY_CELLS)]
        self._logp = [[tuple(log(r[k] if r[k] > fl else fl) for r in rows) for k in range(3)] for rows in self._rows]
        self._cache: dict[int, list[float]] = {}
        self._mix: dict[tuple[int, int], tuple[float, float, float]] = {}

    @property
    def enabled(self) -> bool:
        return self.table is not None

    def observe(self, ob: Observation | None, eff_bb: float) -> None:
        if ob is None or not self._probs:
            return
        cell = family_cell(ob.opportunity, eff_bb)
        acc = self.acc.get(ob.seat)
        if acc is None:
            acc = self.acc[ob.seat] = [0.0] * len(self.names)
        lam = self.forget
        acc[:] = [lam * a + lp for a, lp in zip(acc, self._logp[cell][ob.category])]
        self.count[ob.seat] = self.count.get(ob.seat, 0) + 1
        self._cache.pop(ob.seat, None)
        if self._mix:
            self._mix = {k: v for k, v in self._mix.items() if k[0] != ob.seat}

    def probs(self, seat: int) -> list[float]:
        hit = self._cache.get(seat)
        if hit is not None:
            return hit
        acc = self.acc.get(seat)
        lp = self.log_prior if acc is None else [a + b for a, b in zip(self.log_prior, acc)]
        m = max(lp)
        e = [exp(x - m) for x in lp]
        t = sum(e)
        out = self._cache[seat] = [x / t for x in e]
        return out

    def mixture(self, seat: int, opportunity: int, eff_bb: float) -> tuple[float, float, float]:
        """Family-weighted category probabilities for a context: a prior for the action posterior."""
        if not self._probs:
            return DEFAULT_PRIOR[opportunity]
        cell = family_cell(opportunity, eff_bb)
        hit = self._mix.get((seat, cell))
        if hit is not None:
            return hit
        w = self.probs(seat)
        a = b = c = 0.0
        for wi, row in zip(w, self._rows[cell]):
            a += wi * row[0]
            b += wi * row[1]
            c += wi * row[2]
        out = self._mix[(seat, cell)] = (a, b, c)
        return out

    def summary(self, seat: int) -> FamilySummary:
        p = self.probs(seat)
        pairs = tuple(sorted(zip(self.names, (round(x, 4) for x in p)), key=lambda kv: -kv[1]))
        ent = -sum(x * log(x) for x in p if x > 0) / log(len(p)) if len(p) > 1 else 0.0
        return FamilySummary(pairs, pairs[0][0], pairs[0][1], ent, self.count.get(seat, 0))

    def to_state(self) -> dict[str, Any]:
        return {"acc": {str(s): [round(x, 4) for x in a] for s, a in self.acc.items()},
                "n": {str(s): c for s, c in self.count.items()}}

    def load_state(self, state: Mapping[str, Any]) -> FamilyPosterior:
        self.acc = {int(s): list(a) for s, a in state["acc"].items()}
        self.count = {int(s): int(c) for s, c in state["n"].items()}
        self._cache = {}
        self._mix = {}
        return self
