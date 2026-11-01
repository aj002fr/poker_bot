"""Bucketed Bayesian range updates for every seat, including the hero's own range.

The latent state of a player's hand is its preflop strength decile (bucket 0..9, from
the variant's preflop table), which is fixed for the hand, so beliefs carry across
streets without transition tables. Each public action multiplies the weights by a
likelihood ``L(action | bucket)`` and renormalizes::

    w'_b = w_b L(a | b) / sum_r w_r L(a | r)

The default likelihood is a soft threshold strategy built from the actor's predicted
action frequencies in that context (from :mod:`strategy.beliefs.posterior`): the weakest
part of the *current* range folds, the middle calls, the top raises, a share of raises are
bluffs drawn uniformly, and the soft edge widens on later streets where the preflop
bucket says less about the hand. Every likelihood is floored, so an off-model action
reweights a range instead of zeroing it. A policy that knows its own per-bucket action
probabilities passes them directly to keep the hero's own range exact.

Card removal: the hero's hole cards and the board remove opponents' combos per bucket
using the table's per-card combo counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, log
from operator import add, mul

K = 10
STREET_WIDTH = (0.06, 0.16, 0.22, 0.28)
"""Soft-threshold width by street (preflop, flop, turn, river), in range-quantile units."""
BLUFF_SHARE = (0.10, 0.25, 0.25, 0.30)
"""Share of aggressive actions modelled as bluffs spread over the whole range."""
FLOOR = 0.02


def threshold_likelihood(weights: Sequence[float], probs: Sequence[float], category: int, *, street: int,
                         passive_first: bool = True, floor: float = FLOOR) -> list[float]:
    """``L(category | bucket)`` for a soft threshold strategy over the current range.

    ``probs`` are the actor's (passive-low, middle, aggressive) category probabilities:
    (fold, call, raise) when facing a bet or (check, small bet, large bet) unbet. Unbet,
    both bet sizes share the aggressive region (large bets slightly more polarized).
    """
    inv_tau = 1.0 / STREET_WIDTH[min(street, 3)]
    beta = BLUFF_SHARE[min(street, 3)]
    inv_total = 1.0 / (sum(weights) or 1.0)
    if passive_first:
        p_top = probs[2]
        lo_edge, hi_edge = probs[0], 1.0 - p_top
    else:
        p_top = probs[1] + probs[2]
        lo_edge = hi_edge = probs[0]
    a_top, b_top = 1.0 - beta, beta * p_top
    a_low = 1.0 - beta * p_top
    need_top = category != 0 or not passive_first
    need_low = passive_first and category != 2
    if not passive_first and category != 0:
        share = probs[category] / max(1e-9, p_top)
        tilt = 0.5 if category == 2 else -0.25
    cum = 0.0
    out: list[float] = []
    for w in weights:
        u = (cum + 0.5 * w) * inv_total
        cum += w
        # |x| <= 1 / min(STREET_WIDTH) < 20, so exp cannot overflow.
        top = a_top / (1.0 + exp((hi_edge - u) * inv_tau)) + b_top if need_top else 0.0
        low = a_low / (1.0 + exp((u - lo_edge) * inv_tau)) if need_low else 0.0
        if passive_first:
            v = low if category == 0 else top if category == 2 else 1.0 - low - top
        elif category == 0:
            v = 1.0 - top
        else:
            v = top * share * (1.0 + tilt * (abs(u - 0.5) * 2 - 0.5))
        out.append(v if v > floor else floor)
    return out


@dataclass(frozen=True, slots=True)
class RangeSummary:
    weights: tuple[float, ...]
    mean: float
    """Mean strength in [0, 1] (bucket midpoints)."""
    top: float
    """Mass in the top decile: the near-nut share of the range preflop."""
    top2: float
    middle: float
    """Mass in deciles 3..6 (the middle 40% of all hands)."""
    bottom: float
    entropy: float
    """Normalized entropy: 1.0 is an uninformed range."""
    capped: bool
    """Top-decile mass fell below half its prior."""

    @property
    def polarization(self) -> float:
        return (self.top2 + self.bottom) / max(1e-9, self.middle)


_MIDS = tuple((i + 0.5) / K for i in range(K))
_INV_LOG_K = 1.0 / log(K)


def summarize(weights: Sequence[float], prior_top: float = 0.1) -> RangeSummary:
    inv = 1.0 / (sum(weights) or 1.0)
    w = tuple(x * inv for x in weights)
    mean = sum(map(mul, w, _MIDS))
    ent = -sum(p * log(p) for p in w if p > 0) * _INV_LOG_K
    top = w[-1]
    return RangeSummary(w, mean, top, w[-1] + w[-2], sum(w[3:7]), w[0] + w[1] + w[2], ent, top < prior_top / 2)


class RangeBeliefs:
    """Range weights for every dealt seat in the current hand."""

    __slots__ = ("prior", "bucket_combos", "bucket_card", "weights", "prior_top", "hero", "_board", "_hole",
                 "_f_opp", "_f_hero", "_summ", "_by_card")

    def __init__(self, bucket_combos: Sequence[int] | None = None,
                 bucket_card: Sequence[Sequence[int]] | None = None) -> None:
        self.bucket_combos = list(bucket_combos) if bucket_combos else None
        self.bucket_card = [list(r) for r in bucket_card] if bucket_card else None
        self._by_card = [tuple(r[c] for r in self.bucket_card) for c in range(52)] if self.bucket_card else []
        if self.bucket_combos:
            t = sum(self.bucket_combos)
            self.prior = [c / t for c in self.bucket_combos]
        else:
            self.prior = [1.0 / K] * K
        self.prior_top = self.prior[-1]
        self.weights: dict[int, list[float]] = {}
        self.hero: int | None = None
        self._board: list[int] = []
        self._hole: list[int] = []
        self._f_opp = [1.0] * K
        self._f_hero = [1.0] * K
        self._summ: dict[int, RangeSummary] = {}

    def new_hand(self, seats: Sequence[int], hero: int | None = None, hero_cards: Sequence[int] = ()) -> None:
        self.weights = {s: list(self.prior) for s in seats}
        self._summ = {}
        self.hero = hero
        self._board, self._hole = [], list(hero_cards)
        self._f_opp = [1.0] * K
        self._f_hero = [1.0] * K
        self._apply_removal()

    def add_board(self, cards: Sequence[int]) -> None:
        self._board.extend(cards)
        self._apply_removal()

    def _factors(self, dead: Sequence[int]) -> list[float]:
        """Surviving combo share per bucket. Combos holding two dead cards are subtracted
        twice, a small bias (at most a few percent of a bucket) traded for O(K) per card."""
        bc, by_card = self.bucket_combos, self._by_card
        assert bc is not None
        if not dead:
            return [1.0 if n else 0.0 for n in bc]
        rem = by_card[dead[0]]
        for c in dead[1:]:
            rem = list(map(add, rem, by_card[c]))
        return [(n - r) / n if n > r else 0.0 for n, r in zip(bc, rem)]

    def _apply_removal(self) -> None:
        """Scale weights by each bucket's surviving combo share. The hero's own hole cards
        remove combos only from opponents; board cards remove them from everyone."""
        if not self.bucket_card or not self.bucket_combos:
            return
        f_opp = self._factors(self._board + self._hole)
        f_hero = self._factors(self._board)
        r_opp = [n / o if o > 0 else 0.0 for n, o in zip(f_opp, self._f_opp)]
        r_hero = [n / o if o > 0 else 0.0 for n, o in zip(f_hero, self._f_hero)]
        self._f_opp, self._f_hero = f_opp, f_hero
        for s, w in self.weights.items():
            r = r_hero if s == self.hero else r_opp
            new = [a * b for a, b in zip(w, r)]
            t = sum(new)
            if t > 0:
                w[:] = [x / t for x in new]
        self._summ.clear()

    def update(self, seat: int, likelihood: Sequence[float]) -> None:
        w = self.weights.get(seat)
        if w is None:
            return
        new = [a * b for a, b in zip(w, likelihood)]
        t = sum(new)
        if t <= 0:
            return
        w[:] = [x / t for x in new]
        self._summ.pop(seat, None)

    def fold(self, seat: int) -> None:
        self.weights.pop(seat, None)
        self._summ.pop(seat, None)

    def summary(self, seat: int) -> RangeSummary | None:
        hit = self._summ.get(seat)
        if hit is not None:
            return hit
        w = self.weights.get(seat)
        if w is None:
            return None
        out = self._summ[seat] = summarize(w, self.prior_top)
        return out

    def to_state(self) -> dict:
        return {"w": {str(s): [round(x, 6) for x in w] for s, w in self.weights.items()}, "hero": self.hero,
                "board": list(self._board), "hole": list(self._hole)}

    def load_state(self, state: dict) -> RangeBeliefs:
        self.weights = {int(s): list(w) for s, w in state["w"].items()}
        self._summ = {}
        self.hero = state["hero"]
        self._board, self._hole = list(state["board"]), list(state["hole"])
        if self.bucket_card and self.bucket_combos:
            self._f_opp = self._factors(self._board + self._hole)
            self._f_hero = self._factors(self._board)
        return self
