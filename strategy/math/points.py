"""Point-denominated call thresholds and the ICM risk premium.

The competition pays finishing points, so a call is judged by the points the caller
holds after folding, winning and losing, not by chips. For a two-outcome all-in the
break-even equity is ``(U_fold - U_lose) / (U_win - U_lose)``; the risk premium is
how far that sits above the chip pot-odds threshold.

Before the hand cap, ICM is only a model of future finishes. The ``value_fn`` hook
lets a learned continuation value replace it; at the cap the organizer's ICM is the
scoring rule itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

from arena.tournament.scoring import icm

from strategy.math.odds import div, pot_odds_threshold

ValueFn = Callable[[Sequence[int], Sequence[float]], list[float]]


def point_call_threshold(u_fold: Real, u_win: Real, u_lose: Real) -> Real:
    """Win probability at which a two-outcome call matches folding in points.

    A result above 1 means folding beats even a certain win; below 0 means calling
    beats folding even when the call always loses.
    """
    if u_win <= u_lose:
        raise ValueError("winning must be worth more points than losing")
    return div(u_fold - u_lose, u_win - u_lose)


def call_point_gain(outcomes: Sequence[tuple[Real, Real]], u_fold: Real) -> Real:
    """Expected points from calling minus points from folding.

    ``outcomes`` lists ``(probability, points)`` pairs, for example win, tie and lose,
    and the probabilities must sum to 1.
    """
    total = sum(p for p, _ in outcomes)
    if any(p < 0 for p, _ in outcomes) or abs(total - 1) > 1e-12:
        raise ValueError("outcome probabilities must be non-negative and sum to 1")
    return sum(p * u for p, u in outcomes) - u_fold


def risk_premium(u_fold: Real, u_win: Real, u_lose: Real, pot: Real, call: Real) -> Real:
    """Extra equity a point-maximizing caller needs beyond the chip pot-odds threshold."""
    return point_call_threshold(u_fold, u_win, u_lose) - pot_odds_threshold(pot, call)


@dataclass(frozen=True)
class AllInCall:
    """Hero's point values and thresholds for calling a heads-up all-in."""

    u_fold: float
    u_win: float
    u_lose: float
    matched_call: int
    contested_pot: int
    point_threshold: float
    chip_threshold: float

    @property
    def risk_premium(self) -> float:
        return self.point_threshold - self.chip_threshold


def _points(stacks: Sequence[int], prizes: Sequence[float], value_fn: ValueFn) -> list[float]:
    """Points per player for a between-hands stack vector; busted players take the next-lowest prize."""
    n = len(stacks)
    out = [0.0] * n
    alive = [i for i in range(n) if stacks[i] > 0]
    busted = [i for i in range(n) if stacks[i] == 0]
    if len(busted) > 1:
        raise ValueError("a heads-up all-in eliminates at most one player")
    for i in busted:
        out[i] = float(prizes[n - 1]) if n - 1 < len(prizes) else 0.0
    values = value_fn([stacks[i] for i in alive], list(prizes)[: len(alive)])
    for i, v in zip(alive, values, strict=True):
        out[i] = float(v)
    return out


def icm_all_in_call(
    stacks: Sequence[int],
    prizes: Sequence[float],
    hero: int,
    villain: int,
    hero_committed: int,
    villain_committed: int,
    dead: int = 0,
    value_fn: ValueFn = icm,
) -> AllInCall:
    """Point values of folding, winning and losing when ``hero`` faces ``villain``'s all-in.

    ``stacks`` are the chips each surviving player still holds behind, not counting
    chips already in the pot; ``prizes`` are the points still to be awarded, best
    finish first. ``hero_committed`` and ``villain_committed`` are the chips each has
    put in this hand, and ``dead`` is chips from players who have folded. Every
    other player is assumed out of the hand. A hero who cannot match the shove calls
    all-in, and the villain's unmatched excess is returned. Busting finishes last
    among the current survivors. ``value_fn`` maps positive stacks and the prizes
    they compete for to points per player; it defaults to the arena's exact ICM.
    """
    n = len(stacks)
    if hero == villain or not (0 <= hero < n and 0 <= villain < n):
        raise ValueError("hero and villain must be distinct seats")
    if any(type(s) is not int or s < 0 for s in stacks):
        raise ValueError("stacks must be non-negative integers")
    if stacks[hero] == 0:
        raise ValueError("hero has no chips behind to call with")
    if min(hero_committed, villain_committed, dead) < 0:
        raise ValueError("committed and dead chips must be non-negative")
    if villain_committed <= hero_committed:
        raise ValueError("villain must have bet more than hero has committed")
    if stacks[villain] != 0:
        raise ValueError("villain must be all-in")

    matched_call = min(villain_committed - hero_committed, stacks[hero])
    matched = hero_committed + matched_call
    excess = villain_committed - matched
    contested = 2 * matched + dead

    fold = list(stacks)
    fold[villain] += hero_committed + villain_committed + dead

    win = list(stacks)
    win[hero] += contested - matched_call
    win[villain] += excess

    lose = list(stacks)
    lose[hero] -= matched_call
    lose[villain] += excess + contested

    u_fold = _points(fold, prizes, value_fn)[hero]
    u_win = _points(win, prizes, value_fn)[hero]
    u_lose = _points(lose, prizes, value_fn)[hero]
    return AllInCall(
        u_fold=u_fold,
        u_win=u_win,
        u_lose=u_lose,
        matched_call=matched_call,
        contested_pot=contested,
        point_threshold=point_call_threshold(u_fold, u_win, u_lose),
        chip_threshold=pot_odds_threshold(contested - matched_call, matched_call),
    )
