"""[0,1] games from The Mathematics of Poker, discretised to ``n`` hands.

Each player's hand is an independent uniform draw from ``n`` equally likely
strengths; index 0 is the strongest, and hand ``i`` stands for the interval
``[i/n, (i+1)/n)``. Equal hands split. Thresholds therefore converge to the
book's continuous solutions at rate ~1/n, and gates use tolerances of a few
grid steps. Unless stated, both players ante half the pot so utilities are
zero-sum; by symmetry Y's net value equals the book's ex-showdown value.

Players are X (0, acts first unless stated) and Y (1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from strategy.solve.game import Chance, Decision, Terminal


def compare(a: int, b: int) -> int:
    """+1 if hand ``a`` beats ``b``, -1 if it loses, 0 on a tie."""
    return (a < b) - (a > b)


def hand_midpoints(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


@dataclass(frozen=True)
class _TwoHandDeal:
    n: int = 100
    num_players: int = 2
    player_names: tuple[str, ...] = ("X", "Y")

    @property
    def meta(self) -> dict[str, object]:
        return {"n": self.n}

    def root(self) -> tuple:
        return ()

    def deal(self) -> Chance:
        p = 1.0 / (self.n * self.n)
        return Chance(tuple(((x, y), p, (x, y, ())) for x in range(self.n) for y in range(self.n)))


def _zero_sum(u_y: float) -> Terminal:
    return Terminal((-u_y, u_y))


@dataclass(frozen=True)
class HalfStreetNoFold(_TwoHandDeal):
    """X checks; Y checks or bets 1; X must call (Example 11.2's no-fold game)."""

    pot: float = 1.0

    @property
    def name(self) -> str:
        return f"half-street-no-fold(n={self.n})"

    def expand(self, state: tuple):
        if not state:
            return self.deal()
        x, y, h = state
        if not h:
            return Decision(1, ("Y", y), (("check", (x, y, ("check",))), ("bet", (x, y, ("bet",)))))
        stake = self.pot / 2.0 + (1.0 if h[0] == "bet" else 0.0)
        return _zero_sum(compare(y, x) * stake)


@dataclass(frozen=True)
class HalfStreetWithFold(_TwoHandDeal):
    """X checks; Y checks or bets 1; X calls or folds."""

    pot: float = 2.0

    @property
    def name(self) -> str:
        return f"half-street-fold(n={self.n},P={self.pot:g})"

    def expand(self, state: tuple):
        if not state:
            return self.deal()
        x, y, h = state
        half = self.pot / 2.0
        if not h:
            return Decision(1, ("Y", y), (("check", (x, y, ("check",))), ("bet", (x, y, ("bet",)))))
        if h == ("check",):
            return _zero_sum(compare(y, x) * half)
        if h == ("bet",):
            return Decision(0, ("X", x), (("fold", (x, y, h + ("fold",))), ("call", (x, y, h + ("call",)))))
        if h[1] == "fold":
            return _zero_sum(half)
        return _zero_sum(compare(y, x) * (half + 1.0))


@dataclass(frozen=True)
class JamFold(_TwoHandDeal):
    """Y posts 1/2 and jams to ``stack`` or folds; X posts 1 and calls or folds.

    ``stronger_wins`` = 1 is the deterministic game (Example 12.1); 2/3 is the
    equity-smoothed game (Example 12.2). Values are total-game, blinds included.
    """

    stack: float = 2.0
    stronger_wins: float = 1.0

    @property
    def name(self) -> str:
        return f"jam-fold(n={self.n},S={self.stack:g},q={self.stronger_wins:.3g})"

    def expand(self, state: tuple):
        if not state:
            return self.deal()
        x, y, h = state
        if not h:
            return Decision(1, ("Y", y), (("fold", (x, y, ("fold",))), ("jam", (x, y, ("jam",)))))
        if h == ("fold",):
            return _zero_sum(-0.5)
        if h == ("jam",):
            return Decision(0, ("X", x), (("fold", (x, y, h + ("fold",))), ("call", (x, y, h + ("call",)))))
        if h[1] == "fold":
            return _zero_sum(1.0)
        edge = (2.0 * self.stronger_wins - 1.0) * compare(y, x)
        return _zero_sum(self.stack * edge)


@dataclass(frozen=True)
class FullStreetNoFold(_TwoHandDeal):
    """One street, two bets, no folding; optional check-raise (Examples 16.2 and 16.4).

    X checks or bets. Facing a bet Y calls or raises (X then calls). After a
    check Y checks or bets; X calls or, if allowed, check-raises (Y then calls).
    """

    check_raise: bool = True
    pot: float = 1.0

    @property
    def name(self) -> str:
        return f"full-street-no-fold(n={self.n},check_raise={self.check_raise})"

    @property
    def meta(self) -> dict[str, object]:
        return {"n": self.n, "check_raise": self.check_raise}

    def expand(self, state: tuple):
        if not state:
            return self.deal()
        x, y, h = state

        def child(a: str) -> tuple:
            return (x, y, h + (a,))

        if not h:
            return Decision(0, ("X", x, h), (("check", child("check")), ("bet", child("bet"))))
        if h == ("bet",):
            return Decision(1, ("Y", y, h), (("call", child("call")), ("raise", child("raise"))))
        if h == ("check",):
            return Decision(1, ("Y", y, h), (("check", child("check")), ("bet", child("bet"))))
        if h == ("check", "bet"):
            actions = [("call", child("call"))]
            if self.check_raise:
                actions.append(("raise", child("raise")))
            return Decision(0, ("X", x, h), tuple(actions))
        bets = {("bet", "call"): 1, ("bet", "raise"): 2, ("check", "check"): 0,
                ("check", "bet", "call"): 1, ("check", "bet", "raise"): 2}[h]
        return _zero_sum(compare(y, x) * (self.pot / 2.0 + bets))


@dataclass(frozen=True)
class FullStreetFinitePot(_TwoHandDeal):
    """One street, one bet, folding allowed ([0,1] Game #9, Example 17.1)."""

    pot: float = 2.0

    @property
    def name(self) -> str:
        return f"full-street-finite-pot(n={self.n},P={self.pot:g})"

    def expand(self, state: tuple):
        if not state:
            return self.deal()
        x, y, h = state
        half = self.pot / 2.0

        def child(a: str) -> tuple:
            return (x, y, h + (a,))

        if not h:
            return Decision(0, ("X", x, h), (("check", child("check")), ("bet", child("bet"))))
        if h == ("bet",):
            return Decision(1, ("Y", y, h), (("fold", child("fold")), ("call", child("call"))))
        if h == ("check",):
            return Decision(1, ("Y", y, h), (("check", child("check")), ("bet", child("bet"))))
        if h == ("check", "bet"):
            return Decision(0, ("X", x, h), (("fold", child("fold")), ("call", child("call"))))
        if h == ("bet", "fold"):
            return _zero_sum(-half)
        if h == ("check", "bet", "fold"):
            return _zero_sum(half)
        bets = 0 if h == ("check", "check") else 1
        return _zero_sum(compare(y, x) * (half + bets))


@dataclass(frozen=True)
class DrySidePot:
    """Three-player half-street with Z all-in (Example 29.2, Dry Side Pot Game #2).

    X, Y and Z each ante ``pot / 3``; Z is all-in. X checks dark; Y checks or
    bets 1 into an empty side pot; X calls or folds. The main pot goes to the
    best live hand among the remaining players and Z; the side pot to the
    better of X and Y. Z's hand is never observed and Z never acts, so its
    deal is integrated out exactly at each terminal over the same n-hand grid.
    """

    n: int = 100
    pot: float = 9.0
    num_players: int = 3
    player_names: tuple[str, ...] = ("X", "Y", "Z")

    @property
    def name(self) -> str:
        return f"dry-side-pot(n={self.n},P={self.pot:g})"

    @property
    def meta(self) -> dict[str, object]:
        return {"n": self.n}

    def root(self) -> tuple:
        return ()

    def expand(self, state: tuple):
        n = self.n
        if not state:
            p = 1.0 / (n * n)
            return Chance(tuple(((x, y), p, (x, y, ())) for x in range(n) for y in range(n)))
        x, y, h = state
        if not h:
            return Decision(1, ("Y", y), (("check", (x, y, ("check",))), ("bet", (x, y, ("bet",)))))
        if h == ("bet",):
            return Decision(0, ("X", x), (("fold", (x, y, h + ("fold",))), ("call", (x, y, h + ("call",)))))
        return Terminal(self.utilities(x, y, h))

    def utilities(self, x: int, y: int, h: tuple) -> tuple[float, float, float]:
        n = self.n
        ante = self.pot / 3.0
        won = np.zeros(3)
        live = (0, 1) if h[-1] != "fold" else (1,)
        hands = {0: x, 1: y}
        best = min(hands[i] for i in live)
        leaders = [i for i in live if hands[i] == best]
        # Z's hand z is uniform on the grid: below best -> Z wins, equal -> split with Z.
        p_z_better = best / n
        p_z_tie = 1.0 / n
        p_z_worse = (n - 1 - best) / n
        for i in leaders:
            won[i] += self.pot * (p_z_worse / len(leaders) + p_z_tie / (len(leaders) + 1))
        won[2] += self.pot * (p_z_better + p_z_tie / (len(leaders) + 1))
        if h == ("bet", "call"):
            c = compare(x, y)
            if c > 0:
                won[0] += 2.0
            elif c < 0:
                won[1] += 2.0
            else:
                won[:2] += 1.0
        contributed = np.full(3, ante)
        if h == ("bet", "call"):
            contributed[:2] += 1.0
        return tuple(float(v) for v in won - contributed)


def half_street_no_fold_solution() -> dict[str, float]:
    return {"y_bet": 0.5, "value_y": 0.25}


def half_street_with_fold_solution(pot: float) -> dict[str, float]:
    alpha = 1.0 / (pot + 1.0)
    y1 = (1.0 - alpha) / ((2.0 - alpha) * (1.0 + alpha))
    return {"y_value_bet": y1, "x_call": 2.0 * y1, "y_bluff": alpha * y1, "value_y": (1.0 - alpha) * y1 / 2.0}


def jam_fold_solution(stack: float) -> dict[str, float]:
    s = stack
    return {"x_call": 1.5 / (1.0 + s), "y_jam": 3.0 * s / (1.0 + s) ** 2, "value_y": -0.5 + 9.0 * s / (4.0 * (1.0 + s) ** 2)}


def jam_fold_smoothed_solution(stack: float) -> dict[str, float]:
    """Stronger hand wins 2/3. At S = 3 X is indifferent with every hand above 3/4."""
    s = stack
    if s <= 1.5:
        return {"x_call": 1.0, "y_jam": 1.0, "value_y": 0.0}
    if s <= 3.0:
        return {"x_call": 1.0, "y_jam": (2.0 * s + 3.0) / (4.0 * s), "value_y": (2.0 * s - 3.0) ** 2 / (48.0 * s)}
    return {"x_call": 4.5 / (s + 3.0), "y_jam": 9.0 * s / (s + 3.0) ** 2, "value_y": -0.5 + 27.0 * s / (4.0 * (s + 3.0) ** 2)}


def full_street_no_fold_solution(check_raise: bool) -> dict[str, float]:
    if check_raise:
        return {"x_check_raise": 1 / 9, "x_bet_or_raise": 5 / 9, "y_raise": 1 / 3, "y_bet": 2 / 3, "value_y": 1 / 9}
    return {"x_bet": 1 / 2, "y_raise": 1 / 4, "y_bet": 3 / 4, "value_y": 1 / 8}


def finite_pot_solution(pot: float) -> dict[str, float]:
    if pot != 2.0:
        raise ValueError("source thresholds are tabulated for P = 2 only")
    return {
        "x_value_bet": 1 / 3, "x_bluff": 1 / 9, "x_call": 2 / 3,
        "y_value_bet": 1 / 2, "y_bluff": 1 / 6, "y_call": 2 / 3, "value_y": 1 / 18,
    }


DRY_SIDE_POT_PRINTED = {"x_call": 0.2848, "y_value_bet": 0.1424, "bluff_center": 0.6424, "bluff_width": 0.0190}


def dry_side_pot_solution(pot: float = 9.0) -> dict[str, float]:
    """Solve the book's own three indifference equations for P = 9.

    X at its call threshold:   w (10 - 9x) = y1
    Y at its value threshold:  y1 = x / 2
    Y at both bluff endpoints: 9 (y - x)(1 - y) = x, so
                               w = sqrt((1 + x)^2 - 40x / 9).
    The printed solution (x ~ 0.2848) does not satisfy the last equation; the
    root below does, and the solver reproduces it.
    """
    if pot != 9.0:
        raise ValueError("the source works the P = 9 case only")

    def gap(x: float) -> float:
        return math.sqrt(max((1 + x) ** 2 - 40 * x / 9, 0.0)) - x / (20 - 18 * x)

    lo, hi = 0.3, _double_root()
    for _ in range(200):
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2
    width = x / (20 - 18 * x)
    return {"x_call": x, "y_value_bet": x / 2, "bluff_center": (1 + x) / 2, "bluff_width": width}


def _double_root() -> float:
    """Largest x with a real bluff interval: (1 + x)^2 = 40x / 9."""
    b = 2 - 40 / 9
    return (-b - math.sqrt(b * b - 4)) / 2
