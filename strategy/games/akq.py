"""AKQ half-street games (The Mathematics of Poker, Chs. 13 and 14).

A three-card deck {A, K, Q}; X and Y each get one card without replacement
(six equally likely deals) and the higher card wins. X checks dark; Y checks
or bets; X calls or folds. Both ante half the pot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from strategy.solve.game import Chance, Decision, Terminal

CARDS = ("A", "K", "Q")
SQRT2_MINUS_1 = math.sqrt(2.0) - 1.0


@dataclass(frozen=True)
class _AKQ:
    pot: float = 2.0
    num_players: int = 2
    player_names: tuple[str, ...] = ("X", "Y")

    def deals(self) -> tuple[tuple[str, str, float], ...]:
        """(X card, Y card, probability). Card removal: no shared card."""
        pairs = [(x, y) for x in CARDS for y in CARDS if x != y]
        return tuple((x, y, 1.0 / len(pairs)) for x, y in pairs)

    def root(self) -> tuple:
        return ()

    def sizes(self) -> tuple[float, ...]:
        raise NotImplementedError

    def expand(self, state: tuple):
        if not state:
            return Chance(tuple(((x, y), p, (x, y, ())) for x, y, p in self.deals()))
        x, y, h = state
        half = self.pot / 2.0
        sign = (CARDS.index(y) < CARDS.index(x)) - (CARDS.index(y) > CARDS.index(x))
        if not h:
            actions = [("check", (x, y, ("check",)))]
            actions += [(f"bet{s!r}", (x, y, (f"bet{s!r}",))) for s in self.sizes()]
            return Decision(1, ("Y", y), tuple(actions))
        if h == ("check",):
            return Terminal((-sign * half, sign * half))
        if len(h) == 1:
            return Decision(0, ("X", x, h[0]), (("fold", (x, y, h + ("fold",))), ("call", (x, y, h + ("call",)))))
        if h[1] == "fold":
            return Terminal((-half, half))
        stake = half + float(h[0][3:])
        return Terminal((-sign * stake, sign * stake))


@dataclass(frozen=True)
class LimitAKQ(_AKQ):
    """Pot ``pot``, one unit bet (Example 13.1)."""

    bet: float = 1.0

    @property
    def name(self) -> str:
        return f"limit-akq(P={self.pot:g})"

    def sizes(self) -> tuple[float, ...]:
        return (self.bet,)


@dataclass(frozen=True)
class NoLimitAKQ(_AKQ):
    """Unit pot, Y chooses its bet from ``bet_sizes`` (Example 14.1)."""

    pot: float = 1.0
    bet_sizes: tuple[float, ...] = tuple(round(0.05 * k, 4) for k in range(1, 31))

    @property
    def name(self) -> str:
        return f"no-limit-akq(sizes={len(self.bet_sizes)})"

    def sizes(self) -> tuple[float, ...]:
        return self.bet_sizes


def limit_akq_solution(pot: float) -> dict[str, float]:
    return {
        "y_q_bluff": 1.0 / (pot + 1.0),
        "x_k_call": (pot - 1.0) / (pot + 1.0),
        "value_y": (pot - 1.0) / (6.0 * (pot + 1.0)),
    }


def no_limit_akq_value(size: float) -> float:
    """Y's value when it must use a common size ``s`` into a unit pot."""
    return size * (1.0 - size) / (6.0 * (1.0 + size))


def no_limit_akq_solution() -> dict[str, float]:
    s = SQRT2_MINUS_1
    return {
        "size": s,
        "value_y": s * s / 6.0,
        "y_q_bluff": s / (1.0 + s),
        "x_k_call": (1.0 - s) / (1.0 + s),
    }
