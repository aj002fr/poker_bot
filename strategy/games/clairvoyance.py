"""Clairvoyance games (The Mathematics of Poker, Chs. 11 and 19).

Y knows whether it holds a winner or a dead hand; X holds a bluff-catcher
and never bets. Both players ante half the pot, so utilities are zero-sum
and Y's value is its net expectation.
"""

from __future__ import annotations

from dataclasses import dataclass

from strategy.solve.game import Chance, Decision, Terminal


@dataclass(frozen=True)
class ClairvoyantHalfStreet:
    """X checks; Y checks or bets ``bet``; X calls or folds (Example 11.1)."""

    pot: float = 2.0
    bet: float = 1.0
    winner_fraction: float = 0.5
    num_players: int = 2
    player_names: tuple[str, ...] = ("X", "Y")

    @property
    def name(self) -> str:
        return f"clairvoyance-half-street(P={self.pot:g},w={self.winner_fraction:g})"

    def root(self) -> tuple:
        return ()

    def expand(self, state: tuple):
        half = self.pot / 2.0
        if not state:
            w = self.winner_fraction
            return Chance((("winner", w, ("winner",)), ("dead", 1.0 - w, ("dead",))))
        hand = state[0]
        sign = 1.0 if hand == "winner" else -1.0
        if len(state) == 1:
            return Decision(1, ("Y", hand), (("check", (hand, "check")), ("bet", (hand, "bet"))))
        if state[1] == "check":
            return Terminal((-sign * half, sign * half))
        if len(state) == 2:
            return Decision(0, ("X", "bet"), (("fold", (hand, "bet", "fold")), ("call", (hand, "bet", "call"))))
        if state[2] == "fold":
            return Terminal((-half, half))
        stake = half + self.bet
        return Terminal((-sign * stake, sign * stake))


def half_street_solution(pot: float, bet: float, winner_fraction: float) -> dict[str, float]:
    """Equilibrium when Y has enough dead hands to bluff at the indifference ratio."""
    bluff_to_value = bet / (pot + bet)
    call = pot / (pot + bet)
    w = winner_fraction
    return {
        "bluff_fraction_of_dead": bluff_to_value * w / (1.0 - w),
        "x_call": call,
        "value_y": w * (pot / 2.0 + bet * call) - (1.0 - w) * pot / 2.0,
    }


@dataclass(frozen=True)
class TwoStreetClairvoyant:
    """Static two-street clairvoyance with Y choosing sizes (Example 19.2).

    Y may check or bet any menu size (pot fractions, clipped to the stack, plus
    all-in) on each street; X calls or folds. With pot 1 and stack 4 the
    geometric schedule bets one pot, then the remaining 3 into a pot of 3.
    """

    pot: float = 1.0
    stack: float = 4.0
    winner_fraction: float = 1.0 / 3.0
    fractions: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    num_players: int = 2
    player_names: tuple[str, ...] = ("X", "Y")

    @property
    def name(self) -> str:
        return f"two-street-clairvoyant(P={self.pot:g},S={self.stack:g})"

    def sizes(self, pot: float, remaining: float) -> tuple[float, ...]:
        if remaining <= 0:
            return ()
        out = sorted({round(min(f * pot, remaining), 9) for f in self.fractions} | {round(remaining, 9)})
        return tuple(out)

    def root(self) -> tuple:
        return ()

    def expand(self, state: tuple):
        if not state:
            w = self.winner_fraction
            return Chance((("winner", w, ("winner", ())), ("dead", 1.0 - w, ("dead", ()))))
        hand, history = state
        sign = 1.0 if hand == "winner" else -1.0
        half = self.pot / 2.0
        pot, invested, street, facing = self.pot, 0.0, 1, None
        for token in history:
            if token == "check":
                street += 1
            elif token == "fold":
                return Terminal((-(half + invested), half + invested))
            elif token == "call":
                invested += facing
                pot += 2 * facing
                facing = None
                street += 1
            else:
                facing = float(token)
        if facing is not None:
            key = ("X", history)
            return Decision(0, key, (("fold", (hand, history + ("fold",))), ("call", (hand, history + ("call",)))))
        if street > 2:
            stake = half + invested
            return Terminal((-sign * stake, sign * stake))
        actions = [("check", (hand, history + ("check",)))]
        for size in self.sizes(pot, self.stack - invested):
            actions.append((f"{size:g}", (hand, history + (f"{size:g}",))))
        return Decision(1, ("Y", hand, history), tuple(actions))


def geometric_bet_fraction(stack_to_pot: float, streets: int) -> float:
    """Pot fraction s with (1 + 2s)^streets = 1 + 2 * stack_to_pot."""
    growth = (1.0 + 2.0 * stack_to_pot) ** (1.0 / streets)
    return (growth - 1.0) / 2.0


def two_street_solution(pot: float, stack: float, winner_fraction: float) -> dict[str, float]:
    """Geometric sizing and value (Y's total betting range B1 = 4 w r1 r2 / ((1+r1)(1+r2)))."""
    s = geometric_bet_fraction(stack / pot, 2)
    growth = 1.0 + 2.0 * s
    bet_mass = 4.0 * winner_fraction * growth * growth / (1.0 + growth) ** 2
    if bet_mass > 1.0:
        raise ValueError("Y lacks enough dead hands for the bluffing equilibrium")
    return {
        "first_bet": s * pot,
        "second_bet": s * pot * growth,
        "growth": growth,
        "y_bet_mass": bet_mass,
        "x_call_each_street": 1.0 - s / (1.0 + s),
        "value_y": pot * (bet_mass - 0.5),
    }
