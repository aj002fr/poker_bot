"""Marked-draw two-street game (The Mathematics of Poker, Example 20.2).

Limit hold'em turn with a pot of 4 bets. X has an exposed made hand and bets
1 on the turn (forced). Y holds a flush draw with probability 1/10, otherwise
a dead hand, and calls or folds. The flush arrives with probability 1/5 and is
public. On the river X checks; Y checks or bets 1; X calls or folds. Y wins a
showdown only with a completed flush. Both ante half the turn pot.

Equilibrium: Y calls with every draw plus 1/70 of its range in dead hands
(1/63 of the dead hands), bets every flush and every carried dead hand on the
flush river, and X calls only 1/7 there — not the isolated-river 6/7 —
because the dead hands must be indifferent to the whole call-then-bluff line.
"""

from __future__ import annotations

from dataclasses import dataclass

from strategy.solve.game import Chance, Decision, Terminal


@dataclass(frozen=True)
class MarkedDraw:
    turn_pot: float = 4.0
    bet: float = 1.0
    draw_fraction: float = 0.1
    flush_probability: float = 0.2
    name: str = "marked-draw"
    num_players: int = 2
    player_names: tuple[str, ...] = ("X", "Y")

    def root(self) -> tuple:
        return ()

    def expand(self, state: tuple):
        half = self.turn_pot / 2.0
        if not state:
            d = self.draw_fraction
            return Chance((("draw", d, ("draw",)), ("dead", 1.0 - d, ("dead",))))
        hand = state[0]
        h = state[1:]
        if not h:
            return Decision(1, ("Y", hand, "turn"), (("fold", (hand, "fold")), ("call", (hand, "call"))))
        if h == ("fold",):
            return Terminal((half, -half))
        if len(h) == 1:
            f = self.flush_probability
            return Chance((("flush", f, (hand, "call", "flush")), ("blank", 1.0 - f, (hand, "call", "blank"))))
        river = h[1]
        y_wins = hand == "draw" and river == "flush"
        sign = 1.0 if y_wins else -1.0
        if len(h) == 2:
            return Decision(1, ("Y", hand, river), (("check", state + ("check",)), ("bet", state + ("bet",))))
        if h[2] == "check":
            stake = half + self.bet
            return Terminal((-sign * stake, sign * stake))
        if len(h) == 3:
            return Decision(0, ("X", river), (("fold", state + ("fold",)), ("call", state + ("call",))))
        if h[3] == "fold":
            stake = half + self.bet
            return Terminal((-stake, stake))
        stake = half + 2.0 * self.bet
        return Terminal((-sign * stake, sign * stake))


def marked_draw_solution(
    turn_pot: float = 4.0, draw_fraction: float = 0.1, flush_probability: float = 0.2
) -> dict[str, float]:
    """Closed form for a unit bet, derived from the book's indifference argument."""
    river_pot = turn_pot + 2.0
    f = flush_probability
    bluff_mass = draw_fraction / (river_pot + 1.0)
    # Dead hand indifferent to call-then-bluff: -1 + f * (river_pot - (river_pot + 1) c) = 0.
    x_call_flush = (river_pot - 1.0 / f) / (river_pot + 1.0)
    half = turn_pot / 2.0
    draw_ev = (1.0 - f) * -(half + 1.0) + f * ((half + 1.0) + x_call_flush)
    value_y = draw_fraction * draw_ev - (1.0 - draw_fraction) * half
    return {
        "y_dead_call_mass": bluff_mass,
        "y_dead_call_fraction": bluff_mass / (1.0 - draw_fraction),
        "x_call_flush_river": x_call_flush,
        "x_call_isolated_river": river_pot / (river_pot + 1.0),
        "value_y": value_y,
    }
