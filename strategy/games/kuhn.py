"""Kuhn poker, matching OpenSpiel's ``kuhn_poker`` (two players).

Cards J, Q, K (0, 1, 2); ante 1; one bet of 1. Chance deals player 0's card,
then player 1's. Actions are ``p`` (pass/check/fold) and ``b`` (bet/call), in
OpenSpiel's order. Game value for player 0 is -1/18.
"""

from __future__ import annotations

from dataclasses import dataclass

from strategy.solve.game import Chance, Decision, Terminal

GAME_VALUE_P0 = -1.0 / 18.0
CARDS = ("J", "Q", "K")


@dataclass(frozen=True)
class KuhnPoker:
    name: str = "kuhn"
    num_players: int = 2
    player_names: tuple[str, ...] = ("P0", "P1")

    def root(self) -> tuple[tuple[int, ...], str]:
        return ((), "")

    def expand(self, state: tuple[tuple[int, ...], str]):
        cards, history = state
        if len(cards) < 2:
            remaining = [c for c in range(3) if c not in cards]
            prob = 1.0 / len(remaining)
            return Chance(tuple((CARDS[c], prob, (cards + (c,), history)) for c in remaining))
        if history in ("pp", "bb", "pbb"):
            stake = 1 if history == "pp" else 2
            sign = 1.0 if cards[0] > cards[1] else -1.0
            return Terminal((sign * stake, -sign * stake))
        if history == "bp":
            return Terminal((1.0, -1.0))
        if history == "pbp":
            return Terminal((-1.0, 1.0))
        player = len(history) % 2
        return Decision(
            player,
            (player, CARDS[cards[player]], history),
            (("p", (cards, history + "p")), ("b", (cards, history + "b"))),
        )
