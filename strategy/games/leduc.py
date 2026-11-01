"""Leduc hold'em, matching OpenSpiel's default ``leduc_poker`` (two players).

Six cards: ranks J, Q, K in two suits, card ``c`` has rank ``c // 2``. Ante 1;
two betting rounds with raise sizes 2 then 4, at most two raises per round,
player 0 opens both rounds. Fold is legal only when facing a bet. Chance deals
player 0's card, player 1's card, then (after round one) the public card from
the remaining four. A private card pairing the public card wins; otherwise
the higher rank wins; equal ranks split.
"""

from __future__ import annotations

from dataclasses import dataclass

from strategy.solve.game import Chance, Decision, Terminal

GAME_VALUE_P0 = -0.085606424078
RAISE_SIZES = (2, 4)
MAX_RAISES = 2
FOLD, CALL, RAISE = "f", "c", "r"


@dataclass(frozen=True)
class _State:
    cards: tuple[int, ...] = ()
    public: int = -1
    rounds: tuple[tuple[str, ...], ...] = ((),)


def _round_status(actions: tuple[str, ...], start: tuple[int, int], raise_size: int):
    """Returns (contributions, to_act, round_over, folded_player, raises)."""
    contrib = list(start)
    stake = max(contrib)
    raises = 0
    player = 0
    for i, a in enumerate(actions):
        if a == FOLD:
            return tuple(contrib), None, True, player, raises
        if a == CALL:
            contrib[player] = stake
            if i >= 1:
                return tuple(contrib), None, True, None, raises
        else:
            contrib[player] = stake + raise_size
            stake = contrib[player]
            raises += 1
        player = 1 - player
    return tuple(contrib), player, False, None, raises


def _rank_hand(card: int, public: int) -> tuple[int, int]:
    return (1 if card // 2 == public // 2 else 0, card // 2)


@dataclass(frozen=True)
class LeducPoker:
    name: str = "leduc"
    num_players: int = 2
    player_names: tuple[str, ...] = ("P0", "P1")

    def root(self) -> _State:
        return _State()

    def deck_for_public(self, state: _State) -> list[int]:
        return [c for c in range(6) if c not in state.cards]

    def expand(self, state: _State):
        if len(state.cards) < 2:
            remaining = [c for c in range(6) if c not in state.cards]
            prob = 1.0 / len(remaining)
            return Chance(tuple((c, prob, _State(state.cards + (c,), -1, state.rounds)) for c in remaining))
        contrib = (1, 1)
        for r, actions in enumerate(state.rounds):
            contrib, to_act, over, folded, raises = _round_status(actions, contrib, RAISE_SIZES[r])
            if folded is not None:
                loser = folded
                u = [0.0, 0.0]
                u[loser] = -float(contrib[loser])
                u[1 - loser] = float(contrib[loser])
                return Terminal(tuple(u))
            if not over:
                break
        else:
            to_act = None
        if over:
            if len(state.rounds) == 1:
                deck = self.deck_for_public(state)
                prob = 1.0 / len(deck)
                return Chance(tuple((c, prob, _State(state.cards, c, state.rounds + ((),))) for c in deck))
            h0 = _rank_hand(state.cards[0], state.public)
            h1 = _rank_hand(state.cards[1], state.public)
            stake = float(contrib[0])
            if h0 == h1:
                return Terminal((0.0, 0.0))
            sign = 1.0 if h0 > h1 else -1.0
            return Terminal((sign * stake, -sign * stake))
        legal = []
        if contrib[0] != contrib[1]:
            legal.append(FOLD)
        legal.append(CALL)
        if raises < MAX_RAISES:
            legal.append(RAISE)
        current = state.rounds[-1]
        key = (to_act, state.cards[to_act], state.public, state.rounds[:-1], current)
        children = tuple(
            (a, _State(state.cards, state.public, state.rounds[:-1] + (current + (a,),))) for a in legal
        )
        return Decision(to_act, key, children)
