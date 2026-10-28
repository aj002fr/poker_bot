"""Standard 52-card high-hand ranking for ``practice_nlhe`` and ``uncertainty_plo4``."""

from __future__ import annotations

from collections.abc import Iterable

from tests.oracles.cards import STANDARD_RANKS, Card
from tests.oracles.ranking import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    BestHand,
    HandValue,
    Ranking,
    best_holdem,
    rank_five,
)

# Ace is low only in the wheel A-2-3-4-5, which is a five-high straight.
_STRAIGHTS = {frozenset(range(top - 4, top + 1)): top for top in range(6, 15)}
_STRAIGHTS[frozenset({14, 2, 3, 4, 5})] = 5

STANDARD = Ranking(
    name="standard_high",
    ranks=STANDARD_RANKS,
    straights=_STRAIGHTS,
    order=(
        HIGH_CARD,
        ONE_PAIR,
        TWO_PAIR,
        THREE_OF_A_KIND,
        STRAIGHT,
        FLUSH,
        FULL_HOUSE,
        FOUR_OF_A_KIND,
        STRAIGHT_FLUSH,
    ),
)


def rank5(cards: Iterable[Card]) -> HandValue:
    return rank_five(cards, STANDARD)


def best_nlhe(hole: Iterable[Card], board: Iterable[Card]) -> BestHand:
    return best_holdem(hole, board, STANDARD)
