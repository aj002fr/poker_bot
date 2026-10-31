"""Short-deck (36 cards, ranks 6-A) ranking for ``uncertainty_shortdeck``.

Declared independently of the standard profile: its own deck ranks, its own
straight table where A-6-7-8-9 is the lowest (nine-high) straight and
A-2-3-4-5 does not exist, and flush above full house.

The straight-versus-trips order is an open decision, so it is a parameter:

- ``STRAIGHT_OVER_TRIPS``: the spec's declared profile (section 3.2, E08).
- ``TRIPS_OVER_STRAIGHT``: the "Triton" order common in live short deck,
  which PokerKit is expected to use.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from tests.oracles.cards import SHORT_DECK_RANKS, Card
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


class ShortDeckOrder(Enum):
    STRAIGHT_OVER_TRIPS = "straight_over_trips"
    TRIPS_OVER_STRAIGHT = "trips_over_straight"


SPEC_ORDER = ShortDeckOrder.STRAIGHT_OVER_TRIPS

_STRAIGHTS = {
    frozenset({14, 6, 7, 8, 9}): 9,
    frozenset({6, 7, 8, 9, 10}): 10,
    frozenset({7, 8, 9, 10, 11}): 11,
    frozenset({8, 9, 10, 11, 12}): 12,
    frozenset({9, 10, 11, 12, 13}): 13,
    frozenset({10, 11, 12, 13, 14}): 14,
}

_MIDDLE = {
    ShortDeckOrder.STRAIGHT_OVER_TRIPS: (THREE_OF_A_KIND, STRAIGHT),
    ShortDeckOrder.TRIPS_OVER_STRAIGHT: (STRAIGHT, THREE_OF_A_KIND),
}

RANKINGS = {
    order: Ranking(
        name=f"short_deck_{order.value}",
        ranks=SHORT_DECK_RANKS,
        straights=_STRAIGHTS,
        order=(
            HIGH_CARD,
            ONE_PAIR,
            TWO_PAIR,
            *_MIDDLE[order],
            FULL_HOUSE,
            FLUSH,
            FOUR_OF_A_KIND,
            STRAIGHT_FLUSH,
        ),
    )
    for order in ShortDeckOrder
}


def shortdeck_ranking(order: ShortDeckOrder = SPEC_ORDER) -> Ranking:
    return RANKINGS[ShortDeckOrder(order)]


def rank5(cards: Iterable[Card], order: ShortDeckOrder = SPEC_ORDER) -> HandValue:
    return rank_five(cards, shortdeck_ranking(order))


def best_shortdeck(hole: Iterable[Card], board: Iterable[Card], order: ShortDeckOrder = SPEC_ORDER) -> BestHand:
    return best_holdem(hole, board, shortdeck_ranking(order))
