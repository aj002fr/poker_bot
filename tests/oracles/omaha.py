"""Four-card Omaha high: exactly two hole cards plus exactly three board cards."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import combinations
from typing import NamedTuple

from tests.oracles.cards import Card, InvalidCards, validate_cards
from tests.oracles.ranking import HandValue, Ranking, rank_five_unchecked
from tests.oracles.standard import STANDARD


class OmahaHand(NamedTuple):
    value: HandValue
    hole_used: tuple[Card, Card]
    board_used: tuple[Card, Card, Card]
    combinations_checked: int


def omaha_combinations(
    hole: tuple[Card, ...], board: tuple[Card, ...]
) -> Iterator[tuple[tuple[Card, ...], tuple[Card, ...]]]:
    for h in combinations(hole, 2):
        for b in combinations(board, 3):
            yield h, b


def best_omaha(hole: Iterable[Card], board: Iterable[Card], ranking: Ranking = STANDARD) -> OmahaHand:
    """Best of ``C(4,2) * C(len(board),3)`` hands; 60 on a full board."""
    hole = tuple(hole)
    board = tuple(board)
    if len(hole) != 4:
        raise InvalidCards(f"PLO needs 4 hole cards, got {len(hole)}")
    if len(board) not in (3, 4, 5):
        raise InvalidCards(f"board must have 3-5 cards, got {len(board)}")
    validate_cards(hole + board, ranking.ranks)
    best: OmahaHand | None = None
    checked = 0
    for h, b in omaha_combinations(hole, board):
        checked += 1
        value = rank_five_unchecked(h + b, ranking)
        if best is None or value > best.value:
            best = OmahaHand(value, h, b, 0)  # type: ignore[arg-type]
    assert best is not None
    return best._replace(combinations_checked=checked)
