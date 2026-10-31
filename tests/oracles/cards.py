"""Card parsing and deck validation for the test oracles (stdlib only)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

RANK_CHARS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUE = {ch: value for value, ch in enumerate(RANK_CHARS, start=2)}
VALUE_CHAR = {value: ch for ch, value in RANK_VALUE.items()}

STANDARD_RANKS = tuple(range(2, 15))
SHORT_DECK_RANKS = tuple(range(6, 15))


class InvalidCards(ValueError):
    """An impossible card set: malformed, duplicated, or outside the deck."""


class Card(NamedTuple):
    rank: int
    suit: str

    def __str__(self) -> str:
        return VALUE_CHAR[self.rank] + self.suit


def parse_card(text: str) -> Card:
    if len(text) != 2 or text[0] not in RANK_VALUE or text[1] not in SUITS:
        raise InvalidCards(f"malformed card {text!r}")
    return Card(RANK_VALUE[text[0]], text[1])


def parse_cards(text: str | Iterable[str]) -> tuple[Card, ...]:
    """Parse ``"As Kd"``, ``"AsKd"`` or an iterable of two-character cards."""
    if isinstance(text, str):
        compact = text.replace(" ", "")
        if len(compact) % 2:
            raise InvalidCards(f"malformed card list {text!r}")
        parts = [compact[i : i + 2] for i in range(0, len(compact), 2)]
    else:
        parts = list(text)
    return tuple(parse_card(p) for p in parts)


def make_deck(ranks: Iterable[int]) -> tuple[Card, ...]:
    return tuple(Card(r, s) for r in ranks for s in SUITS)


def standard_deck() -> tuple[Card, ...]:
    return make_deck(STANDARD_RANKS)


def short_deck() -> tuple[Card, ...]:
    return make_deck(SHORT_DECK_RANKS)


def validate_cards(cards: Iterable[Card], allowed_ranks: Iterable[int]) -> tuple[Card, ...]:
    """Reject duplicates and cards whose rank or suit is not in the declared deck."""
    cards = tuple(cards)
    allowed = frozenset(allowed_ranks)
    for card in cards:
        if not isinstance(card, Card):
            raise InvalidCards(f"not a Card: {card!r}")
        if card.rank not in allowed or card.suit not in SUITS:
            raise InvalidCards(f"card {card} is not in this deck")
    if len(set(cards)) != len(cards):
        raise InvalidCards("duplicate card in " + " ".join(map(str, cards)))
    return cards


def validate_deal(hands: Iterable[Iterable[Card]], board: Iterable[Card], allowed_ranks: Iterable[int]) -> None:
    """A whole deal must be duplicate-free across every hand and the board (E06)."""
    everything: list[Card] = [c for hand in hands for c in hand]
    everything.extend(board)
    validate_cards(everything, allowed_ranks)
