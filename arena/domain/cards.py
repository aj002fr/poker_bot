"""Card strings, per-ruleset decks, and deck validation.

Cards are two-character strings: rank (``23456789TJQKA``) then suit (``cdhs``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

RANKS = "23456789TJQKA"
SHORT_DECK_RANKS = "6789TJQKA"
SUITS = "cdhs"


class DeckError(ValueError):
    """An impossible deck or deal: duplicate card, wrong rank, or exhaustion."""


def full_deck(ranks: str) -> tuple[str, ...]:
    return tuple(r + s for r in ranks for s in SUITS)


def parse_cards(text: str | Iterable[str]) -> tuple[str, ...]:
    """Parse ``"AsKd"``, ``"As Kd"`` or ``["As", "Kd"]`` into card strings."""
    if isinstance(text, str):
        compact = text.replace(" ", "")
        if len(compact) % 2:
            raise DeckError(f"malformed card string {text!r}")
        items = [compact[i : i + 2] for i in range(0, len(compact), 2)]
    else:
        items = list(text)
    for card in items:
        if (
            not isinstance(card, str)
            or len(card) != 2
            or card[0] not in RANKS
            or card[1] not in SUITS
        ):
            raise DeckError(f"malformed card {card!r}")
    return tuple(items)


def validate_deck(deck: Sequence[str], ranks: str) -> None:
    """Require a complete permutation of the ruleset's deck."""
    expected = set(full_deck(ranks))
    cards = parse_cards(deck)
    seen: set[str] = set()
    for card in cards:
        if card in seen:
            raise DeckError(f"duplicate card {card}")
        if card[0] not in ranks:
            raise DeckError(f"card {card} has a rank outside this ruleset ({ranks})")
        seen.add(card)
    if seen != expected:
        missing = sorted(expected - seen)
        raise DeckError(f"deck is incomplete; missing {missing}")
