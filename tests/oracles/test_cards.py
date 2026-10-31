import pytest

from tests.oracles.cards import (
    SHORT_DECK_RANKS,
    STANDARD_RANKS,
    Card,
    InvalidCards,
    parse_card,
    parse_cards,
    short_deck,
    standard_deck,
    validate_cards,
    validate_deal,
)


def test_parse_round_trip():
    cards = parse_cards("As Td 2c 9h")
    assert cards == (Card(14, "s"), Card(10, "d"), Card(2, "c"), Card(9, "h"))
    assert " ".join(map(str, cards)) == "As Td 2c 9h"
    assert parse_cards("AsTd") == parse_cards(["As", "Td"])


@pytest.mark.parametrize("bad", ["10d", "1s", "Ax", "as", "A", ""])
def test_parse_rejects_malformed(bad):
    with pytest.raises(InvalidCards):
        parse_card(bad)


def test_deck_sizes():
    assert len(set(standard_deck())) == 52
    assert len(set(short_deck())) == 36
    assert {c.rank for c in short_deck()} == set(range(6, 15))


def test_e06_duplicate_card_rejected():
    with pytest.raises(InvalidCards, match="duplicate"):
        validate_cards(parse_cards("As Kd As"), STANDARD_RANKS)


def test_e06_duplicate_across_deal_rejected():
    with pytest.raises(InvalidCards, match="duplicate"):
        validate_deal([parse_cards("As Ad"), parse_cards("Ks As")], parse_cards("2c 3d 7h"), STANDARD_RANKS)


@pytest.mark.parametrize("card", ["2c", "3d", "4h", "5s"])
def test_e06_e08_short_deck_rejects_low_ranks(card):
    with pytest.raises(InvalidCards, match="not in this deck"):
        validate_cards(parse_cards(["As", card]), SHORT_DECK_RANKS)


def test_validate_rejects_non_cards():
    with pytest.raises(InvalidCards):
        validate_cards([(14, "s")], STANDARD_RANKS)
