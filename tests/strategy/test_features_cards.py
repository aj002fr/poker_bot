import random
from itertools import combinations

import pokerkit as pk
import pytest

from strategy.features.cards import (CARD_NAMES, DRAW_CLASSES, FLUSH, FULL_HOUSE, MADE_LEVELS, POSTFLOP_CELLS,
                                     TEXTURE_KEYS, canonical_hole_key, capabilities, card_ints, draws, evaluate,
                                     hole_label, made_hand, postflop_key, street_delta, texture)
from tests.strategy.feature_support import RULESETS

POKERKIT_HAND = {"practice_nlhe": pk.StandardHighHand, "uncertainty_shortdeck": pk.ShortDeckHoldemHand,
                 "uncertainty_plo4": pk.OmahaHoldemHand}
NL = capabilities("practice_nlhe")
SD = capabilities("uncertainty_shortdeck")
PLO = capabilities("uncertainty_plo4")


def c(*names: str) -> tuple[int, ...]:
    return card_ints(names)


@pytest.mark.parametrize("ruleset_id", RULESETS)
def test_evaluator_matches_pokerkit(ruleset_id: str) -> None:
    caps = capabilities(ruleset_id)
    hand = POKERKIT_HAND[ruleset_id]
    rng = random.Random(ruleset_id)
    s = lambda xs: "".join(CARD_NAMES[x] for x in xs)  # noqa: E731
    for _ in range(800):
        cards = rng.sample(caps.deck, 2 * caps.hole_count + 5)
        a, b, board = cards[:caps.hole_count], cards[caps.hole_count:2 * caps.hole_count], cards[-5:]
        ha, hb = hand.from_game(s(a), s(board)), hand.from_game(s(b), s(board))
        va, vb = evaluate(a, board, caps), evaluate(b, board, caps)
        assert (ha > hb) - (ha < hb) == (va > vb) - (va < vb), (s(a), s(b), s(board))


@pytest.mark.parametrize("ruleset_id", RULESETS)
def test_capabilities_agree_with_the_arena_profiles(ruleset_id: str) -> None:
    from arena.rules.profiles import get_profile

    p, caps = get_profile(ruleset_id), capabilities(ruleset_id)
    assert (caps.ranks, caps.hole_count) == (p.ranks, p.hole_card_count)
    assert (caps.hole_cards_used == 2) == (p.hand_selection == "two_hole_three_board")
    assert (caps.category_order != NL.category_order) == (p.short_deck_ranking == "pokerkit_native")


def test_capabilities_are_metadata() -> None:
    assert (NL.hole_count, NL.hole_cards_used, NL.deck_size) == (2, None, 52)
    assert (SD.hole_count, SD.deck_size) == (2, 36)
    assert (PLO.hole_count, PLO.hole_cards_used) == (4, 2)
    assert SD.strength[FLUSH] > SD.strength[FULL_HOUSE]
    assert NL.strength[FULL_HOUSE] > NL.strength[FLUSH]


@pytest.mark.parametrize(("ruleset_id", "classes"), [("practice_nlhe", 169), ("uncertainty_shortdeck", 81)])
def test_canonical_class_counts(ruleset_id: str, classes: int) -> None:
    caps = capabilities(ruleset_id)
    assert len({canonical_hole_key(h) for h in combinations(caps.deck, 2)}) == classes


def test_canonical_key_is_suit_invariant() -> None:
    rng = random.Random(5)
    for caps in (NL, PLO):
        for _ in range(300):
            hole = rng.sample(caps.deck, caps.hole_count)
            perm = rng.sample(range(4), 4)
            moved = [(x >> 2) * 4 + perm[x & 3] for x in hole]
            assert canonical_hole_key(hole) == canonical_hole_key(moved)
    assert canonical_hole_key(c("Ah", "Kh", "Qd", "Jd")) != canonical_hole_key(c("Ah", "Kd", "Qh", "Jd"))
    assert [hole_label(c(*h)) for h in (("As", "Kd"), ("9h", "Th"), ("7c", "7d"))] == ["AKo", "T9s", "77"]


def test_made_hand_levels() -> None:
    assert made_hand(c("Ah", "Kd"), c("As", "7h", "2c"), NL).level == 3  # top pair
    assert made_hand(c("Qh", "Qd"), c("Js", "7h", "2c"), NL).level == 4  # overpair
    assert made_hand(c("7d", "7c"), c("7s", "Kh", "2c"), NL).level == 6  # set
    assert made_hand(c("4d", "3c"), c("As", "Ah", "Kc", "Kd", "Qs"), NL).level == 0  # plays the board
    assert made_hand(c("Kh", "Qh", "2d", "3c"), c("Ah", "7h", "9h"), PLO).level >= 8  # flush, two hole hearts
    assert made_hand(c("Kh", "Qd", "2d", "3c"), c("Ah", "7h", "9h", "4h"), PLO).category != FLUSH


def test_draws_respect_variant_rules() -> None:
    nfd = draws(c("Kh", "Qh"), c("Ah", "7h", "2c"), NL)
    assert nfd.flush_draw and nfd.nut_flush_draw and nfd.outs == 9
    oesd = draws(c("9d", "8c"), c("7h", "6s", "2c"), NL)
    assert oesd.oesd and oesd.outs == 8
    gut = draws(c("9d", "8c"), c("6h", "5s", "Kc"), NL)
    assert gut.gutshot and not gut.oesd and gut.outs == 4
    assert not draws(c("Kh", "Qd", "Jc", "Tc"), c("Ah", "7h", "2c"), PLO).flush_draw  # one heart is not enough
    assert draws(c("Kh", "Qh", "Jc", "Td"), c("Ah", "7h", "2c"), PLO).flush_draw
    assert draws(c("8d", "7c"), c("6h", "9s", "Kc"), SD).oesd  # T or the short-deck A-6-7-8-9
    assert draws(c("Kh", "Qh"), c("Ah", "7h", "2c", "3d", "9s"), NL).outs == 0  # river


def test_texture_and_turn_delta() -> None:
    mono = texture(c("Ah", "7h", "2h"), NL)
    assert mono.flush_possible and mono.max_suit == 3 and mono.nut_category == FLUSH
    paired = texture(c("Ks", "Kd", "4c"), NL)
    assert paired.paired and not paired.flush_draw_possible
    conn = texture(c("9s", "8d", "7c"), NL)
    assert conn.straight_windows >= 1 and conn.dynamic > texture(c("Ks", "7d", "2c"), NL).dynamic
    flop, turn = c("Ah", "7h", "2c"), c("Ah", "7h", "2c", "9h")
    d = street_delta(texture(flop, NL), texture(turn, NL), turn[-1], flop)
    assert d.completes_flush and d.nut_up and not d.pairs_board
    d2 = street_delta(texture(flop, NL), texture(c("Ah", "7h", "2c", "7d"), NL), c("7d")[0], flop)
    assert d2.pairs_board


def test_postflop_keys_cover_the_table() -> None:
    keys = {postflop_key(s, m, d, t) for s in (1, 2, 3) for m in range(MADE_LEVELS) for d in range(DRAW_CLASSES)
            for t in range(TEXTURE_KEYS)}
    assert keys == set(range(POSTFLOP_CELLS))
    rng = random.Random(2)
    for caps in (NL, SD, PLO):
        for _ in range(200):
            cards = rng.sample(caps.deck, caps.hole_count + 3)
            tx = texture(cards[caps.hole_count:], caps)
            assert 0 <= tx.key < TEXTURE_KEYS
