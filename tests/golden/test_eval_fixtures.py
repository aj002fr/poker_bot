import pytest

from arena.domain.cards import DeckError, full_deck
from arena.rules.pokerkit_adapter import HandSpec, PokerKitHand
from arena.rules.profiles import get_profile
from tests.support.fixtures import load_fixtures, run_fixture

EVAL = load_fixtures("eval")


@pytest.mark.parametrize("fixture", EVAL, ids=[f["id"] for f in EVAL])
def test_eval_fixture(fixture):
    run_fixture(fixture)


def _spec(ruleset, deck):
    return HandSpec(get_profile(ruleset), (0, 1), {0: 2000, 1: 2000}, 0, 0, 1, 10, 20, tuple(deck))


@pytest.mark.parametrize(
    "ruleset,mutate",
    [
        ("practice_nlhe", lambda d: d[:1] + d[:51]),
        ("practice_nlhe", lambda d: d[:51]),
        ("uncertainty_shortdeck", lambda d: ["2c"] + d[1:]),
        ("uncertainty_shortdeck", lambda d: d + ["5h"]),
        ("uncertainty_plo4", lambda d: d[:50] + ["Xx", d[50]]),
    ],
    ids=["duplicate", "missing", "short-deck-rank-2", "short-deck-extra-5", "malformed"],
)
def test_e06_impossible_decks_rejected_before_play(ruleset, mutate):
    deck = mutate(list(full_deck(get_profile(ruleset).ranks)))
    with pytest.raises(DeckError):
        PokerKitHand(_spec(ruleset, deck))
