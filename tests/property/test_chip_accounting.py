"""A raise to a street total debits only the difference from what the seat already has in front of it."""

from __future__ import annotations

import pytest

from arena.domain.models import Action
from tests.support.fixtures import make_hand


def _stacks(hand) -> dict[int, tuple[int, int]]:
    return {p.seat: (p.stack, p.street_committed) for p in hand.players()}


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4"])
def test_raise_to_debits_only_the_delta(ruleset):
    hand = make_hand({"ruleset": ruleset, "setup": {"stacks": {"0": 2000, "1": 2000, "2": 2000}, "button": 0,
                                                    "small_blind_seat": 1, "big_blind_seat": 2}})
    assert _stacks(hand) == {0: (2000, 0), 1: (1990, 10), 2: (1980, 20)}
    hand.apply(Action(type="call"))
    assert _stacks(hand)[0] == (1980, 20)
    hand.apply(Action(type="raise_to", amount=80))
    assert _stacks(hand)[1] == (1920, 80), "SB with 10 posted raising to 80 adds 70"
    target = hand.legal_actions().min_full_raise_to
    assert target == 140
    hand.apply(Action(type="raise_to", amount=target))
    assert _stacks(hand)[2] == (2000 - target, target), "BB with 20 posted raising to 140 adds 120"
    hand.apply(Action(type="raise_to", amount=hand.legal_actions().min_full_raise_to))
    assert _stacks(hand)[0] == (2000 - 200, 200), "a limper re-raising to 200 adds 180"
    added = [e.data["added"] for e in hand.events if e.type == "ActionApplied"]
    assert added == [20, 70, 120, 180]
    assert sum(s for s, _ in _stacks(hand).values()) + sum(c for _, c in _stacks(hand).values()) == 6000
