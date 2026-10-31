"""Fast gate: 100 generated hands per initial N=2..8 per scenario, fixed seeds."""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.property.generated import RULESETS, play_generated_hand

HANDS_PER_CELL = 100


@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("ruleset", RULESETS)
def test_generated_hands(ruleset, n):
    for seed in range(HANDS_PER_CELL):
        try:
            play_generated_hand(seed, ruleset, n)
        except Exception as exc:
            raise AssertionError(f"failing seed: play_generated_hand({seed}, {ruleset!r}, {n})") from exc


@settings(max_examples=150, deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**31), ruleset=st.sampled_from(RULESETS), n=st.integers(2, 8))
def test_generated_hands_hypothesis(seed, ruleset, n):
    play_generated_hand(seed, ruleset, n)


@pytest.mark.release
@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("ruleset", RULESETS)
def test_generated_hands_release_smoke(ruleset, n):
    for seed in range(HANDS_PER_CELL, HANDS_PER_CELL + 2000):
        play_generated_hand(seed, ruleset, n)
