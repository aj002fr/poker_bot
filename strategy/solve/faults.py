"""Deliberate solver bugs, used only by mutation tests to prove the gates bite.

Production code always runs with ``Fault.NONE``. Each other member reproduces
one classic CFR implementation error so the Stage 2 tests can show that the
gate meant to catch it actually fails.
"""

from __future__ import annotations

from enum import IntEnum


class Fault(IntEnum):
    NONE = 0
    # The solver hands back its last iterate instead of the reach-weighted average.
    LAST_ITERATE = 1
    # Counterfactual regret weighted by the acting player's own reach.
    REGRET_OWN_REACH = 2
    # Strategy sums ignore the acting player's reach (plain average of iterates).
    AVERAGE_UNWEIGHTED = 3
    # Every player's regret is computed from player 0's utility.
    PAYOFF_ORIENTATION = 4
    # Chance probability dropped from counterfactual reach (card-weighting error).
    IGNORE_CHANCE_REACH = 5
    # Player reach reset after each non-root chance node (isolated-street solving).
    RESET_REACH_AT_CHANCE = 6
    # External sampling draws opponent actions uniformly instead of on-policy.
    ES_UNIFORM_OPPONENT = 7
    # External sampling draws chance outcomes uniformly instead of by probability.
    ES_UNIFORM_CHANCE = 8
    # Traverser samples one on-policy action without importance weighting (no exploration).
    ES_NO_EXPLORATION = 9


FULL_TRAVERSAL_FAULTS = (
    Fault.LAST_ITERATE,
    Fault.REGRET_OWN_REACH,
    Fault.AVERAGE_UNWEIGHTED,
    Fault.PAYOFF_ORIENTATION,
    Fault.IGNORE_CHANCE_REACH,
    Fault.RESET_REACH_AT_CHANCE,
)

SAMPLING_FAULTS = (
    Fault.LAST_ITERATE,
    Fault.ES_UNIFORM_OPPONENT,
    Fault.ES_UNIFORM_CHANCE,
    Fault.ES_NO_EXPLORATION,
)
