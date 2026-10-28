"""Baseline: check when free, otherwise fold. Never puts in another chip voluntarily."""

from arena.agents.sdk import ExplicitStateBot, check_or_fold


class CheckFold(ExplicitStateBot):
    name = "check_fold"
    rulesets = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")

    def decide(self, observation, state):
        return check_or_fold(observation.legal), state


def make_bot():
    return CheckFold()
