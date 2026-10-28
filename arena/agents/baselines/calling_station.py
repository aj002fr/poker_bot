"""Baseline: check when free, otherwise call any bet. Never raises or folds."""

from arena.agents.sdk import ExplicitStateBot, check_or_call


class CallingStation(ExplicitStateBot):
    name = "calling_station"
    rulesets = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")

    def decide(self, observation, state):
        return check_or_call(observation.legal), state


def make_bot():
    return CallingStation()
