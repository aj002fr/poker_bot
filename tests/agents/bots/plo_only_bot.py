"""Declares support for PLO only."""
from arena.agents.sdk import ExplicitStateBot, check_or_fold


class PloOnly(ExplicitStateBot):
    name = "plo_only"
    rulesets = ("uncertainty_plo4",)

    def decide(self, obs, state):
        return check_or_fold(obs.legal), state


def make_bot():
    return PloOnly()
