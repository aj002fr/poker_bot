"""Always raises to 1 chip, which is never a legal raise."""
from arena.agents.sdk import ExplicitStateBot, raise_to


class Illegal(ExplicitStateBot):
    name = "illegal"
    rulesets = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")

    def decide(self, obs, state):
        return raise_to(1), {"n": state["n"] + 1}

    def initial_state(self, table):
        return {"n": 0}


def make_bot():
    return Illegal()
