"""Loops forever on every decision whose number is divisible by 5."""
from arena.agents.sdk import ExplicitStateBot, check_or_call


class Loop(ExplicitStateBot):
    name = "loop"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        if int(obs.decision_id[2:]) % 5 == 0:
            while True:
                pass
        return check_or_call(obs.legal), state


def make_bot():
    return Loop()
