"""Raises on every decision whose number is divisible by 3; otherwise checks or calls."""
from arena.agents.sdk import ExplicitStateBot, check_or_call


class Raising(ExplicitStateBot):
    name = "raising"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        if int(obs.decision_id[2:]) % 3 == 0:
            raise ValueError("deliberate bot error")
        return check_or_call(obs.legal), state


def make_bot():
    return Raising()
