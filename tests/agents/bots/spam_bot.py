"""Prints about 1 MB per decision through print() and raw fd writes, then checks or calls."""
import os

from arena.agents.sdk import ExplicitStateBot, check_or_call


class Spam(ExplicitStateBot):
    name = "spam"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        print("y" * 500_000)
        try:
            os.write(1, b"z" * 500_000)
        except OSError:
            pass
        return check_or_call(obs.legal), state


def make_bot():
    return Spam()
