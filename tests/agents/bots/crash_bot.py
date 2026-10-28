"""Kills its own process on every decision whose number is divisible by 7."""
import os

from arena.agents.sdk import ExplicitStateBot, check_or_call


class Crash(ExplicitStateBot):
    name = "crash"
    rulesets = ("practice_nlhe",)

    def initial_state(self, table):
        return {"accepted": 0}

    def decide(self, obs, state):
        if int(obs.decision_id[2:]) % 7 == 0:
            os._exit(3)
        return check_or_call(obs.legal), {"accepted": state["accepted"] + 1}


def make_bot():
    return Crash()
