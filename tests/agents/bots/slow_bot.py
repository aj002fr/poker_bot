"""Sleeps 3 ms (three budgets) on every decision whose number is divisible by 4."""
import time

from arena.agents.sdk import ExplicitStateBot, check_or_call


class Slow(ExplicitStateBot):
    name = "slow"
    rulesets = ("practice_nlhe",)

    def initial_state(self, table):
        return {"accepted": 0}

    def decide(self, obs, state):
        if int(obs.decision_id[2:]) % 4 == 0:
            time.sleep(0.003)
            return check_or_call(obs.legal), {"accepted": state["accepted"] + 1, "late": True}
        return check_or_call(obs.legal), {"accepted": state["accepted"] + 1}


def make_bot():
    return Slow()
