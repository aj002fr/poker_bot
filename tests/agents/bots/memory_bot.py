"""Tries to allocate 4 GiB on every decision, beyond the address-space limit."""
from arena.agents.sdk import ExplicitStateBot


class Memory(ExplicitStateBot):
    name = "memory"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        blob = bytearray(4 << 30)
        return {"type": "fold"}, len(blob)


def make_bot():
    return Memory()
