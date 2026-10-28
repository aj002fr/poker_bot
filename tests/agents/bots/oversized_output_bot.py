"""Returns an action padded past the 16 KiB reply cap."""
from arena.agents.sdk import ExplicitStateBot


class OversizedOutput(ExplicitStateBot):
    name = "oversized_output"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        return {"type": "check", "pad": "x" * 20000}, state


def make_bot():
    return OversizedOutput()
