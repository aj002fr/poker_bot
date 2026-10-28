"""Returns a state just over the 64 KiB cap on even decisions; exactly at the cap otherwise."""
from arena.agents.sdk import MAX_STATE_BYTES, ExplicitStateBot, check_or_call


class OversizedState(ExplicitStateBot):
    name = "oversized_state"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        n = int(obs.decision_id[2:])
        size = MAX_STATE_BYTES - 2 + (1 if n % 2 == 0 else 0)  # the JSON string adds 2 quote bytes
        return check_or_call(obs.legal), "x" * size


def make_bot():
    return OversizedState()
