"""Returns an action that fails the Action schema."""
from arena.agents.sdk import ExplicitStateBot


class InvalidSchema(ExplicitStateBot):
    name = "invalid_schema"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        return {"type": "raise_to", "amount": "lots"}, state


def make_bot():
    return InvalidSchema()
