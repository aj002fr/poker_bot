"""Records what a fresh table context looks like: module globals, table info, decisions."""
from arena.agents.sdk import ExplicitStateBot, check_or_call

CONTEXTS = 0


class T12(ExplicitStateBot):
    name = "t12_probe"
    rulesets = ("practice_nlhe",)

    def __init__(self):
        global CONTEXTS
        CONTEXTS += 1
        self.instance_decisions = 0

    def initial_state(self, table):
        return {"module_contexts": CONTEXTS, "table_id": table.table_id, "seat": table.hero_seat,
                "fields": sorted(table.model_dump()), "decisions": 0, "instance_decisions_at_start": self.instance_decisions}

    def decide(self, obs, state):
        self.instance_decisions += 1
        state["decisions"] += 1
        return check_or_call(obs.legal), state


def make_bot():
    return T12()
