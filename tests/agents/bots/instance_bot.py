"""Instance-state bot: counts its decisions in an attribute (lost on restart)."""
from arena.agents.sdk import InstanceStateBot, check_or_call


class Counting(InstanceStateBot):
    name = "instance_counter"
    rulesets = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")

    def start_table(self, table):
        self.seen = 0

    def decide(self, obs):
        self.seen += 1
        return check_or_call(obs.legal)


def make_bot():
    return Counting()
