"""Writes garbage, then a frame header announcing 1 MiB, straight to the reply pipe."""
import os
import struct
import sys

from arena.agents.sdk import ExplicitStateBot

OUT_FD = int(sys.argv[sys.argv.index("--out-fd") + 1])


class Forged(ExplicitStateBot):
    name = "forged_frame"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        if int(obs.decision_id[2:]) % 2:
            os.write(OUT_FD, struct.pack(">II", 2, 0) + b"[]")
        else:
            os.write(OUT_FD, struct.pack(">II", 10, 1 << 20))
        while True:
            pass


def make_bot():
    return Forged()
