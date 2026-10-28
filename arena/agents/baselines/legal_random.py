"""Baseline: a uniformly random legal action kind, then a random legal raise target.

Its random stream lives in the explicit state (``seed`` from ``TableInfo.policy_seed``
plus a draw counter), so a restored worker continues the same sequence.
"""

import hashlib

from arena.agents.sdk import Action, ExplicitStateBot, raise_to


def _draw(state, n):
    """Uniform integer in [0, n) from SHA-256 counter mode; advances ``state['n']``."""
    limit = (1 << 64) - ((1 << 64) % n)
    while True:
        block = hashlib.sha256(f"{state['seed']}|{state['n']}".encode()).digest()
        state["n"] += 1
        value = int.from_bytes(block[:8], "big")
        if value < limit:
            return value % n


class LegalRandom(ExplicitStateBot):
    name = "legal_random"
    rulesets = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")

    def initial_state(self, table):
        return {"seed": table.policy_seed, "n": 0}

    def decide(self, observation, state):
        legal = observation.legal
        kinds = [k for k, ok in (("fold", legal.fold), ("check", legal.check),
                                 ("call", legal.call_amount is not None), ("raise", legal.can_raise)) if ok]
        kind = kinds[_draw(state, len(kinds))]
        if kind != "raise":
            return Action(type=kind), state
        targets = []
        if legal.short_all_in_raise_to is not None:
            targets.append(legal.short_all_in_raise_to)
        if legal.min_full_raise_to is not None and legal.max_raise_to is not None:
            lo, hi = legal.min_full_raise_to, legal.max_raise_to
            pick = _draw(state, 4)
            targets.append(lo if pick == 0 else hi if pick == 1 else lo + _draw(state, hi - lo + 1))
        return raise_to(targets[_draw(state, len(targets))]), state


def make_bot():
    return LegalRandom()
