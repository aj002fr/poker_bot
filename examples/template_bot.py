"""Template bot: copy this file, change ``decide``, then run

    uv run python -m arena.cli match --bot path/to/your_bot.py

Rules of the local SDK (see docs/agent-sdk.md):

* ``make_bot()`` returns your bot. The arena calls it once per table in a fresh,
  sandboxed process: no network, no filesystem writes, no subprocesses, ~1 GiB of
  address space. Import only the standard library, ``arena.agents.sdk`` and
  packages already installed in the arena's environment.
* Every decision must come back within 1 ms, measured by the parent from sending
  the observation to receiving your complete reply. Late, invalid or illegal
  replies become check (or fold, when checking is not legal).
* Keep everything you need between decisions in ``state`` (JSON, at most 64 KiB).
  Returned state is committed only together with an accepted action, and a crashed
  worker is restored from it, so the bot replays exactly. Do not use the clock or
  any randomness that is not derived from ``table.policy_seed``.
* Do your imports and any precomputation in ``make_bot()``/``initial_state()``, which
  run before the first decision and are not timed.
"""

import hashlib

from arena.agents.sdk import (
    ExplicitStateBot,
    Observation,
    TableInfo,
    call,
    check,
    check_or_fold,
    clamp_raise_to,
)

RANKS = "23456789TJQKA"


class TemplateBot(ExplicitStateBot):
    name = "template_bot"
    rulesets = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")

    def initial_state(self, table: TableInfo):
        return {"seed": table.policy_seed, "draws": 0, "hands_seen": 0, "last_hand": None}

    def _random(self, state) -> float:
        """A uniform number in [0, 1) whose stream lives in ``state``."""
        digest = hashlib.sha256(f"{state['seed']}|{state['draws']}".encode()).digest()
        state["draws"] += 1
        return int.from_bytes(digest[:8], "big") / 2**64

    def decide(self, obs: Observation, state):
        if obs.hand_id != state["last_hand"]:
            state["last_hand"] = obs.hand_id
            state["hands_seen"] += 1
        legal = obs.legal
        to_call = legal.call_amount or 0
        high = max(RANKS.index(c[0]) for c in obs.hole_cards)
        paired = len({c[0] for c in obs.hole_cards}) < len(obs.hole_cards)
        strong = paired or high >= RANKS.index("K")

        if strong and self._random(state) < 0.5:
            bet = clamp_raise_to(legal, obs.current_bet_to + obs.pot_total // 2)
            if bet is not None:
                return bet, state
        if strong or to_call * 4 <= obs.pot_total:
            return (call() if to_call else check()), state
        return check_or_fold(legal), state


def make_bot():
    return TemplateBot()
