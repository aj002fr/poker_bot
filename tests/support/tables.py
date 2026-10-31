"""Seeded in-process tables for history, replay and privacy tests."""

from __future__ import annotations

from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng
from arena.table.driver import legal_random, policy_rngs
from arena.table.service import Table, seeded_table

RULESETS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")


def play_table(ruleset: str, n: int, seed: str, *, fallback_every: int = 0, stack: int = 400,
               table_id: str | None = None, **config_kw) -> Table:
    """A full seeded table. Every ``fallback_every``-th decision is consumed by a timeout fallback."""
    config = TournamentConfig(ruleset_id=ruleset, initial_player_count=n, starting_stack=stack, **config_kw)
    table = seeded_table(config, seed, table_id=table_id or f"t_{ruleset}_{n}_{seed}")
    rngs = policy_rngs(seed, config.resolved_seats)
    decisions = 0
    while not table.complete:
        seat = table.actor_seat
        decisions += 1
        if fallback_every and decisions % fallback_every == 0:
            table.apply_fallback(seat, "timeout", {"late_by_ns": 1234, "payload": {"type": "raise_to"}})
            continue
        table.submit(seat, legal_random(seat, table.legal_actions(), rngs[seat]),
                     hand_id=table.hands[-1].hand_id, decision_id=table.decision_id,
                     expected_state_version=table.state_version)
    return table


def policy_rng(seed: str) -> SeededRng:
    return SeededRng(seed, "test-policy")
