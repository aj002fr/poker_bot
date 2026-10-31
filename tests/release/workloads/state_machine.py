"""State-machine gate: generated complete hands per ruleset and initial N, each replayed.

Every hand comes from ``tests.property.generated`` (randomized ring, dead
markers, awkward stacks, all eight blind levels, both local-rule settings, the
full legal raise interval, and illegal probes that must be rejected without
mutation). After every accepted transition the section 12.1 invariants are
checked, the finished hand is re-settled by the independent pot ledger and
evaluators, and then the hand is replayed from its spec and the accepted
manual actions alone: the regenerated event sequence must be identical.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from arena.domain.models import Action
from arena.rules.pokerkit_adapter import PokerKitHand
from tests.property.generated import RULESETS, play_generated_hand
from tests.release.workloads.common import CaseRunner, digest

SEED_BASE = 1_000_000
SHARD_HANDS = 5_000
REQUIRED_PER_N = {"practice_nlhe": 100_000, "uncertainty_shortdeck": 25_000, "uncertainty_plo4": 25_000}
REQUIRED_TOTAL = 1_050_000


class HandReplayMismatch(AssertionError):
    pass


def replay_hand(hand: PokerKitHand) -> None:
    """Rebuild the hand from its spec and re-apply only the accepted manual actions."""
    again = PokerKitHand(hand.spec)
    for event in hand.events:
        if event.type == "ActionApplied" and "auto" not in event.data:
            if again.is_complete or again.actor_seat != event.data["seat"]:
                raise HandReplayMismatch(f"replay actor {again.actor_seat} != recorded {event.data['seat']}")
            again.apply(Action.model_validate(event.data["action"]))
    if again.events != hand.events:
        for i, (a, b) in enumerate(zip(hand.events, again.events)):
            if a != b:
                raise HandReplayMismatch(f"event {i}: recorded {a}, replay {b}")
        raise HandReplayMismatch(f"recorded {len(hand.events)} events, replay {len(again.events)}")
    if again.result().final_stacks != hand.result().final_stacks:
        raise HandReplayMismatch("final stacks differ on replay")


def _tally(hand: PokerKitHand, counts: Counter) -> None:
    types = Counter(e.type for e in hand.events)
    result = hand.result()
    counts["hands"] += 1
    counts["decisions"] += sum(1 for e in hand.events if e.type == "ActionApplied" and "auto" not in e.data)
    counts["auto_actions"] += sum(1 for e in hand.events if e.type == "ActionApplied" and "auto" in e.data)
    counts["fold_wins" if result.ended_by_fold else "showdowns"] += 1
    counts["hands_with_side_pots"] += len(result.pots) > 1
    counts["odd_chip_corrections"] += types["OddChipCorrection"]
    counts["excess_returned"] += types["ExcessReturned"]
    counts["nominal_big_blind_calls"] += sum(1 for e in hand.events
                                             if e.type == "ActionApplied" and e.data.get("nominal_big_blind_call"))
    counts["dead_small_blind_hands"] += hand.spec.small_blind_seat is None and not hand.spec.heads_up
    counts["short_big_blind_hands"] += hand.short_blind_correction
    counts["all_in_players"] += sum(1 for s in hand.seats if result.contributions[s] == result.starting_stacks[s])
    counts["replays_verified"] += 1


def _play(case: dict[str, Any], counts: Counter) -> str:
    stats: dict[str, int] = {}
    hand = play_generated_hand(case["seed"], case["ruleset"], case["players"], stats)
    replay_hand(hand)
    counts.update(stats)
    _tally(hand, counts)
    return digest([[e.type, e.data] for e in hand.events])


def run_case(case: dict[str, Any]) -> str:
    return _play(case, Counter())


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    runner = CaseRunner("state_machine")
    for seed in range(params["start"], params["start"] + params["count"]):
        case = {"ruleset": params["ruleset"], "players": params["players"], "seed": seed}
        runner.run(case, lambda c: _play(c, runner.counts))
    return runner.result(seed_range=[params["start"], params["start"] + params["count"] - 1],
                         seed_streams=f"gen-{params['ruleset']}-{params['players']} (spec), policy (actions)")


def shards() -> list[dict[str, Any]]:
    out = []
    for ruleset in RULESETS:
        for n in range(2, 9):
            for k in range(REQUIRED_PER_N[ruleset] // SHARD_HANDS):
                out.append({"gate": "state_machine", "ruleset": ruleset, "players": n,
                            "start": SEED_BASE + k * SHARD_HANDS, "count": SHARD_HANDS})
    return out


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict) -> list[dict[str, Any]]:
    checks = []
    per_cell: Counter = Counter()
    totals: Counter = Counter()
    for r in results:
        per_cell[(r["params"]["ruleset"], r["params"]["players"])] += r["counts"].get("hands", 0)
        totals.update(r["counts"])
    for ruleset in RULESETS:
        need = REQUIRED_PER_N[ruleset]
        cells = {n: per_cell[(ruleset, n)] for n in range(2, 9)}
        checks.append({"check": f"{ruleset}: >= {need:,} complete hands for each N=2..8",
                       "passed": all(v >= need for v in cells.values()),
                       "detail": {str(n): v for n, v in cells.items()}})
    checks.append({"check": f"total complete hands >= {REQUIRED_TOTAL:,}", "passed": totals["hands"] >= REQUIRED_TOTAL,
                   "detail": f"{totals['hands']:,}"})
    checks.append({"check": "every hand replayed to an identical event sequence",
                   "passed": totals["replays_verified"] == totals["hands"],
                   "detail": f"{totals['replays_verified']:,} replays"})
    return checks
