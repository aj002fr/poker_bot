"""In-process drivers for headless play and generated tests.

These are not the agent SDK baselines (milestone 4); they call the same
:class:`~arena.table.service.Table` transitions without any worker isolation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from arena.domain.models import Action, LegalActions
from arena.domain.rng import Rng, SeededRng

Policy = Callable[[int, LegalActions, Rng], Action]


def legal_random(seat: int, legal: LegalActions, rng: Rng) -> Action:
    """Uniform over action kinds, then uniform over the full legal raise interval."""
    options: list[str] = []
    if legal.fold:
        options.append("fold")
    if legal.check:
        options.append("check")
    if legal.call_amount is not None:
        options.append("call")
    if legal.can_raise:
        options.append("raise")
    kind = options[rng.randbelow(len(options))]
    if kind != "raise":
        return Action(type=kind)
    targets: list[int] = []
    if legal.short_all_in_raise_to is not None:
        targets.append(legal.short_all_in_raise_to)
    if legal.min_full_raise_to is not None and legal.max_raise_to is not None:
        lo, hi = legal.min_full_raise_to, legal.max_raise_to
        pick = rng.randbelow(4)
        if pick == 0:
            targets.append(lo)
        elif pick == 1:
            targets.append(hi)
        else:
            targets.append(lo + rng.randbelow(hi - lo + 1))
    return Action(type="raise_to", amount=targets[rng.randbelow(len(targets))])


def check_or_fold(seat: int, legal: LegalActions, rng: Rng) -> Action:
    return Action(type="check") if legal.check else Action(type="fold")


def check_or_call(seat: int, legal: LegalActions, rng: Rng) -> Action:
    return Action(type="check") if legal.check else Action(type="call")


def shove(seat: int, legal: LegalActions, rng: Rng) -> Action:
    if legal.all_in_raise_to is not None:
        return Action(type="raise_to", amount=legal.all_in_raise_to)
    return Action(type="call") if legal.call_amount is not None else Action(type="check")


POLICIES: dict[str, Policy] = {
    "legal_random": legal_random,
    "check_or_fold": check_or_fold,
    "check_or_call": check_or_call,
    "shove": shove,
}


def policy_rngs(seed: int | str, seats: Sequence[int]) -> dict[int, Rng]:
    """One independent policy stream per seat, separate from the environment stream."""
    return {s: SeededRng(seed, f"policy-seat-{s}") for s in seats}


def play_out(
    table,
    policy: Policy | Mapping[int, Policy],
    rng: Rng | Mapping[int, Rng],
    *,
    on_decision=None,
) -> None:
    while not table.complete:
        seat = table.actor_seat
        legal = table.legal_actions()
        chooser = policy[seat] if isinstance(policy, Mapping) else policy
        seat_rng = rng[seat] if isinstance(rng, Mapping) else rng
        action = chooser(seat, legal, seat_rng)
        if on_decision is not None:
            on_decision(table, seat, legal, action)
        table.submit(seat, action, hand_id=table.hands[-1].hand_id, decision_id=table.decision_id)
