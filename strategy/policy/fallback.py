"""Constant-time legal fallback, and the legalization of a policy intent.

:func:`fallback_action` is what the runtime plays when anything on the decision path
raises or produces an action the legal set does not allow: check when checking is
free, otherwise fold. Every arena decision allows one of the two, so it is always
legal and never inspects history, cards or state.

:func:`legalize` maps an :class:`~strategy.policy.baseline.Intent` onto the legal set
without ever choosing a stronger action than intended: a fold with a free check
checks, a call with nothing to call checks, a raise that is not available calls (or
checks), and a raise target is clamped to the closest legal amount.
"""

from __future__ import annotations

from typing import Any

from arena.agents.sdk import Action, LegalActions, clamp_raise_to

CHECK = Action(type="check")
FOLD = Action(type="fold")
CALL = Action(type="call")


def fallback_action(legal: LegalActions) -> Action:
    return CHECK if legal.check else FOLD


def legalize(kind: str, amount: int | None, legal: LegalActions, all_in_to: int) -> Action:
    """``all_in_to``: the hero's street total if every chip goes in (stack + street commitment)."""
    if kind == "fold" or kind == "check":
        return CHECK if legal.check else FOLD
    if kind == "call":
        if legal.check:
            return CHECK
        return CALL if legal.call_amount is not None else FOLD
    if kind == "raise" or kind == "allin":
        if legal.can_raise:
            target = all_in_to if kind == "allin" or amount is None else amount
            act = clamp_raise_to(legal, target)
            if act is not None:
                return act
        if legal.check:
            return CHECK
        return CALL if legal.call_amount is not None else FOLD
    raise ValueError(f"unknown intent kind {kind!r}")


def is_legal(action: Any, legal: LegalActions) -> bool:
    return isinstance(action, Action) and legal.allows(action)
