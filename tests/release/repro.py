"""Reproduce recorded failing cases and shrink failing generated hands into fixtures.

``python -m tests.release repro CASE_JSON`` reruns one case exactly as its
shard ran it (every recorded failure carries this command).

``python -m tests.release minimize --ruleset R --players N --seed S [--write]``
records the generated hand for that seed, then shrinks it while the *same*
failure (exception type and message up to the first colon, or digits removed)
persists: seats that are not the button or a blind are removed, and stacks are
trimmed. The result is a fixture in the golden format whose expected pots,
contributions and final stacks come from the independent oracles, never from
the engine under test. ``--write`` saves it under ``tests/regressions/fixtures``
and retains the seed in ``tests/regressions/failing_seeds.json``; both run in
the fast suite.
"""

from __future__ import annotations

import json
import re
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena.domain.models import Action
from arena.rules.base import IllegalAction
from arena.rules.pokerkit_adapter import PokerKitHand

ROOT = Path(__file__).resolve().parents[2]
REGRESSIONS = ROOT / "tests" / "regressions"
SEEDS_FILE = REGRESSIONS / "failing_seeds.json"
FIXTURE_DIR = REGRESSIONS / "fixtures"


def repro(case_json: str, *, trace: bool = False) -> int:
    from tests.release.harness import GATE_BY_ID

    case = json.loads(case_json)
    gate = GATE_BY_ID[case["gate"]]
    try:
        result = gate.workload().run_case(case)
    except Exception as exc:
        if trace:
            traceback.print_exc()
        print(f"FAILED {case_json}\n  {type(exc).__name__}: {exc}")
        return 1
    print(f"passed {case_json}\n  result digest {result}")
    return 0


# ------------------------------------------------------------- recording


class _Recorder:
    """Captures the accepted actions of one generated hand, including the one that fails."""

    def __init__(self) -> None:
        self.hand: PokerKitHand | None = None
        self.steps: list[dict[str, Any]] = []

    def cls(self):
        rec = self

        class Recording(PokerKitHand):
            def __init__(self, spec) -> None:
                rec.hand = self
                super().__init__(spec)

            def apply(self, action: Action, *, auto_reason: str | None = None):
                if auto_reason is not None:  # the adapter's own automatic actions are not steps
                    return super().apply(action, auto_reason=auto_reason)
                step = {"actor": self.actor_seat, "do": action.model_dump(exclude_none=True)}
                rec.steps.append(step)
                try:
                    return super().apply(action)
                except IllegalAction:
                    rec.steps.remove(step)
                    raise

        return Recording


def signature(exc: BaseException) -> str:
    """Exception type and message head, digits masked, so a shrunk case must fail the same way."""
    head = re.sub(r"\d+", "#", str(exc).split(":")[0])
    return f"{type(exc).__name__}: {head}"[:120]


def record_generated(seed: int, ruleset: str, players: int) -> tuple[dict[str, Any], str | None]:
    """The generated hand as a fixture, and the failure signature (None when it passes)."""
    from tests.property import generated

    rec = _Recorder()
    original = generated.PokerKitHand
    generated.PokerKitHand = rec.cls()
    error = None
    try:
        generated.play_generated_hand(seed, ruleset, players)
    except Exception as exc:
        error = signature(exc)
    finally:
        generated.PokerKitHand = original
    hand = rec.hand
    assert hand is not None
    spec = hand.spec
    fx = {
        "id": f"R-{ruleset}-{players}-{seed}",
        "kind": "full_trace",
        "ruleset": ruleset,
        "setup": {
            "seats": list(spec.seat_ring),
            "stacks": {str(s): v for s, v in spec.stacks.items()},
            "button": spec.button,
            "small_blind_seat": spec.small_blind_seat,
            "big_blind_seat": spec.big_blind_seat,
            "blinds": [spec.small_blind, spec.big_blind],
            "hole": {str(s): "".join(hand.hole_cards[s]) for s in hand.seats},
            "board": "".join(_full_board(hand)),
        },
        "local_rules": {"short_bet_reopens_checker": spec.local_rules.short_bet_reopens_checker},
        "steps": rec.steps,
        "origin": {"generator": "tests.property.generated.play_generated_hand", "seed": seed, "ruleset": ruleset,
                   "players": players},
    }
    return fx, error


def _full_board(hand: PokerKitHand) -> list[str]:
    """The five board cards this deck deals (the hand may have ended before the river)."""
    ledger = hand.card_ledger()
    board = list(ledger["board"])
    undealt = list(ledger["undealt"])
    while len(board) < 5 and undealt:
        undealt.pop(0)  # burn
        take = 3 if not board else 1
        board += undealt[:take]
        undealt = undealt[take:]
    return board


# ------------------------------------------------------------- shrinking


def replay_fixture(fx: dict[str, Any]) -> tuple[str, PokerKitHand | None]:
    """``("invalid", None)`` when the steps no longer apply, ``(signature, hand)`` on a failure,
    ``("ok", hand)`` when the hand completes."""
    from tests.support.fixtures import make_hand

    try:
        hand = make_hand(fx)
    except Exception:
        return "invalid", None
    for step in fx["steps"]:
        if hand.is_complete or hand.actor_seat != step["actor"]:
            return "invalid", None
        try:
            hand.apply(Action.model_validate(step["do"]))
        except IllegalAction:
            return "invalid", None
        except Exception as exc:
            return signature(exc), hand
    return ("ok" if hand.is_complete else "invalid"), hand


def _without_seat(fx: dict[str, Any], seat: int) -> dict[str, Any]:
    s = fx["setup"]
    out = json.loads(json.dumps(fx))
    o = out["setup"]
    if seat in (s["button"], s["small_blind_seat"], s["big_blind_seat"]):
        return out
    o["seats"] = [x for x in s["seats"] if x != seat]
    o["stacks"].pop(str(seat), None)
    o["hole"].pop(str(seat), None)
    out["steps"] = [st for st in out["steps"] if st["actor"] != seat]
    return out


def _with_stack(fx: dict[str, Any], seat: int, value: int) -> dict[str, Any]:
    out = json.loads(json.dumps(fx))
    out["setup"]["stacks"][str(seat)] = value
    return out


def _complexity(stack: int) -> tuple[int, int]:
    """Rounder first: 100 is simpler than 99."""
    return sum(ch != "0" for ch in str(stack)), stack


def minimize(fx: dict[str, Any], fails: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    """Greedy shrink: drop non-marker seats, then trim stacks, while ``fails`` holds."""
    assert fails(fx), "the starting fixture does not show the failure"
    changed = True
    while changed:
        changed = False
        for seat in sorted(int(x) for x in fx["setup"]["stacks"]):
            cand = _without_seat(fx, seat)
            if cand["setup"]["stacks"] != fx["setup"]["stacks"] and fails(cand):
                fx, changed = cand, True
        for seat in sorted(int(x) for x in fx["setup"]["stacks"]):
            current = fx["setup"]["stacks"][str(seat)]
            for value in sorted({100 * (current // 100), 10 * (current // 10), current // 2, current - 1},
                                key=_complexity):
                if 0 < value < current and _complexity(value) < _complexity(current):
                    cand = _with_stack(fx, seat, value)
                    if fails(cand):
                        fx, changed = cand, True
                        break
    return fx


def oracle_expect(fx: dict[str, Any], hand: PokerKitHand) -> dict[str, Any]:
    """Expected settlement from the independent oracles for a hand that reached its end."""
    from tests.oracles.pots import settle
    from tests.support.oracle_bridge import oracle_strength

    contributions = {s: 0 for s in hand.seats}
    for ev in hand.events:
        if ev.type == "BetsCollected":
            for seat, amount in ev.data["bets"].items():
                contributions[seat] += amount
    live = [s for s in hand.seats if s not in hand.folded]
    board = list(hand.board)
    strengths = {s: oracle_strength(hand.profile.ruleset_id, list(hand.hole_cards[s]), board) for s in live}
    settlement = settle(contributions, tuple(hand.folded), strengths, hand.spec.button)
    final = {s: hand.starting_stacks[s] - contributions[s] + settlement.totals.get(s, 0) for s in hand.seats}
    return {
        "contributions": {str(s): v for s, v in contributions.items()},
        "pots": [{"amount": p.amount, "eligible": sorted(p.eligible),
                  "awards": {str(k): v for k, v in sorted(p.shares.items())}} for p in settlement.pots],
        "final_stacks": {str(s): v for s, v in final.items()},
    }


def minimize_cli(ruleset: str, players: int, seed: int, *, write: bool = False) -> int:
    fx, error = record_generated(seed, ruleset, players)
    if error is None:
        print(f"seed {seed} ({ruleset}, N={players}) passes; nothing to minimize")
        return 0
    print(f"failure: {error}")
    small = minimize(fx, lambda f: replay_fixture(f)[0] == error)
    _, hand = replay_fixture(small)
    assert hand is not None
    if len(hand.board) == 5 or sum(1 for s in hand.seats if s not in hand.folded) == 1:
        small["expect"] = oracle_expect(small, hand)
    small["description"] = (f"Minimized from generated hand seed {seed} ({ruleset}, N={players}), which failed with "
                            f"'{error}'. Expected settlement from the independent pot ledger and evaluators.")
    print(f"{len(fx['setup']['stacks'])} -> {len(small['setup']['stacks'])} seats, "
          f"{len(fx['steps'])} -> {len(small['steps'])} actions")
    text = json.dumps(small, indent=1) + "\n"
    print(text)
    if write:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIXTURE_DIR / f"{small['id']}.json"
        path.write_text(text)
        seeds = json.loads(SEEDS_FILE.read_text()) if SEEDS_FILE.exists() else []
        if not any(s["seed"] == seed and s["ruleset"] == ruleset and s["players"] == players for s in seeds):
            seeds.append({"gate": "state_machine", "ruleset": ruleset, "players": players, "seed": seed,
                          "error": error, "fixture": path.relative_to(ROOT).as_posix(),
                          "found": datetime.now(UTC).date().isoformat()})
            SEEDS_FILE.write_text(json.dumps(seeds, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)} and retained the seed in {SEEDS_FILE.relative_to(ROOT)}")
    return 1
