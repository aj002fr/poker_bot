"""Opponent-style setup, hidden-seat rules, the end-of-table reveal and learning aids.

Privacy rule: while a table runs, a hidden seat's policy, style, label, source,
artifact hash and reasoning notes appear in no player-visible payload. Hidden
seats are masked in :meth:`arena.api.manager.SeatSlot.public`; everything else
here is either computed from public events (stats), from the requesting seat's
own view (aids), or released only once the table is complete (reveal).
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr

from arena.agents.styles.charts import CHARTS, PUBLIC_CHARTS, chart_group, grid, hand_class, in_chart, seat_positions
from arena.agents.styles.engine import coach
from arena.agents.styles.handeval import draws, equity_vs_random, hand_words
from arena.agents.styles.profiles import (
    PROFILES,
    RANDOM_POLICY,
    RANDOM_POOL_DEFAULT,
    STAT_DEFINITIONS,
    style_of_policy,
)
from arena.domain.canonical import canonical_hash
from arena.domain.observations import TableView

Mode = Literal["labelled", "hidden", "mixed", "random"]
MODES: tuple[str, ...] = ("labelled", "hidden", "mixed", "random")
STYLE_RULESETS = ("practice_nlhe",)
LINEUP_DOMAIN = "arena-lineup-v1"
EVIDENCE_STATS = ("vpip", "pfr", "aggression", "fold_to_reraise")


class StylesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Mode = "labelled"
    random_pool: list[StrictStr] | None = None
    learning_aids: bool = True


@dataclass(frozen=True)
class SeatPlan:
    """How one requested seat is set up: hidden during play, and whether its style is drawn at random."""

    hidden: bool = False
    random_pick: bool = False


class StylesInvalid(ValueError):
    def __init__(self, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.details = code, message, details


def random_pool(req: StylesRequest) -> list[str]:
    if req.random_pool is None:
        return list(RANDOM_POOL_DEFAULT)
    pool = list(dict.fromkeys(req.random_pool))
    if not pool:
        raise StylesInvalid("invalid_styles", "random_pool must name at least one style")
    unknown = [s for s in pool if s not in PROFILES]
    if unknown:
        raise StylesInvalid("invalid_styles", "random_pool has unknown style ids", {"unknown": unknown})
    return pool


def plan_seats(seats: Sequence[Any], req: StylesRequest, ruleset_id: str) -> list[SeatPlan]:
    """Check every seat's ``reveal`` flag and random-style use against the mode."""
    plans: list[SeatPlan] = []
    for i, s in enumerate(seats):
        if s.kind == "human":
            if s.reveal is not None:
                raise StylesInvalid("invalid_seat", f"seat {i}: human seats take no reveal flag")
            plans.append(SeatPlan())
            continue
        if req.mode == "random" and s.policy not in (None, RANDOM_POLICY):
            raise StylesInvalid("invalid_styles", f"seat {i}: random mode seats every bot with a random style; "
                                                  f"use {RANDOM_POLICY} or omit the policy")
        is_random = req.mode == "random" or s.policy == RANDOM_POLICY
        if req.mode == "labelled" and s.reveal is False:
            raise StylesInvalid("invalid_styles", f"seat {i}: labelled mode shows every bot; reveal cannot be false")
        if req.mode in ("hidden", "random") and s.reveal is True:
            raise StylesInvalid("invalid_styles", f"seat {i}: {req.mode} mode hides every bot; reveal cannot be true")
        if is_random and s.reveal is True:
            raise StylesInvalid("invalid_styles", f"seat {i}: a random-style seat stays hidden until the table ends")
        if is_random and ruleset_id not in STYLE_RULESETS:
            raise StylesInvalid("unsupported_ruleset", f"seat {i}: style bots do not support {ruleset_id}",
                                {"seat_index": i, "policy": RANDOM_POLICY, "supported": list(STYLE_RULESETS)})
        hidden = is_random or req.mode == "hidden" or (req.mode == "mixed" and s.reveal is False)
        plans.append(SeatPlan(hidden=hidden, random_pick=is_random))
    return plans


def pick_style(pool: Sequence[str], seed: str | None, seat: int) -> str:
    """A style for a random seat: reproducible from the table's private seed, else fresh randomness."""
    if seed is None:
        return secrets.choice(list(pool))
    digest = hashlib.sha256(f"arena-style-pick-v1|{seed}|{seat}".encode()).digest()
    return pool[int.from_bytes(digest[:8], "big") % len(pool)]


def styles_catalog() -> dict[str, Any]:
    return {
        "styles": [p.public() for p in PROFILES.values()],
        "modes": list(MODES),
        "random_pool_default": list(RANDOM_POOL_DEFAULT),
        "stat_definitions": STAT_DEFINITIONS,
        "charts": {c: {g: sorted(CHARTS[c][g], key=_grid_order) for g in ("early", "middle", "late")}
                   for c in PUBLIC_CHARTS},
        "grid": grid(),
    }


@lru_cache(maxsize=1)
def _grid_index() -> dict[str, int]:
    return {h: i for i, h in enumerate(h for row in grid() for h in row)}


def _grid_order(hand: str) -> int:
    return _grid_index()[hand]


# ----------------------------------------------------------------- reveal


def _evidence(stats: Mapping[str, Any], style: str | None) -> list[dict[str, Any]]:
    expected = PROFILES[style].expected if style is not None else None
    readings = {
        "vpip": (stats["vpip"]["rate"], stats["vpip"]["count"], stats["vpip"]["of"]),
        "pfr": (stats["pfr"]["rate"], stats["pfr"]["count"], stats["pfr"]["of"]),
        "aggression": (stats["aggression"]["factor"], stats["aggression"]["bets_raises"],
                       stats["aggression"]["calls"]),
        "fold_to_reraise": (stats["reraise"]["fold_rate"], stats["reraise"]["folded"], stats["reraise"]["faced"]),
    }
    out = []
    for name in EVIDENCE_STATS:
        value, count, of = readings[name]
        band = getattr(expected, name) if expected is not None else None
        in_range = None if value is None or band is None else bool(band[0] <= value <= band[1])
        out.append({"stat": name, "label": STAT_DEFINITIONS[name]["label"], "value": value, "count": count,
                    "of": of, "expected": list(band) if band is not None else None, "in_range": in_range,
                    "book_ref": STAT_DEFINITIONS[name]["book_ref"]})
    return out


def _pct(rate: float | None) -> str:
    return "no sample" if rate is None else f"{round(rate * 100)}%"


def _summary(stats: Mapping[str, Any], style: str | None, evidence: list[dict[str, Any]]) -> str:
    if not stats["hands"]:
        return "Was not dealt a hand, so there is nothing to read."
    played = (f"Played {_pct(stats['vpip']['rate'])} of {stats['hands']} hands and raised "
              f"{_pct(stats['pfr']['rate'])} before the flop")
    if style is None:
        return played + "."
    judged = [e for e in evidence if e["in_range"] is not None]
    fits = sum(1 for e in judged if e["in_range"])
    name = PROFILES[style].name
    if not judged:
        return f"{played}; too few hands to compare with a typical {name}."
    return f"{played}; {fits} of {len(judged)} readings fit a typical {name}."


def lineup_hash(config_hash: str, lineup: list[dict[str, Any]]) -> str:
    return canonical_hash({"config_hash": config_hash, "lineup": lineup}, domain=LINEUP_DOMAIN)


def build_reveal(config_hash: str, seats: list[dict[str, Any]], stats: Mapping[str, Any]) -> dict[str, Any]:
    """``seats``: every slot's real identity (seat, kind, policy, label, source, artifact_hash, hidden, random_pick)."""
    lineup = [{"seat": s["seat"], "kind": s["kind"], "policy": s["policy"], "artifact_hash": s["artifact_hash"]}
              for s in seats]
    out = []
    for s in seats:
        style = style_of_policy(s["policy"]) if s["kind"] == "bot" else None
        seat_stats = stats["seats"][str(s["seat"])]
        evidence = _evidence(seat_stats, style)
        out.append({
            "seat": s["seat"], "kind": s["kind"], "hidden": s["hidden"], "random_pick": s["random_pick"],
            "policy": s["policy"], "label": s["label"], "style": style, "artifact_hash": s["artifact_hash"],
            "book_ref": PROFILES[style].public()["book_ref"] if style is not None else None,
            "summary": _summary(seat_stats, style, evidence),
            "evidence": evidence,
            "showdowns": seat_stats["showdowns"]["items"],
        })
    return {"lineup_hash": lineup_hash(config_hash, lineup), "lineup": lineup, "seats": out}


# ------------------------------------------------------------- learning aids


@lru_cache(maxsize=4096)
def _equity(hole: tuple[str, ...], board: tuple[str, ...], opponents: int) -> tuple[float, int, bool]:
    equity, samples, exact = equity_vs_random(hole, board, opponents)
    return round(equity, 4), samples, exact


def _positions(view: TableView) -> dict[int, dict[str, str]]:
    dealt = next(([int(s) for s in e.data["start_stacks"]] for e in view.action_history
                  if e.type == "HandStarted"), [])
    if not dealt:
        return {}
    return seat_positions(view.seats, dealt, view.button_seat, view.small_blind_seat, view.big_blind_seat)


def build_aids(view: TableView, observation: Any | None) -> dict[str, Any]:
    """Aids for the seat whose own ``view`` this is; ``observation`` only when it is that seat's decision."""
    positions = _positions(view)
    out: dict[str, Any] = {
        "hand_id": view.hand_id,
        "decision_id": view.decision_id if observation is not None else None,
        "positions": {str(s): p for s, p in sorted(positions.items())},
        "hero": None,
    }
    hero, hole = view.hero_seat, view.hole_cards
    if hero is None or not hole or hero not in positions:
        return out
    board = list(view.board)
    position = positions[hero]
    hc = hand_class(hole)
    words, rank_row = hand_words(hole, board)
    draw_list, outs = draws(hole, board) if len(board) in (3, 4) else ([], 0)
    group = chart_group(position["group"])
    opponents = sum(1 for p in view.players if p.seat != hero and p.status in ("active", "all_in"))
    vs_one, samples, exact = _equity(tuple(hole), tuple(board), 1)
    vs_field, samples_f, exact_f = _equity(tuple(hole), tuple(board), max(1, opponents))
    pot_odds = None
    coach_hint = None
    if observation is not None:
        legal = observation.legal
        to_call = (legal.call_amount or 0) if legal is not None else 0
        if to_call > 0:
            pot = view.pot_total
            be = round(to_call / (pot + to_call), 4)
            pot_odds = {"to_call": to_call, "pot": pot, "break_even": be,
                        "words": f"Call {to_call} to win {pot}: you need {round(be * 100)}% to break even"}
        coach_hint = coach(observation)
    out["hero"] = {
        "seat": hero,
        "position": position,
        "hole_cards": list(hole),
        "hand_class": hc,
        "hand_in_words": words,
        "rank_row": rank_row,
        "draws": draw_list,
        "outs": outs,
        "charts": {"group": group, **{c: in_chart(c, group, hc) for c in PUBLIC_CHARTS}},
        "equity_vs_random": {"vs_one": vs_one, "vs_field": vs_field, "opponents": max(1, opponents),
                             "samples": max(samples, samples_f), "exact": exact and exact_f},
        "pot_odds": pot_odds,
        "coach": coach_hint,
    }
    return out
