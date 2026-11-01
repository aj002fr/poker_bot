"""Rule tables for the style bots (No-limit Hold'em only).

Each style is a short set of rules over a :class:`Spot` parsed from the observation:
the hand class and position group preflop, a strength bucket after the flop, and the
betting so far in this hand. Randomness ("balance") comes from the seat's policy seed,
counted in the bot's explicit state, so a seeded table replays exactly. Every decision
carries a private note explaining it (shown only in the post-table reviewer replay).

These are teaching opponents with recognisable patterns, not strong players.

Every number and range in the rule tables is read through :meth:`Spot.kn` with the
style's own value as the default, so a spot without knobs (every :class:`StyleBot`)
plays exactly these rules, while :mod:`strategy.population` samples knob values to turn
each style into a family of related opponents. A spot may also carry its own
``evaluator`` (a variant's post-flop strength buckets) in place of the Hold'em one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from arena.agents.sdk import Action, ExplicitStateBot, clamp_raise_to, with_note
from arena.agents.styles.charts import (
    chart_group,
    chart_name,
    hand_class,
    in_chart,
    in_range,
    preflop_order,
    seat_positions,
)
from arena.agents.styles.handeval import hand_words, strength

STYLE_IDS = ("rock", "aggressive", "super_aggressive", "station", "weak_tight", "switcher")
NO_KNOBS: Mapping[str, Any] = MappingProxyType({})


class Chance:
    """Seeded coin flips; ``deterministic`` answers ``p >= 0.5`` without consuming the stream."""

    def __init__(self, seed: str, n: int = 0, *, deterministic: bool = False) -> None:
        self.seed, self.n, self.deterministic = seed, n, deterministic

    def __call__(self, p: float) -> bool:
        if self.deterministic:
            return p >= 0.5
        if p <= 0.0:
            return False
        if p >= 1.0:
            return True
        self.n += 1
        h = hashlib.blake2b(f"{self.seed}:{self.n}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") / 18446744073709551616.0 < p


@dataclass
class Spot:
    obs: Any
    hero: int
    street: str
    hole: list[str]
    board: list[str]
    hand: str
    sb: int
    bb: int
    to_call: int
    pot: int
    stack: int
    committed: int
    cur: int
    label: str
    group: str
    level: int
    pre_raises: int = 0
    limpers: int = 0
    folded_to: bool = True
    hero_raised_pre: bool = False
    pre_aggressor: int | None = None
    left_to_act: int = 0
    street_raises: int = 0
    hero_checked: bool = False
    hero_bet_street: bool = False
    faced_reraise: bool = False
    opponents: int = 0
    in_position: bool = False
    knobs: Mapping[str, Any] = field(default=NO_KNOBS, repr=False)
    evaluator: Callable[[list[str], list[str]], tuple[str, int, tuple[int, ...]]] | None = field(default=None,
                                                                                                   repr=False)

    def kn(self, key: str, default: Any) -> Any:
        """The knob ``key``, or the style's own ``default`` when it is not overridden."""
        return self.knobs.get(key, default)

    def strength(self) -> tuple[str, int, tuple[int, ...]]:
        return (self.evaluator or strength)(self.hole, self.board)

    @property
    def m(self) -> float:
        return (self.stack + self.committed) / max(1, self.sb + self.bb)

    @property
    def bet_frac(self) -> float:
        """The bet to call as a fraction of the pot before it."""
        return self.to_call / max(1, self.pot - self.to_call)

    @property
    def cg(self) -> str:
        return chart_group(self.group)


def parse(obs: Any) -> Spot:
    hero = obs.hero_seat
    me = next(p for p in obs.players if p.seat == hero)
    legal = obs.legal
    blinds = obs.blinds
    dealt: list[int] = []
    level = 0
    for e in obs.action_history:
        if e.type == "HandStarted":
            dealt = [int(s) for s in e.data["start_stacks"]]
            level = e.data["level"]
            break
    positions = seat_positions(obs.seats, dealt, obs.button_seat, obs.small_blind_seat, obs.big_blind_seat)
    pos = positions.get(hero, {"label": "BB", "group": "blinds"})
    s = Spot(obs=obs, hero=hero, street=obs.street, hole=list(obs.hole_cards), board=list(obs.board),
             hand=hand_class(obs.hole_cards), sb=blinds.small_blind, bb=blinds.big_blind,
             to_call=legal.call_amount or 0, pot=obs.pot_total, stack=me.stack, committed=me.street_committed,
             cur=obs.current_bet_to, label=pos["label"], group=pos["group"], level=level)
    hero_acted_pre = False
    for e in obs.action_history:
        if e.type != "ActionApplied" or e.data.get("auto"):
            continue
        d = e.data
        seat, street, kind = d["seat"], d["street"], d["action"]["type"]
        if street == "preflop":
            if seat == hero:
                hero_acted_pre = True
            elif not hero_acted_pre and kind in ("call", "raise_to"):
                s.folded_to = False
            if kind == "raise_to":
                s.pre_raises += 1
                s.pre_aggressor = seat
                if seat == hero:
                    s.hero_raised_pre = True
            elif kind == "call" and s.pre_raises == 0 and seat != hero:
                s.limpers += 1
        if street == s.street:
            if kind == "raise_to":
                s.street_raises += 1
                if seat == hero:
                    s.hero_bet_street = True
                    s.faced_reraise = False
                elif s.hero_bet_street:
                    s.faced_reraise = True
            elif kind == "check" and seat == hero:
                s.hero_checked = True
    live = {p.seat for p in obs.players if p.status in ("active", "all_in")}
    s.opponents = len(live - {hero})
    active = {p.seat for p in obs.players if p.status == "active"}
    if s.street == "preflop" and dealt:
        order = preflop_order(obs.seats, dealt, obs.big_blind_seat)
        after = order[order.index(hero) + 1:] if hero in order else []
        s.left_to_act = sum(1 for x in after if x in active)
    ring = list(obs.seats)
    if obs.button_seat in ring:
        i = ring.index(obs.button_seat)
        post = [x for x in ring[i + 1:] + ring[: i + 1] if x in live]
        s.in_position = bool(post) and post[-1] == hero
    return s


# ------------------------------------------------------------------ actions

Decision = tuple[Action, str]


def _check_or_fold(s: Spot, why: str) -> Decision:
    if s.obs.legal.check:
        return Action(type="check"), why.replace("folded", "checked", 1)
    return Action(type="fold"), why


def _call(s: Spot, why: str) -> Decision:
    if s.obs.legal.check:
        return Action(type="check"), why
    return Action(type="call"), why


def _all_in(s: Spot, why: str) -> Decision:
    legal = s.obs.legal
    target = legal.all_in_raise_to or legal.max_raise_to or legal.short_all_in_raise_to
    a = clamp_raise_to(legal, target) if target is not None else None
    return (a, why) if a is not None else _call(s, why)


def _raise(s: Spot, target: int, why: str) -> Decision:
    legal = s.obs.legal
    if target >= s.kn("jam_frac", 0.4) * (s.stack + s.committed):
        return _all_in(s, why)
    a = clamp_raise_to(legal, target)
    return (a, why) if a is not None else _call(s, why)


def _open_size(s: Spot, mult: float = 3.0) -> int:
    return int(round(mult * s.bb + s.limpers * s.bb))


def _re(s: Spot, mult: float = 3) -> int:
    """A raise to ``mult`` times the current bet."""
    return int(round(s.cur * mult))


def _bet(s: Spot, frac: float, why: str) -> Decision:
    if s.to_call:
        return _raise(s, _re(s, s.kn("raise_mult", 3)), why)
    return _raise(s, s.cur + max(s.bb, int(frac * s.kn("bet_scale", 1.0) * s.pot)), why)


def _words(s: Spot) -> str:
    return hand_words(s.hole, s.board)[0].lower()


# ------------------------------------------------------------------ preflop


def _rock_pre(s: Spot, ch: Chance, chart: str = "conservative") -> Decision:
    h, lbl, k = s.hand, s.label, s.kn
    chart = k("rock.chart", chart)
    if s.pre_raises == 0:
        if in_chart(chart, s.group, h):
            return _raise(s, _open_size(s, k("rock.open_mult", 3.0)),
                          f"opened {h} from {lbl}: in the {chart_name(chart)} {s.cg}-position chart")
        return _check_or_fold(s, f"folded {h} from {lbl}: outside the {chart_name(chart)} {s.cg}-position chart")
    if s.pre_raises == 1 and not s.hero_raised_pre:
        if in_range(k("rock.reraise", "QQ+, AK"), h):
            return _raise(s, _re(s, k("rock.reraise_mult", 3)), f"reraised with {h}: strong enough to play back at a raise")
        if in_range(k("rock.call", "JJ-77, AQ, AJs, KQs"), h) and s.to_call <= k("rock.call_price", 0.08) * s.stack:
            return _call(s, f"called a raise with {h}: the Gap Concept wants more than an opening hand to call")
        return _check_or_fold(s, f"folded {h} to a raise: the Gap Concept wants a stronger hand to call than to open")
    if in_range(k("rock.jam_3b", "KK+"), h):
        return _all_in(s, f"moved in with {h} over a reraise")
    if (s.hero_raised_pre and in_range(k("rock.call_3b", "QQ, AK"), h)
            and s.to_call <= k("rock.call_3b_price", 0.3) * s.stack):
        return _call(s, f"called the reraise with {h}: good enough to see a flop, not to go broke")
    return _check_or_fold(s, f"laid down {h} to a reraise")


def _aggressive_pre(s: Spot, ch: Chance) -> Decision:
    h, lbl, k = s.hand, s.label, s.kn
    if s.pre_raises == 0:
        if in_chart(k("agg.chart", "aggressive"), s.group, h):
            return _raise(s, _open_size(s, k("agg.open_mult", 3.0)),
                          f"opened {h} from {lbl}: in the aggressive {s.cg}-position chart")
        return _check_or_fold(s, f"folded {h} from {lbl}: outside even the aggressive chart")
    if s.pre_raises == 1 and not s.hero_raised_pre:
        if (in_range(k("agg.reraise", "TT+, AQ+, AJs, KQs"), h)
                or (in_range(k("agg.light_reraise", "A2s-A9s, 54s-T9s"), h) and ch(k("agg.light_reraise_p", 0.15)))):
            return _raise(s, _re(s, k("agg.reraise_mult", 3)), f"reraised with {h} to take the initiative")
        if (in_range(k("agg.call", "22+, A2s+, KJ+, QJs, JTs, T9s, 98s"), h)
                and s.to_call <= k("agg.call_price", 0.12) * s.stack):
            return _call(s, f"called a raise with {h}: playable and cheap enough")
        return _check_or_fold(s, f"folded {h} to a raise")
    if in_range(k("agg.jam_3b", "QQ+, AK"), h):
        return _all_in(s, f"moved in with {h} over a reraise")
    if (s.hero_raised_pre and in_range(k("agg.call_3b", "55+, AT+, A5s+, KQ, KJs, QJs, JTs"), h)
            and s.to_call <= k("agg.call_3b_price", 0.3) * s.stack):
        return _call(s, f"called the reraise with {h}: does not give up a pot easily")
    return _check_or_fold(s, f"gave up {h} to a reraise")


_STEALS = {"BTN": 0.85, "CO": 0.6, "SB": 0.7}


def _super_pre(s: Spot, ch: Chance) -> Decision:
    h, lbl, k = s.hand, s.label, s.kn
    if s.pre_raises == 0:
        steal = k("sup.steal", _STEALS).get(lbl, 0.0)
        if s.folded_to and s.left_to_act <= k("sup.steal_behind", 2) and steal and ch(steal):
            return _raise(s, _open_size(s, k("sup.steal_mult", 2.5)),
                          f"stole with {h} from {lbl}: folded to it, so any two cards")
        if (in_chart(k("sup.chart", "aggressive"), s.group, h)
                or (in_range(k("sup.loose", "K2s+, Q2s+, J7s+"), h) and ch(k("sup.loose_p", 0.5)))):
            return _raise(s, _open_size(s, k("sup.open_mult", 3.0)), f"opened {h} from {lbl}: raises far more than it calls")
        return _check_or_fold(s, f"folded {h} from {lbl}")
    if s.pre_raises == 1 and not s.hero_raised_pre:
        if (in_range(k("sup.reraise", "99+, AT+, KQ"), h)
                or (in_chart(k("sup.light_chart", "aggressive"), "late", h) and ch(k("sup.light_reraise_p", 0.3)))):
            return _raise(s, _re(s, k("sup.reraise_mult", 3)), f"reraised with {h}: plays back at raises often")
        if in_chart(k("sup.call_chart", "aggressive"), "late", h) and s.to_call <= k("sup.call_price", 0.12) * s.stack:
            return _call(s, f"called a raise with {h}")
        return _check_or_fold(s, f"folded {h} to a raise")
    if in_range(k("sup.jam_3b", "JJ+, AQ+"), h) or (in_range(k("sup.light_jam", "Ax, K9s+"), h)
                                                   and ch(k("sup.light_jam_p", 0.3))):
        return _all_in(s, f"moved in with {h} over a reraise: fights back")
    if (s.hero_raised_pre and (in_chart(k("sup.call_chart", "aggressive"), "late", h) or ch(k("sup.call_3b_p", 0.3)))
            and s.to_call <= k("sup.call_3b_price", 0.3) * s.stack):
        return _call(s, f"called the reraise with {h}: rarely lets a reraise push it off")
    return _check_or_fold(s, f"folded {h} to a reraise")


def _station_pre(s: Spot, ch: Chance) -> Decision:
    h, k = s.hand, s.kn
    if in_range(k("sta.raise", "AA, KK"), h) and ch(k("sta.raise_p", 0.35)):
        return _raise(s, max(_open_size(s, k("sta.open_mult", 3.0)), _re(s, k("sta.reraise_mult", 3))),
                      f"raised with {h}: one of the few hands it raises")
    if s.pre_raises == 0:
        if in_chart(k("sta.chart", "station"), s.group, h):
            return _call(s, f"limped with {h}: likes to see cheap flops")
        return _check_or_fold(s, f"folded {h}")
    limit = k("sta.call_price", 0.25) if s.pre_raises == 1 else k("sta.call_3b_price", 0.35)
    if in_chart(k("sta.chart", "station"), s.group, h) and s.to_call <= limit * s.stack:
        return _call(s, f"called with {h}: would rather call than raise or fold")
    return _check_or_fold(s, f"folded {h}: even a station passes on this price")


def _weak_tight_pre(s: Spot, ch: Chance) -> Decision:
    h, lbl, k = s.hand, s.label, s.kn
    if s.pre_raises == 0:
        if in_chart(k("wt.chart", "weak_tight"), s.group, h):
            if in_range(k("wt.raise", "TT+, AK, AQs"), h) or ch(k("wt.raise_p", 0.5)):
                return _raise(s, _open_size(s, k("wt.open_mult", 2.5)),
                              f"made a small raise with {h} from {lbl}: a tight chart")
            return _call(s, f"limped with {h} from {lbl}: would rather not raise")
        return _check_or_fold(s, f"folded {h} from {lbl}: only plays its tight chart")
    if s.pre_raises == 1 and not s.hero_raised_pre:
        if in_range(k("wt.reraise", "AA, KK"), h):
            return _raise(s, _re(s, k("wt.reraise_mult", 3)), f"reraised with {h}")
        if in_range(k("wt.call", "JJ-55, AK, AQ, AJs, KQs"), h) and s.to_call <= k("wt.call_price", 0.08) * s.stack:
            return _call(s, f"called a raise with {h} instead of reraising")
        return _check_or_fold(s, f"folded {h} to a raise")
    if in_range(k("wt.call_3b", "AA, KK"), h):
        return _call(s, f"only called the reraise even with {h}")
    return _check_or_fold(s, f"backed down: folded {h} to a reraise")


def _push_fold_pre(s: Spot, ch: Chance) -> Decision:
    h, k = s.hand, s.kn
    m = f"M {s.m:.0f}"
    if s.pre_raises == 0 and in_chart(k("pf.chart", "push"), s.group, h):
        return _all_in(s, f"moved in with {h} ({m}): short-stacked, so all-in or fold")
    if s.pre_raises == 1 and in_chart(k("pf.chart", "push"), "early", h):
        return _all_in(s, f"moved in over a raise with {h} ({m})")
    if s.pre_raises >= 2 and in_range(k("pf.jam_3b", "KK+, AK"), h):
        return _all_in(s, f"moved in with {h} ({m})")
    return _check_or_fold(s, f"folded {h} ({m}): not worth moving in")


# ----------------------------------------------------------------- postflop


def _rock_post(s: Spot, ch: Chance) -> Decision:
    bucket, outs, _ = s.strength()
    w = _words(s)
    k = s.kn
    raiser = s.pre_aggressor == s.hero
    rr = k("rockp.raise_mult", 3)
    if s.to_call:
        if s.faced_reraise:
            if bucket == "monster":
                return _raise(s, _re(s, rr), f"raised again with {w}")
            if bucket == "strong" and s.bet_frac <= k("rockp.strong_vs_raise", 0.6):
                return _call(s, f"called the raise with {w}")
            return _check_or_fold(s, f"folded {w} to a raise: a rock respects raises")
        if bucket == "monster":
            return _raise(s, _re(s, rr), f"raised for value with {w}")
        if bucket == "strong" and s.bet_frac <= k("rockp.strong_call", 1.2):
            return _call(s, f"called with {w}")
        if bucket == "medium" and s.bet_frac <= k("rockp.medium_call", 0.6):
            return _call(s, f"called a modest bet with {w}")
        if outs >= k("rockp.draw_outs", 8) and s.bet_frac <= k("rockp.draw_call", 0.4):
            return _call(s, f"called with a draw ({outs} outs) at a good price")
        return _check_or_fold(s, f"folded {w}: rarely continues without a hand")
    if bucket in ("monster", "strong"):
        return _bet(s, k("rockp.value_frac", 0.6), f"bet {w} for value")
    if (raiser and s.street == "flop" and 1 <= s.opponents <= k("rockp.cbet_opponents", 1)
            and ch(k("rockp.cbet_medium", 0.6) if bucket == "medium" else k("rockp.cbet_air", 0.3))):
        return _bet(s, k("rockp.cbet_frac", 0.5), f"continuation bet with {w} as the preflop raiser")
    return _call(s, f"checked {w}")


def _aggressive_post(s: Spot, ch: Chance, *, cbet: float = 0.75, barrel: float = 0.4, semi: float = 0.3,
                     bluff_raise: float = 0.05, trap: float = 0.0, fight: float = 0.3, p: str = "aggp") -> Decision:
    bucket, outs, _ = s.strength()
    w = _words(s)
    k = s.kn
    cbet, barrel, semi = k(p + ".cbet", cbet), k(p + ".barrel", barrel), k(p + ".semi", semi)
    bluff_raise, trap, fight = k(p + ".bluff_raise", bluff_raise), k(p + ".trap", trap), k(p + ".fight", fight)
    rr = k(p + ".raise_mult", 3)
    raiser = s.pre_aggressor == s.hero
    if s.to_call:
        if bucket == "monster":
            how = "check-raised" if s.hero_checked else "raised"
            return _raise(s, _re(s, rr), f"{how} with {w}")
        if s.faced_reraise and bucket != "strong" and not (outs >= 8 and s.bet_frac <= k(p + ".draw_call", 0.6)):
            if bucket == "medium" and ch(fight):
                return _call(s, f"called the raise with {w}: does not give up easily")
            if ch(bluff_raise):
                return _raise(s, _re(s, rr), f"reraised as a bluff with {w}")
            return _check_or_fold(s, f"gave up {w} to a raise")
        if bucket == "strong":
            if ch(k(p + ".strong_raise", 0.25)):
                return _raise(s, _re(s, rr), f"raised {w} to take control")
            return _call(s, f"called with {w}")
        if outs >= 8:
            if ch(semi):
                return _raise(s, _re(s, rr), f"semi-bluff raise with a draw ({outs} outs)")
            if s.bet_frac <= k(p + ".draw_call", 0.6):
                return _call(s, f"called with a draw ({outs} outs)")
        if bucket == "medium" and s.bet_frac <= k(p + ".medium_call", 0.6):
            return _call(s, f"called with {w}")
        if ch(bluff_raise):
            return _raise(s, _re(s, rr), f"bluff-raised with {w}")
        return _check_or_fold(s, f"folded {w}")
    if bucket == "monster" and trap and not s.in_position and not s.hero_checked and ch(trap):
        return _call(s, f"checked {w} to set a check-raise trap")
    if bucket in ("monster", "strong"):
        return _bet(s, k(p + ".value_frac", 0.66), f"bet {w} for value")
    if raiser:
        prob = {"flop": cbet, "turn": barrel, "river": barrel * k(p + ".river_barrel", 0.6)}.get(s.street, 0.0)
        if ch(prob):
            return _bet(s, k(p + ".cbet_frac", 0.6), f"kept betting as the preflop raiser with {w}")
    if outs >= 8 and ch(k(p + ".draw_bet", 0.5)):
        return _bet(s, k(p + ".draw_frac", 0.6), f"semi-bluff with a draw ({outs} outs)")
    if bucket == "medium" and ch(k(p + ".thin_value", 0.4)):
        return _bet(s, k(p + ".thin_frac", 0.5), f"bet {w} for thin value")
    if trap and ch(k(p + ".stab", 0.25)):
        return _bet(s, k(p + ".stab_frac", 0.5), f"stabbed at the pot with {w}")
    return _call(s, f"checked {w}")


def _super_post(s: Spot, ch: Chance) -> Decision:
    return _aggressive_post(s, ch, cbet=0.85, barrel=0.55, semi=0.45, bluff_raise=0.15, trap=0.45, fight=0.6,
                            p="supp")


def _station_post(s: Spot, ch: Chance) -> Decision:
    bucket, outs, _ = s.strength()
    w = _words(s)
    k = s.kn
    if s.to_call:
        if bucket == "monster" and ch(k("stap.raise_p", 0.2)):
            return _raise(s, _re(s, k("stap.raise_mult", 3)), f"finally raised, with {w}")
        if bucket in ("monster", "strong", "medium") or outs >= k("stap.draw_outs", 4):
            return _call(s, f"called with {w}: a station keeps calling")
        if s.bet_frac <= k("stap.weak_price", 0.75) and ch(k("stap.weak_call", 0.45)):
            return _call(s, f"called with {w}, hoping to catch up")
        return _check_or_fold(s, f"folded {w}: a rare fold")
    if bucket == "monster" and ch(k("stap.bet_monster", 0.4)):
        return _bet(s, k("stap.bet_frac", 0.4), f"made a small bet with {w}")
    if bucket == "strong" and ch(k("stap.bet_strong", 0.15)):
        return _bet(s, k("stap.bet_frac", 0.4), f"made a small bet with {w}")
    return _call(s, f"checked {w}: prefers calling to betting")


def _weak_tight_post(s: Spot, ch: Chance) -> Decision:
    bucket, outs, _ = s.strength()
    w = _words(s)
    k = s.kn
    raiser = s.pre_aggressor == s.hero
    if s.to_call:
        if s.faced_reraise:
            if bucket == "monster":
                return _call(s, f"only called the raise with {w}")
            return _check_or_fold(s, f"backed down: folded {w} to a raise")
        if bucket == "monster":
            return _call(s, f"called with {w}: slow to raise even when strong")
        if bucket == "strong" and s.bet_frac <= k("wtp.strong_call", 0.7):
            return _call(s, f"called with {w}")
        if bucket == "medium" and s.bet_frac <= k("wtp.medium_call", 0.4):
            return _call(s, f"called a small bet with {w}")
        if outs >= 8 and s.bet_frac <= k("wtp.draw_call", 0.3):
            return _call(s, f"called a small bet with a draw ({outs} outs)")
        return _check_or_fold(s, f"folded {w} to the bet")
    if bucket == "monster" and ch(k("wtp.bet_monster", 0.7)) or bucket == "strong" and ch(k("wtp.bet_strong", 0.35)):
        return _bet(s, k("wtp.bet_frac", 0.4), f"bet {w} small")
    if raiser and s.street == "flop" and bucket == "medium" and ch(k("wtp.cbet_medium", 0.3)):
        return _bet(s, k("wtp.cbet_frac", 0.35), f"small continuation bet with {w}")
    return _call(s, f"checked {w}: cautious after the flop")


# ----------------------------------------------------------------- dispatch

Rule = Callable[[Spot, Chance], Decision]
Rules = tuple[Rule, Rule]

RULES: dict[str, Rules] = {
    "rock": (_rock_pre, _rock_post),
    "aggressive": (_aggressive_pre, _aggressive_post),
    "super_aggressive": (_super_pre, _super_post),
    "station": (_station_pre, _station_post),
    "weak_tight": (_weak_tight_pre, _weak_tight_post),
}
PUSH_FOLD_PRE: Rule = _push_fold_pre
GEAR_RULES: dict[str, Rules] = {
    "conservative": (_rock_pre, _rock_post),
    "aggressive": (_super_pre, _aggressive_post),
    "push_fold": (_push_fold_pre, _aggressive_post),
}


def switcher_gear(s: Spot) -> str:
    """conservative early with a deep stack; aggressive as blinds rise; push/fold when short."""
    if s.m < s.kn("gear.push_m", 7):
        return "push_fold"
    if s.level >= s.kn("gear.aggressive_level", 2) or s.m < s.kn("gear.aggressive_m", 15):
        return "aggressive"
    return "conservative"


def decide_gear(gear: str, s: Spot, ch: Chance) -> Decision:
    pre, post = GEAR_RULES[gear]
    action, note = (pre if s.street == "preflop" else post)(s, ch)
    return action, f"[{gear.replace('_', '/')} gear, M {s.m:.0f}] " + note


def decide_spot(style: str, s: Spot, ch: Chance) -> Decision:
    if style == "switcher":
        return decide_gear(switcher_gear(s), s, ch)
    pre, post = RULES[style]
    return (pre if s.street == "preflop" else post)(s, ch)


def decide(style: str, obs: Any, ch: Chance) -> Decision:
    return decide_spot(style, parse(obs), ch)


def coach(obs: Any) -> dict[str, Any]:
    """What the conservative rules would do here, without randomness, and why."""
    action, note = decide("rock", obs, Chance("coach", deterministic=True))
    reason = note[0].upper() + note[1:] + "."
    return {"action": action.type, "amount": action.amount, "reason": reason, "heuristic": "conservative",
            "disclaimer": "A teaching heuristic in the book's conservative style, not optimal play."}


class StyleBot(ExplicitStateBot):
    rulesets = ("practice_nlhe",)

    def __init__(self, style: str) -> None:
        if style not in STYLE_IDS:
            raise ValueError(f"unknown style {style!r}")
        self.style = style
        self.name = f"style_{style}"

    def initial_state(self, table: Any) -> dict[str, Any]:
        return {"seed": table.policy_seed, "n": 0}

    def decide(self, observation: Any, state: Any) -> tuple[dict, dict[str, Any]]:
        ch = Chance(state["seed"], state["n"])
        action, note = decide(self.style, observation, ch)
        return with_note(action, note), {"seed": state["seed"], "n": ch.n}
