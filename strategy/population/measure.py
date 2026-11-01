"""Measure population members at seeded in-process tables.

:func:`census` seats members (and optionally other bots) at official-format tables and
returns one :class:`Tally` per entry key. A tally counts every decision as the seat makes
it (no observations are kept), so censuses stay small and tallies from separate tables or
processes merge exactly. :meth:`Tally.stats` gives the statistics families declare in
``Family.expected``:

==================  =========================================================================
``vpip``, ``pfr``   from :class:`arena.table.player_stats.StatsTracker` (spectator events)
``aggression``      bets and raises divided by calls, all streets
``fold_to_reraise`` folds after betting and facing a raise on the same street
``limp``            calls when the pot is unraised and the seat owes chips (limps, over-limps)
``iso``             raises when the pot has limpers and no raise
``three_bet``       raises when facing exactly one raise preflop (hero not yet raised)
``squeeze``         the same, when the raise already has at least one caller
``cold_call``       calls in that same spot (one raise, hero not yet raised)
``cbet``            flop bets by the preflop raiser when checked to
``cbet_dry_hu``     ... heads-up on a dry flop; ``cbet_wet`` ... on a wet or monotone flop
``strong_check``    checks with a strong or monster hand when first to act post-flop
``xr_share``        post-flop raises made after checking on that street, among post-flop raises
``allin_share``     raises to the all-in amount among all raises
``min_raise_share`` raises to exactly the minimum among raises facing a bet or blind
``big_call``        calls or raises facing a bet of at least a third of the seat's chips
``short_allin``     preflop raises that are all-in, among preflop raises at M < 8
``junk_entry``      voluntary preflop entries with a starting hand outside the top 35 percent
``size_cv``         coefficient of variation of post-flop bet size (not all-in) as a fraction of the pot
``size_corr``       correlation of that size with the seat's strength bucket
``draw_continue``   calls or raises facing a flop/turn bet with 4+ outs and no strong made hand
``draw_aggr``       bets or raises in those draw spots, facing a bet or checked to
``mw_bluff``        bets with a weak hand and no draw when checked to in a 3+ way pot
``hu_bluff``        the same heads-up
==================  =========================================================================

Tallies use the seat's own observation and its hole cards, so the card-based statistics
are diagnostics for tests, never inputs a learner could read about another seat.
"""

from __future__ import annotations

import math
import multiprocessing
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from arena.agents.sdk import TableInfo, bot_info
from arena.agents.styles.engine import parse
from arena.agents.worker import InProcessSeat, play_table
from arena.benchmark.runner import policy_seed
from arena.domain.models import TournamentConfig
from arena.table.player_stats import StatsTracker
from arena.table.projections import Viewer, project_events
from arena.table.service import seeded_table
from strategy.population import cards
from strategy.population.policy import PopulationBot, scan

OFFICIAL = {"initial_player_count": 8, "starting_stack": 2000, "hand_cap": 200, "hands_per_level": 25}
MIN_SAMPLES = 20
TRACKER_KEYS = ("hands", "vpip", "pfr", "bets_raises", "calls", "faced", "folded")

Entry = tuple[str, "Mapping[str, Any] | Callable[[], object]"]
"""``(key, member spec)`` or ``(key, zero-argument bot factory)``; only specs run in parallel."""


@dataclass
class Tally:
    counts: Counter = field(default_factory=Counter)
    n: dict[str, list[int]] = field(default_factory=dict)
    sizes: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    fallbacks: Counter = field(default_factory=Counter)
    decisions: int = 0
    max_wall_ns: int = 0

    def merge(self, other: Tally) -> Tally:
        self.counts.update(other.counts)
        for k, (hit, den) in other.n.items():
            cur = self.n.setdefault(k, [0, 0])
            cur[0] += hit
            cur[1] += den
        self.sizes += other.sizes
        self.scores += other.scores
        self.fallbacks.update(other.fallbacks)
        self.decisions += other.decisions
        self.max_wall_ns = max(self.max_wall_ns, other.max_wall_ns)
        return self

    def count(self, stat: str, hit: bool) -> None:
        cur = self.n.setdefault(stat, [0, 0])
        cur[0] += int(hit)
        cur[1] += 1

    def add(self, obs: Any, action: Mapping[str, Any]) -> None:
        self.decisions += 1
        _tally(obs, action, self)

    def stats(self) -> dict[str, tuple[float | None, int]]:
        """(value, sample size) for every statistic."""
        c = self.counts
        out: dict[str, tuple[float | None, int]] = {
            "vpip": (c["vpip"] / c["hands"] if c["hands"] else None, c["hands"]),
            "pfr": (c["pfr"] / c["hands"] if c["hands"] else None, c["hands"]),
            "aggression": (c["bets_raises"] / max(1, c["calls"]), c["bets_raises"] + c["calls"]),
            "fold_to_reraise": (c["folded"] / c["faced"] if c["faced"] else None, c["faced"]),
        }
        for k, (hit, den) in self.n.items():
            out[k] = (hit / den if den else None, den)
        sizes = self.sizes
        if len(sizes) >= 2:
            mean = sum(sizes) / len(sizes)
            sd = math.sqrt(sum((x - mean) ** 2 for x in sizes) / (len(sizes) - 1))
            out["size_cv"] = (sd / mean if mean else None, len(sizes))
        out["size_corr"] = (_corr(sizes, self.scores), len(sizes))
        return out


class RecordingSeat(InProcessSeat):
    """An in-process seat that tallies its decisions (and keeps them when ``keep`` is set)."""

    def __init__(self, *args: Any, keep: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tally = Tally()
        self.keep = keep
        self.records: list[tuple[Any, dict | None]] = []

    def decide(self, observation: Any):
        d = super().decide(observation)
        if d.action is not None:
            self.tally.add(observation, d.action)
        if self.keep:
            self.records.append((observation, d.action))
        return d


def _bot(entry: Mapping[str, Any] | Callable[[], object]) -> object:
    return PopulationBot.from_spec(entry) if isinstance(entry, Mapping) else entry()


def play(bots: Sequence[tuple[str, object]], seed: str, ruleset: str = "practice_nlhe",
         config: Mapping[str, Any] | None = None, *, keep: bool = False
         ) -> tuple[Any, dict[int, tuple[str, RecordingSeat]]]:
    """One seeded table with ``(key, bot)`` pairs in seat order."""
    cfg = TournamentConfig(ruleset_id=ruleset, **{**OFFICIAL, **(config or {}), "initial_player_count": len(bots)})
    seats: dict[int, tuple[str, RecordingSeat]] = {}
    for seat, (key, bot) in zip(cfg.resolved_seats, bots, strict=True):
        info = TableInfo(table_id="t_pop", hero_seat=seat, ruleset_id=ruleset, config=cfg.resolved(),
                         config_hash=cfg.config_hash(), seats=cfg.resolved_seats, decision_budget_ns=cfg.bot_budget_ns,
                         policy_seed=policy_seed(seed, seat))
        agent = RecordingSeat(seat, bot, bot_info(bot), info, "population", keep=keep)
        agent.start()
        seats[seat] = (key, agent)
    table = seeded_table(cfg, seed, table_id="t_pop")
    play_table(table, {s: a for s, (_, a) in seats.items()})
    return table, seats


def seatings(count: int, *, tables: int, seed: str, seats: int = 8) -> list[list[int]]:
    """Entry indices per table: a seeded round-robin, each entry at most once per table."""
    order = list(range(count))
    random.Random(f"pop-census|{seed}").shuffle(order)
    out: list[list[int]] = []
    pos = 0
    for _ in range(tables):
        picked: list[int] = []
        while len(picked) < min(seats, count):
            i = order[pos % len(order)]
            pos += 1
            if i not in picked:
                picked.append(i)
        out.append(picked)
    return out


def _run_tables(args: tuple[Sequence[Entry], list[tuple[int, list[int]]], str, str, Mapping[str, Any] | None]
                ) -> dict[str, Tally]:
    entries, plan, seed, ruleset, config = args
    out: dict[str, Tally] = {}
    for t, picked in plan:
        bots = [(entries[i][0], _bot(entries[i][1])) for i in picked]
        table, seated = play(bots, f"{seed}-{t}", ruleset, config)
        tracker = StatsTracker(table.seat_ring)
        tracker.feed(project_events(table.log, Viewer.spectator()))
        for seat, (key, agent) in seated.items():
            tally = agent.tally
            c = tracker.counts[seat]
            for k in TRACKER_KEYS:
                tally.counts[k] += getattr(c, k)
            tally.fallbacks.update(agent.stats.fallbacks)
            tally.max_wall_ns = max([tally.max_wall_ns, *agent.stats.worker_wall_ns])
            out.setdefault(key, Tally()).merge(tally)
    return out


def census(entries: Sequence[Entry], *, tables: int, seed: str, ruleset: str = "practice_nlhe", seats: int = 8,
           config: Mapping[str, Any] | None = None, jobs: int = 1) -> dict[str, Tally]:
    """Seat ``entries`` round-robin over ``tables`` tables; one merged :class:`Tally` per key.

    The result does not depend on ``jobs`` (tables are seeded by index); ``jobs > 1`` spawns
    worker processes and needs every entry to be a member spec.
    """
    plan = list(enumerate(seatings(len(entries), tables=tables, seed=seed, seats=seats)))
    if jobs <= 1:
        return _run_tables((entries, plan, seed, ruleset, config))
    if not all(isinstance(e[1], Mapping) for e in entries):
        raise ValueError("parallel censuses need member specs, not bot factories")
    chunks = [(entries, plan[j::jobs], seed, ruleset, config) for j in range(jobs)]
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        parts = pool.map(_run_tables, chunks)
    out: dict[str, Tally] = {}
    for part in parts:
        for key, tally in part.items():
            out.setdefault(key, Tally()).merge(tally)
    return {k: out[k] for k in sorted(out)}


def family_stats(tallies: Sequence[Tally]) -> dict[str, tuple[float | None, int]]:
    """(value, sample size) for every statistic, pooled over ``tallies``."""
    total = Tally()
    for t in tallies:
        total.merge(t)
    return total.stats()


def _corr(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 10:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _tally(obs: Any, action: Mapping[str, Any], t: Tally) -> None:
    s = parse(obs)
    kind = action["type"]
    raised = kind == "raise_to"
    entered = kind in ("call", "raise_to")
    legal = obs.legal
    all_in = raised and action.get("amount") == legal.all_in_raise_to
    if raised:
        t.count("allin_share", all_in)
        low = legal.min_full_raise_to
        if s.to_call > 0 and low is not None and low != legal.all_in_raise_to:
            t.count("min_raise_share", action.get("amount") == low)
    if s.to_call and s.to_call >= (s.stack + s.committed) / 3:
        t.count("big_call", entered)
    if s.street == "preflop":
        if raised and s.m < 8:
            t.count("short_allin", all_in)
        if entered and not (kind == "call" and s.to_call == 0):
            t.count("junk_entry", cards.preflop_top(obs.hole_cards, obs.ruleset_id) > 0.35)
        if s.pre_raises == 0 and s.to_call > 0:
            t.count("limp", kind == "call")
            if s.limpers:
                t.count("iso", raised)
        if s.pre_raises == 1 and not s.hero_raised_pre:
            t.count("three_bet", raised)
            t.count("cold_call", kind == "call")
            if scan(obs, s.hero).callers >= 1:
                t.count("squeeze", raised)
        return
    bucket, outs, _ = cards.strength_for(obs.ruleset_id)(s.hole, s.board)
    if s.street in ("flop", "turn") and bucket in ("weak", "medium") and outs >= 4:
        t.count("draw_aggr", raised)
        if s.to_call:
            t.count("draw_continue", entered)
    if s.to_call == 0:
        if s.street == "flop" and s.pre_aggressor == s.hero and not s.hero_checked:
            t.count("cbet", raised)
            tex = cards.texture(s.board)
            if tex == "dry" and s.opponents == 1:
                t.count("cbet_dry_hu", raised)
            if tex in ("wet", "monotone"):
                t.count("cbet_wet", raised)
        if bucket in ("strong", "monster") and not s.hero_checked:
            t.count("strong_check", kind == "check")
        if raised and not all_in and obs.pot_total:
            t.sizes.append((action["amount"] - s.cur) / obs.pot_total)
            t.scores.append(cards.BUCKET_SCORE[bucket])
        if bucket == "weak" and outs < 8:
            t.count("mw_bluff" if s.opponents >= 2 else "hu_bluff", raised)
    else:
        if raised:
            t.count("xr_share", s.hero_checked)
