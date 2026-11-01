"""Incremental per-seat feature and belief engine: one :meth:`FeatureEngine.update` per decision.

Each call consumes only the public events it has not seen (by ``seq``) from the
observation's ``action_history`` and, in event order:

* rebuilds the public hand context (:class:`~strategy.features.state.HandContext`);
* for every voluntary action, predicts it from the actor's hierarchical action posterior
  (with the actor's family mixture as the prior), reweights the actor's bucketed range
  by the soft-threshold likelihood, then adds the action to the posterior and to the
  family posterior;
* applies card removal when board cards arrive;
* buffers events for Harrington's table-reading counts
  (:class:`~arena.table.player_stats.StatsTracker`), which catch up when queried
  (:meth:`FeatureEngine.stats_rates`) or when the buffer fills.

Card features are computed once per street and equity is a table lookup, so a decision
costs a bounded number of dictionary and list operations whatever the history length.

:meth:`to_state` keeps only what outlives a hand (posteriors, family evidence, stats)
plus the current hand's ranges, within ``max_bytes``. :meth:`from_state` restores it;
the next observation's ``action_history`` rebuilds the hand context without
re-counting actions the saved state already includes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from arena.table.player_stats import StatsTracker
from strategy.beliefs.posterior import (POST_FACING_BET, POST_UNBET, PRE_FACING_RAISE, PRE_FACING_RERAISE,
                                        ActionPosterior, Prediction, context_cell, from_record)
from strategy.beliefs.ranges import RangeBeliefs, threshold_likelihood
from strategy.beliefs.types import UNKNOWN, FamilyPosterior, FamilyTable
from strategy.features.cards import (NO_BLOCKERS, NO_DELTA, NO_DRAWS, NO_TEXTURE, CardFeatures, canonical_hole_key,
                                     capabilities, card_ints, hole_label, made_hand, postflop_cards, postflop_key,
                                     street_delta, texture)
from strategy.features.equity import EquityTables
from strategy.features.ranges import RangeFeatures, range_features
from strategy.features.state import ActionRecord, HandContext, PointModel, StateFeatures, state_features

STATE_VERSION = 1
MAX_STATE_BYTES = 48 * 1024
"""Default serialized budget, leaving room under the arena's 64 KiB bot state limit."""
STREET_OF_BOARD = {0: 0, 3: 1, 4: 2, 5: 3}
STATS_BUFFER = 2048
"""Events buffered for the lazily fed stats tracker before a forced catch-up."""


@dataclass(frozen=True)
class EngineConfig:
    kappa: tuple[float, float, float] = (10.0, 8.0, 5.0)
    """Pseudo-counts shrinking pool -> seat -> context cell."""
    floor: float = 0.02
    decay: float = 0.997
    """Per-hand recency decay of action counts."""
    family_prior: bool = True
    unknown_prior: float = 0.25
    family_forget: float = 0.995
    family_floor: float = 0.005
    stats: bool = True
    max_showdowns: int = 2
    tables_dir: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionFeatures:
    state: StateFeatures
    cards: CardFeatures
    ranges: RangeFeatures

    def vector(self) -> tuple[float, ...]:
        return self.state.vector() + card_vector(self.cards) + range_vector(self.ranges)


CARD_FIELDS = ("equity", "equity_sd", "percentile", "bucket", "made_level", "made_strength", "outs", "draw_class",
               "flush_draw", "nut_flush_draw", "oesd", "gutshot", "nut_flush_blocker", "nut_fd_blocker",
               "straight_blockers", "max_suit", "paired", "straight_windows", "draw_windows", "dynamic", "saturation",
               "turn_class", "level_delta", "equity_from_table")
RANGE_FIELDS = ("own_rank", "own_middle", "own_mean", "own_top", "main_mean", "main_top", "field_top", "field_mean",
                "field_strongest", "range_advantage", "nut_advantage", "capped_opponents", "fold_all",
                "main_fold_to_raise")


def card_vector(c: CardFeatures) -> tuple[float, ...]:
    m, d, b, t = c.made, c.draws, c.blockers, c.texture
    return (c.equity, c.equity_sd, c.percentile, float(c.bucket), float(m.level) if m else 0.0,
            float(m.value >> 20) if m else 0.0, float(d.outs), float(d.draw_class), float(d.flush_draw),
            float(d.nut_flush_draw), float(d.oesd), float(d.gutshot), float(b.nut_flush), float(b.nut_flush_draw),
            float(b.straight), float(t.max_suit), float(t.paired), float(t.straight_windows), float(t.draw_windows),
            t.dynamic, t.saturation, float(c.delta.turn_class), float(c.level_delta),
            float(c.equity_source != "prior"))


def range_vector(r: RangeFeatures) -> tuple[float, ...]:
    own, main = r.own, r.main
    main_fold = next((o.fold_to_raise for o in r.opponents if o.seat == r.main_seat), 0.0)
    return (r.own_rank, r.own_middle, own.mean if own else 0.0, own.top if own else 0.0, main.mean if main else 0.0,
            main.top if main else 0.0, r.field_top, r.field_mean, r.field_strongest, r.range_advantage,
            r.nut_advantage, float(r.capped_opponents), r.fold_all, main_fold)


def _event(e: Any) -> tuple[int, str | None, str, Mapping[str, Any]]:
    if isinstance(e, dict):
        return e["seq"], e.get("hand_id"), e["type"], e["data"]
    return e.seq, e.hand_id, e.type, e.data


class FeatureEngine:
    def __init__(self, ruleset_id: str, hero_seat: int, seats: Sequence[int], config: EngineConfig | None = None) -> None:
        cfg = self.config = config or EngineConfig()
        self.ruleset_id = ruleset_id
        self.hero = hero_seat
        self.seats = list(seats)
        self.caps = capabilities(ruleset_id)
        self.equity = EquityTables(self.caps, cfg.tables_dir)
        table = FamilyTable.load(ruleset_id, cfg.tables_dir) if cfg.family_prior else None
        self.families = FamilyPosterior(table, unknown_prior=cfg.unknown_prior, forget=cfg.family_forget,
                                        floor=cfg.family_floor)
        self.posterior = ActionPosterior(kappa=cfg.kappa, floor=cfg.floor, decay=cfg.decay,
                                         exclude_from_pool=(hero_seat,))
        self.ranges = RangeBeliefs(self.equity.bucket_combos, self.equity.bucket_card)
        self.stats = StatsTracker(self.seats, max_showdowns=cfg.max_showdowns) if cfg.stats else None
        self.ctx = HandContext(self.seats)
        self.points = PointModel()
        self.last_seq = -1
        self.replay_until = -1
        self.decisions = 0
        self._hand_id: str | None = None
        self._hole: tuple[int, ...] = ()
        self._hole_key = 0
        self._label = ""
        self._pre: dict[int, Any] = {}
        self._board_n = -1
        self._street: tuple[Any, ...] = ()
        self._pending: list[Any] = []

    # --------------------------------------------------------------- events

    def _on_record(self, rec: ActionRecord) -> None:
        seat = rec.seat
        if rec.kind == "fold":
            self.ranges.fold(seat)
        ob = from_record(rec)
        if ob is None:
            return
        hero = seat == self.hero
        if rec.kind != "fold":
            w = self.ranges.weights.get(seat)
            if w is not None:
                prior = None if hero or not self.families.enabled else self.families.mixture(
                    seat, ob.opportunity, rec.eff_bb)
                pred = self.posterior.predict(seat, ob.opportunity, ob.cell, prior=prior)
                self.ranges.update(seat, threshold_likelihood(w, pred.probs, ob.category, street=rec.street,
                                                              passive_first=ob.opportunity != POST_UNBET))
        self.posterior.observe(ob)
        if not hero:
            self.families.observe(ob, rec.eff_bb)

    def feed(self, events: Sequence[Any], hand_id: str | None = None) -> None:
        """Consume public events (any order of arrival; duplicates by ``seq`` are skipped)."""
        ctx = self.ctx
        fresh = []
        for e in events:
            seq, eh, etype, d = _event(e)
            if seq <= self.last_seq:
                continue
            self.last_seq = seq
            fresh.append(e)
            ctx.apply(seq, eh, etype, d)
            if seq <= self.replay_until:
                ctx.records.clear()
                continue
            if etype == "ActionApplied":
                for rec in ctx.drain():
                    self._on_record(rec)
            elif etype == "HandStarted":
                hole = self._hole if eh is not None and eh == hand_id else ()
                self.ranges.new_hand(ctx.dealt, self.hero, hole)
                self.posterior.advance(ctx.hand_number)
            elif etype == "BoardDealt":
                self.ranges.add_board(card_ints(d["cards"]))
        if self.stats is not None and fresh:
            self._pending.extend(fresh)
            if len(self._pending) > STATS_BUFFER:
                self.flush_stats()

    # ------------------------------------------------------------ decisions

    def _new_hand(self, obs: Any) -> None:
        self._hand_id = obs.hand_id
        self._hole = card_ints(obs.hole_cards)
        self._hole_key = canonical_hole_key(self._hole)
        self._label = hole_label(self._hole)
        self._pre = {}
        self._board_n = -1

    def _street_cards(self, board: Sequence[str]) -> tuple[Any, ...]:
        n = len(board)
        if n == self._board_n:
            return self._street
        self._board_n = n
        caps, hole = self.caps, self._hole
        if n == 0 or not hole:
            self._street = (None, NO_DRAWS, NO_BLOCKERS, NO_TEXTURE, NO_DELTA, 0, -1)
            return self._street
        b = card_ints(board)
        made, dr, bl, tx = postflop_cards(hole, b, caps)
        street = STREET_OF_BOARD[n]
        if n > 3:
            prev = b[:-1]
            delta = street_delta(texture(prev, caps), tx, b[-1], prev)
            level_delta = made.level - made_hand(hole, prev, caps).level
        else:
            delta, level_delta = NO_DELTA, 0
        self._street = (made, dr, bl, tx, delta, level_delta, postflop_key(street, made.level, dr.draw_class, tx.key))
        return self._street

    def _preflop(self, opponents: int) -> Any:
        hit = self._pre.get(opponents)
        if hit is None:
            hit = self._pre[opponents] = self.equity.preflop(self._hole, opponents)
        return hit

    def predict(self, seat: int, opportunity: int, *, street: int | None = None, eff_bb: float | None = None,
                live: int | None = None, initiative: bool | None = None) -> Prediction:
        """The posterior's prediction for ``seat`` in ``opportunity`` given the current hand's context."""
        ctx = self.ctx
        st = ctx.street if street is None else street
        if eff_bb is None:
            eff_bb = min(ctx.start.get(seat, 0), ctx.start.get(self.hero, 0)) / max(1, ctx.bb)
        n = len(ctx.dealt) - len(ctx.folded) if live is None else live
        init = (st > 0 and ctx.street_aggressors[st - 1] == seat) if initiative is None else initiative
        cell = context_cell(st, ctx.pos_group(seat), eff_bb, n, init)
        prior = self.families.mixture(seat, opportunity, eff_bb) if self.families.enabled and seat != self.hero \
            else None
        return self.posterior.predict(seat, opportunity, cell, prior=prior)

    def update(self, obs: Any) -> DecisionFeatures:
        if obs.hand_id != self._hand_id:
            self._new_hand(obs)
        self.feed(obs.action_history, obs.hand_id)
        self.decisions += 1
        ctx = self.ctx
        hero = self.hero
        state = state_features(obs, ctx, self.points)
        street = state.street
        k = state.opponents
        made, dr, bl, tx, delta, level_delta, cell = self._street_cards(obs.board)
        pre = self._preflop(k)
        if cell >= 0:
            post = self.equity.postflop(cell, made.level, dr.draw_class, k)
            eq, sd, src = post.equity, post.sd, post.source
        else:
            eq, sd, src = pre.equity, pre.sd, pre.source
        cards = CardFeatures(street, self._hole_key, self._label, eq, sd, src, pre.percentile, pre.bucket, made, dr,
                             bl, tx, delta, level_delta)

        opp_k = (PRE_FACING_RAISE if ctx.raises == 0 else PRE_FACING_RERAISE) if street == 0 else POST_FACING_BET
        live_n = state.live
        hero_start = ctx.start.get(hero, 0)
        bb = max(1, state.bb)
        fold: dict[int, tuple[float, float]] = {}
        fams: dict[int, tuple[str, float]] = {}
        opps = []
        prev_aggr = ctx.street_aggressors[street - 1] if street > 0 else -1
        enabled = self.families.enabled
        names = self.families.names
        for p in obs.players:
            s = p.seat
            if s == hero or p.status not in ("active", "all_in"):
                continue
            opps.append(s)
            if enabled:
                fp = self.families.probs(s)
                top = max(fp)
                fams[s] = (names[fp.index(top)], top)
            else:
                fams[s] = (UNKNOWN, 1.0)
            if p.status == "all_in":
                fold[s] = (0.0, 0.0)
                continue
            eff = min(ctx.start.get(s, 0), hero_start) / bb
            c = context_cell(street, ctx.pos_group(s), eff, live_n, prev_aggr == s)
            prior = self.families.mixture(s, opp_k, eff) if enabled else None
            pr = self.posterior.predict(s, opp_k, c, prior=prior)
            fold[s] = (pr.probs[0], pr.ess_seat)
        ranges = range_features(self.ranges, hero, pre.bucket, pre.percentile, opps, ctx.aggressor, fold, fams)
        return DecisionFeatures(state, cards, ranges)

    def flush_stats(self) -> StatsTracker | None:
        """Bring the table-reading stats up to date (they are fed lazily) and return them."""
        if self.stats is not None and self._pending:
            self.stats.feed(self._pending)
            self._pending = []
        return self.stats

    def stats_rates(self, seat: int) -> dict[str, float | None]:
        stats = self.flush_stats()
        return stats.rates(seat) if stats is not None else {}

    # ---------------------------------------------------------------- state

    def to_state(self, max_bytes: int = MAX_STATE_BYTES) -> dict[str, Any]:
        self.flush_stats()
        state = {"v": STATE_VERSION, "ruleset_id": self.ruleset_id, "hero": self.hero, "last_seq": self.last_seq,
                 "posterior": self.posterior.to_state(), "families": self.families.to_state(),
                 "ranges": self.ranges.to_state(),
                 "stats": self.stats.to_state() if self.stats is not None else None}
        size = len(json.dumps(state, separators=(",", ":")))
        while size > max_bytes:
            cells = len(self.posterior.cell)
            if self.stats is not None and state["stats"] is not None and cells < 8:
                state["stats"] = None
            elif not self.posterior.prune(int(cells * 0.75)):
                break
            state["posterior"] = self.posterior.to_state()
            size = len(json.dumps(state, separators=(",", ":")))
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, Any], seats: Sequence[int],
                   config: EngineConfig | None = None) -> FeatureEngine:
        if state.get("v") != STATE_VERSION:
            raise ValueError(f"unsupported engine state version {state.get('v')!r}")
        eng = cls(state["ruleset_id"], state["hero"], seats, config)
        eng.posterior.load_state(state["posterior"])
        eng.families.load_state(state["families"])
        eng.ranges.load_state(state["ranges"])
        if state.get("stats") is not None and eng.stats is not None:
            eng.stats = StatsTracker.from_state(state["stats"])
        eng.replay_until = state["last_seq"]
        return eng
