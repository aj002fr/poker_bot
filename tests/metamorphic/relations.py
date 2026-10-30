"""Metamorphic relations for the table engine (spec section 12.2).

Each relation plays a *base* table and a transformed *variant* of it, maps the
variant's hand records back into the base frame, and requires exact equality,
hand for hand and event for event (blinds, deals, actions, bet collection, pot
awards, final stacks), plus equal eliminations, placements and points.

- ``suits``: a permutation of the four suits applied to every deck. Actions and
  results must be identical with the cards renamed.
- ``rotation``: every seat label moved by ``r`` around the eight-seat ring
  (``s -> (s + r) mod 8``), with the first button moved the same way and each
  player's policy stream following the player. Markers, actions and results must
  follow the players.
- ``scaling``: starting stacks and every blind level multiplied by ``k``; the
  variant's policy sees the legal actions divided by ``k`` and acts ``k`` times
  larger. Everything must scale exactly. A hand whose base settlement splits a
  pot unevenly (an odd chip) cannot scale exactly; it and the rest of that table
  are excluded and counted, never compared and never passed.
- ``table_id``: the same seed under another anonymous table label. Every hand
  record must be identical.
- ``scheduling``: one benchmark manifest run serially and with four parallel
  jobs; every table's saved history must be byte-identical
  (:func:`serial_vs_parallel`).

Policies are the seeded ``legal_random`` driver, one stream per player.
Legal actions do not depend on cards, so the suit and seat relations preserve
every random choice.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena.domain.models import Action, LegalActions, TournamentConfig, default_blind_levels
from arena.domain.rng import SeededRng
from arena.table.driver import legal_random
from arena.table.service import Table

SUITS = "cdhs"
SEAT_KEYS = {"seat"}
SEAT_LIST_KEYS = {"eligible", "seats", "folded"}
SEAT_DICT_KEYS = {"bets", "awards", "final_stacks", "stacks", "contributions", "transfers", "engine_awards"}
CHIP_KEYS = {"amount", "added", "street_committed"}
CHIP_DICT_KEYS = {"bets", "awards", "final_stacks", "stacks", "contributions", "transfers", "engine_awards"}
CARD_KEYS = {"card"}
CARD_LIST_KEYS = {"cards"}


class RelationMismatch(AssertionError):
    pass


@dataclass
class Frame:
    """How a variant table maps back into the base frame."""

    seat: Callable[[int], int] = lambda s: s
    card: Callable[[str], str] = lambda c: c
    chips: int = 1

    def chip(self, v: int) -> int:
        if v % self.chips:
            raise RelationMismatch(f"variant amount {v} is not a multiple of {self.chips}")
        return v // self.chips

    def value(self, key: str, v: Any) -> Any:
        if key in SEAT_KEYS and isinstance(v, int):
            return self.seat(v)
        if key in SEAT_LIST_KEYS and isinstance(v, (list, tuple)):
            return sorted(self.seat(s) for s in v)
        if key in SEAT_DICT_KEYS and isinstance(v, Mapping):
            scale = key in CHIP_DICT_KEYS
            return {self.seat(int(s)): (self.chip(x) if scale else x) for s, x in sorted(v.items())}
        if key in CHIP_KEYS and isinstance(v, int):
            return self.chip(v)
        if key in CARD_KEYS and isinstance(v, str):
            return self.card(v)
        if key in CARD_LIST_KEYS and isinstance(v, (list, tuple)):
            return [self.card(c) for c in v]
        if key == "action" and isinstance(v, Mapping):
            return {k: (self.chip(x) if k == "amount" and isinstance(x, int) else x) for k, x in v.items()}
        return v

    def event(self, e) -> tuple[str, dict[str, Any]]:
        return e.type, {k: self.value(k, v) for k, v in sorted(e.data.items())}


def hand_trace(table: Table, frame: Frame, upto: int | None = None) -> list[dict[str, Any]]:
    out = []
    for rec in table.hands[:upto]:
        pos = rec.positions
        out.append({
            "hand": rec.hand_number, "level": rec.level,
            "blinds": tuple(frame.chip(b) for b in rec.blinds),
            "positions": (frame.seat(pos.button), frame.seat(pos.small_blind), frame.seat(pos.big_blind),
                          pos.heads_up),
            "start_stacks": {frame.seat(s): frame.chip(v) for s, v in sorted(rec.start_stacks.items())},
            "events": [frame.event(e) for e in rec.events],
            "eliminations": sorted((frame.seat(p.seat), tuple(p.positions)) for p in rec.eliminations),
        })
    return out


def table_outcome(table: Table, frame: Frame) -> dict[str, Any]:
    return {
        "complete": table.complete,
        "stacks": {frame.seat(s): frame.chip(v) for s, v in sorted(table.stacks.items())},
        "placements": sorted((frame.seat(p.seat), tuple(p.positions), p.method)
                             for p in table.standings.placements.values()),
        "points": {frame.seat(s): p for s, p in sorted(table.points().items())},
    }


def _first_diff(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            if isinstance(x, dict) and isinstance(y, dict):
                for k in x:
                    if x.get(k) != y.get(k):
                        if k == "events":
                            for j, (ex, ey) in enumerate(zip(x[k], y[k])):
                                if ex != ey:
                                    return f"hand {x['hand']} event {j}: base {ex} vs variant {ey}"
                            return f"hand {x['hand']}: event count {len(x[k])} vs {len(y[k])}"
                        return f"hand {x.get('hand')}: {k} base {x[k]} vs variant {y.get(k)}"
            return f"item {i}: base {x} vs variant {y}"
    return f"length {len(a)} vs {len(b)}"


# ---------------------------------------------------------------- tables


Policy = Callable[[int, LegalActions], Action]


def play(table: Table, policy: Policy) -> Table:
    while not table.complete:
        seat = table.actor_seat
        table.submit(seat, policy(seat, table.legal_actions()), hand_id=table.hands[-1].hand_id,
                     decision_id=table.decision_id)
    return table


def random_policy(rngs: Mapping[int, SeededRng]) -> Policy:
    return lambda seat, legal: legal_random(seat, legal, rngs[seat])


def base_config(ruleset: str, n: int, *, hand_cap: int = 32, hands_per_level: int = 4,
                starting_stack: int = 2000) -> TournamentConfig:
    return TournamentConfig(ruleset_id=ruleset, initial_player_count=n, hand_cap=hand_cap,
                            hands_per_level=hands_per_level, starting_stack=starting_stack)


@dataclass
class Base:
    config: TournamentConfig
    seed: str
    table: Table
    decks: dict[int, tuple[str, ...]] = field(default_factory=dict)

    @property
    def button(self) -> int:
        return self.table.hands[0].positions.button


def play_base(config: TournamentConfig, seed: str, table_id: str = "t_meta") -> Base:
    table = Table(config, table_id=table_id, env_rng=SeededRng(seed, "env"))
    rngs = {s: SeededRng(seed, f"policy-seat-{s}") for s in config.resolved_seats}
    play(table, random_policy(rngs))
    return Base(config, seed, table, {rec.hand_number: rec.deck for rec in table.hands})


def _deck_source(decks: Mapping[int, tuple[str, ...]], card: Callable[[str], str] = lambda c: c):
    def source(hand_number: int, _profile) -> tuple[str, ...]:
        if hand_number not in decks:
            raise RelationMismatch(f"variant reached hand {hand_number}, which the base table never dealt")
        return tuple(card(c) for c in decks[hand_number])

    return source


def _compare(base: Base, variant: Table, frame: Frame, upto: int | None = None) -> str:
    want = hand_trace(base.table, Frame(), upto)
    got = hand_trace(variant, frame, upto)
    if want != got:
        raise RelationMismatch(_first_diff(want, got))
    if upto is None:
        a, b = table_outcome(base.table, Frame()), table_outcome(variant, frame)
        pa, pb = a.pop("points"), b.pop("points")
        if a != b:
            raise RelationMismatch(f"outcome base {a} vs variant {b}")
        if pa.keys() != pb.keys() or any(abs(pa[s] - pb[s]) > 1e-9 for s in pa):
            raise RelationMismatch(f"points base {pa} vs variant {pb}")
    return hashlib.sha256(json.dumps(got, sort_keys=True, default=str).encode()).hexdigest()


# -------------------------------------------------------------- relations


def suit_renaming(base: Base, perm: str) -> str:
    """``perm`` lists the new suit for each of ``cdhs`` (for example ``"hscd"``)."""
    rename = dict(zip(SUITS, perm))
    back = {v: k for k, v in rename.items()}
    variant = Table(base.config, table_id=base.table.table_id, deck_source=_deck_source(
        base.decks, lambda c: c[0] + rename[c[1]]), initial_button=base.button)
    rngs = {s: SeededRng(base.seed, f"policy-seat-{s}") for s in base.config.resolved_seats}
    play(variant, random_policy(rngs))
    return _compare(base, variant, Frame(card=lambda c: c[0] + back[c[1]]))


def seat_rotation(base: Base, r: int) -> str:
    seats = base.config.resolved_seats
    fwd = {s: (s + r) % 8 for s in seats}
    back = {v: k for k, v in fwd.items()}
    config = base.config.model_copy(update={"seats": sorted(fwd.values())})
    config = TournamentConfig.model_validate(config.model_dump())
    variant = Table(config, table_id=base.table.table_id, deck_source=_deck_source(base.decks),
                    initial_button=fwd[base.button])
    rngs = {fwd[s]: SeededRng(base.seed, f"policy-seat-{s}") for s in seats}
    play(variant, random_policy(rngs))
    return _compare(base, variant, Frame(seat=lambda s: back[s]))


def _scaled_legal(legal: LegalActions, k: int) -> LegalActions:
    data = legal.model_dump()
    for key in ("call_amount", "min_full_raise_to", "max_raise_to", "short_all_in_raise_to", "all_in_raise_to",
                "pot_raise_to"):
        v = data[key]
        if v is not None:
            if v % k:
                raise RelationMismatch(f"scaled legal {key}={v} is not a multiple of {k}")
            data[key] = v // k
    return LegalActions.model_validate(data)


def odd_chip_hand(base: Base) -> int | None:
    """The first base hand whose settlement split a pot unevenly, if any."""
    for rec in base.table.hands:
        for pot in rec.result.pots:
            winners = [s for s, v in pot.awards.items() if v > 0]
            if len(winners) > 1 and pot.amount % len(winners):
                return rec.hand_number
    return None


def chip_scaling(base: Base, k: int) -> tuple[str, int | None]:
    """Returns the trace hash and the first excluded hand (None when the whole table compared)."""
    cfg = base.config
    levels = cfg.blind_levels or default_blind_levels()
    config = TournamentConfig.model_validate({**cfg.model_dump(), "starting_stack": cfg.starting_stack * k,
                                              "blind_levels": [[sb * k, bb * k] for sb, bb in levels]})
    variant = Table(config, table_id=base.table.table_id, deck_source=_deck_source(base.decks),
                    initial_button=base.button)
    rngs = {s: SeededRng(base.seed, f"policy-seat-{s}") for s in cfg.resolved_seats}
    cut = odd_chip_hand(base)

    def policy(seat: int, legal: LegalActions) -> Action:
        action = legal_random(seat, _scaled_legal(legal, k), rngs[seat])
        return action if action.amount is None else Action(type=action.type, amount=action.amount * k)

    # Stop once the excluded hand has been dealt: hands before it are compared, it and later ones are not.
    while not variant.complete and (cut is None or len(variant.hands) < cut):
        seat = variant.actor_seat
        variant.submit(seat, policy(seat, variant.legal_actions()), hand_id=variant.hands[-1].hand_id,
                       decision_id=variant.decision_id)
    upto = None if cut is None else cut - 1
    return _compare(base, variant, Frame(chips=k), upto), cut


def table_relabel(base: Base, table_id: str) -> str:
    variant = Table(base.config, table_id=table_id, env_rng=SeededRng(base.seed, "env"))
    rngs = {s: SeededRng(base.seed, f"policy-seat-{s}") for s in base.config.resolved_seats}
    play(variant, random_policy(rngs))
    return _compare(base, variant, Frame())


def serial_vs_parallel(ruleset: str, players: int, blocks: int, base_seed: str, jobs: int = 4) -> dict[str, Any]:
    """Run one benchmark manifest serially and in parallel; require identical tables and histories."""
    from arena.agents.baselines import baseline_info, baseline_path
    from arena.benchmark.runner import Entrant, run_benchmark

    names = [n for n in ("legal_random", "calling_station", "check_fold", "nlhe_heuristic")
             if ruleset in baseline_info(n).rulesets]
    entrants = [Entrant(f"{names[i % len(names)]}#{i}", baseline_path(names[i % len(names)]), True)
                for i in range(players)]
    config = TournamentConfig(ruleset_id=ruleset, initial_player_count=players, hand_cap=60, hands_per_level=8)
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, j in (("serial", 1), ("parallel", jobs)):
            out = Path(tmp) / label
            run = run_benchmark(config=config, entrants=entrants, blocks=blocks, base_seed=base_seed,
                                isolation="in_process", jobs=j, out_dir=out, save_histories=True)
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((out / "tables").glob("*.json"))}
            data = run.manifest.model_dump(mode="json")
            jobs_used = data["process_layout"]["jobs"]
            runs[label] = {
                "tables": [(r.index, r.table_id, r.completed, r.hands, sorted(r.points.items()), r.error)
                           for r in run.results],
                "histories": files,
                "manifest": _without_measurements(data),
                "jobs": jobs_used,
            }
    s, p = runs["serial"], runs["parallel"]
    for key in ("tables", "histories", "manifest"):
        if s[key] != p[key]:
            raise RelationMismatch(f"serial vs {jobs} jobs differ in {key}")
    if s["jobs"] == p["jobs"]:
        raise RelationMismatch("serial and parallel runs used the same process layout")
    if not s["histories"] or len(s["histories"]) != blocks * players:
        raise RelationMismatch(f"expected {blocks * players} saved histories, got {len(s['histories'])}")
    return {"tables": len(s["tables"]),
            "histories_sha256": hashlib.sha256(repr(sorted(s["histories"].items())).encode()).hexdigest(),
            "manifest_sha256": hashlib.sha256(json.dumps(s["manifest"], sort_keys=True).encode()).hexdigest()}


# Wall-clock measurements and the process layout legitimately differ between runs; nothing else may.
MEASURED_KEYS = {"timing", "table_wall_s_mean", "created_at", "host", "process_layout"}


def _without_measurements(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _without_measurements(v) for k, v in obj.items() if k not in MEASURED_KEYS}
    if isinstance(obj, list):
        return [_without_measurements(v) for v in obj]
    return obj
