"""Authoritative in-process table: hands, eliminations, levels, the cap and scoring.

This is the only place game transitions happen; the headless runner, tests and
(later) the API all drive a :class:`Table` through :meth:`Table.submit`.
Deadlines, persistence and the command queue are layered on top later.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from arena.domain.cards import full_deck
from arena.domain.models import Action, LegalActions, TournamentConfig, level_index
from arena.domain.rng import Rng, SeededRng, SystemRng, shuffled
from arena.rules.base import HandEvent, HandResult, IllegalAction
from arena.rules.pokerkit_adapter import HandSpec, PokerKitHand
from arena.rules.profiles import RuleProfile, get_profile
from arena.table.positions import Positions, initial_positions, next_positions, positions_from_button
from arena.tournament.scoring import Placement, Standings

DeckSource = Callable[[int, RuleProfile], tuple[str, ...]]


class ActionRejected(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass
class HandRecord:
    hand_number: int
    hand_id: str
    level: int
    blinds: tuple[int, int]
    positions: Positions
    start_stacks: dict[int, int]
    deck: tuple[str, ...]
    events: list[HandEvent] = field(default_factory=list)
    result: HandResult | None = None
    eliminations: list[Placement] = field(default_factory=list)


@dataclass(frozen=True)
class Receipt:
    hand_id: str
    decision_id: str
    seat: int
    action: Action
    fallback_reason: str | None
    state_version: int


def seeded_deck_source(rng: Rng) -> DeckSource:
    def source(hand_number: int, profile: RuleProfile) -> tuple[str, ...]:
        return tuple(shuffled(full_deck(profile.ranks), rng))

    return source


class Table:
    def __init__(
        self,
        config: TournamentConfig,
        *,
        table_id: str = "t_local",
        env_rng: Rng | None = None,
        deck_source: DeckSource | None = None,
        initial_button: int | None = None,
        start_stacks: Mapping[int, int] | None = None,
    ) -> None:
        self.config = config
        self.table_id = table_id
        self.profile = get_profile(config.ruleset_id)
        self.seat_ring: tuple[int, ...] = tuple(config.resolved_seats)
        self.env_rng: Rng = env_rng if env_rng is not None else SystemRng()
        self.deck_source = deck_source or seeded_deck_source(self.env_rng)
        if start_stacks is not None:
            if set(start_stacks) != set(self.seat_ring):
                raise ValueError("start_stacks must cover every seat")
            self.stacks = dict(start_stacks)
        else:
            self.stacks = {s: config.starting_stack for s in self.seat_ring}
        self.total_chips = sum(self.stacks.values())
        self.standings = Standings(config.resolved_payouts, self.seat_ring)
        self.hands: list[HandRecord] = []
        self.hand: PokerKitHand | None = None
        self.positions: Positions | None = None
        self.hand_number = 0
        self.state_version = 0
        self._decision_seq = 0
        self.decision_id: str | None = None
        self.receipts: list[Receipt] = []
        self.complete = False
        self._initial_button = initial_button
        self._start_next_hand()

    # ------------------------------------------------------------------ views

    @property
    def live_seats(self) -> list[int]:
        return [s for s in self.seat_ring if self.stacks[s] > 0]

    @property
    def current(self) -> HandRecord | None:
        return self.hands[-1] if self.hands and self.hand is not None else None

    @property
    def actor_seat(self) -> int | None:
        if self.complete or self.hand is None:
            return None
        return self.hand.actor_seat

    def legal_actions(self) -> LegalActions:
        if self.hand is None or self.actor_seat is None:
            raise ActionRejected("no_decision", "no seat is to act")
        return self.hand.legal_actions()

    def points(self) -> dict[int, float]:
        if not self.complete:
            raise RuntimeError("table not complete")
        return self.standings.points()

    # --------------------------------------------------------------- commands

    def submit(
        self,
        seat: int,
        action: Action | Mapping[str, Any],
        *,
        hand_id: str | None = None,
        decision_id: str | None = None,
        expected_state_version: int | None = None,
    ) -> Receipt:
        """Validate and apply one action atomically. Rejections never mutate state."""
        if self.complete or self.hand is None:
            raise ActionRejected("table_complete", "table has finished")
        record = self.hands[-1]
        if hand_id is not None and hand_id != record.hand_id:
            raise ActionRejected("wrong_hand", f"hand {hand_id} is not current")
        if decision_id is not None and decision_id != self.decision_id:
            raise ActionRejected("stale_decision", f"decision {decision_id} is not current")
        if expected_state_version is not None and expected_state_version != self.state_version:
            raise ActionRejected("stale_version", "state version mismatch")
        if seat != self.actor_seat:
            raise ActionRejected("out_of_turn", f"seat {seat} is not to act")
        if not isinstance(action, Action):
            try:
                action = Action.model_validate(action, strict=True)
            except ValidationError as exc:
                raise ActionRejected("invalid_schema", str(exc.errors()[0]["msg"])) from None
        try:
            return self._apply(seat, action, None)
        except IllegalAction as exc:
            raise ActionRejected("illegal_action", str(exc)) from None

    def fallback_action(self) -> Action:
        legal = self.legal_actions()
        return Action(type="check") if legal.check else Action(type="fold")

    def apply_fallback(self, seat: int, reason: str) -> Receipt:
        """Consume the current decision with check if legal, otherwise fold."""
        if seat != self.actor_seat:
            raise ActionRejected("out_of_turn", f"seat {seat} is not to act")
        return self._apply(seat, self.fallback_action(), reason)

    def _apply(self, seat: int, action: Action, fallback_reason: str | None) -> Receipt:
        assert self.hand is not None and self.decision_id is not None
        record = self.hands[-1]
        decision_id = self.decision_id
        events = self.hand.apply(action)
        if fallback_reason is not None:
            record.events.append(HandEvent("ActionFallback", {"seat": seat, "reason": fallback_reason}))
        record.events.extend(events)
        self.state_version += 1
        receipt = Receipt(record.hand_id, decision_id, seat, action, fallback_reason, self.state_version)
        self.receipts.append(receipt)
        if self.hand.is_complete:
            self._settle_hand()
        else:
            self._new_decision()
        return receipt

    # ------------------------------------------------------------- hand loop

    def _new_decision(self) -> None:
        self._decision_seq += 1
        self.decision_id = f"d_{self._decision_seq:05d}"

    def _start_next_hand(self) -> None:
        while True:
            live = self.live_seats
            if len(live) < 2 or self.hand_number >= self.config.hand_cap:
                self._finalize()
                return
            self.hand_number += 1
            if self.positions is None:
                if self._initial_button is not None:
                    self.positions = positions_from_button(self.seat_ring, live, self._initial_button)
                else:
                    self.positions = initial_positions(self.seat_ring, live, self.env_rng)
            else:
                self.positions = next_positions(self.positions, self.seat_ring, live)
            pos = self.positions
            sb, bb = self.config.blinds_for_hand(self.hand_number)
            deck = self.deck_source(self.hand_number, self.profile)
            start = {s: self.stacks[s] for s in live}
            spec = HandSpec(
                profile=self.profile,
                seat_ring=self.seat_ring,
                stacks=start,
                button=pos.button,
                small_blind_seat=pos.small_blind_seat(live),
                big_blind_seat=pos.big_blind,
                small_blind=sb,
                big_blind=bb,
                deck=deck,
                local_rules=self.config.local_rules,
            )
            record = HandRecord(
                hand_number=self.hand_number,
                hand_id=f"h_{self.hand_number:03d}",
                level=level_index(self.hand_number, self.config.hands_per_level),
                blinds=(sb, bb),
                positions=pos,
                start_stacks=start,
                deck=deck,
            )
            self.hands.append(record)
            self.hand = PokerKitHand(spec)
            record.events.extend(self.hand.events)
            self.state_version += 1
            if not self.hand.is_complete:
                self._new_decision()
                return
            self._record_result()

    def _record_result(self) -> None:
        assert self.hand is not None
        record = self.hands[-1]
        result = self.hand.result()
        record.result = result
        for seat, stack in result.final_stacks.items():
            self.stacks[seat] = stack
        if sum(self.stacks.values()) != self.total_chips:
            raise AssertionError("table chips not conserved")
        record.eliminations = self.standings.record_eliminations(
            self.hand_number, record.start_stacks, self.stacks
        )
        self.hand = None
        self.decision_id = None

    def _settle_hand(self) -> None:
        self._record_result()
        self._start_next_hand()

    def _finalize(self) -> None:
        self.hand = None
        self.decision_id = None
        survivors = {s: self.stacks[s] for s in self.standings.remaining}
        self.final_placements = self.standings.finalize(self.hand_number, survivors)
        self.complete = True
        self.state_version += 1


def seeded_table(config: TournamentConfig, seed: int | str, **kwargs: Any) -> Table:
    return Table(config, env_rng=SeededRng(seed, "env"), **kwargs)
