"""Spec 12.2 information-set tests and 12.1 invariant 7 (views hold only authorized cards).

Two worlds share the hero's hole cards, the board and all public history, but
differ in opponents' hole cards, burns and undealt order. Until a card is
revealed, every serialized view the hero or a spectator can obtain must be
byte-identical across the two worlds, and so must a deterministic policy's
choice.
"""

from __future__ import annotations

import re

import pytest

from arena.domain.canonical import canonical_hash, canonical_json
from arena.domain.cards import full_deck
from arena.domain.models import Action, LocalRules, TournamentConfig
from arena.domain.rng import SeededRng, shuffled
from arena.rules.pokerkit_adapter import HandSpec, engine_order
from arena.rules.profiles import get_profile
from arena.table.driver import legal_random
from arena.table.projections import (
    ProjectionDenied,
    Viewer,
    observation,
    table_history_view,
    table_view,
)
from arena.table.service import ActionRejected, Table
from tests.support.tables import RULESETS, play_table

CARD = re.compile(r"^[2-9TJQKA][cdhs]$")


def card_strings(obj) -> set[str]:
    found: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and CARD.match(x):
            found.add(x)

    walk(obj)
    return found


def dumped(model) -> str:
    return canonical_json(model.model_dump(mode="json"))


class TwoWorlds:
    """Deck sources for two tables that differ only in cards the hero cannot see."""

    def __init__(self, config: TournamentConfig, seed: str, hero: int):
        self.config, self.seed, self.hero = config, seed, hero
        self.profile = get_profile(config.ruleset_id)
        self.tables: dict[str, Table] = {}

    def source(self, world: str):
        def deck(hand_number: int, profile) -> tuple[str, ...]:
            base = shuffled(full_deck(profile.ranks), SeededRng(self.seed, f"deck-{hand_number}"))
            table = self.tables[world]
            if world == "a":
                return tuple(base)
            live = [s for s in table.seat_ring if table.stacks[s] > 0]
            pos = table.positions
            spec = HandSpec(profile=profile, seat_ring=table.seat_ring, stacks={s: 1 for s in live},
                            button=pos.button, small_blind_seat=pos.small_blind_seat(live),
                            big_blind_seat=pos.big_blind, small_blind=1, big_blind=2,
                            deck=tuple(base), local_rules=LocalRules())
            order = engine_order(spec)
            n, h = len(order), profile.hole_card_count
            fixed = {h * n + k for k in (1, 2, 3, 5, 7)}
            if self.hero in order:
                fixed |= {r * n + order.index(self.hero) for r in range(h)}
            free = [i for i in range(len(base)) if i not in fixed]
            moved = shuffled([base[i] for i in free], SeededRng(self.seed, f"perm-{hand_number}"))
            out = list(base)
            for i, card in zip(free, moved):
                out[i] = card
            return tuple(out)

        return deck

    def build(self) -> tuple[Table, Table]:
        for world in ("a", "b"):
            # The deck source reads the table's positions while __init__ starts hand 1,
            # so the object must be registered before __init__ runs.
            table = Table.__new__(Table)
            self.tables[world] = table
            Table.__init__(table, self.config, table_id="t_infoset", deck_source=self.source(world),
                           initial_button=self.config.resolved_seats[0])
        return self.tables["a"], self.tables["b"]


def hero_policy(obs) -> Action:
    """Deterministic in the observation alone."""
    legal = obs.legal
    digest = int(canonical_hash(obs.model_dump(mode="json"), domain="hero-policy")[:8], 16)
    options = [Action(type="check") if legal.check else Action(type="call") if legal.call_amount else Action(type="fold")]
    if legal.fold:
        options.append(Action(type="fold"))
    if legal.min_full_raise_to is not None:
        options.append(Action(type="raise_to", amount=legal.min_full_raise_to))
    return options[digest % len(options)]


def revealed(table: Table) -> bool:
    return any(e.type == "CardsRevealed" for e in table.log)


def compare_worlds(ruleset: str, n: int, seed: str) -> tuple[int, int]:
    config = TournamentConfig(ruleset_id=ruleset, initial_player_count=n, starting_stack=300)
    hero = config.resolved_seats[-1]
    a, b = TwoWorlds(config, seed, hero).build()
    compared = differing_private = 0
    while not a.complete and not b.complete:
        if revealed(a) or revealed(b):
            break
        for viewer in (Viewer.for_seat(hero), Viewer.spectator()):
            assert dumped(table_view(a, viewer)) == dumped(table_view(b, viewer))
            assert dumped(table_history_view(a, viewer)) == dumped(table_history_view(b, viewer))
        compared += 1
        if a.hand is not None and b.hand is not None and a.hand.hole_cards != b.hand.hole_cards:
            differing_private += 1
        seat = a.actor_seat
        assert seat == b.actor_seat
        assert a.legal_actions() == b.legal_actions()
        if seat == hero:
            obs_a, obs_b = observation(a, hero), observation(b, hero)
            assert dumped(obs_a) == dumped(obs_b)
            action = hero_policy(obs_a)
            assert hero_policy(obs_b) == action
        else:
            rng = SeededRng(f"{seed}-{a.state_version}", "opponent")
            action = legal_random(seat, a.legal_actions(), rng)
        a.submit(seat, action)
        b.submit(seat, action)
    return compared, differing_private


@pytest.mark.parametrize("n", [2, 3, 5, 8])
@pytest.mark.parametrize("ruleset", RULESETS)
def test_views_identical_across_hidden_card_worlds(ruleset, n):
    compared = differing = 0
    for k in range(6):
        c, d = compare_worlds(ruleset, n, f"iset-{ruleset}-{n}-{k}")
        compared += c
        differing += d
    assert compared >= 6
    assert differing > 0, "the two worlds never differed in private cards; the test would be vacuous"


# -------------------------------------------------------------- leak scanning


def allowed_cards(table: Table, seat: int | None) -> set[str]:
    hand = table.hand
    if hand is None:
        return set()
    allowed = set(hand.board) | {c for cards in hand.revealed.values() for c in cards}
    if seat is not None:
        allowed |= set(hand.hole_cards.get(seat, []))
    return allowed


def history_allowed(table: Table, seat: int | None) -> set[str]:
    out: set[str] = set()
    for e in table.log:
        if e.type == "BoardDealt" or e.type == "CardsRevealed":
            out |= set(e.data["cards"])
        elif e.type == "HoleDealt" and seat is not None and e.data["seat"] == seat:
            out.add(e.data["card"])
    return out


@pytest.mark.parametrize("ruleset", RULESETS)
def test_views_and_errors_never_leak_hidden_cards(ruleset):
    for n in (2, 4, 8):
        config = TournamentConfig(ruleset_id=ruleset, initial_player_count=n, starting_stack=300)
        from arena.table.service import seeded_table

        table = seeded_table(config, f"leak-{n}")
        rng = SeededRng(f"leak-{n}", "policy")
        steps = 0
        while not table.complete:
            steps += 1
            for seat in table.seat_ring:
                view = table_view(table, Viewer.for_seat(seat)).model_dump(mode="json")
                assert card_strings(view) <= allowed_cards(table, seat)
            spectator = table_view(table, Viewer.spectator()).model_dump(mode="json")
            assert card_strings(spectator) <= allowed_cards(table, None)
            if steps % 5 == 0:
                for seat in table.seat_ring:
                    hist = table_history_view(table, Viewer.for_seat(seat)).model_dump(mode="json")
                    assert card_strings(hist) <= history_allowed(table, seat)
                errors = []
                actor = table.actor_seat
                other = next(s for s in table.seat_ring if s != actor)
                for args in ((other, {"type": "check"}), (actor, {"type": "raise_to", "amount": 10**9}),
                             (actor, {"type": "bet", "amount": 3}), (actor, {"type": "call", "amount": 5})):
                    try:
                        table.submit(*args)
                    except ActionRejected as exc:
                        errors.append(str(exc))
                for kwargs in ({"decision_id": "d_bogus"}, {"hand_id": "h_999"}, {"expected_state_version": -1}):
                    try:
                        table.submit(actor, {"type": "fold"}, **kwargs)
                    except ActionRejected as exc:
                        errors.append(str(exc))
                assert len(errors) == 7
                for message in errors:
                    assert not re.search(r"\b[2-9TJQKA][cdhs]\b", message), message
            table.submit(table.actor_seat, legal_random(table.actor_seat, table.legal_actions(), rng))
        for seat in table.seat_ring:
            final = table_history_view(table, Viewer.for_seat(seat)).model_dump(mode="json")
            assert card_strings(final) <= history_allowed(table, seat)
        public = table_history_view(table, Viewer.spectator()).model_dump(mode="json")
        assert card_strings(public) <= history_allowed(table, None)


def test_hole_cards_visible_only_to_their_owner():
    table = play_table("practice_nlhe", 4, "owner", hand_cap=1, hands_per_level=1)
    holes = {}
    for e in table.log:
        if e.type == "HoleDealt":
            holes.setdefault(e.data["seat"], []).append(e.data["card"])
    for seat in table.seat_ring:
        view = table_history_view(table, Viewer.for_seat(seat))
        seen = [e.data["card"] for e in view.events if e.type == "HoleDealt" and "card" in e.data]
        assert seen == holes[seat]
    spectator = table_history_view(table, Viewer.spectator())
    assert all("card" not in e.data for e in spectator.events if e.type == "HoleDealt")
    assert all(e.type not in ("CardBurned", "EngineChipsPushed", "OddChipCorrection") for e in spectator.events)
    assert all("state_hash" not in e.data for e in spectator.events)
    assert all("deck" not in e.data and "start_hash" not in e.data for e in spectator.events)
    assert all("rng" not in e.data for e in spectator.events)
    assert all("rejected" not in e.data for e in spectator.events)


def test_post_table_review_requires_completed_table():
    config = TournamentConfig(initial_player_count=3, starting_stack=300)
    from arena.table.service import seeded_table

    live = seeded_table(config, "review")
    with pytest.raises(ProjectionDenied):
        table_history_view(live, Viewer.reviewer())
    with pytest.raises(ProjectionDenied):
        table_view(live, Viewer.reviewer())
    with pytest.raises(ProjectionDenied):
        table_view(live, Viewer.for_seat(7))
    not_actor = next(s for s in live.seat_ring if s != live.actor_seat)
    with pytest.raises(ProjectionDenied):
        observation(live, not_actor)
    done = play_table("practice_nlhe", 3, "review")
    full = table_history_view(done, Viewer.reviewer())
    assert len(full.events) == len(done.log)
    assert any("deck" in e.data for e in full.events)
    with pytest.raises(ProjectionDenied):
        observation(done, done.seat_ring[0])


def test_observation_serialization_is_stable_and_valid():
    from arena.domain.observations import Observation

    config = TournamentConfig(ruleset_id="uncertainty_plo4", initial_player_count=6, starting_stack=500)
    from arena.table.service import seeded_table

    live = seeded_table(config, "stable")
    rng = SeededRng("stable", "p")
    for _ in range(25):
        if live.complete:
            break
        obs = observation(live, live.actor_seat)
        text = obs.model_dump_json()
        assert Observation.model_validate_json(text) == obs
        assert observation(live, live.actor_seat).model_dump_json() == text
        assert len(obs.hole_cards) == 4
        live.submit(live.actor_seat, legal_random(live.actor_seat, live.legal_actions(), rng))
