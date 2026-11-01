import random
from types import SimpleNamespace

import pytest

from arena.tournament.scoring import icm
from strategy.beliefs.posterior import POST_FACING_BET, POST_UNBET, PRE_FACING_RAISE, PRE_UNOPENED, from_record
from strategy.features.state import (BLINDS, LATE, NUMERIC_FIELDS, HandContext, icm_value, icm_values, state_features,
                                     survivor_icm)
from tests.strategy.feature_support import RULESETS, Actions, hand_events, recorded

PAYOUTS = [50, 30, 20, 0, 0, 0, 0, 0]


def test_icm_matches_the_arena_reference() -> None:
    rng = random.Random(4)
    for _ in range(200):
        n = rng.randint(2, 8)
        stacks = [rng.randint(1, 5000) for _ in range(n)]
        ref = icm(stacks, PAYOUTS[:n])
        assert icm_values(stacks, PAYOUTS[:n]) == pytest.approx(ref, abs=1e-9)
        for i in range(n):
            assert icm_value(stacks, PAYOUTS[:n], i) == pytest.approx(ref[i], abs=1e-9)


def test_survivor_icm_handles_a_bust() -> None:
    assert survivor_icm({0: 0, 1: 500, 2: 500}, [50, 30, 20], 0) == 20
    assert survivor_icm({0: 100}, [50, 30, 20], 0) == 50


def four_handed() -> tuple[HandContext, list[dict]]:
    stacks = {0: 2000, 1: 2000, 2: 2000, 3: 2000}
    events = hand_events(stacks, button=0, sb_seat=1, bb_seat=2)
    a = Actions(events, {1: 10, 2: 20})
    a.act(3, "raise_to", 60).act(0, "call").act(1, "fold").act(2, "call")
    a.board("flop", ["Ah", "7d", "2c"]).act(2, "check").act(3, "raise_to", 90).act(0, "call").act(2, "fold")
    ctx = HandContext([0, 1, 2, 3])
    ctx.feed(events)
    return ctx, events


def test_records_carry_the_opportunity_context() -> None:
    ctx, _ = four_handed()
    recs = ctx.drain()
    assert [r.seat for r in recs] == [3, 0, 1, 2, 2, 3, 0, 2]
    obs = [from_record(r) for r in recs]
    assert [o.opportunity for o in obs] == [PRE_UNOPENED, PRE_FACING_RAISE, PRE_FACING_RAISE, PRE_FACING_RAISE,
                                            POST_UNBET, POST_UNBET, POST_FACING_BET, POST_FACING_BET]
    assert [o.category for o in obs] == [2, 1, 0, 1, 0, 1, 1, 0]
    assert recs[1].to_call == 60 and recs[2].to_call == 50 and recs[3].to_call == 40
    assert recs[5].initiative and not recs[4].initiative  # the preflop raiser c-bets
    assert recs[5].pot_before == 190 and recs[5].live_before == 3
    assert ctx.pre_order == [3, 0, 1, 2] and ctx.post_order == [1, 2, 3, 0]
    assert ctx.pos_group(0) == LATE and ctx.pos_group(2) == BLINDS
    assert ctx.aggressor == 3 and ctx.pot == 370 and ctx.folded == {1, 2}


def test_records_persist_across_hands_until_drained() -> None:
    _, events = four_handed()
    second = hand_events({0: 2000, 1: 2000, 2: 2000, 3: 2000}, button=1, sb_seat=2, bb_seat=3, hand_number=2,
                         seq0=100, hand_id="h2")
    Actions(second, {2: 10, 3: 20}, hand_id="h2").act(0, "fold")
    ctx = HandContext([0, 1, 2, 3])
    ctx.feed(events + second)
    recs = ctx.drain()
    assert len(recs) == 9 and {r.hand_number for r in recs} == {1, 2}
    ctx.feed(events)  # already seen: ignored
    assert ctx.drain() == []


def test_heads_up_and_dead_small_blind_positions() -> None:
    hu = HandContext([0, 1])
    hu.feed(hand_events({0: 1000, 1: 1000}, button=0, sb_seat=0, bb_seat=1, heads_up=True))
    assert hu.pre_order == [0, 1] and hu.post_order == [1, 0]
    assert hu.pos_group(0) == LATE and hu.pos_group(1) == BLINDS
    dead = HandContext([0, 1, 2, 3, 4])
    dead.feed(hand_events({0: 1000, 2: 1000, 3: 1000, 4: 1000}, button=0, sb_seat=None, bb_seat=2))
    assert dead.sb_seat == -1 and dead.pre_order == [3, 4, 0, 2] and dead.post_order == [2, 3, 4, 0]


def _player(seat: int, stack: int, committed: int, status: str = "active") -> SimpleNamespace:
    return SimpleNamespace(seat=seat, stack=stack, hand_committed=committed, street_committed=committed,
                           status=status)


def _obs(players: list, *, hero: int, call: int, events: list, pot: int, bet_to: int) -> SimpleNamespace:
    legal = SimpleNamespace(call_amount=call, can_raise=False, min_full_raise_to=None, short_all_in_raise_to=None,
                            pot_raise_to=None)
    return SimpleNamespace(hero_seat=hero, players=players, blinds=SimpleNamespace(small_blind=10, big_blind=20),
                           street="preflop", legal=legal, current_bet_to=bet_to, pot_total=pot, hands_remaining=50,
                           next_level=None, surviving_player_count=3, hand_number=1, hand_cap=50,
                           initial_player_count=3, payouts=[50, 30, 20], seats=[0, 1, 2], action_history=events)


def test_side_pots_and_coverage() -> None:
    events = hand_events({0: 1000, 1: 300, 2: 2000}, button=0, sb_seat=1, bb_seat=2)
    Actions(events, {1: 10, 2: 20}).act(0, "fold")
    ctx = HandContext([0, 1, 2])
    ctx.feed(events)
    # hero (seat 0) covers the all-in seat 1 and is covered by seat 2
    players = [_player(0, 1000, 0), _player(1, 0, 300, "all_in"), _player(2, 1200, 800)]
    f = state_features(_obs(players, hero=0, call=800, events=events, pot=1100, bet_to=800), ctx)
    assert f.winnable_if_call == 1900 and f.uncontestable == 0 and f.side_pots == 1
    assert (f.covers, f.covered_by) == (1, 1) and f.all_in_opponents == 1
    assert f.pot_odds == pytest.approx(800 / 1900)
    # a short hero can win only what it matches
    players = [_player(0, 200, 0), _player(1, 0, 300, "all_in"), _player(2, 1200, 800)]
    f = state_features(_obs(players, hero=0, call=800, events=events, pot=1100, bet_to=800), ctx)
    assert f.call_amount == 200 and f.winnable_if_call == 600 and f.uncontestable == 700
    assert f.pot_odds_effective == pytest.approx(200 / 600)
    q, premium = f.clash(2)
    assert 0.0 < q < 1.0 and premium == pytest.approx(q - 0.5)


@pytest.mark.parametrize("ruleset_id", RULESETS)
def test_state_features_on_real_observations(ruleset_id: str) -> None:
    checked = heads_up = 0
    for seat, seats, observations in recorded(ruleset_id):
        for obs in observations:
            ctx = HandContext(seats)
            ctx.feed(obs.action_history)
            f = state_features(obs, ctx)
            assert f.to_call == (obs.legal.call_amount or 0)
            assert 0.0 <= f.pot_odds < 1.0 and f.spr >= 0.0
            assert f.is_button == (seat == obs.button_seat) and f.is_bb == (seat == obs.big_blind_seat)
            assert f.live == sum(1 for p in obs.players if p.status in ("active", "all_in"))
            assert sum(f.stack_vector) == pytest.approx(1.0, abs=1e-3)
            assert f.stack_vector[0] == pytest.approx(f.chip_share, abs=1e-4)
            surv = {s: v for s, v in ctx.start.items() if v > 0}
            ordered = sorted(surv)
            ref = icm([surv[s] for s in ordered], obs.payouts[:len(ordered)])[ordered.index(seat)]
            assert f.icm_points == pytest.approx(ref)
            assert len(f.vector()) == len(NUMERIC_FIELDS)
            if f.dealt == 2 and f.street == 0:
                heads_up += 1
                assert f.pre_index == (0 if f.is_button else 1)
            checked += 1
    assert checked > 50
