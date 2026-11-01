import json

import pytest

from strategy.beliefs import (ActionPosterior, FamilyPosterior, FamilyTable, Observation, RangeBeliefs, from_record,
                              summarize, threshold_likelihood)
from strategy.beliefs.posterior import (DEFAULT_PRIOR, POST_FACING_BET, POST_UNBET, PRE_FACING_RAISE, PRE_UNOPENED,
                                        category)
from strategy.beliefs.ranges import K
from strategy.features.cards import capabilities, card_ints
from strategy.features.equity import EquityTables
from strategy.features.state import HandContext


def ob(seat: int, opp: int, cat: int, cell: int = 0, hand: int = 1) -> Observation:
    return Observation(seat, hand, opp, cell, cat)


def test_prediction_without_data_is_the_floored_prior() -> None:
    post = ActionPosterior(floor=0.02)
    p = post.predict(3, PRE_UNOPENED, 0)
    expected = [x * (1 - 3 * 0.02) + 0.02 for x in DEFAULT_PRIOR[PRE_UNOPENED]]
    assert p.probs == pytest.approx(expected) and p.ess_cell == 0


def test_posterior_shrinks_hierarchically_and_floors() -> None:
    post = ActionPosterior(kappa=(10, 8, 5), floor=0.02)
    for _ in range(200):
        post.observe(ob(1, PRE_FACING_RAISE, 1, cell=7))  # always calls in cell 7
    p = post.predict(1, PRE_FACING_RAISE, 7)
    assert p.probs[1] > 0.9 and min(p.probs) >= 0.02 and sum(p.probs) == pytest.approx(1.0)
    assert p.ess_cell == 200 and p.sd[1] < 0.05
    other_cell = post.predict(1, PRE_FACING_RAISE, 8).probs[1]
    assert 0.8 < other_cell < p.probs[1]  # the seat level carries over, the cell refines
    other_seat = post.predict(2, PRE_FACING_RAISE, 7).probs[1]
    assert DEFAULT_PRIOR[PRE_FACING_RAISE][1] < other_seat < other_cell  # only the table pool moves
    assert post.predict(1, POST_FACING_BET, 7).probs == post.predict(9, POST_FACING_BET, 7).probs  # eligibility


def test_decay_and_pool_exclusion() -> None:
    post = ActionPosterior(decay=0.9, exclude_from_pool=(0,))
    for _ in range(50):
        post.observe(ob(0, POST_UNBET, 2))
    assert POST_UNBET not in post.pool
    post.observe(ob(1, POST_UNBET, 0, hand=1))
    before = post.predict(1, POST_UNBET, 0).ess_cell
    post.advance(11)
    assert post.predict(1, POST_UNBET, 0).ess_cell == pytest.approx(before * 0.9 ** 10)


def test_posterior_state_round_trip_and_prune() -> None:
    post = ActionPosterior()
    for i in range(40):
        post.observe(ob(i % 4, i % 5, i % 3, cell=i, hand=i))
    state = json.loads(json.dumps(post.to_state()))
    back = ActionPosterior().load_state(state)
    for s in range(4):
        for o in range(5):
            assert back.predict(s, o, 3).probs == pytest.approx(post.predict(s, o, 3).probs, abs=1e-3)
    assert post.prune(10) == 30 and len(post.cell) == 10


def test_bet_size_categories() -> None:
    assert category(POST_UNBET, "check", 0.0, 0, 100) == 0
    assert category(POST_UNBET, "raise_to", 0.0, 50, 100) == 1
    assert category(POST_UNBET, "raise_to", 0.0, 75, 100) == 2
    assert category(PRE_FACING_RAISE, "raise_to", 0.0, 300, 100) == 2


def test_automatic_actions_are_not_observations() -> None:
    from tests.strategy.feature_support import Actions, hand_events

    events = hand_events({0: 1000, 1: 1000, 2: 1000}, button=0, sb_seat=1, bb_seat=2)
    Actions(events, {1: 10, 2: 20}).act(0, "fold", auto=True).act(1, "call")
    ctx = HandContext([0, 1, 2])
    ctx.feed(events)
    recs = ctx.drain()
    assert from_record(recs[0]) is None and from_record(recs[1]) is not None


def test_threshold_likelihood_shapes() -> None:
    w = [0.1] * K
    raise_ = threshold_likelihood(w, (0.5, 0.3, 0.2), 2, street=0)
    fold = threshold_likelihood(w, (0.5, 0.3, 0.2), 0, street=0)
    assert raise_[-1] > raise_[0] and fold[0] > fold[-1]
    assert all(x >= 0.02 for x in raise_ + fold)
    bet = threshold_likelihood(w, (0.6, 0.25, 0.15), 2, street=1, passive_first=False)
    check = threshold_likelihood(w, (0.6, 0.25, 0.15), 0, street=1, passive_first=False)
    assert bet[-1] > bet[4] and check[0] > check[-1]


def test_range_updates_and_card_removal() -> None:
    eq = EquityTables(capabilities("practice_nlhe"))
    rb = RangeBeliefs(eq.bucket_combos, eq.bucket_card)
    prior_top = rb.prior[-1]
    rb.new_hand([0, 1, 2], hero=0, hero_cards=card_ints(["Ah", "As"]))
    assert rb.weights[0] == pytest.approx(rb.prior)  # the hero's own cards do not remove its own combos
    assert rb.weights[1][-1] < prior_top  # blocking aces removes premium combos from opponents
    rb.update(1, threshold_likelihood(rb.weights[1], (0.6, 0.25, 0.15), 2, street=0))
    s1, s2 = rb.summary(1), rb.summary(2)
    assert s1.mean > s2.mean and s1.top > s2.top and sum(s1.weights) == pytest.approx(1.0)
    before = rb.weights[2][-1]
    rb.add_board(card_ints(["Ad", "Kd", "Kc"]))
    assert rb.weights[2][-1] < before and rb.weights[0][-1] < rb.prior[-1]
    rb.fold(2)
    assert rb.summary(2) is None
    back = RangeBeliefs(eq.bucket_combos, eq.bucket_card).load_state(json.loads(json.dumps(rb.to_state())))
    assert back.weights[1] == pytest.approx(rb.weights[1], abs=1e-5)


def test_summary_of_an_uninformed_range() -> None:
    s = summarize([1.0] * K)
    assert s.entropy == pytest.approx(1.0) and s.mean == pytest.approx(0.5) and not s.capped


def test_family_posterior_is_disabled_without_a_table() -> None:
    fam = FamilyPosterior(None)
    fam.observe(ob(1, PRE_UNOPENED, 2), 50.0)
    assert not fam.enabled and fam.probs(1) == [1.0] and fam.mixture(1, PRE_UNOPENED, 50.0) == DEFAULT_PRIOR[0]


def test_family_posterior_identifies_synthetic_behaviour() -> None:
    table = FamilyTable.load("practice_nlhe")
    fam = FamilyPosterior(table)
    for i in range(60):  # never folds, never raises: a calling station
        fam.observe(ob(4, PRE_FACING_RAISE if i % 2 else POST_FACING_BET, 1), 80.0)
        fam.observe(ob(4, POST_UNBET, 0), 80.0)
    s = fam.summary(4)
    assert s.top in ("calling_station", "station") and s.top_prob > 0.5 and s.observations == 120
    mix = fam.mixture(4, POST_FACING_BET, 80.0)
    assert mix[1] > 0.6 and sum(mix) == pytest.approx(1.0, abs=1e-3)
    back = FamilyPosterior(table).load_state(json.loads(json.dumps(fam.to_state())))
    assert back.probs(4) == pytest.approx(fam.probs(4), abs=1e-3)


def _play(ruleset_id: str, lineup: list[str], hand_cap: int) -> tuple[list[int], FamilyPosterior]:
    from arena.agents.baselines import BASELINES
    from arena.agents.sdk import TableInfo, load_bot
    from arena.agents.styles import STYLES
    from arena.agents.worker import InProcessSeat, play_table
    from arena.benchmark.runner import policy_seed
    from arena.domain.models import TournamentConfig
    from arena.table.projections import Viewer, project_events
    from arena.table.service import seeded_table

    config = TournamentConfig(ruleset_id=ruleset_id, initial_player_count=len(lineup), hand_cap=hand_cap)
    seats = config.resolved_seats
    agents = {}
    for s, name in zip(seats, lineup, strict=True):
        bot, info = load_bot(STYLES.get(name) or BASELINES[name], module_name=f"test_family_{name}")
        ti = TableInfo(table_id="t_fam_test", hero_seat=s, ruleset_id=ruleset_id, config=config.resolved(),
                       config_hash=config.config_hash(), seats=seats, decision_budget_ns=config.bot_budget_ns,
                       policy_seed=policy_seed("family-test", s))
        agents[s] = InProcessSeat(s, bot, info, ti, "family-test")
        agents[s].start()
    table = seeded_table(config, "family-test-holdout", table_id="t_fam_test")
    play_table(table, agents)
    ctx = HandContext(seats)
    ctx.feed(project_events(table.log, Viewer.spectator()))
    fam = FamilyPosterior(FamilyTable.load(ruleset_id))
    for rec in ctx.drain():
        fam.observe(from_record(rec), rec.eff_bb)
    return list(seats), fam


@pytest.mark.parametrize(("ruleset_id", "lineup"), [
    ("practice_nlhe", ["rock", "weak_tight", "check_fold", "nlhe_heuristic", "super_aggressive", "station",
                       "switcher", "aggressive"]),
    ("uncertainty_plo4", ["calling_station", "check_fold", "legal_random", "calling_station", "check_fold",
                          "legal_random", "check_fold", "legal_random"]),
])
def test_family_posterior_identifies_arena_bots_from_public_events(ruleset_id: str, lineup: list[str]) -> None:
    seats, fam = _play(ruleset_id, lineup, 150)
    judged = exact = 0
    for s, name in zip(seats, lineup, strict=True):
        summary = fam.summary(s)
        if name == "switcher" or summary.observations < 60:
            continue
        judged += 1
        exact += summary.top == name
        # the switcher's pooled row overlaps the styles it moves between, so allow it as runner-up
        assert name in (summary.probs[0][0], summary.probs[1][0]), (name, summary.probs[:3])
    assert judged >= 4 and exact >= 4
