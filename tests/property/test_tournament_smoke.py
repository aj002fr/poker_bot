"""Full headless tournaments per scenario and initial N, repeated from fixed seeds."""

import pytest

from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng
from arena.history.canonical import TableHistory, export_history
from arena.history.replay import replay
from arena.table.driver import check_or_call, legal_random, play_out, shove
from arena.table.service import seeded_table
from arena.tournament.scoring import SCORE_TOLERANCE
from tests.oracles.icm import Bust, table_points
from tests.property.generated import RULESETS, check_invariants
from tests.support.oracle_bridge import assert_matches_oracles


def run_tournament(ruleset: str, n: int, seed: int, policy=legal_random, **config_kw):
    config = TournamentConfig(ruleset_id=ruleset, initial_player_count=n, starting_stack=400, **config_kw)
    table = seeded_table(config, seed)
    hands = []

    def on_decision(t, seat, legal, action):
        if not hands or hands[-1] is not t.hand:
            hands.append(t.hand)
        check_invariants(t.hand)

    play_out(table, policy, SeededRng(seed, "players"), on_decision=on_decision)
    for hand in hands:
        assert hand.is_complete
        assert_matches_oracles(hand)
    report = replay(TableHistory.loads(export_history(table).dumps()))
    assert report.complete and report.final_stacks == table.stacks
    return table


def fingerprint(table):
    return (
        [(h.hand_id, h.positions, h.deck, h.result.final_stacks, [(e.type, e.data) for e in h.events]) for h in table.hands],
        [(p.seat, p.positions, p.points) for p in table.final_placements],
    )


def check_scoring(table, n):
    points = table.points()
    assert sorted(points) == sorted(table.seat_ring)
    assert abs(sum(points.values()) - 100) <= SCORE_TOLERANCE
    assert len(table.hands) <= table.config.hand_cap
    survivors = {p.seat: table.stacks[p.seat] for p in table.final_placements if p.method != "elimination"}
    busts = [
        [Bust(e.seat, h.start_stacks[e.seat]) for e in h.eliminations]
        for h in table.hands
        if h.eliminations
    ]
    oracle = table_points(table.config.resolved_payouts, bust_hands=busts, survivors=survivors, exact=False)
    for seat, value in oracle.items():
        assert abs(points[seat] - float(value)) <= 1e-7


@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("ruleset", RULESETS)
def test_tournament_smoke_and_deterministic_repeat(ruleset, n):
    first = run_tournament(ruleset, n, seed=1000 + n)
    assert first.complete
    check_scoring(first, n)
    again = run_tournament(ruleset, n, seed=1000 + n)
    assert fingerprint(again) == fingerprint(first)


@pytest.mark.parametrize("ruleset", RULESETS)
def test_all_in_every_hand_tournament(ruleset):
    table = run_tournament(ruleset, 8, seed=7, policy=shove)
    assert table.complete
    check_scoring(table, 8)


@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("ruleset", RULESETS)
def test_hand_cap_tournament_ends_with_icm(ruleset, n):
    table = run_tournament(ruleset, n, seed=50 + n, policy=check_or_call, hand_cap=6, hands_per_level=2)
    assert len(table.hands) == 6 or len(table.standings.remaining) == 1
    if len(table.standings.remaining) > 1:
        assert {p.method for p in table.final_placements} == {"icm"}
    check_scoring(table, n)


@pytest.mark.release
@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("ruleset", RULESETS)
def test_tournament_release_smoke(ruleset, n):
    for seed in range(20):
        table = run_tournament(ruleset, n, seed=seed)
        check_scoring(table, n)
