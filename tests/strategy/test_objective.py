import pytest

from arena.domain.models import TournamentConfig
from arena.domain.rng import SeededRng
from arena.table.driver import legal_random, play_out, shove
from arena.table.service import seeded_table
from strategy.objective import (
    cap_allocation,
    chip_diagnostics,
    icm_points,
    outcome_from_table,
    rescore,
    rescore_events,
)


def test_icm_points_known_values():
    assert icm_points([100, 100], [60, 40]) == pytest.approx([50, 50])
    # Malmuth-Harville by hand for three stacks.
    a, b, c = 50, 30, 20
    p = {"a": a / 100, "b": b / 100, "c": c / 100}
    second_a = p["b"] * a / (a + c) + p["c"] * a / (a + b)
    assert icm_points([a, b, c], [50, 30, 20])[0] == pytest.approx(50 * p["a"] + 30 * second_a + 20 * (1 - p["a"] - second_a))
    assert sum(icm_points([5, 17, 30, 48], [50, 30, 20, 0])) == pytest.approx(100)


def test_cap_allocation_uses_top_prizes_for_survivors():
    alloc = cap_allocation({2: 300, 5: 100}, [50, 30, 20, 0])
    assert alloc[2] + alloc[5] == pytest.approx(80)
    assert alloc[2] > alloc[5] > 30
    assert cap_allocation({1: 400}, [50, 30, 20]) == {1: 50.0}


def _table(ruleset, n, seed, hand_cap, policy=legal_random):
    config = TournamentConfig(ruleset_id=ruleset, initial_player_count=n, starting_stack=400, hand_cap=hand_cap)
    table = seeded_table(config, seed)
    play_out(table, policy, SeededRng(seed, "players"))
    return table


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4"])
@pytest.mark.parametrize("seed,hand_cap", [(1, 6), (2, 200), (3, 30)])
def test_objective_matches_table_points(ruleset, seed, hand_cap):
    table = _table(ruleset, 6, seed, hand_cap)
    outcome = outcome_from_table(table)
    points = table.points()
    for seat in table.seat_ring:
        assert outcome.points(seat) == pytest.approx(points[seat], abs=1e-9)
        assert outcome.utility(seat) == outcome.points(seat)
    again = rescore_events(table.log)
    assert {s: again.points(s) for s in table.seat_ring} == pytest.approx(points, abs=1e-9)
    if outcome.ended_at_cap:
        assert sum(outcome.cap_icm_points(s) for s in outcome.survivors) > 0
    chips = chip_diagnostics(outcome, table.seat_ring[0])
    assert chips["diagnostic_only"] is True


def test_shove_tournament_ends_with_a_winner():
    table = _table("practice_nlhe", 4, 7, 200, policy=shove)
    outcome = outcome_from_table(table)
    assert not outcome.ended_at_cap
    assert [outcome.finishes[s].method for s in outcome.survivors] == ["winner"]


def test_rescore_rejects_inconsistent_inputs():
    with pytest.raises(AssertionError):
        rescore([50, 30, 20], [0, 1, 2], {0: 100, 1: 100, 2: 100}, [(1, {0: 100, 1: 100, 2: 100},
                                                                       {0: 250, 1: 0, 2: 100})], 10)
