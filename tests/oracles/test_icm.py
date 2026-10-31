import random
from fractions import Fraction
from functools import lru_cache
from math import factorial

import pytest

from tests.oracles.icm import Bust, icm_permutations, table_points

TOL = 1e-9


def subset_recursion(stacks, prizes):
    """Different algorithm from the oracle: memoized over the set of unplaced players."""
    players = tuple(stacks)

    @lru_cache(maxsize=None)
    def ev(remaining: frozenset, position: int) -> tuple:
        if not remaining:
            return tuple(0.0 for _ in players)
        total = sum(stacks[p] for p in remaining)
        acc = [0.0] * len(players)
        for p in remaining:
            w = stacks[p] / total
            sub = ev(remaining - {p}, position + 1)
            for i, q in enumerate(players):
                acc[i] += w * (sub[i] + (prizes[position] if q == p else 0.0))
        return tuple(acc)

    return dict(zip(players, ev(frozenset(players), 0)))


def test_t04_heads_up():
    r = icm_permutations({"a": 75, "b": 25}, [70, 30])
    assert r.points == {"a": 60, "b": 40}
    assert r.orders == 2


def test_t05_three_equal_stacks():
    r = icm_permutations({"a": 500, "b": 500, "c": 500}, [50, 30, 20])
    assert all(v == Fraction(100, 3) for v in r.points.values())
    assert all(abs(float(v) - 100 / 3) <= TOL for v in r.points.values())


def test_t06_exact_fractions():
    r = icm_permutations({"a": 50, "b": 30, "c": 20}, [50, 30, 20])
    assert r.points == {"a": Fraction(1075, 28), "b": Fraction(131, 4), "c": Fraction(202, 7)}
    approx = {k: float(v) for k, v in r.points.items()}
    assert abs(approx["a"] - 38.392857143) < 1e-8
    assert abs(approx["b"] - 32.75) < TOL
    assert abs(approx["c"] - 28.857142857) < 1e-8
    floats = icm_permutations({"a": 50, "b": 30, "c": 20}, [50, 30, 20], exact=False).points
    assert all(abs(floats[k] - float(r.points[k])) <= TOL for k in floats)


@pytest.mark.parametrize("seed", range(5))
def test_t07_winner_take_all_is_chip_share(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    stacks = {i: rng.randint(1, 4000) for i in range(n)}
    total = sum(stacks.values())
    r = icm_permutations(stacks, [100] + [0] * (n - 1))
    assert r.points == {i: Fraction(100 * s, total) for i, s in stacks.items()}


def test_t08_flat_payout_ignores_stacks():
    r = icm_permutations({0: 1, 1: 10, 2: 1000, 3: 7}, [25, 25, 25, 25])
    assert all(v == 25 for v in r.points.values())


def test_t09_six_seat_three_eliminated_uses_top_three_prizes():
    payout = [50, 30, 20, 0, 0, 0]
    busts = [[Bust("p6", 2000)], [Bust("p5", 1500)], [Bust("p4", 900)]]
    points = table_points(payout, busts, {"a": 50, "b": 30, "c": 20})
    assert points["a"] == Fraction(1075, 28)
    assert points["b"] == Fraction(131, 4)
    assert points["c"] == Fraction(202, 7)
    assert points["p4"] == points["p5"] == points["p6"] == 0
    assert sum(points.values()) == 100


def test_eight_players_enumerate_all_40320_orders():
    stacks = {i: s for i, s in enumerate([2000, 1500, 3100, 700, 4200, 1, 2500, 2000])}
    r = icm_permutations(stacks, [50, 30, 20, 0, 0, 0, 0, 0])
    assert r.orders == factorial(8) == 40_320
    assert sum(r.points.values()) == 100
    assert r.points[1] < r.points[0] == r.points[7] < r.points[6]


def test_one_survivor_takes_first_prize():
    r = icm_permutations({"a": 16000}, [50, 30, 20])
    assert r.points == {"a": 50} and r.orders == 1


def test_t03_winner_before_cap_gets_exactly_first_place():
    busts = [[Bust("d", 100)], [Bust("c", 400)], [Bust("b", 3000)]]
    points = table_points([50, 30, 20, 0], busts, {"a": 8000})
    assert points == {"a": 50, "b": 30, "c": 20, "d": 0}


def test_t10_same_hand_busts_rank_by_start_stack():
    busts = [[Bust("small", 300), Bust("big", 900)]]
    points = table_points([50, 30, 20, 0], busts, {"a": 1000, "b": 1000})
    assert points["big"] == 20 and points["small"] == 0
    assert points["a"] == points["b"] == 40


def test_t11_equal_start_stacks_share_occupied_prizes():
    busts = [[Bust("d", 100)], [Bust("x", 500), Bust("y", 500)]]
    points = table_points([50, 30, 20, 0], busts, {"a": 4000})
    assert points["x"] == points["y"] == 25
    assert points["a"] == 50 and points["d"] == 0
    assert sum(points.values()) == 100


def test_hand_cap_busts_rank_below_every_survivor():
    # Bust on hand 200 with the biggest start stack still finishes below the survivors.
    busts = [[Bust("e", 50)], [Bust("d", 9000)]]
    points = table_points([50, 30, 20, 0, 0], busts, {"a": 10, "b": 10, "c": 10})
    assert points["d"] == 0 and points["e"] == 0
    assert all(points[p] == Fraction(100, 3) for p in "abc")


@pytest.mark.parametrize(
    "stacks, prizes",
    [
        ({"a": 0, "b": 10}, [70, 30]),
        ({"a": -5, "b": 10}, [70, 30]),
        ({"a": True, "b": 10}, [70, 30]),
        ({"a": 1.5, "b": 10}, [70, 30]),
        ({"a": 10, "b": 10}, [100]),
        ({}, [100]),
        ({i: 1 for i in range(9)}, [100] * 9),
    ],
)
def test_rejects_invalid_inputs(stacks, prizes):
    with pytest.raises(ValueError):
        icm_permutations(stacks, prizes)


def test_table_points_rejects_inconsistent_inputs():
    with pytest.raises(ValueError):
        table_points([50, 30, 20], [[Bust("a", 10)]], {"a": 5, "b": 5})
    with pytest.raises(ValueError):
        table_points([50, 30, 20], [], {"a": 5, "b": 5})
    with pytest.raises(ValueError):
        table_points([50, 30], [[Bust("a", 10), Bust("b", 10)]], {})


def random_case(rng, n):
    shape = rng.choice(["uniform", "skewed", "tiny", "equal"])
    if shape == "uniform":
        stacks = [rng.randint(1, 16000) for _ in range(n)]
    elif shape == "skewed":
        stacks = [rng.choice([1, 2, rng.randint(1, 100), rng.randint(5000, 16000)]) for _ in range(n)]
    elif shape == "tiny":
        stacks = [rng.randint(1, 3) for _ in range(n)]
    else:
        stacks = [2000] * n
    prize_shape = rng.choice(["top3", "wta", "flat", "sparse", "random"])
    if prize_shape == "top3":
        prizes = ([50, 30, 20] + [0] * 8)[:n]
    elif prize_shape == "wta":
        prizes = [100] + [0] * (n - 1)
    elif prize_shape == "flat":
        prizes = [Fraction(100, n)] * n
    elif prize_shape == "sparse":
        paid = rng.randint(1, n - 1)
        prizes = [Fraction(100, paid)] * paid + [0] * (n - paid)
    else:
        cuts = sorted(rng.randint(0, 100) for _ in range(n - 1))
        prizes = sorted((b - a for a, b in zip([0, *cuts], [*cuts, 100])), reverse=True)
    return dict(enumerate(stacks)), prizes


def check_case(stacks, prizes):
    oracle = icm_permutations(stacks, prizes, exact=False).points
    other = subset_recursion(stacks, [float(p) for p in prizes])
    assert all(abs(oracle[k] - other[k]) <= TOL for k in stacks)
    assert abs(sum(oracle.values()) - float(sum(prizes))) <= TOL
    by_stack = sorted(stacks, key=stacks.get)
    for lo, hi in zip(by_stack, by_stack[1:]):
        assert oracle[lo] <= oracle[hi] + TOL
        if stacks[lo] == stacks[hi]:
            assert abs(oracle[lo] - oracle[hi]) <= TOL


def test_randomized_against_subset_recursion():
    rng = random.Random(44)
    for _ in range(150):
        check_case(*random_case(rng, rng.randint(2, 6)))


@pytest.mark.release
def test_randomized_1000_cases_n2_to_n8_against_subset_recursion():
    rng = random.Random(4004)
    for i in range(1000):
        check_case(*random_case(rng, 2 + i % 7))
