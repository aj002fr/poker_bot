"""Stage 2 fixture registry: toy games, analytic expectations, declared tolerances.

Each `Fixture` builds a game, solves it, measures named quantities from the
*average* policy, and compares them with the source's expected values. The
same registry drives the tests, the gate table (``python -m
strategy.games.fixtures``) and the benchmark.

Measured threshold quantities are probability masses: e.g. ``y_bluff`` is
the total probability, over Y's hands, of betting with hands on the bluff
side of the check region. For [0,1] games discretised to ``n`` hands the
threshold tolerance is one grid step (1/n) and the value tolerance ~1/n^2.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from strategy.games import akq, clairvoyance, kuhn, leduc, marked_draw, matrix, uniform
from strategy.solve.cfr import CFR_PLUS, CFRConfig, CFRSolver
from strategy.solve.evaluate import best_response, expected_values, nash_conv
from strategy.solve.faults import Fault
from strategy.solve.game import ExtensiveGame, GameTree, compile_game
from strategy.solve.matrix import regret_matching_selfplay
from strategy.solve.mccfr import ExternalSamplingMCCFR
from strategy.solve.policy import TabularPolicy


@dataclass(frozen=True)
class Expect:
    value: float
    tolerance: float
    kind: str = "abs"  # "abs": |m - v| <= tol; "max": m <= v; "min": m >= v

    def passes(self, measured: float) -> bool:
        if self.kind == "abs":
            return abs(measured - self.value) <= self.tolerance
        if self.kind == "max":
            return measured <= self.value
        if self.kind == "min":
            return measured >= self.value
        raise ValueError(self.kind)


@dataclass(frozen=True)
class GateResult:
    fixture: str
    check: str
    measured: float
    expect: Expect

    @property
    def passed(self) -> bool:
        return self.expect.passes(self.measured)


Measure = Callable[[GameTree, TabularPolicy], dict[str, float]]


@dataclass(frozen=True)
class Fixture:
    id: str
    title: str
    source: str
    build: Callable[[int], ExtensiveGame]
    measure: Measure
    expected: Callable[[int], dict[str, Expect]]
    iterations: int
    config: CFRConfig = CFR_PLUS
    es_iterations: int | None = None
    uses_resolution: bool = False
    note: str = ""


@dataclass
class FixtureRun:
    fixture: Fixture
    method: str
    resolution: int
    tree: GameTree
    policy: TabularPolicy
    results: list[GateResult]
    nodes_touched: int
    seconds: float
    measured: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]


FAST_RESOLUTION = 60
FULL_RESOLUTION = 200


@lru_cache(maxsize=64)
def _compiled(fixture_id: str, resolution: int) -> GameTree:
    return compile_game(FIXTURES[fixture_id].build(resolution))


def run_fixture(
    fixture: Fixture | str,
    *,
    method: str = "cfr",
    resolution: int = FAST_RESOLUTION,
    iterations: int | None = None,
    config: CFRConfig | None = None,
    fault: Fault = Fault.NONE,
    seed: int = 0,
) -> FixtureRun:
    fx = FIXTURES[fixture] if isinstance(fixture, str) else fixture
    tree = _compiled(fx.id, resolution) if fx.id in FIXTURES else compile_game(fx.build(resolution))
    start = time.perf_counter()
    if method == "cfr":
        solver = CFRSolver(tree, config or fx.config, fault=fault).run(iterations or fx.iterations)
    elif method in ("es", "es-linear"):
        n_iter = iterations or fx.es_iterations
        if n_iter is None:
            raise ValueError(f"{fx.id} declares no sampling budget")
        solver = ExternalSamplingMCCFR(tree, seed=seed, linear=method == "es-linear", fault=fault).run(n_iter)
    else:
        raise ValueError(method)
    seconds = time.perf_counter() - start
    policy = solver.average_policy()
    measured = fx.measure(tree, policy)
    expected = fx.expected(resolution)
    results = [GateResult(fx.id, name, float(measured[name]), exp) for name, exp in expected.items()]
    return FixtureRun(fx, method, resolution, tree, policy, results, solver.nodes_touched, seconds, measured)


# ---------------------------------------------------------------- helpers


def _nash(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    report = nash_conv(tree, policy)
    return {"nash_conv": report.nash_conv}


def _hand_probs(policy: TabularPolicy, key: Callable[[int], object], action: str, n: int) -> np.ndarray:
    return np.array([policy.prob(key(i), action) for i in range(n)])


def _mass(probs: np.ndarray, lo: float, hi: float) -> float:
    mid = uniform.hand_midpoints(len(probs))
    sel = (mid >= lo) & (mid < hi)
    return float(probs[sel].sum() / len(probs))


def _threshold_tol(n: int) -> float:
    return 1.0 / n


def _value_tol(n: int) -> float:
    return 1.0 / n**2 + 2e-4


def _nash_tol(n: int) -> float:
    return 2e-3


# ---------------------------------------------------------------- fixtures


def _matrix_measure(k: int) -> Measure:
    def measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
        row = policy.infoset_probs(tree.infoset("row"))
        col = policy.infoset_probs(tree.infoset("col"))
        return {
            "max_deviation_from_uniform": float(max(np.abs(row - 1 / k).max(), np.abs(col - 1 / k).max())),
            "value_row": float(expected_values(tree, policy)[0]),
            **_nash(tree, policy),
        }

    return measure


def _matrix_expected(_: int) -> dict[str, Expect]:
    return {
        "max_deviation_from_uniform": Expect(0.0, 1e-9),
        "value_row": Expect(0.0, 1e-12),
        "nash_conv": Expect(1e-9, 0.0, "max"),
    }


RM_ITERATIONS = 20_000


def _rm_measure(payoff: np.ndarray, initial: tuple[np.ndarray, np.ndarray]) -> Measure:
    def measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
        k = payoff.shape[0]
        out = regret_matching_selfplay(payoff, iterations=RM_ITERATIONS, initial_regret=initial)
        avg_dev = max(np.abs(out.average[0] - 1 / k).max(), np.abs(out.average[1] - 1 / k).max())
        last_dev = max(np.abs(out.last[0] - 1 / k).max(), np.abs(out.last[1] - 1 / k).max())
        return {
            "average_deviation_from_uniform": float(avg_dev),
            "average_value_row": out.average_value,
            "average_external_regret": float(out.external_regret[-1].max()),
            "last_iterate_deviation": float(last_dev),
        }

    return measure


def _rm_expected(payoff: np.ndarray) -> Callable[[int], dict[str, Expect]]:
    # Regret matching's external-regret bound: payoff range * sqrt(|A| / T).
    bound = float(payoff.max() - payoff.min()) * np.sqrt(payoff.shape[0] / RM_ITERATIONS)

    def expected(_: int) -> dict[str, Expect]:
        return {
            "average_deviation_from_uniform": Expect(0.01, 0.0, "max"),
            "average_value_row": Expect(0.0, 1e-3),
            "average_external_regret": Expect(bound, 0.0, "max"),
            "last_iterate_deviation": Expect(0.25, 0.0, "min"),
        }

    return expected


def _clairvoyance_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    return {
        "bluff_fraction_of_dead": policy.prob(("Y", "dead"), "bet"),
        "winner_bets": policy.prob(("Y", "winner"), "bet"),
        "x_call": policy.prob(("X", "bet"), "call"),
        "value_y": float(expected_values(tree, policy)[1]),
        **_nash(tree, policy),
    }


def _clairvoyance_expected(pot: float, winner_fraction: float) -> Callable[[int], dict[str, Expect]]:
    sol = clairvoyance.half_street_solution(pot, 1.0, winner_fraction)

    def expected(_: int) -> dict[str, Expect]:
        return {
            "bluff_fraction_of_dead": Expect(sol["bluff_fraction_of_dead"], 0.01),
            "winner_bets": Expect(1.0, 1e-3),
            "x_call": Expect(sol["x_call"], 0.01),
            "value_y": Expect(sol["value_y"], 1e-3),
            "nash_conv": Expect(1e-3, 0.0, "max"),
        }

    return expected


def _half_no_fold_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    n = tree.meta["n"]
    bet = _hand_probs(policy, lambda i: ("Y", i), "bet", n)
    return {"y_bet": float(bet.mean()), "value_y": float(expected_values(tree, policy)[1]), **_nash(tree, policy)}


def _half_no_fold_expected(n: int) -> dict[str, Expect]:
    sol = uniform.half_street_no_fold_solution()
    return {
        "y_bet": Expect(sol["y_bet"], _threshold_tol(n)),
        "value_y": Expect(sol["value_y"], _value_tol(n)),
        "nash_conv": Expect(_nash_tol(n), 0.0, "max"),
    }


def _half_fold_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    n = tree.meta["n"]
    bet = _hand_probs(policy, lambda i: ("Y", i), "bet", n)
    call = _hand_probs(policy, lambda i: ("X", i), "call", n)
    return {
        "y_value_bet": _mass(bet, 0.0, 0.6),
        "y_bluff": _mass(bet, 0.6, 1.0),
        "x_call": float(call.mean()),
        "value_y": float(expected_values(tree, policy)[1]),
        **_nash(tree, policy),
    }


def _half_fold_expected(n: int) -> dict[str, Expect]:
    sol = uniform.half_street_with_fold_solution(2.0)
    t = _threshold_tol(n)
    return {
        "y_value_bet": Expect(sol["y_value_bet"], t),
        "y_bluff": Expect(sol["y_bluff"], t),
        "x_call": Expect(sol["x_call"], t),
        "value_y": Expect(sol["value_y"], _value_tol(n)),
        "nash_conv": Expect(_nash_tol(n), 0.0, "max"),
    }


def _jam_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    n = tree.meta["n"]
    jam = _hand_probs(policy, lambda i: ("Y", i), "jam", n)
    call = _hand_probs(policy, lambda i: ("X", i), "call", n)
    return {
        "y_jam": float(jam.mean()),
        "x_call": float(call.mean()),
        "x_call_below_3_4": _mass(call, 0.0, 0.75),
        "value_y": float(expected_values(tree, policy)[1]),
        **_nash(tree, policy),
    }


def _jam_expected(stack: float, smoothed: bool) -> Callable[[int], dict[str, Expect]]:
    def expected(n: int) -> dict[str, Expect]:
        sol = uniform.jam_fold_smoothed_solution(stack) if smoothed else uniform.jam_fold_solution(stack)
        t = _threshold_tol(n)
        out = {"y_jam": Expect(sol["y_jam"], t)}
        if smoothed and stack == 3.0:
            # X is indifferent with every hand above 3/4: calls [0, 3/4] fully, anything above.
            out["x_call_below_3_4"] = Expect(0.75, t)
            out["x_call"] = Expect(0.75 - t, 0.0, "min")
        else:
            out["x_call"] = Expect(sol["x_call"], t)
        out["value_y"] = Expect(sol["value_y"], _value_tol(n))
        out["nash_conv"] = Expect(_nash_tol(n), 0.0, "max")
        return out

    return expected


def _full_no_fold_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    n = tree.meta["n"]
    x_bet = _hand_probs(policy, lambda i: ("X", i, ()), "bet", n)
    y_raise = _hand_probs(policy, lambda i: ("Y", i, ("bet",)), "raise", n)
    y_bet = _hand_probs(policy, lambda i: ("Y", i, ("check",)), "bet", n)
    out = {
        "x_bet": float(x_bet.mean()),
        "y_raise": float(y_raise.mean()),
        "y_bet": float(y_bet.mean()),
        "value_y": float(expected_values(tree, policy)[1]),
        **_nash(tree, policy),
    }
    if tree.meta["check_raise"]:
        x_cr = (1 - x_bet) * _hand_probs(policy, lambda i: ("X", i, ("check", "bet")), "raise", n)
        out["x_check_raise"] = float(x_cr.mean())
        out["x_check_raise_above_1_3"] = _mass(x_cr, 1 / 3 + 1 / n, 1.0)
        out["x_bet_or_raise"] = float((x_bet + x_cr).mean())
    return out


def _full_no_fold_expected(check_raise: bool) -> Callable[[int], dict[str, Expect]]:
    sol = uniform.full_street_no_fold_solution(check_raise)

    def expected(n: int) -> dict[str, Expect]:
        t = _threshold_tol(n)
        out = {"y_raise": Expect(sol["y_raise"], t), "y_bet": Expect(sol["y_bet"], t)}
        if check_raise:
            out["x_check_raise"] = Expect(sol["x_check_raise"], t)
            out["x_check_raise_above_1_3"] = Expect(t, 0.0, "max")
            out["x_bet_or_raise"] = Expect(sol["x_bet_or_raise"], t)
        else:
            out["x_bet"] = Expect(sol["x_bet"], t)
        out["value_y"] = Expect(sol["value_y"], _value_tol(n))
        out["nash_conv"] = Expect(_nash_tol(n), 0.0, "max")
        return out

    return expected


def _finite_pot_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    n = tree.meta["n"]
    x_bet = _hand_probs(policy, lambda i: ("X", i, ()), "bet", n)
    x_call = (1 - x_bet) * _hand_probs(policy, lambda i: ("X", i, ("check", "bet")), "call", n)
    y_call = _hand_probs(policy, lambda i: ("Y", i, ("bet",)), "call", n)
    y_bet = _hand_probs(policy, lambda i: ("Y", i, ("check",)), "bet", n)
    return {
        "x_value_bet": _mass(x_bet, 0.0, 0.61),
        "x_bluff": _mass(x_bet, 0.61, 1.0),
        "x_check_call": float(x_call.mean()),
        "y_call": float(y_call.mean()),
        "y_value_bet": _mass(y_bet, 0.0, 2 / 3),
        "y_bluff": _mass(y_bet, 2 / 3, 1.0),
        "value_y": float(expected_values(tree, policy)[1]),
        **_nash(tree, policy),
    }


def _finite_pot_expected(n: int) -> dict[str, Expect]:
    sol = uniform.finite_pot_solution(2.0)
    t = _threshold_tol(n)
    return {
        "x_value_bet": Expect(sol["x_value_bet"], t),
        "x_bluff": Expect(sol["x_bluff"], t),
        "x_check_call": Expect(sol["x_call"] - sol["x_value_bet"], t),
        "y_call": Expect(sol["y_call"], t),
        "y_value_bet": Expect(sol["y_value_bet"], t),
        "y_bluff": Expect(sol["y_bluff"], t),
        "value_y": Expect(sol["value_y"], _value_tol(n)),
        "nash_conv": Expect(_nash_tol(n), 0.0, "max"),
    }


def _limit_akq_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    return {
        "y_a_bet": policy.prob(("Y", "A"), "bet1.0"),
        "y_k_bet": policy.prob(("Y", "K"), "bet1.0"),
        "y_q_bluff": policy.prob(("Y", "Q"), "bet1.0"),
        "x_a_call": policy.prob(("X", "A", "bet1.0"), "call"),
        "x_k_call": policy.prob(("X", "K", "bet1.0"), "call"),
        "x_q_call": policy.prob(("X", "Q", "bet1.0"), "call"),
        "value_y": float(expected_values(tree, policy)[1]),
        **_nash(tree, policy),
    }


def _limit_akq_expected(_: int) -> dict[str, Expect]:
    sol = akq.limit_akq_solution(2.0)
    return {
        "y_a_bet": Expect(1.0, 0.01),
        "y_k_bet": Expect(0.0, 0.01),
        "y_q_bluff": Expect(sol["y_q_bluff"], 0.01),
        "x_a_call": Expect(1.0, 0.01),
        "x_k_call": Expect(sol["x_k_call"], 0.01),
        "x_q_call": Expect(0.0, 0.01),
        "value_y": Expect(sol["value_y"], 1e-3),
        "nash_conv": Expect(1e-3, 0.0, "max"),
    }


def _nl_akq_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    labels = tree.action_labels[tree.infoset(("Y", "A"))]
    probs = policy.action_probs(("Y", "A"))
    bets = [(float(a[3:]), probs[a]) for a in labels if a != "check"]
    bet_mass = sum(p for _, p in bets)
    mean_size = sum(s * p for s, p in bets) / bet_mass if bet_mass > 0 else 0.0
    return {
        "a_bet": bet_mass,
        "a_mean_size": mean_size,
        "value_y": float(expected_values(tree, policy)[1]),
        **_nash(tree, policy),
    }


def _nl_akq_expected(_: int) -> dict[str, Expect]:
    sol = akq.no_limit_akq_solution()
    return {
        "a_bet": Expect(1.0, 0.01),
        "a_mean_size": Expect(sol["size"], 0.04),
        "value_y": Expect(sol["value_y"], 1e-4),
        "nash_conv": Expect(1e-3, 0.0, "max"),
    }


def _kuhn_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    return {
        "value_p0": float(expected_values(tree, policy)[0]),
        "p1_j_bluff_after_check": policy.prob((1, "J", "p"), "b"),
        "p1_q_call_bet": policy.prob((1, "Q", "b"), "b"),
        "p1_k_bet_after_check": policy.prob((1, "K", "p"), "b"),
        **_nash(tree, policy),
    }


def _kuhn_expected(_: int) -> dict[str, Expect]:
    return {
        "value_p0": Expect(kuhn.GAME_VALUE_P0, 2e-3),
        "p1_j_bluff_after_check": Expect(1 / 3, 0.03),
        "p1_q_call_bet": Expect(1 / 3, 0.03),
        "p1_k_bet_after_check": Expect(1.0, 0.01),
        "nash_conv": Expect(0.01, 0.0, "max"),
    }


def _leduc_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    return {"value_p0": float(expected_values(tree, policy)[0]), **_nash(tree, policy)}


def _leduc_expected(_: int) -> dict[str, Expect]:
    return {"value_p0": Expect(leduc.GAME_VALUE_P0, 2e-3), "nash_conv": Expect(0.05, 0.0, "max")}


def _two_street_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    first = policy.action_probs(("Y", "winner", ()))
    bet_mass = 1.0 - first["check"]
    dead_first = policy.action_probs(("Y", "dead", ()))
    sol = clairvoyance.two_street_solution(1.0, 4.0, 1.0 / 3.0)
    w = 1.0 / 3.0
    y_bet_mass = w * bet_mass + (1 - w) * (1.0 - dead_first["check"])
    return {
        "winner_first_street_pot_bet": first.get("1", 0.0) / bet_mass if bet_mass > 0 else 0.0,
        "winner_second_street_all_in": policy.prob(("Y", "winner", ("1", "call")), "3"),
        "x_call_first_street": policy.prob(("X", ("1",)), "call"),
        "x_call_second_street": policy.prob(("X", ("1", "call", "3")), "call"),
        "y_bet_mass": y_bet_mass,
        "value_y": float(expected_values(tree, policy)[1]),
        "geometric_growth": sol["growth"],
        **_nash(tree, policy),
    }


def _two_street_expected(_: int) -> dict[str, Expect]:
    sol = clairvoyance.two_street_solution(1.0, 4.0, 1.0 / 3.0)
    return {
        "winner_first_street_pot_bet": Expect(1.0, 0.03),
        "winner_second_street_all_in": Expect(1.0, 0.03),
        "x_call_first_street": Expect(sol["x_call_each_street"], 0.03),
        "x_call_second_street": Expect(sol["x_call_each_street"], 0.03),
        "y_bet_mass": Expect(sol["y_bet_mass"], 0.02),
        "value_y": Expect(sol["value_y"], 1e-3),
        "geometric_growth": Expect(3.0, 1e-12),
        "nash_conv": Expect(1e-3, 0.0, "max"),
    }


def _marked_draw_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    game = marked_draw.MarkedDraw()
    return {
        "y_draw_call": policy.prob(("Y", "draw", "turn"), "call"),
        "y_dead_call_mass": (1 - game.draw_fraction) * policy.prob(("Y", "dead", "turn"), "call"),
        "y_flush_bet": policy.prob(("Y", "draw", "flush"), "bet"),
        "y_dead_bluff_flush": policy.prob(("Y", "dead", "flush"), "bet"),
        "x_call_flush_river": policy.prob(("X", "flush"), "call"),
        "value_y": float(expected_values(tree, policy)[1]),
        **_nash(tree, policy),
    }


def _marked_draw_expected(_: int) -> dict[str, Expect]:
    sol = marked_draw.marked_draw_solution()
    return {
        "y_draw_call": Expect(1.0, 0.01),
        "y_dead_call_mass": Expect(sol["y_dead_call_mass"], 0.002),
        "y_flush_bet": Expect(1.0, 0.01),
        "y_dead_bluff_flush": Expect(1.0, 0.02),
        "x_call_flush_river": Expect(sol["x_call_flush_river"], 0.02),
        "value_y": Expect(sol["value_y"], 1e-3),
        "nash_conv": Expect(1e-3, 0.0, "max"),
    }


def _isolated_river_expected(_: int) -> dict[str, Expect]:
    sol = marked_draw.marked_draw_solution()
    base = _clairvoyance_expected(6.0, 0.1)(0)
    base["x_call"] = Expect(sol["x_call_isolated_river"], 0.01)
    return base


def _dry_measure(tree: GameTree, policy: TabularPolicy) -> dict[str, float]:
    n = tree.meta["n"]
    bet = _hand_probs(policy, lambda i: ("Y", i), "bet", n)
    call = _hand_probs(policy, lambda i: ("X", i), "call", n)
    mid = uniform.hand_midpoints(n)
    upper = mid >= 0.5
    bluff_w = bet[upper] / n
    report = nash_conv(tree, policy)
    printed = _threshold_profile(tree, n, x_call=0.2848, y_value=0.1424, band=(0.6424 - 0.0095, 0.6424 + 0.0095))
    printed_gain_y = best_response(tree, printed, 1)[0] - expected_values(tree, printed)[1]
    over = _overbluff(tree, policy, n, extra_width=0.03)
    return {
        "x_call": float(call.mean()),
        "y_value_bet": _mass(bet, 0.0, 0.5),
        "bluff_width": float(bluff_w.sum()),
        "bluff_center": float((bluff_w * mid[upper]).sum() / max(bluff_w.sum(), 1e-12)),
        "max_deviation_gain": report.max_gain,
        "utility_conservation_error": float(np.abs(tree.terminal_utility.sum(axis=1)).max()),
        "printed_profile_y_gain": float(printed_gain_y),
        "z_gain_from_y_overbluff": float(expected_values(tree, over)[2] - report.values[2]),
    }


def _threshold_profile(tree: GameTree, n: int, *, x_call: float, y_value: float, band: tuple[float, float]) -> TabularPolicy:
    probs = TabularPolicy.uniform(tree).probs.copy()
    mid = uniform.hand_midpoints(n)
    for i in range(n):
        y_bets = mid[i] < y_value or band[0] <= mid[i] < band[1]
        probs[tree.slots(tree.infoset(("Y", i)))] = (0.0, 1.0) if y_bets else (1.0, 0.0)
        x_calls = mid[i] < x_call
        probs[tree.slots(tree.infoset(("X", i)))] = (0.0, 1.0) if x_calls else (1.0, 0.0)
    return TabularPolicy(tree, probs)


def _overbluff(tree: GameTree, policy: TabularPolicy, n: int, *, extra_width: float) -> TabularPolicy:
    """Y also bets hands just above its equilibrium bluff band; X unchanged."""
    probs = policy.probs.copy()
    mid = uniform.hand_midpoints(n)
    bet = _hand_probs(policy, lambda i: ("Y", i), "bet", n)
    top = max((mid[i] for i in range(n) if mid[i] >= 0.5 and bet[i] > 0.5), default=0.75)
    for i in range(n):
        if top < mid[i] <= top + extra_width:
            probs[tree.slots(tree.infoset(("Y", i)))] = (0.0, 1.0)
    return TabularPolicy(tree, probs)


def _dry_expected(n: int) -> dict[str, Expect]:
    sol = uniform.dry_side_pot_solution(9.0)
    t = _threshold_tol(n)
    return {
        "x_call": Expect(sol["x_call"], t),
        "y_value_bet": Expect(sol["y_value_bet"], t),
        "bluff_width": Expect(sol["bluff_width"], t),
        "bluff_center": Expect(sol["bluff_center"], 2 * t),
        "max_deviation_gain": Expect(1e-3, 0.0, "max"),
        "utility_conservation_error": Expect(1e-12, 0.0, "max"),
        "printed_profile_y_gain": Expect(0.05, 0.0, "min"),
        "z_gain_from_y_overbluff": Expect(1e-6, 0.0, "min"),
    }


_RM_START = {
    "pennies": (np.array([2.0, 0.0]), np.array([0.0, 1.0])),
    "rps": (np.array([3.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])),
}

_FIXTURE_LIST: list[Fixture] = [
    Fixture(
        "matching-pennies", "Matching pennies (CFR on the extensive form)", "MoP Ch. 10",
        lambda n: matrix.matching_pennies(), _matrix_measure(2), _matrix_expected, iterations=200, es_iterations=2000,
    ),
    Fixture(
        "rock-paper-scissors", "Rock-paper-scissors (CFR on the extensive form)", "MoP Ch. 10",
        lambda n: matrix.rock_paper_scissors(), _matrix_measure(3), _matrix_expected, iterations=200, es_iterations=2000,
    ),
    Fixture(
        "rm-matching-pennies", "Regret matching self-play, matching pennies (biased start)", "Neller & Lanctot",
        lambda n: matrix.matching_pennies(), _rm_measure(matrix.MATCHING_PENNIES, _RM_START["pennies"]),
        _rm_expected(matrix.MATCHING_PENNIES),
        iterations=1,
    ),
    Fixture(
        "rm-rock-paper-scissors", "Regret matching self-play, RPS (biased start)", "Neller & Lanctot",
        lambda n: matrix.rock_paper_scissors(), _rm_measure(matrix.ROCK_PAPER_SCISSORS, _RM_START["rps"]),
        _rm_expected(matrix.ROCK_PAPER_SCISSORS),
        iterations=1,
    ),
    Fixture(
        "clairvoyance", "Clairvoyance half-street, P = 2", "MoP Ex. 11.1",
        lambda n: clairvoyance.ClairvoyantHalfStreet(2.0, 1.0, 0.5), _clairvoyance_measure,
        _clairvoyance_expected(2.0, 0.5), iterations=5000, es_iterations=40_000,
    ),
    Fixture(
        "half-street-no-fold", "[0,1] half-street, no folding", "MoP Ex. 11.2",
        lambda n: uniform.HalfStreetNoFold(n=n), _half_no_fold_measure, _half_no_fold_expected,
        iterations=200, uses_resolution=True, es_iterations=20_000,
    ),
    Fixture(
        "half-street-fold", "[0,1] half-street with folding, P = 2", "MoP Ex. 11.3",
        lambda n: uniform.HalfStreetWithFold(n=n, pot=2.0), _half_fold_measure, _half_fold_expected,
        iterations=2000, uses_resolution=True, es_iterations=200_000,
    ),
    Fixture(
        "jam-fold", "[0,1] jam-or-fold, deterministic, S = 2", "MoP Ex. 12.1",
        lambda n: uniform.JamFold(n=n, stack=2.0), _jam_measure, _jam_expected(2.0, False),
        iterations=2000, uses_resolution=True, es_iterations=200_000,
    ),
    Fixture(
        "jam-fold-smoothed", "[0,1] jam-or-fold, stronger wins 2/3, S = 3", "MoP Ex. 12.2",
        lambda n: uniform.JamFold(n=n, stack=3.0, stronger_wins=2 / 3), _jam_measure,
        _jam_expected(3.0, True), iterations=2000, uses_resolution=True, es_iterations=200_000,
        note="At S = 3 X's call threshold is not unique: every hand above 3/4 is indifferent.",
    ),
    Fixture(
        "jam-fold-smoothed-s4", "[0,1] jam-or-fold, stronger wins 2/3, S = 4 (unique regime)", "MoP Ex. 12.2",
        lambda n: uniform.JamFold(n=n, stack=4.0, stronger_wins=2 / 3), _jam_measure,
        _jam_expected(4.0, True), iterations=2000, uses_resolution=True, es_iterations=200_000,
    ),
    Fixture(
        "limit-akq", "Limit AKQ, pot 2, unit bet", "MoP Ex. 13.1",
        lambda n: akq.LimitAKQ(pot=2.0), _limit_akq_measure, _limit_akq_expected,
        iterations=5000, es_iterations=40_000,
    ),
    Fixture(
        "no-limit-akq", "No-limit half-street AKQ, sizes 0.05..1.5", "MoP Ex. 14.1",
        lambda n: akq.NoLimitAKQ(), _nl_akq_measure, _nl_akq_expected, iterations=5000, es_iterations=100_000,
    ),
    Fixture(
        "no-limit-akq-exact", "No-limit AKQ with sqrt(2) - 1 on the menu", "MoP Ex. 14.1",
        lambda n: akq.NoLimitAKQ(bet_sizes=(0.25, 0.35, akq.SQRT2_MINUS_1, 0.5, 0.75, 1.0)), _nl_akq_measure,
        _nl_akq_expected, iterations=5000, es_iterations=100_000,
    ),
    Fixture(
        "kuhn", "Kuhn poker", "Kuhn 1950; OpenSpiel kuhn_poker",
        lambda n: kuhn.KuhnPoker(), _kuhn_measure, _kuhn_expected, iterations=1000, es_iterations=20_000,
    ),
    Fixture(
        "leduc", "Leduc poker (OpenSpiel default config)", "OpenSpiel leduc_poker",
        lambda n: leduc.LeducPoker(), _leduc_measure, _leduc_expected, iterations=500, es_iterations=400_000,
    ),
    Fixture(
        "full-street-no-fold", "[0,1] full street, two bets, no check-raise, no fold", "MoP Ex. 16.2",
        lambda n: uniform.FullStreetNoFold(n=n, check_raise=False),
        _full_no_fold_measure, _full_no_fold_expected(False), iterations=2000, uses_resolution=True,
    ),
    Fixture(
        "full-street-check-raise", "[0,1] full street, two bets, check-raise, no fold", "MoP Ex. 16.4",
        lambda n: uniform.FullStreetNoFold(n=n, check_raise=True),
        _full_no_fold_measure, _full_no_fold_expected(True), iterations=2000, uses_resolution=True,
        note="X's check-raise region is co-optimal: any 1/9 of hands better than y2 = 1/3.",
    ),
    Fixture(
        "finite-pot-full-street", "[0,1] full street with folding, P = 2", "MoP Ex. 17.1 (Game #9)",
        lambda n: uniform.FullStreetFinitePot(n=n, pot=2.0), _finite_pot_measure, _finite_pot_expected,
        iterations=3000, uses_resolution=True,
    ),
    Fixture(
        "two-street-geometric", "Two-street static clairvoyance, pot 1, stack 4, Y sizes", "MoP Ex. 19.2",
        lambda n: clairvoyance.TwoStreetClairvoyant(), _two_street_measure, _two_street_expected,
        iterations=5000, es_iterations=100_000,
    ),
    Fixture(
        "marked-draw", "Marked-draw turn and river", "MoP Ex. 20.2",
        lambda n: marked_draw.MarkedDraw(), _marked_draw_measure, _marked_draw_expected,
        iterations=5000, es_iterations=100_000,
    ),
    Fixture(
        "isolated-flush-river", "Flush river solved in isolation (contrast for marked draw)", "MoP Ex. 20.1",
        lambda n: clairvoyance.ClairvoyantHalfStreet(pot=6.0, bet=1.0, winner_fraction=0.1), _clairvoyance_measure,
        _isolated_river_expected, iterations=5000, es_iterations=40_000,
    ),
    Fixture(
        "dry-side-pot", "Three-player dry side pot, Z all-in, P = 9", "MoP Ex. 29.2 (Game #2)",
        lambda n: uniform.DrySidePot(n=n, pot=9.0), _dry_measure, _dry_expected,
        iterations=2000, uses_resolution=True,
        note=(
            "The printed solution (x ~ 0.2848, y1 ~ 0.1424, band 0.6424 +/- 0.0095) violates the book's own"
            " bluff-endpoint equation; Y gains by bluffing more against it. Expected values here solve the"
            " book's three indifference equations."
        ),
    ),
]

FIXTURES: dict[str, Fixture] = {f.id: f for f in _FIXTURE_LIST}


def gate_table(results: list[GateResult]) -> str:
    lines = ["| Fixture | Check | Measured | Expected | Tolerance | Pass |", "|---|---|---:|---:|---:|:---:|"]
    for r in results:
        e = r.expect
        tol = f"±{e.tolerance:.3g}" if e.kind == "abs" else ("≤" if e.kind == "max" else "≥")
        lines.append(
            f"| {r.fixture} | {r.check} | {r.measured:.6g} | {e.value:.6g} | {tol} | {'yes' if r.passed else 'NO'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Stage 2 fixture gates and print a Markdown table.")
    parser.add_argument("--resolution", type=int, default=FAST_RESOLUTION)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    all_results: list[GateResult] = []
    for fx in _FIXTURE_LIST:
        if args.only and fx.id not in args.only:
            continue
        run = run_fixture(fx, resolution=args.resolution)
        all_results.extend(run.results)
    print(gate_table(all_results))
    return 0 if all(r.passed for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
