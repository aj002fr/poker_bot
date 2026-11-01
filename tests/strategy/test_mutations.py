"""Deliberately broken solver variants must fail at least one Stage 2 gate.

Each fault is compiled into the production kernels (`strategy.solve.faults`).
Every detector must pass for the correct solver (the control) and fail for
each fault mapped to it; every fault must have a detector.
"""

from collections.abc import Callable
from functools import cache

import pytest

from strategy.games.fixtures import FIXTURES, run_fixture
from strategy.solve import Fault, nash_conv
from strategy.solve.faults import FULL_TRAVERSAL_FAULTS, SAMPLING_FAULTS
from tests.strategy.helpers import average_weighting_error, leduc_tree, reference_mismatch


def fixture_gate(fixture_id: str, method: str = "cfr") -> Callable[[Fault], bool]:
    def passes(fault: Fault) -> bool:
        return run_fixture(fixture_id, method=method, fault=fault).passed

    passes.__name__ = f"gate[{fixture_id},{method}]"
    return passes


def es_convergence(fixture_id: str) -> Callable[[Fault], bool]:
    def passes(fault: Fault) -> bool:
        run = run_fixture(fixture_id, method="es", fault=fault)
        return nash_conv(run.tree, run.policy).max_gain <= FIXTURES[fixture_id].es_tolerance

    passes.__name__ = f"es-convergence[{fixture_id}]"
    return passes


def openspiel_reference(game: str, solver: str) -> Callable[[Fault], bool]:
    def passes(fault: Fault) -> bool:
        return reference_mismatch(game, solver, fault=fault) < 1e-8

    passes.__name__ = f"openspiel-reference[{game},{solver}]"
    return passes


def reach_weighted_average(fault: Fault) -> bool:
    return average_weighting_error(leduc_tree(), "cfr", iterations=10, fault=fault) < 1e-10


DETECTORS: dict[Fault, list[Callable[[Fault], bool]]] = {
    Fault.LAST_ITERATE: [fixture_gate("kuhn"), openspiel_reference("kuhn", "cfr"), es_convergence("kuhn")],
    Fault.REGRET_OWN_REACH: [fixture_gate("kuhn"), fixture_gate("leduc"), openspiel_reference("kuhn", "cfr")],
    Fault.AVERAGE_UNWEIGHTED: [reach_weighted_average, openspiel_reference("leduc", "cfr")],
    Fault.PAYOFF_ORIENTATION: [fixture_gate("kuhn"), fixture_gate("dry-side-pot")],
    Fault.IGNORE_CHANCE_REACH: [fixture_gate("marked-draw"), fixture_gate("two-street-geometric")],
    Fault.RESET_REACH_AT_CHANCE: [fixture_gate("leduc"), fixture_gate("marked-draw")],
    Fault.ES_UNIFORM_OPPONENT: [es_convergence("kuhn"), es_convergence("limit-akq")],
    Fault.ES_UNIFORM_CHANCE: [es_convergence("marked-draw")],
    Fault.ES_NO_EXPLORATION: [es_convergence("kuhn"), es_convergence("clairvoyance")],
}

CASES = [(fault, d) for fault, ds in DETECTORS.items() for d in ds]
ALL_DETECTORS = {d.__name__: d for _, d in CASES}


def test_every_fault_has_a_detector():
    assert set(DETECTORS) == set(Fault) - {Fault.NONE}
    assert set(FULL_TRAVERSAL_FAULTS) | set(SAMPLING_FAULTS) == set(DETECTORS)


@cache
def _control(name: str) -> bool:
    return ALL_DETECTORS[name](Fault.NONE)


@pytest.mark.parametrize("name", sorted(ALL_DETECTORS))
def test_detector_passes_for_the_correct_solver(name):
    assert _control(name)


@pytest.mark.parametrize("fault, detector", CASES, ids=[f"{f.name}-{d.__name__}" for f, d in CASES])
def test_broken_variant_is_caught(fault, detector):
    assert not detector(fault), f"{fault.name} survived {detector.__name__}"
