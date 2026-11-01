"""Small games and shared checks for the strategy tests."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from strategy.games import kuhn, leduc
from strategy.solve import (
    PRESETS,
    CFRSolver,
    Chance,
    Decision,
    Fault,
    GameTree,
    TabularPolicy,
    Terminal,
    compile_game,
    counterfactual_regrets,
    expected_values,
    infoset_reach,
    nash_conv,
    reach_probabilities,
    sampled_regret_estimate,
)

REFERENCE = Path(__file__).parent / "data" / "openspiel_reference.json"


@dataclass(frozen=True)
class SpecGame:
    """A game written as a nested spec.

    ``("T", utilities)``, ``("C", ((label, prob, child), ...))`` or
    ``("D", player, infoset_key, ((label, child), ...))``.
    """

    spec: tuple
    num_players: int = 2
    name: str = "spec"

    @property
    def player_names(self) -> tuple[str, ...]:
        return tuple(f"P{i}" for i in range(self.num_players))

    def root(self) -> tuple:
        return self.spec

    def expand(self, node: tuple):
        if node[0] == "T":
            return Terminal(tuple(node[1]))
        if node[0] == "C":
            return Chance(tuple(node[1]))
        return Decision(node[1], node[2], tuple(node[3]))


@lru_cache(maxsize=None)
def kuhn_tree() -> GameTree:
    return compile_game(kuhn.KuhnPoker())


@lru_cache(maxsize=None)
def leduc_tree() -> GameTree:
    return compile_game(leduc.LeducPoker())


def random_policy(tree: GameTree, seed: int, floor: float = 0.05) -> TabularPolicy:
    rng = np.random.default_rng(seed)
    w = rng.random(tree.num_slots) + floor
    sums = np.add.reduceat(w, tree.infoset_offset[:-1])[tree.slot_infoset]
    return TabularPolicy(tree, w / sums)


def pure_strategies(tree: GameTree, player: int):
    """Every pure strategy of ``player`` as a slot-probability vector over its own infosets."""
    mine = np.flatnonzero(tree.infoset_player == player)
    choices = [range(int(tree.infoset_offset[i + 1] - tree.infoset_offset[i])) for i in mine]
    for pick in itertools.product(*choices):
        probs = np.zeros(tree.num_slots)
        for i, a in zip(mine, pick, strict=True):
            probs[tree.infoset_offset[i] + a] = 1.0
        yield mine, probs


def brute_force_best_response(tree: GameTree, policy: TabularPolicy, player: int) -> float:
    mask = tree.player_slot_mask(player)
    best = -np.inf
    for _, pure in pure_strategies(tree, player):
        probs = np.where(mask, pure, policy.probs)
        best = max(best, float(expected_values(tree, probs)[player]))
    return best


def reference() -> dict:
    return json.loads(REFERENCE.read_text())


def reference_mismatch(game: str, solver: str, *, fault: Fault = Fault.NONE) -> float:
    """Largest gap between our solver and OpenSpiel's frozen checkpoints."""
    tree = {"kuhn": kuhn_tree, "leduc": leduc_tree}[game]()
    rows = reference()["games"][game]["solvers"][solver]
    run = CFRSolver(tree, PRESETS[solver], fault=fault)
    worst = 0.0
    for row in rows:
        run.run(row["iterations"] - run.iterations)
        avg = run.average_policy()
        worst = max(
            worst,
            abs(nash_conv(tree, avg).nash_conv - row["average_nash_conv"]),
            abs(float(expected_values(tree, avg)[0]) - row["average_value_p0"]),
            abs(nash_conv(tree, run.current_policy()).nash_conv - row["current_nash_conv"]),
        )
    return worst


def explicit_average(tree: GameTree, iterates: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Average strategy from an iterate history: weight_t * own reach of I under sigma_t, normalised."""
    total = np.zeros(tree.num_slots)
    decision = np.flatnonzero(tree.node_player >= 0)
    for sigma, w in zip(iterates, weights, strict=True):
        reach = reach_probabilities(tree, sigma)
        own = np.zeros(tree.num_infosets)
        np.maximum.at(own, tree.node_infoset[decision], reach[decision, tree.node_player[decision]])
        total += w * own[tree.slot_infoset] * sigma
    sums = np.add.reduceat(total, tree.infoset_offset[:-1])[tree.slot_infoset]
    return np.where(sums > 0, total / np.where(sums > 0, sums, 1.0), 1.0 / np.diff(tree.infoset_offset)[tree.slot_infoset])


def average_weighting_error(tree: GameTree, solver_name: str = "cfr", iterations: int = 20, *, fault: Fault = Fault.NONE) -> float:
    """Gap between the solver's average policy and the explicit reach-weighted average of its iterates.

    Player p's own slots only change after p's pass, so the profile at the
    start of iteration t holds every player's iterate-t strategy.
    """
    cfg = PRESETS[solver_name]
    solver = CFRSolver(tree, cfg, fault=fault)
    iterates: list[np.ndarray] = []
    weights: list[float] = []
    for t in range(1, iterations + 1):
        iterates.append(solver.sigma.copy())
        weights.append(float(t) ** cfg.gamma if cfg.gamma else 1.0)
        solver.run(1)
    return float(np.abs(solver.average_policy().probs - explicit_average(tree, iterates, weights)).max())


def max_regret_z(
    tree: GameTree,
    policy: TabularPolicy,
    traverser: int,
    samples: int,
    *,
    fault: Fault = Fault.NONE,
    seed: int = 0,
    min_reach: float = 1e-3,
) -> float:
    """Largest |z| between sampled ES regrets and the exact counterfactual regrets.

    Only the traverser's slots whose infoset has external reach >= ``min_reach``
    are tested, so each has enough samples for a normal approximation.
    """
    exact = counterfactual_regrets(tree, policy, traverser)
    est = sampled_regret_estimate(tree, policy, traverser, samples=samples, seed=seed, fault=fault)
    external = infoset_reach(tree, policy).external[tree.slot_infoset]
    test = tree.player_slot_mask(traverser) & (external >= min_reach)
    gap = np.abs(est.mean - exact)[test]
    se = est.stderr[test]
    zero_var = se < 1e-12
    if np.any(gap[zero_var] > 1e-9):
        return np.inf
    return float((gap[~zero_var] / se[~zero_var]).max(initial=0.0))
