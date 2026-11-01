"""Game-agnostic solvers over dense extensive-form game trees.

Typical use::

    from strategy.games import kuhn
    from strategy.solve import CFRSolver, CFR_PLUS, compile_game, nash_conv

    tree = compile_game(kuhn.KuhnPoker())
    solver = CFRSolver(tree, CFR_PLUS).run(1000)
    report = nash_conv(tree, solver.average_policy())
"""

from strategy.solve.cfr import CFR_PLUS, DISCOUNTED, LINEAR, PRESETS, VANILLA, CFRConfig, CFRSolver
from strategy.solve.evaluate import (
    InfosetReach,
    NashConvReport,
    best_response,
    counterfactual_regrets,
    expected_values,
    exploitability,
    infoset_reach,
    nash_conv,
    reach_probabilities,
)
from strategy.solve.faults import Fault
from strategy.solve.game import (
    CHANCE,
    TERMINAL,
    Chance,
    Decision,
    ExtensiveGame,
    GameDefinitionError,
    GameTree,
    Terminal,
    compile_game,
)
from strategy.solve.matrix import MatrixSelfPlay, regret_matching_selfplay
from strategy.solve.mccfr import ExternalSamplingMCCFR, SampledRegret, sampled_regret_estimate
from strategy.solve.policy import TabularPolicy

__all__ = [
    "CHANCE",
    "CFR_PLUS",
    "DISCOUNTED",
    "LINEAR",
    "PRESETS",
    "TERMINAL",
    "VANILLA",
    "CFRConfig",
    "CFRSolver",
    "Chance",
    "Decision",
    "ExtensiveGame",
    "ExternalSamplingMCCFR",
    "Fault",
    "GameDefinitionError",
    "GameTree",
    "InfosetReach",
    "MatrixSelfPlay",
    "NashConvReport",
    "SampledRegret",
    "TabularPolicy",
    "Terminal",
    "best_response",
    "compile_game",
    "counterfactual_regrets",
    "expected_values",
    "exploitability",
    "infoset_reach",
    "nash_conv",
    "reach_probabilities",
    "regret_matching_selfplay",
    "sampled_regret_estimate",
]
