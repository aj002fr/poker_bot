"""Regenerate ``tests/strategy/data/openspiel_reference.json`` from OpenSpiel.

    uv run --group oracle python -m tests.strategy.oracle.openspiel_reference

OpenSpiel is a dev-only oracle: this module and the ``oracle`` tests are the
only code that imports it. The frozen file lets `make verify-fast` check our
solvers against OpenSpiel's numbers without installing it.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import pyspiel
from open_spiel.python import policy as os_policy
from open_spiel.python.algorithms import cfr, discounted_cfr, expected_game_score, exploitability

OUT = Path(__file__).resolve().parents[1] / "data" / "openspiel_reference.json"

GAMES = {"kuhn": "kuhn_poker", "leduc": "leduc_poker"}
CHECKPOINTS = {"kuhn": (1, 2, 3, 10, 100, 1000), "leduc": (1, 2, 3, 10, 50)}
SOLVERS = {
    "cfr": cfr.CFRSolver,
    "cfr+": cfr.CFRPlusSolver,
    "linear-cfr": discounted_cfr.LCFRSolver,
    "dcfr": discounted_cfr.DCFRSolver,
}


def tree_counts(game: pyspiel.Game) -> dict[str, int]:
    counts = {"nodes": 0, "terminal": 0, "chance": 0, "decision": 0}
    infosets: set[str] = set()
    stack = [game.new_initial_state()]
    while stack:
        state = stack.pop()
        counts["nodes"] += 1
        if state.is_terminal():
            counts["terminal"] += 1
            continue
        if state.is_chance_node():
            counts["chance"] += 1
            stack.extend(state.child(a) for a, _ in state.chance_outcomes())
            continue
        counts["decision"] += 1
        infosets.add(state.information_state_string())
        stack.extend(state.child(a) for a in state.legal_actions())
    counts["infosets"] = len(infosets)
    return counts


def policy_value(game: pyspiel.Game, policy: os_policy.Policy) -> float:
    return float(expected_game_score.policy_value(game.new_initial_state(), [policy] * game.num_players())[0])


def build() -> dict:
    out: dict = {"open_spiel": version("open_spiel"), "games": {}}
    for name, short in GAMES.items():
        game = pyspiel.load_game(short)
        uniform = os_policy.UniformRandomPolicy(game)
        entry = {
            "short_name": short,
            "parameters": {k: str(v) for k, v in sorted(game.get_parameters().items())},
            "counts": tree_counts(game),
            "uniform_nash_conv": float(exploitability.nash_conv(game, uniform)),
            "uniform_value_p0": policy_value(game, uniform),
            "solvers": {},
        }
        for solver_name, cls in SOLVERS.items():
            solver = cls(game)
            rows = []
            done = 0
            for k in CHECKPOINTS[name]:
                while done < k:
                    solver.evaluate_and_update_policy()
                    done += 1
                avg = solver.average_policy()
                rows.append({
                    "iterations": k,
                    "average_nash_conv": float(exploitability.nash_conv(game, avg)),
                    "average_value_p0": policy_value(game, avg),
                    "current_nash_conv": float(exploitability.nash_conv(game, solver.current_policy())),
                })
                print(name, solver_name, rows[-1], flush=True)
            entry["solvers"][solver_name] = rows
        out["games"][name] = entry
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
