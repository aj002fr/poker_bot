"""Compare solvers on Kuhn and Leduc by nodes touched and wall time.

Every method gets the same budget of node visits. NashConv of the average
policy is measured at geometric checkpoints; evaluation time and numba JIT
warmup are excluded from the reported solve seconds.

    make bench-solve                                               # full run, writes artifacts/bench-solve/
    uv run --group train python -m strategy.solve.bench --quick    # small budgets (smoke test)

numba is in the opt-in `train` dependency group, so run it with `--group train`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from strategy.games import kuhn, leduc
from strategy.solve.cfr import PRESETS, CFRSolver
from strategy.solve.evaluate import nash_conv
from strategy.solve.game import GameTree, compile_game
from strategy.solve.mccfr import ExternalSamplingMCCFR

METHODS = ("cfr", "cfr+", "linear-cfr", "dcfr", "es-mccfr", "es-mccfr-linear")


@dataclass(frozen=True)
class BenchGame:
    name: str
    build: object
    node_budget: int
    quick_budget: int
    targets: tuple[float, ...]


GAMES = {
    "kuhn": BenchGame("kuhn", kuhn.KuhnPoker, 4_000_000, 100_000, (0.01, 0.001)),
    "leduc": BenchGame("leduc", leduc.LeducPoker, 60_000_000, 2_000_000, (0.05, 0.01)),
}


@dataclass
class Checkpoint:
    iterations: int
    nodes_touched: int
    seconds: float
    nash_conv: float


@dataclass
class MethodRun:
    game: str
    method: str
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def first_reaching(self, target: float) -> Checkpoint | None:
        return next((c for c in self.checkpoints if c.nash_conv <= target), None)


def make_solver(tree: GameTree, method: str, seed: int = 0):
    if method in PRESETS:
        return CFRSolver(tree, PRESETS[method])
    if method == "es-mccfr":
        return ExternalSamplingMCCFR(tree, seed=seed)
    if method == "es-mccfr-linear":
        return ExternalSamplingMCCFR(tree, seed=seed, linear=True)
    raise ValueError(method)


def _checkpoint_targets(budget: int, count: int = 12) -> list[int]:
    return sorted({max(1, int(budget / 2 ** (count - 1 - k))) for k in range(count)})


def run_method(tree: GameTree, game: str, method: str, budget: int, seed: int = 0) -> MethodRun:
    solver = make_solver(tree, method, seed)
    run = MethodRun(game, method)
    seconds = 0.0
    per_iter = float(tree.num_nodes)
    for target in _checkpoint_targets(budget):
        while solver.nodes_touched < target:
            chunk = max(1, math.ceil((target - solver.nodes_touched) / per_iter))
            before = solver.nodes_touched
            start = time.perf_counter()
            solver.run(chunk)
            seconds += time.perf_counter() - start
            per_iter = max((solver.nodes_touched - before) / chunk, 1.0)
        report = nash_conv(tree, solver.average_policy())
        run.checkpoints.append(Checkpoint(solver.iterations, solver.nodes_touched, seconds, report.nash_conv))
    return run


def warmup() -> None:
    tree = compile_game(kuhn.KuhnPoker())
    for method in METHODS:
        solver = make_solver(tree, method).run(2)
        nash_conv(tree, solver.average_policy())


def summary_table(runs: list[MethodRun], targets: tuple[float, ...]) -> str:
    head = "| Method | Iterations | Nodes touched | Seconds | Final NashConv |"
    sep = "|---|---:|---:|---:|---:|"
    for t in targets:
        head += f" Nodes to {t:g} | Seconds to {t:g} |"
        sep += "---:|---:|"
    lines = [head, sep]
    for r in runs:
        last = r.checkpoints[-1]
        row = f"| {r.method} | {last.iterations:,} | {last.nodes_touched:,} | {last.seconds:.3f} | {last.nash_conv:.3g} |"
        for t in targets:
            hit = r.first_reaching(t)
            row += f" {hit.nodes_touched:,} | {hit.seconds:.3f} |" if hit else " not reached | – |"
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", nargs="*", default=list(GAMES))
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/bench-solve"))
    args = parser.parse_args(argv)

    warmup()
    args.out.mkdir(parents=True, exist_ok=True)
    sections = []
    for name in args.games:
        bg = GAMES[name]
        tree = compile_game(bg.build())
        budget = bg.quick_budget if args.quick else bg.node_budget
        runs = [run_method(tree, name, m, budget, args.seed) for m in args.methods]
        table = summary_table(runs, bg.targets)
        sections.append(
            f"## {name} ({tree.num_nodes:,} nodes, {tree.num_infosets:,} infosets, budget {budget:,} node visits)\n\n"
            + table
        )
        (args.out / f"{name}.json").write_text(json.dumps([asdict(r) for r in runs], indent=2))
    report = "\n\n".join(sections)
    (args.out / "summary.md").write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
