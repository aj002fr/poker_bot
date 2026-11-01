"""Command line for the baseline parameter search.

    python -m strategy.search run strategy/search/configs/nlhe-cem.json --jobs 8
    python -m strategy.search run --run-id smoke --generations 1 --population 4 --tables 8 --confirm 8
    python -m strategy.search status artifacts/search/nlhe-cem-v1
    python -m strategy.search population --partition validation
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> int:
    from strategy.search.cem import DEFAULT_RUN_ROOT, SearchConfig, run_search

    raw: dict = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    overrides = {"run_id": args.run_id, "generations": args.generations, "population": args.population,
                 "tables_per_gen": args.tables, "confirm_tables": args.confirm, "algorithm": args.algorithm,
                 "ruleset_id": args.ruleset, "source": args.source, "seed": args.seed}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    if "start_file" in raw:
        raw["start"] = json.loads(Path(raw.pop("start_file")).read_text(encoding="utf-8"))["params"]
    known = {f.name for f in fields(SearchConfig)}
    unknown = set(raw) - known
    if unknown:
        print(f"unknown config keys {sorted(unknown)}", file=sys.stderr)
        return 2
    cfg = SearchConfig(**raw)
    root = Path(args.root) if args.root else DEFAULT_RUN_ROOT
    run_dir = run_search(cfg, root=root, jobs=args.jobs, log=lambda s: print(s, flush=True))
    print(f"run directory: {run_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run = Path(args.run_dir)
    gens = [json.loads(line) for line in (run / "generations.jsonl").read_text().splitlines() if line]
    for g in gens:
        c = g.get("confirm")
        print(f"gen {g['gen']:>3}: inc {g['incumbent_select_mean']:6.2f}  mean {g['mean_select_mean']:6.2f}  "
              f"best {g['best_sample_mean']:6.2f}  "
              + (f"confirm {c['mean']:+5.2f} t {c['t']:5.2f}" if c else "confirm   -        ")
              + f"  {'ACCEPT' if g['accepted'] else 'keep  '}  inc {g['incumbent_hash'][:12]}  {g['wall_s']}s")
    return 0


def cmd_population(args: argparse.Namespace) -> int:
    from strategy.search.population import load_population, population_digest

    print(json.dumps(population_digest(load_population(args.source), args.partition), indent=1, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m strategy.search")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run or resume a search")
    r.add_argument("config", nargs="?")
    r.add_argument("--run-id")
    r.add_argument("--generations", type=int)
    r.add_argument("--population", type=int)
    r.add_argument("--tables", type=int)
    r.add_argument("--confirm", type=int)
    r.add_argument("--algorithm", choices=("cem", "cmaes"))
    r.add_argument("--ruleset")
    r.add_argument("--source")
    r.add_argument("--seed")
    r.add_argument("--root")
    r.add_argument("--jobs", type=int)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("status", help="print a run's generations")
    s.add_argument("run_dir")
    s.set_defaults(fn=cmd_status)
    q = sub.add_parser("population", help="describe a partition of the opponent source")
    q.add_argument("--source", default="auto")
    q.add_argument("--partition", default="train")
    q.set_defaults(fn=cmd_population)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
