"""Command line for the learning lab.

    python -m strategy.cli rules --ruleset practice_nlhe
    python -m strategy.cli conformance --hands 200
    python -m strategy.cli experiment lab-smoke --jobs 4
    python -m strategy.cli experiment null-aa-nlhe --dry-run
    python -m strategy.cli report artifacts/lab/lab-smoke/<manifest_id>
    python -m strategy.cli sample-size --sigma 6.5 --delta 1.0
    python -m strategy.cli sample-size --pilot artifacts/lab/lab-smoke/<manifest_id> --delta 1.0
    python -m strategy.cli null-synthetic --replicates 2000 --blocks 20
    python -m strategy.cli specs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from arena.domain.models import RULESET_IDS, TournamentConfig


def _print_json(obj: object) -> None:
    from strategy.eval.report import finite

    print(json.dumps(finite(obj), indent=1, sort_keys=True))


def cmd_rules(args: argparse.Namespace) -> int:
    from strategy.rules.adapter import RulesAdapter

    out = []
    for r in args.ruleset or RULESET_IDS:
        config = TournamentConfig(ruleset_id=r, initial_player_count=args.players)
        out.append(RulesAdapter.from_config(config).describe())
    if args.json:
        _print_json(out)
        return 0
    for d in out:
        print(f"{d['ruleset_id']}: rules {d['rules_hash'][:16]}  variant {d['variant_hash'][:16]}")
        print(f"   capabilities: {', '.join(d['capabilities'])}")
    return 0


def cmd_conformance(args: argparse.Namespace) -> int:
    from strategy.rules.conformance import run_all

    reports = run_all(args.ruleset or list(RULESET_IDS), hands=args.hands, seed=args.seed)
    if args.json:
        _print_json([r.as_dict() for r in reports])
    else:
        for rep in reports:
            print(f"{rep.ruleset_id} against {rep.simulator}: {'OK' if rep.ok else 'FAILED'}  "
                  f"(variant {rep.variant_hash[:12]})")
            for c in rep.checks:
                print(f"   {c.name:<12} {'ok' if c.ok else 'FAIL'}  passed {c.passed}  failed {c.failed}  "
                      f"skipped {c.skipped}")
                for f in c.failures[:5]:
                    print(f"      {f}")
                for n in c.notes[:3]:
                    print(f"      note: {n}")
    return 0 if all(r.ok for r in reports) else 1


def _jobs(value: int | None) -> int:
    return value or max(1, min(4, (os.cpu_count() or 2) // 2))


def cmd_experiment(args: argparse.Namespace) -> int:
    from strategy.experiments.runner import DEFAULT_LAB_DIR, AlreadyRun, default_run_dir, describe_plan, materialize, \
        run_experiment
    from strategy.experiments.schema import load_spec

    spec = load_spec(args.spec)
    root = Path(args.root) if args.root else DEFAULT_LAB_DIR
    out = Path(args.out) if args.out else None
    if args.dry_run:
        plan = materialize(spec, (out or root / spec.experiment_id) / "_dry_run")
        d = describe_plan(plan)
        d["run_dir"] = str(out or default_run_dir(plan.manifest, root))
        _print_json(d)
        return 0

    def progress(label: str, done: int, total: int) -> None:
        if not args.quiet and (done == total or done % max(1, total // 4) == 0):
            print(f"   {label}: {done}/{total} tables", file=sys.stderr, flush=True)

    try:
        run_dir, report = run_experiment(spec, out_dir=out, root=root, jobs=_jobs(args.jobs),
                                         keep_histories=args.keep_histories, progress=progress)
    except AlreadyRun as exc:
        print(f"{exc}; see its report.md (use --out for a fresh directory)")
        return 0
    _print_summary(run_dir, report)
    return 0


def _print_summary(run_dir: Path, report: dict) -> None:
    a = report["analysis"]
    p, dec = a["primary"], a["decision"]
    print(f"{report['manifest']['experiment_id']}  manifest {report['manifest']['manifest_id']}")
    if p.get("blocks"):
        lo, hi = p["chosen_interval"]
        print(f"   blocks {p['blocks']}  A {p['mean_points_A']:.3f}  B {p['mean_points_B']:.3f}  "
              f"B-A {p['mean_difference']:+.3f}  {int(p['level'] * 100)}% {p['interval_method']} "
              f"[{lo:+.3f}, {hi:+.3f}]")
    print(f"   verdict {dec['verdict']}: {dec['reason']}")
    null = report.get("null_calibration")
    if null:
        for name, r in null["methods"].items():
            print(f"   null {name:<22} false gains {r['false_gains']}/{null['replicates']} "
                  f"(declared {null['declared_false_gain_rate']:.3f})  {'ok' if r['calibrated'] else 'TOO MANY'}")
        if null.get("exact_coupling"):
            c = null["exact_coupling"]
            print(f"   exact coupling: {c['blocks']} blocks identical={c['identical']}")
    print(f"   report {run_dir / 'report.md'}")


def cmd_report(args: argparse.Namespace) -> int:
    from strategy.eval.report import load_report, write_report

    run_dir = Path(args.run_dir)
    report = load_report(run_dir)
    write_report(run_dir, report)
    if args.print:
        sys.stdout.write((run_dir / "report.md").read_text(encoding="utf-8"))
    else:
        _print_summary(run_dir, report)
    return 0


def cmd_sample_size(args: argparse.Namespace) -> int:
    from strategy.eval.report import load_report
    from strategy.eval.stats import sample_size

    sigma = args.sigma
    if args.pilot:
        sigma = load_report(Path(args.pilot))["analysis"]["primary"]["sd_difference"]
    if sigma is None:
        raise SystemExit("give --sigma or --pilot")
    s = sample_size(sigma, args.delta, alpha=args.alpha, power=args.power)
    if args.json:
        _print_json(s.as_dict())
    else:
        print(f"sigma_D {sigma:.4f}  delta {args.delta:g}  alpha {args.alpha:g}  power {args.power:g}: "
              f"{s.blocks} paired blocks ({s.method}; normal approximation {s.normal_blocks:.1f}, "
              f"multiplier {s.multiplier:.3f})")
    return 0


def cmd_null_synthetic(args: argparse.Namespace) -> int:
    from strategy.eval.null import synthetic_calibration

    out = synthetic_calibration(replicates=args.replicates, blocks=args.blocks, alpha=args.alpha,
                                bootstrap=not args.no_bootstrap, resamples=args.resamples)
    if args.json:
        _print_json(out)
        return 0
    print(f"{out['replicates']} synthetic null replicates x {out['blocks_per_replicate']} blocks; declared false-gain "
          f"rate {out['declared_false_gain_rate']:.3f}")
    for name, r in out["methods"].items():
        lo, hi = r["false_gain_rate_ci"]
        print(f"   {name:<22} false gains {r['false_gains']} ({r['false_gain_rate']:.4f}, 95% [{lo:.4f}, {hi:.4f}])  "
              f"mean width {r['mean_interval_width']:.3f}  {'ok' if r['calibrated'] else 'TOO MANY'}")
    return 0 if all(r["calibrated"] for r in out["methods"].values()) else 1


def cmd_specs(args: argparse.Namespace) -> int:
    from strategy.experiments.schema import SPEC_DIR, load_spec

    for p in sorted(SPEC_DIR.glob("*.json")):
        s = load_spec(p)
        print(f"{p.stem:<22} {s.kind:<17} {s.stage:<13} {s.title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="strategy")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rules", help="variant metadata, capabilities and rules hashes")
    p.add_argument("--ruleset", action="append", choices=RULESET_IDS)
    p.add_argument("--players", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_rules)

    p = sub.add_parser("conformance", help="check the engine against variant metadata and arena fixtures")
    p.add_argument("--ruleset", action="append", choices=RULESET_IDS)
    p.add_argument("--hands", type=int, default=200, help="differential hands per variant")
    p.add_argument("--seed", default="conformance-v1")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_conformance)

    p = sub.add_parser("experiment", help="materialize a spec, run it and write report.md/report.json")
    p.add_argument("spec", help="spec file or id in strategy/experiments/specs/")
    p.add_argument("--out", default=None, help="run directory (default artifacts/lab/<id>/<manifest_id>)")
    p.add_argument("--root", default=None)
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="print the manifest plan without playing")
    p.add_argument("--keep-histories", action="store_true", help="keep private table histories in the run dir")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_experiment)

    p = sub.add_parser("report", help="re-render report.md from a run's report.json")
    p.add_argument("run_dir")
    p.add_argument("--print", action="store_true")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("sample-size", help="paired blocks needed for a minimum effect")
    p.add_argument("--sigma", type=float, default=None, help="pilot SD of block differences")
    p.add_argument("--pilot", default=None, help="run directory whose report supplies sigma")
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_sample_size)

    p = sub.add_parser("null-synthetic", help="false-gain calibration on simulated paired tournaments")
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--resamples", type=int, default=500)
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_null_synthetic)

    p = sub.add_parser("specs", help="list the experiment specs")
    p.set_defaults(fn=cmd_specs)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
