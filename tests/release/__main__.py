"""``python -m tests.release``: run, resume and report the release gates.

    run [--gate G ...] [--jobs N] [--fresh] [--no-report]   run missing/stale/failed shards, then report
    report                                                 aggregate shards; exit 1 unless every gate passed
    status                                                 shard states per gate
    repro CASE_JSON [--trace]                              rerun one recorded failing case
    minimize --ruleset R --players N --seed S [--write]    shrink a failing generated hand to a fixture
    shard GATE INDEX                                       (internal) run one shard
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.release")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gate", action="append", dest="gates")
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    r.add_argument("--fresh", action="store_true", help="rerun shards even if they are up to date")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--exclusive-last", action="store_true", help="run idle-machine gates after the parallel ones")
    sub.add_parser("report")
    sub.add_parser("status")
    s = sub.add_parser("shard")
    s.add_argument("gate")
    s.add_argument("index", type=int)
    rp = sub.add_parser("repro")
    rp.add_argument("case")
    rp.add_argument("--trace", action="store_true")
    m = sub.add_parser("minimize")
    m.add_argument("--ruleset", required=True)
    m.add_argument("--players", type=int, required=True)
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--write", action="store_true", help="save the fixture and retain the seed under tests/regressions")
    args = ap.parse_args(argv)

    from tests.release import harness

    if args.cmd == "shard":
        return harness.execute_shard(args.gate, args.index)
    if args.cmd == "run":
        unknown = [g for g in args.gates or [] if g not in harness.GATE_BY_ID]
        if unknown:
            ap.error(f"unknown gate(s) {unknown}; choose from {[g.id for g in harness.GATES]}")
        harness.run(args.gates, jobs=max(1, args.jobs), fresh=args.fresh, exclusive_first=not args.exclusive_last)
        if args.no_report:
            return 0
        from tests.release.report import build_report

        return build_report()
    if args.cmd == "report":
        from tests.release.report import build_report

        return build_report()
    if args.cmd == "status":
        for gate in harness.GATES:
            refs = harness.shard_refs(gate)
            states = Counter(ref.state() for ref in refs)
            print(f"{gate.id:22s} {len(refs):4d} shards  " + "  ".join(f"{k}={v}" for k, v in sorted(states.items())))
        return 0
    if args.cmd == "repro":
        from tests.release.repro import repro

        return repro(args.case, trace=args.trace)
    if args.cmd == "minimize":
        from tests.release.repro import minimize_cli

        return minimize_cli(args.ruleset, args.players, args.seed, write=args.write)
    return 2


if __name__ == "__main__":
    sys.exit(main())
