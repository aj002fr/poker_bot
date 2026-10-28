"""Headless command line for the practice engine.

    python -m arena.cli hand --ruleset practice_nlhe --players 6 --seed 1
    python -m arena.cli tournament --ruleset uncertainty_plo4 --players 8 --seed 7 --save
    python -m arena.cli replay --table-id t_demo --verify-hashes
    python -m arena.cli export-phh --table-id t_demo --out artifacts/t_demo.phhs
    python -m arena.cli history --table-id t_demo --viewer seat:3
    python -m arena.cli run-baseline --ruleset practice_nlhe --players 6 --tables 20 --seed demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arena.domain.canonical import canonical_json, canonical_hash
from arena.domain.models import RULESET_IDS, TournamentConfig
from arena.history.canonical import DEFAULT_HISTORY_DIR, history_path, load_history, write_history
from arena.table.driver import POLICIES, play_out, policy_rngs
from arena.table.service import seeded_table

RUNS_DIR = Path("artifacts/runs")


def _config(args: argparse.Namespace, **extra) -> TournamentConfig:
    return TournamentConfig(
        ruleset_id=args.ruleset, initial_player_count=args.players, starting_stack=args.stack, **extra
    )


def _table_id(args: argparse.Namespace, config: TournamentConfig) -> str:
    if args.table_id:
        return args.table_id
    return "t_" + canonical_hash([config.config_hash(), str(args.seed)], domain="arena-cli-table-v1")[:12]


def _print_events(record) -> None:
    print(f"== {record.hand_id}  blinds {record.blinds[0]}/{record.blinds[1]}  {record.positions.as_dict()}")
    print(f"   start stacks {record.start_stacks}")
    for event in record.events:
        print(f"   {event.type:<18} {json.dumps(event.data, sort_keys=True, default=str)}")


def _play(args: argparse.Namespace, config: TournamentConfig):
    table = seeded_table(config, args.seed, table_id=_table_id(args, config))
    play_out(table, POLICIES[args.policy], policy_rngs(args.seed, config.resolved_seats))
    if args.save:
        print(f"history written to {write_history(table, Path(args.history_dir))}")
    return table


def cmd_hand(args: argparse.Namespace) -> int:
    table = _play(args, _config(args, hand_cap=1, hands_per_level=1))
    _print_events(table.hands[0])
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    config = _config(args)
    table = _play(args, config)
    if args.trace:
        for record in table.hands:
            _print_events(record)
    print(f"table {table.table_id}  ruleset {config.ruleset_id}  players {config.initial_player_count}  "
          f"config {config.config_hash()[:12]}")
    print(f"hands played {len(table.hands)}  (cap {config.hand_cap})")
    for record in table.hands:
        for p in record.eliminations:
            print(f"   {record.hand_id}: seat {p.seat} out, positions {p.positions}, {p.points:g} points")
    for p in table.final_placements:
        print(f"   end: seat {p.seat} positions {p.positions}, {p.points:.6f} points ({p.method})")
    print("practice assumptions; not official Sequoia rules")
    return 0


def _find_history(args: argparse.Namespace) -> Path:
    if args.file:
        return Path(args.file)
    if args.history_dir:
        return history_path(args.table_id, Path(args.history_dir))
    candidates = [history_path(args.table_id, DEFAULT_HISTORY_DIR)]
    candidates += sorted(RUNS_DIR.glob(f"*/tables/{args.table_id}.json"))
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(f"no stored history for table {args.table_id!r} (looked in {DEFAULT_HISTORY_DIR} and {RUNS_DIR})")


def cmd_replay(args: argparse.Namespace) -> int:
    from arena.history.replay import ReplayIncompatible, ReplayMismatch, replay

    path = _find_history(args)
    try:
        report = replay(load_history(path), allow_engine_mismatch=args.allow_engine_mismatch)
    except ReplayIncompatible as exc:
        print(f"HALT {path}: incompatible engine: {exc}", file=sys.stderr)
        return 3
    except (ReplayMismatch, ValueError) as exc:
        print(f"HALT {path}: {exc}", file=sys.stderr)
        return 2
    print(f"table {report.table_id}: replay matches the stored record ({path})")
    print(f"   events {report.events}  hands {report.hands}  complete {report.complete}")
    if args.verify_hashes:
        print(f"   hand-start hashes verified {report.hand_starts_verified}")
        print(f"   decision-boundary state hashes verified {report.decisions_verified}")
    print(f"   final stacks {report.final_stacks}")
    if report.points is not None:
        print(f"   points {report.points}")
    for note in report.notes:
        print(f"   note: {note}")
    return 0


def cmd_export_phh(args: argparse.Namespace) -> int:
    from arena.history.phh import export_history_phhs

    text = export_history_phhs(load_history(_find_history(args)))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="ascii")
        print(f"PHH written to {out}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from arena.table.projections import ProjectionDenied, Viewer, history_view

    history = load_history(_find_history(args))
    if args.viewer == "spectator":
        viewer = Viewer.spectator()
    elif args.viewer == "reviewer":
        viewer = Viewer.reviewer()
    elif args.viewer.startswith("seat:"):
        viewer = Viewer.for_seat(int(args.viewer[5:]))
    else:
        raise SystemExit("--viewer is spectator, reviewer or seat:N")
    try:
        view = history_view(history.events, viewer, table_complete=history.summary.complete)
    except ProjectionDenied as exc:
        print(f"denied: {exc}", file=sys.stderr)
        return 4
    print(canonical_json(view.model_dump(mode="json")))
    return 0


def cmd_run_baseline(args: argparse.Namespace) -> int:
    from arena.benchmark.manifest import default_run_id, run_baseline

    config = _config(args)
    policies = {s: (args.policy, POLICIES[args.policy]) for s in config.resolved_seats}
    run_id = args.run_id or default_run_id(config, str(args.seed), args.tables)
    out = Path(args.out) if args.out else RUNS_DIR / run_id
    manifest = run_baseline(config=config, tables=args.tables, base_seed=str(args.seed), policies=policies,
                            out_dir=out, run_id=run_id)
    r = manifest.results
    print(f"run {manifest.run_id}: {len(r.completed_table_ids)} completed, {len(r.failed_tables)} failed, "
          f"{len(r.incomplete_table_ids)} incomplete of {manifest.scheduled_tables}")
    print(f"   manifest {out / 'manifest.json'}  (content hash {manifest.content_hash()[:16]})")
    print(f"   mean points by seat {r.mean_points_by_seat}")
    return 0 if not r.failed_tables else 1


def cmd_schemas(args: argparse.Namespace) -> int:
    from arena.schemas import write_schemas

    for path in write_schemas(Path(args.out)):
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena")
    sub = parser.add_subparsers(dest="command", required=True)

    def play_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ruleset", choices=RULESET_IDS, default="practice_nlhe")
        p.add_argument("--players", type=int, default=6)
        p.add_argument("--stack", type=int, default=2000)
        p.add_argument("--seed", default="1")
        p.add_argument("--policy", choices=sorted(POLICIES), default="legal_random")

    def table_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--table-id", required=True)
        p.add_argument("--history-dir", default=None, help=f"default: {DEFAULT_HISTORY_DIR}, then {RUNS_DIR}/*/tables")
        p.add_argument("--file", default=None, help="explicit history file")

    for name, fn in (("hand", cmd_hand), ("tournament", cmd_tournament)):
        p = sub.add_parser(name)
        play_args(p)
        p.add_argument("--table-id", default=None)
        p.add_argument("--save", action="store_true", help="write the private history")
        p.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
        if name == "tournament":
            p.add_argument("--trace", action="store_true")
        p.set_defaults(fn=fn)

    p = sub.add_parser("replay", help="replay a stored history and verify it")
    table_args(p)
    p.add_argument("--verify-hashes", action="store_true", help="report hand-start and decision hash checks")
    p.add_argument("--allow-engine-mismatch", action="store_true")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("export-phh", help="export complete hands as PHH")
    table_args(p)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_export_phh)

    p = sub.add_parser("history", help="print a viewer-authorized history")
    table_args(p)
    p.add_argument("--viewer", default="spectator", help="spectator, reviewer or seat:N")
    p.set_defaults(fn=cmd_history)

    p = sub.add_parser("run-baseline", help="play seeded in-process tables and write a run manifest")
    play_args(p)
    p.add_argument("--tables", type=int, default=10)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out", default=None, help=f"run directory (default {RUNS_DIR}/<run_id>)")
    p.set_defaults(fn=cmd_run_baseline)

    p = sub.add_parser("schemas", help="write the JSON Schemas of the public contracts")
    p.add_argument("--out", default="docs/schemas")
    p.set_defaults(fn=cmd_schemas)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
