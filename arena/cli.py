"""Headless command line for the practice engine.

    python -m arena.cli hand --ruleset practice_nlhe --players 6 --seed 1
    python -m arena.cli tournament --ruleset uncertainty_plo4 --players 8 --seed 7 --policy legal_random
"""

from __future__ import annotations

import argparse
import json
import sys

from arena.domain.models import RULESET_IDS, TournamentConfig
from arena.domain.rng import SeededRng
from arena.table.driver import POLICIES, play_out
from arena.table.service import seeded_table


def _config(args: argparse.Namespace, **extra) -> TournamentConfig:
    return TournamentConfig(
        ruleset_id=args.ruleset, initial_player_count=args.players, starting_stack=args.stack, **extra
    )


def _print_events(record) -> None:
    print(f"== {record.hand_id}  blinds {record.blinds[0]}/{record.blinds[1]}  {record.positions.as_dict()}")
    print(f"   start stacks {record.start_stacks}")
    for event in record.events:
        print(f"   {event.type:<18} {json.dumps(event.data, sort_keys=True, default=str)}")


def cmd_hand(args: argparse.Namespace) -> int:
    table = seeded_table(_config(args, hand_cap=1, hands_per_level=1), args.seed)
    play_out(table, POLICIES[args.policy], SeededRng(args.seed, "players"))
    _print_events(table.hands[0])
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    config = _config(args)
    table = seeded_table(config, args.seed)
    play_out(table, POLICIES[args.policy], SeededRng(args.seed, "players"))
    if args.trace:
        for record in table.hands:
            _print_events(record)
    print(f"ruleset {config.ruleset_id}  players {config.initial_player_count}  config {config.config_hash()[:12]}")
    print(f"hands played {len(table.hands)}  (cap {config.hand_cap})")
    for record in table.hands:
        for p in record.eliminations:
            print(f"   {record.hand_id}: seat {p.seat} out, positions {p.positions}, {p.points:g} points")
    for p in table.final_placements:
        print(f"   end: seat {p.seat} positions {p.positions}, {p.points:.6f} points ({p.method})")
    print("practice assumptions; not official Sequoia rules")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("hand", cmd_hand), ("tournament", cmd_tournament)):
        p = sub.add_parser(name)
        p.add_argument("--ruleset", choices=RULESET_IDS, default="practice_nlhe")
        p.add_argument("--players", type=int, default=6)
        p.add_argument("--stack", type=int, default=2000)
        p.add_argument("--seed", default="1")
        p.add_argument("--policy", choices=sorted(POLICIES), default="legal_random")
        if name == "tournament":
            p.add_argument("--trace", action="store_true")
        p.set_defaults(fn=fn)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
