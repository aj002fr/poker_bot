"""Build missing or stale strategy tables (see ``strategy/tables/README.md``).

    python -m strategy.tables.build                 # every missing or stale table, all variants
    python -m strategy.tables.build --check         # list status; exit 1 if a committed table is stale
    python -m strategy.tables.build --force --ruleset practice_nlhe
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from strategy.tables import families, postflop, preflop, store

RULESETS = ("practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4")
OPTIONAL = {"preflop-uncertainty_plo4"}
"""Too large for git (about 450 KB, about a minute on 8 cores); the runtime falls back
to a heuristic preflop percentile until it is built."""


def tables(ruleset_id: str) -> list[tuple[str, Callable[[list[str]], None]]]:
    return [(preflop.table_name(ruleset_id), preflop.main), (postflop.table_name(ruleset_id), postflop.main),
            (families.table_name(ruleset_id), families.main)]


def status(name: str) -> str:
    s = store.stale(name)
    return "missing" if s is None else "stale" if s else "fresh"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ruleset", action="append", choices=RULESETS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-optional", action="store_true", help=f"do not build {sorted(OPTIONAL)}")
    a = ap.parse_args(argv)
    bad = False
    for rs in a.ruleset or RULESETS:
        for name, build in tables(rs):
            st = status(name)
            if a.check:
                print(f"{name:36s} {st}")
                bad |= st == "stale" or (st == "missing" and name not in OPTIONAL)
                continue
            if st == "fresh" and not a.force:
                continue
            if name in OPTIONAL and a.skip_optional:
                continue
            print(f"building {name} ({st})", flush=True)
            build(["--ruleset", rs, "--jobs", str(a.jobs)])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
