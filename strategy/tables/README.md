# Strategy tables

Offline tables that the decision-time feature layer (`strategy/features`, `strategy/beliefs`)
looks up instead of simulating. Every generator is seeded: for a given source tree and
parameters the `data` (and so `data_sha256`) is identical whatever `--jobs` is.

| Table | Generator | Content | Size | Build time (8 cores) | In git |
| --- | --- | --- | --- | --- | --- |
| `preflop-<ruleset>` | `strategy.tables.preflop` | per canonical hole class: equity vs 1..7 random hands, tie share, flop-equity histogram and moments, percentile, strength decile, per-card bucket combo counts | NLHE 8 KB, short deck 4 KB, PLO4 450 KB | 7 s, 4 s, 55 s | yes, except PLO4 |
| `postflop-<ruleset>` | `strategy.tables.postflop` | per (street, made level, draw class, texture) cell: equity vs 1..3 random hands, in-cell equity spread, tie share, fallback level | 11-14 KB | 6 s | yes |
| `families-<ruleset>` | `strategy.tables.families` | per opponent family (the arena's style and baseline bots): action-category probabilities by opportunity and depth, from spectator events of seeded tables | 1-3 KB | 6-20 s | yes |

Files live in `strategy/tables/data/` as gzip-compressed canonical JSON:
`{"provenance": {...}, "data": {...}}`. Override the directory with `STRATEGY_TABLES_DIR`.

## Provenance

Every file records the generator module and its command line, the parameters (sample
counts and seed), the git commit it was built from, the Python version, wall time, a
SHA-256 of the canonical `data`, and `inputs_sha256`, a SHA-256 of the source files
whose definitions the data depends on (the key functions in `features/cards.py`, the
context definitions in `beliefs/`, the generator itself). Loading verifies the data
hash. A table is **stale** when its input sources changed since it was built; the
test suite fails on stale committed tables, so a change to a key definition must ship
with regenerated tables.

## Regenerating

```bash
uv run python -m strategy.tables.build --check     # fresh / stale / missing for every table
uv run python -m strategy.tables.build             # build what is missing or stale (includes PLO4 preflop)
uv run python -m strategy.tables.build --skip-optional
uv run python -m strategy.tables.build --force --ruleset practice_nlhe
uv run python -m strategy.tables.preflop --ruleset practice_nlhe --samples 40000 --jobs 8   # one table, custom size
```

The PLO4 preflop table (16,432 classes) is not committed: it is too large for git
and quick to rebuild. Without it the PLO4 runtime uses a monotone heuristic
percentile (`strategy.features.equity.heuristic_percentile`), reports
`equity_source == "prior"` preflop, and disables card removal in range beliefs.
Build it once per checkout with `uv run python -m strategy.tables.build`.

Short deck and PLO4 have no style bots in the arena, so their family tables contain
only the baselines (`calling_station`, `check_fold`, `legal_random`); every other
opponent is absorbed by the posterior's `unknown` family.
