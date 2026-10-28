# arena: Sequoia-style pokerbot practice engine

This is the authoritative poker engine for a practice arena where bots play short
single-table tournaments. It covers rulesets, dealing and betting, pot settlement,
dead-button seating, eliminations and ICM scoring. It also has a private event log with
explicit decks and state hashes, privacy-safe seat and spectator views, canonical JSON
histories with hash-verified replay, and PHH export. It is headless: there is no web UI
or agent sandbox yet (those are later milestones).

All rules are **practice assumptions**, not official Sequoia rules. See
[`docs/assumptions.md`](docs/assumptions.md) for every local rule choice and every place
where the engine corrects PokerKit.

## Rulesets

| `ruleset_id` | Game |
| --- | --- |
| `practice_nlhe` | No-limit Texas Hold'em |
| `uncertainty_shortdeck` | No-limit short-deck Hold'em (6–A, PokerKit's native hand order) |
| `uncertainty_plo4` | Pot-limit Omaha, four hole cards |

Tables seat 2–8 players. They use a dead button, escalating blinds, no ante, and a hand
cap (default 200). If the cap is reached, the survivors are scored by ICM.

## Layout

```
arena/domain/      cards, seeded RNG, canonical JSON, strict models: config, actions,
                   event schema (events.py), observation and view types (observations.py)
arena/rules/       rule profiles and the PokerKit 0.7.7 adapter (betting overlay, settlement)
arena/table/       dead-button positions, the table service, state hashes, projections, drivers
arena/tournament/  standings, same-hand bust ranking, Malmuth-Harville ICM
arena/history/     canonical JSON histories, replay verification, PHH export
arena/benchmark/   run manifests (the benchmark runner arrives in milestone 4)
tests/golden/      spec fixtures as JSON (G, E and T series), re-settled by the oracles
tests/property/    generated hands and tournaments, information-set and leak tests
tests/history/     event schema, replay and tampering, PHH round trips, manifests
tests/oracles/     independent evaluators, pot ledger and ICM (stdlib only)
docs/              assumptions and discrepancies, G20/G21 states, view contracts,
                   history/replay notes, JSON Schemas (docs/schemas/)
```

## Running locally

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 if it is missing.

```bash
make setup         # uv sync --frozen
make verify-fast   # pytest -m "not release"
make verify-release  # longer generated runs (marked release)
make cli-demo      # one seeded 6-player NLHE tournament
```

The CLI plays seeded headless hands and tournaments with simple built-in policies
(`legal_random`, `check_or_call`, `check_or_fold`, `shove`):

```bash
uv run python -m arena.cli hand --ruleset uncertainty_plo4 --players 4 --seed 3
uv run python -m arena.cli tournament --ruleset uncertainty_shortdeck --players 8 --seed 7 --trace
```

The same seed always produces the same decks, positions, events and scores.

Histories, replay and exports work from files under `artifacts/` (git-ignored):

```bash
uv run python -m arena.cli tournament --players 6 --seed 9 --save --table-id t_demo
uv run python -m arena.cli replay --table-id t_demo --verify-hashes
uv run python -m arena.cli history --table-id t_demo --viewer seat:2     # or spectator / reviewer
uv run python -m arena.cli export-phh --table-id t_demo --out artifacts/t_demo.phhs
uv run python -m arena.cli run-baseline --players 6 --tables 20 --seed demo   # manifest + histories
uv run python -m arena.cli schemas                                         # regenerate docs/schemas
```

The view types other components build on are documented in
[`docs/observations-and-projections.md`](docs/observations-and-projections.md). History,
replay, PHH and manifests are described in
[`docs/history-and-replay.md`](docs/history-and-replay.md).
