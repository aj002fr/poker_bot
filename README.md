# arena: Sequoia-style pokerbot practice engine

This is the authoritative poker engine for a practice arena where bots play short
single-table tournaments. It covers rulesets, dealing and betting, pot settlement,
dead-button seating, eliminations and ICM scoring. It also has a private event log with
explicit decks and state hashes, privacy-safe seat and spectator views, canonical JSON
histories with hash-verified replay, and PHH export. A playable table (FastAPI plus a
React table) lets humans and simple baseline bots play on the same engine, in memory or
durably on PostgreSQL with crash recovery. A local Python bot SDK runs bots in sandboxed
workers with a 1 ms decision deadline, with baseline bots and a paired-seed benchmark
runner. The browser also replays finished tables and launches and reports batch runs.

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
arena/agents/      bot SDK (sdk.py), sandboxed worker pool and deadlines (worker.py),
                   process confinement (sandbox.py), baselines/
arena/benchmark/   run manifests, paired seed-block benchmark runner (runner.py)
arena/api/         FastAPI server: seat credentials, per-table command writer, deadlines, WebSocket
arena/persistence/ memory and PostgreSQL stores, migrations, snapshot + replay recovery
web/               React + TypeScript + Vite: setup, lobby, table, results, replay, batch runs; Playwright e2e
examples/          template_bot.py: copy it to start a bot
tests/agents/      fault-injection bots, sandbox probes, fake-clock deadline tests, restore
tests/api/         API behaviour and HTTP/WebSocket leak checks
tests/persistence/ resume equivalence, store contract, crash/race/outage/fencing faults
tests/golden/      spec fixtures as JSON (G, E and T series), re-settled by the oracles
tests/property/    generated hands and tournaments, information-set and leak tests
tests/history/     event schema, replay and tampering, PHH round trips, manifests
tests/oracles/     independent evaluators, pot ledger and ICM (stdlib only)
docs/              assumptions and discrepancies, G20/G21 states, view contracts,
                   history/replay notes, JSON Schemas (docs/schemas/)
```

## Running locally

You need [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if it is missing)
and Node 20+ for the web table.

```bash
make setup         # uv sync --frozen, npm ci in web/
make dev           # API on :8317, table on http://127.0.0.1:5317 (tables in memory)
make dev-postgres  # same, with tables in PostgreSQL (docker compose, port 54329)
make verify-postgres  # persistence and API suites against PostgreSQL
make verify-web    # web typecheck + Playwright suite (npx playwright install chromium once)
make verify-fast   # pytest -m "not release"
make verify-release  # every release gate as resumable shards -> artifacts/test-report.{json,md}
make verify-release-status  # shard states per gate
make cli-demo      # one seeded 6-player NLHE tournament
```

Release verification (spec sections 12 and 13), its gates, checkpoints, failing-seed
minimizer and mutation checks are described in `docs/release-verification.md`.

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

## Playing in the browser

Run `make dev` and open http://127.0.0.1:5317. Choose a rule scenario, 2–8 seats
and who sits where (a human or a baseline bot; bots that do not support the ruleset
are disabled), then create the table. You take the first human seat. Every other human seat gets an
"Open in new tab" link (each tab holds its own private seat credential), and the host
starts the table once every human seat is taken. With `make dev` tables live in memory,
so restarting the server clears them; `make dev-postgres` keeps them and recovers
them after a restart or crash ([`docs/persistence.md`](docs/persistence.md)).

After a table, **Replay this table** steps through every action and hand at 1–16×,
from your seat or as a spectator; the host can switch to the reviewer view with every
hole card. `make dev` also prints an **operator token** (saved to
`artifacts/operator-token`). Paste it under **Operator** to seat registered bot files
and to launch batch benchmark runs from **Batch runs**, whose reports show config and
artifact hashes, mean points with confidence intervals, timing and fallbacks per
scenario and seat count. The endpoints, deadlines and privacy rules are in
[`docs/playable-table.md`](docs/playable-table.md).

## Bots

Write a bot by copying [`examples/template_bot.py`](examples/template_bot.py), then play
it against the built-in baselines (`check_fold`, `calling_station`, `legal_random`, and
the NLHE-only `nlhe_heuristic`). Each run prints mean points with a 95% interval,
p50/p95/p99/max decision timing in seconds, and fallback counts:

```bash
uv run python -m arena.cli match --bot examples/template_bot.py              # 240 tables
uv run python -m arena.cli match --bot my_bot.py --ruleset uncertainty_shortdeck --tables 60
uv run python -m arena.cli benchmark --bots nlhe_heuristic,calling_station,legal_random,check_fold \
    --out artifacts/runs/demo
```

Sandboxed workers need Linux with util-linux `unshare`, unprivileged user namespaces,
Landlock and x86-64 seccomp. Tests that need them skip themselves elsewhere. See
[`docs/agent-sdk.md`](docs/agent-sdk.md) for the bot contract and
[`docs/agent-sandbox.md`](docs/agent-sandbox.md) for what is and is not hardened, with
timing measured on this VM.

The view types other components build on are documented in
[`docs/observations-and-projections.md`](docs/observations-and-projections.md). History,
replay, PHH and manifests are described in
[`docs/history-and-replay.md`](docs/history-and-replay.md).
