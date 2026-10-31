# Release verification

`make verify-release` runs every spec section 13 gate, plus the 12.2 metamorphic
relations, the 12.4 seeded-defect mutation checks and a deterministic-repeat check, as
checkpointed shards. It writes `artifacts/test-report.json` (schema
`arena-test-report.v1`), a readable `artifacts/test-report.md`, and
`artifacts/release/failing-seeds.json`. It exits 1 unless every applicable gate passed.
Every table and report stays labelled *practice assumptions*. Nothing here claims
compatibility with the official Sequoia rules or anything about bot strength.

```bash
make db-up                      # PostgreSQL for the fault gate (see docs/persistence.md)
make verify-release             # run or resume everything, then report (JOBS=8 by default)
make verify-release-status      # shard states per gate
make verify-release-report      # rebuild the report from the checkpoints only
uv run python -m tests.release run --gate state_machine --jobs 7   # one gate
uv run python -m tests.release run --gate performance --no-report  # one idle-machine gate
```

A fresh VM also needs docker, with `sudo dockerd &` started by hand, before
`make db-up`. The browser gate also needs `npm ci` and `npx playwright install
chromium` in `web/`.

## Shards, checkpoints and resume

- **Every gate is a list of shards.** Each shard runs in its own process and writes
  `artifacts/release/shards/<gate>/<gate>-NNN.json` atomically when it finishes. Its
  log goes to `artifacts/release/logs/`.
- **A shard counts only if it is up to date and passed.** Its record must say `passed`,
  and its fingerprint must match. The fingerprint hashes the gate's code inputs (the
  engine, the oracles, the workload module and so on) together with the shard
  parameters. Change the engine, and every gate that depends on it becomes stale and
  reruns. Changing orchestration or the report (`harness.py`, `report.py`,
  `__main__.py`, `repro.py`) invalidates nothing.
- **Resume.** `run` skips shards that are up to date and runs everything that is
  missing, stale, failed or errored. If a shard process dies without a result, its
  record becomes `error`.
- **Order.**
  1. The parallel pool: bulk, metamorphic and mutation shards.
  2. The *exclusive* gates, one shard at a time on an otherwise idle machine. These are
     timing-sensitive or real-time: fast, fault, browser, isolation, isolated workers
     and performance.
  3. Gates that check other gates' checkpoints (determinism).
  - `--exclusive-last` is the `make` default. It runs the parallel pool first, so the
    idle-machine gates start after it.
- **Gate status.**
  - `failed`: any shard failed or errored, any case failed, or any gate-level check
    failed.
  - `incomplete`: any shard is missing or stale.
  - `passed`: none of the above.
  - `not_applicable`: the official-compatibility gate only, with its reason. It is
    never counted as passed.

## Gates

| Gate | Workload | What must hold |
| --- | --- | --- |
| `fast` | `pytest -m "not release"` with PostgreSQL required | zero failures; no skip except two structural ones, listed with reasons |
| `nlhe_eval`, `shortdeck_eval`, `plo_eval` | all 2,598,960 / 376,992 five-card hands; 200,000 seven-card samples; 200,000 distinct PLO samples incl. ≥10,000 false-flush and board-only | engine class order equals the independent oracle, category totals equal the spec table |
| `state_machine` | 1,050,000 generated hands: NLHE 100,000 per N=2..8, others 25,000 per N | invariants after every transition, illegal probes rejected without mutation, oracle settlement, event-for-event replay |
| `tournaments` | 36,000 complete tables of seeded baseline mixes, in-process | oracle per hand, recomputed markers/blinds/levels, cap or sole survivor, placements and points against the ICM oracle, hash-verified replay, no fallback |
| `icm` | 7,000 randomized cases N=2..8 | ≤1e-9 against the exact permutation oracle, conservation, symmetry |
| `fault` | persistence suite (memory + PostgreSQL), API on both stores, worker fault/deadline/retry/restore suites | named 12.3 injections incl. the 100× action/timeout race on each store pass |
| `browser` | `tsc -b` and the Playwright suite on the deterministic test clock | every spec passes |
| `isolation` | sandbox negative tests, projection/credential/replay authorization tests | all pass, nothing skipped |
| `isolated_tournaments` | 512 eight-bot tables through real sandboxed workers (64 seed blocks × 8 rotations) | every table valid, replayed, hash-verified and re-settled by the oracles |
| `performance` | 128 eight-seat NLHE tables of sandboxed baseline decisions (≥100,000), 2 jobs | p50/p95/p99/max, worker parse / bot / IPC split, throughput, peak RSS; no accepted decision over 1 ms; suitability: fallback rate ≤1% and pooled p99 ≤0.5 ms |
| `metamorphic` | 4,200 base tables × 7 variants + serial/parallel manifests | suit renaming, seat/button rotation, chip scaling, table relabel, serial = 4-job parallel |
| `mutation` | 8 seeded defects in a temporary copy of the code | each is caught by its named test, which passes on the clean copy |
| `determinism` | 15 bulk shards rerun with two other hash seeds | identical result hashes |
| `official_compat` | needs the official kit | not applicable |

The suitability thresholds and the 1 ms budget are fixed. If they fail, the gate fails;
nothing is loosened to make it pass.

## Failing cases and seeds

- **Recording.** Every failing case is recorded with the exact command that reproduces
  it, `uv run python -m tests.release repro '<case json>'`. The report collects these
  under `failing_seeds`.
- **Minimizing.** A failing generated hand is shrunk into a fixture with
  `python -m tests.release minimize --ruleset R --players N --seed S --write`.
  - The minimizer replays the recorded actions.
  - It drops seats that are not the button or a blind and trims stacks, keeping only
    changes under which the same failure signature persists.
  - The fixture's expected pots, contributions and final stacks come from the
    independent pot ledger and evaluators, never from the engine under test.
- **Retained seeds.** `--write` saves the fixture to `tests/regressions/fixtures/` and
  keeps the seed in `tests/regressions/failing_seeds.json`. Both run in the fast suite
  (`tests/regressions/test_failing_seeds.py`).

## Mutation checks

`tests/release/mutation.py` copies `arena/`, `tests/` and `pyproject.toml` to a
temporary directory and checks that `arena` imports from that copy. It then applies
each patch: every replacement must match exactly once, so a refactor cannot leave a
mutant silently testing nothing. Each named test is run twice: it must pass on the
clean copy and fail on the mutant. The working tree is never modified.

| Spec defect | Patch | Named tests |
| --- | --- | --- |
| swap HU action order | heads-up engine order `(button, bb)` | `test_core_fixture[G01]` |
| debit raise target instead of delta | raise also debits the chips already in front | `test_raise_to_debits_only_the_delta`, `test_core_fixture[G14]` |
| reopen after every short all-in | any raise since a seat acted restores raise rights | `test_core_fixture[G09b]`, `[G21a]` |
| drop folded contributions | pot layers count only live seats | `test_core_fixture[G12]` |
| award unmatched excess as a side pot | returned excess kept as a one-player pot | `test_core_fixture[G13]` |
| change an ICM denominator | subset probabilities divide by all stacks | `test_icm_fixture[T04]` |
| expose a hidden card | every hole card shown in every view | `test_views_identical_across_hidden_card_worlds`, `test_no_unauthorized_cards_in_any_body_frame_or_error` |
| apply a duplicate action twice | a retried `action_id` skips the receipt check | `test_action_id_idempotency`, `test_duplicate_identical_actions_racing_commit_once[memory]` |

## Metamorphic relations

These live in `tests/metamorphic/relations.py`. The fast suite runs a small set
(`tests/metamorphic/test_metamorphic.py`), and the release gate runs them at scale.
Each relation plays a base table and a transformed variant, maps the variant's hand
records back into the base frame, and requires exact equality of every blind, deal,
action, collection, award, elimination, placement and point.

- **Suit renaming.** Every deck is permuted by suit.
- **Seat/button rotation.** Seats map `s -> (s + r) mod 8`, and the policy streams
  follow the players.
- **Chip scaling.** Stacks and blinds are multiplied by `k`. The variant's policy acts
  on the legal actions divided by `k`.
  - A hand whose base settlement splits a pot unevenly cannot scale exactly. That hand
    and the rest of its table are excluded and counted under the report's exclusions;
    they are never compared or passed.
- **Table relabel.** The same seed is played under another anonymous table id.
- **Serial versus parallel.** One in-process benchmark manifest runs with 1 and with
  4 jobs. Every saved history must be byte-identical, and the manifest must be
  identical apart from wall-clock measurements and the process layout.
