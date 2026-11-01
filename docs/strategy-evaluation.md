# Strategy lab: rules, objective, paired evaluation and experiments

The `strategy/` package holds the self-play learning work. This document covers the
variant-agnostic layers every later stage builds on:

- `strategy/rules/`: variant metadata, the rules hash, the canonical action menu, and conformance.
- `strategy/objective/`: the terminal-points objective.
- `strategy/eval/`: paired A/B evaluation, statistics, slices, execution metrics, and the report.
- `strategy/experiments/`: experiment specs, immutable run manifests, and the runner.
- `strategy/cli.py`: the `python -m strategy.cli` entry point.

All results are practice assumptions, not official Sequoia rules.

## Decision-time boundary

At decision time a bot may import only the public SDK (`arena.agents.sdk`) and the
observation contracts (`arena.domain.observations`). Inside this package only
`strategy.rules.variants`, `strategy.rules.adapter` and `strategy.rules.canonical`
are decision-time safe. `tests/strategy/test_rules.py` enforces that with an import
check. Everything else (objective, eval, experiments, conformance) is offline and may
use arena internals.

## Rules (`strategy.rules`)

```python
from strategy.rules import RulesAdapter, get_variant, canonical_actions

adapter = RulesAdapter.from_table_info(table_info)  # or .from_config(TournamentConfig | resolved dict)
adapter.rules_hash         # variant + everything that changes terminal points
adapter.variant_hash       # the game alone
adapter.capabilities       # e.g. {"betting.no_limit", "deck.standard_52", "assets.nlhe", ...}
adapter.variant.require("assets.nlhe", asset="preflop charts")  # raises UnsupportedVariant elsewhere
adapter.actions(observation, fractions=(0.5, 1.0))  # canonical, always-legal menu
```

- **`VariantSpec`** is a frozen description of one variant: deck ranks and suits,
  hole cards, board streets, burns, showdown hand selection, category order, lowest
  straight, and betting structure. `KNOWN_VARIANTS` covers `practice_nlhe`,
  `uncertainty_shortdeck` and `uncertainty_plo4`. A surprise variant is a new
  `VariantSpec`, built with `VariantSpec.from_dict` if needed. It needs no new code
  paths.
- **Capabilities** gate asset reuse. Hold'em charts, priors and tables must call
  `require(CAP_NLHE_ASSETS)`. Only exact NLHE has that capability.
- **The rules hash** covers the variant plus seats, stacks, blind schedule, hand cap,
  payouts, button policy, runouts, rake and local rules. Decision budgets and UI
  timers are excluded.
- **The canonical action menu** contains fold, check, call, min raise, pot raise,
  all-in, and pot-fraction raises. A pot fraction `f` raises to
  `current_bet_to + f * (pot_total + to_call)`. Fraction raises outside the legal
  interval are dropped. When two raises land on the same amount, one survives, in
  priority order all-in, then pot, then min, then fraction. Every entry satisfies
  `legal.allows`.

### Conformance (`strategy.rules.conformance`, offline)

`run_conformance(ruleset_id, simulator=...)` checks any object that implements the
`Simulator` and `SimHand` protocols against the variant metadata and the arena
fixtures:

- **metadata**: deck, deal and betting structure.
- **ranking**: probe hands per deck.
- **fixtures**: replay of `tests/golden/fixtures` core and eval fixtures. Decks are
  rebuilt from our own metadata.
- **scoring**: the ICM fixtures.
- **differential**: lockstep random play against the arena reference, with every
  canonical-menu entry checked for legality.

`ArenaSimulator` is the reference implementation. A fast simulator written for
training must pass this before use. Run `make lab-conformance`.

## Objective (`strategy.objective`)

The objective is official terminal points, and only that:

- **Eliminated**: the finishing position's points. Simultaneous busts share
  positions.
- **Last alive**: the first prize.
- **Survivor at the hand cap**: the exact Malmuth-Harville ICM share of the
  unassigned top prizes.

All of this goes through `arena.tournament.scoring`, the code the table itself uses.
Chips appear only in `chip_diagnostics(...)`, labelled `diagnostic_only`. The
experiment schema rejects any other primary metric.

```python
from strategy.objective import outcome_from_events, rescore, cap_allocation, icm_points
outcome = outcome_from_events(table.log)        # or outcome_from_table(table)
outcome.utility(seat); outcome.finishes[seat].method  # "elimination" | "winner" | "icm"
rescore(payouts, seats, start_stacks, [(hand_number, before, after), ...], hand_cap)  # for rollouts
```

## Paired evaluation (`strategy.eval`)

- **Cell**: one ruleset, table config and opponent lineup. The candidate is
  entrant 0 and the opponents are entrants 1 to N−1.
- **Block**: one table seed. Within a block the candidate rotates through every
  seat.
- **Arms**: arms A and B run `arena.benchmark.runner.run_benchmark` with the same
  seeds, so cards, buttons and opponents' seeds match (common random numbers).

Block utility `U = mean candidate points over rotations`, and `D = U(B) − U(A)`. The
block is the only inferential unit. Hands, decisions and tables within a block are
never treated as independent.

- **Statistics** (`strategy.eval.stats`, standard library only): paired t;
  stratified cluster bootstrap; Welch; Wilson and Newcombe for rates;
  Clopper-Pearson; Holm; `sample_size(sigma_d, delta)` (N ≈ (z₁₋α/₂ + z₁₋β)² σ²/δ²,
  t-refined); and `minimum_detectable_effect`.
- **Decision**: `pass` when the interval's lower bound exceeds `min_effect` and every
  gate passes. `fail` when an execution gate breaks or the interval lies entirely
  below zero. `invalid` when any block is incomplete, for example a failed table.
  Otherwise `inconclusive`. Discovery results are labelled as needing fresh locked
  data.
- **Slices**: candidate seat, variant, lineup, and opponent family, each with Holm
  p-values. Descriptive stats per arm cover:
  - finish distribution and finish method;
  - fraction of tables ended at the cap, and cap ICM points;
  - bust level, and points by survivors at the end;
  - lower-tail CVaR10, and chip diagnostics.

  Changed action rates are reported by street and facing, preflop effective depth,
  blind level, hands to the cap, and players in the hand. These come from public
  quantities only and use nominal Newcombe intervals.
- **Execution** (per arm): latency p50, p95, p99, p99.9 and max; fallbacks,
  timeouts, illegal actions, errors and reruns; artifact bytes and state bytes;
  whether timing was competitive (sandboxed). Gates come from the spec.
- **Report**: `report.md` (concise) and `report.json` (full, including every block
  pair).

### Partition interface (population registry stub)

The registry belongs to `strategy.population`. Evaluation needs only this protocol
from `strategy.eval.population`:

```python
class PartitionSource(Protocol):
    registry_id: str
    def registry_hash(self) -> str: ...
    def partition(self, name: str) -> Partition: ...   # "train" | "validation" | "holdout"
    def sample_lineups(self, partition, *, ruleset_id, opponents, count, seed) -> list[Lineup]: ...
```

- A `Partition` carries `registry_id`, `registry_hash`, `seed_namespace`,
  `members: tuple[OpponentRef, ...]` and `metadata`, and exposes
  `partition_hash()`.
- An `OpponentRef` has `member_id`, `family`, a `policy` spec (`baseline:<name>`,
  `style:<name>` or a bot path), the `rulesets` it plays, and `params_hash`.
- `sample_lineups(...)` is a deterministic default sampler.
- A spec names a source as `"stub"` or as `"package.module:attribute"`, which can be
  a source, a class, or a factory.

`StubPartitionSource` splits the arena's baselines and styles by family. Every
manifest marks it `stub: true`, and it is not valid for promotion claims.

The frozen opponent registry implements the protocol as
`"source": "strategy.population:PopulationSource"`. It has 17 sampled families with
train, validation and holdout partitions, plus promoted checkpoint slots; see
[opponent-population.md](opponent-population.md). The spec default is still `stub`, and
`specs/population-smoke.json` shows the registry in use.

## Experiments (`strategy.experiments`)

A spec, in JSON under `strategy/experiments/specs/` or at any path, declares
everything before the run:

- the hypothesis id and stage (`discovery`, `confirmation` or `final_holdout`);
- arms, including an optional `salt` per arm;
- variants, table structure, the population partition and lineups;
- blocks per lineup and the base seed;
- `min_effect`, `alpha`, `power` and the interval method;
- isolation, the decision budget, and execution gates;
- for `kind: null_calibration`, the replicate layout.

Two rules are enforced. The holdout partition is used by, and only by,
`final_holdout` runs. A null calibration must compare identical policies.

`run_experiment(spec)` materializes an **immutable manifest**. Its id,
`manifest_id = "m_" + hash(content)`, pins:

- the objective, including a hash of the scorer source;
- the engine versions;
- each variant's rules hash, variant hash, arena config hash and capabilities;
- each arm's artifact hash (before and after salting);
- the registry and partition hashes and member ids;
- every cell's lineup with opponent artifact hashes;
- a hash of every block seed.

The manifest is written once with mode 0444 and verified on every load.
`ManifestTampered` means the content changed after the fact. `ManifestConflict`
means a different manifest was written to the same path. Seeds stay in
`private/seeds.json`, and the manifest carries only their hashes. A finished run
directory is never re-run (`AlreadyRun`).

```
artifacts/lab/<experiment_id>/<manifest_id>/
  manifest.json  private/seeds.json  cells/<cell>/<arm>/manifest.json  report.json  report.md
```

### Null A/B calibration

Arms A and B are the same artifact. A **salted** arm wraps the bot so that
`TableInfo.policy_seed` becomes `sha256(salt | seed)`. This makes policy noise
independent between arms while card luck stays shared, so the true difference is
exactly zero. The pooled blocks are split into disjoint replicate experiments, and
each replicate is analysed like a real A/B. The false-gain rate, meaning replicates
whose interval lies above zero, must be consistent with `alpha/2` (binomial
p ≥ 0.01). An extra cell runs the same salt in both arms, and every block difference
must be exactly 0.

Measured on this VM (8 CPUs, in-process, stub population):

| spec | blocks | full-sample B − A (95% paired t) | paired t false gains | two-sided rejections: paired t, bootstrap | exact coupling | wall time |
|---|---|---|---|---|---|---|
| `null-aa-nlhe` (6 players, cap 50) | 1200 | +0.077 [−0.228, 0.382] | 4/100 (p = 0.24) | 6.0%, 11.0% | identical | 6.5 min |
| `null-aa-variants` (4 players, cap 40, 3 variants) | 720 | −0.634 [−1.385, 0.117] | 1/60 (p = 0.78) | 6.7%, 16.7% | identical | 2 min |

The paired t interval holds its declared rate. The percentile cluster bootstrap is
too liberal at 12 blocks per replicate, so `paired_t` stays the default decision
interval. Use the bootstrap only as a robustness check, or for runs with many more
blocks. Pairing narrows the interval: the unpaired Welch interval on tables was about
1.6 times as wide in the NLHE run (10.4 against 6.7 points).

## Commands

| command | what it does |
|---|---|
| `make lab-conformance` | conformance of the arena engine for all variants |
| `make lab-report` | run the `lab-smoke` A/B and write its report |
| `make lab-experiment LAB_SPEC=<id or path>` | run any spec |
| `make lab-null` | real A/A null calibration on NLHE (`null-aa-nlhe`) |
| `make lab-null-variants` | the same across NLHE, short deck and PLO4 |
| `make lab-null-synthetic` | 4000-replicate synthetic null calibration |
| `make verify-strategy` | `pytest tests/strategy` (also part of `make verify-fast`) |

More commands are available directly: `python -m strategy.cli rules --ruleset uncertainty_plo4`,
`sample-size --sigma 9.2 --delta 2` (or `--pilot <run dir>`), `report <run dir>`,
`experiment <spec> --dry-run`, and `specs`.
