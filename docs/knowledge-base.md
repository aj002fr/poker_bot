# Strategy knowledge base

`strategy/knowledge/` turns the reading digests (Mathematics of Poker, Harrington on
Hold'em, the GTO Wizard notes, the CFR papers and the MIT 15.S50 notes) into testable
records. A record is a hypothesis about what makes the bot score more tournament points.
Each one links to an experiment spec that can confirm or reject it, or to an invariant
test spec if it is a correctness property. Book claims are only starting points. A
record's status changes only when an experiment result artifact supports the change.

Everything here is our own paraphrase. Records cite a chapter or page location and say
how the source's setting differs from ours, which is an eight-bot points SNG with a hand
cap. They never reproduce book text, charts or tables.

The knowledge base sits above the strategy lab described in
[strategy-evaluation.md](strategy-evaluation.md). Records say what to test, and the lab's
runnable experiment specs and manifests produce the evidence.

`strategy/math/` holds the exact arithmetic that the records and tests rely on. It uses
`Fraction` arithmetic and has property tests. `strategy.math.odds` uses only the
standard library. `strategy.math.points` calls the arena ICM scorer, so under the lab's
decision-time boundary it is offline code, used for training labels, tables and tests,
not inside a deployed bot.

## Counts

| Item | Count |
|---|---:|
| Digest hypotheses indexed (`digest_index.yaml`), all cited by at least one record | 148 |
| Strategy and method records (`H001`–`H072`) | 72 |
| Invariant records (`I001`–`I024`) | 24 |
| Experiment specs (`E001`–`E039`) | 39 |
| Invariant test specs (`T001`–`T024`) | 24 (13 implemented, 11 specified) |

The 148 digest hypotheses were deduplicated by mechanism. Several books often describe
the same lever, for example "call all-ins against the point threshold, not chip pot
odds", and those become a single record that cites every source.

## Layout

```
strategy/
  math/
    odds.py          pot odds, call EV, MDF, bluff break-even and ratios, semi-bluffs,
                     multi-street bluff EV, geometric sizing
    points.py        point-threshold call (q* = (U_fold - U_lose) / (U_win - U_lose)),
                     call point gain, ICM risk premium, all-in call under ICM
  knowledge/
    vocabulary.yaml  features (with an `observable` flag) and metrics (with units)
    sources.yaml     digest files and the source registry with default transfer caveats
    digest_index.yaml  every digest hypothesis key (e.g. mop/H06) with a paraphrased title
                     and location
    records/         H###-slug.yaml and I###-slug.yaml
    experiment_specs/  E###-slug.yaml (schema knowledge.experiment/v1)
    test_specs/      T###-slug.yaml (schema knowledge.invariant-test/v1)
    schema.py        field lists, enums and per-document validators
    registry.py      loader and the `parameter_default` accessor
    lint.py          cross-document lint (python -m strategy.knowledge.lint)
    triage.py        scoring and the generated index (python -m strategy.knowledge.triage)
    index.md, index.yaml  generated ranked index; do not edit by hand
tests/strategy/
  math/              property tests for odds, points and the arena ICM scorer
  knowledge/         lint on the real repository plus mutation tests for every rule
```

## Running it

```bash
uv run python -m strategy.knowledge.lint            # exit 1 with one line per error
uv run python -m strategy.knowledge.triage          # print the top 15
uv run python -m strategy.knowledge.triage --write  # regenerate index.md and index.yaml
uv run python -m strategy.knowledge.triage --check  # fail if the index is stale
make verify-strategy                                # all fast strategy tests, solver tests included
```

`make verify-fast` runs the knowledge tests. A record or spec edit that leaves the index
stale fails `test_triage_index_is_current`, so run `--write` after changing triage scores or
experiment membership.

PyYAML is used by the loader. It is already installed as a transitive dependency of the
locked environment, so `pyproject.toml` is unchanged. Promote it to a direct dependency
if that ever stops being true.

## Record schema

Every field is required. Use `null` or `[]` when a field does not apply, so a missing
field is always a mistake.

| Field | Meaning |
|---|---|
| `id`, `title`, `kind` | Stable id (`H###` for `strategy` or `method`, `I###` for `invariant`); the filename starts with the id |
| `status` | `proposed`, `screening`, `supported`, `rejected`, `inconclusive`, `superseded` (the concluded statuses `supported`, `rejected` and `inconclusive` need a result artifact) |
| `cluster` | Mechanism family used for deduplication and reporting |
| `sources[]` | `source`, `location`, `digest_records` (keys in `digest_index.yaml`) or `digest_sections`, `transfer_caveat` |
| `situation`, `eligible_opportunity` | When the record applies, and the exact denominator for measuring it |
| `observable_inputs` | Vocabulary features that are observable at decision time |
| `training_signals` | Vocabulary features used only as labels or targets (hindsight is allowed here and nowhere else) |
| `intervention`, `baseline` | The change and the arm it is compared with |
| `mechanism`, `exceptions` | Why it should work, and when it should not |
| `variant_scope` | `transfer` (`generic`, `family`, `variant_specific`) and `variants` (`generic` means exactly `[all]`) |
| `source_numbers[]` | Each number taken from a source, labelled with its `use` |
| `parameters[]` | Tunable values with a `basis`, a `default` and optionally a `sweep` or `derivation` |
| `experiment` | Link: `id`, `primary_metric`, `minimum_effect`, `partitions`, `result_artifact` |
| `triage` | Scores for the ranked index (invariants carry `{mandatory: true, rationale}` instead) |
| `owner` | Plan workstream that would run or implement it (e.g. `M4`, `M3-B`) |
| `implementation` | `module` and `version_hash` once code exists |
| `decision_log[]` | Dated status changes; the last entry's status equals `status` |
| `related` | Ids of related records |

### Source numbers never become defaults by accident

Books give numbers for settings that are not ours: chip EV, nine-handed live games and
no hand cap. Each source number is labelled with one of these uses:

- `analytic`: a derivation that holds in our setting too (e.g. pot-odds identities).
- `sweep_bound`: used only to bound a sweep.
- `initialization_candidate`: may seed a learner, but is not a default.
- `illustration`: context only.
- `rejected`: known not to transfer.

Parameters declare how their default is justified:

- `neutral`: the default must be a no-op value (`0`, `1`, `false`, `"off"`, `"baseline"`).
- `analytic`: the default needs a `derivation`.
- `measured`: the default is used only after the record is `supported` and has a result
  artifact.
- `sweep_only`: the default must be `null` and a `sweep` is required.

Lint rejects an `analytic` default that equals a source number labelled anything other
than `analytic`, with the message "unsupported source numbers cannot become defaults".
At runtime, `registry.parameter_default(record, name)` returns only neutral, analytic or
supported-and-measured defaults. In every other case it raises `UnsupportedDefaultError`.
A test checks that no source number in the repository resolves to a default today.

## Lint rules

`lint.py` checks the following, and every rule has a mutation test in
`tests/strategy/knowledge/test_knowledge_base.py`:

- Each document parses. Records, experiment specs and test specs pass their schema
  validators. Ids are unique, and filenames start with the id.
- Every source and digest key a record cites exists. Every digest key in
  `digest_index.yaml` is cited by at least one record, so no hypothesis is silently
  dropped.
- Observable inputs are vocabulary features flagged `observable: true`. Training signals
  are vocabulary features.
- Every strategy or method record links to an experiment spec, and every invariant links
  to a test spec. The spec lists the record back. The record's `primary_metric`,
  `minimum_effect` and `partitions` match the spec.
- The final holdout is always `excluded` and is never used for fitting, screening or
  confirming.
- Experiments have at least two arms and exactly one control. Arm toggles take declared
  levels. Factorial specs have at least two factors, each with records it tests. Every
  spec has train and evaluate lineups, a primary variant, a seed namespace, a paired
  design with a sample-size rule, a CPU budget and promote/reject/execution gates.
- Spec metrics are vocabulary metrics, and the minimum-effect unit equals the metric's
  unit.
- An `implemented` test spec names its tests as `path::test_name`, those functions exist,
  and the record names its implementation module.
- Every runnable spec under `run_specs` matches its design (see below).
- The index is current (`index.md is stale; run …` otherwise).

## Experiment specs

Each spec (`knowledge.experiment/v1`) contains:

- `question` and `kind` (`ab_test`, `factorial`, `sweep`, `ablation`,
  `oracle_comparison`, `null_simulation` or `rehearsal`);
- the arms, the toggles they set, and any factors;
- the train and evaluate lineups and the variants (NLHE primary, with short deck and PLO
  as rehearsal where the record transfers);
- seeds (a `knowledge/E###` namespace, full seat rotation and coupled random streams);
- the paired-block design (N ≈ 7.84σ²/δ² from a 100-block pilot, with Holm correction
  across the discovery screen);
- the primary metric and minimum effect, secondary metrics, slices and guardrails;
- the partitions, a CPU budget and gates;
- `run_specs`, `blocked_by`, `status` and `result_artifact`.

Minimum effects are labelled `provisional` (the plan has not fixed the points minimum
effect yet), `plan_gate` (taken from a plan acceptance gate) or `analytic`. The points
default is 0.004 pool fraction per SNG, provisional. Update that one value when the plan
fixes it.

### Design specs and runnable specs

A knowledge spec is a research design. It can have several arms or factors, and it
records what has to exist before it can run. The runnable unit is a
`strategy.experiments` spec (JSON, see [strategy-evaluation.md](strategy-evaluation.md)).
Each runnable spec is one A/B pair of concrete policy artifacts with a seed, a block
count and an absolute `min_effect` in points. That spec materializes an immutable
manifest. A design is executed by listing its runnable specs under `run_specs`, either as
ids in `strategy/experiments/specs/` or as repository paths. A factorial design usually
needs one runnable spec per comparison. Lint checks each link:

- The runnable spec loads and validates.
- Its `hypothesis_id` is the experiment id or one of the experiment's records.
- It measures the same primary metric. The knowledge metric `mean_point_difference` is
  the same quantity as the runnable `mean_point_difference`, expressed as a fraction of
  the point pool.
- Its kind is `ab`, or `null_calibration` for `null_simulation` designs.
- It plays only declared variants: `nlhe` is `practice_nlhe`, `short_deck` is
  `uncertainty_shortdeck` and `plo` is `uncertainty_plo4`.
- It never uses the final holdout.
- Its `min_effect` is at least the declared minimum effect times that table's point
  pool. With the default 50/30/20 payouts the pool is 100 points, so the provisional
  0.004 becomes 0.4 points per SNG.

A design marked `ready`, `running` or `complete` must list at least one runnable spec.
When a run concludes, its `report.json` under
`artifacts/lab/<experiment_id>/<manifest_id>/` becomes the record's
`experiment.result_artifact`. No design lists a runnable spec today. Every one is waiting
on policies that do not exist yet (the learners, solvers and feature layers named in its
`blocked_by`).

## Triage

Every strategy or method record gets seven benefit scores and three penalty scores, each
from 0 to 3:

- Benefits: objective fit, reach, expected effect, transfer, readiness, statistical
  power, and risk reduction.
- Penalties: rule uncertainty, leakage risk, and complexity.

A record's score is the sum of its benefits minus its penalties. An experiment's priority
is the best score among its records. Ties go to the higher mean score, then to the
experiment with more records, then to the lower id. Invariants are mandatory tests and
are not ranked.

### Top 15

| Rank | Experiment | Priority | Plan rank | Verdict |
|---:|---|---:|---:|---|
| 1 | E010 Paired-block promotion vs unpaired and hand-level promotion | 20 | 5 | confirmed |
| 2 | E001 Terminal points vs chip-EV training | 17 | 1 | confirmed |
| 3 | E003 Point-aware all-in calls and bystander utility | 17 | 3 | confirmed |
| 4 | E009 Opponent-specific effective stacks and side-pot features | 16 | 4 | confirmed |
| 5 | E007 Point-aware push and call tables vs fixed charts | 15 | 7 | confirmed |
| 6 | E008 Joint multiway caller subsets vs heads-up multipliers | 15 | 8 | confirmed |
| 7 | E013 Frozen-checkpoint league vs latest-only mirror self-play | 15 | 11 | confirmed |
| 8 | E016 Trajectory and counterfactual labels | 15 | 14 | confirmed |
| 9 | E004 Tournament clock: cap horizon and blind transitions | 15 | 2 | confirmed |
| 10 | E012 Robust vs raw vs confidence-capped exploit vs restricted response | 15 | 10 | confirmed |
| 11 | E011 Contextual Bayesian beliefs and likelihood floors | 14 | 9 | confirmed |
| 12 | E002 Learned continuation value vs ICM-delta shaping | 14 | | added |
| 13 | E020 Gap: separate first-in and calling thresholds | 14 | | added |
| 14 | E017 Distilled, byte-quantized student vs float teacher | 14 | 15 | confirmed |
| 15 | E005 Short-stack urgency: forced-bet pressure and soft gates | 14 | | added |

Twelve of the plan's fifteen experiments are confirmed. E010 ranks first because it is
the evaluation method every other promotion depends on, and it is ready now.

Three experiments are added:

- E002 makes the points objective usable mid-tournament.
- E020 covers a very common preflop situation and is cheap to parameterise.
- E005 matters a lot in a fast-blind SNG and is cheap to build.

Three plan experiments drop out of the top 15. None of them lost value; each is blocked
by infrastructure that does not exist yet, so its readiness score is low:

- E006 (fold/jam vs full preflop menu at medium-short depth) needs the M4 push/fold solver
  and confirmed legal sizes. It now ranks 23rd.
- E014 (blocker-aware distributional buckets) needs the M5 blueprint trainer and known
  variant rules. It now ranks 24th.
- E015 (small adaptive action menu with off-tree translation) needs the M5 blueprint
  trainer and the T023 translation tests. It now ranks 25th.

They move back up once those blockers land. `strategy/knowledge/index.md` has the full
ranking and the per-record scores.

## Invariant test specs

Invariants are correctness properties that must hold before any experiment result can be
trusted. Each test spec lists its cases (`given` and `expect`), the properties to check,
and at least one mutation the tests must catch.

| Spec | Property | Owner | Status | Test location |
|---|---|---|---|---|
| T001 | Pot odds use only contestable chips | M2-C | implemented | `tests/strategy/math/test_odds.py` |
| T002 | Bluff break-even, MDF and bluff-ratio identities | M2-C | implemented | `tests/strategy/math/test_odds.py` |
| T003 | Multi-street bluff EV accounting | M2-C | implemented | `tests/strategy/math/test_odds.py` |
| T004 | Point call threshold and ICM risk premium | M2-C | implemented | `tests/strategy/math/test_points.py` |
| T005 | ICM scorer and cap allocation invariants | M1-A | implemented | `tests/strategy/math/test_icm_invariants.py` |
| T006 | Point conservation and symmetric copies | M1-C | specified | `tests/strategy/evaluation/test_point_conservation.py` |
| T007 | Half-street toy games reproduce known solutions | M2-A | implemented | `tests/strategy/test_fixture_gates.py`, `test_mutations.py` |
| T008 | Jam-or-fold toy games reproduce known solutions | M2-A | implemented | `tests/strategy/test_fixture_gates.py`, `test_mutations.py` |
| T009 | AKQ games reproduce known solutions | M2-A | implemented | `tests/strategy/test_fixture_gates.py`, `test_reach_and_card_removal.py` |
| T010 | Full-street and multistreet toy games | M2-A | implemented | `tests/strategy/test_fixture_gates.py`, `test_mutations.py` |
| T011 | Three-player dry side-pot thresholds | M2-A | implemented | `tests/strategy/test_fixture_gates.py`, `test_mutations.py` |
| T012 | CFR implementations match OpenSpiel oracles | M2-B | implemented | `tests/strategy/test_openspiel_oracle.py`, `test_regret_matching.py`, `test_sampling.py` |
| T013 | Reach weighting and policy averaging | M2-B | implemented | `tests/strategy/test_averaging.py`, `test_reach_and_card_removal.py`, `test_mutations.py` |
| T014 | Sampled and full traversal agree on tiny games | M2-B | implemented | `tests/strategy/test_sampling.py`, `test_fixture_gates.py` |
| T015 | Decision-time rows contain only information-set data | M3-A | specified | `tests/strategy/features/test_no_leakage.py` |
| T016 | PHH Pluribus import reproduces published counts | M7 | specified | `tests/strategy/data/test_phh_import.py` (release marker) |
| T017 | Mixed sampling and byte tables are exact | M8 | specified | `tests/strategy/runtime/test_byte_tables.py` |
| T018 | Runtime always returns a legal action in constant time | M3-D | specified | `tests/strategy/runtime/test_legality_latency.py` |
| T019 | Variant tables are gated by the rules hash | M9 | specified | `tests/strategy/runtime/test_rules_gate.py` |
| T020 | Rescaled equivalent states map to the same features | M3-A | specified | `tests/strategy/features/test_rescaling.py` |
| T021 | Opponent rates use exact opportunity denominators | M3-A | specified | `tests/strategy/beliefs/test_opportunities.py` |
| T022 | Paired evaluation keeps streams coupled; null A/B is honest | M1-C | specified | `tests/strategy/evaluation/test_pairing.py` |
| T023 | Off-tree translation is deterministic and boundary-safe | M5 | specified | `tests/strategy/abstraction/test_translation.py` |
| T024 | Side-pot and effective-stack features match the engine | M3-A | specified | `tests/strategy/features/test_side_pots.py` |

The Stage 2 solver work (`strategy.games` and `strategy.solve`) covers T007–T014. Their
fixture registry reproduces each toy-game case in these specs, and `test_mutations.py`
breaks the solver in the ways these specs list.

The feature and belief layers may already cover parts of T021 and T024
(`tests/strategy/test_features_state.py` checks opportunity context, side pots and
coverage). They stay `specified` until someone checks their cases against those tests.

When you implement a specified test, follow these steps:

1. Write the tests at the proposed location, or at another location if you update the
   spec.
2. Set `status: implemented` and list each test as `path::test_name` under
   `implemented_by`.
3. Set the record's `implementation.module`.

Lint then checks that the named tests exist.

## Changing a record

- To add evidence, run the linked experiment and store its result artifact. Then set
  `experiment.result_artifact`, append a `decision_log` entry and change `status`.
- To merge or split records, mark the old record `superseded` with `superseded_by`, and
  keep every digest key cited.
- After changing triage scores or experiment membership, run
  `python -m strategy.knowledge.triage --write` and commit the regenerated index.
