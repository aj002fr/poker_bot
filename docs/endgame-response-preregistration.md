# Pre-registration: push/fold against population response curves

Written and committed before any validation run of the specs below. The runs use the
validation partition of `strategy.population`; the holdout partition is not used.

## Question

Do push/fold tables that best-respond, in points, to the population's measured
responses score more tournament points than (a) the point-aware equilibrium tables and
(b) the chip-EV tables, in NLHE and in short deck?

## What is fixed before the runs

- Curves: `strategy.endgame.response`, fitted only on train-partition tournaments
  (200 lineups of seven training members and one point-aware bot, every seat rotation,
  seed 11). The functional form per decision kind was chosen by held-out log loss on
  train-partition lineups (30% held out): class-offset logistic for open, call and
  overcall in both variants. The beliefs soft-threshold form
  (`strategy.beliefs.ranges.threshold_likelihood`) was a candidate and lost.
- Tables: `strategy.tables.exploit` over the same 6000 states as the point-aware
  tables. `response`: exact best response to the curves. `capped`: best response to a
  0.5/0.5 mixture of the curves and the state's point-aware CFR+ equilibrium.
- Bots: `strategy/endgame/bots/{response,capped}.py`. They share the base policy and
  layer limits with the control arms (opens at most 15 BB, 10 BB from the small blind,
  calls at most 30 BB). Only the tables differ.

Table files the runs use (SHA-256 prefix of the `.npz`; experiment manifests pin the
bot files, not the tables). This list was written before the runs but missed the
pre-registration commit `c30fd99`, so it was committed afterwards. The table files
themselves are in `c30fd99` and their hashes match. `c30fd99` (committed
2026-11-02 00:32 UTC, pushed before the first run at 00:32:40) is kept as the tag
`prereg-endgame-response`; rebasing onto main turned it into `ea66c63`, with the same
code, tables and specs.

| table | sha256 |
| --- | --- |
| `response-practice_nlhe` | `c1cf450c586e3000` |
| `response-uncertainty_shortdeck` | `dc3b068e75894f5d` |
| `endgame-response-practice_nlhe` | `ee90b4c824b2f5d0` |
| `endgame-response-uncertainty_shortdeck` | `9efe20a0a5aee577` |
| `endgame-capped-practice_nlhe` | `9daa1f360cb269af` |
| `endgame-capped-uncertainty_shortdeck` | `3004aa7c36e6d9fe` |
| `endgame-points-practice_nlhe` | `86ca270d46f5b0b0` |
| `endgame-points-uncertainty_shortdeck` | `2f955c0888311997` |
| `endgame-chips-practice_nlhe` | `1f39bc820f7cecc2` |
| `endgame-chips-uncertainty_shortdeck` | `b0062a7909f12b03` |

## Primary analysis

Eight co-primary paired comparisons, arm B minus arm A, in mean terminal points per
candidate seat:

| spec | B | A | variant |
| --- | --- | --- | --- |
| `endgame-response-vs-points` | response | points | NLHE |
| `endgame-response-vs-chips` | response | chips | NLHE |
| `endgame-capped-vs-points` | capped | points | NLHE |
| `endgame-capped-vs-chips` | capped | chips | NLHE |
| `endgame-*-shortdeck` (four) | same | same | short deck |

Each spec runs 16 validation lineups × 25 blocks = 400 paired blocks (3200 tables per
arm), with 8-player tables of 2000 chips and a 200-hand cap. The test is a two-sided
paired t-test on block differences. The eight p-values are Holm-corrected at family-wise
α = 0.05. A comparison is a win only if its Holm-adjusted p < 0.05 and the difference is
positive. A significant negative difference is a loss. Anything else is a null. Each
spec declares `min_effect` 0, so the runner's own verdict is the sign test with the
interval excluding zero.

## Secondary (descriptive, no claims)

- The in-model response audit (`python -m strategy.endgame.gates --only response`): on
  held-out synthetic states, one player plays each table while the others play the
  curves or the point equilibrium.
- Action-rate changes, finish distributions and slices by opponent family from the
  experiment reports.

## Reporting

All eight results are reported, including nulls and losses, with intervals and the
table data hashes used. No spec is re-run with new seeds to change a result. If a
result prompts a change, the changed tables are a new experiment with new specs.

## Results (added after the runs)

| spec | B − A (points) | 95% paired t | p | Holm p | outcome |
| --- | --- | --- | --- | --- | --- |
| `endgame-response-vs-points` | +0.954 | [0.408, 1.501] | 0.0007 | 0.004 | win |
| `endgame-response-vs-chips` | +0.543 | [−0.031, 1.117] | 0.064 | 0.19 | null |
| `endgame-capped-vs-points` | +0.964 | [0.472, 1.456] | 0.0001 | 0.001 | win |
| `endgame-capped-vs-chips` | +1.176 | [0.643, 1.708] | 0.00002 | 0.0001 | win |
| `endgame-response-vs-points-shortdeck` | +0.441 | [−0.047, 0.929] | 0.076 | 0.19 | null |
| `endgame-response-vs-chips-shortdeck` | +0.127 | [−0.337, 0.592] | 0.59 | 0.59 | null |
| `endgame-capped-vs-points-shortdeck` | +0.615 | [0.186, 1.045] | 0.005 | 0.026 | win |
| `endgame-capped-vs-chips-shortdeck` | +0.460 | [0.015, 0.904] | 0.043 | 0.17 | null |

Holm uses `strategy.eval.stats.holm`. The runner's own verdict ("pass" when the interval
excludes zero) is uncorrected: it says pass for the short-deck capped-vs-chips run, which
is not significant after correction.
