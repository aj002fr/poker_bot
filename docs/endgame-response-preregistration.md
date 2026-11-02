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
