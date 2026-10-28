# Practice assumptions and engine discrepancies

Everything in this engine is labelled `compatibility_status: practice_assumptions`.
It reproduces the rules in `sequoia-pokerbot-spec.md` (with the approved build-plan
amendments). It is not a statement of the official Sequoia rules.

PokerKit **0.7.7** (MIT) is the rules dependency. It is pinned in `pyproject.toml` and
`uv.lock`, and `arena.rules.pokerkit_adapter.POKERKIT_VERSION` records the version. The
adapter is authoritative: it uses PokerKit for dealing, betting mechanics and hand
evaluation, and corrects PokerKit wherever the spec says otherwise (listed below).
The corrections can only make PokerKit's legal-action set *narrower*. If PokerKit refuses
an action the adapter offers, or its chip pushes differ from the adapter's settlement by
more than odd-chip placement, the hand raises `EngineDiscrepancy` rather than guessing.

## Local rule switches and fixed choices

| Setting | Default | Where | Covered by |
| --- | --- | --- | --- |
| `local_rules.short_bet_reopens_checker` | `false` | `TournamentConfig.local_rules` | G21a/b (false), G21c/d (true) |
| `short_deck_ranking` | `pokerkit_native` (the only value) | `RuleProfile` | E07, E08a/b |
| Call against a short big blind is priced at the nominal BB | always | adapter | G20, G20b, G20c, G20d, G20-plo |
| Pot-limit cap prices the call at the nominal BB | always | `_ArenaState` | G20-plo |
| Short all-in raise does not reopen players who already acted, preflop included | always | betting-rights overlay | G09, G09b |
| Auto-check when the actor owes nothing and nobody else can respond | always | adapter (`auto: "no_contestable_action"`) | G20c |
| One pot per contribution level, never merged | always | adapter settlement | G12, G15, G22 |
| Odd chips one each, clockwise from the first seat after the button | always | adapter settlement | G15, G22, G22b |
| Unmatched excess returned when bets are collected | always (PokerKit) | `ExcessReturned` event | G20c, G13 |
| Same-hand busts ranked by start-of-hand stack; equal start stacks share positions | always | `Standings` | T09, T10 |
| Hand-cap finish by Malmuth-Harville ICM, tolerance `1e-9` | always | `arena.tournament.scoring` | T04–T08, randomized vs oracle |
| Dead button, one runout, no ante, no rake | fixed literals | `TournamentConfig` | G04, G05 family |

Visibility: on a fold win no hole cards are revealed. At any showdown, all-in or river,
every live hand is revealed (`CardsRevealed`) before the remaining board is dealt.

### Short-deck ranking

PokerKit's `ShortDeckHoldemLookup` order, from weakest to strongest, is: high card, pair,
two pair, trips, **straight**, full house, **flush**, quads, straight flush. A straight
beats trips and a flush beats a full house. This is what the spec fixtures expect. It is
not the Triton order the build plan guessed (trips over straight). The profile setting
`short_deck_ranking` exists so a different order can be added later. Today it accepts
only `pokerkit_native`, and nothing in the engine silently re-ranks hands.
`A-6-7-8-9` is the lowest short-deck straight (E08a/b).

## Discrepancy register (PokerKit 0.7.7 versus the spec)

| Id | PokerKit behaviour | Spec / adapter behaviour | Fixture |
| --- | --- | --- | --- |
| D1 | When the BB is posted short (7 of 20), the call is priced at the largest posted blind (10) and the minimum raise is 30. In pot limit, the cap uses the posted level too. | The call is priced at the nominal BB (20) and the minimum full raise-to is 40. While the preflop level is below the nominal BB, the adapter turns a canonical `call` into a PokerKit completion to the nominal amount, lowers PokerKit's minimum completion for that window, and resets PokerKit's raise tracking after each such action. The pot cap is priced at 20 (57 in G20-plo). | G20, G20b, G20c, G20d, G20-plo |
| D2 | A check, then a short all-in bet, then a call: PokerKit lets the original checker raise. | Default `short_bet_reopens_checker=false` blocks the raise. `true` matches PokerKit. | G21a–d |
| D3 | Each street's "last full raise" starts at 0, so preflop a short all-in raise over the BB reopens a limper. | The BB counts as the opening full increment, so the limper may only call or fold. | G09b |
| D4 | Short-deck order, see above. | Same as the spec. Contrary to the plan's guess; reported, not frozen. | E07, E08 |
| D5 | The whole odd-chip remainder of a split pot goes to the first tied winner (104 three ways is 36/34/34). | One chip each, clockwise from the first seat after the button: 35/35/34. The adapter emits `OddChipCorrection`. | G22b, G15, G22 |
| D6 | Adjacent pot layers with the same eligible set are merged before splitting. | One pot per contribution level, each split on its own. Totals differ only through odd-chip placement. | G15, G22 |

Other observations that needed no correction:
- Eight seats work.
- The PLO pot cap is correct when the blinds are posted in full (E11: 70).
- PokerKit's internal deck uses the global `random` module. The adapter overwrites it
  with the stored, seeded deck and checks every draw against it.
- On a fold win, the winner's uncalled street bet comes back through PokerKit's chip
  pull, not as a collected bet.
- There is no separate "effective stack" pre-cap. Unmatched chips are refunded when bets
  are collected, so pots and final stacks match the spec.

## Not yet covered (later milestones)

- T12 at the agent level (worker isolation, budgets, timeouts) is milestone 4. The
  table-level fallback path is tested now.
- The official-rules adapter is not included. Everything here is practice assumptions.
