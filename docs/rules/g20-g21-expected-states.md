# G20 / G21 expected states (milestone 1 exit evidence)

Every row below is asserted by a JSON fixture in `tests/golden/fixtures/core/`, and each
fixture hand is also re-settled by the independent pot-ledger oracle in `tests/oracles/`.
"PokerKit alone" is what PokerKit 0.7.7 does without the adapter's corrections.

## G20: short big blind (blinds 10/20, seat 2 posts 7 all-in)

Seats: 0 = button (A, 2000), 1 = SB (B, 2000), 2 = BB (C, 7). Preflop, A acts first.

| Decision point | PokerKit alone | Adapter (spec) |
| --- | --- | --- |
| A to act: call amount | 10 | **20** |
| A to act: minimum full raise-to | 30 | **40** |
| A to act: maximum raise-to (no limit) | 2000 | 2000 |
| After A calls, B to act: call / minimum raise-to | 0 (check) / 30 | **10 / 40** |
| Flop: hand committed, pot | 10/10/7, 27 | **20/20/7, 47** |
| Pots | 21 (A,B,C) + 6 (A,B) | **21 (A,B,C) + 26 (A,B)** |

The same short blind in the other G20 fixtures:

| Fixture | Line | Expected |
| --- | --- | --- |
| G20b | A raises to 40 | B: call 30, minimum full raise-to **60** (an increment of 20 over the nominal 20) |
| G20c | A folds | B owes nothing more than C's 7 and nobody can respond. B gets an auto-check (`no_contestable_action`) and **3** back (`ExcessReturned`). Pot 14. |
| G20-plo | Pot limit, A to act | Call 20, minimum 40, pot cap **57** = 20 + (10 + 7) + 20. After A calls, the SB cap is **67**. No all-in above the cap. |
| G20d | BB 1 of nominal 100, dead SB; seat 7 calls all-in 17 | Seat 1 can still call **100**. The button's minimum full raise-to is **200**. Pots 4 / 48 / 244. |

## G21: short all-in bet after a check (blinds 50/100)

Seats: 0 = SB (A, 2000), 1 = BB (B, 140), 2 = button (C, 2000). Everyone limps to the
flop at 100. On the flop A checks, B bets all-in for 40 (a full bet would be 100), and C acts.

| Fixture | `short_bet_reopens_checker` | C's action | A's options afterwards |
| --- | --- | --- | --- |
| G21a | `false` (default) | call 40 | call 40 or fold; **no raise** (`raise_blocked_reason` set) |
| G21b | `false` | raise to 140 (full) | call 140, fold, or raise to at least **240** |
| G21c | `true` | call 40 | call 40, fold, or raise to at least **140** (PokerKit's native behaviour) |
| G21d | `true` | raise to 140 | call 140, fold, or raise to at least **240** |

In every case B's short all-in bet is offered as `short_all_in_raise_to = 40` with
`min_full_raise_to = null`, and C's minimum full raise-to is 140.

## G09b: the preflop equivalent

Blinds 10/20, seat 3 (UTG) limps, and the button shoves 30, which is less than a full raise to 40.
The SB and BB may call or raise to at least 50. When the action returns to the limper,
they may only call or fold. PokerKit alone would let the limper re-raise
(discrepancy D3 in `docs/assumptions.md`).
