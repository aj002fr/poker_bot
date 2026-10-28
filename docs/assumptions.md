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

## History, replay and views (milestone 3)

- Visibility allowlists are in `arena/table/projections.py` and documented in
  `docs/observations-and-projections.md`.
  - A showdown reveals every live hand, and no hand is mucked.
  - Folded hands are never shown to players or spectators.
  - The full private log is shown only to a reviewer after the table ends.
- The display pot breakdown (`TableView.pots`) merges collected levels that have the
  same eligible players, which is easier for people to read. Settlement still uses one
  pot per contribution level (`PotAwarded`).
- An ActionFallback's reason code is public, so tables can count timeouts. The
  rejected payload is private.
- Decision and hand-start state hashes are private.
- PHH exports flag hands that depend on the short-big-blind correction (D1) or the
  odd-chip rule (D5). PokerKit's own PHH reader rejects or re-prices those hands, so
  they are counted as explicit exclusions in the PHH cross-check.
- Replay refuses a history recorded with a different PokerKit version unless
  `--allow-engine-mismatch` is given. Any difference from the stored log halts the
  replay, and nothing is repaired.
- RNG streams are separate: `env` for decks and the first button, and one
  `policy-seat-N` stream per baseline seat. Seeds appear only in private histories
  and `private/seeds.json`. The run manifest holds only their hash.

## Playable table server (milestone 3b)

Local choices in `arena/api/`. The details are in [`playable-table.md`](playable-table.md).

| Choice | Default | Where |
| --- | --- | --- |
| Human deadline | `human_turn_ms` (20,000) from when the decision is scheduled, on a monotonic clock; `received <= deadline` is on time, where `received` is when the full request registered as in flight | `TableManager._schedule`, `register_action`, `_expire` |
| Human timeout | `apply_fallback("timeout")`: check if legal, otherwise fold; the late action gets 409 `deadline_passed` | `TableManager` |
| Bot failure | Exception: `bot_error`. Rejected action: `invalid_action`. One fallback, no retry. | `TableManager._bot_act` |
| Bot pacing | 700 ms per move, plus 1,500 ms before a hand's first bot move. Presentation only. | `ARENA_BOT_DELAY_MS`, `ARENA_HAND_PAUSE_MS` |
| Idempotency scope | Per table, keyed by `action_id`, accepted actions only | `LiveTable.receipts` |
| Starting a table | Host credential, every human seat claimed | `TableManager.start` |
| Unauthenticated reads | Spectator view (public information only) | `Principal.viewer` |
| Reviewer history | Host credential and a completed table; otherwise 403 | `TableManager.history` |
| WebSocket origin | Allowlist; a missing `Origin` (non-browser client) is allowed | `create_app` |
| Persistence | In-memory by default; `ARENA_STORE=postgres` for durable tables (milestone 5, below) | `create_app` |

## Service durability (milestone 5)

Local choices in `arena/persistence/` and the manager. The details are in [`persistence.md`](persistence.md).

| Choice | Default | Where |
| --- | --- | --- |
| Commit unit | One transaction per command (create, join, start, action, timeout, bot move, resume); zero-delay bot moves are separate commands | `TableManager._commit` |
| Writer fence | Compare-and-set on (`owner_id`, `version`) at every commit; the lease only governs claims | `Store.commit` |
| Lease and heartbeat | 15 s lease, renewed by commits and a 1 s heartbeat | `TableManager(lease_s, heartbeat_ms)` |
| Snapshots | One per hand start, taken before the hand is dealt; includes the dealer RNG checkpoint | `Table.resume_point` |
| Recovery source | Newest snapshot plus the commands after it; the whole rebuilt log must equal the stored log | `arena/persistence/recovery.py` |
| Irreproducible record | Halt (503 `table_halted`), private reason stored, never repaired | `TableManager._load` |
| Outage think time | Restored from the last durable heartbeat or commit (≤ 1 s stale, in the human's favour) | `TableManager._resume` |
| Graceful shutdown | Open `shutdown` pause with the exact remaining time; closed on the next start | `TableManager.close` |
| Bot artifact change across a restart | Halt; the checkpointed artifact hash must match the running policy code | `TableManager._load` |
| Multiple API processes on one database | Not supported (multi-writer fencing deferred) | build plan wave 4 |

## Agent runtime (milestone 4)

The SDK and runtime are described in `docs/agent-sdk.md`, and the sandbox in
`docs/agent-sandbox.md`. These are the local choices.

- **The SDK is our own contract**, not Sequoia's Agent API. Bots are Python files with
  `make_bot()`. They are loaded only inside a sandboxed worker, except for the
  built-in baselines and runs explicitly marked `--in-process`.
- **Deadline.** The budget is 1,000,000 ns over the whole parent-observed round trip,
  from just before the observation is serialized to the complete reply.
  `received <= deadline` counts as on time. A wake notice with no game data is sent
  before the clock starts. An on-time reply that is processed late still counts.
- **Fallback.** Check, otherwise fold, applied once with no retry. Late replies are
  discarded.
  - A worker still busy at its seat's next decision gets a 20 ms grace period, then is
    killed and restored from committed state.
  - After 25 restarts in one table, the seat's remaining decisions fall back with
    `worker_unavailable`.
- **Crash or timeout labelling.** A worker that dies mid-decision may be labelled
  `timeout` instead of `worker_crash`, because the pipe closes after the helper reaps
  it. The fallback and the restore are the same either way.
- **Commit rule.** Explicit state commits only with an accepted action. Restores and
  every later decision use the parent's canonical copy.
- **Instance state.** Instance-state bots are allowed. After a restart the table
  continues with a fresh instance, the seat is flagged `restorable: false` and the table
  is counted in `state_lost`. These tables are **not** excluded from means. Excluding
  them would reward crashing.
- **Rulesets.** Bots must declare their rulesets. An unsupported pairing is rejected
  before any table starts (CLI exit code 5). It is never played with a fallback policy.
- **Fresh context per table.** Every table gets fresh workers and a fresh anonymous
  `table_id`. Workers are never reused. `PYTHONHASHSEED=0` is set in workers so set and
  dict iteration is reproducible.
- **Benchmark.**
  - Paired seed blocks with full seat rotation.
  - Intervals are a 95% percentile bootstrap over blocks with 2,000 resamples.
  - Points are the tournament payouts per table (100 per table).
- **Parallel jobs.** The default is `max(1, min(4, cpu_count // 4))` (2 on this VM). With
  3 or more jobs on 8 vCPUs, the 1 ms timeout rate jumps because parent and worker both
  busy-poll. See the sandbox doc.
- **Infrastructure reruns (spec 9.2).** A bot-only table interrupted by infrastructure is
  marked invalid and rerun with the same seed and table ID, under an attempt ID
  `<table_id>-a<n>`. The default is up to 3 attempts in total, recorded in the manifest.
  - What counts as infrastructure: a worker that cannot be started, a host OS or memory
    error, a lost job process, or an infrastructure kill of an instance-state bot.
  - An infrastructure kill of an explicit-state bot is restored exactly and does not
    invalidate the table.
  - Agent crashes keep the normal fallback.
  - Engine errors are listed as failed and not rerun.
  - Every attempt is counted and reported. See `docs/agent-sdk.md`.
- **Diagnostic runs.** `--in-process` and `--budget-ns` runs are labelled
  `competitive: false` in the manifest and in the printed output.
- **The playable table's bots** (`arena/api/`) are the engine's trusted in-process
  policies with presentation pacing. They do not use the sandboxed workers or the 1 ms
  budget.

## Not yet covered (later milestones)

- The official-rules adapter is not included. Everything here is practice assumptions.
- Metamorphic tests from spec 12.2 (suit renaming, seat rotation, chip scaling) belong
  to milestone 7. Information-set tests cover the in-process views. HTTP, WebSocket
  and DOM leak checks cover the playable table (`tests/api/test_leaks.py`,
  `web/e2e/`).
