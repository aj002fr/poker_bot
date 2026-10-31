# Playable table (milestones 3b and 6)

A single-process server (`arena/api/`) on the authoritative engine, and a React table
(`web/`). Each table has one command queue, and therefore one writer. By default tables
live in memory, so restarting the server clears them. With `ARENA_STORE=postgres`
(`make dev-postgres`) every command commits to PostgreSQL, and a restart recovers
each table. The API contract is the same either way; see [`persistence.md`](persistence.md).

```bash
make setup   # uv sync + npm ci in web/
make dev     # API on 127.0.0.1:8317, table on http://127.0.0.1:5317
```

The browser talks to the API through the Vite proxy at `/api`.

## Endpoints (spec 9.1 subset)

| Endpoint | Credential | Behaviour |
| --- | --- | --- |
| `GET /catalog` | none, or operator | Rulesets and bot choices with artifact hashes and supported rulesets. Registered files are listed for the operator only. |
| `POST /configs/resolve` | none, or operator | Validates a setup and returns the resolved config and its hash, without creating anything |
| `POST /tables` | none, or operator | `{config, seats: [{kind: human}, {kind: bot, policy, reveal?}], seed?, styles?}`. Returns `table_id` and a private **host** credential. `seed` is optional, private and never returned. Seating a registered file needs the operator credential; a bot that does not support the ruleset gets 422 `unsupported_ruleset`. `styles` sets the style mode (labelled, hidden, mixed, random), random pool and learning aids ([`opponent-styles.md`](opponent-styles.md)). |
| `POST /tables/{id}/join` | none | Claims an open human seat (`{seat?}`) before start. Returns a private **seat** credential, once. |
| `POST /tables/{id}/start` | host | Freezes the table. 409 if a human seat is unclaimed. |
| `GET /tables/{id}` | seat or none | The caller's snapshot: seat view for a seat credential, spectator view otherwise |
| `POST /tables/{id}/actions` | seat | `ActionEnvelope`. The acting seat comes from the credential only. |
| `GET /tables/{id}/history` | seat, none, or host + `?perspective=reviewer` | `HistoryView` for the caller. The reviewer view needs the host credential and a completed table. |
| `GET /tables/{id}/replay?perspective=&hand=` | seat, none, or host | Replay frames for one hand, re-simulated from the private log: `seat` (own cards and showdowns, during and after the table), `spectator`, or `reviewer` (every hole card; host credential and a completed table only) |
| `GET /styles` | none | Style profiles, table modes, stat definitions, opening charts and the 13×13 hand grid |
| `GET /tables/{id}/stats` | any | Table-reading stats per seat from public events, with every showdown. 409 before start. |
| `GET /tables/{id}/aids` | seat | Learning aids from the seat's own view: position, hand in words, draws, equity against random hands, pot odds and a coach hint on its turn |
| `GET /operator` | operator | 200 when the bearer token is the operator credential |
| `GET /bots`, `POST /bots` | operator | List or register a local bot file (`{path}`, absolute). It is copied, hashed and loaded only in a sandboxed worker. |
| `POST /runs`, `GET /runs`, `GET /runs/{id}` | operator | Launch the milestone 4 benchmark runner in the background over scenario × seat-count cells, list runs, and read one run with per-cell manifests and summaries |
| `WS /tables/{id}/stream` | first message | Send `{"type":"auth","token":...}` (or `null` for spectator) first. Then `snapshot` frames arrive, one per commit, each with a larger `version`. |

The operator credential is `ARENA_OPERATOR_TOKEN`; `make dev` generates one per session,
prints it and writes it to `artifacts/operator-token` (git-ignored).

Credentials travel in `Authorization: Bearer ...`, or in the first WebSocket message.
They never go in URLs. A credential for another table gets 403, and an unknown one
gets 401. The WebSocket rejects a browser `Origin` outside the allowlist
(`localhost:5317`, `127.0.0.1:5317`, plus `ARENA_ALLOWED_ORIGINS`).

### Snapshot

`{table_id, status, version, viewer, setup, view, turn, last_hand, results, stats, reveal}`

- `view`: `table_view(table, viewer)` from `arena.table.projections`.
- `turn`: `{seat, kind, decision_id, turn_ms, remaining_ms}`. `remaining_ms` is
  computed from the server's deadline when the snapshot is built.
- `last_hand`: the previous completed hand's events, run through `project_events` for
  this viewer. The engine starts the next hand right after settling, so this is how a
  showdown stays visible.
- `results`: placements, points, payouts, eliminations and per-seat fallback counts,
  built from public events.
- `setup.styles` and per-seat `hidden`/`style`: hidden seats show only "Player N" until
  the table completes.
- `stats`: table-reading stats per seat, from public events (null before start).
- `reveal`: once complete, every seat's real identity, a lineup hash and the evidence
  for its style. See [`opponent-styles.md`](opponent-styles.md).
- Reviewer replay frames carry `bot_note` (a bot's private reasoning) after completion.

### Errors

The body is `{"error": {"code", "message", "details?"}}`. Validation errors list field
locations and messages, and never echo the submitted input.

| Status | Codes |
| --- | --- |
| 401 | `invalid_credential`, `seat_credential_required`, `operator_required` (no credential) |
| 403 | `projection_denied` (`ProjectionDenied`), `credential_wrong_table`, `host_credential_required`, `reviewer_permission_required`, `operator_required` (wrong credential), `learning_aids_disabled` |
| 404 | `table_not_found`, `hand_not_found`, `run_not_found` |
| 409 | `out_of_turn`, `stale_decision`, `stale_version`, `wrong_hand`, `table_complete`, `deadline_passed`, `action_id_conflict`, `seats_unclaimed`, `already_started`, `not_started`, `seat_unavailable`, `no_open_seat`, `bot_start_failed` |
| 422 | `invalid_schema`, `illegal_action`, `table_mismatch`, `invalid_config`, `invalid_seat`, `invalid_styles`, `invalid_request`, `unsupported_ruleset`, `invalid_perspective`, bot registration and run validation codes |

## Actions, idempotency and deadlines

- Accepted actions are stored by `action_id`.
  - The same id, seat and payload returns the stored receipt with `replayed: true`
    and applies nothing.
  - Any other reuse of the id gets 409 `action_id_conflict`.
  - A second id for a consumed decision gets the engine's `stale_decision` or
    `out_of_turn`.
- A human's deadline is `now + human_turn_ms` on a monotonic nanosecond clock, set
  when the decision is scheduled (`arena/api/clock.py`; tests use `FakeClock`).
- The receive time is taken once the full request body has been read.
  - An action received at or before the deadline (`received <= deadline`) is on time,
    even if it is processed later.
  - The timeout sweep skips a deadline while an on-time request for that table is
    still in flight.
  - A request received after the deadline first applies the timeout and then gets
    409 `deadline_passed`.
- Timeout uses `Table.apply_fallback(seat, "timeout")`: check if legal, otherwise
  fold.

## Seats and bots

- Bot seats use the milestone 4 agent runtime (`arena/api/seats.py`):
  - Baselines (`check_fold`, `calling_station`, `legal_random`, and the NLHE-only
    `nlhe_heuristic`) run in-process. The 3b ids `check_or_fold` and `check_or_call`
    stay accepted as aliases.
  - Opponent styles (`style_rock`, ..., and `style_random`) are trusted repository code
    and also run in-process.
  - Registered files (operator only) run in a sandboxed worker under the table's
    `bot_budget_ns`, with the benchmark runner's fallbacks.
  - Each seat sees only its own `Observation`. Its policy seed derives from the
    table's private bot seed and the seat.
- A late, broken or illegal reply consumes the decision once with the matching
  fallback reason.
- The artifact hash is the SHA-256 of the bot file. Every seat's committed explicit
  state is checkpointed with each commit, so a recovered table resumes each bot exactly.
- Bot moves are paced for people to follow: `ARENA_BOT_DELAY_MS` (default 700) per
  move, plus `ARENA_HAND_PAUSE_MS` (default 1500) before the first bot move of a hand.
  - Pacing is a scheduled time checked by the background pump, never a sleep inside a
    transition.
  - It changes only when a bot acts, never what it does.

## Browser

- The browser holds no poker logic. It renders snapshots and posts envelopes.
- It does not move chips until the server commits.
- It shows "Raise to X · add Y", where Y is X minus the hero's `street_committed`.
- Controls are enabled from `legal`, and `raise_blocked_reason` is shown when
  raising is not possible.
- Ambiguous network failures are retried with the same `action_id`. The pending
  submission is kept in `sessionStorage`, so after a reload or a WebSocket reconnect
  the tab resends it: the stored receipt (`replayed: true`) confirms it committed,
  `deadline_passed` says the timeout came first.
- Every control works from the keyboard: F fold, C check/call, R raise amount (Enter
  sends, Esc leaves), A all in; the replay uses ←/→, Shift+←/→, P and 1–4.
- Cards print rank and suit and carry an accessible name; seat states ("To act",
  "Folded", "Out", "Wins") are words, not colour alone.
- Screens: setup (bots filtered by ruleset), lobby, table, results, replay
  (`#/t/{id}/replay`), batch runs and reports (`#/runs`, operator only).
- Credentials live in `sessionStorage`, one per tab, so two tabs can hold two seats.
- At widths up to 760 px the oval becomes a compact seat grid under the board, and at
  480 px or less the action panel sticks to the bottom.

## Tests

- `tests/api/test_tables_api.py` covers create, join and start validation;
  credential binding; out-of-turn, invalid and illegal actions with no mutation; the
  street-total debit; idempotency; two tabs; deadline boundaries (at, just after, and
  queued on time); check-then-fold timeout fallback; bot pacing; 403 mapping; full
  tables; and WebSocket auth, origin and versions.
- `tests/api/test_leaks.py` plays full tables over HTTP for NLHE at 6 and 2 seats,
  PLO at 3 and short deck at 8.
  - Every human seat and a spectator keep a WebSocket open throughout.
  - Every body, frame, receipt, error and history is checked against per-hand
    entitlements rebuilt from the reviewer log.
  - A canary test checks that the checker catches injected cards.
- `tests/api/test_replay.py` and `tests/api/test_bots_runs.py` cover replay frames
  and perspective permissions, bot seats and ruleset filtering, registered files, the
  operator credential and background runs.
- `make verify-web` runs the Playwright suite on a fake clock (`ARENA_FAKE_CLOCK=1`,
  zero bot delay, `POST /__test__/clock/advance`):
  - `happy-path`: two humans (1440 px and 360 px) and four bots finish a six-seat table.
  - `mixes`: human/bot mixes at N = 2, 3, 4, 5, 6 and 8 across all three rulesets,
    and an 8 → 7 elimination. Each checks every rendered card against the reviewer log.
  - `replay`: perspective permissions during and after the table, stepping, hands,
    speeds and play/pause.
  - `reconnect`: a lost response confirmed by `action_id` after a reload, a lost
    request resent when the stream reconnects, and a resend after the timeout.
  - `keyboard`: setup, play, results and replay without a mouse, and the operator dialog.
  - `layout`: the 8-seat table in each ruleset at 360, 768 and 1440 px with no seat
    overlap and no horizontal overflow, and text-labelled cards.
  - `runs`: operator gating, registered files in setup, and a launched run's report.
