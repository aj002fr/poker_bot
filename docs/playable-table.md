# Playable table (milestone 3b)

A single-process, in-memory server (`arena/api/`) on the authoritative engine, and a
React table (`web/`). One process and one lock mean one writer per table. Nothing is
persisted: restarting the server clears every table. Persistence, receipts at rest
and recovery come in milestone 5 under the same API.

```bash
make setup   # uv sync + npm ci in web/
make dev     # API on 127.0.0.1:8317, table on http://127.0.0.1:5317
```

The browser talks to the API through the Vite proxy at `/api`.

## Endpoints (spec 9.1 subset)

| Endpoint | Credential | Behaviour |
| --- | --- | --- |
| `GET /catalog` | none | Rulesets and in-process bot policies with artifact hashes |
| `POST /configs/resolve` | none | Validates a setup and returns the resolved config and its hash, without creating anything |
| `POST /tables` | none | `{config, seats: [{kind: human}, {kind: bot, policy}], seed?}`. Returns `table_id` and a private **host** credential. `seed` is optional, private and never returned. |
| `POST /tables/{id}/join` | none | Claims an open human seat (`{seat?}`) before start. Returns a private **seat** credential, once. |
| `POST /tables/{id}/start` | host | Freezes the table. 409 if a human seat is unclaimed. |
| `GET /tables/{id}` | seat or none | The caller's snapshot: seat view for a seat credential, spectator view otherwise |
| `POST /tables/{id}/actions` | seat | `ActionEnvelope`. The acting seat comes from the credential only. |
| `GET /tables/{id}/history` | seat, none, or host + `?perspective=reviewer` | `HistoryView` for the caller. The reviewer view needs the host credential and a completed table. |
| `WS /tables/{id}/stream` | first message | Send `{"type":"auth","token":...}` (or `null` for spectator) first. Then `snapshot` frames arrive, one per commit, each with a larger `version`. |

Credentials travel in `Authorization: Bearer ...`, or in the first WebSocket message.
They never go in URLs. A credential for another table gets 403, and an unknown one
gets 401. The WebSocket rejects a browser `Origin` outside the allowlist
(`localhost:5317`, `127.0.0.1:5317`, plus `ARENA_ALLOWED_ORIGINS`).

### Snapshot

`{table_id, status, version, viewer, setup, view, turn, last_hand, results}`

- `view`: `table_view(table, viewer)` from `arena.table.projections`.
- `turn`: `{seat, kind, decision_id, turn_ms, remaining_ms}`. `remaining_ms` is
  computed from the server's deadline when the snapshot is built.
- `last_hand`: the previous completed hand's events, run through `project_events` for
  this viewer. The engine starts the next hand right after settling, so this is how a
  showdown stays visible.
- `results`: placements, points, payouts, eliminations and per-seat fallback counts,
  built from public events.

### Errors

The body is `{"error": {"code", "message", "details?"}}`. Validation errors list field
locations and messages, and never echo the submitted input.

| Status | Codes |
| --- | --- |
| 401 | `invalid_credential`, `seat_credential_required` |
| 403 | `projection_denied` (`ProjectionDenied`), `credential_wrong_table`, `host_credential_required`, `reviewer_permission_required` |
| 404 | `table_not_found` |
| 409 | `out_of_turn`, `stale_decision`, `stale_version`, `wrong_hand`, `table_complete`, `deadline_passed`, `action_id_conflict`, `seats_unclaimed`, `already_started`, `not_started`, `seat_unavailable`, `no_open_seat` |
| 422 | `invalid_schema`, `illegal_action`, `table_mismatch`, `invalid_config`, `invalid_seat`, `invalid_request` |

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

- Bots run in-process (`arena/api/bots.py`): `legal_random`, `check_or_fold` and
  `check_or_call`. They get only the acting seat's `Observation` and use one
  `policy-seat-N` RNG stream per seat.
- A bot exception or an illegal action consumes the decision once with
  `bot_error` / `invalid_action`.
- The artifact hash is the SHA-256 of the policy source.
- The agent SDK and isolated workers replace these in milestone 4.
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
- Ambiguous network failures are retried with the same `action_id`.
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
- `web/e2e/happy-path.spec.ts` (`make verify-web`) has two humans (1440 px and
  360 px) and four bots finish a six-seat table.
  - It checks the label, own cards and backs, the raise label and the results.
  - It checks for horizontal overflow at 360 px.
  - A DOM leak check compares every rendered card with the reviewer log.
