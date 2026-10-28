# Service durability (milestone 5)

The playable table keeps its HTTP, WebSocket and UI contract from milestone 3b and
now sits on a store. `arena/persistence/` holds two implementations of one protocol:

- **`MemoryStore`** is the default. It is used by `make dev` and the fast tests, and lives as long as the process.
- **`PostgresStore`** (psycopg 3) keeps tables across restarts and crashes.

```bash
make db-up           # PostgreSQL 16.4 (pinned digest) on 127.0.0.1:54329 via docker compose
make dev-postgres    # db-up, then the API with ARENA_STORE=postgres and the web table
make verify-postgres # persistence + API suites against Postgres, failing if it is unreachable
make db-down
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARENA_STORE` | `memory` | `memory` or `postgres` |
| `ARENA_DATABASE_URL` | `postgresql://arena:arena@127.0.0.1:54329/arena` | server database |
| `ARENA_WRITER_ID` | `local-writer` | this process's ownership id; keep it stable across restarts |
| `ARENA_TEST_DATABASE_URL` | `…/arena_test` | test database (created by `scripts/pg-init/`) |
| `ARENA_REQUIRE_POSTGRES` | unset | `1` turns "Postgres unreachable" from a skip into a failure |
| `ARENA_API_TEST_STORE` | `memory` | `postgres` runs `tests/api` on the test database |

Migrations (`arena/persistence/migrations/*.sql`) are applied on startup in name
order, each in its own transaction, under an advisory lock, and recorded in
`schema_migrations`.

## Single command writer

Each table has one FIFO command queue in the process (`CommandQueue`). Create,
join, start, human action, timeout fallback, bot decision and post-recovery resume
are each one command and one transaction:

1. The engine applies the command in memory and validates it. A rejection mutates nothing and writes nothing.
2. `CommitBatch` collects everything the command changed:
   - the new private events;
   - hand rows (start, then end and final stacks);
   - a hand-start snapshot for each hand that began;
   - the action receipt;
   - bot RNG checkpoints;
   - awards;
   - the table row (status, version, state version, event count, pending-decision scheduler);
   - pause rows;
   - one outbox row.
3. The store writes the batch atomically. In Postgres the table row is locked `FOR UPDATE`, and the commit only goes ahead if `owner_id` is this writer and `version` is the one this writer last committed. Otherwise the result is `OwnershipLost`, and the API answers 409 `writer_fenced`. Unique keys on `(table, seq)`, `(table, action_id)`, `(table, decision_id)` and `(table, outbox version)` stop a double append even if a bug slipped past the check. Event sequence numbers must continue the stored log exactly.
4. Only after the commit does the API answer. If the commit fails, the in-memory table is marked dirty and later rebuilt from the store, so memory never runs ahead of what is durable.

Zero-delay bot moves after a human action are separate commands with separate
commits. A crash between them loses no accepted action.

**Ownership.** A table is owned by a writer id with a lease (15 s). It is claimed at
startup, and renewed by every commit and by a 1 s heartbeat that also records the
human's remaining think time.
- A second process with another writer id cannot claim a table while the lease is live. It serves that table as 409 `writer_fenced`.
- A forced claim (`TableManager(force_claim=True)`) takes the table over. From then on, every commit from the old writer fails the compare-and-set.
- What is **not** built: fencing tokens checked by every downstream side effect, lease handover between live processes, and running several API workers at once. Run one API process per database.

## Outbox and WebSocket

A commit adds outbox version `V`. After the command (and any zero-delay bot
commits) the manager publishes. Each subscriber gets its own projection of the
current state, at version ≥ `V`. The pump then marks those rows published.

If the process dies before broadcasting, the rows stay unpublished. The next process
publishes them during recovery, and any client that reconnects gets a fresh
authorized snapshot whose version is greater than anything it saw before. Frames
are projections, never raw events, and the stream never sends a version lower than
one it already sent.

## Recovery

At startup every stored table is loaded:

1. Load the newest hand-start snapshot. It holds stacks, standings, previous positions, state version, decision counter and the dealer RNG checkpoint, all private.
2. Rebuild the engine from the snapshot (`Table.resume_at_hand`) and deal that hand again:
   - **Seeded tables** re-deal from the checkpointed dealer stream. They must reproduce the stored deck, and future hands continue the original stream.
   - **Unseeded tables** take the deck stored in `HandStarted`.
3. Re-submit every committed command after the snapshot: manual `ActionApplied` events, each with its `ActionFallback` reason if it had one. Automatic actions are not commands, so the engine must reproduce them.
4. The rebuilt log must equal the stored log event for event. That includes the hand-start hash and every decision hash, and the state version, event count and completion flag must match the table row.
5. Restore bot RNGs from their checkpoints. The checkpointed policy and artifact hash must match the running bot code.

Any difference halts the table: the status becomes `halted` with a private reason,
and every endpoint answers 503 `table_halted`. This covers a hash mismatch, a
different PokerKit or arena build, a missing checkpoint or a changed bot artifact.
Nothing is repaired, reshuffled or re-decided, and a halted table stays halted
across restarts.

`tests/persistence/test_resume.py` resumes seeded and unseeded tables in all three
rulesets from every snapshot and every commit boundary, and checks that tampered
snapshots, tampered events, a wrong dealer state and an engine version change are
all refused.

## Infrastructure pauses

The spec requires that an infrastructure outage pause the table, and that think time
be restored according to the stored pause record. A client disconnect does not
pause anything: the clock keeps running.

| Situation | What happens | Pause row |
| --- | --- | --- |
| Storage unavailable (failed commit, heartbeat or publish) | Writes and reads answer 503 `storage_unavailable`; no timeouts or bot moves run. When the store answers again, every table is reloaded from it. | `storage_outage` |
| Process crash | On restart, the remaining time comes from the last durable scheduler (written by every commit and heartbeat). | `process_restart` |
| Graceful shutdown | Remaining time is written at shutdown as an open pause; the next start resumes from it and closes it. | `shutdown` |

Each row records the reason, decision, remaining milliseconds, paused-at and
resumed-at times. Wall-clock times are audit data only; deadlines run on the
monotonic clock.

Remaining time is accurate to the heartbeat interval, and rounding always favours the
human. If the last durable heartbeat was up to 1 s before the outage, the human gets
that time back.

The web client retries an action on `storage_unavailable` with the same `action_id`,
which the server deduplicates. It keeps reconnecting the stream, and shows "Table
paused" or "Table halted" when it has no snapshot to show.

## Spec 12.3 server items and where they are tested

All tests in `tests/persistence/` run against both stores. Postgres cases skip
visibly when it is unreachable, unless `ARENA_REQUIRE_POSTGRES=1`.

| Item | Test |
| --- | --- |
| Duplicate identical action, racing (8 threads) | `test_faults.py::test_duplicate_identical_actions_racing_commit_once` |
| Same ID, changed payload (also after restart); late response to a previous decision | `test_same_action_id_with_changed_payload_conflicts_even_after_restart` |
| Different IDs for one decision; two tabs at once | `test_two_tabs_racing_different_action_ids_for_one_decision` |
| Action racing timeout, 100 repeats on real threads | `test_action_racing_timeout_100_times` |
| On-time action waiting in the queue | `tests/api/test_tables_api.py::test_on_time_action_waiting_in_queue_beats_a_later_timeout` |
| Crash before transaction, inside it (before commit), after commit, before broadcast | `test_recovery.py::test_crash_during_a_human_action`, `test_crash_during_bot_and_timeout_commits` |
| A real `SIGKILL` of a Postgres-backed server, then restart | `test_process_kill.py` |
| No double debit, reshuffle or duplicate payout | `verify_record` in `test_recovery.py`: full replay from scratch, unique decisions, contiguous hands, awards equal the replayed points |
| Infrastructure restart restores bot and dealer state | `test_seeded_bot_table_interrupted_twice_matches_an_uninterrupted_run` |
| Halt on an irreproducible record | `test_corrupt_stored_event_halts_the_table`, `test_resume.py` |
| Database unavailable; ambiguous commit | `test_storage_outage_pauses_and_restores_human_think_time`, `test_ambiguous_commit_is_resolved_from_the_store` (memory store only) |
| Ownership change; obsolete writer tries to commit; reused lease after restart | `test_obsolete_writer_is_fenced_and_cannot_fork_the_table`, `test_live_lease_blocks_a_second_writer`, every restart test (same writer id) |
| Reconnect after a crash before broadcast; stream ordering | `test_reconnect_after_crash_before_broadcast_gets_the_committed_state_first` |
| Store contract (atomic batches, compare-and-set, claims, pauses, migrations) | `test_store_contract.py` |

Fault hooks (`TableManager.faults`, points `before_tx`, `in_tx`, `after_commit`,
`before_broadcast`) raise `SimulatedCrash`, a `BaseException` that nothing in the
server catches. The test then abandons that manager and boots a new one on the same
store, which is how a process restart looks to the store.

## Limits

- **One API process per database.** Multi-writer fencing is deferred, as the build plan says.
- **Whole tables are loaded at startup and kept in memory.** There is no eviction or lazy loading.
- **An ambiguous commit (lost acknowledgement) is only injected in the memory store.** Postgres relies on the same reload path, which the outage test exercises.
- **The bot-only benchmark invalidate-and-rerun policy (spec 9.2) is not implemented.** It belongs with the benchmark runner. Interactive tables pause and resume.
- **Private data at rest is not encrypted.** That includes decks, seeds, hole cards and state hashes. The database is private storage, and credentials are stored only as SHA-256 hashes.
- **Halted tables have no diagnosis tooling yet.** The private reason is in `arena_tables.halted_reason`, and the record can be replayed with `arena.persistence.recovery.rebuild_table`.
