# Local bot SDK and agent runtime (milestone 4)

This is the arena's own local contract for Python bots. It is a practice tool, not
Sequoia's Agent API, and nothing here claims compatibility with an official runtime.
Sandbox details and their limits are in [`agent-sandbox.md`](agent-sandbox.md).

## Writing a bot

A bot is one Python file that defines `make_bot()`. Start from
[`examples/template_bot.py`](../examples/template_bot.py):

```bash
cp examples/template_bot.py my_bot.py
uv run python -m arena.cli match --bot my_bot.py
```

`make_bot()` returns one of the two bot kinds in `arena.agents.sdk`. Each declares a
`name` and the `rulesets` it supports (`practice_nlhe`, `uncertainty_shortdeck`,
`uncertainty_plo4`). A bot is never seated at a table with an undeclared ruleset: the
runtime rejects the pairing before the table starts (`UnsupportedRuleset`, or CLI
exit code 5), and no hands are played.

| Kind | Interface | After a crash or kill |
| --- | --- | --- |
| `ExplicitStateBot` (recommended) | `initial_state(table) -> state`, `decide(obs, state) -> (action, new_state)` | Restored from the last committed state, so it replays exactly |
| `InstanceStateBot` | `start_table(table)`, `decide(obs) -> action`, with ordinary attributes | Restarted as a fresh instance; the seat is flagged `restorable: false` |

- **Explicit state** is JSON (`None`, bool, int, float, str, list, dict with string
  keys). It is limited to 64 KiB (`MAX_STATE_BYTES`) in canonical form: sorted keys,
  compact separators, ASCII and no NaN. A state that cannot be encoded counts as a
  failed decision (`state_invalid` or `state_too_large`).
- **Commit rule.** A returned state becomes the committed state only together with an
  action that the parent accepted: on time, valid schema and legal. A failed decision
  leaves the previous committed state in place. Before every decision the bot gets the
  committed state, decoded from its canonical JSON. So state left over from a rejected
  reply, tuples, or other non-JSON values never carry over.
- **Instance state** is a cache, not a checkpoint. A table continues after an
  instance-state worker is lost, but its results are labelled as not restorable (see
  `docs/assumptions.md`).
- **Randomness.** Use `TableInfo.policy_seed`. It is a per-seat stream
  (`policy-seat-N`), independent of the deck seed. Explicit-state bots should keep their
  random stream position in `state`, as the template does. Do not read the wall clock.
  Workers run with `PYTHONHASHSEED=0`.
- **Imports.** A bot can import the standard library, `arena.agents.sdk` and the
  arena's domain types (already loaded in the worker), plus packages installed in the
  arena's environment. It cannot read the rest of the repository, so engine modules
  such as `arena.table` cannot be imported from a sandboxed worker.
- **Untimed setup.** `make_bot()`, `initial_state()` and `start_table()` run when the
  worker loads, before the first decision. Do imports and precomputation there. Loading
  has a 10 s limit.

`TableInfo` (schema `docs/schemas/table-info.v1.json`) has an anonymous `table_id`,
the hero's physical seat, the seat list, the ruleset and config with its hash, the
decision budget and the policy seed. A new one is issued for every table, and no field
identifies an opponent's account or artifact. The observation is the strict
`Observation` from `docs/observations-and-projections.md`.

Action helpers: `fold()`, `check()`, `call()`, `raise_to(amount)` (a street total, not
an increment), `check_or_fold(legal)`, `check_or_call(legal)`, and
`clamp_raise_to(legal, target)`, which returns the legal raise closest to the target or
`None`. A bot may also return a plain dict that matches the `Action` schema.

## Decisions, deadline and fallback

Each seat's bot runs in its own pre-started, sandboxed worker process. Workers are
fresh for every table and are never reused.

- **Budget:** 1,000,000 ns per decision, measured by the parent over the whole round
  trip. The clock starts just before the observation is serialized. It stops when the
  parent has read the last byte of the reply (`received_ns`). A reply is on time when
  `received_ns <= dispatched_ns + budget_ns`, so the boundary nanosecond counts as on
  time. Serialization, IPC and the worker's validation of the observation all count
  against the bot. An on-time reply is still accepted if the parent validates or applies
  it after the deadline.
- **Wake notice.** When a seat is about to act, the parent sends the worker a "wake"
  frame before it builds the observation (the clock has not started yet, and the frame
  carries no game data). The worker then spins for up to 3 ms waiting for the request.
  This hides the virtual CPU's wake-up latency, which is about 250 µs on this VM.
- **Fallback:** a decision that fails for any reason is consumed once, with check if
  checking is legal and fold otherwise (`Table.apply_fallback`). There is no retry. The
  reason code is public in the history, and the rejected payload is private.

| Reason code | Meaning |
| --- | --- |
| `timeout` | No complete reply by the deadline, including replies that arrive late |
| `worker_crash` | The worker exited or closed its pipe before the deadline |
| `bot_exception` | `decide()` raised |
| `bot_memory_error` | `decide()` raised `MemoryError` (address-space limit) |
| `invalid_output` | The reply is not a valid frame or has no action |
| `oversized_output` | The reply is over 16 KiB of metadata or 64 KiB of state |
| `state_invalid`, `state_too_large` | The returned state is not JSON, or is over 64 KiB |
| `invalid_schema` | The action fails the `Action` schema |
| `illegal_action` | The action is not in the legal set |
| `worker_unavailable` | No worker could be started or restored for this decision |

- **Late replies** are drained and discarded and never apply to a later decision.
  Every request carries a sequence number. When the seat is next asked, the parent
  waits up to 20 ms (`stale_grace_ns`) for the overdue reply. If it still has not
  arrived, the worker is killed (`unresponsive`) and replaced by a fresh worker restored
  from the committed state. After 25 restarts in one table the seat stops being
  restored, and its remaining decisions fall back with `worker_unavailable`.
- A worker that dies is replaced at the seat's next decision in the same way.
- The bot's stdout and stderr go to a capped log (64 KiB in the worker, the first
  4 KiB kept by the parent). Printing cannot corrupt replies or block the worker.

## Timing reports

Three numbers are recorded separately for each decision:

- `e2e_ns`, end to end as the parent sees it (the deadline number),
- `worker_wall_ns`, the wall time of the bot's `decide` call, and
- `worker_cpu_ns`, the CPU time of that call (`time.thread_time_ns`).

Worker timings are reported by the worker and are informational only. Deadlines use
only the parent's clock. Summaries print seconds.

## Commands

```bash
# Your bot against the baselines (default: 240 tables, 6 seats, NLHE).
uv run python -m arena.cli match --bot my_bot.py
uv run python -m arena.cli match --bot my_bot.py --ruleset uncertainty_plo4 --players 4 \
    --opponents calling_station,legal_random --tables 120 --seed my-seed

# Any set of entrants: built-in names or paths.
uv run python -m arena.cli benchmark --bots nlhe_heuristic,calling_station,legal_random,check_fold \
    --tables 480 --out artifacts/runs/demo --save-histories
```

The output gives:

- mean finishing points per entrant, with a 95% bootstrap interval over seed blocks,
- e2e p50/p95/p99/max and worker wall and CPU times in seconds,
- fallback counts by reason, restarts, and tables whose state was lost.

`--json` prints the full summary. `--out` writes `manifest.json` (schema
`docs/schemas/benchmark-manifest.v1.json`) and `private/seeds.json`, plus `tables/` with
`--save-histories`. Other options:

- `--jobs` sets the number of parallel table processes. The default is
  `max(1, min(4, cpu_count // 4))`, which is 2 on an 8-vCPU VM. See the sandbox doc for
  why more jobs is worse.
- `--in-process` runs the bots without the sandbox or deadline. This is only for
  trusted code and debugging, and the manifest marks such runs `competitive: false`.

Built-in baselines (`arena/agents/baselines/`, all explicit-state):

| Name | Rulesets | Policy |
| --- | --- | --- |
| `check_fold` | all | Check if possible, else fold |
| `calling_station` | all | Check or call, never raise |
| `legal_random` | all | Uniform over legal action types, with a uniform raise size, using a state-held RNG |
| `nlhe_heuristic` | NLHE only | A simple hand-strength rule (Chen score preflop, made-hand category later). A sparring partner, not a strong bot |

## Benchmark design

`arena/benchmark/runner.py` plays **paired seed blocks**:

- A block is one deck seed, played N times for N seats. In rotation r, entrant i sits
  at seat (i + r) mod N, so every entrant plays every seat with the same cards.
- Each table gets a fresh anonymous `table_id` derived from a hash of the seed and
  rotation.
- Each seat's policy seed comes from `SeededRng(seed, "policy-seat-N")`.
- Means and intervals are computed per block (a percentile bootstrap with 2,000
  resamples), so the pairing reduces variance from the cards.

The manifest extends the milestone 3 `RunManifest` with:

- the entrants, with their artifact hashes and declared rulesets,
- the isolation mode, the budget, and the block and rotation counts,
- the retry policy (`retry`, with a prose version in `retry_policy`),
- the process layout and a description of the sandbox,
- the summary.

### Infrastructure failures and reruns (spec 9.2)

A table interrupted by infrastructure is marked **invalid** and rerun with the same
seed and the same `table_id`, under an attempt ID `<table_id>-a<n>`. The declared
`RetryPolicy` allows up to 3 attempts in total by default, and the manifest records it.
`run_benchmark(retry_policy=RetryPolicy(max_attempts=1))` records invalid tables
without rerunning them. These are the invalidating causes:

| Cause | When |
| --- | --- |
| `worker_start_failed` | No sandboxed worker could be started for a seat, at the start of the table or for a restore |
| `host_os_error`, `host_memory_error` | `OSError` or `MemoryError` in the table process (file descriptors, disk, memory) |
| `job_process_lost`, `job_error` | The parallel job process running the table died, or could not return the table |
| `instance_state_lost_to_infrastructure` | An infrastructure kill hit an instance-state bot, whose state cannot be restored faithfully |

Some failures do **not** invalidate a table:

- Agent faults (crashes, timeouts, exceptions, invalid or illegal output) use the normal
  fallback. They are the bot's results.
- An infrastructure kill of an explicit-state bot is restored exactly from committed
  state. The table stays valid and is listed under `infrastructure_restored`.
- Engine and other unexpected errors are deterministic. A rerun would only repeat them,
  so they are listed under `tables_failed` and are not rerun.

Rules against hiding results:

- The rerun decision depends only on those infrastructure events, never on scores.
- Means and intervals use each table's valid attempt.
- Decisions, fallbacks, timing and restarts count **every** attempt.
- The summary lists every invalid attempt (`invalid_attempts`), with its reason and its
  points if it finished. Each entrant also gets
  `mean_points_including_invalid_attempts`, so the effect of reruns can be checked.
- Tables still invalid after the last attempt appear in `tables_invalid`, and in the
  manifest's `failed_tables` as "invalid after N attempts". The CLI then exits with
  code 1.
- With `--save-histories`, invalid attempts that produced a table are saved under
  `tables/invalid/a<n>/`.

In parallel runs, at most one table per job process is in flight. So when a job
process dies, only the tables that were actually running are marked
`job_process_lost`, and a fresh process pool takes over. After a start or host failure,
the table process also replaces its worker pool.

Seeds stay private (`private/seeds.json`), and the manifest holds only their hash.
These results are for practice against local baselines. They are not a claim about
strategy strength.
