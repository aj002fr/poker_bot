# Opponent styles, table reading and learning aids

Practice opponents modelled on the playing styles described in *Harrington on Hold'em,
Vol. I* (Part Two, "Playing Styles and Starting Requirements", and Part Three, "Reading
the Table"). Everything here is our own paraphrase and our own rule tables. No book text
is reproduced, and the bots are teaching opponents with recognisable habits, not strong
players. They play No-limit Hold'em only (`practice_nlhe`).

The machine-readable contract the web client builds against is served by `GET /styles`
(profiles, stat definitions, opening charts and the 13×13 grid). That endpoint is the
only copy of this data; the UI does not duplicate it.

## Styles

| Style id | Policy id | Book reference | Habits |
| --- | --- | --- | --- |
| `rock` | `style_rock` | Part Two, the conservative style | Few hands, opens by raising from position charts, rarely bluffs, gives up to resistance |
| `aggressive` | `style_aggressive` | Part Two, the aggressive style | Wider charts (any pair, any ace, broadway, suited connectors), frequent continuation bets |
| `super_aggressive` | `style_super_aggressive` | Part Two, the super-aggressive style | Raises almost anything late, steals constantly, reraises light, sets check-raise traps |
| `station` | `style_station` | Part Three, loose callers | Plays many hands and mostly calls; almost never raises or folds after the flop |
| `weak_tight` | `style_weak_tight` | Part Three, tight players who back down | Very few hands, limps often, folds to pressure unless holding a big hand |
| `switcher` (advanced) | `style_switcher` | Part Two, varying your style | Conservative while deep, aggressive as the blinds rise (or below M 15), push/fold below M 7 |

Each profile declares expected ranges for hands played (VPIP), preflop raise (PFR),
aggression factor and fold-to-reraise. `tests/agents/test_styles.py` plays many seeded
tables (all styles together, and styles mixed with the baselines) and checks that every
style's measured numbers fall inside its declared ranges. Measured over 30 tables of 100
hands with all six styles seated:

| Style | VPIP | PFR | Aggression | Fold to reraise |
| --- | --- | --- | --- | --- |
| rock | 0.13 | 0.12 | 6.6 | 0.72 |
| aggressive | 0.27 | 0.24 | 7.6 | 0.54 |
| super_aggressive | 0.44 | 0.36 | 5.8 | 0.33 |
| station | 0.44 | 0.00 | 0.06 | n/a |
| weak_tight | 0.09 | 0.05 | 1.2 | 0.79 |
| switcher | 0.34 | 0.30 | 5.2 | 0.32 |

### How the bots decide

`arena/agents/styles/engine.py` parses the observation into a spot: hand class
(`AKs`, `T9o`, `77`), position group, the preflop action so far (limpers, raises, who
raised), and a post-flop strength bucket (monster, strong, medium, weak) with draw outs.
Each style is a short rule table over that spot. Opening charts live in
`arena/agents/styles/charts.py` as range strings (`"77+, AJ+, KQs"`).

Position groups follow seats after the button: the button and cutoff are `late`, the
two seats before them `middle`, anything earlier `early`; the blinds are `blinds` and use
the `late` chart when folded to. At six seats or fewer there are no `early` seats.

All style bots are `ExplicitStateBot`s. Their only state is the policy seed and a draw
counter, so a seeded table replays exactly, the bot resumes after a crash, and decisions
take well under 0.1 ms. Every decision carries a short private note ("opened AJs from
CO: in the aggressive chart"), visible only in the reviewer replay after the table ends.

Style files are trusted repository code (`source: "style"`). At live tables they run
in-process like the baselines; in benchmarks they also load in sandboxed workers,
because the worker imports the shared engine before the sandbox hides the repository.
A style's artifact hash covers its file and the shared engine, charts, evaluator and
profiles, so changing the engine changes every style's hash.

## Table modes (`POST /tables` → `styles`)

```json
{"styles": {"mode": "mixed", "random_pool": ["rock", "aggressive", "station"], "learning_aids": true},
 "seats": [{"kind": "human"}, {"kind": "bot", "policy": "style_rock"},
           {"kind": "bot", "policy": "style_aggressive", "reveal": false},
           {"kind": "bot", "policy": "style_random"}]}
```

| Mode | Effect |
| --- | --- |
| `labelled` (default) | Every bot seat shows its policy and style |
| `hidden` | Every bot seat (styles and baselines) shows only "Player N" |
| `mixed` | Per-seat `reveal` (default true) |
| `random` | Every bot seat gets a style drawn from `random_pool` and is hidden |

`style_random` can be used in any mode; its pick is always hidden until the table ends.
With a `seed` the picks are reproducible; without one they use fresh randomness.
Contradictions (for example `reveal: false` in labelled mode, or `reveal: true` on a
random seat) are `422 invalid_styles`.

### Privacy

While a table runs, a hidden seat's policy, style, label, source, artifact hash and
reasoning notes appear in no HTTP body, WebSocket frame, error, history or replay frame,
for any credential including the host's. The engine's event log never names policies;
hidden seats are masked in `setup.seats`; stats come from spectator-visible events;
learning aids use only the requesting seat's own view. The table config hash does not
cover seats. The lineup hash, which does, is published only in the reveal.
`tests/api/test_styles_api.py` plays a mixed table and scans every captured payload for
those strings and for the hidden seats' note texts.

When the table completes, `setup.styles.revealed` turns true, the seats show their real
identities, and the snapshot gains `reveal`: per seat the real policy and style, whether
it was a random pick, the book reference, evidence (each stat with its sample size,
expected range and whether it fits), a one-line summary and its showdowns. Guess scoring
is left to the client.

Style setup is stored with the table (`arena_tables.setup`) and reasoning notes in
`bot_notes`, apart from the event log because replay must regenerate that log exactly.
Both survive restarts (migration `0002_styles_and_notes.sql`).

## Table-reading stats

`arena/table/player_stats.py` (`StatsTracker`) turns public events into the four
table-reading questions of Part Three, with sample sizes:

| Stat | Label | Meaning |
| --- | --- | --- |
| `vpip` | Hands played | Hands where the seat chose to put chips in preflop (blinds and a free big-blind check do not count) |
| `pfr`, `rfi` | Raise-first | Hands with a preflop raise; and of hands folded to the seat, how often it opened with a raise |
| `aggression` | Callers or raisers | Bets and raises divided by calls, all streets |
| `reraise` | Reraises | After betting or raising and then facing a raise, how often it folded, called or raised again |
| `showdowns` | Showdowns | Hands shown, with the board, the hand in words, the line per street and the chips won |

Automatic actions are not counted. Every snapshot carries `stats` (with the three most
recent showdowns per seat); `GET /tables/{id}/stats` returns every showdown.

## Learning aids (`GET /tables/{id}/aids`, seat credential)

Computed from the seat's own cards, the board and public state only:

- position label and group for every dealt seat,
- the hand class, the hand in words and its hand-rankings row,
- flush and straight draws with outs (flop and turn),
- whether the hand is in the conservative and aggressive opening charts for its group,
- equity against one random hand and against the live field (seeded Monte Carlo, so the
  same spot always gives the same number; exact on the river against one hand),
- on the seat's turn, pot odds (`to_call / (pot + to_call)`) and a coach hint: what the
  Rock's rules would do here without randomness, with the reason. The hint is a teaching
  heuristic, not optimal play.

A table created with `learning_aids: false` answers `403 learning_aids_disabled`.

## Command line

```bash
uv run python -m arena.cli match --bot my_bot.py --opponents rock,aggressive,station
uv run python -m arena.cli benchmark --bots rock,aggressive,super_aggressive,station,weak_tight,switcher
```

Style names also accept `style_rock` and a few aliases (`conservative`, `super`,
`weaktight`). The baseline `calling_station` keeps its name; the style is `station`.
