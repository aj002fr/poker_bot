# Pluribus hands: import, study mode and agreement

Stage 7 of the self-play learning plan. The 10,000 six-max hands that Pluribus played
against professionals (Brown and Sandholm, *Science* 2019) are replayed through the arena
engine, turned into pause-and-predict study packs for the browser, and used for an
offline diagnostic of how often our bots choose what Pluribus chose.

```bash
make pluribus-fetch      # pinned download into the cache, hash-checked (a few seconds)
make pluribus-import     # validate every hand in the engine -> artifacts/phh/pluribus-import.json
make study-sample        # 24 fixture hands -> artifacts/study/pluribus-sample (make dev does this)
make study-pack          # all hands -> artifacts/study/pluribus (150 MB, about 2 minutes)
make pluribus-agreement  # bot vs Pluribus agreement -> artifacts/phh/pluribus-agreement.json
```

## Source and cache

`strategy/data/fetch.py` pins the PHH project's
[phh-dataset](https://github.com/uoftcprg/phh-dataset) at commit
`e47fbd5816372360bade4de5d712346fe1bb70f6`, directory `data/pluribus` (git tree
`e0cd07e4…`). It makes a blobless sparse clone of only that directory, checks the tree id,
then checks a content hash: SHA-256 over the sorted lines `"<path> <sha256 of file>\n"` of
all 10,000 `.phh` files, pinned as `1b36ee3d…f1c9`. The copy lives in
`$ARENA_DATA_CACHE`, else `$XDG_CACHE_HOME/arena`, else `~/.cache/arena`, under
`phh-dataset/<commit>/`, never in the repository. `--verify` re-checks a cached copy,
`--where` prints its path and `--force` fetches again.

Only `strategy/data/fixtures/` is committed: 24 byte-identical upstream hands chosen to
cover rivers, showdowns, 3-bets, preflop all-ins, check-raises, big blind defences and a
split pot with an odd chip. `SOURCE.json` lists each file's SHA-256 and what it covers,
plus the pins, citation and DOIs (dataset 10.5281/zenodo.10796885, paper
10.1126/science.aay2400). The dataset is MIT licensed; the licence is copied to
`fixtures/LICENSE-phh-dataset`.

## Import and validation

`strategy/data/phh_import.py` parses each PHH record (TOML) and replays it in the arena's
own `Table` with a cash-game configuration: six seats, blinds 50/100, equal 10,000 stacks,
one blind level and a hand cap high enough never to bind. PHH players 1..6 are SB, BB,
UTG, MP, CO and BTN, so they map to engine seats 0..5 with the button on seat 5. The deck
is built through `strategy.rules.conformance.build_deck` so the engine deals exactly the
recorded hole cards and boards.

A hand is rejected, with a reason code, when any of these fail:

| Code | Check |
|---|---|
| `toml`, `fields`, `cards`, `action_syntax` | the record parses; every field and card is well formed |
| `variant`, `not_six_max`, `antes`, `blinds`, `stacks` | no-limit hold'em, six players, no antes, SB/BB only, `min_bet` equals the big blind, equal stacks |
| `positions`, `hole_cards`, `board` | the engine posts the blinds on players 1 and 2 and deals the recorded cards, flop/turn/river in order |
| `actor`, `illegal_action`, `rejected` | each recorded action comes from the engine's actor and is legal in `LegalActions` |
| `size_mapping` | every raise maps to a legal option of the canonical menu (`strategy.rules`) |
| `incomplete`, `showdown` | the actions finish the hand; shown cards match what was dealt and what the engine reveals |
| `finishing_stacks` | the engine's final stacks equal the record's |

Raise sizes are mapped to the canonical menu of `strategy.rules.RulesAdapter` (fold,
check, call, min raise, ⅓, ½, ¾, 1, 1.5 and 2.5 pot, all-in): the option nearest in log
ratio of the raise increment. The exact amount is still what the engine applies.

The PHH records truncate odd chips in split pots. When the final stacks differ only by a
lost chip, the hand is replayed again at double stakes, where the split is exact. It
passes if every seat then finishes at twice its recorded stack plus one for each
truncated half chip, and if the seats without a half chip also match at the original
scale. Such a hand gets a note; the study pack keeps it at the original scale, with the
arena's odd-chip rule deciding who gets the odd chip.

**Result on the full corpus** (`artifacts/phh/pluribus-import.json`): all 10,000 hands
import, none are rejected. 8 need the double-stakes check for a split pot. They contain
91,356 decisions, 15,169 by Pluribus: 8,007 folds, 2,414 checks, 1,621 calls and 3,127
raises. 2,397 of the raises hit a menu amount exactly; the rest are off by a mean log
ratio of 0.045 (at most 0.47). Pluribus's raises map most often to ½ pot (1,309), the
minimum (727) and pot (410).

## Study packs

`strategy/data/study.py` writes a pack directory: `pack.json` (provenance and caveat),
`index.json` (one summary per hand) and `hands/<id>.json`. Each hand holds:

- frames: the arena's own seat projection for Pluribus (`table_view` with
  `Viewer.for_seat`) after the deal and after every action, so a frame can only show
  Pluribus's cards, the board and cards shown at showdown;
- the hand's public events, projected the same way;
- one entry per Pluribus decision: the frame index, the canonical options with amounts,
  what Pluribus did, and the spot from `strategy.features` (position, what it faces, pot,
  price to call, pot odds, effective stack, SPR, opponents, hand class and preflop
  percentile, equity against random hands, made hand, draws, outs, board texture);
- the result: final stacks, every player's hole cards and who reached showdown. The
  browser shows these only once the hand is over.

The build is deterministic (same input, same bytes) and goes to a temporary directory
that is renamed into place.

## Serving and the browser

`arena/api/study.py` reads packs from `ARENA_STUDY_DIR` (default `artifacts/study`) and
never imports `strategy`; `tests/strategy/test_isolation.py` enforces that for all of
`arena/`. Endpoints:

- `GET /study/packs`: packs with hand counts per filter;
- `GET /study/packs/{pack}/hands?filter=all|postflop|showdown|big_pot&offset&limit`;
- `GET /study/packs/{pack}/hands/{hand}`.

Unknown packs or hands return 404. A pack with another schema version or broken files is
left out of the list and returns 503 `pack_unreadable`.

**Study Pluribus** (`#/learn/pluribus`, linked from Learn) plays the hand from Pluribus's
seat and stops at each of its decisions. You pick fold, check, call or a bet or raise
size (amounts shown in chips), then the reveal says whether you matched and what Pluribus
did. The spot panel shows pot odds against equity, with the reminder that equity against
random hands overstates it against real ranges. At the end every hand is turned face up.
Your score (actions matched, and sizes matched when you both raised) is kept in this
browser. The list filters to hands where Pluribus acted after the flop or reached
showdown, or where the pot reached at least 20 BB.

## Agreement diagnostic

`strategy/data/agreement.py` asks each bot to act on the exact observation Pluribus saw
at every Pluribus decision. Each hand gets a fresh bot, and the bot's state carries
across that hand's decisions. The hand then follows Pluribus's real line whatever the
bot chose. Results are averaged over four policy seeds. *Action agreement* compares fold,
check, call and bet/raise. *Balanced* averages the match rate over Pluribus's four kinds,
so folding everything cannot score well. *Size agreement* counts spots where both raised
and the amounts map to the same menu option.

All 10,000 hands, 15,169 Pluribus decisions (`make pluribus-agreement`, about 20 s on 6 cores):

| Bot | Action | Balanced | Preflop | Flop | Turn | River | Size (both raise) |
|---|---|---|---|---|---|---|---|
| baseline:nlhe-v2 | 76.0% | 64.6% | 82.7% | 61.8% | 60.2% | 64.4% | 31.2% (1,976) |
| baseline:control | 76.3% | 67.4% | 82.6% | 64.3% | 60.8% | 62.9% | 41.5% (1,955) |
| arena:nlhe_heuristic | 72.2% | 57.9% | 77.5% | 63.5% | 60.7% | 56.2% | 12.0% (752) |
| style:rock | 76.3% | 60.7% | 81.7% | 65.8% | 64.9% | 62.9% | 25.4% (1,452) |
| style:aggressive | 76.0% | 63.8% | 83.7% | 61.7% | 58.5% | 56.7% | 16.6% (2,228) |
| style:super_aggressive | 68.3% | 60.4% | 74.8% | 56.3% | 52.9% | 52.7% | 25.8% (2,542) |
| style:station | 61.1% | 63.2% | 59.6% | 65.3% | 65.2% | 60.1% | 15.0% (150) |
| style:weak_tight | 73.3% | 57.7% | 77.4% | 65.2% | 65.1% | 62.3% | 34.2% (643) |
| arena:calling_station | 26.6% | 50.0% | 10.8% | 61.3% | 61.1% | 55.3% | – |
| arena:check_fold | 68.7% | 50.0% | 72.2% | 61.5% | 62.1% | 60.0% | – |
| arena:legal_random | 37.5% | 39.3% | 33.8% | 45.0% | 44.9% | 46.1% | 7.2% (1,268) |

No bot made an illegal move or raised an error. Read this table with care:

- Most Pluribus decisions are preflop folds, so `check_fold` reaches 68.7% action
  agreement by doing nothing. The balanced column and the per-street columns are more
  informative than the headline.
- Pluribus chose one action from a mixed strategy, so a single recorded action is a
  sample, not the right answer. Even a perfect copy of Pluribus would not score 100%.
- Pluribus played 100 BB six-max cash. Our bots are tuned for eight-player tournament
  points, where short stacks and ICM change correct play.
- This is a diagnostic. It is not a training target, a strength measure or a gate.
  Fitting a policy to these numbers would teach it Pluribus's sampling noise.

The diagnostic also exposed an arena bug: the style bots' evaluator crashed on a board
holding four of a kind (no kicker left), which the arena's own games had never dealt.
This is fixed in `arena/agents/styles/handeval.py` and covered by a test.
