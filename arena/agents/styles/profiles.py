"""Style profiles: the one copy of each style's description, book reference, expected
stats and counter-strategy. ``GET /styles`` serves these and the UI renders them.

All text is our own paraphrase of ideas from *Harrington on Hold'em, Vol. I* (Dan
Harrington and Bill Robertie), credited by part and a descriptive section name. No book
text is reproduced. ``expected`` ranges are what :mod:`arena.table.player_stats`
measures for each bot over many seeded hands at mixed 6-seat tables
(``tests/agents/test_styles.py`` checks them).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PART_TWO = "Part Two: Playing Styles and Starting Requirements"
PART_THREE = "Part Three: Reading the Table"

Range = tuple[float, float] | None


@dataclass(frozen=True)
class BookRef:
    part: str
    section: str


@dataclass(frozen=True)
class Expected:
    vpip: Range
    pfr: Range
    aggression: Range
    fold_to_reraise: Range = None


@dataclass(frozen=True)
class StyleProfile:
    id: str
    name: str
    family: str
    description: str
    book_ref: BookRef
    expected: Expected
    counter_strategy: str
    tells: tuple[str, ...]
    advanced: bool = False
    rulesets: tuple[str, ...] = ("practice_nlhe",)

    @property
    def policy_id(self) -> str:
        return POLICY_PREFIX + self.id

    def public(self) -> dict[str, Any]:
        e = self.expected
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "book_ref": asdict(self.book_ref),
            "expected": {k: (list(v) if v is not None else None) for k, v in asdict(e).items()},
            "counter_strategy": self.counter_strategy,
            "tells": list(self.tells),
            "advanced": self.advanced,
            "rulesets": list(self.rulesets),
        }


POLICY_PREFIX = "style_"
RANDOM_POLICY = "style_random"

PROFILES: dict[str, StyleProfile] = {
    p.id: p
    for p in (
        StyleProfile(
            id="rock",
            name="Rock",
            family="Conservative",
            description="Plays few hands, enters by raising from position-based opening charts, and seldom bluffs.",
            book_ref=BookRef(PART_TWO, "The conservative style"),
            expected=Expected(vpip=(0.08, 0.26), pfr=(0.07, 0.25), aggression=(2.0, 60.0),
                              fold_to_reraise=(0.45, 0.95)),
            counter_strategy=("Steal its blinds often and give up when it plays back: a Rock's raise or reraise "
                              "usually means a real hand. The book's Gap Concept explains why it needs more to "
                              "call a raise than to open."),
            tells=("Hands played stays low even after many hands.",
                   "When it does enter, it raises rather than calls.",
                   "Shows down strong, premium-looking hands."),
        ),
        StyleProfile(
            id="aggressive",
            name="Aggressive",
            family="Aggressive",
            description=("Opens any pair, any ace, two face cards and suited connectors from any seat, and "
                         "follows up with frequent continuation bets."),
            book_ref=BookRef(PART_TWO, "The aggressive style"),
            expected=Expected(vpip=(0.20, 0.38), pfr=(0.17, 0.35), aggression=(2.5, 30.0),
                              fold_to_reraise=(0.35, 0.80)),
            counter_strategy=("Its opening range is wide, so reraise it with your strong hands and call down "
                              "lighter. Expect continuation bets on most flops."),
            tells=("Plays many hands and mostly raises them.",
                   "Bets the flop after raising preflop most of the time.",
                   "Shows down medium hands such as weak aces and small pairs."),
        ),
        StyleProfile(
            id="super_aggressive",
            name="Super-aggressive",
            family="Aggressive",
            description=("Raises almost any two cards when folded to it late, steals the blinds constantly, "
                         "reraises light and sometimes sets check-raise traps."),
            book_ref=BookRef(PART_TWO, "The super-aggressive style"),
            expected=Expected(vpip=(0.34, 0.58), pfr=(0.28, 0.48), aggression=(2.5, 15.0),
                              fold_to_reraise=(0.15, 0.50)),
            counter_strategy=("The book's Hammer: when it keeps raising into you, come over the top with an "
                              "all-in reraise using a wider range than usual. Let it bluff into your strong hands."),
            tells=("Raise-first rate is very high, especially from late seats.",
                   "Reraises far more often than other players.",
                   "Shows down hands it had no business raising, like low offsuit cards."),
        ),
        StyleProfile(
            id="station",
            name="Calling station",
            family="Loose-passive",
            description="Plays many hands but mostly calls, and rarely raises or folds once it sees a flop.",
            book_ref=BookRef(PART_THREE, "Loose callers"),
            expected=Expected(vpip=(0.30, 0.70), pfr=(0.0, 0.04), aggression=(0.0, 0.4),
                              fold_to_reraise=None),
            counter_strategy=("Do not bluff it. Value-bet your good hands bigger and more often, because it "
                              "will call with worse."),
            tells=("Hands played is high but raise-first is near zero.",
                   "Calls far more than it bets or raises.",
                   "Shows down weak pairs and missed draws."),
        ),
        StyleProfile(
            id="weak_tight",
            name="Weak-tight",
            family="Tight-passive",
            description="Plays a tight range, makes small raises, and backs down when someone reraises big.",
            book_ref=BookRef(PART_THREE, "Tight players who back down"),
            expected=Expected(vpip=(0.05, 0.17), pfr=(0.03, 0.12), aggression=(0.5, 6.0),
                              fold_to_reraise=(0.50, 1.0)),
            counter_strategy=("Reraise its opens and bet into it after the flop: it folds everything but its "
                              "very best hands. Respect it when it finally plays back."),
            tells=("Hands played is low, close to a Rock's.",
                   "Folds to almost every reraise.",
                   "Rarely reaches showdown, and then with a premium pair."),
        ),
        StyleProfile(
            id="switcher",
            name="Style switcher",
            family="Varying",
            description=("Starts conservative, then turns aggressive as the blinds rise and its stack shrinks, "
                         "moving all-in or folding when short."),
            book_ref=BookRef(PART_TWO, "Varying your style"),
            expected=Expected(vpip=(0.10, 0.50), pfr=(0.08, 0.45), aggression=(1.5, 30.0),
                              fold_to_reraise=None),
            counter_strategy=("Track its stack against the blinds: early it is a Rock, later a blind stealer. "
                              "Adjust as its stack-to-blinds ratio falls."),
            tells=("Stats drift over the table: tight early, loose late.",
                   "Raise-first rises once the blinds go up.",
                   "Short-stacked, it moves all-in rather than raising small."),
            advanced=True,
        ),
    )
}

RANDOM_POOL_DEFAULT: tuple[str, ...] = tuple(p.id for p in PROFILES.values() if not p.advanced)

STAT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "vpip": {"label": "Hands played", "book_ref": asdict(BookRef(PART_THREE, "How many hands they play")),
             "help": "How often the player chooses to put chips in before the flop (blinds do not count)."},
    "pfr": {"label": "Raise-first", "book_ref": asdict(BookRef(PART_THREE, "Callers or raisers")),
            "help": "How often the player raises before the flop."},
    "aggression": {"label": "Callers or raisers", "book_ref": asdict(BookRef(PART_THREE, "Callers or raisers")),
                   "help": "Bets and raises divided by calls. Above 1 means the player raises more than it calls."},
    "fold_to_reraise": {"label": "Reraises", "book_ref": asdict(BookRef(PART_THREE, "Responding to a reraise")),
                        "help": "After betting or raising and being reraised, how often the player folds."},
    "showdowns": {"label": "Showdowns", "book_ref": asdict(BookRef(PART_THREE, "What they show down")),
                  "help": "Hands the player showed at the end, with the line they took to get there."},
}


def style_of_policy(policy: str | None) -> str | None:
    if policy is None or not policy.startswith(POLICY_PREFIX) or policy == RANDOM_POLICY:
        return None
    sid = policy[len(POLICY_PREFIX):]
    return sid if sid in PROFILES else None
