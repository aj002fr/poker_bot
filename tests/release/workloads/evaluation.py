"""Evaluation gates: the engine's evaluator against the independent oracles.

The engine side is exactly what the adapter calls at showdown
(``State.get_hand``): the ruleset's PokerKit hand type, ``from_game(hole, board)``.
The oracle side is ``tests/oracles`` (no PokerKit).

Each shard maps every oracle hand value it sees (strength, tiebreak) to the
engine's lookup entry index and label. A value seen with two engine indexes is
a failure inside the shard. :func:`finalize` merges the maps of all shards and
checks that, ordered by oracle value, the engine indexes are strictly
increasing. That is exact rank agreement including ties across the whole
workload, not just inside one shard. Category labels must match too.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from itertools import combinations, islice
from typing import Any

from pokerkit import Card as PkCard

from arena.rules.profiles import get_profile
from tests.oracles.cards import Card, parse_cards, short_deck, standard_deck
from tests.oracles.omaha import best_omaha
from tests.oracles.ranking import Ranking, best_holdem, rank_five_unchecked
from tests.oracles.shortdeck import shortdeck_ranking
from tests.oracles.standard import STANDARD
from tests.release.workloads.common import CaseRunner, digest

# Section 13 table, and the oracle's independently enumerated short-deck census.
NLHE_FIVE_CARD_TOTALS = {
    "high_card": 1_302_540, "one_pair": 1_098_240, "two_pair": 123_552, "three_of_a_kind": 54_912,
    "straight": 10_200, "flush": 5_108, "full_house": 3_744, "four_of_a_kind": 624, "straight_flush": 40,
}
SHORTDECK_FIVE_CARD_TOTALS = {
    "high_card": 122_400, "one_pair": 193_536, "two_pair": 36_288, "three_of_a_kind": 16_128,
    "straight": 6_120, "flush": 480, "full_house": 1_728, "four_of_a_kind": 288, "straight_flush": 24,
}

GATES = {
    "nlhe_eval": {"ruleset": "practice_nlhe", "deck": standard_deck, "ranking": STANDARD,
                  "five_total": 2_598_960, "five_classes": 7462, "five_totals": NLHE_FIVE_CARD_TOTALS},
    "shortdeck_eval": {"ruleset": "uncertainty_shortdeck", "deck": short_deck, "ranking": shortdeck_ranking(),
                       "five_total": 376_992, "five_classes": None, "five_totals": SHORTDECK_FIVE_CARD_TOTALS},
    "plo_eval": {"ruleset": "uncertainty_plo4", "deck": standard_deck, "ranking": STANDARD},
}

_PK = {str(c): next(iter(PkCard.parse(str(c)))) for c in standard_deck()}


def _hand_type(ruleset: str):
    return get_profile(ruleset).game_class.hand_types[0]


def _label(entry) -> str:
    return entry.label.value.lower().replace(" ", "_")


def _key(value) -> str:
    return f"{value.strength}:{'.'.join(map(str, value.tiebreak))}"


def _cards_text(cards) -> str:
    return " ".join(map(str, cards))


class ClassMap:
    """oracle value key -> [category, engine index, engine label, count]."""

    def __init__(self, runner: CaseRunner) -> None:
        self.runner = runner
        self.classes: dict[str, list[Any]] = {}

    def add(self, value, entry, case: dict[str, Any]) -> None:
        key = _key(value)
        index, label = entry.index, _label(entry)
        if label != value.category:
            self.runner.fail(case, f"engine category {label}, oracle {value.category}")
        row = self.classes.get(key)
        if row is None:
            self.classes[key] = [value.category, index, label, 1]
        elif row[1] != index:
            self.runner.fail(case, f"oracle value {key} maps to engine entries {row[1]} and {index}")
        else:
            row[3] += 1


# --------------------------------------------------------------- generators


def _straights(ranking: Ranking) -> list[frozenset[int]]:
    return sorted(ranking.straights, key=lambda s: ranking.straights[s])


def _made_board(rng: random.Random, deck: list[Card], ranking: Ranking) -> list[Card]:
    """Five board cards that already make a strong hand (board-only cases)."""
    kind = rng.choice(["straight", "flush", "full_house", "quads", "straight_flush"])
    suits = "cdhs"
    if kind in ("straight", "straight_flush"):
        ranks = sorted(rng.choice(_straights(ranking)))
        suit = rng.choice(suits)
        return [Card(r, suit if kind == "straight_flush" else rng.choice(suits)) for r in ranks]
    if kind == "flush":
        suit = rng.choice(suits)
        return rng.sample([c for c in deck if c.suit == suit], 5)
    by_rank: dict[int, list[Card]] = {}
    for c in deck:
        by_rank.setdefault(c.rank, []).append(c)
    a, b = rng.sample(sorted(by_rank), 2)
    if kind == "quads":
        return by_rank[a] + [rng.choice(by_rank[b])]
    return rng.sample(by_rank[a], 3) + rng.sample(by_rank[b], 2)


def _take(rng: random.Random, deck: list[Card], used: list[Card], k: int) -> list[Card]:
    rest = [c for c in deck if c not in used]
    return rng.sample(rest, k)


def holdem_sample(rng: random.Random, deck: list[Card], ranking: Ranking, kind: str) -> tuple[list[Card], list[Card]]:
    if kind == "board_made":
        board = _made_board(rng, deck, ranking)
        return _take(rng, deck, board, 2), board
    if kind == "low_straight":
        low = _straights(ranking)[0]
        picked = [Card(r, rng.choice("cdhs")) for r in sorted(low)]
        cards = picked + _take(rng, deck, picked, 2)
    elif kind == "flushy":
        suit = rng.choice("cdhs")
        picked = rng.sample([c for c in deck if c.suit == suit], rng.choice([5, 6, 7]))
        cards = picked + _take(rng, deck, picked, 7 - len(picked))
    elif kind == "paired":
        ranks = rng.sample(sorted({c.rank for c in deck}), 3)
        pool = [c for c in deck if c.rank in ranks]
        picked = rng.sample(pool, 6)
        cards = picked + _take(rng, deck, picked, 1)
    else:
        cards = rng.sample(deck, 7)
    rng.shuffle(cards)
    return cards[:2], cards[2:]


def omaha_sample(rng: random.Random, deck: list[Card], kind: str) -> tuple[list[Card], list[Card]]:
    if kind == "false_flush":
        suit = rng.choice("cdhs")
        suited = [c for c in deck if c.suit == suit]
        board_suited = rng.choice([3, 4, 5])
        board = rng.sample(suited, board_suited)
        board += _take(rng, [c for c in deck if c.suit != suit], board, 5 - board_suited)
        hole = [rng.choice([c for c in suited if c not in board])]
        hole += _take(rng, [c for c in deck if c.suit != suit], board + hole, 3)
    elif kind == "board_only":
        board = _made_board(rng, deck, STANDARD)
        hole = _take(rng, deck, board, 4)
    elif kind == "real_flush":
        suit = rng.choice("cdhs")
        suited = [c for c in deck if c.suit == suit]
        picked = rng.sample(suited, 5)
        board = picked[:3] + _take(rng, deck, picked, 2)
        hole = picked[3:] + _take(rng, deck, picked + board, 2)
    elif kind == "hole_heavy":
        rank = rng.choice(range(2, 15))
        same = [c for c in deck if c.rank == rank]
        hole = rng.sample(same, rng.choice([2, 3, 4]))
        hole += _take(rng, deck, hole, 4 - len(hole))
        board = _take(rng, deck, hole, 5)
    else:
        cards = rng.sample(deck, 9)
        hole, board = cards[:4], cards[4:]
    rng.shuffle(hole)
    rng.shuffle(board)
    return hole, board


HOLDEM_KINDS = ("random", "random", "board_made", "low_straight", "flushy", "paired")
OMAHA_KINDS = ("random", "random", "false_flush", "board_only", "false_flush", "board_only", "real_flush", "hole_heavy")


# ------------------------------------------------------------------- shards


def _pk(cards) -> list:
    return [_PK[str(c)] for c in cards]


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    gate = params["gate"]
    g = GATES[gate]
    ranking = g["ranking"]
    hand_type = _hand_type(g["ruleset"])
    deck = list(g["deck"]())
    runner = CaseRunner(gate)
    cmap = ClassMap(runner)
    kind = params["kind"]
    if kind == "five":
        start, stop = params["start"], params["stop"]
        for combo in islice(combinations(range(len(deck)), 5), start, stop):
            cards = [deck[i] for i in combo]
            value = rank_five_unchecked(cards, ranking)
            entry = hand_type.from_game(_pk(cards[:2]), _pk(cards[2:])).entry
            runner.counts["five_card_hands"] += 1
            runner.counts[f"five:{value.category}"] += 1
            cmap.add(value, entry, {"kind": "five", "cards": _cards_text(cards)})
    elif kind == "seven":
        rng = random.Random(f"arena-release|{gate}|seven|{params['seed']}")
        for n in range(params["samples"]):
            sample_kind = HOLDEM_KINDS[n % len(HOLDEM_KINDS)]
            hole, board = holdem_sample(rng, deck, ranking, sample_kind)
            value = best_holdem(hole, board, ranking).value
            entry = hand_type.from_game(_pk(hole), _pk(board)).entry
            runner.counts["seven_card_samples"] += 1
            runner.counts[f"seven_kind:{sample_kind}"] += 1
            cmap.add(value, entry, {"kind": "seven", "hole": _cards_text(hole), "board": _cards_text(board)})
    elif kind == "omaha":
        rng = random.Random(f"arena-release|{gate}|omaha|{params['seed']}")
        seen: set[str] = set()
        n = 0
        while len(seen) < params["samples"]:
            sample_kind = OMAHA_KINDS[n % len(OMAHA_KINDS)]
            hole, board = omaha_sample(rng, deck, sample_kind)
            key = " ".join(sorted(map(str, hole))) + "|" + " ".join(sorted(map(str, board)))
            # Each shard owns one residue class of sample keys, so samples are distinct across shards.
            owner = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % params["partitions"]
            if owner != params["partition"] or key in seen:
                runner.counts["omaha_rejected_duplicate_or_other_partition"] += 1
                continue
            seen.add(key)
            n += 1
            value = best_omaha(hole, board).value
            entry = hand_type.from_game(_pk(hole), _pk(board)).entry
            runner.counts["omaha_samples"] += 1
            runner.counts[f"omaha_kind:{sample_kind}"] += 1
            if sample_kind == "false_flush" and value.category in ("flush", "straight_flush"):
                runner.fail({"kind": "omaha", "hole": _cards_text(hole), "board": _cards_text(board)},
                            "false-flush generator produced a real flush")
            cmap.add(value, entry, {"kind": "omaha", "hole": _cards_text(hole), "board": _cards_text(board)})
    else:
        raise ValueError(kind)
    out = runner.result(classes=cmap.classes, lookup=type(hand_type.lookup).__name__)
    out["result_hash"] = digest([out["counts"], cmap.classes, runner.failed_cases])
    return out


def run_case(case: dict[str, Any]) -> str:
    """Re-evaluate one recorded hand on both sides and fail on a category mismatch."""
    g = GATES[case["gate"]]
    hand_type = _hand_type(g["ruleset"])
    if case["kind"] == "five":
        cards = parse_cards(case["cards"])
        value = rank_five_unchecked(cards, g["ranking"])
        entry = hand_type.from_game(_pk(cards[:2]), _pk(cards[2:])).entry
    else:
        hole, board = parse_cards(case["hole"]), parse_cards(case["board"])
        value = (best_omaha(hole, board) if case["kind"] == "omaha" else best_holdem(hole, board, g["ranking"])).value
        entry = hand_type.from_game(_pk(hole), _pk(board)).entry
    print(f"oracle {value}\nengine index {entry.index} label {_label(entry)}")
    assert _label(entry) == value.category, "category differs"
    return f"{_key(value)}={entry.index}"


# ----------------------------------------------------------------- finalize


def _merge(results: list[dict[str, Any]], kinds: set[str]) -> tuple[dict[str, list[Any]], list[str]]:
    merged: dict[str, list[Any]] = {}
    problems = []
    for r in results:
        if r["params"]["kind"] not in kinds:
            continue
        for key, (category, index, label, count) in r["classes"].items():
            row = merged.get(key)
            if row is None:
                merged[key] = [category, index, label, count]
            elif row[1] != index:
                problems.append(f"oracle value {key} maps to engine entries {row[1]} and {index} in different shards")
            else:
                row[3] += count
    return merged, problems


def _monotone(merged: dict[str, list[Any]]) -> list[str]:
    def order(key: str):
        strength, tiebreak = key.split(":")
        return int(strength), tuple(int(x) for x in tiebreak.split(".") if x)

    keys = sorted(merged, key=order)
    problems = []
    for a, b in zip(keys, keys[1:]):
        if not merged[a][1] < merged[b][1]:
            problems.append(f"oracle {a} < {b} but engine entries {merged[a][1]} >= {merged[b][1]}")
    return problems


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Gate-level checks over every shard (and the NLHE map for PLO, which shares PokerKit's StandardLookup)."""
    g = GATES[gate]
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    totals: Counter = Counter()
    for r in results:
        totals.update(r["counts"])
    if gate in ("nlhe_eval", "shortdeck_eval"):
        five, p5 = _merge(results, {"five"})
        seven, p7 = _merge(results, {"seven"})
        check("five-card combinations evaluated", totals["five_card_hands"] == g["five_total"],
              f"{totals['five_card_hands']:,} of {g['five_total']:,}")
        cats = {c: totals.get(f"five:{c}", 0) for c in g["five_totals"]}
        check("five-card category census", cats == g["five_totals"], cats)
        if g["five_classes"]:
            check("five-card equivalence classes", len(five) == g["five_classes"], f"{len(five):,} classes")
        else:
            check("five-card equivalence classes", len(five) > 0, f"{len(five):,} classes")
        problems = p5 + _monotone(five)
        check("five-card ranks strictly ordered and ties equal (global)", not problems, problems[:10] or "ok")
        check("seven-card samples (>= 100,000)", totals["seven_card_samples"] >= 100_000,
              f"{totals['seven_card_samples']:,}")
        disagree = [k for k, row in seven.items() if k not in five or five[k][1] != row[1]] + p7
        check("seven-card best-of-21 agrees with the five-card map (rank and ties)", not disagree,
              disagree[:10] or f"{len(seven):,} classes seen, all agree")
    else:
        omaha, p = _merge(results, {"omaha"})
        check("distinct PLO hole/board samples (>= 100,000)", totals["omaha_samples"] >= 100_000,
              f"{totals['omaha_samples']:,} (shards own disjoint key partitions)")
        for kind in ("false_flush", "board_only"):
            n = totals.get(f"omaha_kind:{kind}", 0)
            check(f"targeted {kind.replace('_', '-')} samples (>= 10,000)", n >= 10_000, f"{n:,}")
        problems = p + _monotone(omaha)
        check("PLO ranks strictly ordered and ties equal (global)", not problems, problems[:10] or "ok")
        nlhe = all_results.get("nlhe_eval", [])
        if nlhe and {r["lookup"] for r in results} == {r["lookup"] for r in nlhe}:
            five, _ = _merge(nlhe, {"five"})
            disagree = [k for k, row in omaha.items() if k not in five or five[k][1] != row[1]]
            check("PLO entries agree with the exhaustive NLHE five-card map (same StandardLookup)", not disagree,
                  disagree[:10] or f"{len(omaha):,} classes seen, all agree")
        else:
            check("PLO entries agree with the exhaustive NLHE five-card map", False, "NLHE five-card shards missing")
    return checks


def shards(gate: str) -> list[dict[str, Any]]:
    g = GATES[gate]
    out = []
    if gate == "plo_eval":
        for k in range(10):
            out.append({"gate": gate, "kind": "omaha", "seed": k, "samples": 20_000, "partition": k, "partitions": 10})
        return out
    parts = 6 if gate == "nlhe_eval" else 2
    step = -(-g["five_total"] // parts)
    for k in range(parts):
        out.append({"gate": gate, "kind": "five", "start": k * step, "stop": min(g["five_total"], (k + 1) * step)})
    for k in range(8):
        out.append({"gate": gate, "kind": "seven", "seed": k, "samples": 25_000})
    return out
