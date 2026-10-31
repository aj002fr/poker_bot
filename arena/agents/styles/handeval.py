"""Small Hold'em hand evaluator for the style bots and the learning aids.

Cards are two-character strings (``"Ah"``, ``"Td"``). :func:`evaluate` ranks any 5-7
cards as a comparable tuple; :func:`describe` and :func:`preflop_words` put a hand in
words; :func:`draws` counts outs; :func:`equity_vs_random` estimates the chance to win
against random hands from the hero's cards and the board only.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from collections.abc import Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RV = {r: i for i, r in enumerate(RANKS)}
DECK: tuple[str, ...] = tuple(r + s for r in RANKS for s in SUITS)

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
RANK_ROWS = ("high_card", "one_pair", "two_pair", "three_of_a_kind", "straight", "flush", "full_house",
             "four_of_a_kind", "straight_flush")

_NAMES = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king", "ace")
_PLURALS = ("twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", "tens", "jacks", "queens",
            "kings", "aces")


def _straight_high(ranks: set[int]) -> int | None:
    for high in range(12, 3, -1):
        if all(r in ranks for r in range(high - 4, high + 1)):
            return high
    if {12, 0, 1, 2, 3} <= ranks:
        return 3
    return None


def evaluate(cards: Sequence[str]) -> tuple[int, ...]:
    """Category first (0 high card .. 8 straight flush), then tie-break ranks."""
    ranks = [RV[c[0]] for c in cards]
    counts = [0] * 13
    for r in ranks:
        counts[r] += 1
    by_suit: dict[str, list[int]] = {}
    for c in cards:
        by_suit.setdefault(c[1], []).append(RV[c[0]])
    flush = next((sorted(rs, reverse=True) for rs in by_suit.values() if len(rs) >= 5), None)
    if flush is not None:
        sf = _straight_high(set(flush))
        if sf is not None:
            return (STRAIGHT_FLUSH, sf)
    quads = [r for r in range(12, -1, -1) if counts[r] == 4]
    trips = [r for r in range(12, -1, -1) if counts[r] == 3]
    pairs = [r for r in range(12, -1, -1) if counts[r] == 2]
    if quads:
        q = quads[0]
        return (QUADS, q, max(r for r in ranks if r != q))
    if trips and (len(trips) > 1 or pairs):
        t = trips[0]
        return (FULL_HOUSE, t, max([*trips[1:], *pairs]))
    if flush is not None:
        return (FLUSH, *flush[:5])
    st = _straight_high(set(ranks))
    if st is not None:
        return (STRAIGHT, st)
    singles = sorted((r for r in range(13) if counts[r] == 1), reverse=True)
    if trips:
        return (TRIPS, trips[0], *singles[:2])
    if len(pairs) >= 2:
        kick = max([*pairs[2:], *singles], default=-1)
        return (TWO_PAIR, pairs[0], pairs[1], kick)
    if pairs:
        return (PAIR, pairs[0], *singles[:3])
    return (HIGH_CARD, *singles[:5])


def describe(value: tuple[int, ...]) -> str:
    cat = value[0]
    n = _NAMES
    p = _PLURALS
    if cat == STRAIGHT_FLUSH:
        return "Royal flush" if value[1] == 12 else f"Straight flush, {n[value[1]]} high"
    if cat == QUADS:
        return f"Four of a kind, {p[value[1]]}"
    if cat == FULL_HOUSE:
        return f"Full house, {p[value[1]]} full of {p[value[2]]}"
    if cat == FLUSH:
        return f"Flush, {n[value[1]]} high"
    if cat == STRAIGHT:
        return f"Straight, {n[value[1]]} high"
    if cat == TRIPS:
        return f"Three of a kind, {p[value[1]]}"
    if cat == TWO_PAIR:
        return f"Two pair, {p[value[1]]} and {p[value[2]]}"
    if cat == PAIR:
        kick = f", {n[value[2]]} kicker" if len(value) > 2 else ""
        return f"Pair of {p[value[1]]}{kick}"
    kick = f", {n[value[2]]} kicker" if len(value) > 2 else ""
    return f"{n[value[1]].capitalize()} high{kick}"


def preflop_words(hole: Sequence[str]) -> str:
    a, b = sorted(hole, key=lambda c: RV[c[0]], reverse=True)
    if a[0] == b[0]:
        return f"Pair of {_PLURALS[RV[a[0]]]}"
    kind = "suited" if a[1] == b[1] else "offsuit"
    return f"{_NAMES[RV[a[0]]].capitalize()}-{_NAMES[RV[b[0]]]} {kind}"


def hand_words(hole: Sequence[str], board: Sequence[str]) -> tuple[str, str]:
    """The hand in words and its hand-rankings row id."""
    if not board:
        return preflop_words(hole), RANK_ROWS[PAIR if hole[0][0] == hole[1][0] else HIGH_CARD]
    value = evaluate([*hole, *board])
    return describe(value), RANK_ROWS[value[0]]


def draws(hole: Sequence[str], board: Sequence[str], *, exact: bool = True) -> tuple[list[dict], int]:
    """Flush and straight draws on the flop or turn, and the outs that complete one.

    ``exact`` counts the unseen cards that make a straight or flush; otherwise the
    per-draw counts are summed (fast enough for a bot's 1 ms budget).
    """
    if len(board) not in (3, 4):
        return [], 0
    known = [*hole, *board]
    current = evaluate(known)[0]
    unseen = [c for c in DECK if c not in known]
    out: list[dict] = []
    suits = {s: sum(1 for c in known if c[1] == s) for s in SUITS}
    flush_suit = next((s for s, k in suits.items() if k == 4 and any(c[1] == s for c in hole)), None)
    if flush_suit is not None and current < FLUSH:
        n = sum(1 for c in unseen if c[1] == flush_suit)
        out.append({"kind": "flush_draw", "outs": n, "words": f"Flush draw, {n} outs"})
    if current < STRAIGHT:
        ranks = {RV[c[0]] for c in known}
        board_ranks = {RV[c[0]] for c in board}
        completing = [r for r in range(13) if r not in ranks and _straight_high(ranks | {r}) is not None
                      and _straight_high(board_ranks | {r}) is None]
        if completing:
            n = sum(1 for c in unseen if RV[c[0]] in completing)
            if len(completing) >= 2:
                out.append({"kind": "open_ended", "outs": n, "words": f"Open-ended straight draw, {n} outs"})
            else:
                out.append({"kind": "gutshot", "outs": n, "words": f"Gutshot straight draw, {n} outs"})
    if not exact:
        return out, sum(d["outs"] for d in out)
    total = 0
    if out:
        for c in unseen:
            cat = evaluate([*known, c])[0]
            if cat in (STRAIGHT, FLUSH, STRAIGHT_FLUSH) and cat > current:
                total += 1
    return out, total


# -------------------------------------------------------------- bot buckets


def strength(hole: Sequence[str], board: Sequence[str]) -> tuple[str, int, tuple[int, ...]]:
    """(bucket, draw outs, value). Buckets: monster, strong, medium, weak; hero cards must matter."""
    value = evaluate([*hole, *board])
    cat = value[0]
    bcat = evaluate(list(board))[0] if board else HIGH_CARD
    top = max(RV[c[0]] for c in board)
    board_ranks = sorted({RV[c[0]] for c in board}, reverse=True)
    h = sorted((RV[c[0]] for c in hole), reverse=True)
    _, outs = draws(hole, board, exact=False)
    if cat >= STRAIGHT and cat > bcat:
        return "monster", outs, value
    if cat == TRIPS and bcat < TRIPS:
        return "monster", outs, value
    if cat == TWO_PAIR and bcat < PAIR:
        return "monster", outs, value
    if cat == TWO_PAIR and bcat == PAIR:
        mine = h[0] if h[0] == h[1] else next((r for r in h if r in board_ranks), None)
        if mine is None:
            return "weak", outs, value
        return ("strong" if mine >= top else "medium"), outs, value
    if cat == PAIR and bcat == HIGH_CARD:
        if h[0] == h[1]:
            if h[0] > top:
                return "strong", outs, value
            return ("medium" if len(board_ranks) < 2 or h[0] > board_ranks[1] else "weak"), outs, value
        paired = next(r for r in h if r in board_ranks)
        kicker = h[1] if paired == h[0] else h[0]
        if paired == top:
            return ("strong" if kicker >= RV["T"] or paired == RV["A"] else "medium"), outs, value
        if len(board_ranks) > 1 and paired == board_ranks[1]:
            return "medium", outs, value
        return "weak", outs, value
    return "weak", outs, value


# ---------------------------------------------------------------- equity


def _seed(hole: Sequence[str], board: Sequence[str], opponents: int) -> int:
    key = f"arena-equity-v1|{','.join(sorted(hole))}|{','.join(board)}|{opponents}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def equity_vs_random(hole: Sequence[str], board: Sequence[str], opponents: int = 1,
                     samples: int = 1500) -> tuple[float, int, bool]:
    """(equity, samples, exact). Ties count as a share. Seeded from the inputs only."""
    known = [*hole, *board]
    deck = [c for c in DECK if c not in known]
    need = 5 - len(board)
    opponents = max(1, opponents)
    if need == 0 and opponents == 1:
        hero = evaluate(known)
        total = 0.0
        combos = list(itertools.combinations(deck, 2))
        for opp in combos:
            v = evaluate([*opp, *board])
            total += 1.0 if hero > v else (0.5 if hero == v else 0.0)
        return total / len(combos), len(combos), True
    rng = random.Random(_seed(hole, board, opponents))
    total = 0.0
    take = need + 2 * opponents
    for _ in range(samples):
        draw = rng.sample(deck, take)
        full = [*board, *draw[:need]]
        hero = evaluate([*hole, *full])
        ties = 0
        lost = False
        for i in range(opponents):
            v = evaluate([draw[need + 2 * i], draw[need + 2 * i + 1], *full])
            if v > hero:
                lost = True
                break
            if v == hero:
                ties += 1
        if not lost:
            total += 1.0 / (ties + 1)
    return total / samples, samples, False
