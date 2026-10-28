"""Baseline: a simple, hand-written NLHE heuristic. Not a strong or tuned strategy.

Preflop it scores the two hole cards with a simplified Chen formula. Postflop it
counts whether the hole cards improve the board to a pair, two pair or better.
Strong hands bet about two thirds of the pot, medium hands call cheap bets, and
everything else checks or folds. Supports ``practice_nlhe`` only.
"""

from collections import Counter

from arena.agents.sdk import ExplicitStateBot, call, check, check_or_fold, clamp_raise_to

RANKS = "23456789TJQKA"


def _rank(card):
    return RANKS.index(card[0])


def chen_score(c1, c2):
    hi, lo = sorted((_rank(c1), _rank(c2)), reverse=True)
    points = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(hi, (hi + 2) / 2)
    if hi == lo:
        return max(points * 2, 5.0)
    if c1[1] == c2[1]:
        points += 2
    gap = hi - lo - 1
    points -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 10:
        points += 1
    return points


def _category(cards):
    """0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads."""
    if not cards:
        return 0
    counts = sorted(Counter(c[0] for c in cards).values(), reverse=True)
    suits = Counter(c[1] for c in cards)
    ranks = {_rank(c) for c in cards} | ({-1} if any(c[0] == "A" for c in cards) else set())
    straight = any(all(r + i in ranks for i in range(5)) for r in range(-1, 9))
    if counts[0] == 4:
        return 7
    if counts[0] == 3 and len(counts) > 1 and counts[1] >= 2:
        return 6
    if max(suits.values()) >= 5:
        return 5
    if straight:
        return 4
    if counts[0] == 3:
        return 3
    if counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        return 2
    return 1 if counts[0] == 2 else 0


class NlheHeuristic(ExplicitStateBot):
    name = "nlhe_heuristic"
    rulesets = ("practice_nlhe",)

    def decide(self, obs, state):
        legal = obs.legal
        to_call = legal.call_amount or 0
        if obs.street == "preflop":
            score = chen_score(*obs.hole_cards)
            if score >= 10:
                bet = clamp_raise_to(legal, obs.current_bet_to * 3)
                return (bet or (call() if to_call else check())), state
            if score >= 7 and to_call <= obs.blinds.big_blind * 3:
                return (call() if to_call else check()), state
            return check_or_fold(legal), state
        mine = _category(obs.hole_cards + obs.board)
        board = _category(obs.board)
        improved = mine > board
        top_pair = improved and mine == 1 and max(_rank(c) for c in obs.hole_cards) >= max(_rank(c) for c in obs.board)
        if improved and mine >= 2:
            bet = clamp_raise_to(legal, obs.current_bet_to + (2 * (obs.pot_total + to_call)) // 3)
            return (bet or (call() if to_call else check())), state
        if top_pair and to_call * 2 <= obs.pot_total:
            return (call() if to_call else check()), state
        return check_or_fold(legal), state


def make_bot():
    return NlheHeuristic()
