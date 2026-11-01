"""Range features: where the hero's hand sits in its own public range, and the field's ranges.

The hero's *public* range is the hero's bucket weights after updating on its own actions
with the same model opponents could build (the hero seat's action posterior), so
``own_rank`` (``q_H``) is the share of that range the hero's actual hand beats: near 1
the hand is the top of what the line represents, near 0 it is the bottom (a bluff or a
give-up candidate). Opponent summaries come from the same bucketed beliefs.

Buckets are preflop strength deciles, so postflop these are range-strength *proxies*:
``range_advantage`` and ``nut_advantage`` compare how much strong preflop holding each
side's line retains, not current hand values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from strategy.beliefs.ranges import RangeBeliefs, RangeSummary


@dataclass(frozen=True, slots=True)
class OpponentRange:
    seat: int
    range: RangeSummary
    family: str
    family_prob: float
    fold_to_raise: float
    """Predicted probability this seat folds if the hero bets or raises now."""
    sample: float
    """Effective observations behind that prediction (seat level, this opportunity)."""


@dataclass(frozen=True, slots=True)
class RangeFeatures:
    hero_bucket: int
    hero_percentile: float
    own: RangeSummary | None
    own_rank: float
    """``q_H``: the hero bucket's mid-rank within the hero's public range, 0..1."""
    own_middle: float
    main_seat: int
    """The aggressor if it is an opponent, else the live opponent with the strongest range (-1: none)."""
    main: RangeSummary | None
    field_top: float
    """Probability at least one live opponent holds a top-decile hand: ``1 - prod(1 - top_j)``."""
    field_mean: float
    field_strongest: float
    range_advantage: float
    """Hero public-range mean minus the main opponent's."""
    nut_advantage: float
    """Hero public top-decile mass minus the largest opponent's."""
    capped_opponents: int
    fold_all: float
    """Probability every live opponent folds to a bet or raise now (independence approximation)."""
    opponents: tuple[OpponentRange, ...]


def own_rank(weights: Sequence[float], bucket: int) -> float:
    total = sum(weights) or 1.0
    below = sum(weights[:bucket])
    return (below + weights[bucket] / 2) / total


def range_features(beliefs: RangeBeliefs, hero: int, hero_bucket: int, hero_percentile: float,
                   opponents: Sequence[int], aggressor: int, fold: Mapping[int, tuple[float, float]],
                   families: Mapping[int, tuple[str, float]]) -> RangeFeatures:
    own = beliefs.summary(hero)
    rank = own_rank(own.weights, hero_bucket) if own is not None else hero_percentile
    opp: list[OpponentRange] = []
    miss_top = 1.0
    fold_all = 1.0
    strongest = -1.0
    strongest_seat = -1
    top_max = 0.0
    capped = 0
    mean_sum = 0.0
    for s in opponents:
        r = beliefs.summary(s)
        if r is None:
            continue
        f, n = fold.get(s, (0.0, 0.0))
        fam, fp = families.get(s, ("unknown", 1.0))
        opp.append(OpponentRange(s, r, fam, fp, f, n))
        miss_top *= 1.0 - r.top
        fold_all *= f
        mean_sum += r.mean
        if r.mean > strongest:
            strongest, strongest_seat = r.mean, s
        if r.top > top_max:
            top_max = r.top
        capped += r.capped
    main_seat = aggressor if aggressor != hero and any(o.seat == aggressor for o in opp) else strongest_seat
    main = next((o.range for o in opp if o.seat == main_seat), None)
    own_mean = own.mean if own is not None else hero_percentile
    ref = main.mean if main is not None else strongest if opp else 0.5
    return RangeFeatures(
        hero_bucket, hero_percentile, own, rank, own.middle if own is not None else 0.0, main_seat, main,
        1.0 - miss_top if opp else 0.0, mean_sum / len(opp) if opp else 0.0, max(strongest, 0.0),
        own_mean - ref, (own.top if own is not None else 0.0) - top_max, capped, fold_all if opp else 0.0,
        tuple(opp))
