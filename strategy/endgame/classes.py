"""Private-hand classes for a variant, derived from its rules (decision-time safe).

A class is a suit-isomorphism class of hole cards (169 in hold'em, 81 in short deck).
Classes are indexed in ascending canonical-key order, the same order as the preflop
equity table. Variants whose class count is too large for dense class-by-class showdown
matrices (four-card Omaha has 16,432) are rejected rather than silently bucketed with
another variant's assets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

from strategy.features.cards import CardCapabilities, canonical_hole_key, capabilities, hole_label
from strategy.rules.variants import CAP_ANY_FIVE, CAP_NO_LIMIT, CAP_TWO_HOLE, get_variant

MAX_DENSE_CLASSES = 256
REQUIRED_CAPABILITIES = (CAP_TWO_HOLE, CAP_ANY_FIVE)
"""Showdown tables enumerate two-card combos and use the best-five evaluator."""
PUSH_FOLD_CAPABILITIES = (*REQUIRED_CAPABILITIES, CAP_NO_LIMIT)
"""A shove must be a legal single action, so push/fold needs no-limit betting."""


@dataclass(frozen=True)
class HandClasses:
    ruleset_id: str
    variant_hash: str
    keys: tuple[int, ...]
    labels: tuple[str, ...]
    combos: tuple[tuple[tuple[int, int], ...], ...]
    """Card pairs per class, ascending."""
    caps: CardCapabilities = field(repr=False, compare=False)
    _index: dict[int, int] = field(repr=False, compare=False, default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.keys)

    @property
    def combo_counts(self) -> tuple[int, ...]:
        return tuple(len(c) for c in self.combos)

    @property
    def total_combos(self) -> int:
        return sum(self.combo_counts)

    def index(self, hole) -> int:
        """Class index of two hole cards (card ints)."""
        return self._index[canonical_hole_key(hole)]

    def label_index(self, label: str) -> int:
        return self.labels.index(label)


@lru_cache(maxsize=None)
def hand_classes(ruleset_id: str) -> HandClasses:
    variant = get_variant(ruleset_id)
    variant.require(*REQUIRED_CAPABILITIES, asset="endgame hand classes")
    caps = capabilities(ruleset_id)
    groups: dict[int, list[tuple[int, int]]] = {}
    for a, b in combinations(caps.deck, 2):
        groups.setdefault(canonical_hole_key((a, b)), []).append((a, b))
    keys = tuple(sorted(groups))
    if len(keys) > MAX_DENSE_CLASSES:
        raise ValueError(f"{ruleset_id}: {len(keys)} hand classes exceed the dense limit {MAX_DENSE_CLASSES}")
    labels = tuple(hole_label(groups[k][0]) for k in keys)
    combos = tuple(tuple(sorted(groups[k])) for k in keys)
    return HandClasses(ruleset_id, variant.variant_hash(), keys, labels, combos, caps,
                       {k: i for i, k in enumerate(keys)})
