"""Rules layer: variant metadata, rules hash, canonical legal actions and conformance.

``variants``, ``adapter`` and ``canonical`` are decision-time safe. ``conformance``
is offline and may import the arena engine.
"""

from strategy.rules.adapter import (
    CanonicalAction,
    RulesAdapter,
    TournamentStructure,
    canonical_actions,
    default_action,
    rules_hash,
)
from strategy.rules.variants import KNOWN_VARIANTS, UnsupportedVariant, VariantSpec, get_variant

__all__ = [
    "KNOWN_VARIANTS",
    "CanonicalAction",
    "RulesAdapter",
    "TournamentStructure",
    "UnsupportedVariant",
    "VariantSpec",
    "canonical_actions",
    "default_action",
    "get_variant",
    "rules_hash",
]
