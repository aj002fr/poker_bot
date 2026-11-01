"""Chip-denominated betting arithmetic: pot odds, defence frequencies and bluff break-evens.

Every function accepts any real numbers (``int``, ``Fraction`` or ``float``) and uses
only field operations, so exact inputs give exact outputs. Amounts are chips or any
common unit; only ratios matter. These are terminal, heads-up, chip-EV identities:
tournament points, future streets, raises and other live players need the fuller
models in :mod:`strategy.math.points` or a solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from numbers import Rational, Real


def div(a: Real, b: Real) -> Real:
    """``a / b`` that stays exact when both operands are integers or fractions."""
    if isinstance(a, Rational) and isinstance(b, Rational):
        return Fraction(a) / Fraction(b)
    return a / b


def _nonneg(name: str, value: Real) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _probability(name: str, value: Real) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def pot_odds_threshold(pot: Real, call: Real) -> Real:
    """Showdown equity at which calling ``call`` to win ``pot`` breaks even in chips.

    ``pot`` is everything the caller can win before calling, including the
    opponent's bet; only contestable chips belong in it (see :func:`effective_pot_odds`).
    """
    _nonneg("pot", pot)
    _nonneg("call", call)
    if pot + call == 0:
        raise ValueError("pot and call cannot both be zero")
    return div(call, pot + call)


def call_ev(equity: Real, pot: Real, call: Real) -> Real:
    """Chip EV of calling, relative to folding, with no future betting."""
    _probability("equity", equity)
    _nonneg("pot", pot)
    _nonneg("call", call)
    return equity * pot - (1 - equity) * call


def effective_pot_odds(pot: Real, to_call: Real, hero_stack: Real) -> Real:
    """Pot-odds threshold when the caller may be unable to match the whole bet.

    ``pot`` includes the opponent's full bet and ``to_call`` is the nominal price.
    A caller with fewer chips matches only ``min(to_call, hero_stack)``; the
    unmatched part of the bet is returned and cannot be won.
    """
    _nonneg("to_call", to_call)
    _nonneg("hero_stack", hero_stack)
    matched = min(to_call, hero_stack)
    unmatched = to_call - matched
    if unmatched > pot:
        raise ValueError("pot must include the full bet being called")
    return pot_odds_threshold(pot - unmatched, matched)


def bluff_break_even(bet: Real, pot: Real) -> Real:
    """Fold probability at which a zero-equity bet of ``bet`` into ``pot`` breaks even."""
    _nonneg("bet", bet)
    _nonneg("pot", pot)
    if bet + pot == 0:
        raise ValueError("bet and pot cannot both be zero")
    return div(bet, pot + bet)


def minimum_defense_frequency(bet: Real, pot: Real) -> Real:
    """Share of the defending range that must continue so a pure bluff cannot profit."""
    return 1 - bluff_break_even(bet, pot)


def bluff_to_value_ratio(bet: Real, pot: Real) -> Real:
    """Bluffs per value bet that make a bluff-catcher indifferent on the last street."""
    return bluff_break_even(bet, pot)


def bluff_share_of_betting_range(bet: Real, pot: Real) -> Real:
    """Share of the betting range that is bluffs at the bluff-catcher's indifference point.

    The caller risks ``bet`` to win ``pot + bet``, so they need this much bluff mass
    in the betting range: ``bet / (pot + 2 * bet)``.
    """
    _nonneg("bet", bet)
    _nonneg("pot", pot)
    if bet + pot == 0:
        raise ValueError("bet and pot cannot both be zero")
    return div(bet, pot + 2 * bet)


def bluff_ev(fold_prob: Real, bet: Real, pot: Real) -> Real:
    """Chip EV of a zero-equity bet that wins ``pot`` when everyone folds."""
    _probability("fold_prob", fold_prob)
    _nonneg("bet", bet)
    _nonneg("pot", pot)
    return fold_prob * pot - (1 - fold_prob) * bet


def semi_bluff_ev(fold_prob: Real, bet: Real, pot: Real, called_equity: Real) -> Real:
    """Chip EV of a bet that keeps ``called_equity`` when called and the hand is checked down."""
    _probability("fold_prob", fold_prob)
    _probability("called_equity", called_equity)
    _nonneg("bet", bet)
    _nonneg("pot", pot)
    called = called_equity * (pot + 2 * bet) - bet
    return fold_prob * pot + (1 - fold_prob) * called


def semi_bluff_break_even(bet: Real, pot: Real, called_equity: Real) -> Real:
    """Smallest fold probability at which :func:`semi_bluff_ev` is non-negative."""
    _probability("called_equity", called_equity)
    _nonneg("bet", bet)
    _nonneg("pot", pot)
    called = called_equity * (pot + 2 * bet) - bet
    if called >= 0:
        return 0 * called
    return div(-called, pot - called)


def pot_fraction_bets(initial_pot: Real, fractions: Sequence[Real]) -> list[Real]:
    """Chip sizes for a line of pot-fraction bets, each sized on the pot after prior calls."""
    _nonneg("initial_pot", initial_pot)
    pot = initial_pot
    bets: list[Real] = []
    for fraction in fractions:
        _nonneg("fraction", fraction)
        bet = fraction * pot
        bets.append(bet)
        pot = pot + 2 * bet
    return bets


def multi_street_bluff_ev(initial_pot: Real, bets: Sequence[Real], fold_probs: Sequence[Real]) -> Real:
    """Chip EV, from the street where it starts, of a zero-equity line of bets.

    ``fold_probs[i]`` is the probability the opponent folds to ``bets[i]`` given they
    called every earlier bet. A fold on street ``i`` wins the initial pot plus the
    opponent's earlier calls; calling every bet wins nothing and costs every bet.
    The model is heads-up, has no raises and treats a called line as dead.
    """
    if len(bets) != len(fold_probs):
        raise ValueError("bets and fold_probs must have the same length")
    if not bets:
        raise ValueError("a bluff line needs at least one bet")
    _nonneg("initial_pot", initial_pot)
    reach = 1 + 0 * initial_pot
    called_so_far = 0 * initial_pot
    ev = 0 * initial_pot
    for bet, fold in zip(bets, fold_probs, strict=True):
        _nonneg("bet", bet)
        _probability("fold_prob", fold)
        ev = ev + reach * fold * (initial_pot + called_so_far)
        reach = reach * (1 - fold)
        called_so_far = called_so_far + bet
    return ev - reach * called_so_far


def multi_street_bluff_last_fold_break_even(
    initial_pot: Real, bets: Sequence[Real], earlier_fold_probs: Sequence[Real]
) -> Real:
    """Fold probability needed on the last bet for :func:`multi_street_bluff_ev` to be zero.

    The result can exceed 1 when no last-street fold rate repays the earlier bets.
    """
    if len(earlier_fold_probs) != len(bets) - 1:
        raise ValueError("need one earlier fold probability per bet except the last")
    at_zero = multi_street_bluff_ev(initial_pot, bets, [*earlier_fold_probs, 0])
    at_one = multi_street_bluff_ev(initial_pot, bets, [*earlier_fold_probs, 1])
    if at_one == at_zero:
        raise ValueError("the last bet is never reached")
    return div(-at_zero, at_one - at_zero)


def geometric_bet_fraction(stack: Real, pot: Real, streets: int) -> float:
    """Pot fraction that, bet and called on each of ``streets`` streets, commits exactly ``stack``.

    Each called bet of fraction ``f`` multiplies the pot by ``1 + 2f``, so equal growth
    ``g = ((pot + 2 * stack) / pot) ** (1 / streets)`` gives ``f = (g - 1) / 2``.
    This is the static, clairvoyant sizing prior; dynamic boards need learned sizes.
    """
    _nonneg("stack", stack)
    if pot <= 0:
        raise ValueError("pot must be positive")
    if streets < 1:
        raise ValueError("streets must be at least 1")
    growth = float(div(pot + 2 * stack, pot)) ** (1 / streets)
    return (growth - 1) / 2
