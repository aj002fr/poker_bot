"""Fast-suite metamorphic relations (spec 12.2); the release gate runs them at scale."""

from __future__ import annotations

import pytest

from tests.metamorphic import relations as R

RULESETS = ["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4"]


@pytest.fixture(scope="module", params=[(r, n) for r in RULESETS for n in (2, 3, 5, 8)],
                ids=lambda p: f"{p[0]}-{p[1]}")
def base(request) -> R.Base:
    ruleset, n = request.param
    return R.play_base(R.base_config(ruleset, n), f"meta-fast|{ruleset}|{n}")


@pytest.mark.parametrize("perm", ["dchs", "hscd", "sdhc"])
def test_suit_renaming(base, perm):
    R.suit_renaming(base, perm)


@pytest.mark.parametrize("r", [1, 4, 7])
def test_seat_and_button_rotation(base, r):
    R.seat_rotation(base, r)


@pytest.mark.parametrize("k", [2, 5, 1000])
def test_chip_scaling(base, k):
    R.chip_scaling(base, k)


def test_table_label_does_not_change_play(base):
    R.table_relabel(base, "t_relabelled")


def test_serial_and_parallel_scheduling_agree():
    assert R.serial_vs_parallel("practice_nlhe", 3, 1, "meta-fast-svp", jobs=3)["tables"] == 3


def test_relations_are_not_vacuous():
    """A wrong back-mapping must be detected: a rotation compared without undoing it fails."""
    base = R.play_base(R.base_config("practice_nlhe", 4), "meta-fast|vacuous")
    with pytest.raises(R.RelationMismatch):
        R._compare(base, base.table, R.Frame(seat=lambda s: (s + 1) % 4))
    with pytest.raises(R.RelationMismatch):
        R._compare(base, base.table, R.Frame(card=lambda c: c[0] + {"c": "d", "d": "c"}.get(c[1], c[1])))


def test_rotation_maps_odd_chip_corrections():
    """Release-gate case metamorphic/plo4/5/160: the correction's seat-keyed payloads follow the rotation."""
    base = R.play_base(R.base_config("uncertainty_plo4", 5), "meta|uncertainty_plo4|5|160")
    assert any(e.type == "OddChipCorrection" for rec in base.table.hands for e in rec.events)
    for r in (2, 4):
        R.seat_rotation(base, r)


def test_odd_chip_hands_are_excluded_not_compared():
    for i in range(40):
        base = R.play_base(R.base_config("practice_nlhe", 8), f"meta-fast|odd|{i}")
        cut = R.odd_chip_hand(base)
        if cut is not None:
            _, excluded = R.chip_scaling(base, 3)
            assert excluded == cut
            return
    pytest.fail("no odd-chip hand in 40 eight-handed tables; the exclusion path was not exercised")
