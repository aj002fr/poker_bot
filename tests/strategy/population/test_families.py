"""Every family's measured statistics fall inside its declared ranges and show its signature."""

from __future__ import annotations

import pytest

from strategy.population.families import FAMILIES, FAMILY_LIST
from strategy.population.measure import Tally, census, family_stats
from strategy.population.registry import Registry
from tests.strategy.population.conftest import JOBS, by_family

MEMBER_STATS = ("vpip", "pfr")
MEMBER_MIN_HANDS = 300

# (family, statistic, direction, reference family, margin): the family's pooled value is
# beyond the reference's by at least ``margin``. Families 7-17 are compared with
# wide_pressure, the family form of their "aggressive" anchor.
CONTRASTS = [
    ("selective_aggressor", "vpip", "<", "wide_pressure", 0.05),
    ("chaotic_aggressor", "three_bet", ">", "wide_pressure", 0.06),
    ("loose_passive", "cold_call", ">", "wide_pressure", 0.25),
    ("tight_surrender", "fold_to_reraise", ">", "wide_pressure", 0.05),
    ("smooth_switcher", "junk_entry", ">", "selective_aggressor", 0.08),
    ("limp_iso", "limp", ">", "wide_pressure", 0.15),
    ("squeezer", "squeeze", ">", "wide_pressure", 0.05),
    ("trapper", "strong_check", ">", "wide_pressure", 0.2),
    ("draw_player", "draw_continue", ">", "wide_pressure", 0.15),
    ("multiway_specialist", "mw_bluff", "<", "wide_pressure", 0.15),
    ("icm_cap", "short_allin", ">", "wide_pressure", 0.2),
    ("odd_sizer", "min_raise_share", ">", "chaotic_aggressor", 0.03),
]


@pytest.fixture(scope="module")
def pooled(nlhe_census, members) -> dict[str, dict[str, tuple[float | None, int]]]:
    return {fam: family_stats(ts) for fam, ts in by_family(nlhe_census, members).items()}


def test_census_covers_every_family_without_fallbacks(nlhe_census, members):
    fams = by_family(nlhe_census, members)
    assert sorted(fams) == sorted(FAMILIES)
    for key, tally in nlhe_census.items():
        assert not tally.fallbacks, (members[key].family, dict(tally.fallbacks))
        assert tally.decisions > 300


@pytest.mark.parametrize("family", [f.id for f in FAMILY_LIST])
def test_declared_ranges(family, pooled, nlhe_census, members):
    stats = pooled[family]
    for stat, (lo, hi) in FAMILIES[family].expected.items():
        value, n = stats.get(stat, (None, 0))
        assert value is not None and n >= 15, f"{family}.{stat}: only {n} samples"
        assert lo <= value <= hi, f"{family}.{stat} = {value:.3f} (n={n}) outside declared [{lo}, {hi}]"
    for key, tally in nlhe_census.items():
        if members[key].family != family or tally.counts["hands"] < MEMBER_MIN_HANDS:
            continue
        mine = tally.stats()
        for stat in MEMBER_STATS:
            lo, hi = FAMILIES[family].expected[stat]
            assert lo <= mine[stat][0] <= hi, f"member {key} of {family}: {stat} = {mine[stat][0]:.3f}"


@pytest.mark.parametrize("family,stat,direction,ref,margin", CONTRASTS, ids=[c[0] for c in CONTRASTS])
def test_signature_against_reference(family, stat, direction, ref, margin, pooled):
    mine, theirs = pooled[family][stat][0], pooled[ref][stat][0]
    assert mine is not None and theirs is not None
    if direction == ">":
        assert mine >= theirs + margin, f"{family}.{stat} {mine:.3f} vs {ref} {theirs:.3f}"
    else:
        assert mine <= theirs - margin, f"{family}.{stat} {mine:.3f} vs {ref} {theirs:.3f}"


def test_texture_cbettor_splits_by_texture(pooled):
    gap = pooled["texture_cbettor"]["cbet_dry_hu"][0] - pooled["texture_cbettor"]["cbet_wet"][0]
    anchor_gap = pooled["wide_pressure"]["cbet_dry_hu"][0] - pooled["wide_pressure"]["cbet_wet"][0]
    assert gap >= 0.2 and gap >= anchor_gap + 0.15


def test_sizing_tell_modes(registry: Registry):
    """Negative tells shrink bets with strength, positive ones grow them, absent ones stay flat."""
    by_mode: dict[int, list] = {}
    for m in sorted(registry.partition("train"), key=lambda m: m.index):
        if m.family == "sizing_tells":
            by_mode.setdefault(int(m.params["tell"]), []).append(m)
    assert sorted(by_mode) == [0, 1, 2]
    picked = [m for mode in (0, 1, 2) for m in by_mode[mode][:2]]
    fillers = [m for m in registry.partition("train") if m.family == "wide_pressure"][:2]
    entries = [(m.member_id, m.spec) for m in picked + fillers]
    tallies = census(entries, tables=16, seed="test-sizing", jobs=JOBS)
    mode_of = {m.member_id: int(m.params["tell"]) for m in picked}
    pooled: dict[int, Tally] = {}
    for key, tally in tallies.items():
        if key in mode_of:
            pooled.setdefault(mode_of[key], Tally()).merge(tally)
    corr = {mode: t.stats()["size_corr"] for mode, t in pooled.items()}
    cv = {mode: t.stats()["size_cv"] for mode, t in pooled.items()}
    assert all(n >= 30 for _, n in corr.values()), corr
    assert corr[0][0] < -0.5, corr
    assert corr[2][0] > 0.5, corr
    assert cv[1][0] < 0.05 < min(cv[0][0], cv[2][0]), cv


def test_variant_noise_against_its_denoised_twin(registry: Registry):
    """The same members with the card noise switched off enter with weak hands less often."""
    noisy = [m for m in registry.partition("train") if m.family == "variant_noisy"][:3]
    entries = []
    for m in noisy:
        entries.append((m.member_id, m.spec))
        twin = {**m.spec, "member_id": m.member_id + "-clean",
                "params": {**m.params, "card_noise": 0.0, "bucket_flip": 0.0}}
        entries.append((twin["member_id"], twin))
    fillers = [m for m in registry.partition("train") if m.family == "wide_pressure"][:2]
    entries += [(m.member_id, m.spec) for m in fillers]
    tallies = census(entries, tables=16, seed="test-noise", jobs=JOBS)
    junk = {k: family_stats([tallies[k] for k in keys])["junk_entry"]
            for k, keys in (("noisy", [m.member_id for m in noisy]),
                            ("clean", [m.member_id + "-clean" for m in noisy]))}
    assert junk["noisy"][1] >= 200 and junk["clean"][1] >= 200, junk
    assert junk["noisy"][0] >= junk["clean"][0] + 0.02, junk


def test_other_variants_play_legally(variant_census):
    ruleset, chosen, tallies = variant_census
    fams = by_family(tallies, chosen)
    assert sorted(fams) == sorted(FAMILIES)
    for key, tally in tallies.items():
        assert not tally.fallbacks, (ruleset, chosen[key].family, dict(tally.fallbacks))
    for fam, ts in fams.items():
        vpip = family_stats(ts)["vpip"][0]
        assert 0.02 < vpip < 0.8, (ruleset, fam, vpip)
