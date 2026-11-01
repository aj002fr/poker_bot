import math
from fractions import Fraction

import pytest

from strategy.eval.null import synthetic_calibration
from strategy.eval.stats import (
    binomial_sf,
    bootstrap_mean,
    clopper_pearson,
    cluster_bootstrap,
    holm,
    mean,
    minimum_detectable_effect,
    newcombe_difference,
    normal_ppf,
    paired_t,
    quantile,
    sample_size,
    t_cdf,
    t_ppf,
    welch_t,
    wilson,
)


@pytest.mark.parametrize("df,expected", [(1, 12.7062047362), (10, 2.2281388520), (30, 2.0422724563),
                                         (200, 1.9718962236)])
def test_t_quantiles(df, expected):
    assert t_ppf(0.975, df) == pytest.approx(expected, abs=1e-7)
    assert t_cdf(expected, df) == pytest.approx(0.975, abs=1e-9)


def test_normal_quantile():
    assert normal_ppf(0.975) == pytest.approx(1.959963985, abs=1e-8)


def test_paired_t():
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    t = paired_t(diffs)
    se = math.sqrt(2.5 / 5)
    assert t.mean == 3.0 and t.se == pytest.approx(se)
    assert (t.lo, t.hi) == pytest.approx((3 - 2.7764451052 * se, 3 + 2.7764451052 * se), abs=1e-7)
    assert t.t_stat == pytest.approx(4.2426406871) and t.p_value == pytest.approx(0.0132356, abs=1e-6)
    one = paired_t([2.0])
    assert one.lo is None and one.hi is None
    flat = paired_t([0.0, 0.0, 0.0])
    assert flat.lo == flat.hi == 0.0


def test_welch_matches_reference():
    # Variances 10/5 and (5/3)/4; Welch-Satterthwaite df = (29/12)^2 / (2^2/4 + (5/12)^2/3).
    w = welch_t([2.0, 4.0, 6.0, 8.0, 10.0], [1.0, 2.0, 3.0, 4.0])
    assert w.mean_difference == pytest.approx(-3.5)
    assert w.df == pytest.approx((29 / 12) ** 2 / (1 + (5 / 12) ** 2 / 3))
    assert w.lo < -3.5 < w.hi < 1.0


def test_quantile_interpolates():
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert quantile([5.0], 0.1) == 5.0


def test_cluster_bootstrap_is_deterministic_and_respects_strata():
    values = [float(i % 7) for i in range(40)]
    a = bootstrap_mean(values, seed="s", resamples=400)
    b = bootstrap_mean(values, seed="s", resamples=400)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert a.lo < mean(values) < a.hi
    # Two strata with constant values: a stratified mean of the pooled sample never moves.
    flat = bootstrap_mean([0.0] * 10 + [1.0] * 10, seed="s", strata=["x"] * 10 + ["y"] * 10, resamples=200)
    assert flat.lo == flat.hi == 0.5
    clusters = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]
    res = cluster_bootstrap(clusters, lambda cs: mean([x for c in cs for x in c]), seed="c", resamples=200)
    assert res.clusters == 3 and res.lo <= res.estimate <= res.hi


def test_proportions():
    lo, hi = wilson(5, 20)
    assert (lo, hi) == pytest.approx((0.1118617, 0.4687009), abs=1e-6)
    assert wilson(0, 0) is None
    d = newcombe_difference(20, 100, 30, 100)
    assert d[0] < 0.1 < d[1]
    assert newcombe_difference(30, 100, 20, 100) == pytest.approx((-d[1], -d[0]))
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(3, 10) == pytest.approx((0.0667395, 0.6524529), abs=1e-6)
    assert binomial_sf(0, 10, 0.3) == 1.0
    assert binomial_sf(3, 10, 0.3) == pytest.approx(0.6172172, abs=1e-6)
    p = Fraction(1, 40)
    exact = float(sum(math.comb(2000, j) * p ** j * (1 - p) ** (2000 - j) for j in range(60, 2001)))
    assert binomial_sf(60, 2000, 0.025) == pytest.approx(exact, rel=1e-9)
    assert binomial_sf(2001, 2000, 0.5) == 0.0


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_sample_size():
    s = sample_size(1.0, 1.0, t_correction=False)
    assert s.normal_blocks == pytest.approx(7.8489, abs=1e-3)
    assert s.blocks == 8
    t = sample_size(10.0, 2.0)
    assert t.blocks >= math.ceil(7.8489 * 25)
    assert minimum_detectable_effect(10.0, t.blocks) <= 2.0 + 1e-9


def test_synthetic_null_is_calibrated():
    res = synthetic_calibration(replicates=1000, blocks=10, players=6, payouts=(50, 30, 20, 0, 0, 0),
                                bootstrap=False, seed="pytest-null")
    t = res["methods"]["paired_t"]
    assert t["calibrated"], t
    lo, hi = t["false_gain_rate_ci"]
    assert lo <= 0.025 <= hi
    assert t["false_promotions"] <= t["false_gains"]
    assert res["methods"]["unpaired_welch_tables"]["mean_interval_width"] > t["mean_interval_width"]
