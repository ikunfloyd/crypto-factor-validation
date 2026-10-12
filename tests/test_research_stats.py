"""
Tests for research/stats.py, the statistical core of the pre-registered study.

All data here is synthetic. Beyond checking each formula against a hand
computation, the simulations check the two properties the decision rests on:
under the null the Hansen-Hodrick t is a standard normal (size), and a
planted effect of the registered kind is found (power). Seeds are fixed, so
the binomial bands below are generous margins, not flakiness allowances.
"""
from __future__ import annotations

import importlib.util
import inspect
import math

import numpy as np
import pytest

from research import stats as S
from research.spec import REPO_ROOT, get_spec

H1 = 3_600_000
M15 = 900_000
DAY = 86_400_000


# --------------------------------------------------------------------------
# synthetic helpers
# --------------------------------------------------------------------------

def ma_signal(rng, n, length=20):
    """Persistent zero-mean unit-variance signal: a moving sum of white noise."""
    e = rng.standard_normal(n + length)
    c = np.concatenate(([0.0], np.cumsum(e)))
    return (c[length:] - c[:-length])[:n] / math.sqrt(length)


def fixed_buckets(g, thr=0.8416):
    """Quintile-like buckets with known thresholds, for speed where edges are not under test."""
    n = len(g)
    return S.assign_buckets(g, np.full(n, -thr), np.full(n, thr))


def null_cell(rng, n_bars, h, sigma=0.01, length=20):
    """iid returns, a persistent signal independent of them, c = 0, d = +1."""
    ts = np.arange(n_bars, dtype=np.int64) * H1
    close = 100 * np.exp(np.cumsum(sigma * rng.standard_normal(n_bars)))
    fwd = S.forward_returns(close, ts, H1, h)
    s = fixed_buckets(ma_signal(rng, n_bars, length))
    return ts, s, fwd


# --------------------------------------------------------------------------
# spec agreement
# --------------------------------------------------------------------------

class TestSpecAgreement:
    def test_defaults_match_appendix_a(self):
        spec = get_spec()
        sig = inspect.signature(S.expanding_edges).parameters
        assert [sig["lo_pct"].default, sig["hi_pct"].default] == spec["buckets"]["edges_pct"]
        assert sig["min_prior"].default == spec["buckets"]["min_prior_values"]
        assert inspect.signature(S.expanding_percentiles).parameters["min_prior"].default == \
            spec["buckets"]["min_prior_values"]
        assert inspect.signature(S.quintile_means).parameters["min_prior"].default == \
            spec["buckets"]["min_prior_values"]
        assert inspect.signature(S.quintile_means).parameters["quantiles"].default == \
            spec["robustness"]["monotonicity"]["buckets"]
        assert inspect.signature(S.direction).parameters["min_active"].default == \
            spec["direction_rule"]["min_active_IS"]
        c = inspect.signature(S.turnover_aware).parameters["cost_bps"].default
        assert c == spec["cost_round_trip_bps"]
        assert c / 2 == spec["robustness"]["turnover_aware"]["fee_per_unit_change_bps"]

    def test_thresholds_are_the_rounded_normal_quantiles(self):
        spec = get_spec()
        m = spec["families"]["F_primary"]["cells"]
        alpha = spec["test"]["alpha"]
        assert round(S.norm_ppf(1 - alpha / m), 3) == spec["first_holm_threshold_t"]
        assert round(S.norm_ppf(1 - alpha / m), 3) == spec["power"]["z_alpha"]
        assert round(S.norm_ppf(spec["power"]["power"]), 3) == spec["power"]["z_power"]
        assert round(S.norm_ppf(0.95), 3) == spec["intervals"]["ub95"]["z"]
        assert round(S.norm_ppf(0.975), 3) == spec["intervals"]["doge_fwd_ci"]["z"]

    def test_norm_helpers(self):
        assert S.norm_cdf(0.0) == 0.5
        assert math.isnan(S.norm_cdf(float("nan")))
        assert S.norm_ppf(0.0) == -math.inf and S.norm_ppf(1.0) == math.inf
        assert math.isnan(S.norm_ppf(1.5))
        x = np.array([-1.0, 0.0, 2.0])
        np.testing.assert_allclose(S.norm_ppf(S.norm_cdf(x)), x, atol=1e-12)

    # Upper-tail normal probabilities to full precision (scipy.stats.norm.sf),
    # written out so the test needs no scipy.
    TAIL = {0.5: 0.3085375387259869, 3.0: 0.001349898031630093, 3.144: 0.0008332771986115927,
            7.0: 1.279812543885835e-12, 8.0: 6.22096057427174e-16, 8.3: 5.2055697448902465e-17,
            10.0: 7.619853024160474e-24, 20.0: 2.753624118606155e-89}

    def test_p_values_keep_their_precision_in_the_tail(self):
        # 1 - Φ(t) cancels: about 7% off at t = 8 and exactly 0 from t ≈ 8.3,
        # which would print a strongly significant cell's p, p_holm, q_bh,
        # q_by and ic_p as 0.0. The p-values are therefore erfc-based.
        for t, sf in self.TAIL.items():
            assert S._p_upper(t) == pytest.approx(sf, rel=1e-13), t
            assert S._p_two_sided(t) == pytest.approx(2 * sf, rel=1e-13), t
            assert S._p_two_sided(-t) == S._p_two_sided(t)
        assert S._p_upper(-3.0) == pytest.approx(0.9986501019683699, rel=1e-15)
        assert 1.0 - S.norm_cdf(8.3) == 0.0                    # what the old formula printed
        assert math.isnan(S._p_upper(math.nan)) and math.isnan(S._p_two_sided(math.nan))
        assert S._p_upper(math.inf) == 0.0 and S._p_upper(-math.inf) == 1.0
        # And through the statistics that use them: a cell with t ≈ 10 keeps a
        # positive one-sided p, the full-window IC a positive two-sided one.
        ts = np.arange(400, dtype=np.int64) * H1
        n = np.where(np.arange(400) % 2 == 0, 0.003, -0.001)   # mu 0.001, SE about 0.002/20: t about 10
        st = S.cell_statistic(n, ts, 0, 400 * H1, H1, 1)
        assert 9 < st["t"] < 11 and st["p"] == pytest.approx(0.5 * math.erfc(st["t"] / math.sqrt(2)), rel=1e-12)
        assert 0.0 < st["p"] < 1e-18
        f = np.arange(400, dtype=np.float64)
        ic = S.full_window_ic(f, f + np.sin(f), ts, 0, 400 * H1, H1, 1)
        assert ic["t"] > 9 and 0.0 < ic["p"] < 1e-18
        assert ic["p"] == pytest.approx(math.erfc(ic["t"] / math.sqrt(2)), rel=1e-12)

    def test_n_evaluable_matches_section_9(self):
        sets = get_spec()["sets"]
        for h in (1, 5, 15):
            assert S.n_evaluable(sets["IS"]["start_ms"], sets["IS"]["end_ms"], H1, h) == 6596 - h - 1
            assert S.n_evaluable(sets["doge_2023"]["start_ms"], sets["doge_2023"]["end_ms"], H1, h) == \
                8784 - h - 1


# --------------------------------------------------------------------------
# forward returns and the evaluable set (§4.1, §6.3)
# --------------------------------------------------------------------------

class TestForwardReturns:
    def test_known_series(self):
        close = np.arange(1.0, 11.0)
        ts = np.arange(10, dtype=np.int64) * M15
        for h in (1, 3):
            fwd = S.forward_returns(close, ts, M15, h)
            for t in range(10):
                if t + 1 + h < 10:
                    assert fwd[t] == pytest.approx(close[t + 1 + h] / close[t + 1] - 1)
                else:
                    assert np.isnan(fwd[t])

    def test_nan_across_a_gap(self):
        ts = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int64) * M15  # bucket 4 dropped
        close = np.linspace(1.0, 2.0, len(ts))
        fwd = S.forward_returns(close, ts, M15, 1)
        # fwd_t needs rows t, t+1, t+2 consecutive.
        assert np.isfinite(fwd[0]) and np.isfinite(fwd[1])
        assert np.isnan(fwd[2]) and np.isnan(fwd[3])      # spans 3->5
        assert np.isfinite(fwd[4]) and np.isfinite(fwd[5])
        assert np.isnan(fwd[6]) and np.isnan(fwd[7])      # past the end

    def test_bad_prices_and_bad_timestamps(self):
        ts = np.arange(6, dtype=np.int64) * M15
        close = np.array([1.0, 0.0, 2.0, 3.0, np.nan, 4.0])
        fwd = S.forward_returns(close, ts, M15, 1)
        assert np.isnan(fwd[0])     # entry close is 0
        assert np.isnan(fwd[2])     # exit close is NaN
        with pytest.raises(ValueError):
            S.forward_returns(close, ts[::-1], M15, 1)
        with pytest.raises(ValueError):
            S.forward_returns(close, ts + np.array([0, 0, 0, 7, 0, 0]), M15, 1)

    def test_evaluable_boundaries(self):
        start, end, h = 10 * H1, 30 * H1, 3
        ts = np.arange(0, 40, dtype=np.int64) * H1
        ev = S.evaluable(ts, start, end, H1, h)
        idx = np.flatnonzero(ev)
        assert idx[0] == 10
        # start(t+1+h) + B <= end  <=>  ts_t + (h+2)B <= end
        assert ts[idx[-1]] == end - (h + 2) * H1
        assert ev.sum() == S.n_evaluable(start, end, H1, h)


# --------------------------------------------------------------------------
# buckets and direction (§6.4, §6.5)
# --------------------------------------------------------------------------

class TestExpandingEdges:
    def test_equals_numpy_percentile_on_prefixes(self):
        rng = np.random.default_rng(0)
        n = 6000
        f = rng.standard_normal(n)
        f[rng.random(n) < 0.05] = np.nan
        f[rng.random(n) < 0.01] = np.inf
        f[::5] = np.round(f[::5], 1)          # ties
        f[3::11] = 0.0                        # a point mass
        lo, hi = S.expanding_edges(f)
        checks = list(range(490, 560)) + list(rng.integers(560, n, 300)) + [n - 1]
        for t in checks:
            pre = f[:t]
            pre = pre[np.isfinite(pre)]
            if len(pre) < 500:
                assert np.isnan(lo[t]) and np.isnan(hi[t])
                continue
            assert lo[t] == np.percentile(pre, 20, method="linear")
            assert hi[t] == np.percentile(pre, 80, method="linear")
        e = S.expanding_percentiles(f, [20, 40, 60, 80])
        for t in rng.integers(600, n, 50):
            pre = f[:t]
            pre = pre[np.isfinite(pre)]
            assert np.array_equal(e[t], np.percentile(pre, [20, 40, 60, 80], method="linear"))

    def test_no_look_ahead(self):
        rng = np.random.default_rng(1)
        f = rng.standard_normal(2000)
        lo, hi = S.expanding_edges(f)
        for t in (500, 501, 1234):
            g = f.copy()
            g[t:] = rng.standard_normal(len(g) - t) * 100 + 50
            lo2, hi2 = S.expanding_edges(g)
            assert np.array_equal(lo[:t + 1], lo2[:t + 1], equal_nan=True)
            assert np.array_equal(hi[:t + 1], hi2[:t + 1], equal_nan=True)

    def test_min_prior_counts_finite_values_only(self):
        f = np.full(1200, np.nan)
        f[::2] = np.arange(600, dtype=float)
        lo, _ = S.expanding_edges(f)
        first = np.flatnonzero(np.isfinite(lo))[0]
        assert np.sum(np.isfinite(f[:first])) == 500

    def test_50k_is_fast(self):
        import time
        f = np.random.default_rng(2).standard_normal(50_000)
        t0 = time.perf_counter()
        S.expanding_edges(f)
        assert time.perf_counter() - t0 < 3.0


class TestBuckets:
    def test_point_mass_on_an_edge_is_never_traded(self):
        rng = np.random.default_rng(3)
        n = 3000
        f = np.where(rng.random(n) < 0.6, 0.0, rng.standard_normal(n))
        lo, hi = S.expanding_edges(f)
        s = S.assign_buckets(f, lo, hi)
        on_edge = np.isfinite(lo) & ((lo == 0) | (hi == 0))
        assert on_edge.sum() > 100           # the mass does sit on an edge
        assert np.all(s[f == 0] == 0)
        assert np.all(s[np.isnan(lo)] == 0)

    def test_strict_edges_and_non_finite(self):
        f = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan, 5.0])
        lo = np.array([-1.0] * 6 + [np.nan])
        hi = np.array([1.0] * 7)
        s = S.assign_buckets(f, lo, hi)
        assert s.dtype == np.int8
        assert s.tolist() == [-1, 0, 0, 0, 1, 0, 0]

    def test_bucket_shares(self):
        ts = np.arange(10, dtype=np.int64) * H1
        s = np.array([1, -1, 0, 1, 0, 0, 0, 0, 0, 0])
        fwd = np.r_[np.full(7, 0.01), np.nan, np.nan, np.nan]
        out = S.bucket_shares(s, fwd, ts, 0, 10 * H1, H1, 1)
        # evaluable bars 0..7 (ts + 3B <= end); fwd defined on 0..6
        assert out["n_eval"] == 7
        assert out["share_top"] == pytest.approx(2 / 7)
        assert out["share_bottom"] == pytest.approx(1 / 7)


class TestDirection:
    def test_rules(self):
        n = 400
        s = np.ones(n, dtype=np.int8)
        mask = np.ones(n, dtype=bool)
        assert S.direction(s, np.full(n, 0.001), mask)[:3] == (1, False, None)
        assert S.direction(s, np.full(n, -0.001), mask)[:3] == (-1, False, None)
        fz = np.r_[np.full(200, 0.001), np.full(200, -0.001)]
        assert S.direction(s, fz, mask)[0] == 1                       # exact zero -> +1
        d, flagged, reason, mean = S.direction(s, np.full(n, -0.001), mask, min_active=401)
        assert (d, flagged, reason) == (1, True, "few_active") and mean == pytest.approx(-0.001)
        d, flagged, reason, mean = S.direction(s, np.full(n, -0.001), mask, missing_input=True)
        assert (d, flagged, reason) == (1, True, "missing_input") and math.isnan(mean)

    def test_only_active_bars_count(self):
        s = np.array([1, -1, 0, 1] * 100, dtype=np.int8)
        fwd = np.array([0.01, 0.01, 5.0, np.nan] * 100)
        mask = np.ones(400, dtype=bool)
        _, flagged, _, mean = S.direction(s, fwd, mask, min_active=200)
        assert not flagged and mean == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Hansen-Hodrick (§6.7)
# --------------------------------------------------------------------------

def brute_omega(z, lag, bartlett=False):
    n = len(z)
    om = 0.0
    for k in range(-lag, lag + 1):
        w = 1 - abs(k) / (lag + 1) if bartlett else 1.0
        for t in range(n):
            if 0 <= t + k < n:
                om += w * z[t] * z[t + k]
    return om


class TestHHOmega:
    def test_h1_is_sum_of_squares(self):
        z = np.random.default_rng(4).standard_normal(500)
        om, method = S.hh_omega(z, 1)
        assert om == pytest.approx(np.sum(z * z)) and method == "hh"

    def test_matches_brute_force(self):
        z = np.random.default_rng(5).standard_normal(60) + 0.3
        for h in (2, 3, 5):
            om, method = S.hh_omega(z, h)
            assert method == "hh"
            assert om == pytest.approx(brute_omega(z, h - 1))

    def test_bartlett_fallback(self):
        z = np.tile([1.0, -1.0], 50)
        assert brute_omega(z, 1) <= 0
        om, method = S.hh_omega(z, 2)
        assert method == "bartlett_fallback"
        assert om == pytest.approx(brute_omega(z, 4, bartlett=True))
        assert om > 0

    def test_overlap_t_is_standard_normal_and_naive_t_is_not(self):
        rng = np.random.default_rng(42)
        h, n_bars = 5, 3000
        t_hh, t_naive = [], []
        for _ in range(300):
            ts, s, fwd = null_cell(rng, n_bars, h)
            n = S.per_trade_net(s, fwd, 1, 0.0)
            st = S.cell_statistic(n, ts, 0, n_bars * H1, H1, h)
            t_hh.append(st["t"])
            a = n[np.isfinite(n) & S.evaluable(ts, 0, n_bars * H1, H1, h)]
            t_naive.append(a.mean() / (a.std(ddof=1) / math.sqrt(len(a))))
        assert 0.85 < np.std(t_hh) < 1.2
        assert abs(np.mean(t_hh)) < 0.2
        assert np.std(t_naive) > 1.6         # ≈ √h: overlap ignored


# --------------------------------------------------------------------------
# cell statistic: hand values, size and power (§6.6-§6.8, §9)
# --------------------------------------------------------------------------

class TestCellStatistic:
    def test_hand_computed_h1(self):
        ts = np.arange(10, dtype=np.int64) * H1
        n = np.array([0.01, np.nan, -0.02, 0.03, np.nan, 0.0, 0.01, 0.5, 0.5, 0.5])
        st = S.cell_statistic(n, ts, 0, 10 * H1, H1, 1)
        # bars 0..7 are evaluable (ts + 3B <= end); 8 and 9 are not
        a = np.array([0.01, -0.02, 0.03, 0.0, 0.01, 0.5])
        mu = a.mean()
        se = math.sqrt(np.sum((a - mu) ** 2)) / 6
        assert st["n_active"] == 6
        assert st["mu"] == pytest.approx(mu)
        assert st["se"] == pytest.approx(se)
        assert st["t"] == pytest.approx(mu / se)
        assert st["p"] == pytest.approx(1 - S.norm_cdf(mu / se))
        assert st["se_method"] == "hh"

    def test_hand_computed_overlap_with_gap(self):
        # bucket 3 dropped: bars 2 and 4 are two grid steps apart, not one
        ts = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64) * H1
        n = np.array([0.01, 0.02, -0.01, 0.03, 0.0, -0.02, np.nan, np.nan, np.nan, np.nan, np.nan])
        h = 2
        st = S.cell_statistic(n, ts, 0, 12 * H1, H1, h)
        a = n[:6]
        z = np.zeros(12)
        z[[0, 1, 2, 4, 5, 6]] = a - a.mean()
        se = math.sqrt(brute_omega(z, 1)) / 6
        assert st["se"] == pytest.approx(se)

    def test_ignores_warmup_and_window_straddling_bars(self):
        ts = np.arange(20, dtype=np.int64) * H1
        n = np.full(20, 0.01)
        n[[6, 8, 11]] = [0.02, -0.01, 0.0]
        st = S.cell_statistic(n, ts, 5 * H1, 15 * H1, H1, 2)
        # evaluable: 5 <= t and t + 4 <= 15  ->  t = 5..11
        assert st["n_active"] == 7

    def test_no_active_bars_gives_nans(self):
        ts = np.arange(10, dtype=np.int64) * H1
        st = S.cell_statistic(np.full(10, np.nan), ts, 0, 10 * H1, H1, 1)
        assert st["n_active"] == 0 and math.isnan(st["t"]) and math.isnan(st["p"])

    def test_zero_se_leaves_t_undefined_and_p_one(self):
        # §6.7: every trade returned exactly μ̂, so SE = 0, t is undefined, p = 1.
        ts = np.arange(20, dtype=np.int64) * H1
        for h in (1, 3):
            st = S.cell_statistic(np.full(20, 0.002), ts, 0, 20 * H1, H1, h)
            assert st["se"] == 0.0 and math.isnan(st["t"]) and st["p"] == 1.0
            assert st["se_method"] == "bartlett_fallback"   # Ω = 0 <= 0
            pooled = S.pooled_statistic([(np.full(20, 0.002), ts)] * 2, 0, 20 * H1, H1, h)
            assert pooled["se"] == 0.0 and math.isnan(pooled["t"]) and pooled["p"] == 1.0

    def test_misaligned_bars_are_refused(self):
        ts = np.arange(10, dtype=np.int64) * H1 + 60_000
        with pytest.raises(ValueError):
            S.cell_statistic(np.zeros(10), ts, 0, 10 * H1, H1, 1)

    def test_size_under_the_null(self):
        """One-sided 5% rejections stay near 5% with the full bucket pipeline."""
        rng = np.random.default_rng(2024)
        h, warm, n_eval, reps = 5, 500, 2000, 400
        n_all = warm + n_eval
        ts = np.arange(n_all, dtype=np.int64) * H1
        start, end = warm * H1, n_all * H1
        rejections = 0
        for _ in range(reps):
            close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n_all)))
            fwd = S.forward_returns(close, ts, H1, h)
            f = ma_signal(rng, n_all)
            lo, hi = S.expanding_edges(f)
            s = S.assign_buckets(f, lo, hi)
            st = S.cell_statistic(S.per_trade_net(s, fwd, 1, 0.0), ts, start, end, H1, h)
            rejections += st["p"] < 0.05
        # Binomial(400, 0.05): mean 20, sd 4.4. The band is about ±3.5 sd.
        assert 5 <= rejections <= 36


def planted_cell(rng, beta, h, cost_bps, n_eval, warm=600, d=None, sigma=0.005):
    """
    A price path in which the factor predicts the next h returns.

    r_u = σ(β·g_{u-2} + ε_u) for a persistent g, so g at bar t moves the
    returns from the close of t+1 onward, which is exactly what a one-bar-
    delayed trade can earn. The factor observed is g plus noise. β < 0 plants
    a reversal, so the direction rule has to learn d = -1 first.
    """
    n_all = warm + n_eval
    g = ma_signal(rng, n_all)
    r = np.zeros(n_all)
    r[2:] = sigma * (beta * g[:-2] + rng.standard_normal(n_all - 2))
    close = 100 * np.exp(np.cumsum(r))
    ts = np.arange(n_all, dtype=np.int64) * H1
    f = g + 0.5 * rng.standard_normal(n_all)
    start, end = warm * H1, n_all * H1
    fwd = S.forward_returns(close, ts, H1, h)
    lo, hi = S.expanding_edges(f)
    s = S.assign_buckets(f, lo, hi)
    if d is None:
        d, *_ = S.direction(s, fwd, S.evaluable(ts, start, end, H1, h))
    st = S.cell_statistic(S.per_trade_net(s, fwd, d, cost_bps), ts, start, end, H1, h)
    return d, st


class TestPower:
    T1 = get_spec()["first_holm_threshold_t"]

    def test_planted_alpha_is_detected_after_cost(self):
        """IS fixes d, then a fresh test window must clear t > 3.144 net of 10 bp."""
        rng = np.random.default_rng(11)
        h, beta = 5, -0.15
        dirs, ts_ = [], []
        for _ in range(40):
            d, _ = planted_cell(rng, beta, h, 0.0, n_eval=3000)
            _, st = planted_cell(rng, beta, h, get_spec()["cost_round_trip_bps"], n_eval=5000, d=d)
            dirs.append(d)
            ts_.append(st["t"])
        assert all(d == -1 for d in dirs)
        assert np.mean(np.array(ts_) > self.T1) >= 0.9

    def test_pure_noise_is_not_detected(self):
        rng = np.random.default_rng(12)
        h = 5
        ts_ = []
        for _ in range(40):
            d, _ = planted_cell(rng, 0.0, h, 0.0, n_eval=3000)
            _, st = planted_cell(rng, 0.0, h, 0.0, n_eval=5000, d=d)   # no cost: a fair null
            ts_.append(st["t"])
        ts_ = np.array(ts_)
        assert np.sum(ts_ > self.T1) <= 1          # nominal 0.083%
        assert np.mean(ts_ > 1.645) <= 0.15        # nominal 5%

    def test_effect_equal_to_the_mde_is_found_80_percent_of_the_time(self):
        """§9's MDE: (3.144 + 0.842)·SE gives 80% power at the first Holm step."""
        rng = np.random.default_rng(13)
        h, n_bars, reps = 5, 2000, 200
        cells = [null_cell(rng, n_bars, h) for _ in range(reps)]
        null = [S.cell_statistic(S.per_trade_net(s, fwd, 1, 0.0), ts, 0, n_bars * H1, H1, h)
                for ts, s, fwd in cells]
        effect = S.mde(np.mean([st["se"] for st in null]))
        hits = 0
        for (ts, s, fwd), st0 in zip(cells, null):
            planted = fwd + effect * s       # d·s·fwd gains exactly `effect` per trade
            st = S.cell_statistic(S.per_trade_net(s, planted, 1, 0.0), ts, 0, n_bars * H1, H1, h)
            assert st["se"] == pytest.approx(st0["se"])   # a constant shift leaves z unchanged
            hits += st["t"] > self.T1
        assert 0.68 <= hits / reps <= 0.92


# --------------------------------------------------------------------------
# multiple testing, bounds (§7, §9)
# --------------------------------------------------------------------------

class TestMultipleTesting:
    P = [0.01, 0.04, 0.03, 0.005]

    def test_holm_hand_values(self):
        # sorted 0.005, 0.01, 0.03, 0.04 -> 4·0.005, 3·0.01, 2·0.03, 1·0.04, running max
        np.testing.assert_allclose(S.holm(self.P), [0.03, 0.06, 0.06, 0.02])
        np.testing.assert_allclose(S.holm([0.5, 0.6]), [1.0, 1.0])
        np.testing.assert_allclose(S.holm([0.01, 0.01, 0.2]), [0.03, 0.03, 0.2])

    def test_holm_first_step_is_bonferroni(self):
        p = np.r_[0.05 / 60, np.full(59, 0.5)]
        assert S.holm(p)[0] == pytest.approx(0.05)

    def test_bh_and_by_hand_values(self):
        np.testing.assert_allclose(S.bh(self.P), [0.02, 0.04, 0.04, 0.02])
        c4 = 1 + 1 / 2 + 1 / 3 + 1 / 4
        np.testing.assert_allclose(S.by(self.P), np.array([0.02, 0.04, 0.04, 0.02]) * c4)
        np.testing.assert_allclose(S.by([0.5, 0.9]), [1.0, 1.0])

    def test_bh_and_by_step_up_minimum(self):
        # m·p/j = 3·0.01/1, 3·0.02/2, 3·0.025/3 = 0.03, 0.03, 0.025: not
        # monotone, so the step-up minimum over j' >= j makes all three 0.025.
        # (The hand vector above is already monotone after scaling and would
        # pass without the minimum.)
        np.testing.assert_allclose(S.bh([0.01, 0.02, 0.025]), [0.025] * 3)
        np.testing.assert_allclose(S.by([0.01, 0.02, 0.025]), np.array([0.025] * 3) * (1 + 1 / 2 + 1 / 3))
        # Input order is kept.
        np.testing.assert_allclose(S.bh([0.025, 0.01, 0.02]), [0.025] * 3)

    def test_bh_and_by_match_scipy(self):
        stats = pytest.importorskip("scipy.stats")
        if not hasattr(stats, "false_discovery_control"):
            pytest.skip("scipy < 1.11 has no false_discovery_control")
        rng = np.random.default_rng(0)
        for _ in range(200):
            m = int(rng.integers(1, 80))
            p = np.round(rng.uniform(0, 1, m) ** 3, 3)          # small values and ties
            np.testing.assert_allclose(S.bh(p), stats.false_discovery_control(p), rtol=1e-12)
            np.testing.assert_allclose(S.by(p), stats.false_discovery_control(p, method="by"), rtol=1e-12)

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            S.holm([0.1, np.nan])

    def test_bounds_and_power_helpers(self):
        spec = get_spec()
        b = S.bounds(0.001, 0.0005, spec)
        assert b["ub95"] == pytest.approx(0.001 + 1.645 * 0.0005)
        assert b["ub95_family"] == pytest.approx(0.001 + 3.144 * 0.0005)
        assert (b["ci95_lo"], b["ci95_hi"]) == pytest.approx((0.001 - 1.96 * 0.0005, 0.001 + 1.96 * 0.0005))
        assert S.mde(0.0005, spec) == pytest.approx((3.144 + 0.842) * 0.0005)
        np.testing.assert_allclose(S.mde(np.array([1.0, 2.0]), spec), [3.986, 7.972])
        assert S.scale_se(2.0, 6590, 8778) == pytest.approx(2.0 * math.sqrt(6590 / 8778))


# --------------------------------------------------------------------------
# pooled F_majors statistic (§7)
# --------------------------------------------------------------------------

class TestPooled:
    def test_single_and_identical_instruments_reduce_to_the_cell(self):
        rng = np.random.default_rng(6)
        ts, s, fwd = null_cell(rng, 1500, 5)
        n = S.per_trade_net(s, fwd, 1, 0.0)
        one = S.cell_statistic(n, ts, 0, 1500 * H1, H1, 5)
        p1 = S.pooled_statistic([(n, ts)], 0, 1500 * H1, H1, 5)
        p3 = S.pooled_statistic([(n, ts)] * 3, 0, 1500 * H1, H1, 5)
        for p in (p1, p3):
            assert p["mu"] == pytest.approx(one["mu"], rel=1e-12)
            # Dividing Ω_w by |A| again would shrink SE by a factor of ~|A|.
            assert p["se"] == pytest.approx(one["se"], rel=1e-10)
        assert p3["n_active_each"] == [one["n_active"]] * 3

    def test_se_matches_the_sd_of_the_pooled_mean(self):
        rng = np.random.default_rng(3)
        h, n_bars, reps = 5, 2000, 300
        ts = np.arange(n_bars, dtype=np.int64) * H1
        mus, ses, se_indep = [], [], []
        for _ in range(reps):
            common = rng.standard_normal(n_bars)
            g_common = ma_signal(rng, n_bars)
            panel, se_i = [], []
            for _i in range(3):
                ret = 0.01 * (0.8 * common + 0.6 * rng.standard_normal(n_bars))
                close = 100 * np.exp(np.cumsum(ret))
                fwd = S.forward_returns(close, ts, H1, h)
                s = fixed_buckets(0.8 * g_common + 0.6 * ma_signal(rng, n_bars))
                n = S.per_trade_net(s, fwd, 1, 0.0)
                panel.append((n, ts))
                se_i.append(S.cell_statistic(n, ts, 0, n_bars * H1, H1, h)["se"])
            st = S.pooled_statistic(panel, 0, n_bars * H1, H1, h)
            mus.append(st["mu"])
            ses.append(st["se"])
            se_indep.append(math.sqrt(sum(x * x for x in se_i)) / 3)
        sd = np.std(mus)
        assert 0.85 < sd / np.mean(ses) < 1.2
        # Treating the instruments as independent understates the SE.
        assert sd / np.mean(se_indep) > 1.2

    def test_different_gaps_and_missing_instrument(self):
        ts = np.arange(100, dtype=np.int64) * H1
        n = np.full(100, 0.001)
        keep = np.ones(100, dtype=bool)
        keep[[10, 11, 50]] = False
        st = S.pooled_statistic([(n, ts), (n[keep], ts[keep])], 0, 100 * H1, H1, 2)
        assert st["n_active_each"] == [97, 94]
        st = S.pooled_statistic([(n, ts), (np.full(100, np.nan), ts)], 0, 100 * H1, H1, 2)
        assert math.isnan(st["t"]) and st["n_active_each"] == [97, 0]


# --------------------------------------------------------------------------
# R1 circular placebo
# --------------------------------------------------------------------------

class TestPlacebo:
    def test_offsets_formula(self):
        for n_grid, bar in ((8784, H1), (35136, M15)):
            d30 = 30 * DAY // bar
            off = S.placebo_offsets(n_grid, d30, 2000)
            j = np.arange(2000)
            ref = d30 + np.round(j * (n_grid - 2 * d30) / 1999).astype(np.int64)
            assert np.array_equal(off, np.unique(ref))
            assert len(off) == 2000
            assert off[0] == d30 and off[-1] == n_grid - d30

    def test_offsets_dedup(self):
        off = S.placebo_offsets(1500, 100, 2000)
        assert len(off) == 1301            # every integer shift from 100 to 1400, once
        assert np.array_equal(off, np.arange(100, 1401))
        with pytest.raises(ValueError):
            S.placebo_offsets(150, 100, 2000)

    def _cell_with_gaps(self, seed):
        rng = np.random.default_rng(seed)
        n_all, warm, h = 1600, 600, 3
        ts = np.arange(n_all, dtype=np.int64) * H1
        keep = rng.random(n_all) > 0.03
        ts = ts[keep]
        close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(ts))))
        fwd = S.forward_returns(close, ts, H1, h)
        f = ma_signal(rng, len(ts))
        lo, hi = S.expanding_edges(f)
        s = S.assign_buckets(f, lo, hi)
        return ts, s, fwd, warm * H1, n_all * H1, h

    def test_zero_rotation_reproduces_t_and_rotation_is_on_the_grid(self):
        ts, s, fwd, start, end, h = self._cell_with_gaps(7)
        d, c = -1, 10.0
        t_obs = S.cell_statistic(S.per_trade_net(s, fwd, d, c), ts, start, end, H1, h)["t"]
        n_grid = (end - start) // H1
        t0 = S.circular_placebo_t(s, fwd, d, c, ts, start, end, H1, h, [0, n_grid])
        assert t0[0] == t_obs and t0[1] == t_obs

        # Brute force: rotate on the grid by hand and read back per bar.
        pos = (ts - start) // H1
        inwin = (ts >= start) & (ts < end)
        grid = np.zeros(n_grid, dtype=np.int8)
        grid[pos[inwin]] = s[inwin]
        for delta in (37, 400, 999):
            s_rot = np.zeros_like(s)
            s_rot[inwin] = grid[(pos[inwin] - delta) % n_grid]
            want = S.cell_statistic(S.per_trade_net(s_rot, fwd, d, c), ts, start, end, H1, h)["t"]
            got = S.circular_placebo_t(s, fwd, d, c, ts, start, end, H1, h, [delta])[0]
            assert got == pytest.approx(want, rel=1e-12)

    def test_p_values_and_resolution(self):
        t_shift = np.array([0.5, 1.0, np.nan, 2.0, -1.0])
        assert S.placebo_p(3.0, t_shift) == pytest.approx(1 / 6)     # the floor, 1/(1+K')
        assert S.placebo_p(1.0, t_shift) == pytest.approx(3 / 6)
        assert S.placebo_p(-5.0, t_shift) == pytest.approx(5 / 6)     # NaN never counts
        assert math.isnan(S.placebo_p(math.nan, t_shift))
        ts, s, fwd, start, end, h = self._cell_with_gaps(8)
        offs = S.placebo_offsets((end - start) // H1, 100, 50)
        t_obs = S.cell_statistic(S.per_trade_net(s, fwd, 1, 0.0), ts, start, end, H1, h)["t"]
        p = S.placebo_p(t_obs, S.circular_placebo_t(s, fwd, 1, 0.0, ts, start, end, H1, h, offs))
        assert 1 / (1 + len(offs)) <= p <= 1.0

    def test_family_p_uses_max_over_testable_cells(self):
        t_obs = np.array([2.5, 1.0, np.nan])
        T = np.array([[0.0, 3.0, 1.0, 0.5],
                      [2.0, 0.0, 0.0, 1.5],
                      [9.0, 9.0, 9.0, 9.0]])       # untestable: excluded from the max
        p = S.family_placebo_p(t_obs, T)
        # column maxima over cells 0,1: [2.0, 3.0, 1.0, 1.5]
        assert p[0] == pytest.approx((1 + 1) / 5)
        assert p[1] == pytest.approx((1 + 4) / 5)
        assert math.isnan(p[2])

    def test_family_max_includes_testable_cells_whose_own_t_is_undefined(self):
        # "max over testable cells of t_j": a testable cell with SE = 0 has no
        # t_HH of its own, but its rotations still belong to the family.
        t_obs = np.array([2.5, np.nan, 1.0])
        T = np.array([[0.0, 3.0, 1.0, 0.5],
                      [2.6, 0.0, 0.0, 0.0],
                      [9.0, 9.0, 9.0, 9.0]])
        p = S.family_placebo_p(t_obs, T, testable=[True, True, False])
        # column maxima over cells 0,1: [2.6, 3.0, 1.0, 0.5]
        assert p[0] == pytest.approx((1 + 2) / 5)
        assert math.isnan(p[1]) and math.isnan(p[2])


# --------------------------------------------------------------------------
# R9: full-window Spearman IC, and the daily IC it replaced (§12)
# --------------------------------------------------------------------------

def brute_avg_rank(a):
    """1-based average ranks by counting, independent of S._avg_rank."""
    a = np.asarray(a, dtype=float)
    return np.array([np.sum(a < x) + (np.sum(a == x) + 1) / 2 for x in a])


def brute_standardised_ranks(a):
    r = brute_avg_rank(a)
    r = r - r.mean()
    return r / math.sqrt(np.mean(r * r))


def rsi_14(close):
    """The registered rsi_14 kernel itself (§5.1), on a bare price array."""
    from quant_core.factor_engine import FactorEngine
    return FactorEngine._calculate_rsi_numba(np.ascontiguousarray(close, dtype=np.float64), 14)


def momentum(close, w=240):
    x = np.log(close)
    out = np.full(len(x), np.nan)
    out[w:] = x[w:] - x[:-w]
    return out


def random_walk_year(rng, bar_ms):
    """A doge_2023-sized set on a pure iid random walk: 90-day warm-up, 366-day window."""
    per_day = DAY // bar_ms
    warm, ev = 90 * per_day, 366 * per_day
    ts = np.arange(warm + ev, dtype=np.int64) * bar_ms
    close = 0.1 * np.exp(np.cumsum(0.005 * rng.standard_normal(len(ts))))
    return ts, close, warm * bar_ms, (warm + ev) * bar_ms


class TestFullWindowIC:
    def test_hand_computed_on_a_grid_with_gaps_ties_and_nans(self):
        rng = np.random.default_rng(20)
        h = 3
        ts = np.delete(np.arange(70, dtype=np.int64), [7, 8, 31, 32, 33, 50]) * H1
        f = np.round(rng.standard_normal(len(ts)), 1)          # ties -> average ranks
        fwd = np.round(rng.standard_normal(len(ts)) * 0.01, 3)
        f[[4, 12]] = np.nan
        fwd[20] = np.nan
        start, end = 5 * H1, 70 * H1
        out = S.full_window_ic(f, fwd, ts, start, end, H1, h)

        ok = (ts >= start) & (ts + (h + 2) * H1 <= end) & np.isfinite(f) & np.isfinite(fwd)
        n = int(ok.sum())
        u, v = brute_standardised_ranks(f[ok]), brute_standardised_ranks(fwd[ok])
        ic = float(np.mean(u * v))
        assert ic == pytest.approx(np.corrcoef(brute_avg_rank(f[ok]), brute_avg_rank(fwd[ok]))[0, 1])
        z = np.zeros((end - start) // H1)                      # the grid: gaps stay zero
        z[(ts[ok] - start) // H1] = u * v - ic
        om, method = brute_omega(z, h - 1), "hh"
        if om <= 0:
            om, method = brute_omega(z, 2 * h, bartlett=True), "bartlett_fallback"
        se = math.sqrt(om) / n
        assert out["n"] == n
        assert out["ic"] == pytest.approx(ic, rel=1e-12)
        assert out["se"] == pytest.approx(se, rel=1e-12)
        assert out["se_method"] == method
        assert out["t"] == pytest.approx(ic / se, rel=1e-12)
        assert out["p"] == pytest.approx(2 * (1 - S.norm_cdf(abs(ic / se))), rel=1e-12)

        # Warm-up bars and bars whose forward window leaves [start, end) never count.
        g = f.copy()
        g[(ts < start) | (ts + (h + 2) * H1 > end)] = 99.0
        again = S.full_window_ic(g, fwd, ts, start, end, H1, h)
        assert again["ic"] == out["ic"] and again["se"] == out["se"] and again["n"] == n

    def test_known_correlation(self):
        rng = np.random.default_rng(9)
        n = 5000
        ts = np.arange(n, dtype=np.int64) * M15
        x = rng.standard_normal(n)
        y = 0.3 * x + math.sqrt(1 - 0.09) * rng.standard_normal(n)
        out = S.full_window_ic(x, y, ts, 0, n * M15, M15, 1)
        assert out["ic"] == pytest.approx(6 / math.pi * math.asin(0.15), abs=0.03)
        assert out["t"] > 10 and out["p"] < 1e-6
        out = S.full_window_ic(x, rng.standard_normal(n), ts, 0, n * M15, M15, 1)
        assert abs(out["t"]) < 3 and 0 < out["p"] <= 1

    def test_degenerate_cases(self):
        ts = np.arange(10, dtype=np.int64) * H1
        end = 10 * H1
        # Two pairs, perfectly ranked: every product equals the IC, so SE = 0,
        # t is undefined and p is 1 (§6.7's rule, which R9's SE inherits).
        f = np.r_[1.0, 2.0, np.full(8, np.nan)]
        out = S.full_window_ic(f, f, ts, 0, end, H1, 1)
        assert out["n"] == 2 and out["ic"] == pytest.approx(1.0)
        assert out["se"] == 0.0 and math.isnan(out["t"]) and out["p"] == 1.0
        # Every f tied: no rank variance, so the IC itself is undefined.
        out = S.full_window_ic(np.ones(10), np.arange(10.0), ts, 0, end, H1, 1)
        assert out["n"] == 8 and math.isnan(out["ic"]) and math.isnan(out["p"]) and out["se_method"] is None
        # Fewer than two pairs.
        out = S.full_window_ic(np.r_[1.0, np.full(9, np.nan)], np.arange(10.0), ts, 0, end, H1, 1)
        assert out["n"] == 1 and math.isnan(out["ic"]) and math.isnan(out["p"])
        with pytest.raises(ValueError):
            S.full_window_ic(np.ones(9), np.ones(10), ts, 0, end, H1, 1)

    def test_size_on_a_random_walk_for_persistent_factors(self):
        """
        Two-sided 5% rejections of R9 stay near 5% where the daily IC failed.

        Pure iid random walks of doge_2023's 1H length; the registered rsi_14
        kernel and a 240-bar momentum (the slowest window of §5.2) are
        persistent and price-derived, the case §12 is about.
        """
        rng = np.random.default_rng(909)
        reps, hs = 300, (1, 5, 15)
        rej = np.zeros((2, len(hs)), dtype=int)
        tvals = []
        for _ in range(reps):
            ts, close, start, end = random_walk_year(rng, H1)
            facs = (rsi_14(close), momentum(close, 240))
            for j, h in enumerate(hs):
                fwd = S.forward_returns(close, ts, H1, h)
                for i, f in enumerate(facs):
                    out = S.full_window_ic(f, fwd, ts, start, end, H1, h)
                    rej[i, j] += out["p"] < 0.05
                    tvals.append(out["t"])
        # Binomial(300, 0.05): mean 15, sd 3.8; each band is about ±3.5 sd.
        assert np.all((rej >= 2) & (rej <= 28)), rej
        assert 0.03 <= rej.sum() / rej.size / reps <= 0.08, rej
        assert 0.85 < np.std(tvals) < 1.15

    # The t bound at 15m is a margin, not this seed's value: |t| is about 14
    # there, and -13 failed for 5 of 20 other seeds. At 1H (|t| about 19) the
    # margin to -13 is wide.
    @pytest.mark.parametrize("bar_ms, expected, t_bound", [(M15, -0.07, -10), (H1, -0.18, -13)])
    def test_the_superseded_daily_ic_is_biased(self, bar_ms, expected, t_bound):
        """
        The §12 deviation: the registered-then-replaced mean of daily ICs.

        On a pure random walk the persistent rsi_14 gets a mean daily IC near
        -0.07 at 15m and -0.18 at 1H, |t| about 14 and 19 at h = 1, because
        ranking within a day centres both series on the day's own mean. The
        full-window IC on the same data is not significant.
        """
        rng = np.random.default_rng(1010)
        ts, close, start, end = random_walk_year(rng, bar_ms)
        f = rsi_14(close)
        fwd = S.forward_returns(close, ts, bar_ms, 1)
        old = S.daily_ic(f, fwd, ts, mask=S.evaluable(ts, start, end, bar_ms, 1))
        assert old["ic_mean"] == pytest.approx(expected, abs=0.03)
        assert old["ic_t"] < t_bound and old["ic_p"] < 1e-12
        new = S.full_window_ic(f, fwd, ts, start, end, bar_ms, 1)
        assert abs(new["t"]) < 3 and abs(new["ic"]) < 0.03

    def test_avg_rank(self):
        np.testing.assert_allclose(S._avg_rank(np.array([3.0, 1.0, 3.0, 2.0])), [3.5, 1.0, 3.5, 2.0])
        a = np.round(np.random.default_rng(21).standard_normal(200), 1)
        np.testing.assert_array_equal(S._avg_rank(a), brute_avg_rank(a))


# --------------------------------------------------------------------------
# R2 v1's statistic
# --------------------------------------------------------------------------

def test_v1_t_sqrt_h_is_v1s_code_unchanged():
    path = REPO_ROOT / "scripts" / "factor_research.py"
    spec = importlib.util.spec_from_file_location("_v1_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(14)
    f = rng.standard_normal(3000)
    fwd = 0.001 * f + 0.01 * rng.standard_normal(3000)
    mask = np.zeros(3000, dtype=bool)
    mask[500:2900] = True
    want = mod._spread_t_stat(mod.layered_returns(f[mask], fwd[mask], 5), 5)
    assert S.v1_t_sqrt_h(f, fwd, 5, mask=mask) == want
    assert S.v1_t_sqrt_h(f[mask], fwd[mask], 5) == want
    assert math.isnan(S.v1_t_sqrt_h(f[:100], fwd[:100], 5))   # v1 declines below 250 obs


# --------------------------------------------------------------------------
# R3-R8, R10
# --------------------------------------------------------------------------

class TestRobustness:
    def test_by_quarter_partitions_trades_by_signal_bar(self):
        rng = np.random.default_rng(15)
        ts = np.arange(400, dtype=np.int64) * H1
        n = rng.standard_normal(400) * 0.01
        h = 5
        rows = S.by_quarter(n, ts, 0, 400 * H1, H1, h, bounds_ms=[100 * H1, 200 * H1, 300 * H1])
        full = S.cell_statistic(n, ts, 0, 400 * H1, H1, h)
        assert [r["quarter"] for r in rows] == [1, 2, 3, 4]
        assert sum(r["n_active"] for r in rows) == full["n_active"]
        # Bar 99's forward window runs into Q2 but the trade stays in Q1.
        assert rows[0]["n_active"] == 100
        assert rows[0]["mu"] == pytest.approx(n[:100].mean())

    def test_by_quarter_defaults_to_appendix_bounds(self):
        sets = get_spec()["sets"]["doge_2023"]
        start, end = sets["start_ms"], sets["end_ms"]
        ts = np.arange(start, end, H1, dtype=np.int64)
        rows = S.by_quarter(np.full(len(ts), 0.001), ts, start, end, H1, 1)
        assert [r["start_ms"] for r in rows[1:]] == get_spec()["robustness"]["quarter_bounds_ms"]

    def test_volatile_days(self):
        rng = np.random.default_rng(16)
        start, end = 0, 5 * DAY
        ts = np.arange(-60, 5 * 1440, dtype=np.int64) * 60_000   # an hour of pre-window minutes
        r = 0.0001 * rng.standard_normal(len(ts))
        r[60 + 3 * 1440 + 100] = 0.05      # day 3
        r[60 + 1 * 1440 + 7] = 0.03        # day 1
        r[10] = 0.5                        # before the window: ignored
        close = 100 * np.cumprod(1 + r)
        days = S.volatile_days(ts, close, start, end, 2)
        assert days.tolist() == [3 * DAY, 1 * DAY]
        assert len(S.volatile_days(ts, close, start, end, 0)) == 0
        # a gap minute: the return across it is not a 1-minute return
        keep = np.ones(len(ts), dtype=bool)
        keep[60 + 3 * 1440 + 100] = False
        days = S.volatile_days(ts[keep], close[keep], start, end, 1)
        assert days.tolist() == [1 * DAY]

    def test_a_return_across_midnight_belongs_to_the_later_minute(self):
        # R4: close(00:00)/close(23:59) - 1 is realised during the 00:00 minute.
        ts = np.arange(3 * 1440, dtype=np.int64) * 60_000
        r = np.full(len(ts), 1e-5)
        r[2 * 1440] = 0.05                       # minute 00:00 of day 2 vs 23:59 of day 1
        close = 100 * np.cumprod(1 + r)
        assert S.volatile_days(ts, close, 0, 3 * DAY, 1).tolist() == [2 * DAY]

    def test_leave_out_volatile_days_boundaries(self):
        h = 2
        ts = np.arange(72, dtype=np.int64) * H1
        n = np.full(72, 0.001)
        out = S.leave_out_volatile_days(n, ts, 0, 72 * H1, H1, h, [DAY])
        # holding span [ts+2h, ts+4h]; overlaps [24h, 48h) for ts in (20h, 46h)
        dropped = [t for t in range(72) if t + 4 <= 72 and (t + 2 < 48 and t + 4 > 24)]
        assert out["n_dropped"] == len(dropped) == 25
        assert out["n_active"] == 69 - 25
        assert 20 not in dropped and 46 not in dropped    # exits exactly at 24h / enters at 48h
        out0 = S.leave_out_volatile_days(n, ts, 0, 72 * H1, H1, h, [])
        assert out0["n_dropped"] == 0 and out0["k"] == 0

    def test_turnover_aware_hand_example(self):
        ts = np.arange(24, dtype=np.int64) * H1
        close = 100 * np.cumprod(1 + np.r_[0.0, np.linspace(-0.01, 0.01, 23)])
        s = np.zeros(24, dtype=np.int8)
        s[0] = 1
        out = S.turnover_aware(s, 1, close, ts, 0, 24 * H1, H1, 2, cost_bps=10.0)
        r = close[1:] / close[:-1] - 1          # r[u] is the return of bar u+1
        gross = 0.5 * r[1] + 0.5 * r[2]          # held over bars 2 and 3
        assert out["n_days"] == 1.0
        assert out["turnover_per_day"] == pytest.approx(1.0)
        assert out["gross_bps_per_day"] == pytest.approx(gross * 1e4)
        assert out["value_bps"] == pytest.approx((gross - 0.0005) * 1e4)
        out_short = S.turnover_aware(s, -1, close, ts, 0, 24 * H1, H1, 2)
        assert out_short["gross_bps_per_day"] == pytest.approx(-gross * 1e4)
        # Per day = the sum over the window divided by the window's length in days.
        ts2 = np.arange(48, dtype=np.int64) * H1
        close2 = np.r_[close, np.full(24, close[-1])]
        s2 = np.r_[s, np.zeros(24, dtype=np.int8)]
        out2 = S.turnover_aware(s2, 1, close2, ts2, 0, 48 * H1, H1, 2, cost_bps=10.0)
        assert out2["n_days"] == 2.0
        assert out2["value_bps"] == pytest.approx(out["value_bps"] / 2)

    def test_turnover_aware_ignores_non_evaluable_signals_and_counts_gaps(self):
        ts = np.arange(24, dtype=np.int64) * H1
        close = np.linspace(100, 101, 24)
        s = np.zeros(24, dtype=np.int8)
        s[22] = 1                                # forward window leaves the window
        out = S.turnover_aware(s, 1, close, ts, 0, 24 * H1, H1, 2)
        assert out["turnover_per_day"] == 0
        keep = np.ones(24, dtype=bool)
        keep[3] = False                          # bucket 3 dropped
        s = np.zeros(24, dtype=np.int8)
        s[0] = 1
        out = S.turnover_aware(s[keep], 1, close[keep], ts[keep], 0, 24 * H1, H1, 2)
        assert out["n_unpriced_steps"] == 1      # p_2 earns C_3/C_2 - 1, and bar 3 is gone
        assert out["turnover_per_day"] == pytest.approx(1.0)

    def test_cost_sensitivity(self):
        rng = np.random.default_rng(17)
        ts, s, fwd = null_cell(rng, 800, 1)
        rows = S.cost_sensitivity(s, fwd, 1, ts, 0, 800 * H1, H1, 1)
        assert [r["cost_bps"] for r in rows] == get_spec()["robustness"]["cost_sensitivity_bps"]
        assert rows[1]["mu"] == pytest.approx(rows[0]["mu"] - 0.0004)
        assert rows[1]["se"] == pytest.approx(rows[0]["se"])

    def test_winsorized_fwd(self):
        rng = np.random.default_rng(18)
        ts = np.arange(1000, dtype=np.int64) * H1
        fwd = rng.standard_t(2, 1000) * 0.01
        fwd[5] = np.nan
        fwd[998] = 10.0                          # not evaluable: does not set the bounds
        w = S.winsorized_fwd(fwd, ts, 0, 1000 * H1, H1, 1)
        ev = S.evaluable(ts, 0, 1000 * H1, H1, 1) & np.isfinite(fwd)
        lo, hi = np.percentile(fwd[ev], [1, 99])
        assert np.nanmin(w) == pytest.approx(lo) and w[998] == pytest.approx(hi)
        assert np.isnan(w[5])
        assert np.sum(w[ev] != fwd[ev]) == pytest.approx(0.02 * ev.sum(), abs=2)

    @pytest.mark.parametrize("prior, x, want", [
        ([1, 2, 3, 4, 5], ("e1", -1), 1),        # Q1 = f < e1
        ([1, 2, 3, 4, 5], ("e1", 0), 2),         # Q2 = [e1, e2)
        ([1, 2, 3, 4, 5], ("e2", -1), 2),
        ([1, 2, 3, 4, 5], ("e2", 0), 3),         # Q3 = [e2, e3]
        ([1, 2, 3, 4, 5], ("e3", 0), 3),
        ([1, 2, 3, 4, 5], ("e3", 1), 4),         # Q4 = (e3, e4]
        ([1, 2, 3, 4, 5], ("e4", 0), 4),
        ([1, 2, 3, 4, 5], ("e4", 1), 5),         # Q5 = f > e4
        ([1, 1, 1, 1, 5], ("e1", 0), 3),         # e1 = e2 = e3: Q2 folds into Q3
        ([1, 1, 1, 1, 5], ("e4", 0), 4),
        ([1, 5, 5, 5, 5], ("e4", 0), 3),         # e2 = e3 = e4: Q4 folds into Q3
        ([1, 5, 5, 5, 5], ("e1", 0), 2),
    ])
    def test_quintile_rule_of_r10(self, prior, x, want):
        """One evaluable bar whose value sits on, just below or just above an edge."""
        e = dict(zip(("e1", "e2", "e3", "e4"), np.percentile(prior, [20, 40, 60, 80])))
        name, step = x
        v = e[name] if step == 0 else np.nextafter(e[name], step * np.inf)
        f = np.r_[np.asarray(prior, dtype=float), v, 0.0, 0.0]
        ts = np.arange(8, dtype=np.int64) * H1
        fwd = S.forward_returns(np.linspace(1.0, 1.7, 8), ts, H1, 1)
        rows = S.quintile_means(f, fwd, ts, 5 * H1, 8 * H1, H1, 1, min_prior=5)   # only bar 5 evaluable
        assert [r["n_active"] for r in rows] == [int(q == want) for q in range(1, 6)]
        assert rows[want - 1]["value"] == pytest.approx(fwd[5])

    def test_quintile_means_point_mass_and_merging(self):
        rng = np.random.default_rng(19)
        n_all, warm, h = 3000, 600, 1
        ts = np.arange(n_all, dtype=np.int64) * H1
        start, end = warm * H1, n_all * H1
        fwd = rng.standard_normal(n_all) * 0.01
        fwd[-2:] = np.nan
        # 60% mass at 0 straddling the middle: e2 == e3 == 0
        f = np.where(rng.random(n_all) < 0.6, 0.0, rng.standard_normal(n_all))
        rows = S.quintile_means(f, fwd, ts, start, end, H1, h)
        lo, hi = S.expanding_edges(f)
        s = S.assign_buckets(f, lo, hi)
        ev = S.evaluable(ts, start, end, H1, h) & np.isfinite(fwd)
        assert rows[0]["n_active"] == int(np.sum(ev & (s == -1)))   # Q1 is §6.4's bottom
        assert rows[4]["n_active"] == int(np.sum(ev & (s == 1)))    # Q5 is §6.4's top
        assert rows[2]["n_active"] >= int(np.sum(ev & (f == 0)))    # the mass sits in Q3
        assert sum(r["n_active"] for r in rows) == int(ev.sum())
        assert rows[0]["value"] == pytest.approx(fwd[ev & (s == -1)].mean())

        # Mass at the bottom: e1 == e2 == 0, so Q2 is merged into Q3.
        g = np.where(rng.random(n_all) < 0.5, 0.0, rng.random(n_all) + 0.1)
        rows = S.quintile_means(g, fwd, ts, start, end, H1, h)
        assert rows[0]["n_active"] == 0            # nothing below the mass
        assert rows[1]["n_active"] == 0 and rows[1]["merged_bars"] == int(ev.sum())
        assert rows[2]["n_active"] >= int(np.sum(ev & (g == 0)))
        assert rows[2]["merged_bars"] == 0


def brute_runs(s, ts, start, end, B, h):
    """Runs by walking the clock bar by bar, with a dict keyed on time (no grid arithmetic)."""
    by_t = {int(t): int(v) for t, v in zip(ts, s) if start <= t < end and t + (h + 2) * B <= end}
    runs, n, prev = 0, 0, 0
    for t in range(start, end, B):
        v = by_t.get(t, 0)
        if v != 0:
            n += 1
            runs += v != prev
        prev = v
    return (n / runs if runs else math.nan), runs


class TestMeanRunLength:
    def test_hand_example(self):
        # Window [10h, 30h), h = 1: grid positions 0..17 are evaluable
        # (ts + 3B <= end), 18 and 19 are not. Position 6 is a dropped bucket.
        # Three warm-up bars before the window carry s = 1 and must not join
        # the first run.
        start, end, h = 10 * H1, 30 * H1, 1
        grid_s = {0: 1, 1: 1, 2: -1, 3: -1, 4: -1, 5: 1, 7: 1, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1,
                  13: -1, 14: -1, 15: -1, 16: -1, 17: 1, 18: 1, 19: 1}
        ts = np.r_[np.array([7, 8, 9]) * H1, start + np.array(sorted(grid_s)) * H1].astype(np.int64)
        s = np.r_[[1, 1, 1], [grid_s[k] for k in sorted(grid_s)]].astype(np.int8)
        # Runs: [0,1] [2..4] [5] (ended by the dropped bucket) [7] (ended by s = 0)
        # [9..12] [13..16] [17] (18 is not evaluable): 7 runs over 16 bars.
        mean, runs = S.mean_run_length(s, ts, start, end, H1, h)
        assert runs == 7 and mean == pytest.approx(16 / 7)
        assert (mean, runs) == pytest.approx(brute_runs(s, ts, start, end, H1, h))
        # Fill the dropped bucket with s = 1: [5, 6, 7] become one run, so
        # 6 runs over 17 bars.
        ts6 = np.sort(np.r_[ts, start + 6 * H1])
        s6 = np.insert(s, np.searchsorted(ts, start + 6 * H1), 1)
        assert S.mean_run_length(s6, ts6, start, end, H1, h) == (pytest.approx(17 / 6), 6)

    def test_no_signal_and_longer_h(self):
        ts = np.arange(50, dtype=np.int64) * M15
        mean, runs = S.mean_run_length(np.zeros(50, dtype=np.int8), ts, 0, 50 * M15, M15, 5)
        assert runs == 0 and math.isnan(mean)
        s = np.ones(50, dtype=np.int8)
        # h = 5: the last h + 1 = 6 grid positions are not evaluable, so one run of 44.
        assert S.mean_run_length(s, ts, 0, 50 * M15, M15, 5) == (44.0, 1)
        with pytest.raises(ValueError):
            S.mean_run_length(s[:-1], ts, 0, 50 * M15, M15, 5)

    def test_matches_a_clock_walk_on_random_signals_with_gaps(self):
        rng = np.random.default_rng(31)
        for h in (1, 5, 15):
            n = 3000
            ts = np.sort(rng.choice(np.arange(n + 300), n, replace=False)).astype(np.int64) * M15
            s = rng.choice(np.array([-1, 0, 1], dtype=np.int8), n, p=[0.2, 0.6, 0.2])
            s = np.where(rng.random(n) < 0.7, np.r_[s[:1], s[:-1]], s).astype(np.int8)  # persistent
            start, end = 200 * M15, (n + 300) * M15
            got = S.mean_run_length(s, ts, start, end, M15, h)
            want = brute_runs(s, ts, start, end, M15, h)
            assert got[1] == want[1] and got[0] == pytest.approx(want[0], rel=1e-12)
            assert got[1] > 100


class TestFunding:
    @staticmethod
    def _setup(h):
        ts = np.arange(48, dtype=np.int64) * H1
        n = np.full(48, 0.001)
        s = np.ones(48, dtype=np.int8)
        fund_ts = np.arange(0, 56, 8, dtype=np.int64) * H1    # 00, 08, 16 each day
        fund_rate = np.full(len(fund_ts), 0.0001)
        return ts, n, s, fund_ts, fund_rate

    def test_window_boundaries(self):
        h = 1
        ts, n, s, fund_ts, fr = self._setup(h)
        out = S.funding_adjusted_net(n, s, 1, ts, H1, h, fund_ts, fr)
        # entry = ts+2h, exit = ts+3h; settlement at 08:00 counts iff entry <= 8h < exit
        assert out[6] == pytest.approx(0.001 - 0.0001)    # entry 08:00: included
        assert out[5] == pytest.approx(0.001)              # exit 08:00: excluded
        assert out[4] == pytest.approx(0.001)
        h = 5
        out = S.funding_adjusted_net(n, s, 1, ts, H1, h, fund_ts, fr)
        assert out[3] == pytest.approx(0.001 - 0.0001)     # [05:00, 10:00) holds 08:00
        assert out[1] == pytest.approx(0.001)              # [03:00, 08:00) does not

    def test_sign_convention(self):
        ts, n, s, fund_ts, fr = self._setup(1)
        short = -s
        assert S.funding_adjusted_net(n, short, 1, ts, H1, 1, fund_ts, fr)[6] == pytest.approx(0.0011)
        assert S.funding_adjusted_net(n, short, -1, ts, H1, 1, fund_ts, fr)[6] == pytest.approx(0.0009)
        neg = -fr
        assert S.funding_adjusted_net(n, s, 1, ts, H1, 1, fund_ts, neg)[6] == pytest.approx(0.0011)

    def test_missing_settlement_is_nan(self):
        ts, n, s, fund_ts, fr = self._setup(1)
        keep = fund_ts != 8 * H1
        out = S.funding_adjusted_net(n, s, 1, ts, H1, 1, fund_ts[keep], fr[keep])
        assert np.isnan(out[6])                            # 08:00 expected, not recorded
        assert out[5] == pytest.approx(0.001)              # 08:00 outside [07:00, 08:00)
        assert out[14] == pytest.approx(0.001 - 0.0001)    # 16:00 present
        fr2 = fr.copy()
        fr2[2] = np.nan                                    # 16:00 recorded as NaN
        assert np.isnan(S.funding_adjusted_net(n, s, 1, ts, H1, 1, fund_ts, fr2)[14])
        # series that stops early: later settlements are missing, not zero
        out = S.funding_adjusted_net(n, s, 1, ts, H1, 1, fund_ts[:3], fr[:3])
        assert np.isnan(out[22])

    def test_inactive_and_gaps(self):
        ts, n, s, fund_ts, fr = self._setup(1)
        n = n.copy()
        n[6] = np.nan
        s = s.copy()
        s[7] = 0
        out = S.funding_adjusted_net(n, s, 1, ts, H1, 1, fund_ts, fr)
        assert np.isnan(out[6]) and np.isnan(out[7])
        with pytest.raises(ValueError):
            S.funding_adjusted_net(n, s, 1, ts, H1, 1, np.r_[fund_ts, fund_ts[-1]], np.r_[fr, 0.0])
