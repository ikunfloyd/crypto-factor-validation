"""
Tests for research/factors.py: the ten factors of §5.

Synthetic data only. Nothing here computes a factor-versus-forward-return
statistic, on synthetic data or otherwise.

The central property is prefix invariance: a factor value at bar t, computed
from data up to bar t's close, must equal the value at t computed on the
whole series. A factor that fails it is using the future.
"""
import copy
import re
from pathlib import Path

import numba
import numpy as np
import polars as pl
import pytest

import research.bars as bars_mod
import research.factors as factors_mod
from research.bars import bar_ms, htf_bar, htf_source_index, resample_complete, to_grid
from research.factors import (
    FACTOR_NAMES,
    _engine,
    _hurst_kernel,
    _ou_kernel,
    compute_factors,
    funding_fresh,
    hurst_exponent,
    hurst_regime,
    irregular_settlements,
    ou_reversion,
    stoch_k,
    triple_screen,
)
from research.spec import MINUTE_MS, get_spec

H = 3_600_000
DAY_MS = 24 * H
# A UTC midnight, so every bucket size and the 8-hourly settlements align.
DAY0 = (1_700_000_000_000 // DAY_MS) * DAY_MS
SPEC = get_spec()["factors"]


# ---------------------------------------------------------------- synthetic data

def _synthetic(n_bars, bar="15m", seed=0, perp_drop=(), spot_drop=(), funding_drop=()):
    """
    Random-walk 1-minute perp and spot candles resampled to `bar` and its HTF,
    plus 8-hourly funding starting 20 days earlier. *_drop name bucket
    indices (or settlement times) to knock out.
    """
    rng = np.random.default_rng(seed)
    per = bar_ms(bar) // MINUTE_MS
    n = n_bars * per
    ts = DAY0 + np.arange(n, dtype=np.int64) * MINUTE_MS
    close = 0.1 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 5e-4, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 5e-4, n))
    vol = rng.uniform(1e3, 1e4, n)
    perp = pl.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": vol, "volume_ccy": vol * close})
    k = pl.Series(rng.normal(-5e-4, 2e-4, n))
    spot = perp.with_columns(*(pl.col(c) * (1 + k) for c in ("open", "high", "low", "close")))

    def holes(df, buckets):
        gone = [int(DAY0 + b * per * MINUTE_MS + 3 * MINUTE_MS) for b in buckets]
        return df.filter(~pl.col("timestamp").is_in(gone))

    perp, spot = holes(perp, perp_drop), holes(spot, spot_drop)
    f_ts = np.arange(DAY0 - 20 * DAY_MS, ts[-1] + DAY_MS, 8 * H, dtype=np.int64)
    f_ts = f_ts[~np.isin(f_ts, np.asarray(funding_drop, dtype=np.int64))]
    funding = pl.DataFrame({"timestamp": f_ts, "rate": rng.normal(1e-4, 5e-5, len(f_ts))})
    return {
        "bar": bar,
        "bars": resample_complete(perp, bar),
        "htf": resample_complete(perp, htf_bar(bar)),
        "spot": resample_complete(spot, bar),
        "funding": funding,
    }


def _small_params():
    """Appendix A with short windows, so every factor is defined on a few hundred bars."""
    p = copy.deepcopy(SPEC)
    p["macd_hist_norm"].update(fast=3, slow=6, signal=3)
    p["rolling_vol_ratio"].update(short=5, long=30)
    p["net_taker_flow"]["lookback"] = 10
    p["basis_z_score"]["lookback"] = 30
    p["funding_cost_pressure"]["lookback_settlements"] = 20
    p["ou_reversion"]["window"] = 20
    p["hurst_regime"].update(window=40, q=5)
    p["triple_screen"]["macd"] = [3, 6, 3]
    return p


def _compute(d, params=None, **kw):
    return compute_factors(d["bars"], d["bar"], d["spot"], d["funding"], d["htf"], params=params, **kw)


def _prefix(d, t):
    """Everything known at the close of bar t."""
    B = bar_ms(d["bar"])
    HTF = bar_ms(htf_bar(d["bar"]))
    s = d["bars"]["timestamp"][t]
    return {
        "bar": d["bar"],
        "bars": d["bars"][: t + 1],
        "spot": d["spot"].filter(pl.col("timestamp") <= s),
        "funding": d["funding"].filter(pl.col("timestamp") <= s),
        "htf": d["htf"].filter(pl.col("timestamp") + HTF <= s + B),
    }


def _assert_prefix_invariant(d, cuts, params=None):
    full = _compute(d, params)
    for name in FACTOR_NAMES:
        assert np.isfinite(full[name].to_numpy()).sum() > 0, f"{name} never defined; test is vacuous"
    for t in cuts:
        pre = _compute(_prefix(d, t), params)
        assert pre.height == t + 1
        for name in FACTOR_NAMES:
            np.testing.assert_array_equal(
                pre[name].to_numpy(), full[name].to_numpy()[: t + 1], err_msg=f"{name} at t={t}"
            )
    return full


# ---------------------------------------------------------------- 3. no look-ahead

def test_prefix_invariance_every_bar_short_windows():
    # Gaps everywhere: perp buckets, spot buckets (one inside the first basis
    # window, where the engine's coverage check bites) and a missing settlement.
    d = _synthetic(200, perp_drop=(60, 120), spot_drop=(5, 70, 150),
                   funding_drop=(DAY0 + 16 * H,), seed=1)
    _assert_prefix_invariant(d, range(d["bars"].height), _small_params())


def test_prefix_invariance_real_parameters():
    d = _synthetic(1700, perp_drop=(800,), spot_drop=(3, 500), funding_drop=(DAY0 + 5 * DAY_MS,), seed=2)
    n = d["bars"].height
    full = _assert_prefix_invariant(d, [33, 34, 240, 1438, 1439, 1440, 1441, 1442, n - 2, n - 1])
    # The long windows really are 1440 bars here, not something shorter.
    assert np.isnan(full["hurst_regime"].to_numpy()[:1440]).all()
    assert np.isfinite(full["hurst_regime"].to_numpy()[1440])
    assert np.isnan(full["ou_reversion"].to_numpy()[:240]).all()
    assert np.isfinite(full["ou_reversion"].to_numpy()[240])


def test_prefix_invariance_one_hour_bars():
    d = _synthetic(260, bar="1H", spot_drop=(8,), seed=3)
    _assert_prefix_invariant(d, range(0, d["bars"].height, 7), _small_params())


def test_prefix_invariance_when_the_settlement_spacing_changes():
    # 8-hourly settlements until DAY0 + 1 day, 4-hourly after, so the whole
    # series' median spacing is 4 h while every prefix ending before the
    # switch sees 8 h. A limit taken from the series would mask the early
    # bars 4-8 h after a settlement in the full run but not in the prefix:
    # later settlements deciding an earlier bar's value. §4.2's fixed 8 h
    # gives the same mask on both.
    d = _synthetic(5 * 96, seed=21)
    switch = DAY0 + DAY_MS
    f_ts = np.r_[np.arange(DAY0 - 4 * DAY_MS, switch, 8 * H),
                 np.arange(switch, DAY0 + 5 * DAY_MS, 4 * H)].astype(np.int64)
    rng = np.random.default_rng(22)
    d["funding"] = pl.DataFrame({"timestamp": f_ts, "rate": rng.normal(1e-4, 5e-5, len(f_ts))})
    assert np.median(np.diff(f_ts)) == 4 * H        # a series median would say 4 h
    early = d["funding"].filter(pl.col("timestamp") < switch)["timestamp"].to_numpy()
    assert np.median(np.diff(early)) == 8 * H       # ... and 8 h on any prefix before the switch

    ts = d["bars"]["timestamp"].to_numpy()
    age = ts - early[np.searchsorted(early, ts, side="right") - 1]
    late_but_fresh = np.flatnonzero((ts < switch) & (age >= 4 * H) & (age < 8 * H))
    assert len(late_but_fresh) == 48                # DAY0 04:00-08:00 and 12:00-16:00, 20:00-24:00
    cuts = [int(late_but_fresh[0]) - 1, *late_but_fresh[::7].tolist(), int(late_but_fresh[-1]),
            int(np.searchsorted(ts, switch)) + 20, d["bars"].height - 1]
    full = _assert_prefix_invariant(d, cuts, _small_params())
    # Those bars are defined: no later settlement withdrew them.
    assert np.isfinite(full["funding_cost_pressure"].to_numpy()[late_but_fresh]).all()


# ---------------------------------------------------------------- 4. undefined -> NaN

def _random_close(n, seed=0):
    rng = np.random.default_rng(seed)
    return 0.1 * np.exp(np.cumsum(rng.normal(0, 2e-3, n)))


def _expected_defined(n, window, bad_idx):
    """Defined iff `window` steps exist and x[t-window .. t] are all finite."""
    ok = np.zeros(n, dtype=bool)
    ok[window:] = True
    for b in bad_idx:
        ok[b: b + window + 1] = False
    return ok


def test_ou_undefined_cases():
    w = SPEC["ou_reversion"]["window"]
    assert np.isnan(ou_reversion(np.full(w + 50, 0.1), w)).all()  # constant: sum (x - xbar)^2 = 0

    close = _random_close(3 * w)
    bad = [w + 10, 2 * w, 2 * w + 30]
    close[bad[0]], close[bad[1]], close[bad[2]] = np.nan, 0.0, -1.0
    out = ou_reversion(close, w)
    np.testing.assert_array_equal(np.isfinite(out), _expected_defined(len(close), w, bad))
    inf_close = _random_close(w + 5)
    inf_close[-1] = np.inf
    assert np.isnan(ou_reversion(inf_close, w)[-1])


def test_ou_sigma_e_zero_with_nonzero_x_variance():
    # An exactly linear log price fits the AR(1) perfectly: sigma_e = 0 while
    # the lagged x varies. Log of a float close is never exactly linear, so
    # this runs the kernel on log prices that are binary fractions.
    x = 0.5 * np.arange(60, dtype=np.float64)
    assert np.isnan(_ou_kernel(x, 10)).all()


def test_ou_sigma_e_zero_after_a_jump_and_w_flat_bars():
    # A jump followed by W equal closes: the W current log prices are equal,
    # so the AR(1) fits them exactly (phi = 0, every residual 0) while the
    # lagged series still varies through the pre-jump price. sigma_e = 0
    # must give NaN. Centring both series on the pre-jump price left the
    # current one's mean off by an ulp and returned noise of order one.
    w = SPEC["ou_reversion"]["window"]
    assert w == 240
    close = np.r_[0.39149, np.full(w, 0.30311)]
    assert np.isnan(ou_reversion(close, w)[-1])
    for jump, flat in ((0.1, 0.3), (0.2718, 0.1234), (1.5, 0.75)):
        assert np.isnan(ou_reversion(np.r_[jump, np.full(20, flat)], 20)[-1])
    # One bar later the jump has left the window: x constant, also NaN.
    assert np.isnan(ou_reversion(np.r_[0.39149, np.full(w + 1, 0.30311)], w)[-1])


def test_hurst_undefined_cases():
    w, q = 60, 5
    assert np.isnan(hurst_regime(np.full(3 * w, 0.1), w, q)).all()  # sigma_a = 0

    close = _random_close(4 * w, seed=4)
    bad = [w + 3, 3 * w]
    close[bad[0]], close[bad[1]] = np.nan, 0.0
    out = hurst_regime(close, w, q)
    np.testing.assert_array_equal(np.isfinite(out), _expected_defined(len(close), w, bad))


def test_hurst_vr_zero_with_nonzero_sigma():
    # Returns repeating with period q: every q-sum equals q * mean return, so
    # VR = 0 although sigma_a > 0. Binary-fraction log prices keep it exact.
    q, w = 5, 20
    r = np.tile(np.array([1.0, -1.0, 0.5, -0.5, 0.0]) * 0.25, 12)
    x = np.r_[0.0, np.cumsum(r)]
    h_hat, mom = _hurst_kernel(x, w, q)
    assert np.isnan(h_hat).all() and np.isnan(mom).all()


def test_stoch_k_matches_brute_force_and_zero_range_is_nan():
    rng = np.random.default_rng(6)
    n, k = 80, SPEC["triple_screen"]["stoch_k"]
    close = rng.uniform(1, 2, n)
    high, low = close + rng.uniform(0, 0.5, n), close - rng.uniform(0, 0.5, n)
    high[40:40 + k] = low[40:40 + k] = close[40:40 + k] = 1.5  # one flat window
    out = stoch_k(high, low, close, k)
    for t in range(n):
        if t < k - 1:
            assert np.isnan(out[t])
            continue
        hh, ll = high[t - k + 1: t + 1].max(), low[t - k + 1: t + 1].min()
        if hh == ll:
            assert np.isnan(out[t])
        else:
            assert out[t] == pytest.approx(100 * (close[t] - ll) / (hh - ll), rel=1e-12)
    assert np.isnan(out[40 + k - 1])


def _htf_frame(close, start=DAY0, period=H):
    ts = start + np.arange(len(close), dtype=np.int64) * period
    return pl.DataFrame({"timestamp": ts, "open": close, "high": close, "low": close,
                         "close": close, "volume": np.ones(len(close))})


def test_triple_screen_undefined_during_htf_warmup_and_before_any_htf_close():
    rng = np.random.default_rng(7)
    htf = _htf_frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 40))))
    fast, slow, signal = SPEC["triple_screen"]["macd"]
    first_valid = slow + signal - 1  # first HTF index known without look-ahead

    q = 15 * MINUTE_MS
    n = 4 * 50
    ts = DAY0 - 5 * H + np.arange(n, dtype=np.int64) * q
    c = 0.1 * np.exp(np.cumsum(rng.normal(0, 2e-3, n)))
    bars = pl.DataFrame({"timestamp": ts, "open": c, "high": c * 1.001, "low": c * 0.999,
                         "close": c, "volume": np.ones(n)})
    out = triple_screen(bars, htf, "15m")

    idx = htf_source_index(ts, q, htf["timestamp"].to_numpy(), H)
    k = stoch_k(bars["high"].to_numpy(), bars["low"].to_numpy(), c, SPEC["triple_screen"]["stoch_k"])
    want = (idx >= first_valid) & np.isfinite(k)
    np.testing.assert_array_equal(np.isfinite(out), want)
    assert (idx == -1).any() and (idx == first_valid - 1).any()  # both cases exercised

    flat = bars.with_columns(pl.lit(0.1).alias(col) for col in ("high", "low", "close"))
    assert np.isnan(triple_screen(flat, htf, "15m")).all()  # %K range zero
    assert np.isnan(triple_screen(bars, None, "15m")).all()


# ---------------------------------------------------------------- 5. triple_screen signs

def _pullback_bars(k_last, start, n=20):
    """15m bars with a 1..101 range, so the last bar's %K equals k_last."""
    ts = start + np.arange(n, dtype=np.int64) * 15 * MINUTE_MS
    close = np.full(n, 51.0)
    close[-1] = 1.0 + k_last
    return pl.DataFrame({"timestamp": ts, "open": close, "high": np.full(n, 101.0),
                         "low": np.full(n, 1.0), "close": close, "volume": np.ones(n)})


@pytest.mark.parametrize("trend,k_last,expected", [
    (+1, 10.0, 0.8), (+1, 90.0, 0.0), (-1, 90.0, -0.8), (-1, 10.0, 0.0),
])
def test_triple_screen_sign_cases(trend, k_last, expected):
    j = np.arange(60, dtype=np.float64)
    htf_close = 100 + 0.05 * j ** 2 if trend > 0 else 1000 - 0.05 * j ** 2
    htf = _htf_frame(htf_close)
    hist = _engine().calculate_macd_hist({"candles": htf}).to_numpy()
    assert np.sign(hist[-1]) == trend  # the HTF really trends the intended way

    bars = _pullback_bars(k_last, DAY0 + 60 * H)
    out = triple_screen(bars, htf, "15m")
    assert out[-1] == pytest.approx(expected, abs=1e-12)
    assert not np.signbit(out[-1]) or expected < 0  # 0 is +0, not -0


# ---------------------------------------------------------------- 6. formulas

def _ou_direct(close, t, w):
    x = np.log(close)
    z, y = x[t - w: t], x[t - w + 1: t + 1]
    (a, phi), *_ = np.linalg.lstsq(np.c_[np.ones(w), z], y, rcond=None)
    e = y - a - phi * z
    return (a + (phi - 1) * x[t]) / np.sqrt(e @ e / (w - 2))


def test_ou_matches_direct_ols():
    w = SPEC["ou_reversion"]["window"]
    close = _random_close(w + 400, seed=8)
    out = ou_reversion(close, w)
    rng = np.random.default_rng(9)
    for t in rng.choice(np.arange(w, len(close)), 25, replace=False):
        assert out[t] == pytest.approx(_ou_direct(close, t, w), rel=1e-6, abs=1e-9)
    # A window above its mean forecasts a fall: the sign convention of §5.2.
    assert np.corrcoef(out[w:], np.log(close)[w:] - np.convolve(
        np.log(close), np.ones(w) / w, "valid")[: len(close) - w])[0, 1] < 0


def _hurst_direct(close, t, w, q):
    x = np.log(close)
    r = np.diff(x)[t - w: t]
    rbar = r.mean()
    var_a = ((r - rbar) ** 2).sum() / (w - 1)
    qsum = np.convolve(r, np.ones(q), "valid")
    var_c = ((qsum - q * rbar) ** 2).sum() / (q * (w - q + 1) * (1 - q / w))
    h = 0.5 + np.log(var_c / var_a) / (2 * np.log(q))
    m = (x[t] - x[t - q]) / (np.sqrt(var_a) * np.sqrt(q))
    return h, (h - 0.5) * m


def test_hurst_matches_direct_formula():
    w, q = SPEC["hurst_regime"]["window"], SPEC["hurst_regime"]["q"]
    close = _random_close(w + 300, seed=10)
    h_out, f_out = hurst_exponent(close, w, q), hurst_regime(close, w, q)
    for t in np.random.default_rng(11).choice(np.arange(w, len(close)), 15, replace=False):
        h, f = _hurst_direct(close, t, w, q)
        assert h_out[t] == pytest.approx(h, rel=1e-9)
        assert f_out[t] == pytest.approx(f, rel=1e-7, abs=1e-12)


def _ar1_walk(n, rho, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1e-3, n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = rho * r[i - 1] + e[i]
    return 0.1 * np.exp(np.cumsum(r))


def test_hurst_near_half_on_random_walk_and_ordered_by_autocorrelation():
    w, q = SPEC["hurst_regime"]["window"], SPEC["hurst_regime"]["q"]
    n = 12 * w
    iid = np.nanmean(hurst_exponent(_ar1_walk(n, 0.0, 12), w, q))
    trending = np.nanmean(hurst_exponent(_ar1_walk(n, 0.3, 13), w, q))
    reverting = np.nanmean(hurst_exponent(_ar1_walk(n, -0.3, 14), w, q))
    assert abs(iid - 0.5) < 0.03
    assert trending > 0.55 and reverting < 0.45


# ---------------------------------------------------------------- 7. no fastmath

def test_new_numba_kernels_are_compiled_without_fastmath():
    kernels = {
        name: obj
        for mod in (factors_mod, bars_mod)
        for name, obj in vars(mod).items()
        if isinstance(obj, numba.core.dispatcher.Dispatcher)
    }
    assert {"_ou_kernel", "_hurst_kernel", "_bad_prefix"} <= set(kernels)
    for name, k in kernels.items():
        assert not k.targetoptions.get("fastmath"), name
    for mod in (factors_mod, bars_mod):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert not re.search(r"fastmath\s*=\s*(?!False\b)", src), mod.__name__


# ---------------------------------------------------------------- 8. basis and funding

def test_dropped_spot_bucket_gives_nan_basis_not_carried_forward():
    p = _small_params()
    look = p["basis_z_score"]["lookback"]
    d = _synthetic(120, spot_drop=(80,), seed=15)
    basis = _compute(d, p)["basis_z_score"].to_numpy()
    assert np.isnan(basis[80])
    assert np.isfinite(basis[79]) and np.isfinite(basis[81])

    # The unchanged engine already leaves that bar NaN once the spot leg is
    # on the perp grid; handed the raw spot bars, it would carry 79 forward.
    ts = d["bars"]["timestamp"].to_numpy()
    eng = _engine()
    on_grid = eng.calculate_basis_z_score(
        {"candles": d["bars"], "spot_candles": to_grid(d["spot"], ts)}, lookback=look).to_numpy()
    raw = eng.calculate_basis_z_score(
        {"candles": d["bars"], "spot_candles": d["spot"]}, lookback=look).to_numpy()
    assert np.isnan(on_grid[80]) and np.isfinite(raw[80])
    np.testing.assert_array_equal(basis, on_grid)


def test_basis_coverage_is_counted_up_to_each_bar():
    # One spot bucket missing inside the first window: the engine's
    # whole-series check would emit bar look-1 from look-1 values, which a
    # run ending at that bar could not. The causal mask withholds it.
    p = _small_params()
    look = p["basis_z_score"]["lookback"]
    d = _synthetic(120, spot_drop=(5,), seed=16)
    basis = _compute(d, p)["basis_z_score"].to_numpy()
    engine_out = _engine().calculate_basis_z_score(
        {"candles": d["bars"], "spot_candles": to_grid(d["spot"], d["bars"]["timestamp"].to_numpy())},
        lookback=look).to_numpy()
    assert np.isfinite(engine_out[look - 1]) and np.isnan(basis[look - 1])
    assert np.isnan(basis[: look]).all() and np.isfinite(basis[look])
    np.testing.assert_array_equal(basis[look:], engine_out[look:])


def test_funding_staleness_mask():
    gap = DAY0 + 16 * H  # missing settlement
    d = _synthetic(3 * 96, funding_drop=(gap,), seed=17)
    ts = d["bars"]["timestamp"].to_numpy()
    fcp = _compute(d)["funding_cost_pressure"].to_numpy()
    engine_out = _engine().calculate_funding_cost_pressure(
        {"candles": d["bars"], "funding_rates": d["funding"]},
        lookback=SPEC["funding_cost_pressure"]["lookback_settlements"]).to_numpy()

    stale = (ts >= gap) & (ts < gap + 8 * H)  # 8 h or more since 08:00, ">=" included
    assert stale.sum() == 32
    assert np.isnan(fcp[stale]).all()
    assert np.isfinite(engine_out[stale]).all()  # the as-of join alone would show 08:00's value
    np.testing.assert_array_equal(fcp[~stale], engine_out[~stale])
    assert np.isfinite(fcp[~stale]).all()

    # The limit is the spec's max_staleness_hours: a 16 h limit masks nothing here.
    assert SPEC["funding_cost_pressure"]["max_staleness_hours"] == 8
    loose = copy.deepcopy(SPEC)
    loose["funding_cost_pressure"]["max_staleness_hours"] = 16
    relaxed = _compute(d, loose)["funding_cost_pressure"].to_numpy()
    np.testing.assert_array_equal(relaxed, engine_out)


def test_funding_limit_is_a_fixed_8h_whatever_the_spacing():
    # 4-hourly settlements with the 20:00 one missing. Under a limit taken
    # from the series (4 h) the bars 20:00-24:00 would be stale; under §4.2's
    # fixed 8 h only bars 8 h or more after 16:00 are, i.e. none before the
    # next settlement at 24:00.
    settle = DAY0 + np.arange(0, 48, 4, dtype=np.int64) * H
    settle = np.delete(settle, 5)
    bar_ts = DAY0 - H + np.arange(52, dtype=np.int64) * H
    fresh = funding_fresh(bar_ts, settle, 8 * H)
    want = (bar_ts >= DAY0) & (bar_ts < settle[-1] + 8 * H)
    np.testing.assert_array_equal(fresh, want)
    assert fresh[(bar_ts >= DAY0 + 20 * H) & (bar_ts < DAY0 + 24 * H)].all()
    # A bar exactly 8 h after its settlement is stale (">=", Appendix A staleness_test: ge).
    assert not funding_fresh(np.array([DAY0 + 8 * H]), np.array([DAY0]), 8 * H)[0]
    assert funding_fresh(np.array([DAY0 + 8 * H - 1]), np.array([DAY0]), 8 * H)[0]

    # The other spacings are reported (§4.2), each with the settlement that ends it.
    reported = irregular_settlements(settle)
    assert reported == [(int(t), int(dt)) for t, dt in zip(settle[1:], np.diff(settle)) if dt != 8 * H]
    assert (int(DAY0 + 24 * H), 8 * H) not in reported and len(reported) == len(settle) - 2
    assert irregular_settlements(DAY0 + np.arange(0, 48, 8, dtype=np.int64) * H) == []
    assert irregular_settlements(settle[:1]) == []


def test_funding_mask_ignores_the_series_spacing_end_to_end():
    # The same bars with 4-hourly funding: compute_factors masks at 8 h after
    # the last settlement, not at 4 h, so dropping one 4-hourly settlement
    # masks nothing.
    d = _synthetic(3 * 96, seed=23)
    f_ts = np.arange(DAY0 - 5 * DAY_MS, DAY0 + 4 * DAY_MS, 4 * H, dtype=np.int64)
    f_ts = f_ts[f_ts != DAY0 + 28 * H]
    d["funding"] = pl.DataFrame({"timestamp": f_ts, "rate": np.random.default_rng(24).normal(1e-4, 5e-5, len(f_ts))})
    fcp = _compute(d)["funding_cost_pressure"].to_numpy()
    engine_out = _engine().calculate_funding_cost_pressure(
        {"candles": d["bars"], "funding_rates": d["funding"]},
        lookback=SPEC["funding_cost_pressure"]["lookback_settlements"]).to_numpy()
    np.testing.assert_array_equal(fcp, engine_out)
    assert np.isfinite(fcp).all()


def test_spec_without_a_fixed_staleness_limit_is_refused():
    d = _synthetic(40, seed=25)
    for change in ({"max_staleness_hours": 0}, {"max_staleness_hours": None}):
        bad = copy.deepcopy(SPEC)
        bad["funding_cost_pressure"].update(change)
        with pytest.raises((ValueError, TypeError)):
            compute_factors(d["bars"], "15m", params=bad)
    old = copy.deepcopy(SPEC)          # the pre-2026-10-11 wording: one settlement interval
    del old["funding_cost_pressure"]["max_staleness_hours"]
    old["funding_cost_pressure"]["max_staleness"] = "one_settlement_interval"
    with pytest.raises(ValueError, match="max_staleness_hours"):
        compute_factors(d["bars"], "15m", params=old)


# ---------------------------------------------------------------- 9. output contract

def test_factor_names_follow_appendix_a():
    assert FACTOR_NAMES == (
        "rsi_14", "macd_hist_norm", "bollinger_pctb", "rolling_vol_ratio", "net_taker_flow",
        "basis_z_score", "funding_cost_pressure", "ou_reversion", "hurst_regime", "triple_screen",
    )


def test_compute_factors_columns_length_and_dtypes():
    d = _synthetic(150, seed=18)
    out = _compute(d, _small_params())
    assert out.columns == ["timestamp", *FACTOR_NAMES]
    assert out.height == d["bars"].height
    assert out["timestamp"].to_list() == d["bars"]["timestamp"].to_list()
    assert out["timestamp"].dtype == pl.Int64
    for name in FACTOR_NAMES:
        assert out[name].dtype == pl.Float64
        assert out[name].null_count() == 0  # undefined is NaN, not null

    bare = compute_factors(d["bars"], "15m", None, None, None, params=_small_params())
    assert bare.columns == out.columns
    for name in ("basis_z_score", "funding_cost_pressure", "triple_screen"):
        assert np.isnan(bare[name].to_numpy()).all()
    for name in ("rsi_14", "ou_reversion", "hurst_regime"):
        np.testing.assert_array_equal(bare[name].to_numpy(), out[name].to_numpy())

    empty = compute_factors(d["bars"][:0], "15m")
    assert empty.columns == out.columns and empty.height == 0


def test_existing_factors_are_the_engine_outputs():
    d = _synthetic(1600, seed=19)
    out = _compute(d)
    data, eng = {"candles": d["bars"]}, _engine()
    np.testing.assert_array_equal(out["rsi_14"].to_numpy(), eng.calculate_rsi(data).to_numpy())
    np.testing.assert_array_equal(out["bollinger_pctb"].to_numpy(),
                                  eng.calculate_bollinger_pctb(data).to_numpy())
    np.testing.assert_array_equal(out["rolling_vol_ratio"].to_numpy(),
                                  eng.calculate_rolling_vol_ratio(data).to_numpy())
    np.testing.assert_array_equal(out["net_taker_flow"].to_numpy(),
                                  eng.calculate_net_taker_flow(data).to_numpy())
    m = SPEC["macd_hist_norm"]
    hist = eng.calculate_macd_hist(data).to_numpy()
    first = m["slow"] + m["signal"] - 1
    norm = out["macd_hist_norm"].to_numpy()
    assert np.isnan(norm[:first]).all()
    np.testing.assert_allclose(norm[first:], hist[first:] / d["bars"]["close"].to_numpy()[first:],
                               rtol=0, atol=0)


def test_compute_factors_rejects_bad_inputs():
    d = _synthetic(40, seed=20)
    with pytest.raises(ValueError):
        compute_factors(d["bars"], "1H")  # 15m starts are not 1H bucket starts
    with pytest.raises(ValueError):
        compute_factors(d["bars"][::-1], "15m")
    with pytest.raises(ValueError):
        compute_factors(d["bars"], "15m", funding=d["funding"].rename({"rate": "fundingRate"}))
    with pytest.raises(ValueError):
        compute_factors(d["bars"], "15m", funding=pl.concat([d["funding"], d["funding"][:1]]))
    bad = copy.deepcopy(SPEC)
    bad["funding_cost_pressure"]["min_settlements"] = 5
    with pytest.raises(ValueError):
        compute_factors(d["bars"], "15m", params=bad)
