"""
The ten factors of the out-of-sample study (§5), on bars from research.bars.

The seven §5.1 factors are the unchanged FactorEngine functions: the
pre-registration defines them as that code, and a test pins it. This module
only feeds them and post-processes their output where §4.2 and §5.1 say so.
The three §5.2 factors are new and live here, never in factor_registry,
because the live model's 17-feature list must not change.

Two conventions matter more than the formulas:

  * **A value at bar t uses only what was known at bar t's close.** The tests
    check it as prefix invariance: the value at t computed on data[:t+1]
    equals the value computed on the whole series. Two engine functions
    decide *whether* to return anything from a whole-series length or
    coverage check, which leaks the future into definedness near the start
    of a series; the causal masks below remove exactly that and nothing else.

  * **Undefined is NaN, never a placeholder.** A missing input series yields
    an all-NaN column, so its cells are untestable (§6.8) instead of being
    fed a constant.

Kernels that test for NaN are compiled without fastmath: under it Numba
assumes NaN cannot occur and folds np.isfinite / x != x to constants.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from loguru import logger
from numba import njit

from research.bars import _as_int_ms, bar_ms, htf_source_index, to_grid
from research.spec import get_spec

# Appendix A order. YAML mappings load in document order, so this is the
# spec's list, not a restatement of it.
FACTOR_NAMES: Tuple[str, ...] = tuple(get_spec()["factors"])

_IMPLEMENTED = {
    "rsi_14", "macd_hist_norm", "bollinger_pctb", "rolling_vol_ratio",
    "net_taker_flow", "basis_z_score", "funding_cost_pressure",
    "ou_reversion", "hurst_regime", "triple_screen",
}
if set(FACTOR_NAMES) != _IMPLEMENTED:
    raise RuntimeError(
        f"Appendix A factors {sorted(FACTOR_NAMES)} differ from those implemented "
        f"{sorted(_IMPLEMENTED)}"
    )

# Choices the engine hard-codes instead of taking as arguments, and the
# variants this module implements. If the spec ever says otherwise, refuse
# to run rather than silently compute something else.
_FIXED = {
    ("macd_hist_norm", "divide_by_close"): True,
    ("bollinger_pctb", "std_ddof"): 0,
    ("bollinger_pctb", "zero_sigma_value"): 0.5,
    ("rolling_vol_ratio", "zero_long_mean_value"): 1.0,
    ("net_taker_flow", "side_data"): "none",
    ("basis_z_score", "std_ddof"): 0,
    ("funding_cost_pressure", "min_settlements"): 10,  # engine: min(10, lookback)
    ("funding_cost_pressure", "std_ddof"): 0,
    ("funding_cost_pressure", "staleness_test"): "ge",
    ("ou_reversion", "gate"): "none",
    ("ou_reversion", "sigma_df"): "W_minus_2",
    ("hurst_regime", "estimator"): "lo_mackinlay_bias_corrected",
    ("triple_screen", "trend"): "sign_of_histogram",
    ("triple_screen", "form"): "continuous_signed",
    ("triple_screen", "htf_bar"): "latest_closed",
}

_HOUR_MS = 3_600_000


def _max_staleness_ms(fp: Dict[str, Any]) -> float:
    """funding_cost_pressure's staleness limit in ms (§4.2: fixed hours, not the series' spacing)."""
    if "max_staleness_hours" not in fp:
        raise ValueError(
            "spec gives funding_cost_pressure no max_staleness_hours; §4.2 fixes the "
            "staleness limit in hours, and this implementation supports nothing else"
        )
    hours = float(fp["max_staleness_hours"])
    if not hours > 0:
        raise ValueError(f"funding_cost_pressure.max_staleness_hours must be positive, got {hours!r}")
    return hours * _HOUR_MS


def _check_params(p: Dict[str, Any]) -> None:
    for (factor, key), want in _FIXED.items():
        got = p.get(factor, {}).get(key, want)
        if got != want:
            raise ValueError(
                f"spec sets {factor}.{key} = {got!r}, but this implementation "
                f"(or the unchanged engine) only supports {want!r}"
            )
    _max_staleness_ms(p["funding_cost_pressure"])  # refuse a spec without the fixed limit


@lru_cache(maxsize=1)
def _engine():
    # The calculate_* methods are stateless; one instance avoids re-logging
    # its start-up line on every call.
    from quant_core.factor_engine import FactorEngine

    return FactorEngine()


def _engine_array(series: Optional[pl.Series], n: int) -> np.ndarray:
    """An engine result as float64, with None (cannot compute) as all-NaN."""
    if series is None:
        return np.full(n, np.nan)
    out = series.to_numpy().astype(np.float64)
    if len(out) != n:
        raise RuntimeError(f"engine returned {len(out)} values for {n} bars")
    return out


def _log_close(close: np.ndarray) -> np.ndarray:
    # A non-positive close becomes -inf/NaN and so makes its windows undefined.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.ascontiguousarray(close, dtype=np.float64))


# ---------------------------------------------------------------- §5.2 kernels

@njit(fastmath=False, cache=False)
def _bad_prefix(x):
    """bad[i] = number of non-finite values in x[:i], for O(1) window checks."""
    n = x.shape[0]
    bad = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        bad[i + 1] = bad[i] + (0 if np.isfinite(x[i]) else 1)
    return bad


@njit(fastmath=False, cache=False)
def _ou_kernel(x, window):
    """
    Rolling AR(1) forecast on log prices x, in residual-sigma units.

    Each window is solved directly, O(W), rather than by rolling sums: the
    centred sums are a small difference of large uncentred ones, and rolling
    updates never return to exactly zero on a flat window, so the
    "undefined if ... = 0" tests would silently stop firing. For the same
    reason the lagged series z = x[t-W .. t-1] and the current series
    y = x[t-W+1 .. t] are each centred on their own first value: whichever
    of them is constant then becomes exact zeros. With one shared
    reference, a jump followed by W flat bars leaves y constant but centred
    on the pre-jump price; its mean is then off by an ulp, the residuals
    come out around 1e-30 instead of 0, and sigma_e = 0 returns noise of
    order one instead of NaN.
    """
    n = x.shape[0]
    out = np.full(n, np.nan)
    w = window
    bad = _bad_prefix(x)
    for t in range(w, n):
        # Pairs (x[s-1], x[s]) for s = t-w+1 .. t use x[t-w .. t].
        if bad[t + 1] - bad[t - w] != 0:
            continue
        rz = x[t - w]       # reference of the lagged series z
        ry = x[t - w + 1]   # reference of the current series y
        zbar = 0.0
        ybar = 0.0
        for j in range(w):
            zbar += x[t - w + j] - rz
            ybar += x[t - w + 1 + j] - ry
        zbar /= w
        ybar /= w
        szz = 0.0
        szy = 0.0
        for j in range(w):
            dz = (x[t - w + j] - rz) - zbar
            dy = (x[t - w + 1 + j] - ry) - ybar
            szz += dz * dz
            szy += dz * dy
        if szz == 0.0:
            continue
        phi = szy / szz
        a = ybar - phi * zbar  # intercept of (y - ry) on (z - rz)
        rss = 0.0
        for j in range(w):
            e = (x[t - w + 1 + j] - ry) - a - phi * (x[t - w + j] - rz)
            rss += e * e
        sigma = math.sqrt(rss / (w - 2))
        if sigma == 0.0:
            continue
        # The original intercept is a + ry - phi*rz, so a_orig + (phi-1)*x_t
        # equals a - (x_t - ry) + phi*(x_t - rz), computed from small numbers.
        out[t] = (a - (x[t] - ry) + phi * (x[t] - rz)) / sigma
    return out


@njit(fastmath=False, cache=False)
def _hurst_kernel(x, window, q):
    """
    Lo-MacKinlay bias-corrected variance ratio over the last `window` returns.

    Returns (H_hat, m) per bar. q-period sums are taken as x[k] - x[k-q],
    which is the sum of the q returns ending at k. Solved per window for the
    same exact-zero reason as _ou_kernel.
    """
    n = x.shape[0]
    h_hat = np.full(n, np.nan)
    mom = np.full(n, np.nan)
    w = window
    big_m = q * (w - q + 1) * (1.0 - q / w)
    two_ln_q = 2.0 * math.log(q)
    bad = _bad_prefix(x)
    for t in range(w, n):
        # Returns r[t-w+1 .. t] use x[t-w .. t].
        if bad[t + 1] - bad[t - w] != 0:
            continue
        rbar = (x[t] - x[t - w]) / w
        var_a = 0.0
        for i in range(t - w + 1, t + 1):
            d = (x[i] - x[i - 1]) - rbar
            var_a += d * d
        var_a /= w - 1
        if var_a == 0.0:
            continue
        var_c = 0.0
        qr = q * rbar
        for k in range(t - w + q, t + 1):
            d = (x[k] - x[k - q]) - qr
            var_c += d * d
        var_c /= big_m
        vr = var_c / var_a
        if vr <= 0.0:
            continue
        h_hat[t] = 0.5 + math.log(vr) / two_ln_q
        mom[t] = (x[t] - x[t - q]) / (math.sqrt(var_a) * math.sqrt(q))
    return h_hat, mom


# ---------------------------------------------------------------- §5.2 factors

def ou_reversion(close: np.ndarray, window: int) -> np.ndarray:
    """
    §5.2 ou_reversion: (a + (phi-1)*x_t) / sigma_e from an OLS AR(1) of x = ln C
    over the last `window` pairs, sigma_e with W-2 degrees of freedom.

    NaN until `window` pairs exist (first value at index `window`), and where
    sigma_e = 0, the lagged x has zero variance, or the window holds a
    non-finite (or non-positive) close.
    """
    window = int(window)
    if window < 3:
        raise ValueError("ou_reversion needs window >= 3 (sigma_e uses W-2 df)")
    return _ou_kernel(_log_close(close), window)


def hurst_exponent(close: np.ndarray, window: int, q: int) -> np.ndarray:
    """H_hat = 1/2 + ln VR / (2 ln q) over the last `window` returns (§5.2); NaN where undefined."""
    window, q = int(window), int(q)
    if q < 2 or window <= q:
        raise ValueError("hurst needs q >= 2 and window > q")
    return _hurst_kernel(_log_close(close), window, q)[0]


def hurst_regime(close: np.ndarray, window: int, q: int) -> np.ndarray:
    """
    §5.2 hurst_regime: (H_hat - 1/2) * m_t, m_t = (x_t - x_{t-q}) / (sigma_a sqrt q).

    NaN until `window` returns exist (first value at index `window`), and
    where VR <= 0, sigma_a = 0, or the window holds a non-finite close.
    """
    window, q = int(window), int(q)
    if q < 2 or window <= q:
        raise ValueError("hurst needs q >= 2 and window > q")
    h_hat, mom = _hurst_kernel(_log_close(close), window, q)
    return (h_hat - 0.5) * mom


def stoch_k(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """
    %K_t = 100 (C_t - min Lo) / (max Hi - min Lo) over bars t-n+1 .. t (§5.2).

    NaN for the first n-1 bars, where the range is zero, and where the window
    holds a non-finite value (min/max propagate NaN).
    """
    n = int(n)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    out = np.full(len(close), np.nan)
    if n < 1 or len(close) < n:
        return out
    view = np.lib.stride_tricks.sliding_window_view
    hh = view(high, n).max(axis=1)
    ll = view(low, n).min(axis=1)
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 100.0 * (close[n - 1:] - ll) / rng
    k[~(rng != 0.0)] = np.nan
    out[n - 1:] = k
    return out


def triple_screen(
    bars: pl.DataFrame,
    htf_bars: Optional[pl.DataFrame],
    bar: str,
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    §5.2 triple_screen: tr * max(0, tr * (50 - %K)) / 50.

    tr is the sign of the unchanged calculate_macd_hist on HTF closes, taken
    from the latest HTF bar already closed when the analysis bar closes
    (htf_source_index). NaN while that histogram is undefined, before any HTF
    bar has closed, or where %K is undefined. htf_bars must be the 4 x B bars
    of the same instrument, built by resample_complete.
    """
    p = params if params is not None else get_spec()["factors"]["triple_screen"]
    n = bars.height
    if htf_bars is None or htf_bars.height == 0 or n == 0:
        return np.full(n, np.nan)

    fast, slow, signal = (int(v) for v in p["macd"])
    period = bar_ms(bar)
    htf_period = period * int(p["htf_multiple"])
    htf_ts = _as_int_ms(htf_bars["timestamp"].to_numpy())
    if np.any(np.diff(htf_ts) <= 0) or np.any(htf_ts % htf_period != 0):
        raise ValueError(f"htf_bars are not sorted {htf_period} ms buckets for bar {bar!r}")

    hist = _engine_array(
        _engine().calculate_macd_hist({"candles": htf_bars}, fast, slow, signal), len(htf_ts)
    )
    # The engine returns nothing below slow+signal bars, although its first
    # value sits at index slow+signal-2; that one value is look-ahead in
    # definedness, so drop it (module docstring).
    hist[: slow + signal - 1] = np.nan
    trend = np.sign(hist)

    idx = htf_source_index(_as_int_ms(bars["timestamp"].to_numpy()), period, htf_ts, htf_period)
    tr = np.where(idx >= 0, trend[np.maximum(idx, 0)], np.nan)

    k = stoch_k(
        bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy(), int(p["stoch_k"])
    )
    centre = float(p["k_center"])
    # + 0.0 turns the -0.0 of a downtrend without a rally into a plain 0.
    return tr * np.maximum(0.0, tr * (centre - k)) / centre + 0.0


# ---------------------------------------------------------------- §4.2 funding

def irregular_settlements(settle_ts: np.ndarray, label: str = "funding") -> List[Tuple[int, int]]:
    """
    (settlement time, spacing since the previous settlement) in ms for every
    spacing other than Appendix A's expected_spacing_hours (8 h), logged.

    §4.2 requires any other spacing to be reported, and nothing more: the
    staleness limit stays the fixed 8 h (funding_fresh). Deriving the limit
    from the series, as a median over the whole set once did, would let
    settlements after a bar decide whether that bar is masked.
    """
    expected = int(get_spec()["data"]["funding"]["expected_spacing_hours"]) * _HOUR_MS
    ts = np.unique(_as_int_ms(settle_ts))
    d = np.diff(ts)
    idx = np.flatnonzero(d != expected)
    out = [(int(ts[i + 1]), int(d[i])) for i in idx]
    if out:
        hours = sorted({sp / _HOUR_MS for _, sp in out})
        logger.warning(
            f"{label}: {len(out)} settlement spacing(s) other than {expected / _HOUR_MS:g} h "
            f"(seen: {', '.join(f'{x:g}' for x in hours)} h; the first ends at {out[0][0]} ms); "
            f"the staleness limit stays {expected / _HOUR_MS:g} h (§4.2)"
        )
    return out


def funding_fresh(bar_ts: np.ndarray, settle_ts: np.ndarray, max_staleness_ms: float) -> np.ndarray:
    """
    True where a bar's start is less than max_staleness_ms after the latest
    settlement at or before it (§4.2); False where it is max_staleness_ms or
    more after it, or no settlement precedes it.

    The limit is a constant (Appendix A's max_staleness_hours, 8 h), so a
    bar's flag depends only on settlements at or before its start: a prefix
    of the data gives the same mask. Without the mask, the as-of join would
    keep showing the last known z-score across a missing settlement as if it
    were current.
    """
    bar_ts = _as_int_ms(bar_ts)
    settle_ts = np.sort(_as_int_ms(settle_ts))
    idx = np.searchsorted(settle_ts, bar_ts, side="right") - 1
    fresh = idx >= 0
    last = settle_ts[np.maximum(idx, 0)] if len(settle_ts) else np.zeros(len(bar_ts), np.int64)
    fresh &= (bar_ts - last) < max_staleness_ms
    return fresh


# ---------------------------------------------------------------- all factors

def _require(df: pl.DataFrame, cols: Tuple[str, ...], what: str) -> None:
    # The engine swallows exceptions and returns None, so a missing column
    # would otherwise surface as a silently all-NaN factor.
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} lacks column(s) {missing}")


def _basis(candles: pl.DataFrame, close: np.ndarray, ts: np.ndarray,
           spot_bars: Optional[pl.DataFrame], lookback: int) -> np.ndarray:
    n = len(ts)
    if spot_bars is None or spot_bars.height == 0:
        return np.full(n, np.nan)
    _require(spot_bars, ("timestamp", "close"), "spot_bars")

    # On the perp grid, a dropped spot bucket is a NaN close at that very
    # timestamp, so the engine's as-of join picks the NaN instead of carrying
    # the previous close forward; the engine then leaves that bar NaN and
    # ignores it inside later windows.
    spot = to_grid(spot_bars.select(["timestamp", "close"]), ts)
    z = _engine_array(
        _engine().calculate_basis_z_score({"candles": candles, "spot_candles": spot}, lookback=lookback),
        n,
    )
    # The engine returns nothing unless the *whole* series has >= lookback
    # finite basis values. Requiring that count up to bar t instead keeps
    # every value it emits and removes the dependence on later spot coverage.
    spot_close = spot["close"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        basis = np.where(spot_close > 0, (close - spot_close) / spot_close, np.nan)
    z[np.cumsum(np.isfinite(basis)) < lookback] = np.nan
    return z


def _funding(candles: pl.DataFrame, ts: np.ndarray, funding: Optional[pl.DataFrame],
             lookback: int, max_staleness_ms: float) -> np.ndarray:
    n = len(ts)
    if funding is None or funding.height == 0:
        return np.full(n, np.nan)
    _require(funding, ("timestamp", "rate"), "funding")

    f_ts = _as_int_ms(funding["timestamp"].to_numpy())
    order = np.argsort(f_ts, kind="stable")
    f_ts = f_ts[order]
    if np.any(np.diff(f_ts) == 0):
        raise ValueError("duplicate settlement times; de-duplicate on fundingTime first (§4.2)")
    rates = pl.DataFrame({
        "timestamp": pl.Series(f_ts, dtype=pl.Int64),
        "rate": pl.Series(funding["rate"].cast(pl.Float64).to_numpy()[order], dtype=pl.Float64),
    })
    z = _engine_array(
        _engine().calculate_funding_cost_pressure(
            {"candles": candles, "funding_rates": rates}, lookback=lookback
        ),
        n,
    )
    z[~funding_fresh(ts, f_ts, max_staleness_ms)] = np.nan
    return z


def compute_factors(
    bars: pl.DataFrame,
    bar: str,
    spot_bars: Optional[pl.DataFrame] = None,
    funding: Optional[pl.DataFrame] = None,
    htf_bars: Optional[pl.DataFrame] = None,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """
    All ten factors on one bar sequence: timestamp + FACTOR_NAMES, float, NaN
    where undefined or where an input series is unavailable.

    bars/spot_bars/htf_bars come from resample_complete (htf_bars at 4 x bar);
    funding has columns 'timestamp' (settlement time, ms) and 'rate'.
    Parameters default to Appendix A; `params` overrides them (tests use
    short windows). The funding staleness limit is params'
    funding_cost_pressure.max_staleness_hours (Appendix A: a fixed 8 h,
    §4.2), never a statistic of the funding series.
    """
    p = params if params is not None else get_spec()["factors"]
    _check_params(p)
    _require(bars, ("timestamp", "high", "low", "close", "volume"), "bars")

    n = bars.height
    ts = _as_int_ms(bars["timestamp"].to_numpy()) if n else np.empty(0, dtype=np.int64)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("bars must be sorted with unique timestamps")
    if np.any(ts % bar_ms(bar) != 0):
        raise ValueError(f"bar timestamps are not {bar} bucket starts")

    candles = bars.with_columns(pl.Series("timestamp", ts, dtype=pl.Int64))
    close = candles["close"].cast(pl.Float64).to_numpy()
    data = {"candles": candles}
    eng = _engine()
    cols: Dict[str, np.ndarray] = {}

    if n:
        cols["rsi_14"] = _engine_array(eng.calculate_rsi(data, period=int(p["rsi_14"]["period"])), n)

        m = p["macd_hist_norm"]
        fast, slow, signal = int(m["fast"]), int(m["slow"]), int(m["signal"])
        hist = _engine_array(eng.calculate_macd_hist(data, fast, slow, signal), n)
        hist[: slow + signal - 1] = np.nan  # causal definedness, as in triple_screen
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = hist / close
        norm[~np.isfinite(norm)] = np.nan
        cols["macd_hist_norm"] = norm

        b = p["bollinger_pctb"]
        cols["bollinger_pctb"] = _engine_array(
            eng.calculate_bollinger_pctb(data, period=int(b["period"]), std_dev=float(b["n_std"])), n
        )
        v = p["rolling_vol_ratio"]
        cols["rolling_vol_ratio"] = _engine_array(
            eng.calculate_rolling_vol_ratio(data, short_window=int(v["short"]), long_window=int(v["long"])), n
        )
        # No 'trades' key: the CLV (Chaikin Money Flow) branch is the registered definition.
        cols["net_taker_flow"] = _engine_array(
            eng.calculate_net_taker_flow(data, lookback=int(p["net_taker_flow"]["lookback"])), n
        )
        cols["basis_z_score"] = _basis(
            candles, close, ts, spot_bars, int(p["basis_z_score"]["lookback"])
        )
        fp = p["funding_cost_pressure"]
        cols["funding_cost_pressure"] = _funding(
            candles, ts, funding, int(fp["lookback_settlements"]),
            _max_staleness_ms(fp),
        )
        cols["ou_reversion"] = ou_reversion(close, int(p["ou_reversion"]["window"]))
        cols["hurst_regime"] = hurst_regime(
            close, int(p["hurst_regime"]["window"]), int(p["hurst_regime"]["q"])
        )
        cols["triple_screen"] = triple_screen(candles, htf_bars, bar, p["triple_screen"])
    else:
        cols = {name: np.empty(0) for name in FACTOR_NAMES}

    return pl.DataFrame(
        [pl.Series("timestamp", ts, dtype=pl.Int64)]
        + [pl.Series(name, cols[name], dtype=pl.Float64) for name in FACTOR_NAMES]
    )
