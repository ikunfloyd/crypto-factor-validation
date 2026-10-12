"""
Statistical core of the out-of-sample study: §6-§9 of docs/preregistration.md
and the robustness rows of §8.

Everything is plain numpy on arrays, so the same functions serve the IS rerun,
the test sets and the synthetic tests that check their size and power.

Conventions every function shares:

  * Bars are rows sorted by ``ts``, the bucket start in int milliseconds.
    Dropped buckets (§6.1) are simply absent, so rows are not a time grid.
  * A set's evaluation window is ``[start_ms, end_ms)``. A bar's grid position
    is ``(ts - start_ms) // bar_ms`` and the grid has
    ``N = (end_ms - start_ms) // bar_ms`` positions. Anything that depends on
    time adjacency (the HAC sum, the placebo rotation, the turnover P&L) runs
    on that grid, never on row indices: across a dropped bucket, row t+1 is
    not bar t+1.
  * Returns are fractions; only keys ending in ``_bps`` are basis points.
"""
from __future__ import annotations

import bisect
import importlib.util
import math
from functools import lru_cache
from statistics import NormalDist
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.spec import MINUTE_MS, REPO_ROOT, get_spec

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
_N01 = NormalDist()

__all__ = [
    "norm_cdf", "norm_ppf",
    "forward_returns", "evaluable", "expanding_percentiles", "expanding_edges",
    "assign_buckets", "bucket_shares", "direction", "per_trade_net",
    "hh_omega", "cell_statistic", "pooled_statistic",
    "holm", "bh", "by", "bounds", "mde", "scale_se", "n_evaluable",
    "placebo_offsets", "circular_placebo_t", "placebo_p", "family_placebo_p",
    "full_window_ic", "daily_ic", "v1_t_sqrt_h",
    "by_quarter", "volatile_days", "leave_out_volatile_days", "turnover_aware",
    "cost_sensitivity", "winsorized_fwd", "funding_adjusted_net", "quintile_means",
    "mean_run_length",
]


# --------------------------------------------------------------------------
# Normal distribution
# --------------------------------------------------------------------------

def norm_cdf(x):
    """Φ(x), scalar or elementwise. NaN propagates."""
    if np.ndim(x) == 0:
        return _N01.cdf(float(x))
    arr = np.asarray(x, dtype=np.float64)
    return np.array([_N01.cdf(v) for v in arr.ravel()]).reshape(arr.shape)


def _ppf1(p: float) -> float:
    if not p == p:
        return math.nan
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf
    if not 0.0 < p < 1.0:
        return math.nan
    return _N01.inv_cdf(p)


def norm_ppf(p):
    """Φ⁻¹(p), scalar or elementwise; ±inf at 0 and 1, NaN outside [0, 1]."""
    if np.ndim(p) == 0:
        return _ppf1(float(p))
    arr = np.asarray(p, dtype=np.float64)
    return np.array([_ppf1(v) for v in arr.ravel()]).reshape(arr.shape)


def _p_upper(t: float) -> float:
    """
    One-sided p = 1 - Φ(t) (§6.7); NaN when t is undefined.

    Computed as erfc(t/√2)/2, not as 1 - Φ(t): the subtraction cancels, so
    from t ≈ 7 the relative error grows and from t ≈ 8.3 p comes out exactly
    0, which would print a strongly significant cell's p (and its Holm and
    BH/BY values) as 0.0. erfc keeps full relative precision in the tail.
    """
    return math.nan if not t == t else 0.5 * math.erfc(t / math.sqrt(2.0))


def _p_two_sided(t: float) -> float:
    """Two-sided p = 2·(1 - Φ(|t|)) = erfc(|t|/√2), for the same reason as _p_upper."""
    return math.nan if not t == t else math.erfc(abs(t) / math.sqrt(2.0))


# --------------------------------------------------------------------------
# Bars, grid and forward returns (§4.1, §6.3)
# --------------------------------------------------------------------------

def _check_ts(ts, bar_ms: int) -> np.ndarray:
    """
    ts as int64, refusing anything that is not one row per bucket.

    Duplicate or unsorted rows, or steps that are not whole bars, would make
    the "consecutive bars" and grid-position arithmetic below silently wrong.
    """
    ts = np.asarray(ts)
    if ts.ndim != 1:
        raise ValueError("ts must be one-dimensional")
    if ts.dtype.kind == "f":
        if not np.all(np.isfinite(ts)) or np.any(ts != np.round(ts)):
            raise ValueError("ts must hold whole milliseconds")
    ts = ts.astype(np.int64)
    d = np.diff(ts)
    if np.any(d <= 0):
        raise ValueError("ts must be strictly increasing (one row per bucket)")
    if np.any(d % bar_ms):
        raise ValueError("every ts step must be a whole number of bars")
    return ts


def _grid(ts, start_ms: int, end_ms: int, bar_ms: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Grid positions of the bars, the in-window mask, and the grid length N."""
    ts = _check_ts(ts, bar_ms)
    if (end_ms - start_ms) % bar_ms:
        raise ValueError("window length must be a whole number of bars")
    n_grid = (end_ms - start_ms) // bar_ms
    inwin = (ts >= start_ms) & (ts < end_ms)
    off = ts - start_ms
    if np.any(off[inwin] % bar_ms):
        raise ValueError("bars are not aligned to the window's bar grid")
    return off // bar_ms, inwin, int(n_grid)


def forward_returns(close, ts, bar_ms: int, h: int) -> np.ndarray:
    """
    fwd_t = close[t+1+h] / close[t+1] - 1 (§6.3): enter one bar after the signal.

    NaN unless bars t .. t+1+h are consecutive. With strictly increasing,
    whole-bar steps, ts[t+1+h] - ts[t] == (h+1)·B holds only if every one of
    the h+1 steps is exactly B, so a single subtraction checks the whole span.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    close = np.asarray(close, dtype=np.float64)
    ts = _check_ts(ts, bar_ms)
    if close.shape != ts.shape:
        raise ValueError("close and ts must have the same length")
    n = len(close)
    out = np.full(n, np.nan)
    k = n - (h + 1)
    if k <= 0:
        return out
    entry, exit_ = close[1:1 + k], close[h + 1:h + 1 + k]
    ok = (
        (ts[h + 1:h + 1 + k] - ts[:k] == (h + 1) * bar_ms)
        & np.isfinite(entry) & np.isfinite(exit_) & (entry > 0) & (exit_ > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:k] = np.where(ok, exit_ / entry - 1.0, np.nan)
    return out


def evaluable(ts, start_ms: int, end_ms: int, bar_ms: int, h: int) -> np.ndarray:
    """
    Bars the window may evaluate (§4.1): start in [start, end) and
    start(t+1+h) + B <= end.

    Written as ts + (h+2)·B <= end, which equals the §4.1 condition whenever
    fwd_t is defined (its bars are then consecutive). A bar with an undefined
    fwd_t is never active anyway.
    """
    ts = np.asarray(ts).astype(np.int64)
    return (ts >= start_ms) & (ts < end_ms) & (ts + (h + 2) * bar_ms <= end_ms)


def n_evaluable(start_ms: int, end_ms: int, bar_ms: int, h: int) -> int:
    """N_set = (end - start)/B - (h + 1): evaluable bars with no gaps (§9)."""
    return int((end_ms - start_ms) // bar_ms - (h + 1))


# --------------------------------------------------------------------------
# Buckets and direction (§6.4, §6.5)
# --------------------------------------------------------------------------

def expanding_percentiles(f, pcts: Sequence[float], min_prior: int = 500) -> np.ndarray:
    """
    For every row t, numpy-'linear' percentiles of the finite f at rows < t.

    Returns shape (len(f), len(pcts)); a row is NaN while fewer than
    ``min_prior`` finite values precede it. Bit-identical to
    ``np.percentile(prefix, pcts, method='linear')``: the virtual index and
    the two-sided lerp below are numpy 1.24's own (function_base._quantile and
    _lerp), so an edge never differs from the documented definition by an ulp
    and a value equal to an edge stays equal to it.

    The prefix is kept as a sorted list (bisect.insort). That is O(n²) in
    memmove but ~0.2 s at 50k rows, far simpler than an order-statistic tree.
    """
    f = np.asarray(f, dtype=np.float64)
    n = len(f)
    q = np.true_divide(np.asarray(pcts, dtype=np.float64), 100)
    k = len(q)
    out = np.full((n, k), np.nan)
    if n == 0 or k == 0:
        return out

    fin = np.isfinite(f)
    m = np.concatenate(([0], np.cumsum(fin)[:-1])).astype(np.int64)  # values before t
    valid = m >= max(1, min_prior)

    mf = m.astype(np.float64)
    vi = (mf[:, None] - 1.0) * q[None, :]
    lo = np.floor(vi)
    above = vi >= (mf[:, None] - 1.0)
    gamma = vi - lo
    lo_i = np.where(above, m[:, None] - 1, lo).astype(np.int64)
    hi_i = np.where(above, m[:, None] - 1, lo + 1).astype(np.int64)
    lo_i[~valid] = 0
    hi_i[~valid] = 0

    lo_l, hi_l = lo_i.tolist(), hi_i.tolist()
    a_rows: List[Optional[List[float]]] = [None] * n
    b_rows: List[Optional[List[float]]] = [None] * n
    sl: List[float] = []
    fl, finl, validl = f.tolist(), fin.tolist(), valid.tolist()
    for t in range(n):
        if validl[t]:
            li, hj = lo_l[t], hi_l[t]
            a_rows[t] = [sl[i] for i in li]
            b_rows[t] = [sl[j] for j in hj]
        if finl[t]:
            bisect.insort(sl, fl[t])

    idx = np.flatnonzero(valid)
    if len(idx):
        a = np.array([a_rows[t] for t in idx], dtype=np.float64)
        b = np.array([b_rows[t] for t in idx], dtype=np.float64)
        # numpy's _lerp: a + (b-a)·g, but b - (b-a)·(1-g) when g >= 0.5.
        diff = b - a
        g = gamma[idx]
        out[idx] = np.where(g >= 0.5, b - diff * (1 - g), a + diff * g)
    return out


def expanding_edges(f, lo_pct: float = 20, hi_pct: float = 80,
                    min_prior: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """(P_lo, P_hi) per row from strictly earlier finite values (§6.4)."""
    e = expanding_percentiles(f, [lo_pct, hi_pct], min_prior)
    return e[:, 0].copy(), e[:, 1].copy()


def assign_buckets(f, p_lo, p_hi) -> np.ndarray:
    """
    s = +1 if f > P_hi, -1 if f < P_lo, else 0 (§6.4), as int8.

    Strict on both sides, so a point mass sitting on an edge (triple_screen
    at 0) stays out of both traded buckets instead of flooding one of them.
    """
    f = np.asarray(f, dtype=np.float64)
    p_lo = np.asarray(p_lo, dtype=np.float64)
    p_hi = np.asarray(p_hi, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(p_lo) & np.isfinite(p_hi)
    s = np.zeros(f.shape, dtype=np.int8)
    s[ok & (f > p_hi)] = 1
    s[ok & (f < p_lo)] = -1
    return s


def bucket_shares(s, fwd, ts, start_ms: int, end_ms: int, bar_ms: int,
                  h: int) -> Dict[str, Any]:
    """
    n_eval and the realised share of each traded bucket (§6.4, §10 columns).

    n_eval counts evaluable bars whose fwd is defined; the shares are taken
    over those bars, so they show how far a point mass pushed a bucket away
    from the nominal 20%.
    """
    s = np.asarray(s)
    m = evaluable(ts, start_ms, end_ms, bar_ms, h) & np.isfinite(np.asarray(fwd, dtype=np.float64))
    n_eval = int(m.sum())
    if n_eval == 0:
        return {"n_eval": 0, "share_top": math.nan, "share_bottom": math.nan}
    return {
        "n_eval": n_eval,
        "share_top": float((s[m] == 1).sum() / n_eval),
        "share_bottom": float((s[m] == -1).sum() / n_eval),
    }


def direction(s, fwd, active_mask, min_active: int = 300,
              missing_input: bool = False) -> Tuple[int, bool, Optional[str], float]:
    """
    In-sample direction d (§6.5) -> (d, flagged, flag_reason, mean_s_fwd).

    ``active_mask`` marks the IS evaluation bars; it is intersected with
    s != 0 and finite fwd, so passing the evaluable mask or the active set
    itself gives the same answer. d is the sign of mean(s·fwd), +1 on an exact
    zero. A cell with fewer than ``min_active`` active bars, or one whose input
    series is missing (the caller says so), gets d = +1 and a flag: a sign
    estimated on a few dozen trades would be a coin flip dressed as a prior.
    mean_s_fwd is a fraction (NaN with no active bars).
    """
    if missing_input:
        return 1, True, "missing_input", math.nan
    s = np.asarray(s)
    fwd = np.asarray(fwd, dtype=np.float64)
    act = np.asarray(active_mask, dtype=bool) & (s != 0) & np.isfinite(fwd)
    n_act = int(act.sum())
    mean = float(np.mean(s[act] * fwd[act])) if n_act else math.nan
    if n_act < min_active:
        return 1, True, "few_active", mean
    return (1 if mean >= 0 else -1), False, None, mean


def _cost_frac(cost_bps: float) -> float:
    return float(cost_bps) / 10_000.0


def per_trade_net(s, fwd, d: int, cost_bps: float, mask=None) -> np.ndarray:
    """
    n_t = d·s_t·fwd_t - c per bar (§6.6); NaN where no trade.

    Every active bar is one round trip paying the full c: long the top and
    short the bottom are two separate trades, which is the undercharge v1
    made (§2, item 1). ``mask`` optionally restricts trades further.
    """
    if d not in (1, -1):
        raise ValueError("d must be +1 or -1")
    s = np.asarray(s)
    fwd = np.asarray(fwd, dtype=np.float64)
    act = (s != 0) & np.isfinite(fwd)
    if mask is not None:
        act &= np.asarray(mask, dtype=bool)
    with np.errstate(invalid="ignore"):
        n = d * s.astype(np.float64) * fwd - _cost_frac(cost_bps)
    return np.where(act, n, np.nan)


# --------------------------------------------------------------------------
# Hansen-Hodrick standard error (§6.7) and the two statistics (§6.7, §7)
# --------------------------------------------------------------------------

def _weighted_autocov_sum(z: np.ndarray, lag: int, bartlett: bool) -> float:
    om = float(np.dot(z, z))
    n = len(z)
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0) if bartlett else 1.0
        om += 2.0 * w * float(np.dot(z[:-k], z[k:]))
    return om


def hh_omega(z, h: int) -> Tuple[float, str]:
    """
    Ω = Σ_{|k|<=ℓ} Σ_t z_t z_{t+k} with ℓ = h-1 and uniform weights (§6.7).

    ℓ = h-1 is exactly the overlap of h-bar forward windows, which is why the
    uniform (Hansen-Hodrick) kernel and not Newey-West is the default. Uniform
    weights can make Ω negative; then Bartlett weights 1 - |k|/(ℓ+1) with
    ℓ = 2h, which cannot, and the method says so. ``z`` must already sit on
    the regular bar grid, zeros where a bar is inactive or dropped.
    """
    z = np.asarray(z, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise ValueError("z must be finite (zeros where inactive)")
    om = _weighted_autocov_sum(z, h - 1, bartlett=False)
    if om > 0:
        return om, "hh"
    return _weighted_autocov_sum(z, 2 * h, bartlett=True), "bartlett_fallback"


def _nan_stat() -> Dict[str, Any]:
    return {"mu": math.nan, "se": math.nan, "t": math.nan, "p": math.nan,
            "n_active": 0, "se_method": None}


def _stat_on_grid(n_grid: np.ndarray, active: np.ndarray, h: int) -> Dict[str, Any]:
    """
    μ̂, SE, t and one-sided p from per-bar net returns already on the grid.

    Shared by cell_statistic and the placebo so that a zero rotation
    reproduces the observed t bit for bit.
    """
    cnt = int(active.sum())
    if cnt == 0:
        return _nan_stat()
    vals = n_grid[active]
    mu = float(vals.mean())
    z = np.zeros(len(n_grid))
    z[active] = vals - mu
    omega, method = hh_omega(z, h)
    se = math.sqrt(omega) / cnt if omega > 0 else 0.0
    # SE = 0 only if every trade returned exactly μ̂: t is then undefined
    # and p is 1 (§6.7).
    t = mu / se if se > 0 else math.nan
    return {"mu": mu, "se": se, "t": t, "p": _p_upper(t) if se > 0 else 1.0,
            "n_active": cnt, "se_method": method}


def _on_grid(n, ts, start_ms, end_ms, bar_ms, h) -> Tuple[np.ndarray, np.ndarray]:
    """Place the window's active per-bar values on the grid."""
    n = np.asarray(n, dtype=np.float64)
    pos, inwin, n_grid = _grid(ts, start_ms, end_ms, bar_ms)
    act = np.isfinite(n) & evaluable(ts, start_ms, end_ms, bar_ms, h)
    vals = np.full(n_grid, np.nan)
    active = np.zeros(n_grid, dtype=bool)
    vals[pos[act]] = n[act]
    active[pos[act]] = True
    return vals, active


def cell_statistic(n, ts, start_ms: int, end_ms: int, bar_ms: int, h: int) -> Dict[str, Any]:
    """
    μ̂, Hansen-Hodrick SE = √Ω/|A|, t_HH and one-sided p (§6.6, §6.7).

    ``n`` is the per-bar net return, NaN where inactive. Values outside the
    evaluable bars of [start, end) are ignored rather than trusted, so a
    warm-up trade or one whose exit falls past the window cannot leak in.
    Returns NaNs (n_active 0) with no active bar. Whether |A| reaches the
    300 needed to be testable (§6.8) is the caller's decision.
    """
    vals, active = _on_grid(n, ts, start_ms, end_ms, bar_ms, h)
    return _stat_on_grid(vals, active, h)


def pooled_statistic(panel: Sequence[Tuple[Any, Any]], start_ms: int, end_ms: int,
                     bar_ms: int, h: int) -> Dict[str, Any]:
    """
    F_majors pooled cell (§7): μ̂_pool = mean of per-instrument μ̂_i.

    ``panel`` is a list of (n, ts), one per instrument, on a common window.
    SE_pool = √Ω_w with w_t = (1/k) Σ_i z_{i,t}/|A_i| on the shared grid. w
    already carries each 1/|A_i|, since μ̂_pool - E μ̂_pool = Σ_t w_t, so Ω_w
    is not divided by any |A| again; summing the instruments before squaring
    is what keeps their same-time correlation in the variance. NaNs if any
    instrument has no active bar.
    """
    k = len(panel)
    if k == 0:
        raise ValueError("empty panel")
    n_grid = (end_ms - start_ms) // bar_ms
    w = np.zeros(n_grid)
    mus: List[float] = []
    counts: List[int] = []
    for n, ts in panel:
        vals, active = _on_grid(n, ts, start_ms, end_ms, bar_ms, h)
        cnt = int(active.sum())
        counts.append(cnt)
        if cnt == 0:
            continue
        mu_i = float(vals[active].mean())
        mus.append(mu_i)
        w[active] += (vals[active] - mu_i) / cnt
    if any(c == 0 for c in counts):
        out = _nan_stat()
        out.pop("n_active")
        out["n_active_each"] = counts
        return out
    w /= k
    omega, method = hh_omega(w, h)
    se = math.sqrt(omega) if omega > 0 else 0.0
    mu = float(np.mean(mus))
    t = mu / se if se > 0 else math.nan  # SE = 0: t undefined, p = 1 (§6.7)
    return {"mu": mu, "se": se, "t": t, "p": _p_upper(t) if se > 0 else 1.0,
            "n_active_each": counts, "se_method": method}


# --------------------------------------------------------------------------
# Multiple testing (§7)
# --------------------------------------------------------------------------

def _check_p(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("p must be one-dimensional")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        # Untestable cells enter the family with p = 1 (§6.8), never NaN.
        raise ValueError("p-values must be finite and in [0, 1]")
    return p


def holm(p) -> np.ndarray:
    """
    Holm step-down adjusted p, in input order, capped at 1.

    p̃_(j) = max_{i<=j} min(1, (m-i+1)·p_(i)); reject where p̃ <= α. The
    running max makes the adjusted values monotone in the sorted order, which
    is what "step down and stop at the first non-rejection" means.
    """
    p = _check_p(p)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    adj = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty(m)
    out[order] = adj
    return out


def _step_up(p: np.ndarray, scale: float) -> np.ndarray:
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    q = scale * m * p[order] / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg q-values: min_{j>=i} m·p_(j)/j, capped at 1."""
    return _step_up(_check_p(p), 1.0)


def by(p) -> np.ndarray:
    """Benjamini-Yekutieli q-values: BH × Σ_{k<=m} 1/k, capped at 1."""
    p = _check_p(p)
    c_m = float(np.sum(1.0 / np.arange(1, len(p) + 1))) if len(p) else 1.0
    return _step_up(p, c_m)


# --------------------------------------------------------------------------
# Bounds and power (§9)
# --------------------------------------------------------------------------

def bounds(mu, se, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    ub95 = μ̂ + z·SE (one cell), ub95_family = μ̂ + 3.144·SE (all 60 at once),
    and the two-sided doge_fwd interval μ̂ ∓ 1.960·SE; z from Appendix A.
    """
    iv = (spec or get_spec())["intervals"]
    mu = np.asarray(mu, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    zc = float(iv["doge_fwd_ci"]["z"])
    out = {
        "ub95": mu + float(iv["ub95"]["z"]) * se,
        "ub95_family": mu + float(iv["ub95_family"]["z"]) * se,
        "ci95_lo": mu - zc * se,
        "ci95_hi": mu + zc * se,
    }
    if mu.ndim == 0 and se.ndim == 0:
        out = {key: float(v) for key, v in out.items()}
    return out


def mde(se, spec: Optional[Dict[str, Any]] = None):
    """
    Minimum detectable effect at 80% power: (z_α + z_power)·SE (§9).

    z_α is the first Holm threshold, i.e. the MDE assumes every other cell
    is null; it is a net effect, in the units of ``se``.
    """
    pw = (spec or get_spec())["power"]
    z = float(pw["z_alpha"]) + float(pw["z_power"])
    return z * float(se) if np.ndim(se) == 0 else z * np.asarray(se, dtype=np.float64)


def scale_se(se_is, n_is, n_primary):
    """SE_IS·√(N_IS/N_primary) (§9); keeps the cell's own IS active share."""
    return se_is * np.sqrt(np.asarray(n_is, dtype=np.float64) / np.asarray(n_primary, dtype=np.float64))


# --------------------------------------------------------------------------
# R1: circular-shift placebo
# --------------------------------------------------------------------------

def placebo_offsets(n_grid: int, d30_bars: int, k: int) -> np.ndarray:
    """
    δ_j = D + round(j·(N - 2D)/(K - 1)), j = 0..K-1, deduplicated (§8 R1).

    Rounding is half up, done in exact integer arithmetic so no float error
    can move an offset. With K = 2000, K - 1 is odd and j·(N - 2D)/(K - 1) is
    never an exact half (that would need an even number to equal an odd one),
    so half-up, half-to-even (np.round) and Python's round all agree.
    Offsets run from D to N - D: no rotation brings a signal within 30 days
    of where it started, on either side of the circle.
    """
    n_grid, d30_bars, k = int(n_grid), int(d30_bars), int(k)
    span = n_grid - 2 * d30_bars
    if span <= 0:
        raise ValueError("grid shorter than two minimum shifts")
    if k < 2:
        raise ValueError("need K >= 2")
    j = np.arange(k, dtype=np.int64)
    off = d30_bars + (2 * j * span + (k - 1)) // (2 * (k - 1))
    return np.unique(off)


def circular_placebo_t(s, fwd, d: int, cost_bps: float, ts, start_ms: int, end_ms: int,
                       bar_ms: int, h: int, offsets) -> np.ndarray:
    """
    t_HH after rotating s by each offset on the window's grid (§8 R1).

    s is laid on the N-position grid (0 at dropped buckets and outside the
    window), rotated so s'_g = s_{(g-δ) mod N}, and read back at each bar's
    grid position; A, n and t_HH are then recomputed with the same d and c.
    Rotating on the grid rather than on rows keeps every signal's distance
    in time, so the placebo inherits the signal's persistence and the
    realised return path exactly. A rotation with no active bar gives NaN,
    which never counts as exceeding t_obs.
    """
    if d not in (1, -1):
        raise ValueError("d must be +1 or -1")
    s = np.asarray(s)
    fwd = np.asarray(fwd, dtype=np.float64)
    pos, inwin, n_grid = _grid(ts, start_ms, end_ms, bar_ms)
    ev = evaluable(ts, start_ms, end_ms, bar_ms, h) & np.isfinite(fwd)
    s_grid = np.zeros(n_grid, dtype=np.int8)
    s_grid[pos[inwin]] = s[inwin]
    fwd_grid = np.full(n_grid, np.nan)
    fwd_grid[pos[ev]] = fwd[ev]
    fwd_ok = np.isfinite(fwd_grid)
    c = _cost_frac(cost_bps)
    out = np.empty(len(offsets))
    with np.errstate(invalid="ignore"):
        for j, delta in enumerate(np.asarray(offsets, dtype=np.int64)):
            sr = np.roll(s_grid, int(delta))
            active = fwd_ok & (sr != 0)
            n_grid_vals = d * sr.astype(np.float64) * fwd_grid - c
            out[j] = _stat_on_grid(n_grid_vals, active, h)["t"]
    return out


def placebo_p(t_obs: float, t_shift) -> float:
    """(1 + #{t_j >= t_obs}) / (1 + K'); smallest attainable p is 1/(1+K')."""
    t_shift = np.asarray(t_shift, dtype=np.float64)
    if not t_obs == t_obs:
        return math.nan
    with np.errstate(invalid="ignore"):
        hits = int(np.sum(t_shift >= t_obs))
    return (1 + hits) / (1 + len(t_shift))


def family_placebo_p(t_obs_by_cell, t_shift_matrix, testable=None) -> np.ndarray:
    """
    Family-wise placebo p per cell (§8 R1), single-step max-t.

    Row c of ``t_shift_matrix`` is cell c's placebo t's; column j must be the
    same offset index j for every cell (the 15m and 1H grids give different
    δ in bars for the same j). For each column, take the max over testable
    cells; cell c's p is (1 + #{max_j >= t_obs_c}) / (1 + K'). The smallest
    of these is the family p of the best cell.

    ``testable`` (default: t_obs finite) says which cells' rotations enter
    the max. A testable cell whose own t_HH is undefined (SE = 0) still
    contributes its rotations but gets no p of its own; untestable cells get
    NaN and stay out of the max.
    """
    t_obs = np.asarray(t_obs_by_cell, dtype=np.float64)
    T = np.asarray(t_shift_matrix, dtype=np.float64)
    if T.ndim != 2 or T.shape[0] != len(t_obs):
        raise ValueError("t_shift_matrix must be (n_cells, K')")
    in_max = np.isfinite(t_obs) if testable is None else np.asarray(testable, dtype=bool)
    out = np.full(len(t_obs), np.nan)
    if not in_max.any():
        return out
    sub = np.where(np.isfinite(T[in_max]), T[in_max], -np.inf)
    col_max = sub.max(axis=0)
    kp = T.shape[1]
    for c in np.flatnonzero(in_max & np.isfinite(t_obs)):
        out[c] = (1 + int(np.sum(col_max >= t_obs[c]))) / (1 + kp)
    return out


# --------------------------------------------------------------------------
# R9: full-window Spearman IC
# --------------------------------------------------------------------------

def _avg_rank(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank."""
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    starts = np.flatnonzero(np.r_[True, sa[1:] != sa[:-1]])
    ends = np.r_[starts[1:], n]
    avg = (starts + ends - 1) / 2.0 + 1.0
    ranks = np.empty(n)
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def _standardised_ranks(a: np.ndarray) -> Optional[np.ndarray]:
    """Average ranks scaled to mean 0 and (population) variance 1; None if all tied."""
    r = _avg_rank(a)
    r -= r.mean()
    var = float(np.dot(r, r)) / len(r)
    if not var > 0.0:
        return None
    return r / math.sqrt(var)


def full_window_ic(f, fwd, ts, start_ms: int, end_ms: int, bar_ms: int, h: int) -> Dict[str, Any]:
    """
    Spearman IC over the whole evaluation window with a §6.7 SE (§8 R9).

    The pairs are the evaluable bars of [start, end) where f and fwd are
    both finite. u and v are the average ranks of f and fwd among those
    pairs, each standardised to mean 0 and variance 1, so

        IC = mean(u_t·v_t)

    is Spearman's rank correlation. Its SE is §6.7's Hansen-Hodrick SE of
    that mean: the centred products u_t·v_t - IC sit on the window's bar
    grid, zero where a bar has no pair (dropped bucket, undefined f or fwd),
    Ω uses ℓ = h-1 with uniform weights and the Bartlett fallback (ℓ = 2h)
    when Ω <= 0, and SE = √Ω / n. The test is two-sided. As in §6.7, SE = 0
    leaves t undefined and makes p = 1. With fewer than two pairs, or with
    every pair tied in f or in fwd, the IC itself is undefined (all NaN);
    the R9 family then counts that cell's p as 1.

    Ranking over the whole window, not within each day, is what keeps the
    statistic centred for persistent price-derived factors (§12).
    """
    f = np.asarray(f, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)
    pos, _, n_grid = _grid(ts, start_ms, end_ms, bar_ms)
    if f.shape != pos.shape or fwd.shape != pos.shape:
        raise ValueError("f, fwd and ts must have the same length")
    ok = evaluable(ts, start_ms, end_ms, bar_ms, h) & np.isfinite(f) & np.isfinite(fwd)
    n = int(ok.sum())
    out = {"ic": math.nan, "se": math.nan, "t": math.nan, "p": math.nan, "n": n, "se_method": None}
    if n < 2:
        return out
    u = _standardised_ranks(f[ok])
    v = _standardised_ranks(fwd[ok])
    if u is None or v is None:
        return out
    prod = u * v
    ic = float(prod.mean())
    z = np.zeros(n_grid)
    z[pos[ok]] = prod - ic
    omega, method = hh_omega(z, h)
    se = math.sqrt(omega) / n if omega > 0 else 0.0
    t = ic / se if se > 0 else math.nan
    p = _p_two_sided(t) if se > 0 else 1.0
    out.update(ic=ic, se=se, t=t, p=p, se_method=method)
    return out


def daily_ic(f, fwd, ts, min_pairs: int = 20, mask=None) -> Dict[str, Any]:
    """
    SUPERSEDED: the mean of daily Spearman ICs first registered as R9.

    Not used by the pipeline. It is kept only so that the tests can show
    why it was replaced before publication (§12, 2026-10-09): ranking within
    a UTC day subtracts the day's own mean from both the forward returns
    and the recent returns a price-derived factor is built from, so on a
    pure random walk a persistent factor gets a negative mean daily IC with
    |t| far above any threshold. full_window_ic is the registered R9.

    Each pair goes to the UTC day of bar t's start. A day needs ``min_pairs``
    pairs with finite f and fwd and non-zero rank variance in both, else it is
    dropped. Days sit on the calendar-day grid with dropped days contributing
    0, so the lag-1 term only pairs days that really are adjacent. ℓ = 1 is
    hh_omega with h = 2, including its Bartlett fallback (ℓ = 4). Two-sided p.
    """
    f = np.asarray(f, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)
    ts = np.asarray(ts).astype(np.int64)
    ok = np.isfinite(f) & np.isfinite(fwd)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    day = ts[ok] // DAY_MS
    fx, fy = f[ok], fwd[ok]
    order = np.argsort(day, kind="mergesort")
    day, fx, fy = day[order], fx[order], fy[order]

    days: List[int] = []
    ics: List[float] = []
    if len(day):
        cut = np.flatnonzero(np.diff(day)) + 1
        for lo, hi in zip(np.r_[0, cut], np.r_[cut, len(day)]):
            if hi - lo < min_pairs:
                continue
            rx = _avg_rank(fx[lo:hi])
            ry = _avg_rank(fy[lo:hi])
            rx -= rx.mean()
            ry -= ry.mean()
            sxx, syy = float(np.dot(rx, rx)), float(np.dot(ry, ry))
            if sxx == 0.0 or syy == 0.0:
                continue
            days.append(int(day[lo]))
            ics.append(float(np.dot(rx, ry)) / math.sqrt(sxx * syy))

    nd = len(ics)
    if nd == 0:
        return {"ic_mean": math.nan, "ic_se": math.nan, "ic_t": math.nan,
                "ic_p": math.nan, "n_days": 0, "se_method": None}
    ic = np.asarray(ics)
    dd = np.asarray(days, dtype=np.int64)
    mean = float(ic.mean())
    z = np.zeros(int(dd[-1] - dd[0]) + 1)
    z[dd - dd[0]] = ic - mean
    omega, method = hh_omega(z, 2)
    se = math.sqrt(omega) / nd if omega > 0 else 0.0
    t = mean / se if se > 0 else math.nan
    p = _p_two_sided(t)
    return {"ic_mean": mean, "ic_se": se, "ic_t": t, "ic_p": p, "n_days": nd,
            "se_method": method}


# --------------------------------------------------------------------------
# R2: v1's statistic, through v1's own code
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _v1_module() -> ModuleType:
    """
    scripts/factor_research.py, loaded by path.

    R2 must be v1's statistic as written, not a re-implementation of it, and
    the script is not a package.
    """
    path = REPO_ROOT / "scripts" / "factor_research.py"
    spec = importlib.util.spec_from_file_location("_factor_research_v1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def v1_t_sqrt_h(f, fwd, h: int, quantiles: int = 5, mask=None) -> float:
    """
    v1's top-minus-bottom t ÷ √h (§8 R2), gross and not multiplied by d.

    Pass the evaluation bars (or a ``mask`` selecting them): layered_returns
    then sets its edges over the whole window, as v1 did. NaN where v1's
    code declines to produce a number.
    """
    mod = _v1_module()
    f = np.asarray(f, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        f, fwd = f[m], fwd[m]
    layers = mod.layered_returns(f, fwd, quantiles)
    if layers is None:
        return math.nan
    return float(mod._spread_t_stat(layers, h))


# --------------------------------------------------------------------------
# R3-R8, R10, RL
# --------------------------------------------------------------------------

def by_quarter(n, ts, start_ms: int, end_ms: int, bar_ms: int, h: int,
               bounds_ms: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """
    §6.7 statistics per quarter (R3); trades belong to the quarter of bar t.

    A trade near a boundary keeps its whole forward return in the quarter
    where its signal fired; the quarter boundaries do not truncate holdings.
    Inner bounds default to Appendix A's quarter_bounds_ms.
    """
    inner = list(get_spec()["robustness"]["quarter_bounds_ms"] if bounds_ms is None else bounds_ms)
    edges = [int(start_ms)] + [int(b) for b in inner] + [int(end_ms)]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("quarter bounds must lie strictly inside the window, ascending")
    n = np.asarray(n, dtype=np.float64)
    ts64 = np.asarray(ts).astype(np.int64)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        nq = np.where((ts64 >= lo) & (ts64 < hi), n, np.nan)
        st = cell_statistic(nq, ts64, start_ms, end_ms, bar_ms, h)
        rows.append({"quarter": i + 1, "start_ms": lo, "end_ms": hi, **st})
    return rows


def volatile_days(ts_1m, close_1m, start_ms: int, end_ms: int, k: int) -> np.ndarray:
    """
    Start (ms) of the k UTC days with the largest Σ r² of simple 1-minute
    returns inside the window (R4), most volatile first.

    A return is close_i / close_{i-1} - 1 over two consecutive minutes and is
    realised during minute i, so it belongs to minute i's day and counts when
    minute i starts in [start, end); returns across a missing minute are not
    1-minute returns and are skipped. Ties go to the earlier day.
    """
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    ts = _check_ts(ts_1m, MINUTE_MS)
    c = np.asarray(close_1m, dtype=np.float64)
    if len(ts) < 2:
        return np.zeros(0, dtype=np.int64)
    prev, cur = c[:-1], c[1:]
    t_cur = ts[1:]
    ok = (
        (np.diff(ts) == MINUTE_MS) & (t_cur >= start_ms) & (t_cur < end_ms)
        & np.isfinite(prev) & np.isfinite(cur) & (prev > 0)
    )
    if not ok.any():
        return np.zeros(0, dtype=np.int64)
    r = cur[ok] / prev[ok] - 1.0
    day = t_cur[ok] // DAY_MS
    uday, inv = np.unique(day, return_inverse=True)
    ss = np.bincount(inv, weights=r * r)
    order = np.lexsort((uday, -ss))
    return (uday[order[:k]] * DAY_MS).astype(np.int64)


def leave_out_volatile_days(n, ts, start_ms: int, end_ms: int, bar_ms: int, h: int,
                            days_ms) -> Dict[str, Any]:
    """
    §6.7 statistics without trades exposed to the given UTC days (R4).

    A trade is held from the close of t+1 to the close of t+1+h, i.e. over
    [ts_t + 2B, ts_t + (h+2)B] for the consecutive bars every defined trade
    has. It is dropped if that span shares a stretch of positive length with
    [D, D + 1 day): a trade that exits exactly at midnight earned none of the
    next day's returns and stays in.
    """
    n = np.asarray(n, dtype=np.float64)
    ts64 = np.asarray(ts).astype(np.int64)
    act = np.isfinite(n) & evaluable(ts64, start_ms, end_ms, bar_ms, h)
    entry = ts64 + 2 * bar_ms
    exit_ = ts64 + (h + 2) * bar_ms
    hit = np.zeros(len(n), dtype=bool)
    days = np.asarray(days_ms, dtype=np.int64).ravel()
    for d0 in days:
        hit |= (entry < d0 + DAY_MS) & (exit_ > d0)
    drop = act & hit
    st = cell_statistic(np.where(drop, np.nan, n), ts64, start_ms, end_ms, bar_ms, h)
    return {"k": int(len(days)), "n_dropped": int(drop.sum()), **st}


def turnover_aware(s, d: int, close, ts, start_ms: int, end_ms: int, bar_ms: int, h: int,
                   cost_bps: float = 10.0) -> Dict[str, Any]:
    """
    Capital-constrained P&L of the staggered portfolio, per day (R5).

    p_u = (d/h) Σ_{j=1..h} s_{u-j} on the grid: each signal holds 1/h of
    capital from the close of t+1 to the close of t+1+h, so overlapping
    trades share capital instead of each counting in full as they do in μ̂.
    It earns p_u·r_{u+1} and pays c/2 per unit of |p_u - p_{u-1}|.

    Only evaluable bars' signals are traded, so every position opens and
    closes inside the window. A return across a dropped bucket is unknown:
    it contributes nothing and is counted in n_unpriced_steps, while the
    position, and so its cost, still evolves on the grid.
    """
    if d not in (1, -1):
        raise ValueError("d must be +1 or -1")
    s = np.asarray(s)
    close = np.asarray(close, dtype=np.float64)
    pos, inwin, n_grid = _grid(ts, start_ms, end_ms, bar_ms)
    ev = evaluable(ts, start_ms, end_ms, bar_ms, h)
    s_grid = np.zeros(n_grid, dtype=np.int64)
    s_grid[pos[ev]] = s[ev]
    cs = np.concatenate(([0], np.cumsum(s_grid)))
    u = np.arange(n_grid)
    p = d * (cs[u] - cs[np.maximum(u - h, 0)]) / h
    if n_grid and p[-1] != 0:  # cannot happen: evaluable stops h+1 bars early
        raise AssertionError("position open at the end of the window")

    c_grid = np.full(n_grid, np.nan)
    c_grid[pos[inwin]] = close[inwin]
    r_next = np.full(n_grid, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_next[:-1] = c_grid[1:] / c_grid[:-1] - 1.0
    held = p != 0
    priced = held & np.isfinite(r_next)
    gross = float(np.sum(p[priced] * r_next[priced]))
    turnover = float(np.sum(np.abs(np.diff(np.concatenate(([0.0], p))))))
    cost = turnover * _cost_frac(cost_bps) / 2.0
    n_days = (end_ms - start_ms) / DAY_MS
    return {
        "value_bps": (gross - cost) / n_days * 1e4,
        "gross_bps_per_day": gross / n_days * 1e4,
        "cost_bps_per_day": cost / n_days * 1e4,
        "turnover_per_day": turnover / n_days,
        "n_days": n_days,
        "n_unpriced_steps": int((held & ~np.isfinite(r_next)).sum()),
    }


def cost_sensitivity(s, fwd, d: int, ts, start_ms: int, end_ms: int, bar_ms: int, h: int,
                     costs_bps: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    """§6.7 statistics at each alternative round-trip cost (R6); 0 is gross."""
    costs = get_spec()["robustness"]["cost_sensitivity_bps"] if costs_bps is None else costs_bps
    rows = []
    for c in costs:
        st = cell_statistic(per_trade_net(s, fwd, d, c), ts, start_ms, end_ms, bar_ms, h)
        rows.append({"cost_bps": float(c), **st})
    return rows


def winsorized_fwd(fwd, ts, start_ms: int, end_ms: int, bar_ms: int, h: int,
                   pct: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    fwd clipped at the 1st/99th percentiles of the window's defined fwd (R7).

    The bounds come only from evaluable bars with a defined fwd for this bar
    size and h (numpy 'linear' percentiles); every finite value is clipped,
    though only the window's bars matter downstream.
    """
    lo_p, hi_p = get_spec()["robustness"]["winsorize_pct"] if pct is None else pct
    fwd = np.asarray(fwd, dtype=np.float64)
    m = evaluable(ts, start_ms, end_ms, bar_ms, h) & np.isfinite(fwd)
    out = fwd.copy()
    if not m.any():
        return out
    lo, hi = np.percentile(fwd[m], [lo_p, hi_p])
    fin = np.isfinite(out)
    out[fin] = np.clip(out[fin], lo, hi)
    return out


def funding_adjusted_net(n, s, d: int, ts, bar_ms: int, h: int, fund_ts, fund_rate,
                         spacing_ms: Optional[int] = None) -> np.ndarray:
    """
    n_t - d·s_t·Σ FR_u over settlements with entry <= τ_u < exit (R8).

    entry = ts[t+1] + B and exit = ts[t+1+h] + B, the closing times of the
    entry and exit bars. A long position pays a positive rate. Settlement
    times are expected every ``spacing_ms`` (default and pipeline value:
    Appendix A's expected_spacing_hours, 8 h, fixed like §4.2's staleness
    limit, so that no series statistic decides which settlements count as
    missing), on the phase most recorded settlements share. If an expected
    time inside [entry, exit) has no finite recorded rate, the trade's
    funding is unknown and the result is NaN rather than a silent zero; a
    recorded off-phase settlement is still charged. NaN wherever n is.
    """
    if d not in (1, -1):
        raise ValueError("d must be +1 or -1")
    sp = int(spacing_ms) if spacing_ms is not None else \
        int(get_spec()["data"]["funding"]["expected_spacing_hours"]) * HOUR_MS
    n = np.asarray(n, dtype=np.float64)
    s = np.asarray(s)
    ts64 = _check_ts(ts, bar_ms)
    ft = np.asarray(fund_ts).astype(np.int64)
    fr = np.asarray(fund_rate, dtype=np.float64)
    if len(ft) and np.any(np.diff(ft) <= 0):
        raise ValueError("funding times must be strictly increasing (de-duplicated)")
    out = np.full(len(n), np.nan)
    m = len(n)
    act = np.isfinite(n) & (s != 0)
    idx = np.flatnonzero(act)
    idx = idx[idx + 1 + h < m]
    if len(idx) == 0:
        return out
    consecutive = ts64[idx + 1 + h] - ts64[idx] == (h + 1) * bar_ms
    idx = idx[consecutive]
    entry = ts64[idx + 1] + bar_ms
    exit_ = ts64[idx + 1 + h] + bar_ms

    rec = np.isfinite(fr)
    ft_r, fr_r = ft[rec], fr[rec]
    if len(ft):
        phases, counts = np.unique(ft % sp, return_counts=True)
        phase = int(phases[np.argmax(counts)])
    else:
        phase = 0
    csum = np.concatenate(([0.0], np.cumsum(fr_r)))
    lo = np.searchsorted(ft_r, entry, side="left")
    hi = np.searchsorted(ft_r, exit_, side="left")
    fsum = csum[hi] - csum[lo]

    def n_grid_before(x):  # expected settlement times strictly before x
        return -((phase - x) // sp)

    expected = n_grid_before(exit_) - n_grid_before(entry)
    on = ft_r[(ft_r - phase) % sp == 0]
    recorded = np.searchsorted(on, exit_, side="left") - np.searchsorted(on, entry, side="left")
    ok = recorded >= expected
    sel = idx[ok]
    out[sel] = n[sel] - d * s[sel].astype(np.float64) * fsum[ok]
    return out


def quintile_means(f, fwd, ts, start_ms: int, end_ms: int, bar_ms: int, h: int,
                   min_prior: int = 500, quantiles: int = 5) -> List[Dict[str, Any]]:
    """
    Mean gross fwd in each expanding-window quintile, with §6.7 stats (R10).

    Edges P20/P40/P60/P80 come from earlier bars, as in §6.4. A value equal
    to an edge moves toward the middle quintile (Appendix A's edge_ties:
    middle), so Q1 and Q5 are exactly §6.4's bottom and top buckets:

        Q1: f < e1   Q2: e1 <= f < e2   Q3: e2 <= f <= e3
        Q4: e3 < f <= e4   Q5: f > e4

    Merging coincident edges: at a bar where Q2's edges coincide (e1 == e2)
    it is empty and its tied values land in Q3; likewise Q4 into Q3 when
    e3 == e4. Since edges move bar by bar, this is decided per bar and
    ``merged_bars`` counts the bars at which a quintile was folded into its
    middle neighbour; one merged at every bar has n_active 0. Q3 with
    e2 == e3 is not merged: it is the point mass itself. ``se``, ``t`` and
    ``p`` are §6.7 on the quintile's own bars (gross, no cost, no d).
    """
    if quantiles < 3 or quantiles % 2 == 0:
        raise ValueError("the middle-tie rule needs an odd number of buckets >= 3")
    f = np.asarray(f, dtype=np.float64)
    fwd = np.asarray(fwd, dtype=np.float64)
    pcts = [100.0 * i / quantiles for i in range(1, quantiles)]
    e = expanding_percentiles(f, pcts, min_prior)
    ev = (
        evaluable(ts, start_ms, end_ms, bar_ms, h) & np.isfinite(fwd) & np.isfinite(f)
        & np.all(np.isfinite(e), axis=1)
    )
    mid = (quantiles + 1) // 2  # 1-based middle bucket
    label = np.ones(len(f), dtype=np.int64)
    for i in range(1, quantiles):  # edge e_i, 1-based
        col = e[:, i - 1]
        label += (f >= col) if i < mid else (f > col)
    label[~ev] = 0

    rows = []
    for q in range(1, quantiles + 1):
        if q in (1, mid, quantiles):
            merged = np.zeros(len(f), dtype=bool)
        else:  # quintile q has edges e_{q-1}, e_q
            merged = ev & (e[:, q - 2] == e[:, q - 1])
        st = cell_statistic(np.where(label == q, fwd, np.nan), ts, start_ms, end_ms, bar_ms, h)
        rows.append({"quintile": q, "value": st["mu"], "merged_bars": int(merged.sum()), **st})
    return rows


def mean_run_length(s, ts, start_ms: int, end_ms: int, bar_ms: int, h: int) -> Tuple[float, int]:
    """
    Realised mean run length of s_t in bars, and the number of runs (§7, row RL).

    A run is a maximal stretch of consecutive grid positions holding the same
    non-zero s, taken over the evaluable bars of [start, end) (the bars
    whose signals the window may trade, as in R5). A sign change, s = 0, a
    dropped bucket or a non-evaluable bar ends a run: the signal is laid on
    the grid with 0 everywhere else, so two equal signals either side of a
    missing bucket count as two runs, never one. NaN (0 runs) when no
    evaluable bar has s != 0.

    §7 asks for this next to every confirmed cell's decision, so the
    pipeline writes it for every cell and no second pass over test data is
    needed after the one run.
    """
    s = np.asarray(s)
    pos, _, n_grid = _grid(ts, start_ms, end_ms, bar_ms)
    if s.shape != pos.shape:
        raise ValueError("s and ts must have the same length")
    ev = evaluable(ts, start_ms, end_ms, bar_ms, h)
    g = np.zeros(n_grid, dtype=np.int8)
    g[pos[ev]] = np.sign(s[ev])
    nz = g != 0
    starts = nz & np.r_[True, g[1:] != g[:-1]]
    runs = int(starts.sum())
    return (float(nz.sum()) / runs if runs else math.nan), runs
