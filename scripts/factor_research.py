#!/usr/bin/env python3
"""
Factor validity study on real OKX history.

Answers the three questions that decide whether a factor is worth trading,
across several holding horizons:

    1. Does it predict?      -> IC (Spearman rank correlation vs forward return)
    2. Is that stable?       -> IR = mean(rolling IC) / std(rolling IC)
    3. Does it survive cost? -> layered backtest, net of round-trip fees

Two conventions here matter more than the numbers they produce:

  * **Entry is delayed one bar.** A factor computed from bar t's close cannot
    be traded at bar t's close — that price is already gone. Returns are
    therefore measured from t+1 to t+1+h. Skipping this is the single most
    common way a crypto backtest invents an edge that does not exist.

  * **IC here is time-series, not cross-sectional.** With one instrument there
    is no cross-section to rank, so this measures whether the factor's own
    level predicts its own future return. It is the weaker of the two claims;
    do not quote it as if it were a cross-sectional IC.

Usage:
    python scripts/factor_research.py
    python scripts/factor_research.py --bars 1m 5m 15m 1H --horizons 1 5 15
    python scripts/factor_research.py --fee-bps 10 --quantiles 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

HISTORY_DIR = REPO_ROOT / "data" / "history"

# Resample rules: how many 1m bars make up each target bar.
BAR_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1H": 60, "4H": 240}


def load_history(inst_id: str) -> Dict[str, pl.DataFrame]:
    """Load whatever fetch_history.py produced for this instrument."""
    base = HISTORY_DIR / inst_id
    if not base.exists():
        raise FileNotFoundError(
            f"No history at {base.relative_to(REPO_ROOT)}. "
            f"Run: python scripts/fetch_history.py --inst-id {inst_id} --with-spot"
        )

    out: Dict[str, pl.DataFrame] = {}

    candles_path = base / "candles_1m.parquet"
    if not candles_path.exists():
        raise FileNotFoundError(f"Missing {candles_path.relative_to(REPO_ROOT)}")
    out["candles"] = pl.read_parquet(candles_path)

    for key, fname in (
        ("spot_candles", "spot_candles_1m.parquet"),
        ("funding_rates", "funding_rates.parquet"),
        ("open_interest", "open_interest.parquet"),
    ):
        path = base / fname
        if path.exists():
            out[key] = pl.read_parquet(path)

    return out


def resample(df: pl.DataFrame, minutes: int) -> pl.DataFrame:
    """Aggregate 1m OHLCV into a coarser bar."""
    if minutes == 1:
        return df

    every_ms = minutes * 60_000
    return (
        df.with_columns((pl.col("timestamp") // every_ms * every_ms).alias("bucket"))
        .group_by("bucket")
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
            pl.col("volume_ccy").sum(),
        )
        .rename({"bucket": "timestamp"})
        .sort("timestamp")
    )


def forward_returns(close: np.ndarray, horizon: int, entry_lag: int = 1) -> np.ndarray:
    """
    Return from entry to exit for a signal observed at bar t.

    With entry_lag=1: buy at close[t+1], sell at close[t+1+h]. The lag is what
    keeps the signal tradeable — see the module docstring.
    """
    n = len(close)
    out = np.full(n, np.nan)
    start, end = entry_lag, entry_lag + horizon

    valid = n - end
    if valid <= 0:
        return out

    entry = close[start : start + valid]
    exit_ = close[end : end + valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:valid] = np.where(entry > 0, exit_ / entry - 1.0, np.nan)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation over the pairwise-complete observations."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return np.nan

    xr = _rank(x[mask])
    yr = _rank(y[mask])
    if xr.std() == 0 or yr.std() == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not distort the correlation."""
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)

    # Average the ranks inside each group of equal values.
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def rolling_ic(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """IC computed on consecutive non-overlapping windows, for the IR estimate."""
    ics = []
    for start in range(0, len(x) - window + 1, window):
        ic = spearman(x[start : start + window], y[start : start + window])
        if np.isfinite(ic):
            ics.append(ic)
    return np.array(ics)


def layered_returns(
    factor: np.ndarray, fwd: np.ndarray, quantiles: int
) -> Optional[Dict[str, np.ndarray]]:
    """
    Bucket bars by factor value and average the forward return in each bucket.

    A usable factor is monotone across buckets: the story "higher factor ->
    higher return" should hold in the middle, not only at the extremes.
    """
    mask = np.isfinite(factor) & np.isfinite(fwd)
    if mask.sum() < quantiles * 50:
        return None

    f, r = factor[mask], fwd[mask]
    edges = np.quantile(f, np.linspace(0, 1, quantiles + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    means, counts, variances = [], [], []
    for i in range(quantiles):
        bucket = (f >= edges[i]) & (f < edges[i + 1])
        n = int(bucket.sum())
        means.append(r[bucket].mean() if n else np.nan)
        variances.append(r[bucket].var(ddof=1) if n > 1 else np.nan)
        counts.append(n)

    return {
        "mean_return": np.array(means),
        "count": np.array(counts),
        "variance": np.array(variances),
        "n_obs": int(mask.sum()),
    }


def _spread_t_stat(layers: Dict[str, np.ndarray], horizon: int) -> float:
    """
    t-statistic for the top-minus-bottom quintile spread.

    Two-sample difference of means, then divided by sqrt(horizon): with a
    horizon longer than one bar the forward-return windows overlap, so
    consecutive observations are not independent and the naive t is inflated
    by roughly sqrt(h). This is the cheap version of a Newey-West correction
    and errs toward calling things insignificant.
    """
    means, var, cnt = layers["mean_return"], layers["variance"], layers["count"]

    if not (np.isfinite(means[0]) and np.isfinite(means[-1])):
        return np.nan
    if not (np.isfinite(var[0]) and np.isfinite(var[-1])):
        return np.nan
    if cnt[0] < 2 or cnt[-1] < 2:
        return np.nan

    se = np.sqrt(var[-1] / cnt[-1] + var[0] / cnt[0])
    if se <= 0:
        return np.nan

    return float((means[-1] - means[0]) / se / np.sqrt(max(1, horizon)))


def analyze(
    factors_df: pl.DataFrame,
    close: np.ndarray,
    factor_names: List[str],
    horizons: List[int],
    bar: str,
    fee_bps: float,
    quantiles: int,
    ic_window: int,
    min_obs: int,
) -> List[Dict]:
    """Run the full battery for one bar size."""
    rows: List[Dict] = []
    skipped: List[str] = []

    for horizon in horizons:
        fwd = forward_returns(close, horizon)

        for name in factor_names:
            factor = factors_df[name].to_numpy().astype(np.float64)

            # An IC computed on a few dozen points is noise dressed as a
            # number — the exchange only retains a short history for some
            # inputs (OI, funding), so guard on sample size before reporting.
            available = int((np.isfinite(factor) & np.isfinite(fwd)).sum())
            if available < min_obs:
                skipped.append(f"{name}(h={horizon}, n={available})")
                continue

            ic = spearman(factor, fwd)
            ics = rolling_ic(factor, fwd, ic_window)
            ir = float(ics.mean() / ics.std()) if len(ics) > 2 and ics.std() > 0 else np.nan

            layers = layered_returns(factor, fwd, quantiles)
            if layers is not None:
                means = layers["mean_return"]
                spread_bps = (means[-1] - means[0]) * 10_000
                t_stat = _spread_t_stat(layers, horizon)
                n_obs = layers["n_obs"]
                # Monotonicity: correlation between bucket index and its mean return.
                finite = np.isfinite(means)
                mono = (
                    float(np.corrcoef(np.arange(quantiles)[finite], means[finite])[0, 1])
                    if finite.sum() > 2
                    else np.nan
                )
            else:
                spread_bps, mono, t_stat = np.nan, np.nan, np.nan
                n_obs = int((np.isfinite(factor) & np.isfinite(fwd)).sum())

            rows.append(
                {
                    "bar": bar,
                    "horizon": horizon,
                    "factor": name,
                    "n_obs": n_obs,
                    "ic": ic,
                    "ir": ir,
                    "n_windows": len(ics),
                    "spread_bps": spread_bps,
                    "t_stat": t_stat,
                    "net_spread_bps": spread_bps - fee_bps,
                    "monotonicity": mono,
                }
            )

    if skipped:
        logger.warning(
            f"{bar}: skipped {len(skipped)} combination(s) with fewer than "
            f"{min_obs:,} usable observations: {', '.join(skipped[:6])}"
            + (" ..." if len(skipped) > 6 else "")
        )

    return rows


def print_table(rows: List[Dict], fee_bps: float) -> None:
    """Print results grouped by bar, sorted by net long-short spread."""
    if not rows:
        logger.warning("No results to show")
        return

    df = pl.DataFrame(rows)

    for bar in df["bar"].unique(maintain_order=True):
        sub = df.filter(pl.col("bar") == bar)

        logger.info("")
        logger.info("=" * 96)
        logger.info(f"  bar = {bar}   (round-trip cost {fee_bps:.1f} bp)")
        logger.info("=" * 96)
        logger.info(
            f"{'factor':<22} {'h':>3} {'n':>9} {'IC':>8} {'IR':>6} "
            f"{'spread':>9} {'net':>9} {'t':>6} {'mono':>6}"
        )
        logger.info("-" * 96)

        for row in sub.sort("net_spread_bps", descending=True, nulls_last=True).iter_rows(named=True):
            net = row["net_spread_bps"]
            t = row["t_stat"]
            # Only call something a survivor if it clears cost AND is
            # distinguishable from noise. |t| < 2 on an in-sample spread is
            # not evidence of anything.
            if np.isfinite(net) and net > 0:
                flag = "  <-- clears cost, t>2" if np.isfinite(t) and abs(t) > 2 else "  <-- clears cost (t<2, noise)"
            else:
                flag = ""
            logger.info(
                f"{row['factor']:<22} {row['horizon']:>3} {row['n_obs']:>9,} "
                f"{_fmt(row['ic'], 4):>8} {_fmt(row['ir'], 2):>6} "
                f"{_fmt(row['spread_bps'], 2):>9} {_fmt(net, 2):>9} "
                f"{_fmt(t, 2):>6} {_fmt(row['monotonicity'], 2):>6}{flag}"
            )


def _fmt(v: Optional[float], nd: int) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def print_correlations(factors_df: pl.DataFrame, names: List[str], min_obs: int = 1000) -> None:
    """
    Factor rank-correlation matrix, computed pairwise.

    Pairwise rather than listwise: requiring every factor to be present on a
    row would shrink the sample to whatever the thinnest series covers (open
    interest is retained for days, RSI for the full year), and a correlation
    on a couple hundred rows says nothing. Each cell instead uses the rows
    where *that pair* is defined, and its sample size is reported separately.

    Highly correlated factors are not independent signals: feeding several of
    them to a tree model inflates apparent feature importance and hides how
    few distinct ideas the model actually has.
    """
    cols = {n: factors_df[n].to_numpy().astype(np.float64) for n in names}
    k = len(names)
    corr = np.full((k, k), np.nan)
    pair_n = np.zeros((k, k), dtype=int)

    for i in range(k):
        for j in range(i, k):
            a, b = cols[names[i]], cols[names[j]]
            mask = np.isfinite(a) & np.isfinite(b)
            pair_n[i, j] = pair_n[j, i] = int(mask.sum())

            if mask.sum() < min_obs:
                continue
            ar, br = _rank(a[mask]), _rank(b[mask])
            if ar.std() == 0 or br.std() == 0:
                continue
            c = float(np.corrcoef(ar, br)[0, 1])
            corr[i, j] = corr[j, i] = c

    logger.info("")
    logger.info("=" * 96)
    logger.info("  Factor rank-correlation matrix (pairwise complete)")
    logger.info("=" * 96)

    width = max(len(n) for n in names) + 2
    logger.info(" " * width + "".join(f"{n[:7]:>9}" for n in names))
    for i, name in enumerate(names):
        cells = "".join(
            f"{corr[i, j]:>9.2f}" if np.isfinite(corr[i, j]) else f"{'--':>9}"
            for j in range(k)
        )
        logger.info(f"{name:<{width}}{cells}")

    logger.info(f"  ('--' = fewer than {min_obs:,} rows where both are defined)")
    logger.info("-" * 96)

    pairs = [
        (names[i], names[j], corr[i, j], pair_n[i, j])
        for i in range(k)
        for j in range(i + 1, k)
        if np.isfinite(corr[i, j]) and abs(corr[i, j]) > 0.7
    ]
    if pairs:
        logger.warning("Near-duplicate factors (|rho| > 0.7):")
        for a, b, c, n in sorted(pairs, key=lambda p: -abs(p[2])):
            logger.warning(f"    {a} <-> {b}: {c:+.2f}  (n={n:,})")
        logger.warning("")
        logger.warning("These are not independent signals. The effective number of")
        logger.warning("ideas in the factor set is smaller than the factor count.")
    else:
        logger.success("No factor pair exceeds |rho| = 0.7")


def run(args: argparse.Namespace) -> int:
    from quant_core.factor_engine import FactorEngine

    data = load_history(args.inst_id)
    base = data["candles"]
    logger.info(
        f"Loaded {base.height:,} 1m candles"
        + (", spot leg present" if "spot_candles" in data else ", no spot leg")
        + (", OI present" if "open_interest" in data else ", no OI")
    )

    engine = FactorEngine()
    all_rows: List[Dict] = []
    corr_done = False

    for bar in args.bars:
        if bar not in BAR_MINUTES:
            logger.warning(f"Unknown bar {bar}, skipping")
            continue

        minutes = BAR_MINUTES[bar]
        candles = resample(base, minutes)
        if candles.height < args.min_bars:
            logger.warning(f"{bar}: only {candles.height} bars, skipping")
            continue

        logger.info(f"{bar}: {candles.height:,} bars -> computing factors")

        payload = {"candles": candles}
        if "spot_candles" in data:
            payload["spot_candles"] = resample(data["spot_candles"], minutes)
        for key in ("funding_rates", "open_interest"):
            if key in data:
                payload[key] = data[key]

        factors_df = engine.calculate_all_factors(payload)
        if factors_df.is_empty():
            logger.warning(f"{bar}: factor engine returned nothing, skipping")
            continue

        names = [c for c in factors_df.columns if c not in ("timestamp", "datetime")]
        # Drop factors that came out constant — they carry no information and
        # would otherwise show up as n/a rows cluttering the table.
        live = [n for n in names if np.nanstd(factors_df[n].to_numpy().astype(float)) > 0]
        dead = sorted(set(names) - set(live))
        if dead:
            logger.warning(f"{bar}: constant factors dropped: {', '.join(dead)}")

        if not live:
            continue

        close = candles["close"].to_numpy().astype(np.float64)
        all_rows.extend(
            analyze(
                factors_df, close, live, args.horizons, bar,
                args.fee_bps, args.quantiles, args.ic_window, args.min_obs,
            )
        )

        if not corr_done:
            print_correlations(factors_df, live, args.min_obs)
            corr_done = True

    print_table(all_rows, args.fee_bps)

    if all_rows and args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(all_rows).write_parquet(out_path)
        logger.success(f"Results -> {out_path}")

    _print_verdict(all_rows, args.fee_bps)
    return 0


def _print_verdict(rows: List[Dict], fee_bps: float) -> None:
    if not rows:
        return

    df = pl.DataFrame(rows).filter(pl.col("net_spread_bps").is_finite())
    if df.is_empty():
        return

    n_tests = df.height
    clears_cost = df.filter(pl.col("net_spread_bps") > 0)
    # Significant only if it also stands apart from noise.
    significant = clears_cost.filter(
        pl.col("t_stat").is_finite() & (pl.col("t_stat").abs() > 2.0)
    )
    # Bonferroni: testing n hypotheses at once inflates the false-positive
    # rate, so the per-test bar has to rise with n.
    bonferroni_t = _bonferroni_threshold(n_tests)
    robust = clears_cost.filter(
        pl.col("t_stat").is_finite() & (pl.col("t_stat").abs() > bonferroni_t)
    )

    logger.info("")
    logger.info("=" * 96)
    logger.info("  Verdict")
    logger.info("=" * 96)
    logger.info(f"  {n_tests} factor x horizon combinations tested")
    logger.info(f"  {clears_cost.height:>3} clear the {fee_bps:.0f} bp round-trip cost")
    logger.info(f"  {significant.height:>3} of those also reach |t| > 2")
    logger.info(f"  {robust.height:>3} survive Bonferroni for {n_tests} tests (|t| > {bonferroni_t:.2f})")

    if robust.is_empty():
        logger.info("")
        logger.warning("  No combination is distinguishable from noise once the")
        logger.warning("  number of tests is accounted for.")
        logger.warning("")
        logger.warning("  Searching 36 combinations and reporting the best few is")
        logger.warning("  how backtests manufacture edges that do not exist. A")
        logger.warning("  positive net spread at |t| < 2 is a coin flip, not a signal.")
        logger.warning("")
        logger.warning("  Productive next steps, roughly in order of expected payoff:")
        logger.warning("    - lengthen the holding period further (hours, not minutes)")
        logger.warning("    - switch to maker orders, where the fee is far smaller")
        logger.warning("    - find factors with genuine economic content; more model")
        logger.warning("      capacity cannot manufacture an edge the data lacks")
    else:
        logger.info("")
        logger.info("  Survives cost and multiple testing:")
        for r in robust.sort("net_spread_bps", descending=True).head(5).iter_rows(named=True):
            logger.info(
                f"    {r['factor']:<22} bar={r['bar']:<4} h={r['horizon']:<3} "
                f"net={r['net_spread_bps']:+7.2f} bp  t={r['t_stat']:+5.2f}  "
                f"n={r['n_obs']:,}"
            )
        logger.info("")
        logger.info("  Still in-sample. Confirm on a held-out period before building")
        logger.info("  anything on top of these.")

    # Unequal sample sizes make the rows incomparable; say so explicitly.
    spans = df.group_by("factor").agg(pl.col("n_obs").max().alias("n"))
    widest = spans["n"].max()
    thin = spans.filter(pl.col("n") < widest * 0.5).sort("n")
    if thin.height:
        logger.info("")
        logger.warning("  Note — these factors were measured on far fewer bars than")
        logger.warning(f"  the others (widest coverage = {widest:,} bars), because the")
        logger.warning("  exchange only retains a short history for their inputs:")
        for r in thin.iter_rows(named=True):
            logger.warning(f"    {r['factor']:<24} {r['n']:>9,} bars")
        logger.warning("  Their IC/spread figures are not comparable to the rest.")

    logger.info("=" * 96)


def _bonferroni_threshold(n_tests: int) -> float:
    """
    Two-sided t threshold for alpha=0.05 after Bonferroni correction.

    Uses the normal approximation, which is fine at these sample sizes.
    """
    from math import sqrt

    alpha = 0.05 / max(1, n_tests)
    # Inverse normal CDF via a rational approximation (Acklam), two-sided.
    p = 1.0 - alpha / 2.0
    return _norm_ppf(p)


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    import math

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure factor validity against real history, net of costs."
    )
    p.add_argument("--inst-id", default="DOGE-USDT-SWAP")
    p.add_argument("--bars", nargs="+", default=["1m", "5m", "15m", "1H"])
    p.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 5, 15],
        help="forward return horizons, in bars of the current size",
    )
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--quantiles", type=int, default=5)
    p.add_argument("--ic-window", type=int, default=1440, help="bars per rolling-IC window")
    p.add_argument("--min-bars", type=int, default=500)
    p.add_argument(
        "--min-obs", type=int, default=1000,
        help="minimum usable observations before a factor x horizon is reported",
    )
    p.add_argument("--out", default="data/history/factor_research.parquet")
    return p.parse_args()


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    try:
        return run(parse_args())
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
