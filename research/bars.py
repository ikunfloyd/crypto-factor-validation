"""
Bars for the out-of-sample study (§6.1), and the joins that put other series
on their grid.

Two rules from the pre-registration shape everything here:

  * **No interpolation.** A bucket is kept only if every one of its 1-minute
    bars exists. A partial bucket's high, low and volume understate the real
    bar and its close may be stale; filling it would invent prices.

  * **Joins are by timestamp, never by position.** The spot leg and the
    higher timeframe have their own gaps, so pairing rows by index lines up
    different times (CONTRIBUTING.md: "Join coarser series as-of, never by
    index").
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import polars as pl

from research.spec import MINUTE_MS, get_spec

_BAR_RE = re.compile(r"^([1-9]\d*)(m|H)$")
_OHLCV = ("open", "high", "low", "close", "volume")


def bar_ms(bar: str) -> int:
    """Length of a bar label such as '15m', '1H' or '4H', in milliseconds."""
    m = _BAR_RE.match(bar)
    if m is None:
        raise ValueError(f"Unsupported bar {bar!r}; expected e.g. '15m', '1H', '4H'")
    minutes = int(m.group(1)) * (60 if m.group(2) == "H" else 1)
    return minutes * MINUTE_MS


def htf_bar(bar: str) -> str:
    """Label of the higher timeframe for an analysis bar (§5.2: 15m -> 1H, 1H -> 4H)."""
    mult = int(get_spec()["factors"]["triple_screen"]["htf_multiple"])
    minutes = bar_ms(bar) * mult // MINUTE_MS
    return f"{minutes // 60}H" if minutes % 60 == 0 else f"{minutes}m"


def _as_int_ms(values: Any) -> np.ndarray:
    """
    Timestamps as int64 milliseconds.

    Float input is accepted only when every value is integral: a truncating
    cast would silently move a bar into the neighbouring bucket.
    """
    a = np.asarray(values)
    if a.dtype.kind in "iu":
        return a.astype(np.int64)
    if a.dtype.kind == "M":
        return a.astype("datetime64[ms]").astype(np.int64)
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or np.any(a != np.round(a)):
            raise ValueError("timestamps must be whole milliseconds")
        return a.astype(np.int64)
    raise TypeError(f"timestamps must be numeric, got dtype {a.dtype}")


def resample_complete(df_1m: pl.DataFrame, bar: str) -> pl.DataFrame:
    """
    Aggregate 1-minute candles into complete bars of size `bar` (§6.1).

    Bucket k covers [k*B, (k+1)*B) in UTC ms and is stamped with its start.
    OHLCV are first/max/min/last/sum (and volume_ccy is summed when present);
    a bucket with fewer than B/1min one-minute bars is dropped, not filled.
    Works for perp and spot candles alike; the output is sorted by time.
    """
    missing = [c for c in ("timestamp", *_OHLCV) if c not in df_1m.columns]
    if missing:
        raise ValueError(f"1-minute candles lack column(s) {missing}")

    period = bar_ms(bar)
    need = period // MINUTE_MS
    value_cols = list(_OHLCV) + (["volume_ccy"] if "volume_ccy" in df_1m.columns else [])
    schema = {"timestamp": pl.Int64, **{c: pl.Float64 for c in value_cols}}
    if df_1m.height == 0:
        return pl.DataFrame(schema=schema)

    ts = _as_int_ms(df_1m["timestamp"].to_numpy())
    if np.any(ts % MINUTE_MS != 0):
        raise ValueError("1-minute timestamps must fall on whole minutes")

    df = pl.DataFrame(
        [pl.Series("timestamp", ts, dtype=pl.Int64)]
        + [df_1m[c].cast(pl.Float64).alias(c) for c in value_cols]
    ).sort("timestamp")

    # Counting rows only proves completeness if each minute appears once; a
    # duplicate would also make first/last depend on row order.
    if np.any(np.diff(df["timestamp"].to_numpy()) == 0):
        raise ValueError("duplicate 1-minute timestamps; de-duplicate before resampling")
    # max/min/sum over a NaN or null is ill-defined; corrupt rows must be
    # removed (and so counted as missing) upstream, not aggregated here.
    if not np.isfinite(df.select(value_cols).to_numpy()).all():
        raise ValueError("1-minute candles contain null or non-finite values")

    aggs = [
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ]
    if "volume_ccy" in value_cols:
        aggs.append(pl.col("volume_ccy").sum())
    aggs.append(pl.col("timestamp").count().alias("_n"))

    out = (
        df.with_columns((pl.col("timestamp") // period * period).alias("_bucket"))
        .group_by("_bucket", maintain_order=True)
        .agg(aggs)
        .filter(pl.col("_n") == need)
        .drop("_n")
        .rename({"_bucket": "timestamp"})
        .sort("timestamp")
    )
    return out.select([pl.col(c).cast(t) for c, t in schema.items()])


def to_grid(df: pl.DataFrame, ref_ts: np.ndarray) -> pl.DataFrame:
    """
    Put `df` on exactly the timestamps `ref_ts` (sorted), NaN where df has no row.

    Used to place the spot leg on the perp bar grid: a spot bucket that was
    dropped then shows up as a NaN close at that bar, so an as-of join cannot
    carry the previous spot close forward into it (§5.1, basis_z_score).
    Rows of df whose timestamp is not in ref_ts are discarded.
    """
    if "timestamp" not in df.columns:
        raise ValueError("df has no 'timestamp' column")
    ref = _as_int_ms(ref_ts).ravel()
    if np.any(np.diff(ref) < 0):
        raise ValueError("ref_ts must be sorted ascending")

    src_ts = _as_int_ms(df["timestamp"].to_numpy())
    order = np.argsort(src_ts, kind="stable")
    src_ts = src_ts[order]
    if np.any(np.diff(src_ts) == 0):
        raise ValueError("df has duplicate timestamps")

    if len(src_ts):
        pos = np.minimum(np.searchsorted(src_ts, ref), len(src_ts) - 1)
        hit = src_ts[pos] == ref
    else:
        pos = np.zeros(len(ref), dtype=np.int64)
        hit = np.zeros(len(ref), dtype=bool)

    columns = [pl.Series("timestamp", ref, dtype=pl.Int64)]
    for c in df.columns:
        if c == "timestamp":
            continue
        src = df[c].cast(pl.Float64).to_numpy()[order]
        col = np.full(len(ref), np.nan)
        col[hit] = src[pos[hit]]
        columns.append(pl.Series(c, col, dtype=pl.Float64))
    return pl.DataFrame(columns)


def htf_source_index(
    bar_ts: np.ndarray, bar_ms: int, htf_ts: np.ndarray, htf_ms: int
) -> np.ndarray:
    """
    For each analysis bar, the index into htf_ts of the HTF bar it may use, or -1.

    A bar starting at s closes at s + B. The latest HTF bar already closed
    then starts at or before floor((s + B) / HTF) * HTF - HTF (§5.2); if that
    bucket was dropped, the latest existing one before it is used. Because
    HTF starts are multiples of HTF, this equals "the latest HTF bar whose
    close time start + HTF is <= s + B". Mapping an HTF bar onto the base bars
    inside it instead would be the look-ahead §5.2 removes.
    """
    bar_ts = _as_int_ms(bar_ts).ravel()
    htf_ts = _as_int_ms(htf_ts).ravel()
    if np.any(np.diff(htf_ts) <= 0):
        raise ValueError("htf_ts must be strictly increasing")
    bar_ms, htf_ms = int(bar_ms), int(htf_ms)
    target = np.floor_divide(bar_ts + bar_ms, htf_ms) * htf_ms - htf_ms
    return (np.searchsorted(htf_ts, target, side="right") - 1).astype(np.int64)
