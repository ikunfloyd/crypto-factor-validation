"""
Tests for research/bars.py: complete-bucket resampling (§6.1), the spot-leg
grid join, and the higher-timeframe mapping of triple_screen (§5.2).

Synthetic candles only; nothing here reads market data.
"""
import numpy as np
import polars as pl
import pytest

from research.bars import bar_ms, htf_bar, htf_source_index, resample_complete, to_grid
from research.spec import BAR_MS, MINUTE_MS

H = 3_600_000
# A UTC midnight, so 15m, 1H and 4H buckets all start on it.
DAY0 = (1_700_000_000_000 // (24 * H)) * 24 * H


def _minutes(start_ms, n, seed=0, with_ccy=True):
    rng = np.random.default_rng(seed)
    ts = start_ms + np.arange(n, dtype=np.int64) * MINUTE_MS
    close = 0.1 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 1e-3, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 1e-3, n))
    cols = {"timestamp": ts, "open": open_, "high": high, "low": low,
            "close": close, "volume": rng.uniform(1, 10, n)}
    if with_ccy:
        cols["volume_ccy"] = cols["volume"] * close
    return pl.DataFrame(cols)


def _drop_minutes(df, ts_list):
    return df.filter(~pl.col("timestamp").is_in(list(ts_list)))


# ---------------------------------------------------------------- bar labels

def test_bar_ms_matches_spec_and_htf_is_4x():
    assert bar_ms("15m") == BAR_MS["15m"]
    assert bar_ms("1H") == BAR_MS["1H"]
    assert bar_ms("4H") == 4 * BAR_MS["1H"]
    assert htf_bar("15m") == "1H"
    assert htf_bar("1H") == "4H"
    for bad in ("15", "0m", "1D", "h1"):
        with pytest.raises(ValueError):
            bar_ms(bad)


# ---------------------------------------------------------------- resample_complete

def test_resample_drops_incomplete_bucket_and_aggregates_ohlcv():
    df = _minutes(DAY0, 45)
    gap = DAY0 + 20 * MINUTE_MS  # inside the second 15m bucket
    out = resample_complete(_drop_minutes(df, [gap]), "15m")

    B = 15 * MINUTE_MS
    assert out["timestamp"].to_list() == [DAY0, DAY0 + 2 * B]
    assert out.columns == ["timestamp", "open", "high", "low", "close", "volume", "volume_ccy"]
    assert out["timestamp"].dtype == pl.Int64

    for row, k in zip(out.iter_rows(named=True), (0, 2)):
        part = df[15 * k: 15 * (k + 1)]
        assert row["open"] == part["open"][0]
        assert row["high"] == part["high"].max()
        assert row["low"] == part["low"].min()
        assert row["close"] == part["close"][-1]
        assert row["volume"] == pytest.approx(part["volume"].sum(), rel=1e-12)
        assert row["volume_ccy"] == pytest.approx(part["volume_ccy"].sum(), rel=1e-12)


def test_resample_htf_needs_every_minute():
    df = _minutes(DAY0, 8 * 60)
    gap = DAY0 + 5 * H + 37 * MINUTE_MS  # hour 5, so the second 4H bucket
    holed = _drop_minutes(df, [gap])

    one_h = resample_complete(holed, "1H")
    assert one_h["timestamp"].to_list() == [DAY0 + k * H for k in range(8) if k != 5]

    four_h = resample_complete(holed, "4H")
    assert four_h["timestamp"].to_list() == [DAY0]
    first = df[:240]
    assert four_h["high"][0] == first["high"].max()
    assert four_h["close"][0] == first["close"][-1]


def test_resample_buckets_are_epoch_aligned():
    # Data starting at 10:07 leaves the 10:00 bucket incomplete: it is
    # dropped, not re-anchored to the first minute.
    start = DAY0 + 10 * H + 7 * MINUTE_MS
    out = resample_complete(_minutes(start, 60), "15m")
    assert out["timestamp"].to_list() == [DAY0 + 10 * H + k * 15 * MINUTE_MS for k in (1, 2, 3)]


def test_resample_spot_candles_without_volume_ccy_and_unsorted_input():
    spot = _minutes(DAY0, 120, seed=3, with_ccy=False)
    shuffled = spot.sample(fraction=1.0, shuffle=True, seed=7)
    a = resample_complete(spot, "15m")
    b = resample_complete(shuffled, "15m")
    assert "volume_ccy" not in a.columns
    assert a.height == 8
    assert a.equals(b)


def test_resample_rejects_duplicates_misaligned_and_nonfinite():
    df = _minutes(DAY0, 30)
    with pytest.raises(ValueError):
        resample_complete(pl.concat([df, df[:1]]), "15m")
    with pytest.raises(ValueError):
        resample_complete(df.with_columns(pl.col("timestamp") + 1), "15m")
    with pytest.raises(ValueError):
        resample_complete(df.with_columns(pl.Series("close", np.r_[np.nan, df["close"][1:]])), "15m")
    with pytest.raises(ValueError):
        resample_complete(df.drop("volume"), "15m")


def test_resample_empty():
    out = resample_complete(_minutes(DAY0, 5)[:0], "1H")
    assert out.height == 0
    assert out.columns[0] == "timestamp"


# ---------------------------------------------------------------- to_grid

def test_to_grid_nan_where_missing_and_ignores_extra_rows():
    B = 15 * MINUTE_MS
    ref = DAY0 + np.arange(6, dtype=np.int64) * B
    src = pl.DataFrame({
        "timestamp": [ref[4], ref[0], ref[1], ref[3], ref[5] + B],  # unsorted, ref[2] missing, one extra
        "close": [4.0, 0.0, 1.0, 3.0, 99.0],
    })
    out = to_grid(src, ref)
    assert out["timestamp"].to_list() == ref.tolist()
    assert out["timestamp"].dtype == pl.Int64
    close = out["close"].to_numpy()
    np.testing.assert_array_equal(close, [0.0, 1.0, np.nan, 3.0, 4.0, np.nan])
    assert out["close"].null_count() == 0  # NaN, not null: the engine reads NaN

    with pytest.raises(ValueError):
        to_grid(src, ref[::-1])
    with pytest.raises(ValueError):
        to_grid(pl.concat([src, src[:1]]), ref)


def test_to_grid_empty_source():
    ref = DAY0 + np.arange(3, dtype=np.int64) * H
    out = to_grid(pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Int64),
                                "close": pl.Series([], dtype=pl.Float64)}), ref)
    assert out.height == 3 and np.isnan(out["close"].to_numpy()).all()


# ---------------------------------------------------------------- htf_source_index

def _brute(bar_ts, B, htf_ts, HTF):
    """Latest HTF bar whose close (start + HTF) is at or before the analysis bar's close."""
    out = np.full(len(bar_ts), -1, dtype=np.int64)
    for i, s in enumerate(bar_ts):
        for j, h in enumerate(htf_ts):
            if h + HTF <= s + B:
                out[i] = j
    return out


def test_htf_examples_from_section_5_2():
    q = 15 * MINUTE_MS
    htf_1h = DAY0 + np.array([8, 9, 10, 11]) * H
    idx = htf_source_index(np.array([DAY0 + 10 * H + 30 * MINUTE_MS]), q, htf_1h, H)
    assert htf_1h[idx[0]] == DAY0 + 9 * H  # 15m bar 10:30 -> 1H bar 09:00

    # 10:45 closes at 11:00, exactly when the 10:00 1H bar closes.
    idx = htf_source_index(np.array([DAY0 + 10 * H + 45 * MINUTE_MS]), q, htf_1h, H)
    assert htf_1h[idx[0]] == DAY0 + 10 * H

    htf_4h = DAY0 + np.array([0, 4, 8, 12]) * H
    idx = htf_source_index(np.array([DAY0 + 10 * H, DAY0 + 11 * H]), H, htf_4h, 4 * H)
    assert htf_4h[idx[0]] == DAY0 + 4 * H   # 1H bar 10:00 -> 4H bar 04:00
    assert htf_4h[idx[1]] == DAY0 + 8 * H   # 1H bar 11:00 closes 12:00 -> 4H 08:00


def test_htf_falls_back_to_latest_existing_and_minus_one():
    htf = DAY0 + np.array([4, 6]) * H  # 1H bars; 07:00 bucket was dropped
    bars = DAY0 + np.array([3 * H, 7 * H + 45 * MINUTE_MS, 8 * H + 45 * MINUTE_MS])
    idx = htf_source_index(bars, 15 * MINUTE_MS, htf, H)
    assert idx.tolist() == [-1, 1, 1]
    assert htf_source_index(bars, 15 * MINUTE_MS, np.array([], dtype=np.int64), H).tolist() == [-1, -1, -1]


@pytest.mark.parametrize("B,HTF", [(15 * MINUTE_MS, H), (H, 4 * H)])
def test_htf_matches_brute_force_with_random_gaps(B, HTF):
    rng = np.random.default_rng(11)
    n_htf = 120
    htf_grid = DAY0 + np.arange(n_htf, dtype=np.int64) * HTF
    htf_ts = htf_grid[rng.random(n_htf) < 0.7]
    per = HTF // B
    bar_grid = DAY0 - 3 * HTF + np.arange((n_htf + 6) * per, dtype=np.int64) * B
    bar_ts = bar_grid[rng.random(len(bar_grid)) < 0.6]
    np.testing.assert_array_equal(
        htf_source_index(bar_ts, B, htf_ts, HTF), _brute(bar_ts, B, htf_ts, HTF)
    )


def test_htf_bar_never_uses_an_unclosed_bar():
    rng = np.random.default_rng(5)
    B, HTF = 15 * MINUTE_MS, H
    htf_ts = DAY0 + np.sort(rng.choice(200, 150, replace=False)).astype(np.int64) * HTF
    bar_ts = DAY0 + np.arange(200 * 4, dtype=np.int64) * B
    idx = htf_source_index(bar_ts, B, htf_ts, HTF)
    used = idx >= 0
    assert np.all(htf_ts[idx[used]] + HTF <= bar_ts[used] + B)
