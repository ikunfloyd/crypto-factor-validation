"""
Synthetic order-book days for the maker-probe engine (research/maker_probe.py):
its property tests and its benchmark.

The engine was written before any of the probe's archive files was
downloaded (pre-registration §12, 2026-10-12), so every check of it runs on
days made here. A day is generated from a "true" book that the generator keeps
itself, so the stream is a real book's history rather than noise, with the
irregularities the engine has to survive:

  * trades at the touch, each followed by the book decrease it causes, and
    sweeps that empty the best level and print part of the next;
  * many events sharing a millisecond, and book rows that arrive a few
    milliseconds late (``p_jitter``), so a trade can precede the book
    decrease it causes, and the rebuilt book is briefly crossed;
  * a snapshot every ``snap_every_ms`` (both sides reset at a few ms past
    the mark, i.e. during the life of the orders posted at that mark), and,
    with ``snap_noise``, snapshots that differ from the book rebuilt before
    them.

Nothing here calibrates to DOGE; the sizes only have to exercise the rules.

Benchmark (one day of about 10 million book rows and 1 million trades):
    PYTHONPATH=. python tests/maker_probe_synthetic.py --benchmark
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import date
from typing import Tuple

import numpy as np
from numba import njit

from research import maker_probe as probe
from research.maker_probe import ASK, BID, BookStream, TradeStream, _apply_level

# A day inside the in-sample year, so research.spec's protected-window guard
# lets it through (any day outside the test windows would do).
DEFAULT_DAY = date(2026, 1, 5)


@njit(fastmath=False, cache=False)
def _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, t, s, p, z, reset, g, p_jitter, jitter_ms):
    """Append one book row to message g (messages are numbered by their first row, so msg never decreases).
    A late row (p_jitter) is stamped up to jitter_ms after t; no row is stamped before the previous one, so
    the book stream lags the trades but keeps its own order, as a sequenced feed does (a reordered "new
    total size" row would leave stale levels behind). Only single-row messages are jittered: a snapshot's
    rows keep one timestamp."""
    ts = t
    if p_jitter > 0.0 and not reset and np.random.random() < p_jitter:
        ts = t + np.random.randint(1, jitter_ms + 1)
    if nb > 0 and ts < ob_ts[nb - 1]:
        ts = ob_ts[nb - 1]
    ob_ts[nb] = ts
    ob_side[nb] = BID if s == 0 else ASK
    ob_px[nb] = p
    ob_sz[nb] = z
    ob_rs[nb] = reset
    ob_msg[nb] = g
    return nb + 1


@njit(fastmath=False, cache=False)
def _generate(seed, t_lo, t_hi, n_book, n_trade, width, depth0, size_max, mean_offset, max_offset,
              max_levels, min_levels, p_inside, p_delete, p_sweep, p_jitter, jitter_ms,
              snap_every_ms, snap_noise):
    np.random.seed(seed)
    tb = np.sort(np.random.randint(t_lo, t_hi, n_book))
    tt = np.sort(np.random.randint(t_lo, t_hi, n_trade))
    n_snaps = (t_hi - t_lo) // snap_every_ms + 2
    cap_b = n_book + 4 * n_trade + (n_snaps + 1) * (2 * max_levels + 4) + 4 * depth0 + 16
    cap_t = 2 * n_trade + 4
    ob_ts = np.empty(cap_b, np.int64)
    ob_side = np.empty(cap_b, np.int8)
    ob_px = np.empty(cap_b, np.int64)
    ob_sz = np.empty(cap_b, np.int64)
    ob_rs = np.empty(cap_b, np.bool_)
    ob_msg = np.empty(cap_b, np.int64)
    ot_ts = np.empty(cap_t, np.int64)
    ot_px = np.empty(cap_t, np.int64)
    ot_sz = np.empty(cap_t, np.int64)
    ot_ag = np.empty(cap_t, np.int8)
    nb = 0
    nt = 0

    levels = np.zeros((2, width), np.int64)
    best = np.full(2, -1, np.int64)
    nlev = np.zeros(2, np.int64)
    lo = np.full(2, width, np.int64)
    hi = np.full(2, -1, np.int64)
    mid = width // 2

    # Initial snapshot, 1 s before the window.
    t_init = t_lo - 1000
    for s in range(2):
        for k in range(depth0):
            p = mid - 1 - k if s == 0 else mid + 1 + k
            z = np.random.randint(1, size_max + 1)
            _apply_level(levels, best, nlev, lo, hi, s, p, z)
            nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, t_init, s, p, z, k == 0, 0, 0.0, 0)

    next_snap = (t_lo // snap_every_ms + 1) * snap_every_ms + np.random.randint(0, 50)
    ib = 0
    it = 0
    while ib < n_book or it < n_trade:
        is_trade = it < n_trade and (ib >= n_book or tt[it] < tb[ib])
        t = tt[it] if is_trade else tb[ib]

        while next_snap <= t:
            if snap_noise > 0.0 and np.random.random() < snap_noise:
                s = np.random.randint(0, 2)
                k = best[s]
                if k >= 0:
                    _apply_level(levels, best, nlev, lo, hi, s, k, np.random.randint(1, size_max + 1))
            g = nb
            for s in range(2):
                first = True
                for p in range(lo[s], hi[s] + 1):
                    if levels[s, p] > 0:
                        nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, next_snap, s, p,
                                        levels[s, p], first, g, 0.0, 0)
                        first = False
                if first:
                    nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, next_snap, s, 1, 0, True, g,
                                    0.0, 0)
            next_snap = (next_snap // snap_every_ms + 1) * snap_every_ms + np.random.randint(0, 50)

        if is_trade:
            it += 1
            aggr = 1 if np.random.random() < 0.5 else -1
            s = 1 if aggr == 1 else 0          # a buyer lifts the asks, a seller hits the bids
            p = best[s]
            if p < 0:
                continue
            lv = levels[s, p]
            if np.random.random() < p_sweep and nlev[s] > min_levels + 1:
                ot_ts[nt] = t
                ot_px[nt] = p
                ot_sz[nt] = lv
                ot_ag[nt] = aggr
                nt += 1
                _apply_level(levels, best, nlev, lo, hi, s, p, 0)
                nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, t, s, p, 0, False, nb,
                                p_jitter, jitter_ms)
                p = best[s]
                lv = levels[s, p]
                part = np.random.randint(1, lv + 1)
                ot_ts[nt] = t
                ot_px[nt] = p
                ot_sz[nt] = part
                ot_ag[nt] = aggr
                nt += 1
                _apply_level(levels, best, nlev, lo, hi, s, p, lv - part)
                nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, t, s, p, lv - part, False, nb,
                                p_jitter, jitter_ms)
            else:
                z = np.random.randint(1, lv + 1)
                ot_ts[nt] = t
                ot_px[nt] = p
                ot_sz[nt] = z
                ot_ag[nt] = aggr
                nt += 1
                _apply_level(levels, best, nlev, lo, hi, s, p, lv - z)
                nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, t, s, p, lv - z, False, nb,
                                p_jitter, jitter_ms)
            if nlev[s] < min_levels:
                # Refill beyond the far end so a side never empties.
                p = lo[0] - 1 if s == 0 else hi[1] + 1
                if 1 <= p <= width - 2:
                    z = np.random.randint(1, size_max + 1)
                    _apply_level(levels, best, nlev, lo, hi, s, p, z)
                    nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, t, s, p, z, False, nb,
                                    p_jitter, jitter_ms)
            continue

        ib += 1
        s = np.random.randint(0, 2)
        bb = best[0]
        ba = best[1]
        if bb < 0 or ba < 0:
            continue
        if np.random.random() < p_inside and ba - bb > 1:
            p = np.random.randint(bb + 1, ba)
            z = np.random.randint(1, size_max + 1)
        else:
            k = int(-math.log(1.0 - np.random.random()) * mean_offset)
            if k > max_offset:
                k = max_offset
            p = bb - k if s == 0 else ba + k
            if p < 1 or p > width - 2:
                continue
            if levels[s, p] > 0 and nlev[s] > min_levels and np.random.random() < p_delete:
                z = 0
            else:
                z = np.random.randint(1, size_max + 1)
        _apply_level(levels, best, nlev, lo, hi, s, p, z)
        nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, t, s, p, z, False, nb,
                        p_jitter, jitter_ms)
        while nlev[s] > max_levels:
            # Drop the deepest level of the side.
            if s == 0:
                q = lo[0]
                while levels[0, q] == 0:
                    q += 1
                lo[0] = q
            else:
                q = hi[1]
                while levels[1, q] == 0:
                    q -= 1
                hi[1] = q
            _apply_level(levels, best, nlev, lo, hi, s, q, 0)
            nb = _emit_book(ob_ts, ob_side, ob_px, ob_sz, ob_rs, ob_msg, nb, t, s, q, 0, False, nb,
                            p_jitter, jitter_ms)

    return (ob_ts[:nb], ob_side[:nb], ob_px[:nb], ob_sz[:nb], ob_rs[:nb], ob_msg[:nb],
            ot_ts[:nt], ot_px[:nt], ot_sz[:nt], ot_ag[:nt])


def synthetic_day(
    seed: int,
    *,
    day: date = DEFAULT_DAY,
    start_offset_ms: int = 0,
    duration_ms: int = probe.DAY_MS + 15 * probe.MINUTE_MS,
    n_book: int = 200_000,
    n_trade: int = 20_000,
    center_tick: int = 20_000,
    width: int = 8192,
    depth0: int = 40,
    size_max: int = 200,
    mean_offset: float = 4.0,
    max_offset: int = 60,
    max_levels: int = 120,
    min_levels: int = 5,
    p_inside: float = 0.15,
    p_delete: float = 0.35,
    p_sweep: float = 0.05,
    p_jitter: float = 0.2,
    jitter_ms: int = 5,
    snap_every_ms: int = 15 * probe.MINUTE_MS,
    snap_noise: float = 0.0,
) -> Tuple[BookStream, TradeStream]:
    """
    A synthetic (BookStream, TradeStream) pair covering
    [day 00:00 + start_offset_ms - 1 s, day 00:00 + start_offset_ms + duration_ms).

    ``n_book`` counts the book events drawn, not rows: every trade adds one
    or two decrease rows, refills and depth trimming add more, and each
    snapshot adds a row per level. Book rows that late stamping (``p_jitter``)
    pushes to or past the end are dropped. Prices are ticks around ``center_tick``.
    """
    t_lo = probe._day_start_ms(day) + int(start_offset_ms)
    t_hi = t_lo + int(duration_ms)
    out = _generate(np.int64(seed), np.int64(t_lo), np.int64(t_hi), np.int64(n_book), np.int64(n_trade),
                    np.int64(width), np.int64(depth0), np.int64(size_max), float(mean_offset), np.int64(max_offset),
                    np.int64(max_levels), np.int64(min_levels), float(p_inside), float(p_delete), float(p_sweep),
                    float(p_jitter), np.int64(jitter_ms), np.int64(snap_every_ms), float(snap_noise))
    b_ts, b_side, b_px, b_sz, b_rs, b_msg, t_ts, t_px, t_sz, t_ag = out
    keep = b_ts < t_hi                 # a message's rows share a timestamp, so none is cut in two
    shift = int(center_tick) - width // 2
    book = BookStream(b_ts[keep], b_side[keep], b_px[keep] + shift, b_sz[keep], b_rs[keep], b_msg[keep],
                      start_ms=t_lo - 1000, end_ms=t_hi, tick_size=1e-5, size_unit=1.0)
    trades = TradeStream(t_ts, t_px + shift, t_sz, t_ag, start_ms=t_lo - 1000, end_ms=t_hi)
    return book, trades


def mirror(book: BookStream, trades: TradeStream, center2: int) -> Tuple[BookStream, TradeStream]:
    """
    The day reflected through price center2 / 2: price p -> center2 - p,
    bids <-> asks, buyers <-> sellers. A buy in the original faces exactly
    what the mirrored sell faces, so the engine's sell results must equal its
    buy results, with prices reflected.
    """
    b = BookStream(book.ts_ms, -book.side, center2 - book.price, book.size, book.reset, book.msg,
                   start_ms=book.start_ms, end_ms=book.end_ms, tick_size=book.tick_size, size_unit=book.size_unit,
                   gaps=book.gaps)
    t = TradeStream(trades.ts_ms, center2 - trades.price, trades.size, -trades.aggressor,
                    start_ms=trades.start_ms, end_ms=trades.end_ms, gaps=trades.gaps)
    return b, t


def benchmark(seed: int = 1, n_book: int = 8_900_000, n_trade: int = 1_000_000) -> None:
    """Generate one full synthetic day of about 10M book rows and 1M trades and time ``simulate_day`` on it."""
    cfg = probe.ProbeConfig.from_spec()
    start, end = probe.required_coverage(DEFAULT_DAY, cfg)
    # Compile both kernels on a small day first, so the timings below are run time only.
    b, t = synthetic_day(seed, n_book=2000, n_trade=200)
    probe.simulate_day(b, t, DEFAULT_DAY, cfg)

    t0 = time.perf_counter()
    book, trades = synthetic_day(seed, duration_ms=end - start, n_book=n_book, n_trade=n_trade)
    t1 = time.perf_counter()
    orders = probe.simulate_day(book, trades, DEFAULT_DAY, cfg)
    t2 = time.perf_counter()
    check = probe.check_snapshots(book, depth=400)
    t3 = time.perf_counter()
    summary = probe.daily_summary(orders, cfg)
    print(f"book rows {len(book):,}  trades {len(trades):,}  snapshots checked {check.n_checked}")
    print(f"generate {t1 - t0:.1f} s  simulate_day {t2 - t1:.2f} s  check_snapshots {t3 - t2:.2f} s")
    print(summary.select(["side", "model", "n_decisions", "n_fills", "fill_rate",
                          "markout_1m_bps_mean", "shortfall_diff_bps_mean"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", action="store_true", help="time one synthetic day of ~10M book rows, ~1M trades")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.benchmark:
        benchmark(args.seed)
    else:
        ap.print_help()
