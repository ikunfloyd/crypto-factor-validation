"""
Tests for research/maker_probe.py, the maker-probe engine (pre-registration §11).

All data is synthetic: the engine was written and tested before any of the
probe's archive files was downloaded (§12, 2026-10-12). Three kinds of test:

  * hand-built scenarios with every expected fill, fill time, markout and
    shortfall written out from the book they are given;
  * a slow, independent reference (dict books, each order replayed on its
    own) that must agree with the Numba engine field by field on random
    synthetic days with gaps in both streams, including days whose
    timestamps are rounded to 100 ms so that book rows, trades, decision
    times, expiries and gap edges collide; on adversarial streams with no
    market behind them (several messages per millisecond, emptied sides,
    crossed books); and on short queue scenarios built so that snapshots
    lower a level while the queue ahead is partly used up;
  * properties that hold whatever the day: every M1 fill is an M2 fill no
    later, M2 fills at least as often, the queue never exceeds its level,
    both models' shortfalls cover the same decisions, and a day reflected
    through a price (bids <-> asks) gives the sell side the buy side's
    results.
"""
from __future__ import annotations

import bisect
import dataclasses
import math
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import pytest

import maker_probe_synthetic as synthetic
from research import maker_probe as probe
from research.maker_probe import ASK, BID, BUY, SELL, BookStream, ProbeConfig, TradeStream
from research.spec import get_spec

CFG = ProbeConfig.from_spec()
DAY = date(2026, 1, 5)
T0 = probe._day_start_ms(DAY)
S = 1000
MIN = 60_000
LIFE = CFG.order_life_ms
END = T0 + 30 * MIN
M0 = 10000.5                     # mid of the initial book below


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def initial(ts: int = T0 - 5 * S) -> List[Tuple]:
    """Bids 10000 x 10, 9999 x 5; asks 10001 x 8, 10002 x 4 (ticks, size units): one snapshot message."""
    return [(ts, BID, 10000, 10, True), (ts, BID, 9999, 5, False),
            (ts, ASK, 10001, 8, True), (ts, ASK, 10002, 4, False)]


def make_book(rows: Sequence[Tuple], start: int = T0 - 10 * S, end: int = END, gaps: Sequence = ()) -> BookStream:
    """
    Rows (ts, side, price, size[, reset[, part]]), stably sorted by ts, so
    rows of one ms keep the given order. ``part`` (default 0) numbers the
    messages inside a millisecond: by default every millisecond is one
    message.
    """
    full = [tuple(r) + (False, 0)[len(r) - 4:] for r in rows]
    full.sort(key=lambda r: r[0])
    if not full:
        z = np.zeros(0, np.int64)
        return BookStream(z, z, z, z, np.zeros(0, bool), z, start, end, 1e-5, 1.0, gaps=gaps)
    keys = [(r[0], r[5]) for r in full]
    rank = {k: n for n, k in enumerate(sorted(set(keys)))}
    ts, side, px, sz, rs, _ = (np.array(c) for c in zip(*full))
    return BookStream(ts.astype(np.int64), side.astype(np.int64), px.astype(np.int64), sz.astype(np.int64),
                      rs.astype(bool), np.array([rank[k] for k in keys], np.int64), start, end, 1e-5, 1.0,
                      gaps=gaps)


def make_trades(rows: Sequence[Tuple], start: int = T0 - 10 * S, end: int = END, gaps: Sequence = ()) -> TradeStream:
    """Rows (ts, price, size, aggressor), stably sorted by ts."""
    full = sorted(rows, key=lambda r: r[0])
    if not full:
        z = np.zeros(0, np.int64)
        return TradeStream(z, z, z, z, start, end, gaps=gaps)
    ts, px, sz, ag = (np.array(c, dtype=np.int64) for c in zip(*full))
    return TradeStream(ts, px, sz, ag, start, end, gaps=gaps)


def run(book_rows: Sequence[Tuple] = (), trade_rows: Sequence[Tuple] = (), decisions: Sequence[int] = (T0,),
        end: int = END, base: Optional[List[Tuple]] = None, day: str = "d", book_gaps: Sequence = (),
        trade_gaps: Sequence = ()) -> pl.DataFrame:
    book = make_book((initial() if base is None else base) + list(book_rows), end=end, gaps=book_gaps)
    return probe.simulate(book, make_trades(trade_rows, end=end, gaps=trade_gaps), list(decisions), CFG, day=day)


def get(df: pl.DataFrame, side: str, model: str, t0: int = T0) -> Dict[str, Any]:
    out = df.filter((pl.col("side") == side) & (pl.col("model") == model) & (pl.col("decision_ts") == t0))
    assert out.height == 1
    return out.row(0, named=True)


def bps_buy(mid: float, px: float) -> float:
    """(mid / px - 1) * 1e4, §11's buy markout."""
    return (mid / px - 1.0) * 1e4


def bps_sell(mid: float, px: float) -> float:
    """(1 - mid / px) * 1e4, §11's sell markout."""
    return (1.0 - mid / px) * 1e4


# The engine computes (mid - px) / px, which rounds differently from
# mid / px - 1 in the last bits. A shortfall can nearly cancel against its
# fee (5 bp - 4.9992 bp), and mid / px - 1 alone then carries about 1e-12 bp
# of rounding error, so values compared on random days get an absolute
# tolerance of a billionth of a basis point.
approx = lambda x: pytest.approx(x, rel=1e-9, abs=1e-12)  # noqa: E731
approx_bp = lambda x: pytest.approx(x, rel=1e-9, abs=1e-9)  # noqa: E731

CROSS_BUY = bps_buy(10001, M0) + 5.0     # buy at the ask now:  +0.49997500 + 5
CROSS_SELL = bps_sell(10000, M0) + 5.0   # sell at the bid now: +0.49997500 + 5
POST_BUY = bps_buy(10000, M0) + 2.0      # rest at the bid:     -0.49997500 + 2
POST_SELL = bps_sell(10001, M0) + 2.0    # rest at the ask:     -0.49997500 + 2


def daily_row(df: pl.DataFrame, side: str, model: str) -> Dict[str, Any]:
    out = probe.daily_summary(df, CFG).filter((pl.col("side") == side) & (pl.col("model") == model))
    assert out.height == 1
    return out.row(0, named=True)


def assert_unfilled(r: Dict[str, Any]) -> None:
    assert r["filled"] is False and r["fill_ts"] is None
    for lab in CFG.markout_labels:
        assert r[f"markout_{lab}_bps"] is None and r[f"markout_{lab}_status"] == "no_fill"


# --------------------------------------------------------------------------
# spec and contract
# --------------------------------------------------------------------------

class TestSpec:
    def test_config_matches_appendix_a(self):
        mp = get_spec()["maker_probe"]
        assert CFG.decision_every_ms == mp["decision_every_minutes"] * MIN == 300_000
        assert CFG.order_life_ms == mp["order_life_seconds"] * 1000 == 60_000
        assert CFG.size_contracts == mp["size_contracts"] == 1
        assert CFG.markout_ms == tuple(m * MIN for m in mp["markout_minutes"]) == (60_000, 300_000, 900_000)
        fees = mp["fees_bps"]
        assert (CFG.maker_fee_bps, CFG.taker_fee_bps) == (fees["maker"], fees["taker"]) == (2.0, 5.0)
        assert CFG.fees_verified is False and CFG.n_days == 14
        assert CFG.markout_labels == ("1m", "5m", "15m")

    def test_refuses_other_fill_models_and_sizes(self):
        spec = get_spec()
        bad = {**spec, "maker_probe": {**spec["maker_probe"], "fill_models": ["M1_trade_through"]}}
        with pytest.raises(ValueError, match="fill_models"):
            ProbeConfig.from_spec(bad)
        bad = {**spec, "maker_probe": {**spec["maker_probe"], "size_contracts": 2}}
        with pytest.raises(ValueError, match="size_contracts"):
            ProbeConfig.from_spec(bad)

    def test_decision_grid_and_coverage(self):
        dec = probe.decision_times("2026-01-05", CFG)
        assert dec.size == 288 and dec[0] == T0 and np.all(np.diff(dec) == 300_000)
        assert dec[-1] == T0 + 86_400_000 - 300_000
        lo, hi = probe.required_coverage(DAY, CFG)
        assert lo == T0 and hi == T0 + 86_400_000 + 11 * MIN + 1      # D+1 00:11:00.001

    def test_parsers_are_stubs(self):
        with pytest.raises(NotImplementedError):
            probe.read_book([], DAY)
        with pytest.raises(NotImplementedError):
            probe.read_trades([], DAY)


class TestContract:
    def test_rejects_unsorted_and_out_of_coverage(self):
        with pytest.raises(ValueError, match="never re-sorts"):
            BookStream(np.array([T0 + 2, T0 + 1]), np.array([BID, BID]), np.array([1, 1]), np.array([1, 1]),
                       np.array([True, False]), np.array([0, 1]), T0, END, 1e-5, 1.0)
        with pytest.raises(ValueError, match="coverage"):
            make_trades([(END, 10000, 1, BUY)])

    def test_rejects_bad_values(self):
        with pytest.raises(ValueError, match="side"):
            make_book([(T0, 0, 10000, 1, True)])
        with pytest.raises(TypeError, match="integers"):
            BookStream(np.array([T0]), np.array([BID]), np.array([0.1]), np.array([1]), np.array([True]),
                       np.array([0]), T0, END, 1e-5, 1.0)
        with pytest.raises(ValueError, match="positive"):
            make_trades([(T0, 10000, 0, BUY)])
        with pytest.raises(ValueError, match="aggressor"):
            make_trades([(T0, 10000, 1, 0)])

    def test_message_rules(self):
        def book(ts, side, px, rs, msg):
            n = len(ts)
            return BookStream(np.array(ts), np.array(side), np.array(px), np.ones(n, np.int64), np.array(rs),
                              np.array(msg), T0 - S, END, 1e-5, 1.0)
        with pytest.raises(ValueError, match="msg must be non-decreasing"):
            book([T0, T0], [BID, ASK], [10000, 10001], [True, True], [1, 0])
        with pytest.raises(ValueError, match="share one timestamp"):
            book([T0, T0 + 1], [BID, ASK], [10000, 10001], [True, True], [0, 0])
        with pytest.raises(ValueError, match="first row"):
            book([T0, T0], [BID, BID], [10000, 9999], [False, True], [0, 0])
        with pytest.raises(ValueError, match="at most once"):
            book([T0, T0], [ASK, ASK], [10001, 10001], [False, False], [0, 0])
        # The same price on both sides, or twice in two messages of one ms, is fine.
        book([T0, T0, T0], [BID, ASK, ASK], [10000, 10000, 10000], [True, True, False], [0, 0, 1])

    def test_gap_rules(self):
        snap = initial(T0 + 2 * MIN)
        make_book(initial() + snap, gaps=[(T0 + MIN, T0 + 2 * MIN)])
        with pytest.raises(ValueError, match="does not end at a snapshot"):
            make_book(initial() + snap, gaps=[(T0 + MIN, T0 + 3 * MIN)])
        with pytest.raises(ValueError, match="no ask reset row"):
            make_book(initial() + snap[:2], gaps=[(T0 + MIN, T0 + 2 * MIN)])
        with pytest.raises(ValueError, match="sorted and disjoint"):
            make_trades([], gaps=[(T0 + 2 * S, T0 + 4 * S), (T0 + 3 * S, T0 + 5 * S)])
        with pytest.raises(ValueError, match="inside the coverage"):
            make_trades([], gaps=[(END - S, END + S)])
        with pytest.raises(ValueError, match="lo < hi"):
            make_trades([], gaps=[(T0, T0)])
        with pytest.raises(TypeError, match="pair of integer"):
            make_trades([], gaps=[(T0, T0 + 0.5)])

    def test_rejects_overlapping_orders(self):
        with pytest.raises(ValueError, match="order life"):
            run(decisions=[T0, T0 + LIFE - 1])

    def test_refuses_protected_windows(self):
        t = probe._day_start_ms(date(2024, 1, 10))          # inside doge_2023
        book = make_book([(t, BID, 10000, 1, True), (t, ASK, 10001, 1, True)], start=t, end=t + 30 * MIN)
        with pytest.raises(ValueError, match="protected"):
            probe.simulate(book, make_trades([], start=t, end=t + 30 * MIN), [t + S], CFG)


# --------------------------------------------------------------------------
# hand-built scenarios
# --------------------------------------------------------------------------

class TestM1:
    def test_trade_through_fills_with_markouts_and_shortfall(self):
        f = T0 + 10 * S
        df = run(
            book_rows=[
                (f + MIN, BID, 10000, 0),          # at exactly fill + 1 min: bid 9999 / ask 10001, mid 10000
                (f + 4 * MIN, ASK, 10001, 0),      # bid 9999 / ask 10002, mid 10000.5
                (f + 10 * MIN, BID, 10001, 3),     # bid 10001 / ask 10002, mid 10001.5
            ],
            trade_rows=[(f, 9999, 1, SELL)],
        )
        for model in ("M1", "M2"):
            r = get(df, "buy", model)
            assert r["status"] == "ok" and r["filled"] is True and r["fill_ts"] == f
            assert (r["best_bid"], r["best_ask"], r["order_price"], r["queue_at_post"]) == (10000, 10001, 10000, 10)
            # The 1-minute read includes the row stamped exactly then (state at or before).
            assert r["markout_1m_bps"] == approx(bps_buy(10000.0, 10000)) == 0.0
            assert r["markout_5m_bps"] == approx(bps_buy(10000.5, 10000)) == approx(0.5)
            assert r["markout_15m_bps"] == approx(bps_buy(10001.5, 10000)) == approx(1.5)
            assert r["shortfall_cross_now_bps"] == approx(CROSS_BUY) == approx(5.49997500125)
            assert r["shortfall_post_then_cross_bps"] == approx(POST_BUY) == approx(1.50002499875)
            assert r["shortfall_diff_bps"] == approx(POST_BUY - CROSS_BUY) == approx(-3.9999500025)
            # The sell rested at 10001 and nothing traded at or above it.
            s = get(df, "sell", model)
            assert_unfilled(s)
            assert s["expiry_best_bid"] == 10000
            assert s["shortfall_post_then_cross_bps"] == approx(bps_sell(10000, M0) + 5.0)
            assert s["shortfall_diff_bps"] == approx(0.0)
        assert get(df, "sell", "M2")["queue_left"] == 8

    def test_opposite_best_crossing(self):
        df = run(book_rows=[(T0 + 5 * S, ASK, 10000, 2)])          # an ask at the buy's price
        for model in ("M1", "M2"):
            assert get(df, "buy", model)["fill_ts"] == T0 + 5 * S
            assert_unfilled(get(df, "sell", model))                 # best bid 10000 < 10001
        df = run(book_rows=[(T0 + 5 * S, BID, 10003, 2)])          # a bid through the sell's price
        for model in ("M1", "M2"):
            assert get(df, "sell", model)["fill_ts"] == T0 + 5 * S
            assert_unfilled(get(df, "buy", model))

    def test_cross_between_two_messages_of_one_millisecond(self):
        # Message 0 puts an ask at 10000, message 1 in the same ms removes it: the end-of-ms book is not
        # crossed, so M1 (lower bound) does not fill, while M2 (upper bound) sees message 0's end state.
        df = run(book_rows=[(T0 + 5 * S, ASK, 10000, 2, False, 0), (T0 + 5 * S, ASK, 10000, 0, False, 1)])
        assert_unfilled(get(df, "buy", "M1"))
        r = get(df, "buy", "M2")
        assert r["fill_ts"] == T0 + 5 * S and (r["queue_left"], r["level_size_left"]) == (10, 10)
        for model in ("M1", "M2"):
            assert_unfilled(get(df, "sell", model))

    def test_trade_at_the_price_is_not_an_m1_fill(self):
        df = run(trade_rows=[(T0 + 5 * S, 10000, 3, SELL)])
        assert_unfilled(get(df, "buy", "M1"))
        r = get(df, "buy", "M2")
        assert_unfilled(r)
        assert (r["queue_left"], r["level_size_left"]) == (7, 10)
        assert get(df, "buy", "M1")["shortfall_post_then_cross_bps"] == approx(CROSS_BUY)   # crossed at 10001


class TestM2:
    def test_queue_depleted_by_trades(self):
        # Q: 10 -> 6 -> 0 (a trade of exactly Q does not reach the order) -> the next trade fills it.
        df = run(trade_rows=[(T0 + S, 10000, 4, SELL), (T0 + 2 * S, 10000, 6, SELL), (T0 + 3 * S, 10000, 1, SELL)])
        r = get(df, "buy", "M2")
        assert r["fill_ts"] == T0 + 3 * S and r["queue_left"] == 0
        assert r["shortfall_post_then_cross_bps"] == approx(POST_BUY)
        assert_unfilled(get(df, "buy", "M1"))
        assert get(df, "buy", "M1")["shortfall_diff_bps"] == approx(0.0)

    def test_trade_larger_than_remaining_queue_fills(self):
        # Q: 10 -> 6; a trade of 7 gives 6 to the orders ahead and 1 to this one.
        df = run(trade_rows=[(T0 + S, 10000, 4, SELL), (T0 + 2 * S, 10000, 7, SELL)])
        r = get(df, "buy", "M2")
        assert r["fill_ts"] == T0 + 2 * S and r["queue_left"] == 6

    def test_queue_depleted_by_size_decreases(self):
        df = run(
            book_rows=[(T0 + S, BID, 10000, 4),          # -6: Q 4
                       (T0 + 1500, BID, 10000, 50),      # an increase joins behind: Q 4
                       (T0 + 2 * S, BID, 10000, 47)],    # -3: Q 1
            trade_rows=[(T0 + 3 * S, 10000, 1, SELL),    # = Q: Q 0, no fill
                        (T0 + 4 * S, 10000, 1, SELL)],   # > Q: fill
        )
        r = get(df, "buy", "M2")
        assert r["fill_ts"] == T0 + 4 * S
        assert (r["queue_left"], r["level_size_left"]) == (0, 47)
        assert_unfilled(get(df, "buy", "M1"))

    def test_trade_and_its_decrease_both_count(self):
        # A trade of 3 and the decrease it causes, 1 ms apart: Q 10 -> 7 -> 4 (double counting, as written).
        df = run(book_rows=[(T0 + S + 1, BID, 10000, 7)], trade_rows=[(T0 + S, 10000, 3, SELL)])
        r = get(df, "buy", "M2")
        assert_unfilled(r)
        assert (r["queue_left"], r["level_size_left"]) == (4, 7)

    def test_same_millisecond_book_rows_come_first(self):
        # Book first: the decrease takes Q to 3, then the trade of 7 > 3 fills the order.
        df = run(book_rows=[(T0 + S, BID, 10000, 3)], trade_rows=[(T0 + S, 10000, 7, SELL)])
        assert get(df, "buy", "M2")["fill_ts"] == T0 + S
        # The same decrease 1 ms after the trade: Q 10 -> 3 -> 0 and no later trade, so no fill.
        df = run(book_rows=[(T0 + S + 1, BID, 10000, 3)], trade_rows=[(T0 + S, 10000, 7, SELL)])
        r = get(df, "buy", "M2")
        assert_unfilled(r)
        assert (r["queue_left"], r["level_size_left"]) == (0, 3)

    def test_snapshot_decrease_counts(self):
        # Q 10 -> 6 after a trade of 4 that the book never shows; a snapshot then shows the level at 3.
        # The snapshot is a decrease of 7 at the level, as an ordinary update would be: Q = max(6 - 7, 0).
        snapshot = [(T0 + S, BID, 10000, 3, True), (T0 + S, BID, 9999, 5), (T0 + S, ASK, 10001, 8, True),
                    (T0 + S, ASK, 10002, 4)]
        trades = [(T0 + 500, 10000, 4, SELL), (T0 + 2 * S, 10000, 3, SELL), (T0 + 3 * S, 10000, 1, SELL)]
        for rows in (snapshot, [(T0 + S, BID, 10000, 3)]):
            r = get(run(book_rows=rows, trade_rows=trades), "buy", "M2")
            assert r["fill_ts"] == T0 + 2 * S and (r["queue_left"], r["level_size_left"]) == (0, 3)

    def test_snapshot_raising_the_level(self):
        # A snapshot that raises the level is no decrease: the extra 5 stand behind the order.
        snapshot = [(T0 + S, BID, 10000, 15, True), (T0 + S, ASK, 10001, 8, True)]
        df = run(book_rows=snapshot, trade_rows=[(T0 + 2 * S, 10000, 10, SELL), (T0 + 3 * S, 10000, 1, SELL)])
        r = get(df, "buy", "M2")
        assert r["fill_ts"] == T0 + 3 * S and (r["queue_left"], r["level_size_left"]) == (0, 15)

    def test_snapshot_then_update_in_one_millisecond(self):
        # Message 0, a snapshot, takes the level from 10 to 12 (no decrease); message 1, an update in the
        # same ms, takes it to 9 (a decrease of 3): Q 7. Read as one snapshot it would be 10 -> 9, Q 9,
        # and the trades below would leave Q 1 without a fill.
        rows = [(T0 + S, BID, 10000, 12, True, 0), (T0 + S, BID, 9999, 5, False, 0),
                (T0 + S, ASK, 10001, 8, True, 0), (T0 + S, ASK, 10002, 4, False, 0),
                (T0 + S, BID, 10000, 9, False, 1)]
        df = run(book_rows=rows, trade_rows=[(T0 + 2 * S, 10000, 7, SELL), (T0 + 3 * S, 10000, 1, SELL)])
        r = get(df, "buy", "M2")
        assert r["fill_ts"] == T0 + 3 * S and (r["queue_left"], r["level_size_left"]) == (0, 9)
        book = make_book(initial() + rows)
        px, sz = probe.snapshot_levels(book, 4, depth=5)
        assert px.tolist() == [10000, 9999] and sz.tolist() == [12, 5]

    def test_level_missing_from_a_snapshot(self):
        snapshot = [(T0 + S, BID, 9999, 5, True), (T0 + S, ASK, 10001, 8, True)]
        df = run(book_rows=snapshot, trade_rows=[(T0 + 2 * S, 10000, 1, SELL)])
        assert get(df, "buy", "M2")["fill_ts"] == T0 + 2 * S
        assert_unfilled(get(df, "buy", "M1"))

    def test_level_vanishing(self):
        df = run(book_rows=[(T0 + S, BID, 10000, 0)],                       # Q 0; best bid now 9999
                 trade_rows=[(T0 + 2 * S, 10000, 1, SELL), (T0 + 3 * S, 9999, 1, SELL)])
        r2 = get(df, "buy", "M2")
        assert r2["fill_ts"] == T0 + 2 * S and (r2["queue_left"], r2["level_size_left"]) == (0, 0)
        assert get(df, "buy", "M1")["fill_ts"] == T0 + 3 * S               # only the trade through 10000


class TestTiming:
    def test_expiry_at_exactly_sixty_seconds(self):
        df = run(trade_rows=[(T0 + LIFE, 9999, 1, SELL)])
        for model in ("M1", "M2"):
            assert get(df, "buy", model)["fill_ts"] == T0 + LIFE

        rows = [(T0 + LIFE, ASK, 10001, 0), (T0 + LIFE, ASK, 10002, 0), (T0 + LIFE, ASK, 10003, 1),
                (T0 + LIFE + 1, ASK, 10003, 0), (T0 + LIFE + 1, ASK, 10005, 1)]
        df = run(book_rows=rows, trade_rows=[(T0 + LIFE + 1, 9999, 1, SELL)])
        for model in ("M1", "M2"):
            r = get(df, "buy", model)
            assert_unfilled(r)
            assert r["expiry_best_ask"] == 10003               # the state at t0 + 60 s, rows stamped then included
            assert r["shortfall_post_then_cross_bps"] == approx(bps_buy(10003, M0) + 5.0)
            assert r["shortfall_diff_bps"] == approx(bps_buy(10003, M0) - bps_buy(10001, M0))

    def test_events_at_the_decision_time_come_before_the_order(self):
        df = run(book_rows=[(T0, BID, 10000, 12)], trade_rows=[(T0, 9999, 1, SELL)])
        for model in ("M1", "M2"):
            r = get(df, "buy", model)
            assert_unfilled(r)
            assert r["queue_at_post"] == 12

    def test_back_to_back_orders(self):
        # Decisions one life apart: the first order's expiry and the second's post read the same state.
        df = run(trade_rows=[(T0 + LIFE, 9999, 1, SELL), (T0 + LIFE + 1, 9998, 1, SELL)],
                 decisions=[T0, T0 + LIFE])
        assert get(df, "buy", "M1", T0)["fill_ts"] == T0 + LIFE
        assert get(df, "buy", "M1", T0 + LIFE)["fill_ts"] == T0 + LIFE + 1


class TestSellSide:
    def test_sell_mirrors_buy(self):
        # test_trade_through_fills_with_markouts_and_shortfall reflected through 10000.5.
        f = T0 + 10 * S
        df = run(
            book_rows=[(f + MIN, ASK, 10001, 0),          # bid 10000 / ask 10002, mid 10001
                       (f + 4 * MIN, BID, 10000, 0),      # bid 9999 / ask 10002, mid 10000.5
                       (f + 10 * MIN, ASK, 10000, 3)],    # bid 9999 / ask 10000, mid 9999.5
            trade_rows=[(f, 10002, 1, BUY)],
        )
        for model in ("M1", "M2"):
            r = get(df, "sell", model)
            assert r["fill_ts"] == f and r["order_price"] == 10001 and r["queue_at_post"] == 8
            assert r["markout_1m_bps"] == approx(bps_sell(10001.0, 10001)) == 0.0
            assert r["markout_5m_bps"] == approx(bps_sell(10000.5, 10001)) == approx(0.49995000499950)
            assert r["markout_15m_bps"] == approx(bps_sell(9999.5, 10001)) == approx(1.49985001499850)
            assert r["shortfall_cross_now_bps"] == approx(CROSS_SELL)
            assert r["shortfall_post_then_cross_bps"] == approx(POST_SELL)
            assert r["shortfall_diff_bps"] == approx(-3.9999500025)
            assert_unfilled(get(df, "buy", model))

    def test_sell_queue(self):
        df = run(trade_rows=[(T0 + S, 10001, 8, BUY), (T0 + 2 * S, 10001, 1, BUY)])
        assert get(df, "sell", "M2")["fill_ts"] == T0 + 2 * S
        assert_unfilled(get(df, "sell", "M1"))


class TestMarkoutsAndSkips:
    def test_markout_beyond_the_data(self):
        f = T0 + 10 * S
        df = run(trade_rows=[(f, 9999, 1, SELL)], end=f + 6 * MIN)
        r = get(df, "buy", "M1")
        assert r["markout_1m_status"] == "ok" and r["markout_1m_bps"] == approx(0.5)
        assert r["markout_5m_status"] == "ok" and r["markout_5m_bps"] == approx(0.5)
        assert r["markout_15m_status"] == "beyond_data" and r["markout_15m_bps"] is None
        day = daily_row(df, "buy", "M1")
        assert (day["n_fills"], day["n_markout_15m"], day["n_markout_15m_beyond_data"]) == (1, 0, 1)
        assert day["markout_15m_bps_mean"] is None and day["markout_1m_bps_mean"] == approx(0.5)

    def test_markout_on_a_one_sided_book(self):
        f = T0 + 10 * S
        df = run(book_rows=[(f + 30 * S, ASK, 10001, 0), (f + 30 * S, ASK, 10002, 0),
                            (f + 2 * MIN, ASK, 10002, 4)],
                 trade_rows=[(f, 9999, 1, SELL)])
        r = get(df, "buy", "M1")
        assert r["markout_1m_status"] == "invalid_book" and r["markout_1m_bps"] is None
        assert r["markout_5m_bps"] == approx(bps_buy(10001.0, 10000))      # bid 10000 / ask 10002

    def test_one_sided_and_crossed_books_are_skipped(self):
        through = [(T0 + 5 * S, 9999, 1, SELL), (T0 + 5 * S, 10002, 1, BUY)]
        df = run(book_rows=[(T0 - S, ASK, 10001, 0), (T0 - S, ASK, 10002, 0)], trade_rows=through)
        assert set(df["status"]) == {"one_sided"} and df["filled"].null_count() == df.height
        assert df["shortfall_post_then_cross_bps"].null_count() == df.height
        day = probe.daily_summary(df, CFG)
        assert day["n_decisions"].to_list() == [0] * 4 and day["n_skipped_one_sided"].to_list() == [1] * 4
        assert day["n_fills"].to_list() == [0] * 4 and day["fill_rate"].null_count() == 4

        df = run(book_rows=[(T0 - S, ASK, 10000, 1)], trade_rows=through)     # locked: bid = ask = 10000
        assert set(df["status"]) == {"crossed"}

    def test_no_data(self):
        # The first snapshot comes after the decision; and a decision whose life runs past the coverage.
        df = run(base=initial(T0 + 1))
        assert set(df["status"]) == {"no_data"}
        df = run(decisions=[T0, END - LIFE])
        assert get(df, "buy", "M1", T0)["status"] == "ok"
        assert get(df, "buy", "M1", END - LIFE)["status"] == "no_data"

    def test_invalid_book_at_expiry_leaves_the_shortfall_undefined(self):
        df = run(book_rows=[(T0 + 30 * S, ASK, 10001, 0), (T0 + 30 * S, ASK, 10002, 0)])
        r = get(df, "buy", "M1")
        assert r["shortfall_post_then_cross_bps"] is None and r["shortfall_diff_bps"] is None
        day = daily_row(df, "buy", "M1")
        assert (day["n_shortfall"], day["n_shortfall_undefined"]) == (0, 1)
        assert day["shortfall_cross_now_bps_mean"] is None

    def test_shortfall_sample_does_not_depend_on_the_model(self):
        # A sell of 11 at the bid fills the buy under M2 (11 > Q = 10) but not M1; the ask side then
        # empties until after expiry. The decision leaves both models' shortfall means, so the two
        # models' means describe the same decisions.
        df = run(book_rows=[(T0 + 30 * S, ASK, 10001, 0), (T0 + 30 * S, ASK, 10002, 0)],
                 trade_rows=[(T0 + 5 * S, 10000, 11, SELL)])
        assert get(df, "buy", "M2")["filled"] is True and get(df, "buy", "M1")["filled"] is False
        for model in ("M1", "M2"):
            r = get(df, "buy", model)
            assert r["shortfall_cross_now_bps"] == approx(CROSS_BUY)
            assert r["shortfall_post_then_cross_bps"] is None and r["shortfall_diff_bps"] is None
            day = daily_row(df, "buy", model)
            assert (day["n_shortfall"], day["n_shortfall_undefined"]) == (0, 1)
            assert day["shortfall_cross_now_bps_mean"] is None


class TestGaps:
    def test_decisions_meeting_a_book_gap(self):
        # Gap [3 min, 10 min), ending at a snapshot. A life [t0, t0 + 60 s] that touches it is skipped:
        # 0 min no, 2 min yes (its expiry is the gap's first ms), 5 min yes, 10 min no (the snapshot is trusted).
        dec = [T0, T0 + 2 * MIN, T0 + 5 * MIN, T0 + 10 * MIN]
        df = run(book_rows=initial(T0 + 10 * MIN), decisions=dec, book_gaps=[(T0 + 3 * MIN, T0 + 10 * MIN)])
        assert [get(df, "buy", "M1", t)["status"] for t in dec] == ["ok", "gap", "gap", "ok"]
        assert get(df, "buy", "M1", T0 + 2 * MIN)["filled"] is None
        day = daily_row(df, "sell", "M2")
        assert (day["n_decisions"], day["n_skipped_gap"], day["n_skipped_no_data"]) == (2, 2, 0)

    def test_markouts_inside_a_book_gap(self):
        # Fill at 10 s; reads at 70 s, 310 s, 910 s. Gap A [70 s, 71 s) holds the first read (lo is
        # inside); gap B [200 s, 310 s) ends exactly at the second read, which is trusted (hi is not).
        f = T0 + 10 * S
        gaps = [(T0 + 70 * S, T0 + 71 * S), (T0 + 200 * S, T0 + 310 * S)]
        df = run(book_rows=initial(T0 + 71 * S) + initial(T0 + 310 * S), trade_rows=[(f, 9999, 1, SELL)],
                 book_gaps=gaps)
        r = get(df, "buy", "M1")
        assert r["status"] == "ok" and r["fill_ts"] == f
        assert r["markout_1m_status"] == "gap" and r["markout_1m_bps"] is None
        assert r["markout_5m_status"] == "ok" and r["markout_5m_bps"] == approx(0.5)
        assert r["markout_15m_status"] == "ok" and r["markout_15m_bps"] == approx(0.5)
        day = daily_row(df, "buy", "M1")
        assert (day["n_markout_1m"], day["n_markout_1m_gap"], day["n_markout_5m_gap"]) == (0, 1, 0)
        assert day["markout_1m_bps_mean"] is None

    def test_decisions_meeting_a_trade_gap(self):
        for gap, want in (((T0 + 30 * S, T0 + 31 * S), "gap"),      # inside the life
                          ((T0 + LIFE, T0 + LIFE + S), "gap"),      # starts at the expiry
                          ((T0 - S, T0), "ok")):                    # ends at t0
            assert get(run(trade_gaps=[gap]), "buy", "M2")["status"] == want

    def test_touch_mismatch_gaps(self):
        rows = initial() + [(T0 + S, BID, 9998, 2)]
        rows += [(T0 + 3 * MIN, *r[1:]) for r in initial()] + [(T0 + 3 * MIN, BID, 9998, 2)]   # matches
        snap6 = initial(T0 + 6 * MIN)
        snap6[2] = (T0 + 6 * MIN, ASK, 10001, 5, True)                                     # touch size differs
        rows += snap6 + [(T0 + 6 * MIN, BID, 9998, 2)]
        snap9 = initial(T0 + 9 * MIN)
        snap9[2] = (T0 + 9 * MIN, ASK, 10001, 5, True)                                     # as rebuilt
        rows += snap9                                                                      # 9998 missing: rank 2
        book = make_book(rows)
        check = probe.check_snapshots(book, depth=3)
        assert (check.n_checked, check.n_with_mismatch, check.n_touch_mismatch) == (6, 2, 1)
        assert sorted(check.mismatches.filter(pl.col("rank") == 0)["ts_ms"].to_list()) == [T0 + 6 * MIN]
        gaps = probe.touch_mismatch_gaps(book, check)
        assert gaps == [(T0 + 3 * MIN + 1, T0 + 6 * MIN)]
        book2 = dataclasses.replace(book, gaps=gaps)
        df = probe.simulate(book2, make_trades([]), [T0, T0 + 5 * MIN, T0 + 6 * MIN], CFG)
        assert [get(df, "buy", "M1", t)["status"] for t in (T0, T0 + 5 * MIN, T0 + 6 * MIN)] == ["ok", "gap", "ok"]
        # Rechecked with the gap declared, the mismatch is marked as expected and adds nothing.
        check2 = probe.check_snapshots(book2, depth=3)
        assert check2.mismatches.filter(pl.col("rank") == 0)["after_gap"].to_list() == [True]
        assert probe.touch_mismatch_gaps(book2, check2) == gaps

    def test_merge_gaps(self):
        assert probe.merge_gaps([(5, 7), (1, 3), (3, 4), (6, 9)]) == [(1, 4), (5, 9)]
        assert probe.merge_gaps([]) == []


class TestAggregation:
    def three_days(self) -> pl.DataFrame:
        days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
        out = []
        for d, fill, move in zip(days, (True, False, True), (False, False, True)):
            t0 = probe._day_start_ms(d)
            f = t0 + 10 * S
            rows = initial(t0 - 5 * S) + ([(f + 30 * S, BID, 10000, 0)] if move else [])
            trades = [(f, 9999, 1, SELL)] if fill else []
            book = make_book(rows, start=t0 - 10 * S, end=t0 + 30 * MIN)
            tr = make_trades(trades, start=t0 - 10 * S, end=t0 + 30 * MIN)
            out.append(probe.simulate(book, tr, [t0], CFG, day=d.isoformat()))
        return pl.concat(out)

    def test_across_day_mean_and_se(self):
        daily = probe.daily_summary(self.three_days(), CFG)
        assert daily.height == 12
        agg = probe.across_days(daily, CFG)
        row = lambda m: agg.filter((pl.col("side") == "buy") & (pl.col("model") == "M1")  # noqa: E731
                                   & (pl.col("metric") == m)).row(0, named=True)
        fr = row("fill_rate")                     # daily 1, 0, 1
        assert fr["n_days"] == 3 and fr["mean"] == approx(2 / 3)
        assert fr["sd"] == approx(math.sqrt(1 / 3)) and fr["se"] == approx(1 / 3)
        mk = row("markout_1m_bps_mean")           # day 1: mid 10000.5 -> 0.5 bp; day 2: no fill; day 3: mid 10000 -> 0
        assert mk["n_days"] == 2 and mk["mean"] == approx(0.25)
        assert mk["sd"] == approx(math.sqrt(0.125)) and mk["se"] == approx(0.25)
        n = row("n_fills")
        assert (n["total"], n["n_days"], n["mean"]) == (2, 3, approx(2 / 3))
        diff = row("shortfall_diff_bps_mean")     # filled days: POST - CROSS; day 2: crossed at 10001 after 60 s: 0
        v = np.array([POST_BUY - CROSS_BUY, 0.0, POST_BUY - CROSS_BUY])
        assert diff["mean"] == approx(v.mean()) and diff["se"] == approx(v.std(ddof=1) / math.sqrt(3))
        assert set(agg["maker_fee_bps"]) == {2.0} and set(agg["fees_verified"]) == {False}

    def test_repeated_days_are_refused(self):
        daily = probe.daily_summary(self.three_days(), CFG)
        with pytest.raises(ValueError, match="repeated"):
            probe.across_days(pl.concat([daily, daily.head(1)]), CFG)


# --------------------------------------------------------------------------
# snapshot check
# --------------------------------------------------------------------------

class TestSnapshotCheck:
    def test_compare_book(self):
        found = probe.compare_book(([100, 99], [5, 3]), ([101], [2]), ([100, 99], [5, 4]), ([101, 102], [2, 7]), 5)
        assert found == [probe.Mismatch("bid", 1, 99, 3, 99, 4), probe.Mismatch("ask", 1, None, 0, 102, 7)]
        assert probe.compare_book(([100], [5]), ([101], [2]), ([100, 99], [5, 4]), ([101], [2]), 1) == []

    def test_check_snapshots_on_a_hand_stream(self):
        book = make_book(initial() + [
            (T0 + S, BID, 9998, 2), (T0 + 2 * S, ASK, 10002, 0),
            # Snapshot: the bid side matches; the ask side has 10002 x 6, which the rebuilt book deleted.
            (T0 + 3 * S, BID, 10000, 10, True), (T0 + 3 * S, BID, 9999, 5), (T0 + 3 * S, BID, 9998, 2),
            (T0 + 3 * S, ASK, 10001, 8, True), (T0 + 3 * S, ASK, 10002, 6),
        ])
        res = probe.check_snapshots(book, depth=3)
        assert (res.n_checked, res.n_with_mismatch, res.n_touch_mismatch) == (2, 1, 0)
        assert res.mismatches.select(["side", "rank", "rebuilt_price", "rebuilt_size", "reference_price",
                                      "reference_size", "after_gap"]).rows() == [("ask", 1, None, 0, 10002, 6, False)]
        px, sz = probe.rebuilt_levels_before(book, [len(book)], 3)
        assert px[0, 0].tolist() == [10000, 9999, 9998] and px[0, 1].tolist() == [10001, 10002, -1]
        assert sz[0, 1].tolist() == [8, 6, 0]

    def test_snapshot_ends_with_its_message(self):
        t = T0 + 15 * MIN
        # A snapshot equal to the rebuilt book, then an ask-only update message in the same ms: no mismatch.
        rows = initial() + initial(t) + [(t, ASK, 10002, 0, False, 1)]
        res = probe.check_snapshots(make_book(rows), depth=5)
        assert (res.n_checked, res.n_with_mismatch) == (2, 0)
        # A snapshot whose best ask is 3 against 8 rebuilt, then an update message setting it to 8: the
        # mismatch is the snapshot's, and is reported.
        snap = initial(t)
        snap[2] = (t, ASK, 10001, 3, True)
        res = probe.check_snapshots(make_book(initial() + snap + [(t, ASK, 10001, 8, False, 1)]), depth=5)
        assert (res.n_with_mismatch, res.n_touch_mismatch) == (1, 1)
        assert res.mismatches.select(["side", "rank", "rebuilt_size", "reference_size"]).rows() == [("ask", 0, 8, 3)]

    def test_synthetic_days(self):
        book, _ = synthetic.synthetic_day(5, duration_ms=6 * 3600_000, n_book=40_000, n_trade=4_000, p_jitter=0.0)
        res = probe.check_snapshots(book, depth=400)
        # Marks at 00:15 ... 05:45 (06:00 is the end): 23 snapshots, two sides each.
        assert res.n_checked == 2 * 23 and res.n_with_mismatch == 0
        # Every snapshot changes the size of one side's best level before it is taken.
        book, _ = synthetic.synthetic_day(5, duration_ms=6 * 3600_000, n_book=40_000, n_trade=4_000, p_jitter=0.0,
                                          snap_noise=1.0)
        res = probe.check_snapshots(book, depth=400)
        assert 20 <= res.n_with_mismatch <= 23 and set(res.mismatches["rank"]) == {0}
        assert res.n_touch_mismatch == res.n_with_mismatch


# --------------------------------------------------------------------------
# independent reference and properties on random days
# --------------------------------------------------------------------------

class _Replay:
    """Book states after each distinct millisecond of a BookStream, as dicts, for state-at-t lookups."""

    def __init__(self, book: BookStream):
        self.times: List[int] = []
        self.states: List[Tuple[Dict[int, int], Dict[int, int]]] = []
        sides = {BID: {}, ASK: {}}
        n = len(book)
        k = 0
        while k < n:
            t = int(book.ts_ms[k])
            while k < n and book.ts_ms[k] == t:
                _apply(sides, int(book.side[k]), int(book.price[k]), int(book.size[k]), bool(book.reset[k]))
                k += 1
            self.times.append(t)
            self.states.append((dict(sides[BID]), dict(sides[ASK])))

    def at(self, t: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        k = bisect.bisect_right(self.times, t) - 1
        return self.states[k] if k >= 0 else ({}, {})


def _apply(sides: Dict[int, Dict[int, int]], side: int, price: int, size: int, reset: bool) -> int:
    book = sides[side]
    if reset:
        book.clear()
    old = book.get(price, 0)
    if size == 0:
        book.pop(price, None)
    else:
        book[price] = size
    return old


def _valid(bids: Dict[int, int], asks: Dict[int, int]) -> bool:
    return bool(bids) and bool(asks) and max(bids) < min(asks)


def _crosses(sides: Dict[int, Dict[int, int]], own: int, price: int) -> bool:
    opp = sides[-own]
    if not opp:
        return False
    return min(opp) <= price if own == BID else max(opp) >= price


def reference(book: BookStream, trades: TradeStream, decisions: Sequence[int], cfg: ProbeConfig) -> Dict[Tuple, Any]:
    """
    §11's rules, replayed order by order with dict books: the engine's
    output, keyed by (decision, side, model, field), recomputed without its
    single pass, dense arrays or query scheduling.
    """
    life = cfg.order_life_ms
    rep = _Replay(book)
    valid_from = book.valid_from_ms()
    book_gaps = [tuple(g) for g in book.gaps.tolist()]
    all_gaps = book_gaps + [tuple(g) for g in trades.gaps.tolist()]
    out: Dict[Tuple, Any] = {}
    for t0 in decisions:
        covered = (valid_from is not None and t0 >= valid_from and t0 + life < book.end_ms
                   and t0 >= trades.start_ms and t0 + life < trades.end_ms)
        bids0, asks0 = rep.at(t0)
        if not covered:
            status = "no_data"
        elif any(max(lo, t0) < min(hi, t0 + life + 1) for lo, hi in all_gaps):     # [t0, t0 + life] meets [lo, hi)
            status = "gap"
        elif not bids0 or not asks0:
            status = "one_sided"
        elif max(bids0) >= min(asks0):
            status = "crossed"
        else:
            status = "ok"
        out[(t0, "status")] = status
        if status != "ok":
            continue
        out[(t0, "bests")] = (max(bids0), min(asks0))
        exp_bids, exp_asks = rep.at(t0 + life)
        out[(t0, "expiry")] = (max(exp_bids) if exp_bids else None, min(exp_asks) if exp_asks else None)
        b_lo, b_hi = np.searchsorted(book.ts_ms, [t0, t0 + life], side="right")
        t_lo, t_hi = np.searchsorted(trades.ts_ms, [t0, t0 + life], side="right")
        stamps = sorted(set(book.ts_ms[b_lo:b_hi].tolist()) | set(trades.ts_ms[t_lo:t_hi].tolist()))
        for side, own in (("buy", BID), ("sell", ASK)):
            sides = {BID: dict(bids0), ASK: dict(asks0)}
            price = max(bids0) if own == BID else min(asks0)
            q = sides[own][price]
            out[(t0, side, "queue_at_post")] = q
            f1: Optional[int] = None
            f2: Optional[int] = None
            left: Optional[Tuple[int, int]] = None
            bk, tk = b_lo, t_lo
            for m in stamps:
                while bk < b_hi and book.ts_ms[bk] == m:
                    g = book.msg[bk]
                    before: Optional[int] = None        # size at the price before a snapshot of the own side
                    while bk < b_hi and book.msg[bk] == g:
                        sd, p, z, rs = int(book.side[bk]), int(book.price[bk]), int(book.size[bk]), bool(book.reset[bk])
                        if rs and sd == own:
                            before = sides[own].get(price, 0)
                        old = _apply(sides, sd, p, z, rs)
                        if sd == own and p == price and f2 is None and before is None:
                            if z < old:
                                q = max(q - (old - z), 0)
                            q = min(q, z)
                        bk += 1
                    if f2 is None:
                        if before is not None:
                            now = sides[own].get(price, 0)
                            q = min(max(q - max(before - now, 0), 0), now)
                        if _crosses(sides, own, price):
                            f2, left = m, (q, sides[own].get(price, 0))
                while tk < t_hi and trades.ts_ms[tk] == m:
                    p, z = int(trades.price[tk]), int(trades.size[tk])
                    through = p < price if own == BID else p > price
                    if through:
                        if f1 is None:
                            f1 = m
                        if f2 is None:
                            f2, left = m, (q, sides[own].get(price, 0))
                    elif p == price and f2 is None:
                        if z > q:
                            f2, left = m, (q, sides[own].get(price, 0))
                        else:
                            q -= z
                    tk += 1
                if f1 is None and _crosses(sides, own, price):
                    f1 = m
                    if f2 is None:
                        f2, left = m, (q, sides[own].get(price, 0))
            if f2 is None:
                left = (q, sides[own].get(price, 0))
            out[(t0, side, "left")] = left
            for model, ft in (("M1", f1), ("M2", f2)):
                out[(t0, side, model, "fill_ts")] = ft
                for off, lab in zip(cfg.markout_ms, cfg.markout_labels):
                    if ft is None:
                        val: Any = ("no_fill", None)
                    elif ft + off >= book.end_ms:
                        val = ("beyond_data", None)
                    elif any(lo <= ft + off < hi for lo, hi in book_gaps):
                        val = ("gap", None)
                    else:
                        b, a = rep.at(ft + off)
                        val = ("ok", (max(b) + min(a)) / 2.0) if _valid(b, a) else ("invalid_book", None)
                    out[(t0, side, model, lab)] = val
    return out


def expected_shortfalls(ref: Dict[Tuple, Any], t0: int, side: str, model: str,
                        cfg: ProbeConfig) -> Tuple[float, Optional[float]]:
    """(cross now, post then cross) in bp from the reference's prices, on §11's formulas."""
    bb, ba = ref[(t0, "bests")]
    eb, ea = ref[(t0, "expiry")]
    m0 = (bb + ba) / 2.0
    cost = (lambda x: (x / m0 - 1.0) * 1e4) if side == "buy" else (lambda x: (1.0 - x / m0) * 1e4)
    cross = cost(ba if side == "buy" else bb) + cfg.taker_fee_bps
    if eb is None or ea is None or eb >= ea:
        return cross, None
    if ref[(t0, side, model, "fill_ts")] is not None:
        return cross, cost(bb if side == "buy" else ba) + cfg.maker_fee_bps
    return cross, cost(ea if side == "buy" else eb) + cfg.taker_fee_bps


def compare_with_reference(df: pl.DataFrame, ref: Dict[Tuple, Any], cfg: ProbeConfig) -> Dict[str, int]:
    """Assert the engine's rows equal the reference; return counts of what was compared."""
    seen = {"fills": 0, "gap_decisions": 0, "gap_markouts": 0}
    for r in df.iter_rows(named=True):
        t0, side, model = r["decision_ts"], r["side"], r["model"]
        assert r["status"] == ref[(t0, "status")], (t0, side, model)
        seen["gap_decisions"] += r["status"] == "gap"
        if r["status"] != "ok":
            continue
        assert r["queue_at_post"] == ref[(t0, side, "queue_at_post")]
        assert r["fill_ts"] == ref[(t0, side, model, "fill_ts")], (t0, side, model)
        assert (r["expiry_best_bid"], r["expiry_best_ask"]) == ref[(t0, "expiry")]
        if model == "M2":
            assert (r["queue_left"], r["level_size_left"]) == ref[(t0, side, "left")], (t0, side)
        seen["fills"] += r["filled"]
        for lab in cfg.markout_labels:
            st, mid = ref[(t0, side, model, lab)]
            assert r[f"markout_{lab}_status"] == st, (t0, side, model, lab)
            seen["gap_markouts"] += st == "gap"
            if st == "ok":
                want = bps_buy(mid, r["order_price"]) if side == "buy" else bps_sell(mid, r["order_price"])
                assert r[f"markout_{lab}_bps"] == approx(want)
        cross, post = expected_shortfalls(ref, t0, side, model, cfg)
        assert r["shortfall_cross_now_bps"] == approx_bp(cross)
        if post is None:
            assert r["shortfall_post_then_cross_bps"] is None and r["shortfall_diff_bps"] is None
        else:
            assert r["shortfall_post_then_cross_bps"] == approx_bp(post)
            assert r["shortfall_diff_bps"] == approx_bp(post - cross)
    return seen


def quantize(book: BookStream, trades: TradeStream, step: int) -> Tuple[BookStream, TradeStream]:
    """Round every timestamp down to a multiple of ``step``: order is kept, collisions multiply."""
    b = BookStream(book.ts_ms // step * step, book.side, book.price, book.size, book.reset, book.msg,
                   book.start_ms, book.end_ms, book.tick_size, book.size_unit)
    t = TradeStream(trades.ts_ms // step * step, trades.price, trades.size, trades.aggressor,
                    trades.start_ms, trades.end_ms)
    return b, t


HOURS = 3
RANDOM_DAYS = [(seed, step) for seed in (11, 12, 13) for step in (1, 100)]


def random_day(seed: int, step: int, **kw: Any) -> Tuple[BookStream, TradeStream, np.ndarray]:
    book, trades = synthetic.synthetic_day(seed, duration_ms=HOURS * 3600_000, n_book=30_000, n_trade=4_000,
                                           max_levels=40, depth0=20, size_max=40, snap_noise=1.0, **kw)
    if step > 1:
        book, trades = quantize(book, trades, step)
    # A book gap of 7 minutes ending at the snapshot after 00:45, and two trade gaps: one inside the
    # 01:25 order's life, and the last millisecond before 02:06, which the 02:05 order's life reaches
    # and the 02:06 order's does not.
    snaps = np.unique(book.ts_ms[book.reset & (book.ts_ms >= T0 + 45 * MIN)])
    hi = int(snaps[0])
    book = dataclasses.replace(book, gaps=[(hi - 7 * MIN, hi)])
    trades = dataclasses.replace(trades, gaps=[(T0 + 85 * MIN + 30 * S, T0 + 85 * MIN + 31 * S),
                                               (T0 + 126 * MIN - 1, T0 + 126 * MIN)])
    # Decisions every 5 minutes on the marks, then 5 s before them, so that the 15-minute snapshots
    # (stamped a few ms past a mark) land in some orders' first milliseconds and in others' sixth
    # second, after the queue has fallen below its level; then back-to-back orders one life apart.
    dec = np.concatenate([T0 + np.arange(0, 60 * MIN, 5 * MIN),
                          T0 + 64 * MIN + 55 * S + np.arange(0, 55 * MIN, 5 * MIN),
                          T0 + 120 * MIN + np.arange(0, 30) * LIFE])
    return book, trades, dec.astype(np.int64)


@pytest.fixture(scope="module", params=RANDOM_DAYS, ids=[f"seed{s}-step{q}" for s, q in RANDOM_DAYS])
def day(request) -> Tuple[BookStream, TradeStream, np.ndarray, pl.DataFrame]:
    book, trades, dec = random_day(*request.param)
    return book, trades, dec, probe.simulate(book, trades, dec, CFG, day="r")


class TestRandomDays:
    def test_engine_matches_reference(self, day):
        book, trades, dec, df = day
        seen = compare_with_reference(df, reference(book, trades, dec, CFG), CFG)
        # The comparison covers fills and both kinds of gap, not only empty orders.
        assert seen["fills"] > 50 and seen["gap_decisions"] >= 4 and seen["gap_markouts"] > 0

    def test_m1_fills_are_m2_fills_no_later(self, day):
        df = day[3].filter(pl.col("status") == "ok")
        wide = (df.filter(pl.col("model") == "M1").select(["decision_ts", "side", pl.col("fill_ts").alias("f1")])
                .join(df.filter(pl.col("model") == "M2").select(["decision_ts", "side", pl.col("fill_ts").alias("f2")]),
                      on=["decision_ts", "side"]))
        m1 = wide.filter(pl.col("f1").is_not_null())
        assert m1.height > 0
        assert m1["f2"].null_count() == 0 and (m1["f2"] <= m1["f1"]).all()

    def test_m2_fill_rate_at_least_m1(self, day):
        daily = probe.daily_summary(day[3], CFG)
        for side in probe.SIDES:
            rate = {r["model"]: r["fill_rate"] for r in daily.filter(pl.col("side") == side).iter_rows(named=True)}
            assert rate["M2"] >= rate["M1"]
        assert (daily.filter(pl.col("model") == "M2")["fill_rate"]
                > daily.filter(pl.col("model") == "M1")["fill_rate"]).any()

    def test_queue_never_exceeds_its_level(self, day):
        m2 = day[3].filter((pl.col("model") == "M2") & (pl.col("status") == "ok"))
        assert (m2["queue_left"] >= 0).all() and (m2["queue_left"] <= m2["level_size_left"]).all()

    def test_shortfall_sample_is_shared_by_the_models(self, day):
        daily = probe.daily_summary(day[3], CFG)
        for side in probe.SIDES:
            g = {r["model"]: r for r in daily.filter(pl.col("side") == side).iter_rows(named=True)}
            assert g["M1"]["n_shortfall"] == g["M2"]["n_shortfall"]
            assert g["M1"]["shortfall_cross_now_bps_mean"] == g["M2"]["shortfall_cross_now_bps_mean"]
            for m in probe.MODELS:
                assert g[m]["shortfall_diff_bps_mean"] == pytest.approx(
                    g[m]["shortfall_post_then_cross_bps_mean"] - g[m]["shortfall_cross_now_bps_mean"], abs=1e-9)

    def test_mirrored_day_swaps_buy_and_sell(self, day):
        book, trades, dec, df = day
        c2 = 40_001
        mb, mt = synthetic.mirror(book, trades, c2)
        mdf = probe.simulate(mb, mt, dec, CFG, day="r")
        assert (df["status"] == mdf["status"]).all()
        ok = df.filter(pl.col("status") == "ok")
        mok = mdf.filter(pl.col("status") == "ok")
        for side, mside in (("buy", "sell"), ("sell", "buy")):
            a = ok.filter(pl.col("side") == side).sort(["decision_ts", "model"])
            b = mok.filter(pl.col("side") == mside).sort(["decision_ts", "model"])
            for c in ("fill_ts", "queue_at_post", "queue_left", "level_size_left"):
                assert a[c].to_list() == b[c].to_list(), c
            assert (a["order_price"] + b["order_price"] == c2).all()
            for lab in CFG.markout_labels:
                assert a[f"markout_{lab}_status"].to_list() == b[f"markout_{lab}_status"].to_list()
                # Equal in ticks: the bp values divide by reflected prices.
                ta = (a[f"markout_{lab}_bps"] * a["order_price"]).to_numpy()
                tb = (b[f"markout_{lab}_bps"] * b["order_price"]).to_numpy()
                assert np.allclose(ta, tb, rtol=0, atol=1e-6, equal_nan=True)


def fuzz_day(seed: int) -> Tuple[BookStream, TradeStream, np.ndarray]:
    """
    An adversarial stream with no market behind it: 20 minutes of random
    messages on a 250 ms grid, so that several messages, trades, decisions,
    expiries and gap edges share a millisecond; snapshots that redraw both
    sides while orders are live, sides emptied, locked and crossed books
    (some lasting only from one message to the next), trades at and through
    the touch, and a gap in each stream.
    """
    rng = np.random.default_rng(seed)
    book_rows: List[Tuple] = initial()
    trade_rows: List[Tuple] = []
    parts: Dict[int, int] = {}
    snaps: List[int] = []
    for t in np.sort(T0 + 250 * rng.integers(0, 4800, 1500)).tolist():
        u = rng.random()
        if u < 0.6:
            part = parts.get(t, 0)
            parts[t] = part + 1
            if u < 0.05:
                top = int(rng.integers(9997, 10004))
                for side, start, step in ((BID, top, -1), (ASK, top + int(rng.integers(1, 3)), 1)):
                    n = 0 if rng.random() < 0.15 else int(rng.integers(1, 6))
                    if n == 0:
                        book_rows.append((t, side, 10000, 0, True, part))
                    for k in range(n):
                        book_rows.append((t, side, start + step * k, int(rng.integers(1, 21)), k == 0, part))
                snaps.append(t)
            else:
                cells = {(int(rng.choice([BID, ASK])), int(rng.integers(9994, 10008)))
                         for _ in range(rng.integers(1, 4))}
                book_rows += [(t, side, px, int(rng.integers(0, 21)), False, part) for side, px in sorted(cells)]
        else:
            trade_rows.append((t, int(rng.integers(9997, 10005)), int(rng.integers(1, 13)),
                               int(rng.choice([BUY, SELL]))))
    end = T0 + 40 * MIN
    later = [t for t in snaps if t >= T0 + 5 * MIN]
    book_gaps = [(later[0] - int(rng.integers(1, 120)) * S, later[0])] if later else []
    lo = T0 + 7 * MIN + 250 * int(rng.integers(0, 40))
    trade_gaps = [(lo, lo + 250 * int(rng.integers(1, 40)))]
    book = make_book(book_rows, end=end, gaps=book_gaps)
    trades = make_trades(trade_rows, end=end, gaps=trade_gaps)
    return book, trades, T0 + np.arange(0, 18 * MIN, 61 * S, dtype=np.int64)


def test_engine_matches_reference_on_adversarial_streams():
    seen = {"fills": 0, "gap_decisions": 0, "gap_markouts": 0}
    statuses: set = set()
    undefined_shortfalls = 0
    for seed in range(12):
        book, trades, dec = fuzz_day(seed)
        df = probe.simulate(book, trades, dec, CFG, day="f")
        for k, v in compare_with_reference(df, reference(book, trades, dec, CFG), CFG).items():
            seen[k] += v
        statuses |= set(df["status"])
        undefined_shortfalls += int(probe.daily_summary(df, CFG)["n_shortfall_undefined"].sum())
        ok = df.filter(pl.col("status") == "ok")
        m2 = ok.filter(pl.col("model") == "M2")
        assert (m2["queue_left"] <= m2["level_size_left"]).all()
        f1 = ok.filter(pl.col("model") == "M1")["fill_ts"].to_list()
        f2 = m2["fill_ts"].to_list()
        assert all(b is not None and b <= a for a, b in zip(f1, f2) if a is not None)
    assert {"ok", "gap", "crossed"} <= statuses
    assert seen["fills"] > 100 and seen["gap_decisions"] > 0 and seen["gap_markouts"] > 0
    assert undefined_shortfalls > 0


def queue_day(seed: int, n: int = 200) -> Tuple[BookStream, TradeStream, np.ndarray]:
    """
    ``n`` short queue scenarios, one per decision, 61 s apart: each starts
    from a fresh snapshot of the touch with random sizes, then mixes, on a
    500 ms grid, trades at either order's price, updates of its level and
    snapshots that set the level to a new size, some in the same
    millisecond as other messages. These are the events where the queue is
    partly used up when a snapshot lowers the level, so counting the
    snapshot's decrease and only capping at the new size give different
    queues; the random days above rarely produce them.
    """
    rng = np.random.default_rng(seed)
    rows: List[Tuple] = []
    trades: List[Tuple] = []
    dec = T0 + np.arange(n, dtype=np.int64) * 61 * S
    for t0 in dec.tolist():
        rows += [(t0 - 500, BID, 10000, int(rng.integers(5, 31)), True), (t0 - 500, BID, 9999, 5),
                 (t0 - 500, ASK, 10001, int(rng.integers(5, 31)), True), (t0 - 500, ASK, 10002, 4)]
        parts: Dict[int, int] = {}
        for t in np.sort(t0 + 500 * rng.integers(1, 121, int(rng.integers(6, 16)))).tolist():
            u = rng.random()
            own, px, aggr = (BID, 10000, SELL) if rng.random() < 0.5 else (ASK, 10001, BUY)
            if u < 0.4:
                trades.append((t, px, int(rng.integers(1, 7)), aggr))
                continue
            part = parts.get(t, 0)
            parts[t] = part + 1
            if u < 0.7:
                rows.append((t, own, px, int(rng.integers(0, 31)), False, part))
                continue
            for side, top, nxt in ((BID, 10000, 9999), (ASK, 10001, 10002)):
                z = int(rng.integers(0, 31))
                rows += ([(t, side, top, z, True, part), (t, side, nxt, 5, False, part)] if z
                         else [(t, side, nxt, 5, True, part)])
    end = int(dec[-1]) + 20 * MIN
    return make_book(rows, end=end), make_trades(trades, end=end), dec


def test_engine_matches_reference_on_queue_scenarios():
    for seed in range(3):
        book, trades, dec = queue_day(seed)
        df = probe.simulate(book, trades, dec, CFG, day="q")
        assert compare_with_reference(df, reference(book, trades, dec, CFG), CFG)["fills"] > 150
        m2 = df.filter(pl.col("model") == "M2")
        assert (m2["queue_left"] <= m2["level_size_left"]).all()


def test_full_synthetic_day():
    book, trades = synthetic.synthetic_day(2, n_book=150_000, n_trade=15_000)
    df = probe.simulate_day(book, trades, DAY, CFG)
    assert df.height == 288 * 4 and set(df["day"]) == {DAY.isoformat()}
    assert df.filter(pl.col("status") == "ok").height == df.height
    daily = probe.daily_summary(df, CFG)
    assert daily.height == 4 and (daily["n_decisions"] == 288).all()
