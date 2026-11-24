"""
Simulation engine of the maker-execution probe: pre-registration §11 and the
``maker_probe`` block of Appendix A.

§11 asks whether resting one contract at the touch beats crossing the spread
once adverse selection is counted. The probe's archive files were not to be
downloaded before ``doge_fwd`` had run (§12, 2026-10-12), so the engine was
written against an explicit in-memory input contract (BookStream,
TradeStream), checked on synthetic days only, and committed before any file
was downloaded. The two archive parsers, ``read_book`` and ``read_trades``,
depend on the files' format, so they were written once the files were here,
to that contract; their section records the format as found.

Time. "The state at t" is the book after every row with ``ts <= t``. A
decision at t0 posts against the state at t0, so rows and trades stamped t0
come before the order. The order then sees events with ``t0 < ts <= t0 +
life``; an event at exactly t0 + 60 s can still fill it. At t0 + life an
unfilled order is cancelled and "post, then cross" crosses at the state at
t0 + life. A markout at k minutes reads the state at fill_ts + k.

Within one millisecond the engine applies the book rows message by message,
in input order, then the trades in input order, and then runs M1's crossing
test on the state at the end of the millisecond. The two fill models:

  * **M1, lower bound.** A buy at P fills in the first millisecond in which
    a trade prints below P, or the end-of-millisecond best ask is at or
    below P. A sell mirrors it: a trade above P, or a best bid at or above
    P. An ask at P counts as crossing (a choice; §11 says only "crosses"):
    it could not rest at P while bids did, so in a consistent book the whole
    bid queue at P, this order included, is gone.
  * **M2, upper bound.** Q, the queue ahead, starts at the displayed size at
    P in the state at t0: the order joins the back of its level. Every
    update row at P on the order's side that lowers the size by x lowers Q
    by x (not below 0). A snapshot of the order's side is one change at P:
    Q falls by the size at P just before the snapshot minus the size the
    snapshot sets, if that is positive. A trade at P of size z fills the
    order if z > Q: the first Q units go to the orders ahead, and the rest
    reach this one. Otherwise Q -= z, so a trade of exactly Q exhausts the
    queue without touching the order, and the next trade at P fills it. Q
    is capped at the displayed size at P. The order also fills whenever M1
    would, with the crossing test run after every message rather than once
    per millisecond.

Why M2 is an upper bound. A trade at P normally arrives with a book row that
lowers P's size by the same amount, and both count (§11 as written), so a
traded unit removes two units of queue. Trades at P count whatever their
aggressor side. A positive residual of any size fills the whole contract. A
cross that lasts only from one message to the next inside a millisecond
fills it.

The cap. Q starts equal to the displayed size; every decrease of x, a
snapshot's included, lowers both by x, while an increase (orders joining
behind) raises only the size. So Q <= displayed size holds by construction
and §11's cap never binds; the engine applies it as a guard. A level that
vanishes leaves Q = 0, and orders that join it later stand behind this one.

Why the same-millisecond order keeps the bounds. Book rows only lower Q, and
the map they apply to Q, f, satisfies f(x - s) >= f(x) - s for s >= 0, so
applying them before a millisecond's trades leaves each trade facing a queue
no larger than the reverse order would, and leaves no larger a queue for
later: M2 fills weakly earlier. M2's crossing test sees every message's end
state, M1's only the millisecond's last, so a cross that exists only between
two messages of one millisecond fills M2 and not M1. M1's trade test ignores
the book. Inside a message the crossing test waits for the last row: the
rows of a message are one book change, and in a well-formed message (each
level at most once, a snapshot clearing its side before setting it, both
checked by BookStream) no row can put the opposite best across P unless the
message's end state has it there too.

A decision whose book at t0 is one-sided or crossed (best bid >= best ask) is
counted and skipped; so is one the data does not cover (``no_data``) and one
whose life [t0, t0 + life] meets a gap in either stream (``gap``). A markout
whose time lies at or beyond the book's coverage, inside a book gap, or on a
one-sided or crossed book is counted and excluded, never imputed.
"""
from __future__ import annotations

import io
import json
import math
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
from numba import njit

from research.spec import get_spec, overlaps_protected

MINUTE_MS = 60_000
DAY_MS = 86_400_000

# BookStream.side and TradeStream.aggressor (the taker's side).
BID, ASK = 1, -1
BUY, SELL = 1, -1

SIDES = ("buy", "sell")
MODELS = ("M1", "M2")
STATUS_NAMES = ("ok", "one_sided", "crossed", "no_data", "gap")
MARKOUT_STATUS_NAMES = ("ok", "beyond_data", "invalid_book", "no_fill", "gap")
FILL_MODELS_SPEC = ["M1_trade_through", "M2_queue_position_or_M1"]

# Kernel codes; the *_NAMES tuples above are indexed by them.
_ST_OK, _ST_ONE_SIDED, _ST_CROSSED, _ST_NO_DATA, _ST_GAP = 0, 1, 2, 3, 4
_MK_OK, _MK_BEYOND, _MK_INVALID, _MK_NO_FILL, _MK_GAP = 0, 1, 2, 3, 4
_INF = np.int64(np.iinfo(np.int64).max)
# A day's price range in ticks: DOGE near 0.2 at a 0.00001 tick is 20,000
# ticks, so this is far beyond any real day and only stops a corrupt price
# from allocating gigabytes.
MAX_LEVEL_SPAN = 1 << 22

__all__ = [
    "BID", "ASK", "BUY", "SELL", "ProbeConfig", "BookStream", "TradeStream",
    "decision_times", "required_coverage", "simulate", "simulate_day",
    "daily_summary", "across_days", "rebuilt_levels_before", "snapshot_levels",
    "compare_book", "check_snapshots", "touch_mismatch_gaps", "merge_gaps",
    "Mismatch", "SnapshotCheck", "read_book", "read_trades", "max_depth", "trade_book_lags",
    "BOOK_DEPTH", "TICK_SIZE", "SIZE_UNIT", "MAX_SILENCE_MS",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeConfig:
    """The probe's constants, as Appendix A's ``maker_probe`` block fixes them."""

    decision_every_ms: int
    order_life_ms: int
    size_contracts: float
    markout_ms: Tuple[int, ...]
    maker_fee_bps: float
    taker_fee_bps: float
    fees_verified: bool
    n_days: int

    @classmethod
    def from_spec(cls, spec: Optional[Dict[str, Any]] = None) -> "ProbeConfig":
        """Read the block from Appendix A, refusing fill models this engine does not implement."""
        mp = (spec or get_spec())["maker_probe"]
        if list(mp["fill_models"]) != FILL_MODELS_SPEC:
            raise ValueError(f"maker_probe.fill_models is {mp['fill_models']}; "
                             f"this engine implements {FILL_MODELS_SPEC}")
        fees = mp["fees_bps"]
        cfg = cls(
            decision_every_ms=int(mp["decision_every_minutes"]) * MINUTE_MS,
            order_life_ms=int(mp["order_life_seconds"]) * 1000,
            size_contracts=float(mp["size_contracts"]),
            markout_ms=tuple(int(m) * MINUTE_MS for m in mp["markout_minutes"]),
            maker_fee_bps=float(fees["maker"]),
            taker_fee_bps=float(fees["taker"]),
            fees_verified=bool(fees["verified"]),
            n_days=int(mp["n_days"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise ValueError on constants the engine's rules do not cover."""
        if self.size_contracts != 1.0:
            # M2 fills on any positive residual, so a larger order would be
            # silently treated as one contract.
            raise ValueError(f"size_contracts is {self.size_contracts}; the fill rules model one contract")
        if self.order_life_ms <= 0 or self.decision_every_ms <= 0:
            raise ValueError("order life and decision spacing must be positive")
        if DAY_MS % self.decision_every_ms:
            raise ValueError("decision spacing must divide the day")
        if not self.markout_ms or list(self.markout_ms) != sorted(set(self.markout_ms)) or self.markout_ms[0] <= 0:
            raise ValueError(f"markout horizons must be positive, ascending and distinct: {self.markout_ms}")

    @property
    def markout_labels(self) -> Tuple[str, ...]:
        """Column labels of the markout horizons: '1m', '5m', '15m'."""
        return tuple(_horizon_label(m) for m in self.markout_ms)


def _horizon_label(ms: int) -> str:
    return f"{ms // MINUTE_MS}m" if ms % MINUTE_MS == 0 else f"{ms}ms"


def _default(config: Optional[ProbeConfig]) -> ProbeConfig:
    cfg = config or ProbeConfig.from_spec()
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------
# input contract
# --------------------------------------------------------------------------

def _int_array(name: str, values: Any) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if a.size and a.dtype.kind not in "iu":
        raise TypeError(f"{name} must hold integers (got {a.dtype}); convert exactly in the parser")
    return a.astype(np.int64, copy=False)


def _check_times(name: str, ts: np.ndarray, start_ms: int, end_ms: int, order_hint: str) -> None:
    if end_ms <= start_ms:
        raise ValueError(f"{name}: coverage [{start_ms}, {end_ms}) is empty")
    if ts.size:
        if np.any(np.diff(ts) < 0):
            raise ValueError(f"{name}: timestamps must be non-decreasing ({order_hint})")
        if ts[0] < start_ms or ts[-1] >= end_ms:
            raise ValueError(f"{name}: timestamps fall outside the declared coverage [{start_ms}, {end_ms})")


def _gap_array(name: str, gaps: Any, start_ms: int, end_ms: int) -> np.ndarray:
    """Gaps as an (n, 2) int64 array of half-open [lo, hi), checked sorted, disjoint and inside the coverage."""
    rows = gaps.tolist() if isinstance(gaps, np.ndarray) else list(gaps)
    out = []
    for r in rows:
        r = tuple(r)
        if len(r) != 2 or not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in r):
            raise TypeError(f"{name}: a gap is a pair of integer ms (lo, hi), got {r}")
        out.append((int(r[0]), int(r[1])))
    g = np.array(out, dtype=np.int64).reshape(-1, 2)
    if g.size:
        if np.any(g[:, 0] >= g[:, 1]):
            raise ValueError(f"{name}: every gap [lo, hi) needs lo < hi")
        if np.any(g[1:, 0] < g[:-1, 1]):
            raise ValueError(f"{name}: gaps must be sorted and disjoint (see merge_gaps)")
        if g[0, 0] < start_ms or g[-1, 1] > end_ms:
            raise ValueError(f"{name}: gaps must lie inside the coverage [{start_ms}, {end_ms})")
    return g


def _level_span(price: np.ndarray) -> Tuple[int, int]:
    """(base tick, number of level slots) of the dense book arrays."""
    if price.size == 0:
        return 0, 1
    base = int(price.min())
    span = int(price.max()) - base + 1
    if span > MAX_LEVEL_SPAN:
        raise ValueError(f"book prices span {span} ticks in one day; a price is corrupt or not in ticks")
    return base, span


_MSG_RULES = {
    1: "msg must be non-decreasing along the rows (sequence order)",
    2: "the rows of one message must share one timestamp",
    3: "a reset row must be its side's first row in its message",
    4: "a message may set each (side, price) at most once",
}


@njit(fastmath=False, cache=False)
def _message_violation(ts, side_idx, idx, reset, msg, n_levels):
    """(rule broken, row) for the first row breaking BookStream's message rules, or (0, -1)."""
    seen_side = np.full(2, -1, np.int64)
    seen_level = np.full((2, n_levels), -1, np.int64)
    start = 0
    for i in range(ts.shape[0]):
        if i > 0 and msg[i] == msg[i - 1]:
            if ts[i] != ts[i - 1]:
                return 2, i
        else:
            if i > 0 and msg[i] < msg[i - 1]:
                return 1, i
            start = i
        s = side_idx[i]
        if reset[i] and seen_side[s] == start:
            return 3, i
        if seen_level[s, idx[i]] == start:
            return 4, i
        seen_side[s] = start
        seen_level[s, idx[i]] = start
    return 0, -1


@dataclass(frozen=True)
class BookStream:
    """
    Order-book level updates for one probe day, in application order.

    Row k sets the total displayed size at ``price[k]`` on ``side[k]`` (BID
    = +1, ASK = -1) to ``size[k]``; size 0 deletes the level. ``msg[k]``
    numbers the archive message the row comes from: non-decreasing, so a
    message's rows are contiguous, and they share one timestamp. If
    ``reset[k]`` is true the whole side is cleared first, so a snapshot of a
    side is its reset row (carrying the first level, or size 0 if the side
    is empty) followed by the side's other rows in the same message. A reset
    row is its side's first row in its message, and a message sets each
    (side, price) at most once. Rows are in the archive's sequence order,
    never re-sorted by time, and ``ts_ms`` is non-decreasing along it.

    Prices are int64 ticks (price / tick_size, exact) and sizes int64
    multiples of ``size_unit`` contracts, in the same unit as TradeStream's
    sizes: M2 compares the two directly. Every ``ts_ms`` lies in the
    half-open coverage ``[start_ms, end_ms)``, and the stream is complete
    over it outside ``gaps``: the state at any such t is the state after the
    rows with ts <= t. The book is defined from the first time both sides
    have been reset.

    ``gaps`` holds half-open intervals [lo, hi) whose states are not to be
    trusted (a sequence break; a snapshot that disagrees with the rebuilt
    book at the touch): sorted, disjoint, inside the coverage, and each
    ending at a snapshot, i.e. both sides have a reset row stamped hi.
    Decisions whose life meets one and markouts read inside one are skipped
    and counted.
    """

    ts_ms: np.ndarray
    side: np.ndarray
    price: np.ndarray
    size: np.ndarray
    reset: np.ndarray
    msg: np.ndarray
    start_ms: int
    end_ms: int
    tick_size: float
    size_unit: float
    gaps: Any = ()

    def __post_init__(self) -> None:
        ts = _int_array("ts_ms", self.ts_ms)
        side = _int_array("side", self.side)
        price = _int_array("price", self.price)
        size = _int_array("size", self.size)
        msg = _int_array("msg", self.msg)
        reset = np.asarray(self.reset)
        if reset.ndim != 1 or (reset.size and reset.dtype.kind not in "biu"):
            raise TypeError("reset must be a one-dimensional boolean array")
        if reset.size and reset.dtype.kind != "b":
            if not np.isin(reset, (0, 1)).all():
                raise ValueError("reset must be 0/1")
        reset = reset.astype(np.bool_, copy=False)
        n = ts.size
        if not (side.size == price.size == size.size == reset.size == msg.size == n):
            raise ValueError("BookStream arrays differ in length")
        start, end = int(self.start_ms), int(self.end_ms)
        _check_times("BookStream", ts, start, end,
                     "rows stay in sequence order; the parser raises a timestamp that goes back to the "
                     "previous one's and reports it, and never re-sorts rows by time")
        if n:
            if not np.isin(side, (BID, ASK)).all():
                raise ValueError("BookStream.side must be BID (+1) or ASK (-1)")
            if price.min() <= 0:
                raise ValueError("BookStream prices must be positive ticks")
            if size.min() < 0:
                raise ValueError("BookStream sizes must be >= 0")
            base, span = _level_span(price)
            rule, row = _message_violation(ts, (side == ASK).astype(np.int8), price - base, reset, msg, span)
            if rule:
                raise ValueError(f"BookStream row {row}: {_MSG_RULES[rule]}")
        if not (self.tick_size > 0 and self.size_unit > 0):
            raise ValueError("tick_size and size_unit must be positive")
        gaps = _gap_array("BookStream", self.gaps, start, end)
        if gaps.size:
            r = np.flatnonzero(reset)
            for lo, hi in gaps.tolist():
                for s, name in ((BID, "bid"), (ASK, "ask")):
                    if not np.any((ts[r] == hi) & (side[r] == s)):
                        raise ValueError(f"BookStream gap [{lo}, {hi}) does not end at a snapshot: "
                                         f"no {name} reset row is stamped {hi}")
        for k, v in (("ts_ms", ts), ("side", side.astype(np.int8)), ("price", price), ("size", size),
                     ("reset", reset), ("msg", msg), ("gaps", gaps)):
            object.__setattr__(self, k, v)
        object.__setattr__(self, "start_ms", start)
        object.__setattr__(self, "end_ms", end)

    def __len__(self) -> int:
        return int(self.ts_ms.size)

    def valid_from_ms(self) -> Optional[int]:
        """The first time both sides have been reset, or None if one never is."""
        out = []
        for s in (BID, ASK):
            hit = np.flatnonzero(self.reset & (self.side == s))
            if hit.size == 0:
                return None
            out.append(int(self.ts_ms[hit[0]]))
        return max(out)


@dataclass(frozen=True)
class TradeStream:
    """
    Trades for one probe day: time, price (int64 ticks), size (int64, the
    BookStream's size unit) and the taker's side (BUY = +1, SELL = -1).
    Sorted by ``ts_ms``, non-decreasing, all inside the half-open coverage
    ``[start_ms, end_ms)``, which is complete outside ``gaps``: an interval
    without trades had none. ``gaps`` holds half-open intervals [lo, hi) in
    which trades may be missing (sorted, disjoint, inside the coverage);
    decisions whose life meets one are skipped and counted.

    The fill rules use price and size only; ``aggressor`` is validated and
    carried for the parser's checks (a trade at the bid should mostly be a
    sell).
    """

    ts_ms: np.ndarray
    price: np.ndarray
    size: np.ndarray
    aggressor: np.ndarray
    start_ms: int
    end_ms: int
    gaps: Any = ()

    def __post_init__(self) -> None:
        ts = _int_array("ts_ms", self.ts_ms)
        price = _int_array("price", self.price)
        size = _int_array("size", self.size)
        aggr = _int_array("aggressor", self.aggressor)
        if not (price.size == size.size == aggr.size == ts.size):
            raise ValueError("TradeStream arrays differ in length")
        start, end = int(self.start_ms), int(self.end_ms)
        _check_times("TradeStream", ts, start, end, "sort by (ts, trade id), stably")
        if ts.size:
            if price.min() <= 0:
                raise ValueError("TradeStream prices must be positive ticks")
            if size.min() <= 0:
                raise ValueError("TradeStream sizes must be positive")
            if not np.isin(aggr, (BUY, SELL)).all():
                raise ValueError("TradeStream.aggressor must be BUY (+1) or SELL (-1)")
        gaps = _gap_array("TradeStream", self.gaps, start, end)
        for k, v in (("ts_ms", ts), ("price", price), ("size", size), ("aggressor", aggr.astype(np.int8)),
                     ("gaps", gaps)):
            object.__setattr__(self, k, v)
        object.__setattr__(self, "start_ms", start)
        object.__setattr__(self, "end_ms", end)

    def __len__(self) -> int:
        return int(self.ts_ms.size)


# --------------------------------------------------------------------------
# day grid
# --------------------------------------------------------------------------

def _day_start_ms(day: Union[str, date]) -> int:
    d = date.fromisoformat(day) if isinstance(day, str) else day
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def decision_times(day: Union[str, date], config: Optional[ProbeConfig] = None) -> np.ndarray:
    """The 288 decision times of UTC day D: D 00:00:00.000 + k * 5 min, as int64 ms."""
    cfg = _default(config)
    n = DAY_MS // cfg.decision_every_ms
    return _day_start_ms(day) + np.arange(n, dtype=np.int64) * cfg.decision_every_ms


def required_coverage(day: Union[str, date], config: Optional[ProbeConfig] = None) -> Tuple[int, int]:
    """
    Half-open [start, end) ms that both streams must cover for day D.

    start is D 00:00 UTC; end lies 1 ms past the last decision + order life +
    longest markout (D+1 00:11:00.001 under Appendix A). The parsers crop to
    it, so no row outside it is read; markouts falling past a shorter
    coverage are counted as beyond the data.
    """
    cfg = _default(config)
    dec = decision_times(day, cfg)
    return int(dec[0]), int(dec[-1]) + cfg.order_life_ms + max(cfg.markout_ms) + 1


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

@njit(fastmath=False, cache=False)
def _apply_level(levels, best, nlev, lo, hi, s, p, z):
    """Set side s's size at index p to z, keeping best, level count and touched bounds; return the old size."""
    old = levels[s, p]
    if old == z:
        return old
    levels[s, p] = z
    if old == 0:
        nlev[s] += 1
        if p < lo[s]:
            lo[s] = p
        if p > hi[s]:
            hi[s] = p
        b = best[s]
        if b < 0 or (s == 0 and p > b) or (s == 1 and p < b):
            best[s] = p
    elif z == 0:
        nlev[s] -= 1
        if p == best[s]:
            if nlev[s] == 0:
                best[s] = -1
            elif s == 0:
                k = p - 1
                while levels[0, k] == 0:
                    k -= 1
                best[0] = k
            else:
                k = p + 1
                while levels[1, k] == 0:
                    k += 1
                best[1] = k
    return old


@njit(fastmath=False, cache=False)
def _clear_side(levels, best, nlev, lo, hi, s):
    if hi[s] >= lo[s]:
        levels[s, lo[s]:hi[s] + 1] = 0
    nlev[s] = 0
    best[s] = -1
    lo[s] = levels.shape[1]
    hi[s] = -1


@njit(fastmath=False, cache=False)
def _record_fill(s, m, tn, od, P, Q, levels, filled, fill_ts, q_left, lvl_left,
                 mk_off, pq_t, pq_d, pq_s, pq_m, pq_k, npend):
    """Mark model m of side s's live order filled at tn and queue its markout reads; return the new pending count."""
    d = od[s]
    filled[s, m] = True
    fill_ts[d, s, m] = tn
    if m == 1:
        q_left[d, s] = Q[s]
        lvl_left[d, s] = levels[s, P[s]]
    for k in range(mk_off.shape[0]):
        pq_t[npend] = tn + mk_off[k]
        pq_d[npend] = d
        pq_s[npend] = s
        pq_m[npend] = m
        pq_k[npend] = k
        npend += 1
    return npend


@njit(fastmath=False, cache=False)
def _simulate_kernel(b_ts, b_side, b_idx, b_sz, b_reset, b_msg, n_levels,
                     t_ts, t_idx, t_sz, dec_ts, dec_ok, life_ms, mk_off, book_end):
    """
    One pass over a day's merged book and trade events.

    Prices are level indices (ticks minus a base); sides are 0 (bid, where a
    buy rests) and 1 (ask, where a sell rests). Queries (posts at t0, expiries
    at t0 + life, markout reads at fill + k) are answered just before the
    first event stamped later than the query, so each sees the state at its
    time. Decision times must be at least ``life_ms`` apart: one live order
    per side.
    """
    nd = dec_ts.shape[0]
    nk = mk_off.shape[0]
    nb = b_ts.shape[0]
    nt = t_ts.shape[0]

    status = np.full(nd, _ST_NO_DATA, np.int8)
    bid0 = np.full(nd, -1, np.int64)
    ask0 = np.full(nd, -1, np.int64)
    queue0 = np.full((nd, 2), -1, np.int64)
    fill_ts = np.full((nd, 2, 2), -1, np.int64)
    exp_bid = np.full(nd, -1, np.int64)
    exp_ask = np.full(nd, -1, np.int64)
    q_left = np.full((nd, 2), -1, np.int64)
    lvl_left = np.full((nd, 2), -1, np.int64)
    mk_sum = np.full((nd, 2, 2, nk), -1, np.int64)
    mk_st = np.full((nd, 2, 2, nk), _MK_NO_FILL, np.int8)

    levels = np.zeros((2, n_levels), np.int64)
    best = np.full(2, -1, np.int64)
    nlev = np.zeros(2, np.int64)
    lo = np.full(2, n_levels, np.int64)
    hi = np.full(2, -1, np.int64)

    live = np.zeros(2, np.bool_)
    od = np.zeros(2, np.int64)
    P = np.zeros(2, np.int64)
    Q = np.zeros(2, np.int64)
    filled = np.zeros((2, 2), np.bool_)
    # A snapshot of a live M2 order's side, open until its message ends, and
    # the size at P just before it.
    snap_open = np.zeros(2, np.bool_)
    snap_before = np.zeros(2, np.int64)

    cap = nd * 4 * nk + 1
    pq_t = np.empty(cap, np.int64)
    pq_d = np.empty(cap, np.int64)
    pq_s = np.empty(cap, np.int64)
    pq_m = np.empty(cap, np.int64)
    pq_k = np.empty(cap, np.int64)
    npend = 0
    next_dyn = _INF

    nf = 2 * nd
    f = 0
    i = 0
    j = 0
    while True:
        tb = b_ts[i] if i < nb else _INF
        tt = t_ts[j] if j < nt else _INF
        tn = tb if tb < tt else tt

        # Answer every query whose time is before the next event.
        while True:
            if f < nf:
                d = f >> 1
                qf = dec_ts[d] + (life_ms if (f & 1) == 1 else 0)
            else:
                d = -1
                qf = _INF
            if qf >= tn and next_dyn >= tn:
                break
            if qf <= next_dyn:
                if (f & 1) == 0:
                    if dec_ok[d]:
                        bb = best[0]
                        ba = best[1]
                        if bb < 0 or ba < 0:
                            status[d] = _ST_ONE_SIDED
                        elif bb >= ba:
                            status[d] = _ST_CROSSED
                        else:
                            status[d] = _ST_OK
                            bid0[d] = bb
                            ask0[d] = ba
                            for s in range(2):
                                live[s] = True
                                od[s] = d
                                P[s] = bb if s == 0 else ba
                                Q[s] = levels[s, P[s]]
                                queue0[d, s] = Q[s]
                                filled[s, 0] = False
                                filled[s, 1] = False
                else:
                    if status[d] == _ST_OK:
                        exp_bid[d] = best[0]
                        exp_ask[d] = best[1]
                        for s in range(2):
                            if not filled[s, 1]:
                                q_left[d, s] = Q[s]
                                lvl_left[d, s] = levels[s, P[s]]
                            live[s] = False
                f += 1
            else:
                qd = next_dyn
                e = 0
                while e < npend:
                    if pq_t[e] == qd:
                        d2 = pq_d[e]
                        s2 = pq_s[e]
                        m2 = pq_m[e]
                        k2 = pq_k[e]
                        if qd >= book_end:
                            mk_st[d2, s2, m2, k2] = _MK_BEYOND
                        elif best[0] < 0 or best[1] < 0 or best[0] >= best[1]:
                            mk_st[d2, s2, m2, k2] = _MK_INVALID
                        else:
                            mk_st[d2, s2, m2, k2] = _MK_OK
                            mk_sum[d2, s2, m2, k2] = best[0] + best[1]
                        npend -= 1
                        pq_t[e] = pq_t[npend]
                        pq_d[e] = pq_d[npend]
                        pq_s[e] = pq_s[npend]
                        pq_m[e] = pq_m[npend]
                        pq_k[e] = pq_k[npend]
                    else:
                        e += 1
                next_dyn = _INF
                for e in range(npend):
                    if pq_t[e] < next_dyn:
                        next_dyn = pq_t[e]

        if tn == _INF:
            break

        # 1. Book rows of this millisecond, message by message (a message's
        #    rows share its timestamp).
        while i < nb and b_ts[i] == tn:
            g = b_msg[i]
            while i < nb and b_msg[i] == g:
                s = b_side[i]
                p = b_idx[i]
                z = b_sz[i]
                m2_live = live[s] and not filled[s, 1]
                if b_reset[i]:
                    if m2_live:
                        snap_open[s] = True
                        snap_before[s] = levels[s, P[s]]
                    _clear_side(levels, best, nlev, lo, hi, s)
                old = _apply_level(levels, best, nlev, lo, hi, s, p, z)
                if m2_live and not snap_open[s] and p == P[s]:
                    if z < old:
                        Q[s] -= old - z
                        if Q[s] < 0:
                            Q[s] = 0
                    if Q[s] > z:
                        Q[s] = z
                i += 1
            # End of the message: a snapshot of the order's side is one change
            # at P, then M2's crossing test on the message's end state.
            for s in range(2):
                if snap_open[s]:
                    snap_open[s] = False
                    lv = levels[s, P[s]]
                    if snap_before[s] > lv:
                        Q[s] -= snap_before[s] - lv
                        if Q[s] < 0:
                            Q[s] = 0
                    if Q[s] > lv:
                        Q[s] = lv
                if live[s] and not filled[s, 1]:
                    o = best[1 - s]
                    if o >= 0 and (o <= P[s] if s == 0 else o >= P[s]):
                        npend = _record_fill(s, 1, tn, od, P, Q, levels, filled, fill_ts, q_left, lvl_left,
                                             mk_off, pq_t, pq_d, pq_s, pq_m, pq_k, npend)
                        if tn + mk_off[0] < next_dyn:
                            next_dyn = tn + mk_off[0]

        # 2. Trades of this millisecond, in input order.
        while j < nt and t_ts[j] == tn:
            p = t_idx[j]
            z = t_sz[j]
            for s in range(2):
                if not live[s]:
                    continue
                through = p < P[s] if s == 0 else p > P[s]
                if through:
                    for m in range(2):
                        if not filled[s, m]:
                            npend = _record_fill(s, m, tn, od, P, Q, levels, filled, fill_ts, q_left, lvl_left,
                                                 mk_off, pq_t, pq_d, pq_s, pq_m, pq_k, npend)
                            if tn + mk_off[0] < next_dyn:
                                next_dyn = tn + mk_off[0]
                elif p == P[s] and not filled[s, 1]:
                    if z > Q[s]:
                        npend = _record_fill(s, 1, tn, od, P, Q, levels, filled, fill_ts, q_left, lvl_left,
                                             mk_off, pq_t, pq_d, pq_s, pq_m, pq_k, npend)
                        if tn + mk_off[0] < next_dyn:
                            next_dyn = tn + mk_off[0]
                    else:
                        Q[s] -= z
            j += 1

        # 3. M1's crossing test on the end-of-millisecond state. M2 has
        #    already seen this state, as the end of the last message.
        for s in range(2):
            if not live[s] or filled[s, 0]:
                continue
            o = best[1 - s]
            crossed = o >= 0 and (o <= P[s] if s == 0 else o >= P[s])
            if crossed:
                for m in range(2):
                    if not filled[s, m]:
                        npend = _record_fill(s, m, tn, od, P, Q, levels, filled, fill_ts, q_left, lvl_left,
                                             mk_off, pq_t, pq_d, pq_s, pq_m, pq_k, npend)
                        if tn + mk_off[0] < next_dyn:
                            next_dyn = tn + mk_off[0]

    return status, bid0, ask0, queue0, fill_ts, exp_bid, exp_ask, q_left, lvl_left, mk_sum, mk_st


@njit(fastmath=False, cache=False)
def _levels_before_kernel(b_side, b_idx, b_sz, b_reset, n_levels, rows, depth):
    """Top ``depth`` levels per side of the book rebuilt from rows [0, r), for each r in ascending ``rows``."""
    nr = rows.shape[0]
    out_px = np.full((nr, 2, depth), -1, np.int64)
    out_sz = np.zeros((nr, 2, depth), np.int64)
    levels = np.zeros((2, n_levels), np.int64)
    best = np.full(2, -1, np.int64)
    nlev = np.zeros(2, np.int64)
    lo = np.full(2, n_levels, np.int64)
    hi = np.full(2, -1, np.int64)
    i = 0
    for r in range(nr):
        stop = rows[r]
        while i < stop:
            s = b_side[i]
            if b_reset[i]:
                _clear_side(levels, best, nlev, lo, hi, s)
            _apply_level(levels, best, nlev, lo, hi, s, b_idx[i], b_sz[i])
            i += 1
        for s in range(2):
            k = best[s]
            n = 0
            if k < 0:
                continue
            while n < depth and lo[s] <= k and k <= hi[s]:
                if levels[s, k] > 0:
                    out_px[r, s, n] = k
                    out_sz[r, s, n] = levels[s, k]
                    n += 1
                k += -1 if s == 0 else 1
    return out_px, out_sz


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def _price_base(book: BookStream) -> Tuple[int, int]:
    """(base tick, number of level slots) of the dense book arrays."""
    return _level_span(book.price)


def _meets_gaps(lo: np.ndarray, hi: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """For each closed interval [lo, hi], whether it meets any half-open gap [g_lo, g_hi)."""
    out = np.zeros(lo.shape, dtype=bool)
    for g_lo, g_hi in gaps.tolist():
        out |= (lo < g_hi) & (hi >= g_lo)
    return out


def _nullable(values: np.ndarray, valid: np.ndarray, dtype: Any) -> pl.Series:
    return pl.Series([v if ok else None for v, ok in zip(values.tolist(), valid.tolist())], dtype=dtype)


def simulate(book: BookStream, trades: TradeStream, decision_ts: Sequence[int],
             config: Optional[ProbeConfig] = None, *, day: Optional[str] = None) -> pl.DataFrame:
    """
    Run both fill models for every decision in ``decision_ts`` (int ms,
    ascending, at least one order life apart).

    Returns one row per (decision, side, model), in that order, with:

      * ``status``: ok, one_sided, crossed (best bid >= best ask), no_data
        (the book is not yet defined at t0, or either stream's coverage
        misses t0 or t0 + life) or gap ([t0, t0 + life] meets a gap of
        either stream). Every other column is null unless ok.
      * ``best_bid``, ``best_ask``, ``order_price`` (ticks, state at t0) and
        ``queue_at_post``, the displayed size at the order's price then.
      * ``filled``, ``fill_ts``; the fill price is ``order_price``.
      * ``markout_<k>_bps`` gross, from the fill price to the mid at
        fill_ts + k: buy (mid / fill - 1) * 1e4, sell (1 - mid / fill) * 1e4.
        The maker fee is not netted. Computed as +-(mid - fill) / fill * 1e4,
        the same quantity, exact for tick prices (mid - fill is a multiple of
        half a tick). ``markout_<k>_status`` says why one is missing:
        beyond_data, gap (inside a book gap), invalid_book (one-sided or
        crossed), no_fill.
      * ``expiry_best_bid``, ``expiry_best_ask``: the state at t0 + life.
      * Shortfalls in bp from the decision-time mid m0, fees included:
        ``shortfall_cross_now_bps`` = buy (ask0 / m0 - 1) * 1e4 + taker,
        sell (1 - bid0 / m0) * 1e4 + taker; ``shortfall_post_then_cross_bps``
        = the order price on the same formula + maker if filled, else the
        opposite best at t0 + life + taker; ``shortfall_diff_bps`` = post -
        cross, so a negative value means resting was cheaper. Post-then-cross
        and the difference are null whenever the book at t0 + life is
        one-sided or crossed, filled or not, so that both models' shortfalls
        cover the same decisions; cross-now is the same for both models.
      * ``queue_left``, ``level_size_left`` (M2 rows only): Q and the
        displayed size at the order's price when the order filled or expired.
    """
    cfg = _default(config)
    dec = _int_array("decision_ts", decision_ts)
    if dec.size == 0:
        raise ValueError("no decision times")
    if np.any(np.diff(dec) < cfg.order_life_ms):
        raise ValueError("decision times must ascend at least one order life apart (one live order per side)")
    lo_ms = min(book.start_ms, trades.start_ms)
    hi_ms = max(book.end_ms, trades.end_ms)
    if overlaps_protected(lo_ms, hi_ms):
        raise ValueError(f"input coverage [{lo_ms}, {hi_ms}) reaches into a protected test window (research.spec)")

    life = cfg.order_life_ms
    valid_from = book.valid_from_ms()
    covered = (
        (dec >= (valid_from if valid_from is not None else np.iinfo(np.int64).max))
        & (dec + life < book.end_ms)
        & (dec >= trades.start_ms)
        & (dec + life < trades.end_ms)
    )
    in_gap = _meets_gaps(dec, dec + life, book.gaps) | _meets_gaps(dec, dec + life, trades.gaps)
    base, n_levels = _price_base(book)
    mk_off = np.asarray(cfg.markout_ms, dtype=np.int64)
    (status, bid0, ask0, queue0, fill_ts, exp_bid, exp_ask, q_left, lvl_left,
     mk_sum, mk_st) = _simulate_kernel(
        book.ts_ms, (book.side == ASK).astype(np.int8), book.price - base, book.size, book.reset, book.msg,
        n_levels, trades.ts_ms, trades.price - base, trades.size,
        dec, covered & ~in_gap, np.int64(life), mk_off, np.int64(book.end_ms),
    )
    status[covered & in_gap] = _ST_GAP
    # A markout read inside a book gap is not trusted, whatever the state there.
    for k in range(mk_off.size):
        t_read = fill_ts + mk_off[k]
        st = mk_st[..., k]
        st[(fill_ts >= 0) & (st != _MK_BEYOND) & _meets_gaps(t_read, t_read, book.gaps)] = _MK_GAP

    nd = dec.size
    D = np.repeat(np.arange(nd), 4)
    S = np.tile(np.array([0, 0, 1, 1]), nd)
    M = np.tile(np.array([0, 1, 0, 1]), nd)
    ok = status[D] == _ST_OK
    bb = (bid0 + base).astype(np.float64)[D]
    ba = (ask0 + base).astype(np.float64)[D]
    m0 = (bb + ba) / 2.0
    is_buy = S == 0
    px = np.where(is_buy, bb, ba)
    ft = fill_ts[D, S, M]
    filled = ft >= 0
    sign = np.where(is_buy, 1.0, -1.0)

    cols: Dict[str, pl.Series] = {
        "day": pl.Series([day] * D.size, dtype=pl.Utf8),
        "decision_ts": pl.Series(dec[D], dtype=pl.Int64),
        "side": pl.Series(np.array(SIDES)[S].tolist(), dtype=pl.Utf8),
        "model": pl.Series(np.array(MODELS)[M].tolist(), dtype=pl.Utf8),
        "status": pl.Series(np.array(STATUS_NAMES)[status[D]].tolist(), dtype=pl.Utf8),
        "best_bid": _nullable(bid0[D] + base, ok, pl.Int64),
        "best_ask": _nullable(ask0[D] + base, ok, pl.Int64),
        "order_price": _nullable(np.where(is_buy, bid0[D], ask0[D]) + base, ok, pl.Int64),
        "queue_at_post": _nullable(queue0[D, S], ok, pl.Int64),
        "filled": _nullable(filled, ok, pl.Boolean),
        "fill_ts": _nullable(ft, ok & filled, pl.Int64),
    }
    for k, lab in enumerate(cfg.markout_labels):
        st = mk_st[D, S, M, k]
        good = ok & filled & (st == _MK_OK)
        mid_k = (mk_sum[D, S, M, k] + 2 * base) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            bps = sign * (mid_k - px) / px * 1e4
        cols[f"markout_{lab}_bps"] = pl.Series(np.where(good, bps, np.nan), dtype=pl.Float64, nan_to_null=True)
        cols[f"markout_{lab}_status"] = _nullable(np.array(MARKOUT_STATUS_NAMES)[st], ok, pl.Utf8)

    eb = exp_bid[D] + base
    ea = exp_ask[D] + base
    exp_valid = (exp_bid[D] >= 0) & (exp_ask[D] >= 0) & (exp_bid[D] < exp_ask[D])
    cols["expiry_best_bid"] = _nullable(eb, ok & (exp_bid[D] >= 0), pl.Int64)
    cols["expiry_best_ask"] = _nullable(ea, ok & (exp_ask[D] >= 0), pl.Int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        cross_px = np.where(is_buy, ba, bb)
        cross_now = sign * (cross_px - m0) / m0 * 1e4 + cfg.taker_fee_bps
        post_filled = sign * (px - m0) / m0 * 1e4 + cfg.maker_fee_bps
        late_px = np.where(is_buy, ea, eb).astype(np.float64)
        post_late = sign * (late_px - m0) / m0 * 1e4 + cfg.taker_fee_bps
    # The sample depends on the market at t0 + life only, never on a fill
    # model: a decision whose expiry book is invalid leaves both models.
    post = np.where(ok & exp_valid, np.where(filled, post_filled, post_late), np.nan)
    cross_now = np.where(ok, cross_now, np.nan)
    cols["shortfall_cross_now_bps"] = pl.Series(cross_now, dtype=pl.Float64, nan_to_null=True)
    cols["shortfall_post_then_cross_bps"] = pl.Series(post, dtype=pl.Float64, nan_to_null=True)
    cols["shortfall_diff_bps"] = pl.Series(post - cross_now, dtype=pl.Float64, nan_to_null=True)

    m2 = ok & (M == 1)
    cols["queue_left"] = _nullable(q_left[D, S], m2, pl.Int64)
    cols["level_size_left"] = _nullable(lvl_left[D, S], m2, pl.Int64)
    return pl.DataFrame([s.alias(name) for name, s in cols.items()])


def simulate_day(book: BookStream, trades: TradeStream, day: Union[str, date],
                 config: Optional[ProbeConfig] = None) -> pl.DataFrame:
    """``simulate`` over the 288 decision times of UTC day D, with ``day`` set to D (ISO)."""
    cfg = _default(config)
    label = day if isinstance(day, str) else day.isoformat()
    return simulate(book, trades, decision_times(label, cfg), cfg, day=label)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _shortfall_names() -> List[str]:
    return ["shortfall_cross_now_bps", "shortfall_post_then_cross_bps", "shortfall_diff_bps"]


def daily_metrics(config: Optional[ProbeConfig] = None) -> List[str]:
    """The per-day means that ``across_days`` averages over days."""
    cfg = _default(config)
    return (["fill_rate"] + [f"markout_{lab}_bps_mean" for lab in cfg.markout_labels]
            + [f"{n}_mean" for n in _shortfall_names()])


def daily_counts(config: Optional[ProbeConfig] = None) -> List[str]:
    """The per-day counts of ``daily_summary``, in column order."""
    cfg = _default(config)
    out = ["n_decisions", "n_skipped_one_sided", "n_skipped_crossed", "n_skipped_no_data", "n_skipped_gap",
           "n_fills"]
    for lab in cfg.markout_labels:
        out += [f"n_markout_{lab}", f"n_markout_{lab}_beyond_data", f"n_markout_{lab}_gap",
                f"n_markout_{lab}_invalid_book"]
    return out + ["n_shortfall", "n_shortfall_undefined"]


def daily_summary(orders: pl.DataFrame, config: Optional[ProbeConfig] = None) -> pl.DataFrame:
    """
    Per (day, side, model): counts, fill rate and mean markouts and shortfalls.

    ``n_decisions`` counts decisions with status ok; skipped ones are counted
    by reason. ``fill_rate`` = n_fills / n_decisions. A markout mean is over
    the fills whose markout is defined (``n_markout_<k>``); those beyond the
    data, in a gap or on an invalid book are counted, not imputed. A
    shortfall mean is over the decisions where post-then-cross is defined
    (``n_shortfall``), the same decisions for M1 and M2, so the cross-now
    mean is the same for both and each diff mean is the difference of its
    model's two other means. Means with nothing to average are null. Fees
    are columns of their own.
    """
    cfg = _default(config)
    ok = pl.col("status") == "ok"
    aggs = [
        ok.sum().cast(pl.Int64).alias("n_decisions"),
        (pl.col("status") == "one_sided").sum().cast(pl.Int64).alias("n_skipped_one_sided"),
        (pl.col("status") == "crossed").sum().cast(pl.Int64).alias("n_skipped_crossed"),
        (pl.col("status") == "no_data").sum().cast(pl.Int64).alias("n_skipped_no_data"),
        (pl.col("status") == "gap").sum().cast(pl.Int64).alias("n_skipped_gap"),
        pl.col("filled").sum().cast(pl.Int64).alias("n_fills"),
    ]
    for lab in cfg.markout_labels:
        st = pl.col(f"markout_{lab}_status")
        aggs += [
            (st == "ok").sum().cast(pl.Int64).alias(f"n_markout_{lab}"),
            (st == "beyond_data").sum().cast(pl.Int64).alias(f"n_markout_{lab}_beyond_data"),
            (st == "gap").sum().cast(pl.Int64).alias(f"n_markout_{lab}_gap"),
            (st == "invalid_book").sum().cast(pl.Int64).alias(f"n_markout_{lab}_invalid_book"),
            pl.col(f"markout_{lab}_bps").mean().alias(f"markout_{lab}_bps_mean"),
        ]
    post = pl.col("shortfall_post_then_cross_bps")
    aggs += [
        post.is_not_null().sum().cast(pl.Int64).alias("n_shortfall"),
        (ok & post.is_null()).sum().cast(pl.Int64).alias("n_shortfall_undefined"),
    ]
    aggs += [pl.col(n).filter(post.is_not_null()).mean().alias(f"{n}_mean") for n in _shortfall_names()]

    out = orders.group_by(["day", "side", "model"], maintain_order=True).agg(aggs)
    out = out.with_columns(
        pl.when(pl.col("n_decisions") > 0)
        .then(pl.col("n_fills") / pl.col("n_decisions"))
        .otherwise(None)
        .alias("fill_rate"),
        pl.lit(cfg.maker_fee_bps).alias("maker_fee_bps"),
        pl.lit(cfg.taker_fee_bps).alias("taker_fee_bps"),
        pl.lit(cfg.fees_verified).alias("fees_verified"),
    )
    order = ["day", "side", "model"] + daily_counts(cfg) + daily_metrics(cfg) + ["maker_fee_bps", "taker_fee_bps",
                                                                                  "fees_verified"]
    return out.select(order).sort(["day", "side", "model"])


def across_days(daily: pl.DataFrame, config: Optional[ProbeConfig] = None) -> pl.DataFrame:
    """
    Per (side, model, metric): the mean of the daily values and its standard
    error across days, sd(ddof=1) / sqrt(n_days).

    Days weigh equally: the fill rate is the mean of daily rates, not total
    fills over total decisions, and a day with 3 fills counts as much in a
    markout mean as one with 200. For a mean metric, ``n_days`` counts the
    days on which it is defined (a day without fills has no markout mean);
    sd and se are null below two days. For a count, ``total`` sums it over
    days and ``mean`` is per day.
    """
    cfg = _default(config)
    if daily.select(["day", "side", "model"]).is_duplicated().any():
        raise ValueError("daily summary has repeated (day, side, model) rows")
    rows: List[Dict[str, Any]] = []
    for side in SIDES:
        for model in MODELS:
            g = daily.filter((pl.col("side") == side) & (pl.col("model") == model))
            if g.height == 0:
                continue
            for metric in daily_metrics(cfg):
                v = g[metric].drop_nulls().to_numpy().astype(np.float64)
                v = v[np.isfinite(v)]
                n = int(v.size)
                sd = float(np.std(v, ddof=1)) if n >= 2 else None
                rows.append({"side": side, "model": model, "metric": metric, "n_days": n, "total": None,
                             "mean": float(v.mean()) if n else None, "sd": sd,
                             "se": sd / math.sqrt(n) if sd is not None else None})
            for metric in daily_counts(cfg):
                v = g[metric].to_numpy()
                rows.append({"side": side, "model": model, "metric": metric, "n_days": int(g.height),
                             "total": int(v.sum()), "mean": float(v.mean()), "sd": None, "se": None})
    schema = {"side": pl.Utf8, "model": pl.Utf8, "metric": pl.Utf8, "n_days": pl.Int64, "total": pl.Int64,
              "mean": pl.Float64, "sd": pl.Float64, "se": pl.Float64}
    out = pl.DataFrame(rows, schema=schema)
    return out.with_columns(
        pl.lit(cfg.maker_fee_bps).alias("maker_fee_bps"),
        pl.lit(cfg.taker_fee_bps).alias("taker_fee_bps"),
        pl.lit(cfg.fees_verified).alias("fees_verified"),
    )


# --------------------------------------------------------------------------
# rebuilt book against the archive's snapshots
# --------------------------------------------------------------------------

class Mismatch(NamedTuple):
    """One level, by rank, where the rebuilt book and a reference differ. A missing level is price None, size 0."""

    side: str
    rank: int
    rebuilt_price: Optional[int]
    rebuilt_size: int
    reference_price: Optional[int]
    reference_size: int


@dataclass(frozen=True)
class SnapshotCheck:
    """
    ``check_snapshots``'s result: snapshot sides compared, how many differed
    at any rank and how many at rank 0 (the touch, which sets the order
    price, the queue and the mids), and the mismatches themselves.
    """

    n_checked: int
    n_with_mismatch: int
    n_touch_mismatch: int
    mismatches: pl.DataFrame


def rebuilt_levels_before(book: BookStream, rows: Sequence[int], depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    The top ``depth`` levels of each side of the book rebuilt from rows
    [0, r), for each r in ``rows``.

    Returns (prices, sizes), each shaped (len(rows), 2, depth): axis 1 is
    bid then ask, levels best first, prices in ticks; a side with fewer
    levels is padded with price -1 and size 0.
    """
    r = _int_array("rows", rows)
    if depth <= 0:
        raise ValueError("depth must be positive")
    if r.size and (r.min() < 0 or r.max() > len(book)):
        raise ValueError("row index outside the stream")
    order = np.argsort(r, kind="stable")
    base, n_levels = _price_base(book)
    px, sz = _levels_before_kernel((book.side == ASK).astype(np.int8), book.price - base, book.size,
                                   book.reset, n_levels, r[order], int(depth))
    px = np.where(px >= 0, px + base, -1)
    out_px = np.empty_like(px)
    out_sz = np.empty_like(sz)
    out_px[order] = px
    out_sz[order] = sz
    return out_px, out_sz


def snapshot_levels(book: BookStream, row: int, depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    The levels a snapshot of one side sets, best first, at most ``depth``.

    ``row`` is the side's reset row; the snapshot is that row and the side's
    other rows in the same message (BookStream's contract), so an update
    message stamped in the same millisecond is not part of it. Returns
    (prices, sizes).
    """
    if not (0 <= row < len(book)) or not book.reset[row]:
        raise ValueError(f"row {row} is not a reset row")
    s, g = int(book.side[row]), int(book.msg[row])
    sizes: Dict[int, int] = {}
    k = row
    while k < len(book) and book.msg[k] == g:
        if book.side[k] == s:
            sizes[int(book.price[k])] = int(book.size[k])
        k += 1
    prices = sorted((p for p, z in sizes.items() if z > 0), reverse=(s == BID))[:depth]
    return np.array(prices, dtype=np.int64), np.array([sizes[p] for p in prices], dtype=np.int64)


def compare_book(rebuilt_bids: Tuple[Sequence[int], Sequence[int]], rebuilt_asks: Tuple[Sequence[int], Sequence[int]],
                 ref_bids: Tuple[Sequence[int], Sequence[int]], ref_asks: Tuple[Sequence[int], Sequence[int]],
                 depth: int) -> List[Mismatch]:
    """
    Compare a rebuilt book with a reference snapshot over the top ``depth``
    levels of each side, rank by rank.

    Each argument is (prices, sizes), best first; price -1 or size 0 marks a
    missing level. A level differs if its price or size differs, or if only
    one book has it. Rank-wise comparison means a single missing level shows
    as a mismatch at every deeper rank; the first mismatching rank is the
    informative one.
    """
    out: List[Mismatch] = []
    for name, mine, ref in (("bid", rebuilt_bids, ref_bids), ("ask", rebuilt_asks, ref_asks)):
        a = _levels_list(mine, depth)
        b = _levels_list(ref, depth)
        for rank in range(depth):
            pa, za = a[rank] if rank < len(a) else (None, 0)
            pb, zb = b[rank] if rank < len(b) else (None, 0)
            if pa != pb or za != zb:
                out.append(Mismatch(name, rank, pa, za, pb, zb))
    return out


def _levels_list(levels: Tuple[Sequence[int], Sequence[int]], depth: int) -> List[Tuple[int, int]]:
    px, sz = levels
    return [(int(p), int(z)) for p, z in zip(list(px)[:depth], list(sz)[:depth]) if p >= 0 and z > 0]


def check_snapshots(book: BookStream, depth: int, rows: Optional[Sequence[int]] = None) -> SnapshotCheck:
    """
    Compare, at each snapshot, the side as rebuilt from the rows before it
    with what the snapshot sets (``compare_book``, top ``depth`` levels).

    ``rows`` are reset rows; by default every reset row except each side's
    first, before which nothing has been rebuilt. Run it on the parsed
    archive day before ``simulate``: §11 checks the rebuilt book against the
    archive's snapshots (§11 expected one every 15 minutes; the archive has
    one every minute, and each is checked). ``after_gap`` marks mismatches at a
    snapshot that ends one of ``book.gaps``, where a difference is expected.
    """
    if rows is None:
        resets = np.flatnonzero(book.reset)
        first = {int(np.flatnonzero(book.reset & (book.side == s))[0]) for s in (BID, ASK)
                 if (book.reset & (book.side == s)).any()}
        r = np.array([k for k in resets.tolist() if k not in first], dtype=np.int64)
    else:
        r = _int_array("rows", rows)
    gap_ends = set(book.gaps[:, 1].tolist())
    records: List[Dict[str, Any]] = []
    n_bad = 0
    n_touch = 0
    if r.size:
        px, sz = rebuilt_levels_before(book, r, depth)
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        for n, row in enumerate(r.tolist()):
            s = int(book.side[row])
            si = 0 if s == BID else 1
            mine = (px[n, si], sz[n, si])
            ref = snapshot_levels(book, row, depth)
            if s == BID:
                found = compare_book(mine, empty, ref, empty, depth)
            else:
                found = compare_book(empty, mine, empty, ref, depth)
            n_bad += bool(found)
            n_touch += any(m.rank == 0 for m in found)
            t = int(book.ts_ms[row])
            records += [{"row": row, "ts_ms": t, "after_gap": t in gap_ends, **m._asdict()} for m in found]
    schema = {"row": pl.Int64, "ts_ms": pl.Int64, "after_gap": pl.Boolean, "side": pl.Utf8, "rank": pl.Int64,
              "rebuilt_price": pl.Int64, "rebuilt_size": pl.Int64, "reference_price": pl.Int64,
              "reference_size": pl.Int64}
    return SnapshotCheck(int(r.size), n_bad, n_touch, pl.DataFrame(records, schema=schema))


def merge_gaps(gaps: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Half-open intervals sorted, with those that overlap or touch merged."""
    out: List[Tuple[int, int]] = []
    for lo, hi in sorted((int(a), int(b)) for a, b in gaps):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def touch_mismatch_gaps(book: BookStream, check: SnapshotCheck) -> List[Tuple[int, int]]:
    """
    ``book.gaps`` merged with the gaps that ``check`` (from
    ``check_snapshots(book, ...)``) implies.

    A snapshot side whose best level differs from the rebuilt book (rank 0,
    in price or size) shows that the stream went wrong somewhere since that
    side's previous snapshot, so the states in between are not trusted: the
    gap is [previous snapshot + 1 ms, this snapshot). A snapshot that ends
    one of ``book.gaps`` is expected to differ and adds nothing. Use the
    result as ``dataclasses.replace(book, gaps=...)``, which checks that
    every gap ends at a snapshot of both sides.
    """
    found = [tuple(g) for g in book.gaps.tolist()]
    touch = check.mismatches.filter((pl.col("rank") == 0) & ~pl.col("after_gap"))["row"].unique().sort()
    resets = {s: np.flatnonzero(book.reset & (book.side == s)) for s in (BID, ASK)}
    for row in touch.to_list():
        own = resets[int(book.side[row])]
        k = int(np.searchsorted(own, row))
        if k == 0:
            continue                      # the side's first snapshot: nothing was rebuilt before it
        lo, hi = int(book.ts_ms[own[k - 1]]) + 1, int(book.ts_ms[row])
        if lo < hi:
            found.append((lo, hi))
    return merge_gaps(found)


# --------------------------------------------------------------------------
# archive parsers: written once the files were downloaded
# --------------------------------------------------------------------------

# The archive as read on 2026-11-24, after doge_fwd had run and before any
# probe statistic was computed (pre-registration §12):
#
#   order book (module 4), a .tar.gz holding one file: one JSON object per
#   line, {"instId", "action": "snapshot" | "update", "ts", "asks", "bids"},
#   each level [price, size, orders] as strings. There is no sequence number
#   and no checksum. Each minute opens with a snapshot of both sides (400
#   levels each, stamped a few ms past the minute), not every 15 minutes as
#   §11 assumed; updates come in batches about 10 ms apart, each setting a
#   level's new total size (0 deletes it). A level that leaves the top 400 is
#   deleted explicitly: the rebuilt book never holds more than 400 levels a
#   side (``max_depth`` checks this on every day).
#   trades (module 1), a .zip holding one CSV: instrument_name, trade_id,
#   side (the taker's), price, size, created_time (UTC ms), and from the
#   files dated 2025-11-02 on a seventh column, source (0 or 1); trade ids are
#   consecutive across both sources. Source-1 trades do not behave as prints
#   of the public book: on 2026-09-11, 12% of them printed at the touch and
#   27% at or beyond the opposite best, and none matched in size the book
#   decrease that followed, against 73%, under 1% and 60% for source 0. They
#   are kept for the trade-id check and left out of the fill rules.
#
# Prices are whole multiples of DOGE-USDT-SWAP's tick, 0.00001. Book and trade
# sizes carry at most two decimals, in contracts: one unit is 0.01 contract,
# OKX's lot size for this swap; trade_book_lags checks that a trade's size
# and the book decrease it causes are in the same unit.
BOOK_DEPTH = 400
TICK_SIZE = 0.00001
SIZE_UNIT = 0.01
# Without sequence numbers a lost stretch of the stream can only show as a
# pause. Updates arrive about every 10 ms, and the longest pause on the first
# day read was 0.6 s; a pause of MAX_SILENCE_MS or more counts as missing
# data (decided before any probe statistic was computed).
MAX_SILENCE_MS = 5_000
_GRID_TOL = 1e-6              # a price or size this far off its grid, in units, is refused
_CHUNK_MESSAGES = 250_000
_TRADE_COLUMNS = ["instrument_name", "trade_id", "side", "price", "size", "created_time"]
_SOURCE = "source"


def _to_units(values: np.ndarray, unit: float, what: str) -> np.ndarray:
    """Decimal strings (or floats) as exact int64 multiples of ``unit``; raise on any value off the grid."""
    x = np.asarray(values, dtype=np.float64) / unit
    r = np.rint(x)
    bad = np.flatnonzero(np.abs(x - r) > _GRID_TOL)
    if bad.size:
        raise ValueError(f"{what} {values[bad[0]]!r} is not a whole multiple of {unit}")
    return r.astype(np.int64)


def _book_messages(paths: Sequence[Path]):
    """Each order-book message of the files in turn, in file order, as parsed JSON."""
    for path in paths:
        with tarfile.open(path, "r|gz") as tf:
            n_files = 0
            for member in tf:
                if not member.isfile():
                    continue
                n_files += 1
                if n_files > 1:
                    raise ValueError(f"{Path(path).name}: more than one file in the archive")
                for line in tf.extractfile(member):
                    if line.strip():
                        yield json.loads(line)


def read_book(paths: Sequence[Path], day: Union[str, date], config: Optional[ProbeConfig] = None, *,
              inst_id: str = "DOGE-USDT-SWAP", report: Optional[Dict[str, Any]] = None) -> BookStream:
    """
    Parse OKX's 400-level order-book archive (module 4) for probe day D into
    a BookStream.

    ``paths`` are the files dated D and D+1, in that order. They are dated in
    UTC; each covers its date's 00:00 -> 24:00 UTC, and D+1 supplies the
    markouts of D's last decisions. Only messages stamped in
    ``required_coverage(D)`` = [D 00:00, D+1 00:11:00.001) are kept, and
    reading stops at the first message past it. If the files end before it,
    ``end_ms`` is lowered to 1 ms past the last message, and the markouts
    beyond are counted as beyond the data.

    Rows keep the files' order: they are never re-sorted by time, since a row
    sets a level's total size and two rows applied out of order would leave a
    wrong level until the next snapshot. ``msg`` numbers the messages; every
    row of a message carries its timestamp. A message stamped earlier than
    the one before it is raised to that one's timestamp, and counted.

    A snapshot message becomes, for each side, a reset row carrying its best
    level followed by the side's other levels (an empty side is a reset row
    of size 0); an update message becomes one row per level, with the new
    total size (0 deletes the level).

    Gaps. The archive has neither sequence numbers nor checksums, so a
    sequence break cannot be seen directly; a pause of MAX_SILENCE_MS or
    more between two messages is treated as missing data, a gap from the
    last message before it to the next snapshot. A pause with no snapshot
    after it lowers ``end_ms`` to 1 ms past the last message before it, and
    the rows after it are dropped.

    ``report``, if given, receives the counts the run reports: messages,
    rows, snapshots, other instruments' messages skipped, clamped messages
    and the largest clamp, the longest pause, the pauses treated as missing
    data, and whether the files ended inside the coverage.

    Then run ``check_snapshots(book, BOOK_DEPTH)``. The probe is simulated
    twice (scripts/run_maker_probe.py; pre-registration §12, 2026-11-24):
    with only the gaps found here, and with ``touch_mismatch_gaps`` added,
    which drops the interval since a side's previous snapshot wherever a
    snapshot's best level differs from the rebuilt book. Both are reported.
    """
    cfg = _default(config)
    start, end = required_coverage(day, cfg)
    ts_parts: List[np.ndarray] = []
    side_parts: List[np.ndarray] = []
    px_parts: List[np.ndarray] = []
    sz_parts: List[np.ndarray] = []
    reset_parts: List[np.ndarray] = []
    msg_parts: List[np.ndarray] = []
    c_ts: List[int] = []
    c_side: List[int] = []
    c_px: List[str] = []
    c_sz: List[str] = []
    c_reset: List[bool] = []
    c_msg: List[int] = []

    def flush() -> None:
        if c_ts:
            ts_parts.append(np.array(c_ts, dtype=np.int64))
            side_parts.append(np.array(c_side, dtype=np.int8))
            px_parts.append(_to_units(np.array(c_px, dtype=np.float64), TICK_SIZE, "price"))
            sz_parts.append(_to_units(np.array(c_sz, dtype=np.float64), SIZE_UNIT, "size"))
            reset_parts.append(np.array(c_reset, dtype=np.bool_))
            msg_parts.append(np.array(c_msg, dtype=np.int64))
            for buf in (c_ts, c_side, c_px, c_sz, c_reset, c_msg):
                buf.clear()

    n_msg = n_snap = n_other = n_clamped = max_clamp = longest = 0
    last_ts: Optional[int] = None
    last_px = {BID: None, ASK: None}
    silences: List[Tuple[int, int]] = []
    snapshot_ts: List[int] = []
    reached_end = False
    for obj in _book_messages(paths):
        if obj.get("instId") != inst_id:
            n_other += 1
            continue
        ts = int(obj["ts"])
        if ts < start:
            continue
        if ts >= end:
            reached_end = True
            break
        action = obj.get("action")
        if action not in ("snapshot", "update"):
            raise ValueError(f"message at {ts}: unknown action {action!r}")
        if last_ts is not None:
            if ts < last_ts:
                n_clamped += 1
                max_clamp = max(max_clamp, last_ts - ts)
                ts = last_ts
            pause = ts - last_ts
            longest = max(longest, pause)
            if pause >= MAX_SILENCE_MS:
                silences.append((last_ts, ts))
        last_ts = ts
        snap = action == "snapshot"
        if snap:
            n_snap += 1
            snapshot_ts.append(ts)
        for key, s in (("bids", BID), ("asks", ASK)):
            levels = obj.get(key) or []
            if snap and not levels:
                other = obj.get("asks" if s == BID else "bids") or []
                p = last_px[s] or (other[0][0] if other else last_px[-s])
                if p is None:
                    raise ValueError(f"message at {ts}: a snapshot with both sides empty")
                levels = [[p, "0", "0"]]
            if not levels:
                continue
            k = len(levels)
            c_ts.extend([ts] * k)
            c_side.extend([s] * k)
            c_px.extend(lv[0] for lv in levels)
            c_sz.extend(lv[1] for lv in levels)
            c_reset.extend([snap] + [False] * (k - 1))
            c_msg.extend([n_msg] * k)
            last_px[s] = levels[0][0]
        n_msg += 1
        if n_msg % _CHUNK_MESSAGES == 0:
            flush()
    flush()
    ts_all = np.concatenate(ts_parts) if ts_parts else np.zeros(0, np.int64)
    cols = [ts_all,
            np.concatenate(side_parts) if side_parts else np.zeros(0, np.int8),
            np.concatenate(px_parts) if px_parts else np.zeros(0, np.int64),
            np.concatenate(sz_parts) if sz_parts else np.zeros(0, np.int64),
            np.concatenate(reset_parts) if reset_parts else np.zeros(0, np.bool_),
            np.concatenate(msg_parts) if msg_parts else np.zeros(0, np.int64)]
    if last_ts is None:
        raise ValueError(f"no order-book message of {inst_id} in [{start}, {end})")
    end_ms = end if reached_end else min(end, last_ts + 1)
    snaps = np.array(snapshot_ts, dtype=np.int64)
    gaps: List[Tuple[int, int]] = []
    for lo, after in silences:
        k = int(np.searchsorted(snaps, after))
        if k < snaps.size:
            gaps.append((lo, int(snaps[k])))
        else:
            end_ms = min(end_ms, lo + 1)
            break
    keep = ts_all < end_ms
    if not keep.all():
        cols = [c[keep] for c in cols]
    gaps = [(lo, hi) for lo, hi in merge_gaps(gaps) if hi <= end_ms]
    if report is not None:
        report.update(messages=n_msg, rows=int(cols[0].size), snapshots=n_snap, other_instruments=n_other,
                      clamped=n_clamped, max_clamp_ms=max_clamp, longest_pause_ms=longest,
                      pauses_as_gaps=[list(g) for g in silences], ended_early=not reached_end,
                      start_ms=start, end_ms=end_ms)
    return BookStream(ts_ms=cols[0], side=cols[1], price=cols[2], size=cols[3], reset=cols[4], msg=cols[5],
                      start_ms=start, end_ms=end_ms, tick_size=TICK_SIZE, size_unit=SIZE_UNIT, gaps=gaps)


def read_trades(paths: Sequence[Path], day: Union[str, date], config: Optional[ProbeConfig] = None, *,
                inst_id: str = "DOGE-USDT-SWAP", report: Optional[Dict[str, Any]] = None) -> TradeStream:
    """
    Parse OKX's trade archive (module 1) for probe day D into a TradeStream.

    ``paths`` are the files dated D and D+1. They are dated in UTC+8: the file
    dated D covers D-1 16:00 -> D 16:00 UTC, so UTC day D needs both. The two
    are joined, de-duplicated on trade id (a repeated id must repeat the
    whole row) and cropped to ``required_coverage(D)``; ``end_ms`` is lowered
    to 1 ms past the last trade if no trade in the files lies past the
    coverage.

    Trades whose ``source`` is 1 are not prints of the public book (see the
    format notes above): they count in the trade-id check below and are then
    left out. Files without the column have source-0 trades only.

    Each trade keeps its own timestamp; rows are sorted by (ts, trade id).
    Prices become ticks and sizes units of 0.01 contract, exactly, in the
    BookStream's units. ``aggressor`` is BUY for side "buy" (the taker
    bought) and SELL for "sell". One row per print: a taker order that
    matched several makers stays several rows.

    Gaps. Trade ids are consecutive, so a jump in id between two trades of
    the coverage is a gap from the earlier of their timestamps to 1 ms past
    the later. ``report``, if given, receives the counts the run reports.
    """
    cfg = _default(config)
    start, end = required_coverage(day, cfg)
    frames = []
    for path in paths:
        with zipfile.ZipFile(path) as zf:
            names = [i.filename for i in zf.infolist() if not i.is_dir()]
            if len(names) != 1:
                raise ValueError(f"{Path(path).name}: expected one file in the archive, found {len(names)}")
            raw = zf.read(names[0])
        # Every column as text, under the file's own names: polars 0.20 renames
        # the columns by position when given a dtypes mapping of full width.
        df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
        if df.columns not in (_TRADE_COLUMNS, _TRADE_COLUMNS + [_SOURCE]):
            raise ValueError(f"{Path(path).name}: columns {df.columns}, expected {_TRADE_COLUMNS} "
                             f"(and optionally {_SOURCE!r})")
        if _SOURCE not in df.columns:
            df = df.with_columns(pl.lit("0").alias(_SOURCE))
        frames.append(df)
    df = (pl.concat(frames) if frames
          else pl.DataFrame(schema={c: pl.Utf8 for c in _TRADE_COLUMNS + [_SOURCE]}))
    n_other = int((df["instrument_name"] != inst_id).sum())
    df = df.filter(pl.col("instrument_name") == inst_id).with_columns(
        pl.col("trade_id").cast(pl.Int64), pl.col("created_time").cast(pl.Int64))
    n_rows = df.height
    distinct = df.unique(maintain_order=True)
    if distinct["trade_id"].n_unique() != distinct.height:
        raise ValueError("a trade id repeats with different contents")
    n_duplicates = n_rows - distinct.height
    past_end = bool((distinct["created_time"] >= end).any())
    df = distinct.filter((pl.col("created_time") >= start) & (pl.col("created_time") < end))
    if df.height == 0:
        raise ValueError(f"no trade of {inst_id} in [{start}, {end})")
    bad_side = df.filter(~pl.col("side").is_in(["buy", "sell"]))
    if bad_side.height:
        raise ValueError(f"trade {bad_side['trade_id'][0]}: side {bad_side['side'][0]!r}")
    bad_source = df.filter(~pl.col(_SOURCE).is_in(["0", "1"]))
    if bad_source.height:
        raise ValueError(f"trade {bad_source['trade_id'][0]}: source {bad_source[_SOURCE][0]!r}")
    by_id = df.sort("trade_id")
    ids = by_id["trade_id"].to_numpy()
    id_ts = by_id["created_time"].to_numpy()
    jumps = np.flatnonzero(np.diff(ids) != 1)
    gaps = merge_gaps([(int(min(id_ts[j], id_ts[j + 1])), int(max(id_ts[j], id_ts[j + 1])) + 1) for j in jumps])
    n_source_1 = int((df[_SOURCE] == "1").sum())
    df = df.filter(pl.col(_SOURCE) == "0").sort(["created_time", "trade_id"])
    if df.height == 0:
        raise ValueError(f"no source-0 trade of {inst_id} in [{start}, {end})")
    ts = df["created_time"].to_numpy().astype(np.int64)
    end_ms = end if past_end else int(ts[-1]) + 1
    gaps = [(lo, min(hi, end_ms)) for lo, hi in gaps if lo < end_ms]
    if report is not None:
        report.update(trades=df.height, source_1_trades=n_source_1, other_instruments=n_other,
                      duplicate_rows=n_duplicates,
                      id_jumps=int(jumps.size), ids_out_of_time_order=int(np.sum(np.diff(id_ts) < 0)),
                      ended_early=not past_end, start_ms=start, end_ms=end_ms)
    return TradeStream(
        ts_ms=ts,
        price=_to_units(df["price"].cast(pl.Float64).to_numpy(), TICK_SIZE, "price"),
        size=_to_units(df["size"].cast(pl.Float64).to_numpy(), SIZE_UNIT, "size"),
        aggressor=np.where(df["side"].to_numpy() == "buy", BUY, SELL).astype(np.int64),
        start_ms=start, end_ms=end_ms, gaps=gaps)


# --------------------------------------------------------------------------
# checks of the parsed day
# --------------------------------------------------------------------------

@njit(fastmath=False, cache=False)
def _depth_kernel(b_side, b_idx, b_sz, b_reset, b_msg, n_levels):
    """Largest number of levels each side holds at the end of any message."""
    levels = np.zeros((2, n_levels), np.int64)
    best = np.full(2, -1, np.int64)
    nlev = np.zeros(2, np.int64)
    lo = np.full(2, n_levels, np.int64)
    hi = np.full(2, -1, np.int64)
    out = np.zeros(2, np.int64)
    n = b_side.shape[0]
    for i in range(n):
        s = b_side[i]
        if b_reset[i]:
            _clear_side(levels, best, nlev, lo, hi, s)
        _apply_level(levels, best, nlev, lo, hi, s, b_idx[i], b_sz[i])
        if i == n - 1 or b_msg[i + 1] != b_msg[i]:
            for k in range(2):
                if nlev[k] > out[k]:
                    out[k] = nlev[k]
    return out


def max_depth(book: BookStream) -> Tuple[int, int]:
    """(bid, ask): the most levels a side of the rebuilt book holds after any message."""
    if not len(book):
        return 0, 0
    base, n_levels = _price_base(book)
    out = _depth_kernel((book.side == ASK).astype(np.int8), book.price - base, book.size, book.reset,
                        book.msg, n_levels)
    return int(out[0]), int(out[1])


@njit(fastmath=False, cache=False)
def _old_size_kernel(b_side, b_idx, b_sz, b_reset, n_levels):
    """For each row, the size its level had just before the row (0 after a reset of its side)."""
    levels = np.zeros((2, n_levels), np.int64)
    best = np.full(2, -1, np.int64)
    nlev = np.zeros(2, np.int64)
    lo = np.full(2, n_levels, np.int64)
    hi = np.full(2, -1, np.int64)
    old = np.zeros(b_side.shape[0], np.int64)
    for i in range(b_side.shape[0]):
        s = b_side[i]
        if b_reset[i]:
            _clear_side(levels, best, nlev, lo, hi, s)
        old[i] = _apply_level(levels, best, nlev, lo, hi, s, b_idx[i], b_sz[i])
    return old


def trade_book_lags(book: BookStream, trades: TradeStream, window_ms: int = 100) -> Dict[str, Any]:
    """
    How the two clocks and the two size units line up. For each trade, the
    first decrease of the level at its price, on the side it hit (the bid
    for a taker sell, the ask for a taker buy), in an update message stamped
    from the trade's millisecond to ``window_ms`` after it.

    Returns the share of trades with such a decrease, quantiles of its lag
    (book time minus trade time, ms) and the shares within 10 and 20 ms; the
    share of trades whose level also decreased in the 10 ms before them (a
    book clock running ahead would raise it); and, for trades alone at their
    price and side in their millisecond, the share whose decrease equals the
    trade's size and the median ratio of the two. A size unit that differed
    would put that ratio far from 1.
    """
    if not len(book) or not len(trades):
        return {"trades": len(trades), "matched_share": None}
    base, n_levels = _price_base(book)
    side_idx = (book.side == ASK).astype(np.int8)
    old = _old_size_kernel(side_idx, book.price - base, book.size, book.reset, n_levels)
    snap_msgs = np.unique(book.msg[book.reset])
    dec = (old > book.size) & ~np.isin(book.msg, snap_msgs)
    d_ts, d_side = book.ts_ms[dec], side_idx[dec].astype(np.int64)
    d_px, d_amt = book.price[dec], (old - book.size)[dec]
    t0 = min(int(book.start_ms), int(trades.start_ms)) - window_ms
    shift = 31                                          # ms since t0 fit in 31 bits (24 days)
    key = ((d_side << 24) | d_px) << shift | (d_ts - t0)
    order = np.argsort(key, kind="stable")
    key, d_ts, d_amt = key[order], d_ts[order], d_amt[order]
    t_side = (trades.aggressor == BUY).astype(np.int64)  # a taker buy hits the ask (side index 1)
    level = ((t_side << 24) | trades.price) << shift

    def first_from(lo_ms: np.ndarray, hi_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = np.minimum(np.searchsorted(key, level | (lo_ms - t0), side="left"), key.size - 1)
        hit = ((key[k] >> shift) == (level >> shift)) & (d_ts[k] >= lo_ms) & (d_ts[k] <= hi_ms)
        return hit, k

    same, kk = first_from(trades.ts_ms, trades.ts_ms + window_ms)
    before, _ = first_from(trades.ts_ms - 10, trades.ts_ms - 1)
    lag = (d_ts[kk] - trades.ts_ms)[same]
    out: Dict[str, Any] = {"trades": len(trades), "matched_share": float(same.mean()),
                           "decrease_in_10ms_before_share": float(before.mean())}
    if lag.size:
        q = np.quantile(lag, [0.05, 0.25, 0.5, 0.75, 0.95])
        out.update(lag_ms_q05=float(q[0]), lag_ms_q25=float(q[1]), lag_ms_median=float(q[2]),
                   lag_ms_q75=float(q[3]), lag_ms_q95=float(q[4]),
                   lag_within_10ms_share=float(np.mean(lag <= 10)), lag_within_20ms_share=float(np.mean(lag <= 20)))
    tk = pl.DataFrame({"ts": trades.ts_ms, "side": t_side, "px": trades.price})
    alone = tk.select(pl.struct(["ts", "side", "px"]).is_unique()).to_series().to_numpy()
    pick = same & alone
    if pick.any():
        amt = d_amt[kk][pick]
        size = trades.size[pick]
        out.update(alone_matched=int(pick.sum()), decrease_equals_size_share=float(np.mean(amt == size)),
                   decrease_over_size_median=float(np.median(amt / size)))
    return out
