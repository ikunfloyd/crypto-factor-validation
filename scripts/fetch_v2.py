#!/usr/bin/env python3
"""
Fetch OKX history for the pre-registered out-of-sample study (§4.2).

Two series, both from OKX's public endpoints (no key, no signing):

    candles   1-minute bars   /api/v5/market/history-candles   (confirm == "1" only)
    funding   settlements     monthly archive (module 3), then
                              /api/v5/public/funding-rate-history for months
                              the archive does not have yet

Each run writes one parquet file per (series, instrument) and one manifest
(§4.2: source per UTC day, rows, first/last timestamp, gaps, SHA-256).
Nothing is ever overwritten: an existing output or manifest stops the run
before the first request for data.

§10 step 4 commits one manifest per set, so one invocation can fetch
everything a set needs: --candles for the perpetuals and their spot legs,
--funding for the perpetuals, all over the same window and into the same
manifest. The single-kind form (--kind with --inst-id) still works. Either
way the guard below judges the whole request at once.

The guard matters more than the fetching. The study is only confirmatory if
no test-set data is downloaded before the freeze (§10), so a window that
overlaps a protected range (research.spec.protected_windows) is refused
unless --allow-test-data is given *and* git shows the freeze was done as §10
steps 2-3 say (research.freeze.verify_freeze: amendment committed, matching
the tag, pushed), with the push confirmed by the public repository itself
(git ls-remote), not by a local ref. A protected range that has not closed
yet (doge_fwd until 2026-11-23 20:00 UTC) is refused even then: §10 puts its
download after its end. Whether it has closed is judged by OKX's clock, and
a local clock more than a minute away from it stops the run. A window that
merely borders a protected range is allowed, but the
requests are shaped so that they cannot return data from across the border:
every page of a backward walk asks for one slice [floor, cursor) of at most
``limit`` grid steps, bounded on both sides with ``before`` and ``after``, so
no hole in the data can carry a page below the window's start, and a window
lying in the future gets empty pages rather than OKX's latest bars. An
archive month whose file would contain protected settlements is not
downloaded while test data is locked, nor while the protected range it
touches is still open.

What the guards are for: they stop every accident and every single-step
mistake, and they make any deliberate deviation a visible, deliberate act.
They cannot stop a determined author who controls this machine; the public,
timestamped pre-registration (GitHub Release and Software Heritage) and
reproducibility from public data cover that.

Gaps are reported, never filled.

Usage:
    python scripts/fetch_v2.py --kind candles --inst-id DOGE-USDT \\
        --start '2025-09-24 00:00' --end '2026-09-23 20:00' \\
        --out-dir data/history/is_supp --manifest results/manifest_is_supp_candles.json
    python scripts/fetch_v2.py --kind funding --inst-id DOGE-USDT-SWAP \\
        --start '2025-09-24 00:00' --end '2026-09-23 20:00' \\
        --out-dir data/history/is_supp --manifest results/manifest_is_supp_funding.json

    # One manifest per set (what research/pipeline.py reads), e.g. the
    # in-sample supplement and, after the freeze, the primary set:
    python scripts/fetch_v2.py --candles DOGE-USDT --funding DOGE-USDT-SWAP \\
        --start '2025-09-24 00:00' --end '2026-09-23 20:00' \\
        --out-dir data/history/is_supp --manifest results/manifest_is_supp.json
    python scripts/fetch_v2.py --candles DOGE-USDT-SWAP DOGE-USDT --funding DOGE-USDT-SWAP \\
        --start '2023-06-25 20:00' --end '2024-09-23 20:00' --allow-test-data \\
        --out-dir data/history/oos/doge_2023 --manifest results/oos/manifest_doge_2023.json
"""
from __future__ import annotations

import argparse
import atexit
import csv
import http.client
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
# Startup, the same in run_oos.py, run_insample_v2.py and fetch_v2.py. This
# checkout's packages come first on sys.path, ahead of any namesake: a
# PYTHONPATH entry naming another checkout (the project's .env sets one) would
# otherwise supply ``research`` while the git checks inspect this one
# (research/integrity.py; run() refuses if it happens all the same).
sys.path[:] = [str(REPO_ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]
if __name__ == "__main__":
    # Compile this checkout's modules from source into a fresh bytecode cache,
    # removed at exit: a __pycache__ .pyc whose header carries the source's
    # mtime and size would otherwise run in place of the .py file the git
    # checks vouch for. That covers ``python scripts/<name>.py``, the documented
    # invocation. Under ``python -m scripts.<name>``, runpy has already read
    # this script's own cached bytecode before this line runs. (The third-party
    # modules imported above are not this checkout's code.)
    sys.pycache_prefix = tempfile.mkdtemp(prefix=f"{Path(__file__).stem}-pycache-")
    atexit.register(shutil.rmtree, sys.pycache_prefix, True)

from loguru import logger

import research.spec as spec
from research import freeze, gitrepo, integrity
from research.io import manifest_entry, write_once_json, write_once_parquet

OKX_REST_BASE = "https://www.okx.com"
CANDLES_PATH = "/api/v5/market/history-candles"
FUNDING_REST_PATH = "/api/v5/public/funding-rate-history"
ARCHIVE_PATH = "/api/v5/public/market-data-history"
TIME_PATH = "/api/v5/public/time"

CANDLE_PAGE_LIMIT = 300
FUNDING_PAGE_LIMIT = 400
# A REST funding slice is cut in hours, not in 8-hour settlement steps, so
# that it holds at most FUNDING_PAGE_LIMIT settlements even if OKX shortens
# the funding interval (it can) and settlements come hourly. Each page then
# returns its whole slice, whichever end OKX serves first.
FUNDING_SLOT_MS = 3_600_000

# OKX caps an archive listing at 10 months, but does not say whether begin/end
# are read in UTC or UTC+8. Our bounds can spill one month over at each end
# under either reading, so 8 requested months stay within the cap.
ARCHIVE_CHUNK_MONTHS = 8

# Sliding-window limits per endpoint group. history-candles allows 20 req/2 s
# per IP; staying at 15 leaves room for clock skew. The others are used for
# a handful of calls per run, so they are kept low without measurable cost.
RATE_LIMITS: Dict[str, Tuple[int, float]] = {
    "candles": (15, 2.0),
    "funding_rest": (8, 2.0),
    "archive": (4, 2.0),
    "time": (4, 2.0),
}

# How far the local clock may be from OKX's before the run is refused. A
# request's round trip is well inside it; a clock set to another day is not.
MAX_CLOCK_SKEW_MS = 60_000
LS_REMOTE_TIMEOUT_S = 30

MAX_RETRIES = 6
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0
HTTP_TIMEOUT_S = 30

MINUTE_MS = spec.MINUTE_MS
DAY_MS = 86_400_000
FUNDING_STEP_MS = int(spec.get_spec()["data"]["funding"]["expected_spacing_hours"]) * 3_600_000
UTC8 = timezone(timedelta(hours=8))

OUTPUT_FILE = {"candles": "candles_1m.parquet", "funding": "funding_rates.parquet"}
STEP_MS = {"candles": MINUTE_MS, "funding": FUNDING_STEP_MS}

# Source labels match the names used in Appendix A's `data` block.
SRC_CANDLES_REST = "rest_history_candles"
SRC_FUNDING_ARCHIVE = "archive_module_3"
SRC_FUNDING_REST = "rest_funding_rate_history"
SRC_MISSING = "missing"
SRC_NONE_EXPECTED = "none_expected"

EXIT_OK = 0
EXIT_FETCH_FAILED = 1
EXIT_USAGE = 2
EXIT_PROTECTED = 3
EXIT_EXISTS = 4

CANDLE_SCHEMA = {
    "timestamp": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "volume_ccy": pl.Float64,
}
FUNDING_SCHEMA = {
    "timestamp": pl.Int64,
    "rate": pl.Float64,
    "realized_rate": pl.Float64,
    "source": pl.Utf8,
}


class FetchError(RuntimeError):
    """A request kept failing, or a response could not be trusted."""


class OkxError(Exception):
    """OKX answered, but with a non-"0" code (rate limit, bad parameter, ...)."""


# --------------------------------------------------------------------------
# The network layer. Every byte this script receives from OKX passes through
# these two functions, and the one question asked of the public git remote
# through _ls_remote_main, so tests replace them and no test can reach the
# network.
# --------------------------------------------------------------------------

_HEADERS = {"User-Agent": "doge-usdt-trading/fetch_v2", "Accept": "*/*"}


def http_get_bytes(url: str) -> bytes:
    """GET ``url`` and return the body; raises on HTTP and transport errors."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return resp.read()


def http_get_json(url: str) -> Any:
    """GET ``url`` and decode its JSON body."""
    return json.loads(http_get_bytes(url).decode("utf-8"))


# Indirection so tests can run the throttle and backoff on a fake clock.
_now: Callable[[], float] = time.monotonic
_sleep: Callable[[float], None] = time.sleep
# The local calendar clock, in epoch ms. Separate from _now, which is
# monotonic and only measures intervals. Whether a protected window has
# closed is judged by OKX's clock (_server_ms); this one is only compared
# with it.
_wall_ms: Callable[[], int] = lambda: int(time.time() * 1000)

# The checkout whose freeze unlocks test data: the one this script belongs
# to. Tests point it at a throwaway repository; nothing else should.
FREEZE_ROOT: Path = REPO_ROOT


def _ls_remote_main(url: str) -> Optional[str]:
    """
    The commit ``url``'s main branch points to, as the remote itself reports
    it (``git ls-remote <url> refs/heads/main``); None if git fails, times
    out, or does not answer with one full 40-hex commit.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-remote", url, "refs/heads/main"], capture_output=True, timeout=LS_REMOTE_TIMEOUT_S,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},       # fail rather than ask for a password
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    fields = proc.stdout.decode("utf-8", "replace").split()
    if (proc.returncode != 0 or len(fields) != 2 or fields[1] != "refs/heads/main"
            or not gitrepo.FULL_SHA1.fullmatch(fields[0])):
        return None
    return fields[0]


def freeze_problem() -> Tuple[Optional[str], Optional[str]]:
    """
    (None, the public remote's main commit) if test data may be unlocked,
    else (the reason, None).

    A file at docs/preregistration-freeze.md is not enough: it must be the
    committed, pushed amendment that names the commit ``prereg-freeze``
    points to and agrees with the directions in it (§10 steps 2-3,
    research.freeze.verify_freeze). Offline, "pushed" can only mean
    refs/remotes/origin/main, which one ``git update-ref`` writes without
    any push. So the public repository is asked as well: the main branch
    it reports must be a commit this checkout has, and must contain the
    last change to the amendment (freeze.amendment_missing_from).
    """
    try:
        freeze.verify_freeze(FREEZE_ROOT)
        url = freeze.PUBLIC_REMOTE
        remote = _ls_remote_main(url)
        if remote is None:
            return f"git ls-remote could not read the main branch of {url}, so nothing shows the push", None
        if gitrepo.commit_of(FREEZE_ROOT, remote) is None:
            return (f"{url} has main at {remote[:12]}, a commit this checkout does not have; "
                    "fetch it and retry"), None
        problem = freeze.amendment_missing_from(FREEZE_ROOT, remote, f"main at {url} ({remote[:12]})")
        if problem:
            return f"{problem}; §10 step 3 pushes the amendment before any download", None
    except (freeze.FreezeNotVerified, gitrepo.GitError) as exc:
        return str(exc), None
    return None, remote


def open_protected(lo: int, hi: int, now_ms: int) -> List[Tuple[int, int]]:
    """The protected windows overlapping [lo, hi) that have not closed by ``now_ms``."""
    return [(a, b) for a, b in spec.protected_overlapping(lo, hi) if b > now_ms]

RETRYABLE = (OSError, ValueError, http.client.HTTPException, OkxError, zipfile.BadZipFile)


class RateLimiter:
    """At most ``max_calls`` in any ``period_s`` window (sliding, not bucketed)."""

    def __init__(self, max_calls: int, period_s: float) -> None:
        self.max_calls = max_calls
        self.period_s = period_s
        self._calls: deque = deque()

    def wait(self) -> None:
        while True:
            now = _now()
            while self._calls and now - self._calls[0] >= self.period_s:
                self._calls.popleft()
            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return
            _sleep(self._calls[0] + self.period_s - now)


class OkxClient:
    """Throttled, retrying access to OKX through the injectable HTTP layer."""

    def __init__(self) -> None:
        self.limiters = {k: RateLimiter(*v) for k, v in RATE_LIMITS.items()}
        self.requests = 0

    def _retry(self, what: str, bucket: str, once: Callable[[], Any]) -> Any:
        last: Optional[BaseException] = None
        for attempt in range(MAX_RETRIES):
            self.limiters[bucket].wait()
            self.requests += 1
            try:
                return once()
            except RETRYABLE as exc:
                last = exc
            if attempt < MAX_RETRIES - 1:
                wait = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** attempt)
                logger.warning(
                    f"{type(last).__name__}: {last} — retry {attempt + 1}/{MAX_RETRIES - 1} in {wait:g}s"
                )
                _sleep(wait)
        raise FetchError(f"Giving up on {what} after {MAX_RETRIES} attempts: {last}")

    def get_json(self, path: str, params: Dict[str, str], bucket: str) -> Dict[str, Any]:
        url = f"{OKX_REST_BASE}{path}?{urllib.parse.urlencode(params)}"

        def once() -> Dict[str, Any]:
            payload = http_get_json(url)
            if not isinstance(payload, dict):
                raise OkxError(f"unexpected payload type {type(payload).__name__}")
            # OKX reports application errors in the body with HTTP 200.
            code = str(payload.get("code"))
            if code != "0":
                raise OkxError(f"OKX code {code}: {payload.get('msg', '')}")
            return payload

        return self._retry(url, bucket, once)

    def get_zip(self, url: str) -> bytes:
        def once() -> bytes:
            data = http_get_bytes(url)
            # A truncated download is a transport failure, so check it here,
            # inside the retry, rather than when parsing.
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"corrupt member {bad}")
            return data

        return self._retry(url, "archive", once)


def _server_ms() -> int:
    """
    OKX's clock in epoch ms (GET /api/v5/public/time, data[0].ts).

    It, not the local clock, says whether a protected window has closed: a
    local clock set ahead would let a doge_fwd download start today and
    write a manifest whose window claims the whole set. Raises FetchError
    if the clock cannot be read.
    """
    payload = OkxClient().get_json(TIME_PATH, {}, "time")
    try:
        return int(payload["data"][0]["ts"])
    except (KeyError, IndexError, TypeError, ValueError):
        raise FetchError(f"{TIME_PATH} answered without data[0].ts: {payload!r}") from None


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------

def parse_utc(text: str) -> int:
    """'YYYY-MM-DD HH:MM' in UTC -> epoch ms."""
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fmt_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _add_months(y: int, m: int, k: int) -> Tuple[int, int]:
    n = y * 12 + (m - 1) + k
    return n // 12, n % 12 + 1


def utc8_month_of(ms: int) -> Tuple[int, int]:
    """The archive month (files are cut at UTC+8 month boundaries) holding ``ms``."""
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC8)
    return dt.year, dt.month


def utc8_month_span(y: int, m: int) -> Tuple[int, int]:
    """[lo, hi) in UTC ms of archive month (y, m), e.g. 2024-09 -> 08-31 16:00 .. 09-30 16:00."""
    ny, nm = _add_months(y, m, 1)
    return _ms(datetime(y, m, 1, tzinfo=UTC8)), _ms(datetime(ny, nm, 1, tzinfo=UTC8))


def utc8_months(start_ms: int, end_ms: int) -> List[Tuple[int, int]]:
    """Archive months whose files hold any settlement in [start_ms, end_ms)."""
    first, last = utc8_month_of(start_ms), utc8_month_of(end_ms - 1)
    out = [first]
    while out[-1] != last:
        out.append(_add_months(*out[-1], 1))
    return out


def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _float_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    return float(text) if text else math.nan


def _parse_ms(value: str) -> int:
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    f = float(text)
    if not f.is_integer():
        raise ValueError(f"not an epoch-ms timestamp: {value!r}")
    return int(f)


def _advance(cursor: int, oldest: int, stalled: bool, label: str) -> Tuple[int, bool]:
    """
    The next backward cursor after a page whose oldest row is ``oldest``.

    OKX documents ``after`` as exclusive. Were it inclusive, a last page
    capped to the rows left could hold only the cursor's own row; stepping
    back 1 ms is then correct under either reading. Two stalls in a row mean
    the server ignores ``after`` altogether: stop instead of looping.
    """
    if oldest < cursor:
        return oldest, False
    if stalled:
        raise FetchError(f"{label}: pagination did not advance past {fmt_utc(cursor)}")
    return cursor - 1, True


# --------------------------------------------------------------------------
# Candles
# --------------------------------------------------------------------------

def fetch_candles(
    client: OkxClient, inst_id: str, start_ms: int, end_ms: int
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Walk 1-minute candles backward from ``end_ms`` to ``start_ms``.

    Each page asks for one slice [floor, cursor): ``after`` (exclusive) is
    the cursor, ``before`` (exclusive) is floor - 1, and ``limit`` is the
    number of minutes in the slice, at most CANDLE_PAGE_LIMIT. A slice cannot
    hold more bars than the page may return, so the answer does not depend on
    which end of the range OKX serves first, and floor never goes below
    ``start_ms``: however large a hole in the data, no page reaches the bars
    before the window, which may be a protected test window. A page shorter
    than ``limit`` does not mean history ended (OKX may cap the page below
    what was asked), and neither does an empty one, which may be a hole: the
    walk steps over the slice and goes on to ``start_ms``.

    Unconfirmed bars are dropped: OKX keeps rewriting the in-progress bar, and
    one kept by accident is a look-ahead.
    """
    rows: List[List[str]] = []
    cursor = end_ms
    stalled = False
    pages = unconfirmed = served_outside = 0
    exhausted_at: Optional[int] = None

    while cursor > start_ms:
        n = min(CANDLE_PAGE_LIMIT, -(-(cursor - start_ms) // MINUTE_MS))
        floor = max(start_ms, cursor - n * MINUTE_MS)
        params = {
            "instId": inst_id,
            "bar": "1m",
            "after": str(cursor),
            "before": str(floor - 1),
            "limit": str(n),
        }
        data = client.get_json(CANDLES_PATH, params, "candles").get("data") or []
        pages += 1
        if not data:
            # A hole, or no history yet: step over the slice. Remember where
            # the empty stretch began, in case it runs on to start_ms.
            if exhausted_at is None:
                exhausted_at = cursor
            cursor, stalled = floor, False
            continue
        exhausted_at = None

        page_ts = []
        for r in data:
            ts = int(r[0])
            page_ts.append(ts)
            if not start_ms <= ts < end_ms:
                served_outside += 1
                continue
            if len(r) < 9 or r[8] != "1":
                unconfirmed += 1
                continue
            rows.append(r)

        oldest = min(page_ts)
        cursor, stalled = _advance(cursor, oldest, stalled, inst_id)

        if pages % 200 == 0:
            logger.info(f"  {inst_id}: page {pages}, back to {fmt_utc(oldest)} UTC")

    if exhausted_at is not None and exhausted_at > start_ms:
        logger.warning(
            f"{inst_id}: no bars before {fmt_utc(exhausted_at)} UTC; "
            f"[{fmt_utc(start_ms)}, {fmt_utc(exhausted_at)}) is missing"
        )
    if unconfirmed:
        logger.info(f"{inst_id}: dropped {unconfirmed} unconfirmed bar(s)")

    df = pl.DataFrame(
        {
            "timestamp": [int(r[0]) for r in rows],
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
            "volume_ccy": [float(r[6]) for r in rows],
        },
        schema=CANDLE_SCHEMA,
    )
    df, conflicts = _dedupe(df, inst_id)

    off_grid = int((df["timestamp"] % MINUTE_MS != 0).sum())
    if off_grid:
        logger.warning(f"{inst_id}: {off_grid} bar(s) not on the minute grid")

    meta = {
        "pages": pages,
        "unconfirmed_dropped": unconfirmed,
        "served_outside_window": served_outside,
        "duplicate_conflicts": conflicts,
        "off_grid_rows": off_grid,
        "no_data_before_ms": exhausted_at,
    }
    return df, meta


def _dedupe(df: pl.DataFrame, label: str) -> Tuple[pl.DataFrame, int]:
    """
    Keep the first row per timestamp; count timestamps whose rows disagree.

    Duplicates with identical values are harmless page overlap. Disagreeing
    ones mean two sources (or two pages) report different numbers for the
    same moment, which must be visible in the manifest, not resolved quietly.
    """
    value_cols = [c for c in df.columns if c != "source"]
    distinct = df.select(value_cols).unique()
    n_ts = df["timestamp"].n_unique()
    conflicts = int(distinct.height - n_ts)
    if conflicts:
        logger.warning(f"{label}: {conflicts} timestamp(s) with disagreeing duplicates; kept the first")
    out = df.unique(subset=["timestamp"], keep="first", maintain_order=True).sort("timestamp")
    return out, conflicts


# --------------------------------------------------------------------------
# Funding
# --------------------------------------------------------------------------

_MONTH_DASHED = re.compile(r"(20\d{2})-(0[1-9]|1[0-2])(?![\d-])")
_MONTH_PACKED = re.compile(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)")


def _iter_file_records(
    node: Any, family: Optional[str] = None
) -> Iterator[Tuple[Dict[str, Any], Optional[str]]]:
    """
    (record, instFamily) for every dict carrying a download ``url`` in a listing.

    The listing's nesting is not documented precisely enough to pin, so the
    walk does not depend on it: the family is inherited from the nearest
    ancestor that names one. What matters is that each file's month can be
    told before it is downloaded (see _file_month).
    """
    if isinstance(node, dict):
        family = node.get("instFamily") or family
        url = node.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            yield node, family
        for v in node.values():
            if isinstance(v, (dict, list)):
                yield from _iter_file_records(v, family)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_file_records(v, family)


def _file_month(rec: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    The archive month of a listed file, or None if it cannot be told.

    The file name (or the URL's YYYYMM directory) is trusted first: a
    timestamp field might be a publication time rather than the data month.
    A file whose month cannot be told is never downloaded, since it might
    hold protected settlements.
    """
    url_path = urllib.parse.urlparse(rec["url"]).path
    name = str(rec.get("fileName") or rec.get("filename") or url_path.rsplit("/", 1)[-1])
    m = _MONTH_DASHED.search(name) or _MONTH_PACKED.search(url_path)
    if m:
        return int(m.group(1)), int(m.group(2))
    v = rec.get("dateTs")
    if v not in (None, "") and str(v).strip().isdigit():
        # A month-start stamp lands in the right month in UTC+8 whether OKX
        # stamps the UTC or the UTC+8 start of the month.
        return utc8_month_of(int(v))
    return None


def list_funding_archive(
    client: OkxClient, inst_family: str, months: Sequence[Tuple[int, int]]
) -> Dict[Tuple[int, int], List[str]]:
    """Archive file URLs per UTC+8 month, for the months asked about only."""
    wanted = set(months)
    out: Dict[Tuple[int, int], List[str]] = {}
    for i in range(0, len(months), ARCHIVE_CHUNK_MONTHS):
        chunk = months[i:i + ARCHIVE_CHUNK_MONTHS]
        # From the UTC+8 start of the first month to the last ms of the last
        # month in UTC: inclusive under either reading of the bounds.
        begin = utc8_month_span(*chunk[0])[0]
        ny, nm = _add_months(*chunk[-1], 1)
        end = _ms(datetime(ny, nm, 1, tzinfo=timezone.utc)) - 1
        params = {
            "module": "3",
            "instType": "SWAP",
            "instFamilyList": inst_family,
            "dateAggrType": "monthly",
            "begin": str(begin),
            "end": str(end),
        }
        payload = client.get_json(ARCHIVE_PATH, params, "archive")
        for rec, fam in _iter_file_records(payload.get("data")):
            if fam not in (None, inst_family):
                continue
            month = _file_month(rec)
            if month is None:
                raise FetchError(f"Cannot tell which month archive file {rec['url']} covers")
            if month in wanted:
                urls = out.setdefault(month, [])
                if rec["url"] not in urls:
                    urls.append(rec["url"])
    return out


def parse_funding_zip(data: bytes, inst_id: str) -> List[Tuple[int, float, float]]:
    """
    (funding_time ms, funding_rate, realized rate or NaN) for ``inst_id``.

    Archive files can hold every swap of a family, so rows are filtered on
    instrument_name; a file without that column is refused rather than
    assumed to be single-instrument.
    """
    out: List[Tuple[int, float, float]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError("archive holds no CSV file")
        for name in members:
            reader = csv.DictReader(io.StringIO(zf.read(name).decode("utf-8-sig")))
            cols = {(f or "").strip().lower(): f for f in (reader.fieldnames or [])}
            for need in ("instrument_name", "funding_rate", "funding_time"):
                if need not in cols:
                    raise ValueError(f"{name}: no column {need!r} in {reader.fieldnames}")
            realized_col = cols.get("real_funding_rate") or cols.get("realized_rate")
            for rec in reader:
                if (rec[cols["instrument_name"]] or "").strip() != inst_id:
                    continue
                out.append(
                    (
                        _parse_ms(rec[cols["funding_time"]]),
                        float(rec[cols["funding_rate"]]),
                        _float_or_nan(rec.get(realized_col)) if realized_col else math.nan,
                    )
                )
    return out


def fetch_funding_rest(
    client: OkxClient, inst_id: str, lo: int, hi: int
) -> List[Tuple[int, float, float, str]]:
    """
    Settlements in [lo, hi) from funding-rate-history, paging backward.

    As with candles, every page asks for one slice [floor, cursor) bounded on
    both sides, here of at most FUNDING_PAGE_LIMIT hours, so no page can
    spill across ``lo`` and an empty slice (a hole, or a stretch older than
    REST keeps) is stepped over rather than taken as the end.
    """
    out: List[Tuple[int, float, float, str]] = []
    cursor = hi
    stalled = False
    while cursor > lo:
        n = min(FUNDING_PAGE_LIMIT, -(-(cursor - lo) // FUNDING_SLOT_MS))
        floor = max(lo, cursor - n * FUNDING_SLOT_MS)
        params = {
            "instId": inst_id,
            "after": str(cursor),
            "before": str(floor - 1),
            "limit": str(n),
        }
        data = client.get_json(FUNDING_REST_PATH, params, "funding_rest").get("data") or []
        if not data:
            cursor, stalled = floor, False
            continue
        page_ts = []
        for r in data:
            ts = int(r["fundingTime"])
            page_ts.append(ts)
            if lo <= ts < hi:
                out.append(
                    (ts, float(r["fundingRate"]), _float_or_nan(r.get("realizedRate")), SRC_FUNDING_REST)
                )
        cursor, stalled = _advance(cursor, min(page_ts), stalled, inst_id)
    return out


def fetch_funding(
    client: OkxClient, inst_id: str, start_ms: int, end_ms: int, guarded: bool,
    now_ms: Optional[int] = None,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Funding settlements in [start_ms, end_ms): archive first, REST for the rest.

    Archive files cover whole UTC+8 months, so a window that starts or ends
    mid-month still needs the whole file. A month whose file would also hold
    settlements from a protected window is not downloaded when ``guarded``
    (test data still locked), nor, even unlocked, when that protected window
    has not closed by ``now_ms`` (default: OKX's clock); its in-window
    part goes to REST instead, which can be bounded exactly. REST only
    reaches back a few months, so such days may end up missing, and the
    manifest says so.
    """
    if not inst_id.endswith("-SWAP"):
        raise FetchError(f"{inst_id}: funding exists only for swaps")
    family = inst_id[: -len("-SWAP")]
    now = _server_ms() if now_ms is None else int(now_ms)
    months = utc8_months(start_ms, end_ms)
    listing = list_funding_archive(client, family, months)

    recs: List[Tuple[int, float, float, str]] = []
    used, skipped, foreign = [], [], 0
    for month in months:
        urls = listing.get(month)
        if not urls:
            continue
        lo, hi = utc8_month_span(*month)
        label = f"{month[0]:04d}-{month[1]:02d}"
        if spec.overlaps_protected(lo, hi) and (guarded or open_protected(lo, hi, now)):
            why = "test data is locked" if guarded else "a protected window it covers has not closed yet"
            logger.warning(
                f"{inst_id}: archive month {label} also covers protected test data and {why}; "
                f"not downloading it, using REST for its in-window part"
            )
            skipped.append(label)
            continue
        n_month = 0
        for url in urls:
            data = client.get_zip(url)
            try:
                parsed = parse_funding_zip(data, inst_id)
            except (ValueError, KeyError, zipfile.BadZipFile) as exc:
                raise FetchError(f"Cannot parse archive file {url}: {exc}") from exc
            for ts, rate, realized in parsed:
                n_month += 1
                if not lo <= ts < hi:
                    foreign += 1
                if start_ms <= ts < end_ms:
                    recs.append((ts, rate, realized, SRC_FUNDING_ARCHIVE))
        if n_month:
            used.append(month)

    if foreign:
        logger.warning(f"{inst_id}: {foreign} archive row(s) outside their file's UTC+8 month")

    rest_months = [m for m in months if m not in used]
    rest_ranges = []
    for lo, hi in _merge_ranges([utc8_month_span(*m) for m in rest_months]):
        lo, hi = max(lo, start_ms), min(hi, end_ms)
        if lo < hi:
            rest_ranges.append([lo, hi])
            recs.extend(fetch_funding_rest(client, inst_id, lo, hi))

    df = pl.DataFrame(
        {
            "timestamp": [r[0] for r in recs],
            "rate": [r[1] for r in recs],
            "realized_rate": [r[2] for r in recs],
            "source": [r[3] for r in recs],
        },
        schema=FUNDING_SCHEMA,
    )
    # Archive rows come first, so on a shared settlement the archive wins: it
    # is the primary source in §4.2.
    df, conflicts = _dedupe(df, inst_id)

    ts = df["timestamp"].to_numpy()
    off = [[int(a), int(b)] for a, b in zip(ts[:-1], ts[1:]) if b - a != FUNDING_STEP_MS]
    for a, b in off:
        logger.warning(
            f"{inst_id}: settlement spacing {(b - a) / 3_600_000:g} h "
            f"({fmt_utc(a)} -> {fmt_utc(b)} UTC)"
        )

    meta = {
        "archive_months_used": [f"{y:04d}-{m:02d}" for y, m in used],
        "archive_months_skipped_protected": skipped,
        "rest_ranges": rest_ranges,
        "duplicate_conflicts": conflicts,
        "archive_rows_outside_file_month": foreign,
        "spacing_other_than_expected": off,
    }
    return df, meta


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def source_by_day(
    ts: np.ndarray, sources: Sequence[str], start_ms: int, end_ms: int, step_ms: int
) -> Dict[str, str]:
    """
    Source label for every UTC day the window touches.

    A day with no rows is 'missing' if the step grid (epoch-aligned: whole
    minutes, or 00/08/16 UTC settlements) puts at least one timestamp in the
    part of that day inside the window, and 'none_expected' otherwise, so a
    window opening at 20:00 does not report its first funding day as a hole.
    """
    by_day: Dict[int, set] = {}
    for t, s in zip(np.asarray(ts, dtype=np.int64).tolist(), sources):
        by_day.setdefault(t // DAY_MS, set()).add(s)
    out: Dict[str, str] = {}
    for d in range(start_ms // DAY_MS, -(-end_ms // DAY_MS)):
        key = datetime.fromtimestamp(d * DAY_MS / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if d in by_day:
            out[key] = "+".join(sorted(by_day[d]))
            continue
        lo, hi = max(d * DAY_MS, start_ms), min((d + 1) * DAY_MS, end_ms)
        first_on_grid = -(-lo // step_ms) * step_ms
        out[key] = SRC_MISSING if first_on_grid < hi else SRC_NONE_EXPECTED
    return out


def edge_gaps(ts: np.ndarray, start_ms: int, end_ms: int, step_ms: int) -> List[List[int]]:
    """
    Missing stretches at the window's ends, half-open like the interior gaps.

    The interior gap list cannot see a series that starts late or stops
    early; these two entries close that hole.
    """
    if len(ts) == 0:
        return [[start_ms, end_ms]]
    out = []
    if int(ts[0]) - start_ms >= step_ms:
        out.append([start_ms, int(ts[0])])
    if end_ms - int(ts[-1]) > step_ms:
        out.append([int(ts[-1]) + step_ms, end_ms])
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch OKX candles or funding for the pre-registered study (§4.2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Two ways to say what to fetch: one kind for several instruments
    # (--kind/--inst-id), or everything a set needs in one manifest
    # (--candles/--funding). run() refuses a mix of the two.
    p.add_argument("--kind", choices=sorted(OUTPUT_FILE), help="single-kind form, with --inst-id")
    p.add_argument("--inst-id", nargs="+", help="OKX instrument id(s), with --kind")
    p.add_argument("--candles", nargs="+", metavar="ID", help="1-minute candles for these instruments")
    p.add_argument("--funding", nargs="+", metavar="ID", help="funding settlements for these swaps")
    p.add_argument("--start", required=True, help="'YYYY-MM-DD HH:MM' UTC, inclusive")
    p.add_argument("--end", required=True, help="'YYYY-MM-DD HH:MM' UTC, exclusive")
    p.add_argument("--out-dir", required=True, type=Path, help="writes <out-dir>/<instId>/<file>")
    p.add_argument("--manifest", required=True, type=Path, help="manifest JSON to create")
    p.add_argument(
        "--allow-test-data",
        action="store_true",
        help="permit a closed protected window; effective only once the freeze is verified "
             "(amendment committed, matching prereg-freeze, and pushed)",
    )
    return p.parse_args(argv)


def _jobs(args: argparse.Namespace) -> Tuple[Optional[List[Tuple[str, str]]], str]:
    """
    The (kind, instId) pairs one invocation fetches, or (None, reason).

    The same instrument may appear under both kinds (a perpetual's candles
    and its funding are different files), but not twice under one kind.
    """
    single = args.kind is not None or args.inst_id is not None
    per_set = args.candles is not None or args.funding is not None
    if single and per_set:
        return None, "use either --kind/--inst-id or --candles/--funding, not both"
    if single:
        if args.kind is None or args.inst_id is None:
            return None, "--kind and --inst-id go together"
        jobs = [(args.kind, i) for i in args.inst_id]
    elif per_set:
        jobs = [("candles", i) for i in args.candles or []] + [("funding", i) for i in args.funding or []]
    else:
        return None, "nothing to fetch: give --kind/--inst-id or --candles/--funding"
    if len(set(jobs)) != len(jobs):
        return None, "duplicate instrument id"
    not_swaps = [i for k, i in jobs if k == "funding" and not i.endswith("-SWAP")]
    if not_swaps:
        return None, f"Funding exists only for swaps: {not_swaps}"
    return jobs, ""


def run(args: argparse.Namespace) -> int:
    # The guards below are only as good as the code running them, so first:
    # is every module of this repository this checkout's?
    foreign = integrity.foreign_code_problem(REPO_ROOT)
    if foreign:
        logger.error(f"Refusing to fetch: {foreign}")
        return EXIT_PROTECTED
    try:
        start_ms, end_ms = parse_utc(args.start), parse_utc(args.end)
    except ValueError as exc:
        logger.error(f"Bad --start/--end ({exc}); expected 'YYYY-MM-DD HH:MM'")
        return EXIT_USAGE
    if start_ms >= end_ms:
        logger.error("--start must be before --end")
        return EXIT_USAGE
    jobs, why_not = _jobs(args)
    if jobs is None:
        logger.error(why_not)
        return EXIT_USAGE
    kinds = sorted({k for k, _ in jobs})
    manifest_kind = args.kind if args.kind is not None else "+".join(kinds)

    window = f"[{fmt_utc(start_ms)}, {fmt_utc(end_ms)}) UTC"
    # 'Now' is OKX's clock (see _server_ms). A local clock far from it is
    # refused rather than overruled: it is a broken machine or a deliberate
    # change, and either deserves a look before any download.
    try:
        now_ms = _server_ms()
    except FetchError as exc:
        logger.error(f"Cannot read OKX's clock, which decides whether a protected window has closed: {exc}")
        return EXIT_FETCH_FAILED
    off_ms = _wall_ms() - now_ms
    if abs(off_ms) > MAX_CLOCK_SKEW_MS:
        logger.error(f"The local clock is {off_ms / 1000:+.0f} s off OKX's ({fmt_utc(now_ms)} UTC), more than "
                     f"{MAX_CLOCK_SKEW_MS // 1000} s. Refusing; set the clock right and retry.")
        return EXIT_FETCH_FAILED
    still_open = open_protected(start_ms, end_ms, now_ms)
    if still_open:
        # Whatever the unlock says: §10 downloads doge_fwd after its window
        # closes, so a partial download cannot pick an earlier stop date.
        logger.error(
            f"{window} overlaps a protected window that has not closed yet ("
            + ", ".join(f"[{fmt_utc(lo)}, {fmt_utc(hi)})" for lo, hi in still_open)
            + "); §10 downloads it after its end. Refusing."
        )
        return EXIT_PROTECTED

    overlaps = spec.overlaps_protected(start_ms, end_ms)
    # git is asked only when the flag is given, and asked even when the
    # window does not overlap, so the manifest's test_data_unlocked is true
    # exactly when the freeze was verified.
    problem, remote_main = freeze_problem() if args.allow_test_data else ("--allow-test-data was not given", None)
    unlocked = problem is None
    if overlaps and not unlocked:
        logger.error(
            f"{window} overlaps a protected test window ("
            + ", ".join(f"[{fmt_utc(lo)}, {fmt_utc(hi)})" for lo, hi in spec.protected_windows())
            + f"), and test data is locked: {problem}. Refusing."
        )
        return EXIT_PROTECTED
    if args.allow_test_data and not unlocked:
        logger.warning(f"--allow-test-data has no effect: {problem}")

    targets = {(k, i): Path(args.out_dir) / i / OUTPUT_FILE[k] for k, i in jobs}
    existing = [str(p) for p in [*targets.values(), Path(args.manifest)] if p.exists()]
    if existing:
        logger.error(f"Refusing to overwrite: {existing}")
        return EXIT_EXISTS

    # Everything is fetched before anything is written, so a failure halfway
    # through leaves no partial output that would block a retry.
    client = OkxClient()
    fetched: Dict[Tuple[str, str], Tuple[pl.DataFrame, Dict[str, Any]]] = {}
    try:
        for kind, inst in jobs:
            logger.info(f"{kind} {inst}: [{fmt_utc(start_ms)}, {fmt_utc(end_ms)}) UTC")
            if kind == "candles":
                fetched[(kind, inst)] = fetch_candles(client, inst, start_ms, end_ms)
            else:
                fetched[(kind, inst)] = fetch_funding(client, inst, start_ms, end_ms,
                                                      guarded=not unlocked, now_ms=now_ms)
    except FetchError as exc:
        logger.error(str(exc))
        return EXIT_FETCH_FAILED

    series = []
    try:
        for (kind, inst), (df, meta) in fetched.items():
            step = STEP_MS[kind]
            ts = df["timestamp"].to_numpy()
            if kind == "candles":
                sources = [SRC_CANDLES_REST] * df.height
                out_df = df
            else:
                sources = df["source"].to_list()
                out_df = df.drop("source")
            path = write_once_parquet(out_df, targets[(kind, inst)])
            entry = manifest_entry(path, source_by_day(ts, sources, start_ms, end_ms, step), ts, step)
            entry.update(
                {
                    "inst_id": inst,
                    "kind": kind,
                    "columns": out_df.columns,
                    "edge_gaps": edge_gaps(ts, start_ms, end_ms, step),
                    **meta,
                }
            )
            n_gaps = len(entry["gaps"]) + len(entry["edge_gaps"])
            logger.info(f"{inst}: {df.height:,} rows, {n_gaps} gap(s) -> {path}")
            series.append(entry)

        manifest = {
            "manifest_version": 1,
            "tool": "scripts/fetch_v2.py",
            "kind": manifest_kind,
            "window": {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_utc": fmt_utc(start_ms),
                "end_utc": fmt_utc(end_ms),
                "half_open": True,
            },
            "overlaps_protected": overlaps,
            "test_data_unlocked": unlocked,
            # What the unlock rested on: the public repository and the main
            # commit it reported. Null while test data stayed locked.
            "remote_url": freeze.PUBLIC_REMOTE if unlocked else None,
            "remote_main_commit": remote_main,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "requests": client.requests,
            "series": series,
        }
        write_once_json(manifest, args.manifest)
    except FileExistsError as exc:
        logger.error(f"Refusing to overwrite: {exc.filename}")
        return EXIT_EXISTS
    logger.info(f"Manifest -> {args.manifest} ({client.requests} requests)")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        raise SystemExit(130)
