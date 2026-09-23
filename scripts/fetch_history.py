#!/usr/bin/env python3
"""
Fetch real OKX market history for research and backtesting.

Pulls three series for a perpetual swap and stores them as Parquet under
``data/history/``:

    1. 1-minute OHLCV candles   /api/v5/market/history-candles
    2. Funding rate history     /api/v5/public/funding-rate-history
    3. Open-interest history    /api/v5/rubik/stat/contracts/open-interest-volume

Only *closed* candles are kept: OKX marks an unfinished bar with ``confirm="0"``
and such a bar keeps mutating until its minute ends. Training or backtesting on
one is a look-ahead bug, so they are dropped at ingest rather than downstream.

All endpoints are public — no API key, no signing. Requests are throttled to
stay inside OKX's per-IP rate limit and retried with exponential backoff.

Usage:
    python scripts/fetch_history.py --days 365
    python scripts/fetch_history.py --days 30 --inst-id BTC-USDT-SWAP
    python scripts/fetch_history.py --days 365 --skip-oi
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import polars as pl

# Allow `python scripts/fetch_history.py` as well as `python -m scripts.fetch_history`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

OKX_REST_BASE = "https://www.okx.com"

# OKX caps history endpoints at 100 rows per call.
PAGE_LIMIT = 100

# Per-IP limit on the market endpoints is 20 requests / 2s. Stay well under it:
# a full year of 1m candles is ~5300 calls and getting rate-limited halfway
# through is far more expensive than running a little slower.
REQUEST_INTERVAL_S = 0.15

MAX_RETRIES = 5
MINUTE_MS = 60_000

HISTORY_DIR = REPO_ROOT / "data" / "history"


class OKXHistoryClient:
    """Minimal async client for OKX's public history endpoints."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self.session = session
        self._last_request_ts = 0.0
        self.request_count = 0

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < REQUEST_INTERVAL_S:
            await asyncio.sleep(REQUEST_INTERVAL_S - elapsed)
        self._last_request_ts = time.monotonic()

    async def get(self, path: str, params: Dict[str, Any]) -> List[List[str]]:
        """GET one page, retrying on transport errors and OKX rate-limit codes."""
        url = f"{OKX_REST_BASE}{path}"

        for attempt in range(MAX_RETRIES):
            await self._throttle()
            try:
                async with self.session.get(url, params=params) as resp:
                    self.request_count += 1

                    # 429 has no body worth parsing; back off and retry.
                    if resp.status == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited (HTTP 429), retrying in {wait}s")
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    payload = await resp.json()

                    # OKX signals application errors in the body, not the status.
                    code = payload.get("code")
                    if code != "0":
                        msg = payload.get("msg", "unknown error")
                        # 50011 = rate limit; anything else is a real error.
                        if code == "50011":
                            wait = 2 ** attempt
                            logger.warning(f"OKX rate limit (50011), retrying in {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        raise RuntimeError(f"OKX error {code}: {msg} (path={path})")

                    return payload.get("data", [])

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = 2 ** attempt
                logger.warning(
                    f"Request failed ({type(exc).__name__}: {exc}), "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {wait}s"
                )
                await asyncio.sleep(wait)

        raise RuntimeError(f"Giving up on {path} after {MAX_RETRIES} retries")


async def fetch_candles(
    client: OKXHistoryClient,
    inst_id: str,
    bar: str,
    start_ms: int,
    end_ms: int,
) -> pl.DataFrame:
    """
    Page backwards through 1m candles from ``end_ms`` down to ``start_ms``.

    OKX returns each page newest-first and paginates with ``after`` = "give me
    bars older than this timestamp", so we walk backwards and reverse at the end.

    Row format: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    """
    rows: List[List[str]] = []
    cursor = end_ms
    expected_pages = max(1, (end_ms - start_ms) // MINUTE_MS // PAGE_LIMIT)
    page = 0

    logger.info(f"Fetching {bar} candles for {inst_id} (~{expected_pages} pages)")

    while cursor > start_ms:
        data = await client.get(
            "/api/v5/market/history-candles",
            {"instId": inst_id, "bar": bar, "after": str(cursor), "limit": str(PAGE_LIMIT)},
        )

        # Empty page means we ran past the start of available history.
        if not data:
            logger.info("Reached the end of available history")
            break

        rows.extend(data)
        page += 1

        # Oldest row on this page becomes the next cursor.
        oldest_ts = int(data[-1][0])
        if oldest_ts >= cursor:
            # Defensive: a non-advancing cursor would loop forever.
            logger.warning("Pagination cursor stopped advancing, stopping early")
            break
        cursor = oldest_ts

        if page % 50 == 0:
            got = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)
            logger.info(f"  page {page}/{expected_pages} — back to {got:%Y-%m-%d %H:%M} UTC")

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(
        {
            "timestamp": [int(r[0]) for r in rows],
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
            "volume_ccy": [float(r[6]) for r in rows],
            "confirm": [r[8] for r in rows],
        }
    )

    unconfirmed = int((df["confirm"] != "1").sum())
    if unconfirmed:
        # Almost always just the current in-progress minute.
        logger.info(f"Dropping {unconfirmed} unconfirmed (still-forming) candle(s)")

    return (
        df.filter(pl.col("confirm") == "1")
        .drop("confirm")
        .filter(pl.col("timestamp") >= start_ms)
        .unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
    )


async def fetch_funding_rates(
    client: OKXHistoryClient, inst_id: str, start_ms: int, end_ms: int
) -> pl.DataFrame:
    """Page backwards through funding-rate settlements (every 8h for most swaps)."""
    rows: List[Dict[str, str]] = []
    cursor = end_ms

    logger.info(f"Fetching funding rate history for {inst_id}")

    while cursor > start_ms:
        data = await client.get(
            "/api/v5/public/funding-rate-history",
            {"instId": inst_id, "after": str(cursor), "limit": str(PAGE_LIMIT)},
        )
        if not data:
            break

        rows.extend(data)
        oldest_ts = int(data[-1]["fundingTime"])
        if oldest_ts >= cursor:
            break
        cursor = oldest_ts

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(
            {
                "timestamp": [int(r["fundingTime"]) for r in rows],
                "rate": [float(r["fundingRate"]) for r in rows],
                "realized_rate": [float(r.get("realizedRate") or "nan") for r in rows],
            }
        )
        .filter(pl.col("timestamp") >= start_ms)
        .unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
    )


async def fetch_open_interest(
    client: OKXHistoryClient, ccy: str, period: str = "5m"
) -> pl.DataFrame:
    """
    Fetch open-interest history from OKX's 'rubik' statistics endpoint.

    Caveat worth knowing before you build a factor on this: the series is
    aggregated *per currency* across contract types, not per instrument, and
    OKX only retains a short window of it. It is an approximation of the
    DOGE-USDT-SWAP open interest, not the real thing — which is why this is
    the one series the script will skip without failing.

    Row format: [ts, openInterest, volume]
    """
    logger.info(f"Fetching open-interest history for {ccy} (period={period})")

    data = await client.get(
        "/api/v5/rubik/stat/contracts/open-interest-volume",
        {"ccy": ccy, "period": period},
    )
    if not data:
        return pl.DataFrame()

    return pl.DataFrame(
        {
            "timestamp": [int(r[0]) for r in data],
            "oi": [float(r[1]) for r in data],
            "oi_volume": [float(r[2]) for r in data],
        }
    ).sort("timestamp")


def spot_inst_id(inst_id: str) -> str:
    """DOGE-USDT-SWAP -> DOGE-USDT. Returns the input unchanged if not a swap."""
    return inst_id[: -len("-SWAP")] if inst_id.endswith("-SWAP") else inst_id


def report_gaps(df: pl.DataFrame, bar_ms: int = MINUTE_MS) -> None:
    """
    Report holes in the candle series.

    Gaps are normal (exchange maintenance, genuinely zero-trade minutes), but
    they must be known: a backtest that silently treats a 3-hour hole as three
    consecutive bars will produce a return that never existed.
    """
    if df.height < 2:
        return

    ts = df["timestamp"].to_numpy()
    deltas = ts[1:] - ts[:-1]
    gaps = deltas[deltas > bar_ms]

    if len(gaps) == 0:
        logger.success("No gaps — candle series is continuous")
        return

    missing = int((gaps // bar_ms - 1).sum())
    pct = 100.0 * missing / (df.height + missing)
    logger.warning(
        f"{len(gaps)} gap(s), {missing} missing bar(s) ({pct:.2f}% of the span); "
        f"largest = {int(gaps.max() // bar_ms)} bars"
    )


def summarize_candles(df: pl.DataFrame, fee_bps_round_trip: float = 10.0) -> None:
    """
    Print the return distribution against round-trip cost.

    This is the arithmetic that decides whether a 1-minute strategy on this
    instrument is viable at all: if typical per-bar moves are the same size as
    the fees, there is no room for a taker strategy regardless of how good the
    model is.
    """
    if df.height < 2:
        return

    returns = (df["close"].pct_change() * 10_000).drop_nulls()  # basis points
    abs_returns = returns.abs()
    cost = fee_bps_round_trip

    logger.info("")
    logger.info("=" * 62)
    logger.info("1-minute return distribution (basis points, 1bp = 0.01%)")
    logger.info("=" * 62)
    logger.info(f"  std dev           {returns.std():8.2f} bp")
    logger.info(f"  mean |return|     {abs_returns.mean():8.2f} bp")
    for q in (0.50, 0.75, 0.90, 0.99):
        logger.info(f"  |return| p{int(q * 100):<2}      {abs_returns.quantile(q):8.2f} bp")

    beats = 100.0 * (abs_returns > cost).mean()
    logger.info("-" * 62)
    logger.info(f"  round-trip cost   {cost:8.2f} bp  (taker in + taker out)")
    logger.info(f"  bars whose full move exceeds cost: {beats:.1f}%")
    logger.info("")
    logger.info("  Read this as an upper bound, not a forecast: it assumes you")
    logger.info("  capture the entire bar move with perfect direction calls.")
    logger.info("  A real edge captures a fraction of it, so if this number is")
    logger.info("  small, a 1m taker strategy cannot work here — lengthen the")
    logger.info("  holding period or switch to maker orders.")
    logger.info("=" * 62)


def write_parquet(df: pl.DataFrame, path: Path, label: str) -> None:
    if df.is_empty():
        logger.warning(f"{label}: nothing fetched, not writing")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

    start = datetime.fromtimestamp(df["timestamp"][0] / 1000, tz=timezone.utc)
    end = datetime.fromtimestamp(df["timestamp"][-1] / 1000, tz=timezone.utc)
    size_mb = path.stat().st_size / 1024 / 1024
    logger.success(
        f"{label}: {df.height:,} rows  "
        f"{start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC  "
        f"({size_mb:.1f} MB) -> {path.relative_to(REPO_ROOT)}"
    )


async def run(args: argparse.Namespace) -> int:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 24 * 60 * 60 * 1000
    out_dir = HISTORY_DIR / args.inst_id
    ccy = args.inst_id.split("-")[0]

    logger.info(
        f"Target: {args.inst_id}  bar={args.bar}  "
        f"span={args.days}d  ->  {out_dir.relative_to(REPO_ROOT)}"
    )

    timeout = aiohttp.ClientTimeout(total=30)
    started = time.monotonic()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = OKXHistoryClient(session)

        candles = await fetch_candles(client, args.inst_id, args.bar, start_ms, end_ms)
        if candles.is_empty():
            logger.error("No candles returned — check the instrument id and try again")
            return 1

        report_gaps(candles)
        write_parquet(candles, out_dir / f"candles_{args.bar}.parquet", "Candles")

        if args.with_spot:
            spot_inst = spot_inst_id(args.inst_id)
            if spot_inst == args.inst_id:
                logger.warning(f"{args.inst_id} is not a swap; skipping the spot leg")
            else:
                logger.info(f"Fetching spot leg {spot_inst} for basis calculation")
                spot = await fetch_candles(client, spot_inst, args.bar, start_ms, end_ms)
                write_parquet(
                    spot, out_dir / f"spot_candles_{args.bar}.parquet", "Spot candles"
                )

        funding = await fetch_funding_rates(client, args.inst_id, start_ms, end_ms)
        write_parquet(funding, out_dir / "funding_rates.parquet", "Funding rates")

        if not args.skip_oi:
            try:
                oi = await fetch_open_interest(client, ccy, args.oi_period)
                write_parquet(oi, out_dir / "open_interest.parquet", "Open interest")
            except Exception as exc:
                # Non-fatal by design: this endpoint is the flakiest of the three
                # and the candles are what actually matter.
                logger.warning(f"Open-interest fetch failed ({exc}); continuing without it")

        summarize_candles(candles, args.fee_bps)

        elapsed = time.monotonic() - started
        logger.info(f"Done in {elapsed:.1f}s across {client.request_count} requests")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch real OKX history into Parquet for research and backtesting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--inst-id", default="DOGE-USDT-SWAP", help="OKX instrument id")
    p.add_argument("--bar", default="1m", help="candle interval (1m, 5m, 15m, 1H, ...)")
    p.add_argument("--days", type=int, default=365, help="how many days back to fetch")
    p.add_argument("--skip-oi", action="store_true", help="skip the open-interest series")
    p.add_argument(
        "--with-spot",
        action="store_true",
        help="also fetch the matching spot pair (DOGE-USDT-SWAP -> DOGE-USDT), "
             "needed to compute a real perp-vs-spot basis",
    )
    p.add_argument("--oi-period", default="5m", help="open-interest granularity")
    p.add_argument(
        "--fee-bps",
        type=float,
        default=10.0,
        help="round-trip taker cost in bp used by the viability summary (default 10 = 0.05%% each way)",
    )
    return p.parse_args()


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    try:
        return asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
