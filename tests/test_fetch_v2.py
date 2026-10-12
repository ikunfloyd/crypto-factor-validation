"""
scripts/fetch_v2.py against a fake OKX.

No test here reaches the network: an autouse fixture replaces the script's
two HTTP functions (and urlopen itself) with ones that fail, and each test
installs a fake server explicitly. The public git remote is a bare repository
under tmp_path, and ``git ls-remote`` is replaced by a read of it. The clocks
are fake too: the monotonic one, so retries and the throttle are checked
without sleeping, and the calendar one, fixed at 2026-10-11 12:00 UTC, while
doge_fwd's window is still open; OKX's clock follows it unless a test says
otherwise.

What matters most is the guard (§10): a window overlapping a protected test
range is refused before any request, unless --allow-test-data is given and
the freeze is verified in git (a throwaway repository here, holding a real
tag, amendment and push) and on the public remote; a protected range that has
not closed by OKX's clock is refused even then; and an allowed window next to
one never receives a row from across the border, whatever holes the data has.
"""
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from loguru import logger

from research import freeze, spec
from research.io import sha256_file

_SPEC = importlib.util.spec_from_file_location(
    "fetch_v2", spec.REPO_ROOT / "scripts" / "fetch_v2.py"
)
fv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fv)
# The real seams, before the autouse fixture replaces them; tests that call
# them give them only a local path or a fake HTTP layer.
REAL_LS_REMOTE = fv._ls_remote_main
REAL_SERVER_MS = fv._server_ms

MIN = 60_000
H8 = 8 * 3_600_000


def utc_ms(text):
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


NOW = utc_ms("2026-10-11 12:00")              # inside doge_fwd's window, which closes 2026-11-23 20:00
FWD_CLOSE = spec.get_spec()["sets"]["doge_fwd"]["end_ms"]


class FakeClock:
    def __init__(self):
        self.t = 1_000.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError(f"unexpected network access: {args}")

    clock = FakeClock()
    monkeypatch.setattr(fv, "http_get_json", no_network)
    monkeypatch.setattr(fv, "http_get_bytes", no_network)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(fv, "_now", clock.now)
    monkeypatch.setattr(fv, "_sleep", clock.sleep)
    monkeypatch.setattr(fv, "_wall_ms", lambda: NOW)
    # OKX's clock agrees with whatever the local one is set to in a test.
    monkeypatch.setattr(fv, "_server_ms", lambda: fv._wall_ms())
    # The freeze is never looked up in the real checkout: by default in a
    # directory that is not a repository at all, so test data stays locked.
    no_repo = tmp_path / "no-freeze"
    no_repo.mkdir()
    monkeypatch.setattr(fv, "FREEZE_ROOT", no_repo)
    # The public remote is a bare repository here (frozen_repo makes it the
    # origin), and ls-remote reads its main without leaving the machine.
    monkeypatch.setattr(freeze, "PUBLIC_REMOTE", str(tmp_path / "public-origin.git"))

    def ls_remote_local(url):
        assert url == freeze.PUBLIC_REMOTE, f"unexpected ls-remote of {url}"
        proc = subprocess.run(["git", "-C", url, "rev-parse", "--verify", "--quiet", "refs/heads/main"],
                              capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None

    monkeypatch.setattr(fv, "_ls_remote_main", ls_remote_local)
    # git (for the throwaway freeze repositories) without the user's config.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(k, raising=False)
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")
    return clock


@pytest.fixture
def messages():
    out = []
    hid = logger.add(lambda m: out.append(m.record["message"]), level="INFO")
    yield out
    logger.remove(hid)


# ---------------------------------------------------------------- a frozen repository

def git(cwd, *args):
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def insample_text():
    """A well-formed 60-row results/insample_v2.csv."""
    lines = [",".join(freeze.INSAMPLE_COLUMNS)]
    for i, (b, h, f) in enumerate(freeze.expected_cells()):
        lines.append(f"{b},{h},{f},{-1 if i % 2 else 1},false,,900,0.5,1.25,hh,{3 + i / 10:.10g}")
    return "\n".join(lines) + "\n"


def frozen_repo(path, *, commit=True, push=True, amendment=None):
    """
    A repository shaped as §10 steps 2-3 leave it: code, then a commit adding
    only results/insample_v2.csv (tagged prereg-freeze), then the amendment,
    pushed to its 'origin', the bare repository at freeze.PUBLIC_REMOTE.
    ``commit``/``push`` False stop one step short (origin is set either way);
    ``amendment`` replaces the rendered amendment text.
    """
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path.parent, "init", "-q", "--bare", freeze.PUBLIC_REMOTE)
    git(path, "remote", "add", "origin", freeze.PUBLIC_REMOTE)
    (path / "research").mkdir()
    (path / "research" / "code.py").write_text("X = 1\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "code")
    text = insample_text()
    (path / "results").mkdir()
    (path / "results" / "insample_v2.csv").write_text(text)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "in-sample directions")
    tagged = git(path, "rev-parse", "HEAD")
    git(path, "tag", "-a", "prereg-freeze", "-m", "freeze")
    (path / "docs").mkdir()
    body = freeze.render_amendment_blocks(tagged, text) if amendment is None else amendment
    (path / "docs" / "preregistration-freeze.md").write_text("# Freeze amendment\n\n" + body)
    if commit:
        git(path, "add", "-A")
        git(path, "commit", "-q", "-m", "freeze amendment")
        if push:
            git(path, "push", "-q", "origin", "main")
    return path


def unlock(monkeypatch, tmp_path):
    """Point the script at a repository in which the freeze verifies."""
    repo = frozen_repo(tmp_path / "frozen")
    assert freeze.verify_freeze(repo).commit == git(repo, "rev-parse", "prereg-freeze^{commit}")
    monkeypatch.setattr(fv, "FREEZE_ROOT", repo)
    return repo


def ok(data):
    return {"code": "0", "msg": "", "data": data}


# ---------------------------------------------------------------- fake OKX

def candle_row(ts, confirm="1"):
    p = 0.1 + (ts // MIN % 97) * 1e-4
    vol = ts // MIN % 1000 + 1
    return [str(ts), f"{p:.6f}", f"{p * 1.001:.6f}", f"{p * 0.999:.6f}", f"{p:.6f}",
            str(vol), f"{vol * 10.5}", "0", confirm]


def rate_of(ts):
    return 1e-4 * ((ts // H8) % 7 - 3)


def funding_rec(inst, ts, rate=None):
    r = rate_of(ts) if rate is None else rate
    return {"instType": "SWAP", "instId": inst, "fundingRate": repr(r),
            "realizedRate": repr(r * 0.9), "fundingTime": str(ts), "method": "current_period"}


def make_zip(rows, bom=True, header=("instrument_name", "contract_type", "funding_rate",
                                     "real_funding_rate", "funding_time")):
    """rows: (instrument_name, rate, ts)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for inst, rate, ts in rows:
        rec = {"instrument_name": inst, "contract_type": "SWAP", "funding_rate": repr(rate),
               "real_funding_rate": repr(rate * 0.9), "funding_time": str(ts)}
        w.writerow([rec[h] for h in header])
    text = ("﻿" if bom else "") + buf.getvalue()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("funding.csv", text.encode("utf-8"))
    return out.getvalue()


def archive_url(fam, y, m):
    return (f"https://static.okx.com/cdn/okex/traderecords/swaprate/monthly/{y:04d}{m:02d}/"
            f"{fam}-swaprate-{y:04d}-{m:02d}.zip")


class FakeOkx:
    """Just enough of OKX's paging semantics to exercise the client."""

    def __init__(self, clock=None, cap=100, honour_before=True, honour_limit=True, inclusive_after=False):
        self.candles = {}        # inst -> {ts: row}
        self.funding = {}        # inst -> {ts: rec}
        self.archive = {}        # family -> {(y, m): url}
        self.zips = {}           # url -> bytes
        self.cap = cap
        self.honour_before = honour_before
        self.honour_limit = honour_limit
        self.inclusive_after = inclusive_after
        self.clock = clock
        self.json_calls = []     # (path, params)
        self.byte_calls = []
        self.call_times = []
        self.served = []         # every candle/funding ts handed out

    def install(self, monkeypatch):
        monkeypatch.setattr(fv, "http_get_json", self.get_json)
        monkeypatch.setattr(fv, "http_get_bytes", self.get_bytes)
        return self

    def _page(self, store, q):
        after = int(q["after"])
        before = int(q["before"]) if ("before" in q and self.honour_before) else None
        limit = int(q["limit"]) if self.honour_limit else 10 ** 9
        keys = sorted(
            (t for t in store
             if (t <= after if self.inclusive_after else t < after) and (before is None or t > before)),
            reverse=True,
        )[: min(limit, self.cap)]
        self.served.extend(keys)
        return [store[t] for t in keys]

    def get_json(self, url):
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        self.json_calls.append((u.path, q))
        if self.clock:
            self.call_times.append(self.clock.t)
        if u.path == fv.CANDLES_PATH:
            assert q["bar"] == "1m"
            return ok(self._page(self.candles.get(q["instId"], {}), q))
        if u.path == fv.FUNDING_REST_PATH:
            return ok(self._page(self.funding.get(q["instId"], {}), q))
        if u.path == fv.ARCHIVE_PATH:
            assert (q["module"], q["instType"], q["dateAggrType"]) == ("3", "SWAP", "monthly")
            fam = q["instFamilyList"]
            begin, end = int(q["begin"]), int(q["end"])
            files = []
            for (y, m), url in sorted(self.archive.get(fam, {}).items()):
                lo, hi = fv.utc8_month_span(y, m)
                if lo <= end and hi > begin:
                    files.append({"fileName": url.rsplit("/", 1)[-1], "url": url, "sizeMB": "0.01"})
            return ok([{"dateAggrType": "monthly", "totalSizeMB": "0.1",
                        "details": [{"instFamily": fam, "instType": "SWAP", "groupDetails": files}]}])
        raise AssertionError(f"unexpected path {u.path}")

    def get_bytes(self, url):
        self.byte_calls.append(url)
        return self.zips[url]


# ---------------------------------------------------------------- candles

class TestCandles:
    START = utc_ms("2026-02-01 00:00")
    END = START + 1500 * MIN

    def server(self, monkeypatch, clock, **kw):
        fake = FakeOkx(clock=clock, **kw).install(monkeypatch)
        rng = np.random.default_rng(7)
        rows = {}
        for k in range(-500, 2000):
            ts = self.START + k * MIN
            if 300 <= k < 340 or 900 <= k < 905:           # two holes
                continue
            rows[ts] = candle_row(ts, "0" if rng.random() < 0.03 else "1")
        fake.candles["DOGE-USDT-SWAP"] = rows
        expected = [t for t, r in sorted(rows.items()) if self.START <= t < self.END and r[8] == "1"]
        return fake, expected

    def test_backward_paging_confirm_filter_and_columns(self, monkeypatch, isolated):
        fake, expected = self.server(monkeypatch, isolated)
        df, meta = fv.fetch_candles(fv.OkxClient(), "DOGE-USDT-SWAP", self.START, self.END)

        assert df["timestamp"].to_list() == expected
        assert df.columns == ["timestamp", "open", "high", "low", "close", "volume", "volume_ccy"]
        assert df.dtypes == [pl.Int64] + [pl.Float64] * 6
        row = fake.candles["DOGE-USDT-SWAP"][expected[10]]
        assert df.row(10) == (int(row[0]), *map(float, row[1:7]))

        afters = [int(q["after"]) for _, q in fake.json_calls]
        assert afters[0] == self.END                       # half-open end
        assert all(b < a for a, b in zip(afters, afters[1:]))
        assert all(int(q["limit"]) <= fv.CANDLE_PAGE_LIMIT for _, q in fake.json_calls)
        # Pages shorter than asked (cap 100 < 300) did not end the walk early.
        assert len(fake.json_calls) >= (self.END - self.START) // MIN // fake.cap
        # Every page asks for one slice [floor, cursor) of at most `limit`
        # minutes, bounded below as well as above, and never below the start.
        for _, q in fake.json_calls:
            after, before, limit = int(q["after"]), int(q["before"]), int(q["limit"])
            assert limit == min(fv.CANDLE_PAGE_LIMIT, -(-(after - self.START) // MIN))
            assert before + 1 == max(self.START, after - limit * MIN)
        assert fake.json_calls[-1][1]["before"] == str(self.START - 1)
        # Nothing before the window was ever handed out.
        assert min(fake.served) >= self.START
        n_unconfirmed = sum(1 for t, r in fake.candles["DOGE-USDT-SWAP"].items()
                            if self.START <= t < self.END and r[8] != "1")
        assert meta["unconfirmed_dropped"] == n_unconfirmed > 0

    def test_dedupe_when_pages_overlap(self, monkeypatch, isolated):
        fake, expected = self.server(monkeypatch, isolated, inclusive_after=True)
        df, meta = fv.fetch_candles(fv.OkxClient(), "DOGE-USDT-SWAP", self.START, self.END)
        assert len(fake.served) > len(set(fake.served))     # the server did repeat rows
        assert df["timestamp"].to_list() == expected
        assert meta["duplicate_conflicts"] == 0

    def test_disagreeing_duplicates_are_counted(self, monkeypatch, isolated):
        start, end = self.START, self.START + 3 * MIN
        good = [candle_row(start + 2 * MIN), candle_row(start + MIN), candle_row(start)]
        twisted = list(good[1])
        twisted[4] = "9.9"
        pages = iter([ok(good[:2] + [twisted] + good[2:]), ok([])])
        monkeypatch.setattr(fv, "http_get_json", lambda url: next(pages))
        df, meta = fv.fetch_candles(fv.OkxClient(), "X", start, end)
        assert df.height == 3 and meta["duplicate_conflicts"] == 1
        assert df.filter(pl.col("timestamp") == start + MIN)["close"].item() == float(good[1][4])

    def test_clip_when_server_ignores_bounds(self, monkeypatch, isolated):
        fake, expected = self.server(monkeypatch, isolated, cap=300, honour_before=False, honour_limit=False)
        df, meta = fv.fetch_candles(fv.OkxClient(), "DOGE-USDT-SWAP", self.START, self.END)
        assert df["timestamp"].to_list() == expected
        assert min(fake.served) < self.START
        assert meta["served_outside_window"] > 0

    def test_history_exhausted_is_reported(self, monkeypatch, isolated):
        fake = FakeOkx().install(monkeypatch)
        first = self.START + 100 * MIN
        fake.candles["X"] = {first + k * MIN: candle_row(first + k * MIN) for k in range(200)}
        df, meta = fv.fetch_candles(fv.OkxClient(), "X", self.START, first + 200 * MIN)
        assert df.height == 200
        assert meta["no_data_before_ms"] == first
        assert fv.edge_gaps(df["timestamp"].to_numpy(), self.START, first + 200 * MIN, MIN) == [[self.START, first]]

    def test_non_advancing_cursor_raises(self, monkeypatch):
        page = ok([candle_row(self.END)])
        monkeypatch.setattr(fv, "http_get_json", lambda url: page)
        with pytest.raises(fv.FetchError):
            fv.fetch_candles(fv.OkxClient(), "X", self.START, self.END)

    def test_hole_longer_than_a_page_is_walked_over(self, monkeypatch, isolated):
        # An empty slice is a hole, not the end of history: the bars before it
        # are still fetched, and none from before the window is handed out.
        fake = FakeOkx(clock=isolated, cap=300).install(monkeypatch)
        rows = {self.START + k * MIN: candle_row(self.START + k * MIN)
                for k in range(-400, 1500) if not 200 <= k < 1100}             # 900-minute hole
        fake.candles["X"] = rows
        df, meta = fv.fetch_candles(fv.OkxClient(), "X", self.START, self.END)
        assert df["timestamp"].to_list() == [t for t in sorted(rows) if self.START <= t < self.END]
        assert min(fake.served) >= self.START
        assert meta["no_data_before_ms"] is None and meta["served_outside_window"] == 0


# ---------------------------------------------------------------- retries & throttle

class TestRetryAndThrottle:
    def test_backoff_on_http_errors(self, monkeypatch, isolated):
        outcomes = [OSError("reset"), urllib.error.URLError("timeout"), ok([1])]

        def flaky(url):
            o = outcomes.pop(0)
            if isinstance(o, Exception):
                raise o
            return o

        monkeypatch.setattr(fv, "http_get_json", flaky)
        assert fv.OkxClient().get_json("/x", {}, "candles")["data"] == [1]
        assert isolated.sleeps == [1.0, 2.0]

    def test_backoff_on_okx_error_codes(self, monkeypatch, isolated):
        outcomes = [{"code": "50011", "msg": "rate limit"}, {"code": "50001", "msg": "busy"}, ok([])]
        monkeypatch.setattr(fv, "http_get_json", lambda url: outcomes.pop(0))
        assert fv.OkxClient().get_json("/x", {}, "candles")["data"] == []
        assert isolated.sleeps == [1.0, 2.0]

    def test_gives_up_after_max_retries(self, monkeypatch, isolated):
        calls = []

        def broken(url):
            calls.append(url)
            raise ValueError("bad json")

        monkeypatch.setattr(fv, "http_get_json", broken)
        with pytest.raises(fv.FetchError):
            fv.OkxClient().get_json("/x", {}, "candles")
        assert len(calls) == fv.MAX_RETRIES
        assert isolated.sleeps == [2.0 ** k for k in range(fv.MAX_RETRIES - 1)]

    def test_corrupt_zip_is_retried(self, monkeypatch, isolated):
        good = make_zip([("DOGE-USDT-SWAP", 1e-4, 0)])
        outcomes = [good[: len(good) // 2], good]
        monkeypatch.setattr(fv, "http_get_bytes", lambda url: outcomes.pop(0))
        assert fv.OkxClient().get_zip("https://x/y.zip") == good
        assert isolated.sleeps == [1.0]

    def test_candle_requests_stay_under_20_per_2s(self, monkeypatch, isolated):
        fake = FakeOkx(clock=isolated).install(monkeypatch)
        client = fv.OkxClient()
        for _ in range(100):
            client.get_json(fv.CANDLES_PATH, {"instId": "X", "bar": "1m", "after": "1", "limit": "1"}, "candles")
        t = np.array(fake.call_times)
        busiest = max(int(((t >= x) & (t < x + 2.0)).sum()) for x in t)
        assert busiest < 20
        assert busiest == fv.RATE_LIMITS["candles"][0]


# ---------------------------------------------------------------- funding archive

class TestArchiveParsing:
    def test_utc8_month_cut(self):
        assert fv.utc8_month_span(2024, 9) == (utc_ms("2024-08-31 16:00"), utc_ms("2024-09-30 16:00"))
        assert fv.utc8_month_span(2024, 12) == (utc_ms("2024-11-30 16:00"), utc_ms("2024-12-31 16:00"))
        assert fv.utc8_month_of(utc_ms("2024-08-31 16:00")) == (2024, 9)
        assert fv.utc8_month_of(utc_ms("2024-08-31 16:00") - 1) == (2024, 8)
        # A window opening at 20:00 UTC on the last day already needs the next month's file only.
        assert fv.utc8_months(utc_ms("2024-08-31 20:00"), utc_ms("2024-10-01 00:00")) == [(2024, 9), (2024, 10)]
        assert fv.utc8_months(utc_ms("2024-12-15 00:00"), utc_ms("2025-01-31 16:00")) == [(2024, 12), (2025, 1)]
        assert fv.utc8_months(utc_ms("2024-12-15 00:00"), utc_ms("2025-01-31 16:01")) == [(2024, 12), (2025, 1), (2025, 2)]

    def test_parse_filters_instrument_and_handles_bom(self):
        ts0 = utc_ms("2024-09-01 00:00")
        data = make_zip([("DOGE-USDT-SWAP", 1e-4, ts0), ("BTC-USDT-SWAP", 5e-4, ts0),
                         ("DOGE-USDT-SWAP", -2e-5, ts0 + H8)])
        rows = fv.parse_funding_zip(data, "DOGE-USDT-SWAP")
        assert [(t, r) for t, r, _ in rows] == [(ts0, 1e-4), (ts0 + H8, -2e-5)]
        assert rows[0][2] == pytest.approx(0.9e-4)

    def test_parse_without_realized_column(self):
        data = make_zip([("DOGE-USDT-SWAP", 1e-4, 0)], bom=False,
                        header=("instrument_name", "funding_rate", "funding_time"))
        (row,) = fv.parse_funding_zip(data, "DOGE-USDT-SWAP")
        assert row[:2] == (0, 1e-4) and np.isnan(row[2])

    def test_parse_refuses_unknown_layout(self):
        data = make_zip([("DOGE-USDT-SWAP", 1e-4, 0)], header=("funding_rate", "funding_time"))
        with pytest.raises(ValueError):
            fv.parse_funding_zip(data, "DOGE-USDT-SWAP")

    def test_file_month_from_listing(self):
        url = archive_url("DOGE-USDT", 2024, 9)
        assert fv._file_month({"url": url}) == (2024, 9)
        assert fv._file_month({"url": "https://x/monthly/202409/f.zip"}) == (2024, 9)
        assert fv._file_month({"url": "https://x/f.zip", "dateTs": str(utc_ms("2024-08-31 16:00"))}) == (2024, 9)
        assert fv._file_month({"url": "https://x/f.zip", "dateTs": str(utc_ms("2024-09-01 00:00"))}) == (2024, 9)
        # The name beats a timestamp that may be a publication time.
        assert fv._file_month({"url": url, "dateTs": str(utc_ms("2024-10-02 00:00"))}) == (2024, 9)
        # Unknown or daily: no month, so never downloaded.
        assert fv._file_month({"url": "https://x/f.zip"}) is None
        assert fv._file_month({"url": "https://x/daily/20240901/f-2024-09-01.zip"}) is None

    def test_file_of_unknown_month_stops_before_download(self, monkeypatch):
        fake = FakeOkx().install(monkeypatch)
        monkeypatch.setattr(fv, "http_get_json", lambda url: ok([{"details": [{"instFamily": "DOGE-USDT",
                            "groupDetails": [{"url": "https://x/mystery.zip"}]}]}]))
        with pytest.raises(fv.FetchError):
            fv.fetch_funding(fv.OkxClient(), "DOGE-USDT-SWAP", utc_ms("2026-01-01 00:00"),
                             utc_ms("2026-02-01 00:00"), guarded=True)
        assert fake.byte_calls == []

    def test_listing_chunks_stay_within_ten_months(self, monkeypatch):
        fake = FakeOkx().install(monkeypatch)
        months = fv.utc8_months(utc_ms("2023-07-01 00:00"), utc_ms("2025-03-01 00:00"))
        assert len(months) == 21
        for y, m in months:
            fake.archive.setdefault("DOGE-USDT", {})[(y, m)] = archive_url("DOGE-USDT", y, m)
        fake.archive["BTC-USDT"] = {(2024, 1): archive_url("BTC-USDT", 2024, 1)}
        out = fv.list_funding_archive(fv.OkxClient(), "DOGE-USDT", months)
        assert sorted(out) == months
        assert all(len(v) == 1 for v in out.values())
        assert len(fake.json_calls) == 3
        for _, q in fake.json_calls:
            assert q["instFamilyList"] == "DOGE-USDT"
            assert int(q["end"]) - int(q["begin"]) < 9 * 31 * 86_400_000
        assert not fake.byte_calls                                   # listing only


class TestFundingRest:
    def test_paging_bounded_below(self, monkeypatch):
        fake = FakeOkx(cap=100).install(monkeypatch)
        base = utc_ms("2026-01-01 00:00")
        fake.funding["DOGE-USDT-SWAP"] = {base + k * H8: funding_rec("DOGE-USDT-SWAP", base + k * H8)
                                          for k in range(700)}
        lo, hi = base + 50 * H8 + 3_600_000, base + 650 * H8           # lo off-grid on purpose
        recs = fv.fetch_funding_rest(fv.OkxClient(), "DOGE-USDT-SWAP", lo, hi)
        ts = sorted(r[0] for r in recs)
        assert ts == [base + k * H8 for k in range(51, 650)]
        assert len(ts) == len(set(ts))
        assert all(src == fv.SRC_FUNDING_REST for *_, src in recs)
        assert min(fake.served) >= lo
        assert len(fake.json_calls) >= 6
        assert all(int(q["limit"]) <= fv.FUNDING_PAGE_LIMIT for _, q in fake.json_calls)
        assert all(int(q["before"]) >= lo - 1 for _, q in fake.json_calls)
        assert fake.json_calls[-1][1]["before"] == str(lo - 1)

    def test_hole_at_the_lower_bound_is_not_crossed(self, monkeypatch):
        # The in-sample supplement's REST funding starts where doge_2024 ends.
        # A three-week hole right after that border must not let any page
        # reach back into the protected settlements before it.
        fake = FakeOkx(cap=100).install(monkeypatch)
        lo = utc_ms("2025-09-23 20:00")
        hi = lo + 120 * 86_400_000
        first = -(-(lo - 60 * 86_400_000) // H8) * H8
        hole = (lo, lo + 21 * 86_400_000)
        fake.funding["DOGE-USDT-SWAP"] = {t: funding_rec("DOGE-USDT-SWAP", t)
                                          for t in range(first, hi + 30 * 86_400_000, H8)
                                          if not hole[0] <= t < hole[1]}
        recs = fv.fetch_funding_rest(fv.OkxClient(), "DOGE-USDT-SWAP", lo, hi)
        assert sorted(r[0] for r in recs) == [t for t in sorted(fake.funding["DOGE-USDT-SWAP"]) if lo <= t < hi]
        assert fake.served and min(fake.served) >= lo
        assert not any(spec.overlaps_protected(t, t + 1) for t in fake.served)
        for _, q in fake.json_calls:
            after, before, limit = int(q["after"]), int(q["before"]), int(q["limit"])
            assert limit <= fv.FUNDING_PAGE_LIMIT and before + 1 == max(lo, after - limit * fv.FUNDING_SLOT_MS)

    def test_each_page_asks_for_exactly_its_slice(self, monkeypatch):
        # 655 settlements = 5240 h: thirteen full slices, then one of 40 h. A
        # page asking for more rows than its slice holds would depend on which
        # end OKX serves first; `limit` is the slice's length in hours.
        fake = FakeOkx(cap=100).install(monkeypatch)
        base = utc_ms("2026-01-01 00:00")
        fake.funding["DOGE-USDT-SWAP"] = {base + k * H8: funding_rec("DOGE-USDT-SWAP", base + k * H8)
                                          for k in range(700)}
        fv.fetch_funding_rest(fv.OkxClient(), "DOGE-USDT-SWAP", base, base + 655 * H8)
        limits = []
        for _, q in fake.json_calls:
            after, before, limit = int(q["after"]), int(q["before"]), int(q["limit"])
            assert limit == -(-(after - (before + 1)) // fv.FUNDING_SLOT_MS) <= fv.FUNDING_PAGE_LIMIT
            limits.append(limit)
        assert limits[-1] < fv.FUNDING_PAGE_LIMIT                        # the short last slice was asked for

    def test_hole_longer_than_a_page_is_walked_over(self, monkeypatch):
        # A 720-hour hole is longer than one page's slice: the empty slices are
        # stepped over and the settlements before the hole are still fetched.
        fake = FakeOkx(cap=100).install(monkeypatch)
        lo = utc_ms("2026-01-01 00:00")
        hi = lo + 90 * 86_400_000
        hole = (lo + 30 * 86_400_000, lo + 60 * 86_400_000)
        fake.funding["DOGE-USDT-SWAP"] = {t: funding_rec("DOGE-USDT-SWAP", t) for t in range(lo, hi, H8)
                                          if not hole[0] <= t < hole[1]}
        recs = fv.fetch_funding_rest(fv.OkxClient(), "DOGE-USDT-SWAP", lo, hi)
        assert sorted(r[0] for r in recs) == sorted(fake.funding["DOGE-USDT-SWAP"])


def funding_server(monkeypatch, clock):
    return FakeOkx(clock=clock, cap=100).install(monkeypatch)


def add_month_file(fake, inst, y, m, drop=(), extra=(), others=True):
    fam = inst[: -len("-SWAP")]
    lo, hi = fv.utc8_month_span(y, m)
    rows = [(inst, rate_of(t), t) for t in range(-(-lo // H8) * H8, hi, H8) if t not in drop]
    rows += [(inst, rate, t) for t, rate in extra]
    if others:
        rows += [("BTC-USDT-SWAP", 7e-4, t) for t in range(-(-lo // H8) * H8, hi, H8)]
    url = archive_url(fam, y, m)
    fake.archive.setdefault(fam, {})[(y, m)] = url
    fake.zips[url] = make_zip(rows)
    return url


def run_main(tmp_path, kind, inst_ids, start, end, *extra):
    out_dir = tmp_path / "out"
    manifest = tmp_path / "results" / "manifest.json"
    code = fv.main(["--kind", kind, "--inst-id", *inst_ids, "--start", start, "--end", end,
                    "--out-dir", str(out_dir), "--manifest", str(manifest), *extra])
    return code, out_dir, manifest


class TestFundingEndToEnd:
    INST = "DOGE-USDT-SWAP"

    def test_archive_then_rest(self, monkeypatch, isolated, tmp_path):
        fake = funding_server(monkeypatch, isolated)
        gap = utc_ms("2026-02-10 08:00")
        odd = utc_ms("2026-02-15 04:00")
        boundary = utc_ms("2026-03-31 16:00")                         # first settlement of April (UTC+8)
        add_month_file(fake, self.INST, 2026, 1, extra=[(utc_ms("2026-01-31 16:00"), rate_of(utc_ms("2026-01-31 16:00")))])
        add_month_file(fake, self.INST, 2026, 2, drop={gap}, extra=[(odd, 3e-4)])
        add_month_file(fake, self.INST, 2026, 3, extra=[(boundary, 9e-4)])
        lo_all = utc_ms("2025-12-01 00:00")
        fake.funding[self.INST] = {lo_all + k * H8: funding_rec(self.INST, lo_all + k * H8) for k in range(600)}

        start, end = "2026-01-15 00:00", "2026-04-10 00:00"
        code, out_dir, manifest = run_main(tmp_path, "funding", [self.INST], start, end)
        assert code == fv.EXIT_OK

        s_ms, e_ms = utc_ms(start), utc_ms(end)
        df = pl.read_parquet(out_dir / self.INST / "funding_rates.parquet")
        assert df.columns == ["timestamp", "rate", "realized_rate"]
        expected = sorted({t for t in range(s_ms, e_ms, H8) if t != gap} | {odd})
        assert df["timestamp"].to_list() == expected
        # The archive wins a shared settlement against REST (§4.2's primary source).
        assert df.filter(pl.col("timestamp") == boundary)["rate"].item() == 9e-4
        assert df.filter(pl.col("timestamp") == odd)["rate"].item() == 3e-4
        assert df["realized_rate"].is_not_nan().all()

        m = json.loads(manifest.read_text())
        (e,) = m["series"]
        assert m["kind"] == "funding" and m["window"]["start_ms"] == s_ms and m["window"]["end_ms"] == e_ms
        assert e["sha256"] == sha256_file(out_dir / self.INST / "funding_rates.parquet")
        assert e["rows"] == len(expected) and e["step_ms"] == H8
        assert e["gaps"] == [[gap, gap + H8]]
        assert e["spacing_other_than_expected"] == [
            [gap - H8, gap + H8], [utc_ms("2026-02-15 00:00"), odd], [odd, utc_ms("2026-02-15 08:00")]]
        assert e["archive_months_used"] == ["2026-01", "2026-02", "2026-03"]
        assert e["rest_ranges"] == [[boundary, e_ms]]
        assert e["duplicate_conflicts"] == 1
        assert e["source_by_day"]["2026-01-15"] == fv.SRC_FUNDING_ARCHIVE
        assert e["source_by_day"]["2026-03-31"] == fv.SRC_FUNDING_ARCHIVE
        assert e["source_by_day"]["2026-04-05"] == fv.SRC_FUNDING_REST
        assert "2026-04-10" not in e["source_by_day"]
        assert e["edge_gaps"] == []

        rest_calls = [q for p, q in fake.json_calls if p == fv.FUNDING_REST_PATH]
        assert int(rest_calls[0]["after"]) == e_ms
        assert rest_calls[-1]["before"] == str(boundary - 1)

    @pytest.mark.parametrize("unlocked", [False, True])
    def test_archive_month_straddling_a_protected_window(self, monkeypatch, isolated, tmp_path, unlocked):
        # The in-sample supplement starts 2025-09-24 00:00, but the September
        # 2025 file (UTC+8) also holds doge_2024's last settlements.
        fake = funding_server(monkeypatch, isolated)
        sept = add_month_file(fake, self.INST, 2025, 9)
        add_month_file(fake, self.INST, 2025, 10)
        add_month_file(fake, self.INST, 2025, 11)
        lo_all = utc_ms("2025-09-01 00:00")
        fake.funding[self.INST] = {lo_all + k * H8: funding_rec(self.INST, lo_all + k * H8) for k in range(300)}
        extra = []
        if unlocked:
            unlock(monkeypatch, tmp_path)
            extra = ["--allow-test-data"]

        start, end = "2025-09-24 00:00", "2025-11-10 00:00"
        code, out_dir, manifest = run_main(tmp_path, "funding", [self.INST], start, end, *extra)
        assert code == fv.EXIT_OK
        s_ms, e_ms = utc_ms(start), utc_ms(end)
        df = pl.read_parquet(out_dir / self.INST / "funding_rates.parquet")
        assert df["timestamp"].to_list() == list(range(s_ms, e_ms, H8))
        m = json.loads(manifest.read_text())
        e = m["series"][0]
        assert m["overlaps_protected"] is False and m["test_data_unlocked"] is unlocked

        if unlocked:
            # doge_2024 closed long ago, so once unlocked the whole file is fine.
            assert sept in fake.byte_calls
            assert e["archive_months_skipped_protected"] == []
            assert e["source_by_day"]["2025-09-25"] == fv.SRC_FUNDING_ARCHIVE
        else:
            assert sept not in fake.byte_calls
            assert e["archive_months_skipped_protected"] == ["2025-09"]
            assert e["rest_ranges"] == [[s_ms, utc_ms("2025-09-30 16:00")]]
            assert e["source_by_day"]["2025-09-25"] == fv.SRC_FUNDING_REST
            assert e["source_by_day"]["2025-10-05"] == fv.SRC_FUNDING_ARCHIVE
            # Not a single protected settlement was handed to the client.
            assert not any(spec.overlaps_protected(t, t + 1) for t in fake.served)


    @pytest.mark.parametrize("now", [NOW, FWD_CLOSE])
    def test_archive_month_touching_an_open_window_waits_for_its_close(self, monkeypatch, isolated, tmp_path, now):
        # In-sample funding for September 2026 needs the 2026-09 file (UTC+8),
        # which also holds doge_fwd's first week. Unlocked or not, that file
        # is not downloaded before doge_fwd closes.
        monkeypatch.setattr(fv, "_wall_ms", lambda: now)
        fake = funding_server(monkeypatch, isolated)
        sept = add_month_file(fake, self.INST, 2026, 9)
        lo_all = utc_ms("2026-08-01 00:00")
        fake.funding[self.INST] = {lo_all + k * H8: funding_rec(self.INST, lo_all + k * H8) for k in range(300)}
        unlock(monkeypatch, tmp_path)
        start, end = "2026-09-01 00:00", "2026-09-23 20:00"
        code, out_dir, manifest = run_main(tmp_path, "funding", [self.INST], start, end, "--allow-test-data")
        assert code == fv.EXIT_OK
        df = pl.read_parquet(out_dir / self.INST / "funding_rates.parquet")
        assert df["timestamp"].to_list() == list(range(utc_ms(start), utc_ms(end), H8))
        e = json.loads(manifest.read_text())["series"][0]
        assert not any(spec.overlaps_protected(t, t + 1) for t in fake.served)
        if now < FWD_CLOSE:
            assert sept not in fake.byte_calls
            assert e["archive_months_skipped_protected"] == ["2026-09"]
        else:
            assert sept in fake.byte_calls
            assert e["archive_months_used"] == ["2026-09"]


# ---------------------------------------------------------------- guard & write-once

DOGE_2023 = ("2023-09-23 20:00", "2024-09-23 20:00")


class TestGuard:
    @pytest.mark.parametrize("kind,inst", [("candles", "DOGE-USDT-SWAP"), ("funding", "DOGE-USDT-SWAP")])
    @pytest.mark.parametrize("flag,frozen", [(False, False), (True, False), (False, True)])
    def test_protected_window_refused(self, monkeypatch, tmp_path, kind, inst, flag, frozen):
        fake = FakeOkx().install(monkeypatch)
        if frozen:
            unlock(monkeypatch, tmp_path)
        extra = ["--allow-test-data"] if flag else []
        code, out_dir, manifest = run_main(tmp_path, kind, [inst], *DOGE_2023, *extra)
        assert code == fv.EXIT_PROTECTED
        assert fake.json_calls == [] and fake.byte_calls == []
        assert not out_dir.exists() and not manifest.exists()

    @pytest.mark.parametrize("state,needle", [
        ("touched", "not a git work tree"),                  # `touch docs/preregistration-freeze.md`
        ("draft", "is not committed"),                       # written, not committed (§10 step 3)
        ("unpushed", "refs/remotes/origin/main"),            # committed, not pushed
        ("empty", "freeze-commit"),                          # committed and pushed, but says nothing
        ("wrong_commit", "but the freeze amendment records"),
    ])
    def test_a_freeze_that_does_not_verify_unlocks_nothing(self, monkeypatch, tmp_path, messages, state, needle):
        fake = FakeOkx().install(monkeypatch)
        root = tmp_path / "freeze"
        if state == "touched":
            (root / "docs").mkdir(parents=True)
            (root / "docs" / "preregistration-freeze.md").write_text("")
        elif state == "draft":
            frozen_repo(root, commit=False)
        elif state == "unpushed":
            frozen_repo(root, push=False)
        elif state == "empty":
            frozen_repo(root, amendment="")
        else:
            frozen_repo(root, amendment=freeze.render_amendment_blocks("f" * 40, insample_text()))
        monkeypatch.setattr(fv, "FREEZE_ROOT", root)
        code, out_dir, manifest = run_main(
            tmp_path, "candles", ["DOGE-USDT-SWAP"], DOGE_2023[0], "2023-09-23 21:00", "--allow-test-data")
        assert code == fv.EXIT_PROTECTED
        assert fake.json_calls == [] and not out_dir.exists() and not manifest.exists()
        assert needle in "\n".join(messages)

    def test_window_overlapping_by_one_minute_is_refused(self, monkeypatch, tmp_path):
        fake = FakeOkx().install(monkeypatch)
        code, *_ = run_main(tmp_path, "candles", ["DOGE-USDT"], "2025-09-23 19:59", "2025-09-24 00:00")
        assert code == fv.EXIT_PROTECTED and fake.json_calls == []

    def test_window_touching_protected_range_with_a_hole_at_the_border(self, monkeypatch, isolated, tmp_path):
        # Starts exactly where doge_2024 ends (half-open: no overlap, allowed).
        # The first minutes after the border are missing, so a page asking for
        # "300 bars older than the cursor" would reach back across it.
        border = utc_ms("2025-09-23 20:00")
        assert spec.overlaps_protected(border - MIN, border) and not spec.overlaps_protected(border, border + MIN)
        fake = FakeOkx(clock=isolated, cap=300).install(monkeypatch)
        hole = set(range(border, border + 7 * MIN, MIN))
        fake.candles["DOGE-USDT"] = {t: candle_row(t) for t in range(border - 600 * MIN, border + 900 * MIN, MIN)
                                     if t not in hole}
        code, out_dir, manifest = run_main(tmp_path, "candles", ["DOGE-USDT"], "2025-09-23 20:00", "2025-09-24 06:01")
        assert code == fv.EXIT_OK
        assert fake.served and not any(spec.overlaps_protected(t, t + 1) for t in fake.served)
        assert all(int(q["before"]) >= border - 1 for _, q in fake.json_calls)
        e = json.loads(manifest.read_text())["series"][0]
        assert e["rows"] == 601 - 7
        assert e["edge_gaps"] == [[border, border + 7 * MIN]]

    def test_window_in_the_future_gets_no_bars(self, monkeypatch, isolated, tmp_path):
        # A window after doge_fwd does not overlap it, but OKX answers a bare
        # "older than <future>" with its latest bars, which today are doge_fwd's.
        fake = FakeOkx(clock=isolated, cap=300).install(monkeypatch)
        fake.candles["DOGE-USDT-SWAP"] = {t: candle_row(t) for t in range(NOW - 2000 * MIN, NOW, MIN)}
        start, end = "2026-12-01 00:00", "2026-12-01 06:00"
        assert not spec.overlaps_protected(utc_ms(start), utc_ms(end))
        code, out_dir, manifest = run_main(tmp_path, "candles", ["DOGE-USDT-SWAP"], start, end)
        assert code == fv.EXIT_OK
        assert fake.served == []
        assert all(int(q["before"]) >= utc_ms(start) - 1 for _, q in fake.json_calls)
        e = json.loads(manifest.read_text())["series"][0]
        assert e["rows"] == 0 and e["edge_gaps"] == [[utc_ms(start), utc_ms(end)]]

    def test_unlocked_after_freeze(self, monkeypatch, isolated, tmp_path):
        fake = FakeOkx(clock=isolated).install(monkeypatch)
        start = utc_ms(DOGE_2023[0])
        fake.candles["DOGE-USDT-SWAP"] = {start + k * MIN: candle_row(start + k * MIN) for k in range(120)}
        repo = unlock(monkeypatch, tmp_path)
        code, out_dir, manifest = run_main(
            tmp_path, "candles", ["DOGE-USDT-SWAP"], DOGE_2023[0], "2023-09-23 22:00", "--allow-test-data")
        assert code == fv.EXIT_OK
        m = json.loads(manifest.read_text())
        assert m["overlaps_protected"] is True and m["test_data_unlocked"] is True
        # The manifest names the remote the unlock was checked against and what its main was.
        assert m["remote_url"] == freeze.PUBLIC_REMOTE
        assert m["remote_main_commit"] == git(repo, "rev-parse", "HEAD")
        assert m["window"]["start_ms"] == start and m["window"]["end_ms"] == utc_ms("2023-09-23 22:00")
        assert m["series"][0]["rows"] == 120

    @pytest.mark.parametrize("case,needle", [
        ("hand_written_ref", "is not on main"),        # never pushed; `git update-ref` stands in for the push
        ("edited_after_push", "is not on main"),       # pushed, then changed again, ref moved by hand
        ("ls_remote_fails", "ls-remote could not"),
        ("remote_main_unknown", "does not have"),      # main moved on the remote; not fetched here
    ])
    def test_the_push_is_asked_of_the_public_remote(self, monkeypatch, isolated, tmp_path, messages, case, needle):
        # Every case passes freeze.verify_freeze, which can read only the local
        # refs/remotes/origin/main; the remote itself must agree.
        fake = FakeOkx(clock=isolated).install(monkeypatch)
        start = utc_ms(DOGE_2023[0])
        fake.candles["DOGE-USDT-SWAP"] = {start + k * MIN: candle_row(start + k * MIN) for k in range(60)}
        repo = frozen_repo(tmp_path / "frozen", push=case != "hand_written_ref")
        if case == "hand_written_ref":
            git(repo, "push", "-q", "origin", "prereg-freeze^{commit}:refs/heads/main")    # code, not the amendment
            git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif case == "edited_after_push":
            with open(repo / "docs" / "preregistration-freeze.md", "a") as f:
                f.write("\nA sentence added after the push.\n")
            git(repo, "commit", "-q", "-am", "edit the amendment")
            git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        elif case == "ls_remote_fails":
            monkeypatch.setattr(fv, "_ls_remote_main", lambda url: None)
        else:
            other = tmp_path / "elsewhere"
            git(tmp_path, "clone", "-q", "-b", "main", freeze.PUBLIC_REMOTE, str(other))
            git(other, "commit", "-q", "--allow-empty", "-m", "a later commit")
            git(other, "push", "-q", "origin", "main")
        freeze.verify_freeze(repo)                                         # local evidence alone would unlock
        monkeypatch.setattr(fv, "FREEZE_ROOT", repo)
        code, out_dir, manifest = run_main(
            tmp_path, "candles", ["DOGE-USDT-SWAP"], DOGE_2023[0], "2023-09-23 21:00", "--allow-test-data")
        assert code == fv.EXIT_PROTECTED
        assert fake.json_calls == [] and not out_dir.exists() and not manifest.exists()
        assert needle in "\n".join(messages)

    def test_ls_remote_reads_main_from_the_remote(self, tmp_path):
        # The real seam, against a local bare repository (no network).
        repo = frozen_repo(tmp_path / "frozen")
        assert REAL_LS_REMOTE(freeze.PUBLIC_REMOTE) == git(repo, "rev-parse", "HEAD")
        empty = tmp_path / "empty.git"
        git(tmp_path, "init", "-q", "--bare", str(empty))
        assert REAL_LS_REMOTE(str(empty)) is None                          # no main yet
        assert REAL_LS_REMOTE(str(tmp_path / "missing.git")) is None

    def test_open_window_refused_even_when_unlocked(self, monkeypatch, isolated, tmp_path):
        # §10: doge_fwd's download (step 4) comes after 2026-11-23 20:00.
        fake = FakeOkx(clock=isolated, cap=300).install(monkeypatch)
        lo = utc_ms("2026-09-23 20:00")
        fake.candles["DOGE-USDT-SWAP"] = {t: candle_row(t) for t in range(lo, lo + 120 * MIN, MIN)}
        unlock(monkeypatch, tmp_path)
        argv = ("candles", ["DOGE-USDT-SWAP"], "2026-09-23 20:00", "2026-09-23 22:00", "--allow-test-data")
        code, out_dir, manifest = run_main(tmp_path, *argv)
        assert code == fv.EXIT_PROTECTED and fake.json_calls == [] and not manifest.exists()
        monkeypatch.setattr(fv, "_wall_ms", lambda: FWD_CLOSE - 1)
        code, *_ = run_main(tmp_path, *argv)
        assert code == fv.EXIT_PROTECTED and fake.json_calls == []
        # Closed, unlocked: allowed. Closed, locked: still refused.
        monkeypatch.setattr(fv, "_wall_ms", lambda: FWD_CLOSE)
        code, out_dir, manifest = run_main(tmp_path, *argv[:-1])
        assert code == fv.EXIT_PROTECTED and fake.json_calls == []
        code, out_dir, manifest = run_main(tmp_path, *argv)
        assert code == fv.EXIT_OK
        assert json.loads(manifest.read_text())["series"][0]["rows"] == 120

    def test_insample_supplement_border_is_not_crossed(self, monkeypatch, isolated, tmp_path):
        # The supplement starts four hours after doge_2024 ends. A hole of 570
        # minutes at its start leaves 30 bars in the window, so a page asking
        # for "300 bars older than 10:00" would take those, the four unprotected
        # hours (240 bars) and 30 of doge_2024's. No page may reach back.
        fake = FakeOkx(clock=isolated, cap=300).install(monkeypatch)
        border = utc_ms("2025-09-23 20:00")
        start = utc_ms("2025-09-24 00:00")
        hole = set(range(start, start + 570 * MIN, MIN))
        fake.candles["DOGE-USDT"] = {border + k * MIN: candle_row(border + k * MIN) for k in range(-600, 900)
                                     if border + k * MIN not in hole}
        code, out_dir, manifest = run_main(tmp_path, "candles", ["DOGE-USDT"], "2025-09-24 00:00", "2025-09-24 10:00")
        assert code == fv.EXIT_OK
        assert min(fake.served) >= start
        assert not any(spec.overlaps_protected(t, t + 1) for t in fake.served)
        assert json.loads(manifest.read_text())["series"][0]["rows"] == 600 - 570


class TestClockAndCode:
    """OKX's clock decides whether a window has closed; only this checkout's code runs."""

    def test_wrong_local_clock_is_refused(self, monkeypatch, isolated, tmp_path):
        # A local clock set past doge_fwd's close would let an unlocked
        # doge_fwd download run today: OKX can serve bars only up to its real
        # now, yet the manifest's window would claim the whole set.
        fake = FakeOkx(clock=isolated, cap=300).install(monkeypatch)
        fake.candles["DOGE-USDT-SWAP"] = {t: candle_row(t) for t in range(NOW - 600 * MIN, NOW, MIN)}
        unlock(monkeypatch, tmp_path)
        monkeypatch.setattr(fv, "_wall_ms", lambda: FWD_CLOSE + MIN)
        monkeypatch.setattr(fv, "_server_ms", lambda: NOW)
        code, out_dir, manifest = run_main(tmp_path, "candles", ["DOGE-USDT-SWAP"],
                                           fv.fmt_utc(FWD_CLOSE - 2 * 86_400_000), fv.fmt_utc(FWD_CLOSE),
                                           "--allow-test-data")
        assert code == fv.EXIT_FETCH_FAILED
        assert fake.json_calls == [] and not out_dir.exists() and not manifest.exists()

    @pytest.mark.parametrize("off_ms", [fv.MAX_CLOCK_SKEW_MS + 1, -fv.MAX_CLOCK_SKEW_MS - 1])
    def test_any_skew_beyond_a_minute_is_refused(self, monkeypatch, tmp_path, off_ms):
        fake = FakeOkx().install(monkeypatch)
        monkeypatch.setattr(fv, "_server_ms", lambda: NOW - off_ms)
        code, *_ = run_main(tmp_path, "candles", ["DOGE-USDT"], "2026-01-01 00:00", "2026-01-02 00:00")
        assert code == fv.EXIT_FETCH_FAILED and fake.json_calls == []

    def test_unreadable_okx_clock_is_refused(self, monkeypatch, tmp_path):
        fake = FakeOkx().install(monkeypatch)

        def unreachable():
            raise fv.FetchError("no answer")

        monkeypatch.setattr(fv, "_server_ms", unreachable)
        code, *_ = run_main(tmp_path, "candles", ["DOGE-USDT"], "2026-01-01 00:00", "2026-01-02 00:00")
        assert code == fv.EXIT_FETCH_FAILED and fake.json_calls == []

    @pytest.mark.parametrize("local,okx,expected", [
        (FWD_CLOSE - 30_000, FWD_CLOSE, fv.EXIT_OK),               # OKX says closed: allowed
        (FWD_CLOSE + 30_000, FWD_CLOSE - 1, fv.EXIT_PROTECTED),    # OKX says open: refused
    ])
    def test_okx_clock_decides_whether_a_window_has_closed(self, monkeypatch, isolated, tmp_path,
                                                            local, okx, expected):
        fake = FakeOkx(clock=isolated, cap=300).install(monkeypatch)
        lo = utc_ms("2026-09-23 20:00")
        fake.candles["DOGE-USDT-SWAP"] = {t: candle_row(t) for t in range(lo, lo + 120 * MIN, MIN)}
        unlock(monkeypatch, tmp_path)
        monkeypatch.setattr(fv, "_wall_ms", lambda: local)
        monkeypatch.setattr(fv, "_server_ms", lambda: okx)
        code, *_ = run_main(tmp_path, "candles", ["DOGE-USDT-SWAP"], "2026-09-23 20:00", "2026-09-23 22:00",
                            "--allow-test-data")
        assert code == expected

    def test_server_time_comes_from_public_time(self, monkeypatch):
        urls = []

        def fake_get(url):
            urls.append(urllib.parse.urlparse(url).path)
            return ok([{"ts": str(NOW)}])

        monkeypatch.setattr(fv, "http_get_json", fake_get)
        assert REAL_SERVER_MS() == NOW and urls == [fv.TIME_PATH]
        for bad in (ok([]), ok([{}]), {"code": "0"}):
            monkeypatch.setattr(fv, "http_get_json", lambda url: bad)
            with pytest.raises(fv.FetchError):
                REAL_SERVER_MS()

    def test_a_module_loaded_from_elsewhere_is_refused(self, monkeypatch, tmp_path, messages):
        # As if PYTHONPATH had put another checkout's research package first.
        planted = types.ModuleType("research.planted")
        planted.__file__ = str(tmp_path / "other-checkout" / "research" / "planted.py")
        monkeypatch.setitem(sys.modules, "research.planted", planted)
        fake = FakeOkx().install(monkeypatch)
        asked = []
        monkeypatch.setattr(fv, "_server_ms", lambda: asked.append(1) or NOW)
        code, out_dir, manifest = run_main(tmp_path, "candles", ["DOGE-USDT"], "2026-01-01 00:00", "2026-01-02 00:00")
        assert code == fv.EXIT_PROTECTED
        assert asked == [] and fake.json_calls == [] and not out_dir.exists() and not manifest.exists()
        assert "research.planted" in "\n".join(messages)


class TestWriteOnceAndManifest:
    def test_existing_output_refused_before_any_request(self, monkeypatch, tmp_path):
        fake = FakeOkx().install(monkeypatch)
        target = tmp_path / "out" / "DOGE-USDT" / "candles_1m.parquet"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"keep")
        code, _, manifest = run_main(tmp_path, "candles", ["DOGE-USDT"], "2026-01-01 00:00", "2026-01-02 00:00")
        assert code == fv.EXIT_EXISTS
        assert fake.json_calls == [] and target.read_bytes() == b"keep" and not manifest.exists()

    def test_existing_manifest_refused(self, monkeypatch, tmp_path):
        fake = FakeOkx().install(monkeypatch)
        manifest = tmp_path / "results" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        code, out_dir, _ = run_main(tmp_path, "funding", ["DOGE-USDT-SWAP"], "2026-01-01 00:00", "2026-01-02 00:00")
        assert code == fv.EXIT_EXISTS
        assert fake.json_calls == [] and manifest.read_text() == "{}" and not out_dir.exists()

    def test_candle_manifest(self, monkeypatch, isolated, tmp_path):
        fake = FakeOkx(clock=isolated, cap=300).install(monkeypatch)
        start = utc_ms("2026-03-01 00:00")
        for inst in ("DOGE-USDT-SWAP", "DOGE-USDT"):
            fake.candles[inst] = {start + k * MIN: candle_row(start + k * MIN)
                                  for k in range(5, 2 * 1440 - 10) if not 100 <= k < 103}
        code, out_dir, manifest = run_main(
            tmp_path, "candles", ["DOGE-USDT-SWAP", "DOGE-USDT"], "2026-03-01 00:00", "2026-03-04 00:00")
        assert code == fv.EXIT_OK
        m = json.loads(manifest.read_text())
        assert [e["inst_id"] for e in m["series"]] == ["DOGE-USDT-SWAP", "DOGE-USDT"]
        for e in m["series"]:
            path = out_dir / e["inst_id"] / "candles_1m.parquet"
            assert e["sha256"] == sha256_file(path)
            assert e["rows"] == pl.read_parquet(path).height == 2 * 1440 - 15 - 3
            assert e["first_ts"] == start + 5 * MIN and e["last_ts"] == start + (2 * 1440 - 11) * MIN
            assert e["gaps"] == [[start + 100 * MIN, start + 103 * MIN]]
            assert e["edge_gaps"] == [[start, start + 5 * MIN],
                                      [start + (2 * 1440 - 10) * MIN, utc_ms("2026-03-04 00:00")]]
            assert e["source_by_day"] == {"2026-03-01": fv.SRC_CANDLES_REST,
                                          "2026-03-02": fv.SRC_CANDLES_REST,
                                          "2026-03-03": fv.SRC_MISSING}
            assert e["kind"] == "candles" and e["step_ms"] == MIN

    def test_source_by_day_labels(self):
        d0 = utc_ms("2026-01-01 00:00")
        ts = np.array([d0, d0 + H8, d0 + 86_400_000 + H8])
        src = ["archive_module_3", "rest_funding_rate_history", "archive_module_3"]
        out = fv.source_by_day(ts, src, d0 - 4 * 3_600_000, d0 + 3 * 86_400_000, H8)
        assert out == {
            "2025-12-31": fv.SRC_NONE_EXPECTED,          # window opens at 20:00, next settlement is 00:00
            "2026-01-01": "archive_module_3+rest_funding_rate_history",
            "2026-01-02": "archive_module_3",
            "2026-01-03": fv.SRC_MISSING,
        }

    @pytest.mark.parametrize("argv_tail", [
        ["--kind", "candles", "--inst-id", "X", "--start", "2026-01-02 00:00", "--end", "2026-01-01 00:00"],
        ["--kind", "candles", "--inst-id", "X", "--start", "2026-01-01", "--end", "2026-01-02 00:00"],
        ["--kind", "funding", "--inst-id", "DOGE-USDT", "--start", "2026-01-01 00:00", "--end", "2026-01-02 00:00"],
        ["--kind", "candles", "--inst-id", "X", "X", "--start", "2026-01-01 00:00", "--end", "2026-01-02 00:00"],
    ])
    def test_usage_errors(self, monkeypatch, tmp_path, argv_tail):
        fake = FakeOkx().install(monkeypatch)
        code = fv.main(argv_tail + ["--out-dir", str(tmp_path / "o"), "--manifest", str(tmp_path / "m.json")])
        assert code == fv.EXIT_USAGE and fake.json_calls == []


# ---------------------------------------------------------------- one manifest per set

class TestOneManifestPerSet:
    """--candles / --funding: everything a set needs, one invocation, one manifest (§10 step 4)."""

    START, END = "2026-03-01 00:00", "2026-03-03 00:00"

    def server(self, monkeypatch, clock):
        fake = FakeOkx(clock=clock, cap=300).install(monkeypatch)
        s = utc_ms(self.START)
        for inst in ("DOGE-USDT-SWAP", "DOGE-USDT"):
            fake.candles[inst] = {s + k * MIN: candle_row(s + k * MIN) for k in range(2 * 1440)}
        base = utc_ms("2026-02-20 00:00")
        fake.funding["DOGE-USDT-SWAP"] = {base + k * H8: funding_rec("DOGE-USDT-SWAP", base + k * H8)
                                          for k in range(60)}
        return fake

    def run(self, tmp_path, *argv):
        out_dir = tmp_path / "out"
        manifest = tmp_path / "results" / "manifest_set.json"
        code = fv.main([*argv, "--start", self.START, "--end", self.END,
                        "--out-dir", str(out_dir), "--manifest", str(manifest)])
        return code, out_dir, manifest

    def test_candles_and_funding_in_one_manifest(self, monkeypatch, isolated, tmp_path):
        self.server(monkeypatch, isolated)
        code, out_dir, manifest = self.run(
            tmp_path, "--candles", "DOGE-USDT-SWAP", "DOGE-USDT", "--funding", "DOGE-USDT-SWAP")
        assert code == fv.EXIT_OK
        m = json.loads(manifest.read_text())
        assert m["kind"] == "candles+funding"
        # run_oos.check_manifest reads these two keys: the window must cover
        # the set, and test data must have been unlocked for a test set.
        assert (m["window"]["start_ms"], m["window"]["end_ms"]) == (utc_ms(self.START), utc_ms(self.END))
        assert m["window"]["half_open"] is True
        assert m["overlaps_protected"] is False and m["test_data_unlocked"] is False
        assert m["remote_url"] is None and m["remote_main_commit"] is None
        assert [(e["kind"], e["inst_id"]) for e in m["series"]] == [
            ("candles", "DOGE-USDT-SWAP"), ("candles", "DOGE-USDT"), ("funding", "DOGE-USDT-SWAP")]
        for e in m["series"]:
            path = out_dir / e["inst_id"] / fv.OUTPUT_FILE[e["kind"]]
            assert e["sha256"] == sha256_file(path)
            assert e["step_ms"] == fv.STEP_MS[e["kind"]]
        assert m["series"][0]["rows"] == m["series"][1]["rows"] == 2 * 1440
        assert m["series"][2]["rows"] == 6  # six 8-hour settlements in two days
        assert pl.read_parquet(out_dir / "DOGE-USDT-SWAP" / "funding_rates.parquet").columns == \
            ["timestamp", "rate", "realized_rate"]

    def test_only_funding_is_allowed_too(self, monkeypatch, isolated, tmp_path):
        self.server(monkeypatch, isolated)
        code, _, manifest = self.run(tmp_path, "--funding", "DOGE-USDT-SWAP")
        assert code == fv.EXIT_OK
        assert json.loads(manifest.read_text())["kind"] == "funding"

    @pytest.mark.parametrize("flag", [False, True])
    def test_guard_judges_the_whole_request(self, monkeypatch, tmp_path, flag):
        fake = FakeOkx().install(monkeypatch)
        out_dir, manifest = tmp_path / "out", tmp_path / "m.json"
        argv = ["--candles", "DOGE-USDT-SWAP", "DOGE-USDT", "--funding", "DOGE-USDT-SWAP",
                "--start", DOGE_2023[0], "--end", DOGE_2023[1],
                "--out-dir", str(out_dir), "--manifest", str(manifest)]
        code = fv.main(argv + (["--allow-test-data"] if flag else []))
        assert code == fv.EXIT_PROTECTED
        assert fake.json_calls == [] and fake.byte_calls == []
        assert not out_dir.exists() and not manifest.exists()

    def test_any_existing_output_stops_before_any_request(self, monkeypatch, tmp_path):
        fake = FakeOkx().install(monkeypatch)
        target = tmp_path / "out" / "DOGE-USDT-SWAP" / "funding_rates.parquet"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"keep")
        code, out_dir, manifest = self.run(
            tmp_path, "--candles", "DOGE-USDT-SWAP", "--funding", "DOGE-USDT-SWAP")
        assert code == fv.EXIT_EXISTS
        assert fake.json_calls == [] and target.read_bytes() == b"keep" and not manifest.exists()
        assert not (out_dir / "DOGE-USDT-SWAP" / "candles_1m.parquet").exists()

    @pytest.mark.parametrize("argv", [
        ["--kind", "candles", "--inst-id", "X", "--funding", "DOGE-USDT-SWAP"],   # both forms
        ["--kind", "candles"],                                                   # half a form
        ["--inst-id", "X"],
        [],                                                                      # nothing to fetch
        ["--funding", "DOGE-USDT"],                                              # not a swap
        ["--candles", "X", "X"],                                                 # duplicate
    ])
    def test_usage_errors(self, monkeypatch, tmp_path, argv):
        fake = FakeOkx().install(monkeypatch)
        code, *_ = self.run(tmp_path, *argv)
        assert code == fv.EXIT_USAGE and fake.json_calls == []
