"""
scripts/fetch_history.py, the v1 fetcher, is locked for test data.

It always fetches [now - days, now) into data/history/<instId>/, so run as
README and docs/factor-research.md used to show, it would download data from
the pre-registered test windows and overwrite the in-sample files whose
SHA-256 Appendix A pins. These tests check that it refuses before the first
request, and that the requests it does send (with a fake client) never reach
below the window, whatever holes the data has.

No test here reaches the network: an autouse fixture replaces
aiohttp.ClientSession with one that fails on construction, and a test that
needs pages served installs a fake client explicitly. None touches the real
data/ either: the same fixture points the history directory and the
in-sample directory into tmp_path, so even a regressed guard writes there.
"""
import asyncio
import importlib.util
from datetime import datetime, timezone

import polars as pl
import pytest
from loguru import logger

from research import spec

_SPEC = importlib.util.spec_from_file_location("fetch_history", spec.REPO_ROOT / "scripts" / "fetch_history.py")
fh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fh)

MIN = 60_000
H8 = 8 * 3_600_000
DAY = 86_400_000
INST = "XYZ-USDT-SWAP"           # a made-up instrument: nothing here is real market data


def utc_ms(text):
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


FWD_CLOSE = spec.get_spec()["sets"]["doge_fwd"]["end_ms"]          # 2026-11-23 20:00 UTC
# The real locations, read before any test points them elsewhere; only
# compared as paths, never written to.
REAL_HISTORY_DIR, REAL_INSAMPLE_DIR = fh.HISTORY_DIR, fh.insample_dir()


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    class NoSession:
        def __init__(self, *args, **kwargs):
            raise AssertionError("unexpected network access")

    monkeypatch.setattr(fh.aiohttp, "ClientSession", NoSession)
    # The real layout, moved into tmp_path: <history>/DOGE-USDT-SWAP is the
    # in-sample directory, as data/history/DOGE-USDT-SWAP is Appendix A's.
    monkeypatch.setattr(fh, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(fh, "insample_dir", lambda: (tmp_path / "DOGE-USDT-SWAP").resolve())


@pytest.fixture
def messages():
    out = []
    hid = logger.add(lambda m: out.append(m.record["message"]), level="INFO")
    yield out
    logger.remove(hid)


def run_at(monkeypatch, now_ms, *argv):
    monkeypatch.setattr(fh, "_now_ms", lambda: now_ms)
    return asyncio.run(fh.run(fh.parse_args(list(argv))))


# ---------------------------------------------------------------- a fake OKX

def candle_row(ts):
    p = 0.1 + (ts // MIN % 97) * 1e-4
    return [str(ts), f"{p:.6f}", f"{p * 1.001:.6f}", f"{p * 0.999:.6f}", f"{p:.6f}", "10", "1.5", "0", "1"]


class FakeClient:
    """Just enough of OKX's paging (exclusive ``after`` and ``before``, ``limit``) to watch the requests."""

    def __init__(self):
        self.candles = {}        # instId -> {ts: row}
        self.funding = {}        # instId -> {ts: rec}
        self.oi = []             # rows the OI endpoint returns, whatever is asked
        self.calls = []          # (path, params)
        self.served = []         # every candle / funding ts handed out
        self.request_count = 0

    def _page(self, store, q):
        after, limit = int(q["after"]), int(q["limit"])
        before = int(q["before"]) if "before" in q else None
        keys = sorted((t for t in store if t < after and (before is None or t > before)), reverse=True)[:limit]
        self.served.extend(keys)
        return [store[t] for t in keys]

    async def get(self, path, params):
        self.calls.append((path, dict(params)))
        self.request_count += 1
        if path.endswith("/history-candles"):
            return self._page(self.candles.get(params["instId"], {}), params)
        if path.endswith("/funding-rate-history"):
            return self._page(self.funding.get(params["instId"], {}), params)
        if path.endswith("/open-interest-volume"):
            return self.oi
        raise AssertionError(f"unexpected path {path}")


class DummySession:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def install(monkeypatch, fake):
    monkeypatch.setattr(fh.aiohttp, "ClientSession", DummySession)
    monkeypatch.setattr(fh, "OKXHistoryClient", lambda session: fake)
    return fake


# ---------------------------------------------------------------- refusals

class TestRefusesBeforeAnyRequest:
    def test_the_documented_invocation_is_refused_today(self, monkeypatch, messages):
        # `--days 365 --with-spot` on 2026-10-11 reaches into doge_fwd's open
        # window and would write over the in-sample files.
        code = run_at(monkeypatch, utc_ms("2026-10-11 12:00"), "--days", "365", "--with-spot")
        assert code == fh.EXIT_PROTECTED
        text = "\n".join(messages)
        assert "test window" in text and "scripts/fetch_v2.py" in text

    @pytest.mark.parametrize("now,days", [
        ("2027-03-01 00:00", 750),          # reaches back through doge_fwd into doge_2024
        ("2026-11-24 00:00", 1),            # [11-23 00:00, 11-24 00:00) holds doge_fwd's last 20 h
        ("2025-01-01 00:00", 30),           # inside doge_2024
        ("2026-09-24 00:00", 2),            # straddles the end of IS and the start of doge_fwd
    ])
    def test_any_window_touching_a_protected_range(self, monkeypatch, tmp_path, now, days):
        code = run_at(monkeypatch, utc_ms(now), "--inst-id", INST, "--days", str(days))
        assert code == fh.EXIT_PROTECTED
        assert list(tmp_path.iterdir()) == []

    def test_the_insample_directory_is_never_a_target(self, monkeypatch, messages):
        # A window clear of every protected range, but the default instrument
        # writes into Appendix A's insample_dir (data/history/DOGE-USDT-SWAP;
        # the sandbox's copy of that layout here).
        assert fh.is_inside(REAL_HISTORY_DIR / "DOGE-USDT-SWAP" / "candles_1m.parquet", REAL_INSAMPLE_DIR)
        now = FWD_CLOSE + 30 * DAY
        assert not spec.overlaps_protected(now - 7 * DAY, now)
        code = run_at(monkeypatch, now, "--days", "7", "--with-spot")
        assert code == fh.EXIT_PROTECTED
        assert "insample_dir" in "\n".join(messages)

    @pytest.mark.parametrize("link", [True, False])
    def test_no_path_spelling_reaches_the_insample_dir(self, monkeypatch, tmp_path, link):
        # A symlink to the in-sample directory, or a '..' through a made-up
        # one, with a bar whose file does not exist there yet (so the
        # write-once check cannot catch it): only resolving the path can.
        ins = fh.insample_dir()
        ins.mkdir()
        if link:
            (tmp_path / "DOGE-LINK").symlink_to(ins, target_is_directory=True)
        fake = install(monkeypatch, FakeClient())
        inst = "DOGE-LINK" if link else "X/../DOGE-USDT-SWAP"
        code = run_at(monkeypatch, utc_ms("2026-12-10 00:00"), "--inst-id", inst, "--bar", "5m", "--days", "3",
                      "--skip-oi")
        assert code == fh.EXIT_PROTECTED
        assert fake.calls == [] and list(ins.iterdir()) == []

    @pytest.mark.parametrize("name", ["candles_1m.parquet", "spot_candles_1m.parquet",
                                      "funding_rates.parquet", "open_interest.parquet"])
    def test_an_existing_output_is_refused(self, monkeypatch, tmp_path, name):
        target = tmp_path / INST / name
        target.parent.mkdir(parents=True)
        target.write_bytes(b"keep")
        code = run_at(monkeypatch, FWD_CLOSE + 30 * DAY, "--inst-id", INST, "--days", "2", "--with-spot")
        assert code == fh.EXIT_EXISTS
        assert target.read_bytes() == b"keep"
        assert sorted(p.name for p in target.parent.iterdir()) == [name]

    @pytest.mark.parametrize("argv", [["--bar", "1M"], ["--bar", "fortnight"], ["--days", "0"]])
    def test_usage_errors(self, monkeypatch, tmp_path, argv):
        code = run_at(monkeypatch, FWD_CLOSE + 30 * DAY, "--inst-id", INST, *argv)
        assert code == fh.EXIT_USAGE


def test_write_parquet_never_overwrites_and_never_touches_the_insample_dir(tmp_path):
    df = pl.DataFrame({"timestamp": [1, 2], "x": [1.0, 2.0]})
    target = fh.insample_dir() / "never_written.parquet"
    with pytest.raises(PermissionError):
        fh.write_parquet(df, target, "x")
    assert not target.exists()
    path = tmp_path / "a" / "x.parquet"
    fh.write_parquet(df, path, "x")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        fh.write_parquet(df.head(1), path, "x")
    assert path.read_bytes() == before


# ---------------------------------------------------------------- the requests it does send

class TestBoundedRequests:
    def test_hole_at_the_border_serves_nothing_from_before_it(self):
        # A window opening exactly where doge_fwd closes, whose first 150
        # minutes are missing: a bare "100 bars older than the cursor" page
        # would reach back across the border.
        start, end = FWD_CLOSE, FWD_CLOSE + 600 * MIN
        fake = FakeClient()
        hole = set(range(start, start + 150 * MIN, MIN))
        fake.candles["X"] = {t: candle_row(t) for t in range(start - 300 * MIN, end + 60 * MIN, MIN) if t not in hole}
        df = asyncio.run(fh.fetch_candles(fake, "X", "1m", start, end))
        assert df["timestamp"].to_list() == [t for t in range(start + 150 * MIN, end, MIN)]
        assert min(fake.served) >= start
        assert not any(spec.overlaps_protected(t, t + 1) for t in fake.served)
        for _, q in fake.calls:
            after, before, limit = int(q["after"]), int(q["before"]), int(q["limit"])
            assert limit <= fh.PAGE_LIMIT and before + 1 == max(start, after - limit * MIN)

    def test_funding_hole_at_the_border(self):
        start, end = FWD_CLOSE, FWD_CLOSE + 60 * DAY
        fake = FakeClient()
        first = -(-(start - 30 * DAY) // H8) * H8
        fake.funding["X"] = {t: {"fundingTime": str(t), "fundingRate": "0.0001", "realizedRate": ""}
                             for t in range(first, end, H8) if not start <= t < start + 10 * DAY}
        df = asyncio.run(fh.fetch_funding_rates(fake, "X", start, end))
        assert df["timestamp"].min() >= start + 10 * DAY
        assert min(fake.served) >= start

    def test_allowed_window_end_to_end_and_written_once(self, monkeypatch, tmp_path):
        now = utc_ms("2027-01-10 00:00")
        start = now - 2 * DAY
        fake = install(monkeypatch, FakeClient())
        for inst in (INST, "XYZ-USDT"):
            fake.candles[inst] = {t: candle_row(t) for t in range(start - 500 * MIN, now, MIN)
                                  if not start <= t < start + 200 * MIN}       # hole at the start
        fake.funding[INST] = {t: {"fundingTime": str(t), "fundingRate": "0.0001", "realizedRate": "0.0001"}
                              for t in range(-(-(start - 10 * DAY) // H8) * H8, now, H8)}
        fake.oi = [[str(t), "1000", "10"] for t in range(start - DAY, now, 5 * MIN)]   # ignores begin/end
        argv = ("--inst-id", INST, "--days", "2", "--with-spot")
        code = run_at(monkeypatch, now, *argv)
        assert code == fh.EXIT_OK
        assert min(fake.served) >= start
        oi_q = [q for p, q in fake.calls if p.endswith("/open-interest-volume")]
        assert oi_q == [{"ccy": "XYZ", "period": "5m", "begin": str(start), "end": str(now - 1)}]
        out = tmp_path / INST
        assert sorted(p.name for p in out.iterdir()) == [
            "candles_1m.parquet", "funding_rates.parquet", "open_interest.parquet", "spot_candles_1m.parquet"]
        assert pl.read_parquet(out / "candles_1m.parquet")["timestamp"].min() == start + 200 * MIN
        assert pl.read_parquet(out / "open_interest.parquet")["timestamp"].min() >= start
        # A second run is refused before any request.
        n = len(fake.calls)
        assert run_at(monkeypatch, now, *argv) == fh.EXIT_EXISTS
        assert len(fake.calls) == n
