"""
scripts/maker_probe_days.py (§11, "Days") against a fake OKX archive listing.

No test reaches the network: an autouse fixture replaces the HTTP functions
of the fetch_v2 module the script asks OKX through (and urlopen itself) with
ones that fail, and each listing test installs a fake that answers as OKX was
seen to answer: begin and end read as dates in UTC+8 for module 1 (trades)
and in UTC for module 4 (order book), at most ten dates per request, newest
file first, each file's dateTs at its date's midnight in the module's zone.
The throttle runs on a fake clock, and git runs without the user's config.

``list`` is checked for the rule itself (a day qualifies only with all four
files, D+1 included at the population's end), for reading each module's
dates in its own zone, for stopping rather than guessing a date, and for
writing once. ``draw`` is checked for drawing exactly what §11's numpy call
draws, and only from a candidate list committed once and left unmodified,
under the pinned numpy.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from research.spec import REPO_ROOT, get_spec


def _load(name):
    found = importlib.util.spec_from_file_location(f"_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(found)
    found.loader.exec_module(mod)
    return mod


mp = _load("maker_probe_days")
fv = mp.fv

SPEC = get_spec()
INST, FAMILY = "DOGE-USDT-SWAP", "DOGE-USDT"
CANDIDATES = SPEC["maker_probe"]["candidates_file"]
UTC, UTC8 = timezone.utc, timezone(timedelta(hours=8))
NOW = int(datetime(2026, 10, 11, 12, tzinfo=UTC).timestamp() * 1000)
FROM, TO = date(2025, 10, 1), date(2025, 10, 12)


def days(lo, hi):
    return [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]


def midnight(d, tz):
    return int(datetime(d.year, d.month, d.day, tzinfo=tz).timestamp() * 1000)


def spec_with(lo=FROM, hi=TO, **probe):
    """Appendix A with a shorter population (and any other maker_probe values)."""
    s = copy.deepcopy(SPEC)
    s["maker_probe"]["population"].update({"from": lo.isoformat(), "to": hi.isoformat()})
    s["maker_probe"].update(probe)
    return s


class FakeClock:
    def __init__(self):
        self.t = 1_000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError(f"unexpected network access: {args}")

    clock = FakeClock()
    monkeypatch.setattr(fv, "http_get_json", no_network)
    monkeypatch.setattr(fv, "http_get_bytes", no_network)
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    monkeypatch.setattr(fv, "_now", clock.now)
    monkeypatch.setattr(fv, "_sleep", clock.sleep)
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


class Repo:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True)
        self.git("init", "-q", "-b", "main")
        self.write("README.md", "x\n")
        self.commit("start")

    def git(self, *args):
        proc = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def write(self, rel, text):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    return Repo(tmp_path / "repo")


# ---------------------------------------------------------------- fake OKX

def ok(data):
    return {"code": "0", "msg": "", "data": data}


def file_name(module, d):
    return (f"{INST}-trades-{d.isoformat()}.zip" if module == "1"
            else f"{INST}-L2orderbook-400lv-{d.isoformat()}.tar.gz")


def file_url(module, d, name=None):
    kind = "trades/daily" if module == "1" else "orderbook/L2/400lv/daily"
    return f"https://static.okx.com/cdn/okex/traderecords/{kind}/{d.strftime('%Y%m%d')}/{name or file_name(module, d)}"


class FakeArchive:
    """market-data-history as observed, plus /public/time."""

    def __init__(self, lo, hi):
        self.dates = {"1": set(days(lo, hi)), "4": set(days(lo, hi))}
        self.read_zone = {"1": UTC8, "4": UTC}        # how OKX reads begin/end
        self.stamp_zone = {"1": UTC8, "4": UTC}       # which midnight dateTs holds
        self.extra = []                               # (module, date, record, family)
        self.calls = []

    def install(self, monkeypatch):
        monkeypatch.setattr(fv, "http_get_json", self.get_json)
        return self

    def drop(self, module, d):
        self.dates[module].discard(d)

    def record(self, module, d):
        return {"dateTs": str(midnight(d, self.stamp_zone[module])), "filename": file_name(module, d),
                "sizeMB": "12.5", "url": file_url(module, d)}

    def get_json(self, url):
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == fv.TIME_PATH:
            return ok([{"ts": str(NOW)}])
        assert u.path == fv.ARCHIVE_PATH, u.path
        self.calls.append(q)
        assert (q["instType"], q["dateAggrType"]) == ("SWAP", "daily") and "instIdList" not in q
        module, zone = q["module"], self.read_zone[q["module"]]
        first = datetime.fromtimestamp(int(q["begin"]) / 1000, zone).date()
        last = datetime.fromtimestamp(int(q["end"]) / 1000, zone).date()
        if last < first or (last - first).days + 1 > 10:
            return {"code": "51000", "msg": "Parameter end error", "data": []}
        groups = {}
        if q["instFamilyList"] == FAMILY:
            groups[FAMILY] = [self.record(module, d) for d in sorted(self.dates[module], reverse=True)
                              if first <= d <= last]
        for m, d, rec, fam in self.extra:
            if m == module and first <= d <= last:
                groups.setdefault(fam, []).append(rec)
        details = [{"ccy": "", "dateRangeStart": str(midnight(first, zone)), "dateRangeEnd": str(midnight(last, zone)),
                    "groupDetails": g, "groupSizeMB": "1.0", "instFamily": fam, "instId": "", "instType": "SWAP"}
                   for fam, g in groups.items() if g]
        return ok([{"dateAggrType": "daily", "details": details, "totalSizeMB": "1.0", "ts": str(NOW)}])


def run_list(repo, spec):
    return mp.main(["list"], root=repo.path, spec=spec)


def manifest(repo):
    return json.loads((repo.path / CANDIDATES).read_text())


def expected_candidates(fake, population):
    """The §11 rule, stated independently of the script."""
    out = []
    for d in population:
        nxt = d + timedelta(days=1)
        if {d, nxt} <= fake.dates["4"] and {d, nxt} <= fake.dates["1"]:
            out.append(d.isoformat())
    return out


def request_dates(q, zone):
    return (datetime.fromtimestamp(int(q["begin"]) / 1000, zone).date(),
            datetime.fromtimestamp(int(q["end"]) / 1000, zone).date())


# ---------------------------------------------------------------- list

class TestList:
    def test_every_day_with_its_four_files_qualifies(self, monkeypatch, repo, messages):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        assert run_list(repo, spec_with()) == mp.EXIT_OK, messages
        m = manifest(repo)
        assert m["candidates"] == [d.isoformat() for d in days(FROM, TO)]
        assert m["n_days"] == m["n_candidates"] == 12
        assert (m["inst_id"], m["inst_family"]) == (INST, FAMILY)
        assert m["server_time_ms"] == NOW
        day = m["days"]["2025-10-05"]
        assert day == {
            "qualifies": True, "missing": [],
            "orderbook_D": [file_name("4", date(2025, 10, 5))], "orderbook_D1": [file_name("4", date(2025, 10, 6))],
            "trades_D": [file_name("1", date(2025, 10, 5))], "trades_D1": [file_name("1", date(2025, 10, 6))],
        }
        assert m["files"]["trades"]["2025-10-05"][0]["url"] == file_url("1", date(2025, 10, 5))
        # Requests: both modules, 13 dates each in chunks of at most 7, family not instId.
        assert sorted({q["module"] for q in fake.calls}) == ["1", "4"]
        assert len(fake.calls) == len(m["requests"]) == 4
        for q in fake.calls:
            lo, hi = request_dates(q, UTC)
            assert (hi - lo).days + 1 <= mp.DAILY_CHUNK_DAYS
            assert q["instFamilyList"] == FAMILY
        # Metadata only: sizes are not kept, and no file was fetched (http_get_bytes would have raised).
        text = (repo.path / CANDIDATES).read_text()
        assert "sizeMB" not in text and "12.5" not in text

    @pytest.mark.parametrize("req", list(mp.REQUIREMENTS))
    def test_a_day_lacking_one_of_its_four_files_does_not_qualify(self, monkeypatch, repo, req):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        d0 = date(2025, 10, 5)
        kind, offset = mp.REQUIREMENTS[req]
        fake.drop(mp.MODULES[kind][0], d0 + timedelta(days=offset))
        assert run_list(repo, spec_with()) == mp.EXIT_OK
        m = manifest(repo)
        assert m["days"][d0.isoformat()]["missing"] == [req]
        assert m["days"][d0.isoformat()]["qualifies"] is False and m["days"][d0.isoformat()][req] == []
        assert d0.isoformat() not in m["candidates"]
        # The same missing file is the other requirement of the neighbouring day, and nothing else changes.
        assert m["candidates"] == expected_candidates(fake, days(FROM, TO))
        assert len(m["candidates"]) == 10

    @pytest.mark.parametrize("module,req", [(None, None), ("4", "orderbook_D1"), ("1", "trades_D1")])
    def test_the_last_day_needs_the_files_of_the_day_after(self, monkeypatch, repo, module, req):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        if module:
            fake.drop(module, TO + timedelta(days=1))
        assert run_list(repo, spec_with()) == mp.EXIT_OK
        m = manifest(repo)
        assert (TO.isoformat() in m["candidates"]) is (module is None)
        assert m["days"][TO.isoformat()]["missing"] == ([req] if req else [])
        assert TO.isoformat() == max(m["days"])                # D+1 is asked about, not judged
        # Each module is asked about the dates up to D+1 of the last day, no further.
        for mod, zone in (("1", UTC8), ("4", UTC)):
            ends = [request_dates(q, zone)[1] for q in fake.calls if q["module"] == mod]
            begins = [request_dates(q, zone)[0] for q in fake.calls if q["module"] == mod]
            assert max(ends) == TO + timedelta(days=1) and min(begins) == FROM

    def test_each_module_is_read_in_its_own_zone(self, monkeypatch, repo):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        assert run_list(repo, spec_with()) == mp.EXIT_OK
        m = manifest(repo)
        d = date(2025, 10, 5)
        trade = m["files"]["trades"][d.isoformat()][0]
        book = m["files"]["orderbook"][d.isoformat()][0]
        # The trade file dated 10-05 is stamped 10-04 16:00 UTC (10-05 00:00 UTC+8) and covers
        # 10-04 16:00 -> 10-05 16:00 UTC; its date comes from its name, not from reading dateTs in UTC.
        assert trade["filename"] == file_name("1", d)
        assert trade["dateTs"] == midnight(d, UTC8) == midnight(d, UTC) - 8 * 3_600_000
        assert trade["covers_utc"] == [midnight(d, UTC8), midnight(d + timedelta(days=1), UTC8)]
        assert book["dateTs"] == midnight(d, UTC)
        assert book["covers_utc"] == [midnight(d, UTC), midnight(d + timedelta(days=1), UTC)]
        assert m["modules"] == {"orderbook": {"module": "4", "file_dates_in": "UTC"},
                                "trades": {"module": "1", "file_dates_in": "UTC+8"}}
        # begin and end name the same date under either zone (04:00 UTC = 12:00 UTC+8).
        for q in fake.calls:
            assert request_dates(q, UTC) == request_dates(q, UTC8)

    def test_either_reading_of_begin_and_end_lists_the_same_days(self, monkeypatch, repo, tmp_path):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        fake.drop("1", date(2025, 10, 8))
        assert run_list(repo, spec_with()) == mp.EXIT_OK
        first = manifest(repo)["candidates"]
        other = Repo(tmp_path / "other")
        swapped = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        swapped.drop("1", date(2025, 10, 8))
        swapped.read_zone = {"1": UTC, "4": UTC8}
        assert run_list(other, spec_with()) == mp.EXIT_OK
        assert manifest(other)["candidates"] == first == expected_candidates(fake, days(FROM, TO))

    @pytest.mark.parametrize("module", ["1", "4"])
    def test_a_datets_in_the_other_zone_stops_the_run(self, monkeypatch, repo, messages, module):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        fake.stamp_zone[module] = UTC if module == "1" else UTC8
        assert run_list(repo, spec_with()) == mp.EXIT_FETCH_FAILED
        assert not (repo.path / CANDIDATES).exists()
        assert "dateTs" in "\n".join(messages)

    def test_a_file_dated_outside_the_request_stops_the_run(self, monkeypatch, repo, messages):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        stray = date(2025, 11, 30)
        fake.extra.append(("4", FROM, {"dateTs": str(midnight(stray, UTC)), "filename": file_name("4", stray),
                                       "url": file_url("4", stray)}, FAMILY))
        assert run_list(repo, spec_with()) == mp.EXIT_FETCH_FAILED
        assert not (repo.path / CANDIDATES).exists()
        assert "for a request covering" in "\n".join(messages)

    def test_a_name_without_exactly_one_date_stops_the_run(self, monkeypatch, repo, messages):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        name = f"{INST}-trades-latest.zip"
        fake.extra.append(("1", FROM, {"filename": name, "url": file_url("1", FROM, name)}, FAMILY))
        assert run_list(repo, spec_with()) == mp.EXIT_FETCH_FAILED
        assert "cannot tell which date" in "\n".join(messages)

    def test_other_instruments_are_skipped_and_extra_parts_kept(self, monkeypatch, repo):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        d = date(2025, 10, 3)
        fake.drop("1", d)
        # Another swap of the family, and a file of this name under another family: neither counts.
        other = f"DOGE-USDC-SWAP-trades-{d}.zip"
        fake.extra.append(("1", d, {"dateTs": str(midnight(d, UTC8)), "filename": other,
                                    "url": file_url("1", d, other)}, FAMILY))
        fake.extra.append(("1", d, fake.record("1", d), "DOGE-USD"))
        # A second file for one order-book date is recorded beside the first.
        part = f"{INST}-L2orderbook-400lv-{date(2025, 10, 6)}-part2.tar.gz"
        fake.extra.append(("4", date(2025, 10, 6), {"dateTs": str(midnight(date(2025, 10, 6), UTC)),
                                                    "filename": part, "url": file_url("4", date(2025, 10, 6), part)},
                           FAMILY))
        assert run_list(repo, spec_with()) == mp.EXIT_OK
        m = manifest(repo)
        assert m["days"][d.isoformat()]["missing"] == ["trades_D"]
        assert sum(r["other_files_skipped"] for r in m["requests"]) == 2
        assert m["days"]["2025-10-06"]["orderbook_D"] == [file_name("4", date(2025, 10, 6)), part]
        assert "2025-10-06" in m["candidates"]

    def test_the_real_population_and_its_edge_files(self, monkeypatch, repo):
        pop = mp.population(SPEC)
        assert (pop[0], pop[-1]) == (date(2025, 9, 24), date(2026, 9, 22))
        fake = FakeArchive(pop[0], pop[-1] + timedelta(days=1)).install(monkeypatch)
        assert run_list(repo, SPEC) == mp.EXIT_OK
        m = manifest(repo)
        assert len(m["candidates"]) == 364 and m["candidates"][-1] == "2026-09-22"
        assert m["population"]["from"] == "2025-09-24"
        assert all((request_dates(q, UTC)[1] - request_dates(q, UTC)[0]).days < mp.DAILY_CHUNK_DAYS
                   for q in fake.calls)
        # Two files reach outside the in-sample year, into doge_2024 and doge_fwd.
        assert m["files_overlapping_protected"] == sorted([file_name("4", date(2026, 9, 23)),
                                                           file_name("1", date(2025, 9, 24))])

    def test_a_listing_that_keeps_failing_writes_nothing(self, monkeypatch, repo, messages):
        FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        monkeypatch.setattr(mp, "DAILY_CHUNK_DAYS", 11)       # over OKX's cap of 10: every request fails
        assert run_list(repo, spec_with()) == mp.EXIT_FETCH_FAILED
        assert not (repo.path / CANDIDATES).exists()
        assert "Giving up" in "\n".join(messages)


class TestListWritesOnce:
    def test_an_existing_file_is_refused_before_any_request(self, monkeypatch, repo):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        repo.write(CANDIDATES, "{}\n")
        assert run_list(repo, spec_with()) == mp.EXIT_EXISTS
        assert fake.calls == [] and (repo.path / CANDIDATES).read_text() == "{}\n"

    def test_a_second_run_is_refused(self, monkeypatch, repo):
        FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        assert run_list(repo, spec_with()) == mp.EXIT_OK
        before = (repo.path / CANDIDATES).read_bytes()
        assert run_list(repo, spec_with()) == mp.EXIT_EXISTS
        assert (repo.path / CANDIDATES).read_bytes() == before

    def test_a_list_committed_and_removed_still_counts(self, monkeypatch, repo, messages):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        repo.write(CANDIDATES, "{}\n")
        repo.commit("candidates")
        repo.git("rm", "-q", CANDIDATES)
        repo.commit("remove them")
        assert run_list(repo, spec_with()) == mp.EXIT_EXISTS
        assert fake.calls == [] and "was committed before" in "\n".join(messages)

    def test_foreign_code_is_refused(self, monkeypatch, repo):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        monkeypatch.setattr(mp.integrity, "foreign_code_problem", lambda root: "research from elsewhere")
        assert run_list(repo, spec_with()) == mp.EXIT_REFUSED
        assert fake.calls == []


# ---------------------------------------------------------------- draw

def candidate_manifest(cands, extra_days=()):
    table = {d: {"qualifies": True, "missing": []} for d in cands}
    table.update({d: {"qualifies": False, "missing": ["trades_D"]} for d in extra_days})
    return json.dumps({"manifest_version": 1, "inst_id": INST, "candidates": cands,
                       "days": dict(sorted(table.items()))}, indent=2) + "\n"


def numpy_draw(cands, n=14, seed=20261008):
    return [str(x) for x in np.random.default_rng(seed).choice(cands, size=min(n, len(cands)), replace=False)]


ALL_DAYS = [d.isoformat() for d in mp.population(SPEC)]
SOME = [d for i, d in enumerate(ALL_DAYS) if i % 3]          # 242 candidates, a few days short


def committed(repo, cands, **kw):
    repo.write(CANDIDATES, candidate_manifest(cands, **kw))
    repo.commit("candidate days")
    return repo


def draw(repo, spec=SPEC):
    return mp.main(["draw"], root=repo.path, spec=spec)


class TestDraw:
    def test_draw_is_the_numpy_call_of_section_11(self, repo, messages):
        committed(repo, SOME, extra_days=[d for i, d in enumerate(ALL_DAYS) if not i % 3])
        assert draw(repo) == mp.EXIT_OK, messages
        want = numpy_draw(SOME)
        got = mp.committed_draw(repo.path, SPEC)
        assert got.candidates == SOME and got.order == want and got.days == sorted(want) and len(want) == 14
        assert mp.drawn_days(SOME, SPEC) == sorted(want)
        text = "\n".join(messages)
        assert " ".join(want) in text and " ".join(sorted(want)) in text

    def test_fewer_than_n_days_uses_them_all(self, repo):
        few = SOME[:5]
        committed(repo, few)
        got = mp.committed_draw(repo.path, SPEC)
        assert got.days == few and sorted(got.order) == few and got.order == numpy_draw(few)

    def test_list_then_draw(self, monkeypatch, repo):
        fake = FakeArchive(FROM, TO + timedelta(days=1)).install(monkeypatch)
        fake.drop("4", date(2025, 10, 9))
        spec = spec_with(n_days=4)
        assert run_list(repo, spec) == mp.EXIT_OK
        repo.commit("candidate days")
        cands = manifest(repo)["candidates"]
        assert len(cands) == 10
        assert mp.committed_draw(repo.path, spec).order == numpy_draw(cands, n=4)

    def test_an_untracked_list_is_refused(self, repo, messages):
        repo.write(CANDIDATES, candidate_manifest(SOME))
        assert draw(repo) == mp.EXIT_REFUSED
        assert "is not committed" in "\n".join(messages)

    @pytest.mark.parametrize("how", ["modified", "staged", "deleted"])
    def test_a_list_with_uncommitted_changes_is_refused(self, repo, messages, how):
        committed(repo, SOME)
        if how == "deleted":
            (repo.path / CANDIDATES).unlink()
        else:
            repo.write(CANDIDATES, candidate_manifest(SOME[1:]))
            if how == "staged":
                repo.git("add", CANDIDATES)
        assert draw(repo) == mp.EXIT_REFUSED
        assert "has uncommitted changes" in "\n".join(messages)

    def test_a_list_changed_after_its_first_commit_is_refused(self, repo, messages):
        committed(repo, SOME)
        committed(repo, SOME[1:])
        assert draw(repo) == mp.EXIT_REFUSED
        assert "changed by 2 commits" in "\n".join(messages)

    def test_the_committed_list_is_read_not_the_working_tree(self, repo):
        committed(repo, SOME)
        repo.git("update-index", "--skip-worktree", CANDIDATES)
        (repo.path / CANDIDATES).write_text(candidate_manifest(SOME[::2]))
        assert mp.committed_draw(repo.path, SPEC).candidates == SOME

    def test_another_numpy_is_refused(self, repo, messages, monkeypatch):
        committed(repo, SOME)
        monkeypatch.setattr(np, "__version__", "1.26.4")
        assert draw(repo) == mp.EXIT_REFUSED
        assert "numpy is 1.26.4, but §11 draws under numpy 1.24.3" in "\n".join(messages)
        with pytest.raises(mp.Refusal, match="numpy"):
            mp.drawn_days(SOME, SPEC)

    @pytest.mark.parametrize("cands,extra,needle", [
        (SOME[::-1], (), "not sorted ascending"),
        (SOME + [SOME[-1]], (), "not sorted ascending"),
        (["2025-09-23"] + SOME, (), "outside the population"),
        (["2025-9-30"], (), "not a YYYY-MM-DD date"),
        ([], (), "no candidate days"),
    ])
    def test_a_malformed_list_is_refused(self, repo, messages, cands, extra, needle):
        committed(repo, cands, extra_days=extra)
        assert draw(repo) == mp.EXIT_REFUSED
        assert needle in "\n".join(messages)

    def test_a_list_disagreeing_with_its_own_table_is_refused(self, repo, messages):
        text = json.loads(candidate_manifest(SOME))
        text["days"][SOME[0]]["qualifies"] = False
        repo.write(CANDIDATES, json.dumps(text))
        repo.commit("candidate days")
        assert draw(repo) == mp.EXIT_REFUSED
        assert "differs from the days the file's own table says qualify" in "\n".join(messages)

    def test_drawn_order_refuses_an_unsorted_list(self):
        with pytest.raises(mp.Refusal, match="sorted"):
            mp.drawn_order(SOME[::-1], SPEC)
