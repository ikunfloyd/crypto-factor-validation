"""
scripts/fetch_maker_probe.py (§11, the probe's archive files) without the network.

The candidate list is a small fake in the shape scripts/maker_probe_days.py
writes; downloads go through a fake opener; git runs in a throwaway
repository without the user's config. Checked: which files the days need,
each once; the refusals before any request (doge_fwd not run or not pushed,
days that are not §11's draw, a file the list does not name once or flags as
reaching into a test window); that a file is read to its end before it is
kept, retried when it is not, never overwritten, and resumed when present.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import subprocess
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from research.spec import REPO_ROOT


def _load(name):
    found = importlib.util.spec_from_file_location(f"_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(found)
    found.loader.exec_module(mod)
    return mod


fmp = _load("fetch_maker_probe")


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

def _entry(kind, d, flagged=False):
    name = (f"DOGE-USDT-SWAP-L2orderbook-400lv-{d}.tar.gz" if kind == "orderbook"
            else f"DOGE-USDT-SWAP-trades-{d}.zip")
    return {"filename": name, "url": f"https://example.invalid/{name}", "dateTs": 0,
            "covers_utc": [0, 1], "overlaps_protected": flagged}


def candidate_doc(days, extra_dates=(), flagged=()):
    """A candidate list in which ``days`` qualify, with one file per date and kind."""
    from datetime import date, timedelta
    dates = sorted({d for x in days for d in (x, (date.fromisoformat(x) + timedelta(days=1)).isoformat())}
                   | set(extra_dates))
    files = {k: {d: [_entry(k, d, d in flagged)] for d in dates} for k in ("orderbook", "trades")}
    rows = {}
    for x in days:
        nxt = (date.fromisoformat(x) + timedelta(days=1)).isoformat()
        rows[x] = {"qualifies": True, "missing": [],
                   "orderbook_D": [files["orderbook"][x][0]["filename"]],
                   "orderbook_D1": [files["orderbook"][nxt][0]["filename"]],
                   "trades_D": [files["trades"][x][0]["filename"]],
                   "trades_D1": [files["trades"][nxt][0]["filename"]]}
    return {"inst_id": "DOGE-USDT-SWAP", "candidates": sorted(days), "days": rows, "files": files,
            "files_overlapping_protected": [files[k][d][0]["filename"] for k in files for d in flagged]}


def zip_bytes(payload=b"trade_id,price\n1,0.2\n" * 50):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("DOGE-USDT-SWAP-trades.csv", payload)
    return buf.getvalue()


def targz_bytes(payload=b'{"action":"snapshot"}\n' * 200):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("DOGE-USDT-SWAP-L2orderbook-400lv.data")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class FakeResponse:
    def __init__(self, body, length=None):
        self._body = io.BytesIO(body)
        n = len(body) if length is None else length
        self.headers = {"Content-Length": str(n), "Last-Modified": "Mon, 01 Jan 2026 00:00:00 GMT",
                        "ETag": '"abc"'}

    def read(self, n=-1):
        return self._body.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Serves ``bodies[url]``, one per call in turn (the last repeats); records each (url, method)."""

    def __init__(self, bodies):
        self.bodies = {u: list(b) for u, b in bodies.items()}
        self.calls = []

    def __call__(self, url, method="GET"):
        self.calls.append((url, method))
        queue = self.bodies[url]
        body = queue.pop(0) if len(queue) > 1 else queue[0]
        if method == "HEAD":
            return FakeResponse(b"", length=len(body))
        return FakeResponse(body)


def archive(kind="trades", d="2025-10-19"):
    e = _entry(kind, d)
    return fmp.ArchiveFile(kind, d, e["filename"], e["url"], [0, 1])


# --------------------------------------------------------------------------
# which files
# --------------------------------------------------------------------------

def test_day_files_lists_each_needed_file_once():
    doc = candidate_doc(["2025-10-19", "2025-10-20"], extra_dates=["2025-12-01"])
    files = fmp.day_files(doc, ["2025-10-19", "2025-10-20"])
    names = [f.filename for f in files]
    assert len(names) == len(set(names)) == 6           # D, D+1, D+2 for each kind; 10-20 is shared
    assert {f.file_date for f in files} == {"2025-10-19", "2025-10-20", "2025-10-21"}
    assert {f.kind for f in files} == {"orderbook", "trades"}
    assert all(f.url.endswith(f.filename) for f in files)


@pytest.mark.parametrize("break_it, why", [
    (lambda doc: doc["days"]["2025-10-19"].update(qualifies=False), "not a qualifying day"),
    (lambda doc: doc["days"]["2025-10-19"]["trades_D1"].append("other.zip"), "names 2 files"),
    (lambda doc: doc["files"]["trades"].pop("2025-10-20"), "listed 0 times"),
    (lambda doc: doc["files"]["orderbook"]["2025-10-19"][0].update(overlaps_protected=True), "test window"),
    (lambda doc: doc["files_overlapping_protected"].append("DOGE-USDT-SWAP-trades-2025-10-20.zip"), "test window"),
])
def test_day_files_refuses(break_it, why):
    doc = candidate_doc(["2025-10-19"])
    break_it(doc)
    with pytest.raises(fmp.Refusal, match=why):
        fmp.day_files(doc, ["2025-10-19"])


def test_probe_days_must_be_the_draw(monkeypatch):
    text = "```maker-probe-days\n2025-10-19\n2025-11-02\n```\n"
    monkeypatch.setattr(fmp.gitrepo, "show_file", lambda root, rev, path: text)
    spec = {"freeze": {"amendment_file": "docs/preregistration-freeze.md"}}
    monkeypatch.setattr(fmp.mpd, "committed_draw", lambda root, s: SimpleNamespace(days=["2025-10-19", "2025-11-02"]))
    assert fmp.probe_days(REPO_ROOT, spec) == ["2025-10-19", "2025-11-02"]
    monkeypatch.setattr(fmp.mpd, "committed_draw", lambda root, s: SimpleNamespace(days=["2025-10-19", "2025-11-03"]))
    with pytest.raises(fmp.Refusal, match="differ from §11's draw"):
        fmp.probe_days(REPO_ROOT, spec)


# --------------------------------------------------------------------------
# doge_fwd first (§12, 2026-10-12)
# --------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True,
                              text=True).stdout.strip()

    git("init", "-q", "-b", "main")
    (tmp_path / "README").write_text("x\n")
    git("add", "README")
    git("commit", "-q", "-m", "start")
    return SimpleNamespace(path=tmp_path, git=git)


def test_refuses_before_doge_fwd_is_committed(repo):
    with pytest.raises(fmp.Refusal, match="not committed"):
        fmp.doge_fwd_commit(repo.path)


def test_refuses_until_doge_fwd_is_pushed(repo):
    (repo.path / "results" / "oos").mkdir(parents=True)
    (repo.path / fmp.DOGE_FWD_RESULTS).write_text("cell\n")
    repo.git("update-ref", "refs/remotes/origin/main", "HEAD")      # the remote holds only the start
    repo.git("add", fmp.DOGE_FWD_RESULTS)
    repo.git("commit", "-q", "-m", "doge_fwd")
    with pytest.raises(fmp.Refusal, match="not on origin/main"):
        fmp.doge_fwd_commit(repo.path)
    repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
    assert fmp.doge_fwd_commit(repo.path) == repo.git("rev-parse", "HEAD")


# --------------------------------------------------------------------------
# reading archives to their end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("make, suffix", [(zip_bytes, ".zip"), (targz_bytes, ".tar.gz")])
def test_archive_members_reads_to_the_end(tmp_path, make, suffix):
    good = tmp_path / f"good{suffix}"
    good.write_bytes(make())
    (member,) = fmp.archive_members(good)
    assert member["bytes"] > 0
    cut = tmp_path / f"cut{suffix}"
    cut.write_bytes(make()[:-40])
    with pytest.raises(fmp.ARCHIVE_ERRORS):
        fmp.archive_members(cut)


def test_archive_members_refuses_other_files(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a\n")
    with pytest.raises(fmp.FetchError, match="not a .zip or .tar.gz"):
        fmp.archive_members(p)


# --------------------------------------------------------------------------
# downloading
# --------------------------------------------------------------------------

def test_download_hashes_and_keeps_a_complete_file(tmp_path):
    f, body = archive(), zip_bytes()
    entry = fmp.download(f, tmp_path, opener=FakeOpener({f.url: [body]}), sleep=lambda s: None)
    assert (tmp_path / f.filename).read_bytes() == body
    assert entry["sha256"] == hashlib.sha256(body).hexdigest()
    assert entry["bytes"] == len(body) and entry["resumed"] is False
    assert entry["etag"] == '"abc"' and entry["members"][0]["name"] == "DOGE-USDT-SWAP-trades.csv"
    assert list(tmp_path.iterdir()) == [tmp_path / f.filename]          # no .part left


def test_download_retries_a_truncated_file(tmp_path):
    f, body = archive("orderbook"), targz_bytes()
    waits = []
    opener = FakeOpener({f.url: [body[:-40], body]})
    entry = fmp.download(f, tmp_path, opener=opener, sleep=waits.append)
    assert len(waits) == 1 and entry["sha256"] == hashlib.sha256(body).hexdigest()


def test_download_gives_up_and_leaves_nothing(tmp_path):
    f = archive()
    with pytest.raises(fmp.FetchError, match="giving up"):
        fmp.download(f, tmp_path, opener=FakeOpener({f.url: [zip_bytes()[:-40]]}), sleep=lambda s: None)
    assert list(tmp_path.iterdir()) == []


def test_download_resumes_without_fetching_again(tmp_path):
    f, body = archive(), zip_bytes()
    (tmp_path / f.filename).write_bytes(body)
    opener = FakeOpener({f.url: [body]})
    entry = fmp.download(f, tmp_path, opener=opener, sleep=lambda s: None)
    assert opener.calls == [(f.url, "HEAD")] and entry["resumed"] is True
    assert entry["sha256"] == hashlib.sha256(body).hexdigest()


def test_download_refuses_a_kept_file_of_the_wrong_size(tmp_path):
    f, body = archive(), zip_bytes()
    (tmp_path / f.filename).write_bytes(body[:-40])
    with pytest.raises(fmp.FetchError, match="delete it and rerun"):
        fmp.download(f, tmp_path, opener=FakeOpener({f.url: [body]}), sleep=lambda s: None)
    assert (tmp_path / f.filename).read_bytes() == body[:-40]           # never overwritten
