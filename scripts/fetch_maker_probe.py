#!/usr/bin/env python3
"""
Download the maker probe's archive files (pre-registration §11).

§11 replays OKX's order book and trades on the 14 days that the freeze
amendment's ``maker-probe-days`` block records. Day D needs four files
(scripts/maker_probe_days.py):

    orderbook D, D+1   module 4, 400-level order book, dated in UTC
    trades    D, D+1   module 1, trades, dated in UTC+8

They are fetched from the URLs in the committed candidate list
(results/manifest_maker_probe.json) into data/history/maker_probe/, and
results/maker_probe/manifest_archives.json records each one: its URL, size,
SHA-256 and members, and the server's Last-Modified and ETag.

Before the first request:

  * §12 (2026-10-12) downloads the files only after the last test set has
    run, so results/oos/doge_fwd.csv must be committed, and the commit that
    added it must be on origin/main.
  * The days are the amendment's block as committed, and the block must
    equal §11's draw from the committed candidate list
    (maker_probe_days.committed_draw).
  * Each of the four files of every day must be listed exactly once in the
    candidate list, and none may be one that it flags as reaching into a
    test window.
  * The manifest must not exist yet.

A file already on disk is kept, so an interrupted run resumes. A new one is
written under a temporary name, read to its end (every gzip or zip CRC
checked) and only then given its name. The manifest is written once, after
every file is present and has been read to its end.

Once the manifest is committed, the script serves readers instead: it
fetches each file the committed manifest lists that is not on disk yet,
checks every file's SHA-256 against the manifest, and writes nothing else.
scripts/run_maker_probe.py --verify then recomputes the probe from them.

Usage:
    python scripts/fetch_maker_probe.py
"""
from __future__ import annotations

import argparse
import atexit
import http.client
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
# Startup, as in the other study scripts: this checkout's packages first on
# sys.path, compiled from source into a fresh bytecode cache.
sys.path[:] = [str(REPO_ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]
if __name__ == "__main__":
    sys.pycache_prefix = tempfile.mkdtemp(prefix=f"{Path(__file__).stem}-pycache-")
    atexit.register(shutil.rmtree, sys.pycache_prefix, True)

from loguru import logger

from research import freeze, gitrepo, integrity
from research.io import sha256_file, write_once_json
from research.spec import get_spec

mpd = integrity.load_script(REPO_ROOT, "maker_probe_days")
wfa = integrity.load_script(REPO_ROOT, "write_freeze_amendment")

EXIT_OK = 0
EXIT_FETCH_FAILED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_EXISTS = 4

DOGE_FWD_RESULTS = "results/oos/doge_fwd.csv"
ARCHIVE_DIR = "data/history/maker_probe"
MANIFEST = "results/maker_probe/manifest_archives.json"

HTTP_TIMEOUT_S = 60
MAX_RETRIES = 4
BACKOFF_S = 5.0
_CHUNK = 1 << 20
_HEADERS = {"User-Agent": "doge-usdt-trading/fetch_maker_probe", "Accept": "*/*"}


class Refusal(Exception):
    """A protocol check failed; nothing was downloaded."""

    def __init__(self, message: str, code: int = EXIT_REFUSED) -> None:
        super().__init__(message)
        self.code = code


class FetchError(Exception):
    """A download or integrity check kept failing."""


# What a short or corrupt download raises, here or while it is read to its end.
ARCHIVE_ERRORS = (OSError, ValueError, EOFError, zlib.error, zipfile.BadZipFile, tarfile.TarError,
                  http.client.HTTPException, FetchError)


class ArchiveFile(NamedTuple):
    kind: str                  # "orderbook" or "trades"
    file_date: str             # the date in its name, in its module's time zone
    filename: str
    url: str
    covers_utc: List[int]      # [lo, hi) ms, as the candidate list records it


# --------------------------------------------------------------------------
# what to download
# --------------------------------------------------------------------------

def doge_fwd_commit(root: Path) -> str:
    """The commit that added doge_fwd's results, which must be on origin/main (§12, 2026-10-12)."""
    if not gitrepo.is_tracked(root, DOGE_FWD_RESULTS):
        raise Refusal(f"{DOGE_FWD_RESULTS} is not committed; §12 (2026-10-12) downloads the probe's files only "
                      "after doge_fwd has run")
    added = gitrepo.commits_touching(root, DOGE_FWD_RESULTS, added_only=True)
    if len(added) != 1:
        raise Refusal(f"{DOGE_FWD_RESULTS} was added by {len(added)} commits, not one")
    remote = gitrepo.commit_of(root, freeze.REMOTE_MAIN)
    if remote is None or not gitrepo.is_ancestor(root, added[0], remote):
        raise Refusal(f"the commit that added {DOGE_FWD_RESULTS} ({added[0][:12]}) is not on origin/main; push it "
                      "first")
    return added[0]


def probe_days(root: Path, spec: Dict[str, Any]) -> List[str]:
    """The amendment's maker-probe-days block as committed, checked against §11's draw."""
    amendment = spec["freeze"]["amendment_file"]
    days = wfa.maker_probe_days_from(gitrepo.show_file(root, "HEAD", amendment))
    drawn = mpd.committed_draw(root, spec).days
    if days != drawn:
        raise Refusal(f"the amendment's maker-probe days {days} differ from §11's draw {drawn}")
    return days


def day_files(doc: Dict[str, Any], days: Sequence[str]) -> List[ArchiveFile]:
    """
    The files the days need, each once, in the order the days list them:
    for every day, the four of maker_probe_days.REQUIREMENTS, as the
    candidate list names and locates them.
    """
    flagged = set(doc.get("files_overlapping_protected") or [])
    out: Dict[str, ArchiveFile] = {}
    for d in days:
        row = (doc.get("days") or {}).get(d)
        if not isinstance(row, dict) or row.get("qualifies") is not True:
            raise Refusal(f"{d} is not a qualifying day of the candidate list")
        for slot, (kind, after) in mpd.REQUIREMENTS.items():
            names = row.get(slot) or []
            if len(names) != 1:
                raise Refusal(f"{d}: the candidate list names {len(names)} files for {slot}, not one")
            file_date = (date.fromisoformat(d) + timedelta(days=after)).isoformat()
            listed = [f for f in (doc.get("files") or {}).get(kind, {}).get(file_date, [])
                      if f.get("filename") == names[0]]
            if len(listed) != 1:
                raise Refusal(f"{d}: {names[0]} is listed {len(listed)} times under {kind} {file_date}")
            f = listed[0]
            if f.get("overlaps_protected") is not False or f["filename"] in flagged:
                raise Refusal(f"{d}: {f['filename']} reaches into a test window")
            out.setdefault(f["filename"], ArchiveFile(kind, file_date, f["filename"], f["url"],
                                                      list(f["covers_utc"])))
    return list(out.values())


# --------------------------------------------------------------------------
# downloading
# --------------------------------------------------------------------------

def archive_members(path: Path) -> List[Dict[str, Any]]:
    """
    Read an archive to its end and list its files (name, bytes). A truncated
    or corrupt file raises: tarfile and zipfile check every CRC on the way.
    """
    name = path.name
    members: List[Dict[str, Any]] = []
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"corrupt member {bad}")
            members = [{"name": i.filename, "bytes": i.file_size} for i in zf.infolist() if not i.is_dir()]
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                fh = tf.extractfile(m)
                n = 0
                for block in iter(lambda: fh.read(_CHUNK), b""):
                    n += len(block)
                if n != m.size:
                    raise tarfile.ReadError(f"{m.name}: read {n} of {m.size} bytes")
                members.append({"name": m.name, "bytes": m.size})
    else:
        raise FetchError(f"{name}: not a .zip or .tar.gz archive")
    if not members:
        raise FetchError(f"{name}: the archive holds no file")
    return members


def _open(url: str, method: str = "GET") -> Any:
    return urllib.request.urlopen(urllib.request.Request(url, headers=_HEADERS, method=method),
                                  timeout=HTTP_TIMEOUT_S)


def _server_headers(resp: Any) -> Dict[str, Optional[str]]:
    return {"last_modified": resp.headers.get("Last-Modified"), "etag": resp.headers.get("ETag"),
            "content_length": resp.headers.get("Content-Length")}


def download(f: ArchiveFile, dest_dir: Path, *, opener: Callable[..., Any] = _open,
             sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """
    ``f`` at dest_dir/<filename>, downloaded unless it is already there, and
    its manifest entry. A new file is renamed into place only after it has
    been read to its end; a file already there is read to its end too.
    """
    dest = dest_dir / f.filename
    entry: Dict[str, Any] = dict(f._asdict())
    if dest.exists():
        with opener(f.url, "HEAD") as resp:
            entry.update(_server_headers(resp))
        entry.update(downloaded_utc=None, resumed=True)
    else:
        last: Optional[BaseException] = None
        for attempt in range(MAX_RETRIES):
            tmp = dest_dir / f".part-{f.filename}"       # keeps the suffix archive_members reads
            try:
                with opener(f.url) as resp, open(tmp, "wb") as out:
                    headers = _server_headers(resp)
                    for block in iter(lambda: resp.read(_CHUNK), b""):
                        out.write(block)
                want = headers["content_length"]
                if want is not None and int(want) != tmp.stat().st_size:
                    raise FetchError(f"{f.filename}: got {tmp.stat().st_size} of {want} bytes")
                members = archive_members(tmp)
                os.link(tmp, dest)               # fails if dest appeared meanwhile: never overwrite
                tmp.unlink()
                entry.update(headers, downloaded_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             resumed=False, members=members)
                break
            except FileExistsError:
                tmp.unlink(missing_ok=True)
                raise
            except ARCHIVE_ERRORS as exc:
                last = exc
                tmp.unlink(missing_ok=True)
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_S * 2 ** attempt
                    logger.warning(f"{f.filename}: {type(exc).__name__}: {exc} — retry {attempt + 1} in {wait:g}s")
                    sleep(wait)
        else:
            raise FetchError(f"giving up on {f.filename} after {MAX_RETRIES} attempts: {last}")
    entry["bytes"] = dest.stat().st_size
    if entry.get("content_length") is not None and int(entry["content_length"]) != entry["bytes"]:
        raise FetchError(f"{f.filename} on disk has {entry['bytes']} bytes, the server {entry['content_length']}; "
                         "delete it and rerun")
    entry["sha256"] = sha256_file(dest)
    if entry["resumed"]:
        entry["members"] = archive_members(dest)
    return entry


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def fetch_listed(root: Path, opener: Callable[..., Any] = _open) -> int:
    """Readers' mode: the committed manifest's files, each fetched if missing and checked against its SHA-256."""
    if gitrepo.path_status(root, MANIFEST):
        logger.error(f"Refusing to download: {MANIFEST} differs from its committed version")
        return EXIT_REFUSED
    doc = json.loads(gitrepo.show_file(root, "HEAD", MANIFEST))
    dest_dir = root / doc["archive_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    wrong = []
    try:
        for i, f in enumerate(doc["files"], 1):
            got = download(ArchiveFile(f["kind"], f["file_date"], f["filename"], f["url"], list(f["covers_utc"])),
                           dest_dir, opener=opener)
            ok = got["sha256"] == f["sha256"]
            wrong += [] if ok else [f["filename"]]
            logger.info(f"[{i}/{len(doc['files'])}] {f['filename']}: " + ("SHA-256 matches" if ok else
                        f"SHA-256 {got['sha256']}, the manifest says {f['sha256']}"))
    except ARCHIVE_ERRORS as exc:
        logger.error(f"Download failed: {exc}. Rerun to resume; files already complete are kept.")
        return EXIT_FETCH_FAILED
    if wrong:
        logger.error(f"{len(wrong)} files differ from the manifest: {', '.join(wrong)}")
        return EXIT_FETCH_FAILED
    logger.info(f"All {len(doc['files'])} files match {MANIFEST}; nothing was written but the files.")
    return EXIT_OK


def run(*, root: Path = REPO_ROOT, spec: Optional[Dict[str, Any]] = None,
        opener: Callable[..., Any] = _open) -> int:
    foreign = integrity.foreign_code_problem(REPO_ROOT)
    if foreign:
        logger.error(f"Refusing to download: {foreign}")
        return EXIT_REFUSED
    s = spec or get_spec()
    root = Path(root)
    manifest = root / MANIFEST
    try:
        if gitrepo.is_tracked(root, MANIFEST):
            return fetch_listed(root, opener)
        if manifest.exists():
            raise Refusal(f"{MANIFEST} exists but is not committed; commit it, or remove it to download afresh",
                          EXIT_EXISTS)
        fwd = doge_fwd_commit(root)
        days = probe_days(root, s)
        cand_rel = s["maker_probe"]["candidates_file"]
        doc = json.loads(gitrepo.show_file(root, "HEAD", cand_rel))
        files = day_files(doc, days)
    except Refusal as exc:
        logger.error(f"Refusing to download: {exc}")
        return exc.code
    except (gitrepo.GitError, mpd.Refusal, ValueError) as exc:
        logger.error(f"Refusing to download: {exc}")
        return EXIT_REFUSED
    dest_dir = root / ARCHIVE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{len(days)} days, {len(files)} files -> {ARCHIVE_DIR}/")
    entries = []
    try:
        for i, f in enumerate(files, 1):
            entries.append(download(f, dest_dir, opener=opener))
            e = entries[-1]
            logger.info(f"[{i}/{len(files)}] {f.filename}: {e['bytes'] / 1e6:,.1f} MB"
                        + (" (already here)" if e["resumed"] else ""))
    except ARCHIVE_ERRORS as exc:
        logger.error(f"Download failed: {exc}. Rerun to resume; files already complete are kept.")
        return EXIT_FETCH_FAILED
    write_once_json({
        "manifest_version": 1,
        "tool": "scripts/fetch_maker_probe.py",
        "section": "§11",
        "inst_id": doc.get("inst_id"),
        "days": days,
        "doge_fwd_results_commit": fwd,
        "candidates_file": cand_rel,
        "candidates_commit": gitrepo.commits_touching(root, cand_rel)[0],
        "archive_dir": ARCHIVE_DIR,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": entries,
    }, manifest)
    logger.info(f"Manifest -> {MANIFEST}. Commit it before the files are parsed.")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None, **seams: Any) -> int:
    argparse.ArgumentParser(description="Download the maker probe's archive files (§11).").parse_args(argv)
    return run(**seams)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        raise SystemExit(130)
