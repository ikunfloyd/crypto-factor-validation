#!/usr/bin/env python3
"""
The candidate days of the maker-execution probe, and the draw from them (§11, "Days").

§11 replays OKX's order book and trades on 14 days of the in-sample year. A
random draw is only as good as the list it draws from, so the list is fixed
by rule and committed before anyone draws:

    list   asks OKX's archive which files exist for each UTC day D from
           maker_probe.population.from to .to (Appendix A), and writes the
           days for which all four files below are listed to
           maker_probe.candidates_file (results/manifest_maker_probe.json),
           exclusively. Only listing metadata is requested: no archive file
           is downloaded.
    draw   reads that file as committed and prints
               numpy.random.default_rng(seed).choice(candidates,
                   size=min(n_days, len(candidates)), replace=False)
           in drawn order and sorted. scripts/write_freeze_amendment.py
           records the same draw (committed_draw) in the freeze amendment
           (§10 step 3); it writes nothing here.

The four files for day D (names as in maker_probe.population.requires):

    orderbook_D, orderbook_D1   module 4, 400-level order book, dated in UTC: D and D+1
    trades_D, trades_D1         module 1, trades, dated in UTC+8: D and D+1

The trade file dated D covers D-1 16:00 -> D 16:00 UTC, so UTC day D needs
the files dated D and D+1; the order book dated D+1 carries the markouts of
D's last decisions. The instrument is the in-sample set's only one
(sets.IS.inst, DOGE-USDT-SWAP), whose family is DOGE-USDT.

What the listing looks like. OKX documents GET /api/v5/public/market-data-history
("Get historical market data", 5 requests per 2 s per IP), and four metadata-only
calls on 2026-10-11 for DOGE-USDT over 2025-10-01 .. 2025-10-10 confirmed it:

    module          1 trades, 2 candles, 3 funding, 4 order book 400 levels,
                    5 order book 5000 levels (from 2025-11-01), 6 50 levels
    instType        SWAP
    instFamilyList  DOGE-USDT. instIdList applies to SPOT only; for a swap the
                    family names it, and both modules answered to it
    dateAggrType    daily
    begin, end      epoch ms, both inclusive. Only the date counts, read in
                    UTC+8 for modules 1-3 and in UTC for 4-6: begin = end =
                    2025-10-04 20:00 UTC returned the trade file dated 10-05
                    and the order book dated 10-04. At most 10 days per
                    request; 10-01 .. 10-10 returned all ten files
    answer          data[0] = {ts (server time of the answer), dateAggrType,
                    totalSizeMB, details: [{instFamily, instId (""), instType,
                    ccy, dateRangeStart, dateRangeEnd, groupSizeMB,
                    groupDetails: [{filename, dateTs, sizeMB, url}]}]}, files
                    newest first, e.g.
                      DOGE-USDT-SWAP-trades-2025-10-01.zip           dateTs = 10-01 00:00 UTC+8
                      DOGE-USDT-SWAP-L2orderbook-400lv-2025-10-01.tar.gz  dateTs = 10-01 00:00 UTC
                    (the docs' field table spells it dataTs; both are read).
                    dateRangeStart/End describe the files returned, not the
                    request, so they cannot show that a range was cut short.

Assumptions, and what happens if one fails:

  * begin and end are sent as 04:00 UTC of the first and last date, which is
    12:00 UTC+8 of the same date, so either time-zone reading asks for the
    same dates.
  * A request spans DAILY_CHUNK_DAYS = 7 dates, inside the cap of 10. The
    docs also say a query beyond its "record limits" silently returns the
    files nearest ``end``; a short request stays clear of that.
  * A file's date is the one in its name. Its dateTs must be that date's
    midnight in the module's time zone, and the date must be one the request
    asked for; otherwise the listing is not what was learned here and the run
    stops rather than guess a date.
  * Files of other instruments in the family are skipped. A day may have more
    than one file of a kind; any one satisfies the requirement, and all are
    recorded.
  * A missing file is not an error: the day is not a candidate, and the
    manifest says which of the four it lacked.

Sizes are not recorded. A file's size tracks the activity it holds, and two
of the files the population needs reach into test windows: the trade file
dated 2025-09-24 starts at 2025-09-23 16:00 UTC, inside doge_2024, and the
order book dated 2026-09-23 runs to 2026-09-24 00:00 UTC, inside doge_fwd.
The manifest flags such files (overlaps_protected) for the probe.

Usage:
    python scripts/maker_probe_days.py list     # once; then commit the file
    python scripts/maker_probe_days.py draw     # after the commit
"""
from __future__ import annotations

import argparse
import atexit
import json
import re
import shutil
import sys
import tempfile
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

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

from research import gitrepo, integrity
from research.io import write_once_json
from research.spec import get_spec, overlaps_protected


# The OKX client, its throttle and retries, the archive-listing walk and the
# UTC+8 zone all come from fetch_v2, so the two scripts ask OKX the same way.
fv = integrity.load_script(REPO_ROOT, "fetch_v2")

EXIT_OK = 0
EXIT_FETCH_FAILED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_EXISTS = 4

ORDERBOOK = "orderbook"
TRADES = "trades"
# Archive module and the time zone its file dates (and dateTs) are read in.
MODULES: Dict[str, Tuple[str, timezone]] = {
    ORDERBOOK: ("4", timezone.utc),
    TRADES: ("1", fv.UTC8),
}
# Appendix A's names for the four files of day D: (kind, days after D).
REQUIREMENTS: Dict[str, Tuple[str, int]] = {
    "orderbook_D": (ORDERBOOK, 0),
    "orderbook_D1": (ORDERBOOK, 1),
    "trades_D": (TRADES, 0),
    "trades_D1": (TRADES, 1),
}
DAILY_CHUNK_DAYS = 7
# 04:00 UTC is 12:00 UTC+8, the same calendar date in both zones.
REQUEST_HOUR_UTC = 4

_DATE_IN_NAME = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")


class Refusal(Exception):
    """A precondition of §11 does not hold; the message says which."""

    def __init__(self, message: str, code: int = EXIT_REFUSED) -> None:
        super().__init__(message)
        self.code = code


class Draw(NamedTuple):
    """The committed candidate list and what §11's draw takes from it."""

    candidates: List[str]
    order: List[str]          # as numpy drew them
    days: List[str]           # the same days, sorted


# ---------------------------------------------------------------- dates

def parse_day(text: str) -> date:
    """'YYYY-MM-DD' -> date, refusing any other spelling (so '2025-9-24' is not a second name for a day)."""
    d = datetime.strptime(text, "%Y-%m-%d").date()
    if d.isoformat() != text:
        raise ValueError(f"not a YYYY-MM-DD date: {text!r}")
    return d


def _midnight_ms(d: date, tz: timezone) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=tz).timestamp() * 1000)


def _request_ms(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, REQUEST_HOUR_UTC, tzinfo=timezone.utc).timestamp() * 1000)


def population(spec: Dict[str, Any]) -> List[date]:
    """Every UTC day from maker_probe.population.from to .to, both included, ascending."""
    pop = spec["maker_probe"]["population"]
    try:
        lo, hi = parse_day(str(pop["from"])), parse_day(str(pop["to"]))
    except ValueError as exc:
        raise Refusal(f"maker_probe.population in Appendix A: {exc}") from None
    if hi < lo:
        raise Refusal(f"maker_probe.population runs backwards ({lo} .. {hi})")
    if set(pop.get("requires", [])) != set(REQUIREMENTS):
        raise Refusal(f"maker_probe.population.requires is {pop.get('requires')}, "
                      f"but this script checks {list(REQUIREMENTS)}")
    if pop.get("order", "ascending") != "ascending":
        raise Refusal(f"maker_probe.population.order is {pop.get('order')!r}; §11 sorts the candidates ascending")
    return [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]


def probe_instrument(spec: Dict[str, Any]) -> Tuple[str, str]:
    """(instId, instFamily) of the probe: the in-sample set's single perpetual."""
    insts = list(spec["sets"]["IS"]["inst"])
    if len(insts) != 1 or not str(insts[0]).endswith("-SWAP"):
        raise Refusal(f"sets.IS.inst is {insts}; the probe needs exactly one perpetual swap")
    inst = str(insts[0])
    return inst, inst[: -len("-SWAP")]


# ---------------------------------------------------------------- listing

def _file_date(name: str) -> Optional[date]:
    """The one date in a file name, or None if there is none or more than one."""
    found = set(_DATE_IN_NAME.findall(name))
    if len(found) != 1:
        return None
    try:
        return parse_day(found.pop())
    except ValueError:
        return None


def _response_ts(payload: Dict[str, Any]) -> Optional[int]:
    try:
        return int(payload["data"][0]["ts"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def list_module(
    client: Any, kind: str, inst_id: str, family: str, dates: Sequence[date]
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    ({file date: [file record, ...]}, [request record, ...]) for one module over ``dates``.

    ``dates`` are file dates in the module's own time zone, ascending and
    consecutive. Raises FetchError (fetch_v2's) if OKX keeps failing or
    answers in a way that would make a file's date a guess.
    """
    module, tz = MODULES[kind]
    files: Dict[str, List[Dict[str, Any]]] = {}
    requests: List[Dict[str, Any]] = []
    for i in range(0, len(dates), DAILY_CHUNK_DAYS):
        chunk = list(dates[i:i + DAILY_CHUNK_DAYS])
        asked = {d.isoformat() for d in chunk}
        params = {
            "module": module,
            "instType": "SWAP",
            "instFamilyList": family,
            "dateAggrType": "daily",
            "begin": str(_request_ms(chunk[0])),
            "end": str(_request_ms(chunk[-1])),
        }
        payload = client.get_json(fv.ARCHIVE_PATH, params, "archive")
        listed = skipped = 0
        for rec, fam in fv._iter_file_records(payload.get("data")):
            url = rec["url"]
            name = str(rec.get("filename") or rec.get("fileName")
                       or urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])
            if fam not in (None, family) or not name.startswith(f"{inst_id}-"):
                skipped += 1
                continue
            d = _file_date(name)
            if d is None:
                raise fv.FetchError(f"module {module}: cannot tell which date archive file {name} holds")
            label = d.isoformat()
            if label not in asked:
                raise fv.FetchError(
                    f"module {module}: OKX listed {name} for a request covering {chunk[0]} .. {chunk[-1]}; "
                    "it no longer reads begin/end as this script assumes"
                )
            stamp = rec.get("dateTs", rec.get("dataTs"))
            if stamp not in (None, ""):
                text = str(stamp).strip()
                if not text.isdigit() or int(text) != _midnight_ms(d, tz):
                    raise fv.FetchError(
                        f"module {module}: {name} has dateTs {stamp!r}, not {label} 00:00 in its zone "
                        f"({_midnight_ms(d, tz)}); its date would be a guess"
                    )
            lo, hi = _midnight_ms(d, tz), _midnight_ms(d + timedelta(days=1), tz)
            entry = {
                "filename": name,
                "url": url,
                "dateTs": int(str(stamp).strip()) if stamp not in (None, "") else None,
                "covers_utc": [lo, hi],
                "overlaps_protected": overlaps_protected(lo, hi),
            }
            same_date = files.setdefault(label, [])
            if all(e["url"] != url for e in same_date):
                same_date.append(entry)
            listed += 1
        requests.append({
            "kind": kind,
            "params": params,
            "response_ts": _response_ts(payload),
            "files_listed": listed,
            "other_files_skipped": skipped,
        })
    return files, requests


def qualify(
    days: Sequence[date], files: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Dict[str, Any]]:
    """Per day D: the file names found for each of the four requirements, what is missing, and whether D qualifies."""
    out: Dict[str, Dict[str, Any]] = {}
    for d in days:
        rec: Dict[str, Any] = {}
        for req, (kind, offset) in REQUIREMENTS.items():
            rec[req] = [e["filename"] for e in files[kind].get((d + timedelta(days=offset)).isoformat(), [])]
        missing = [r for r in REQUIREMENTS if not rec[r]]
        out[d.isoformat()] = {"qualifies": not missing, "missing": missing, **rec}
    return out


def run_list(root: Path, spec: Dict[str, Any]) -> int:
    """
    ``list``: ask OKX about every file the population needs, then write the
    manifest in one step, so a failure halfway leaves nothing that would
    block a retry.
    """
    rel = spec["maker_probe"]["candidates_file"]
    out = root / rel
    # Write-once before the first request, against the history as well: a
    # list committed once and later removed was the list (run_insample_v2.py
    # judges its output the same way).
    if out.exists():
        raise Refusal(f"{rel} exists; the candidate list is written once (§11)", EXIT_EXISTS)
    if gitrepo.is_work_tree(root) and gitrepo.commits_touching(root, rel, all_refs=True, reflog=True):
        raise Refusal(f"{rel} was committed before; the candidate list is written once (§11)", EXIT_EXISTS)
    inst, family = probe_instrument(spec)
    days = population(spec)
    dates = days + [days[-1] + timedelta(days=1)]          # D+1 of the last day

    client = fv.OkxClient()
    try:
        now_ms = fv._server_ms()
        files: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        requests: List[Dict[str, Any]] = []
        for kind in MODULES:
            files[kind], made = list_module(client, kind, inst, family, dates)
            requests += made
    except fv.FetchError as exc:
        logger.error(f"Listing failed, nothing written: {exc}")
        return EXIT_FETCH_FAILED

    table = qualify(days, files)
    candidates = sorted(d for d, r in table.items() if r["qualifies"])
    flagged = sorted({e["filename"] for by_date in files.values() for same in by_date.values()
                      for e in same if e["overlaps_protected"]})
    manifest = {
        "manifest_version": 1,
        "tool": "scripts/maker_probe_days.py",
        "section": "§11 Days",
        "inst_id": inst,
        "inst_family": family,
        "population": dict(spec["maker_probe"]["population"]),
        "modules": {k: {"module": m, "file_dates_in": "UTC" if tz is timezone.utc else "UTC+8"}
                    for k, (m, tz) in MODULES.items()},
        "endpoint": fv.OKX_REST_BASE + fv.ARCHIVE_PATH,
        "server_time_ms": now_ms,
        "server_time_utc": fv.fmt_utc(now_ms),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requests": requests,
        "n_days": len(table),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "days": table,
        "files": {k: {d: v for d, v in sorted(by_date.items())} for k, by_date in files.items()},
        "files_overlapping_protected": flagged,
    }
    try:
        write_once_json(manifest, out)
    except FileExistsError:
        raise Refusal(f"{rel} exists; the candidate list is written once (§11)", EXIT_EXISTS) from None

    logger.info(f"{len(candidates)} of {len(table)} days qualify -> {rel} ({client.requests} listing requests)")
    for req in REQUIREMENTS:
        lacking = [d for d, r in table.items() if req in r["missing"]]
        if lacking:
            logger.info(f"  lacking {req}: {len(lacking)} day(s), e.g. {', '.join(lacking[:5])}")
    if flagged:
        logger.warning("Files reaching into a protected test window (the probe must not read those parts): "
                       + ", ".join(flagged))
    logger.info(f"Next: commit {rel} (§11 commits it before the draw), then: python scripts/maker_probe_days.py draw")
    return EXIT_OK


# ---------------------------------------------------------------- draw

def check_numpy(spec: Dict[str, Any]) -> None:
    """The draw runs under the numpy §11 names: another version may turn the same seed into other days."""
    want = str(spec["maker_probe"]["numpy_version"])
    if np.__version__ != want:
        raise Refusal(f"numpy is {np.__version__}, but §11 draws under numpy {want} "
                      "(maker_probe.numpy_version, as environment.yml pins it)")


def drawn_order(candidates: Sequence[str], spec: Dict[str, Any]) -> List[str]:
    """
    §11's draw, in the order numpy returns it:
    ``numpy.random.default_rng(seed).choice(candidates, size=min(n_days, len(candidates)), replace=False)``.

    The result depends on the order of ``candidates``, so they must be the
    sorted list §11 commits; and on numpy's version (check_numpy).
    """
    check_numpy(spec)
    cands = list(candidates)
    if not cands:
        raise Refusal("there are no candidate days (§11: if the data cannot be had, the report says what is "
                      "missing and stops)")
    if cands != sorted(set(cands)):
        raise Refusal("the candidates are not sorted ascending without repeats, as §11 draws from them")
    mp = spec["maker_probe"]
    rng = np.random.default_rng(int(mp["seed"]))
    picked = rng.choice(cands, size=min(int(mp["n_days"]), len(cands)), replace=False)
    return [str(d) for d in picked]


def drawn_days(candidates: Sequence[str], spec: Dict[str, Any]) -> List[str]:
    """The drawn days, sorted ascending."""
    return sorted(drawn_order(candidates, spec))


def parse_candidates(text: str, spec: Dict[str, Any]) -> List[str]:
    """
    The candidate list from the manifest's text, refusing one that is not
    what ``list`` writes: a sorted list of unique population days of the
    probe's instrument, equal to the days its own table says qualify.
    """
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise Refusal(f"the candidate file is not JSON: {exc}") from None
    cands = doc.get("candidates") if isinstance(doc, dict) else None
    if not isinstance(cands, list) or not all(isinstance(c, str) for c in cands):
        raise Refusal("the candidate file has no list of date strings under 'candidates'")
    inst, _ = probe_instrument(spec)
    if doc.get("inst_id") != inst:
        raise Refusal(f"the candidate file is for {doc.get('inst_id')!r}, not {inst}")
    days = population(spec)
    lo, hi = days[0].isoformat(), days[-1].isoformat()
    for c in cands:
        try:
            parse_day(c)
        except ValueError:
            raise Refusal(f"candidate {c!r} is not a YYYY-MM-DD date") from None
        if not lo <= c <= hi:
            raise Refusal(f"candidate {c} lies outside the population {lo} .. {hi}")
    if cands != sorted(set(cands)):
        raise Refusal("the candidates are not sorted ascending without repeats (§11)")
    table = doc.get("days")
    if isinstance(table, dict):
        qualifying = sorted(d for d, r in table.items() if isinstance(r, dict) and r.get("qualifies") is True)
        if qualifying != cands:
            raise Refusal("the candidate list differs from the days the file's own table says qualify")
    return cands


def committed_candidates(root: Path, spec: Dict[str, Any]) -> List[str]:
    """
    The candidate list as committed at HEAD.

    §11 commits it before the draw, so it must be tracked, without
    uncommitted change, and changed by exactly one commit: a list revised
    after its first commit could have been revised after a look at a draw.
    It is read from the commit, not the working tree, so an index flag such
    as skip-worktree cannot put other text under the committed name.
    """
    rel = spec["maker_probe"]["candidates_file"]
    if not gitrepo.is_work_tree(root):
        raise Refusal(f"{root} is not a git work tree, so the committed {rel} cannot be read")
    if not gitrepo.is_tracked(root, rel):
        raise Refusal(f"{rel} is not committed; §11 commits the candidate list before the draw")
    dirty = gitrepo.path_status(root, rel)
    if dirty:
        raise Refusal(f"{rel} has uncommitted changes ({dirty[0][:2].strip()}); the draw uses the list as committed")
    touched = gitrepo.commits_touching(root, rel)
    if len(touched) != 1:
        raise Refusal(f"{rel} was changed by {len(touched)} commits; §11 commits the list once, as written by "
                      "`list`, and draws from that")
    return parse_candidates(gitrepo.show_file(root, "HEAD", rel), spec)


def committed_draw(root: Path, spec: Dict[str, Any]) -> Draw:
    """§11's draw from the committed candidate list, with every refusal of ``draw``."""
    check_numpy(spec)
    cands = committed_candidates(root, spec)
    order = drawn_order(cands, spec)
    return Draw(candidates=cands, order=order, days=sorted(order))


def run_draw(root: Path, spec: Dict[str, Any]) -> int:
    """``draw``: print the draw; nothing is written (the freeze amendment records it)."""
    d = committed_draw(root, spec)
    mp = spec["maker_probe"]
    logger.info(f"{len(d.order)} of {len(d.candidates)} candidate days "
                f"(seed {mp['seed']}, n_days {mp['n_days']}, numpy {np.__version__}):")
    logger.info("drawn order: " + " ".join(d.order))
    logger.info("sorted:      " + " ".join(d.days))
    return EXIT_OK


# ---------------------------------------------------------------- CLI

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="List the maker-probe candidate days, or draw from the committed list (§11).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="write the candidate list (once), from OKX's archive listing")
    sub.add_parser("draw", help="print §11's draw from the committed candidate list")
    return p.parse_args(argv)


def run(args: argparse.Namespace, *, root: Path = REPO_ROOT, spec: Optional[Dict[str, Any]] = None) -> int:
    """
    The CLI's body. ``root`` and ``spec`` are seams for tests (a throwaway
    repository, a shorter population); the real run passes neither.
    """
    foreign = integrity.foreign_code_problem(REPO_ROOT)
    if foreign:
        logger.error(f"Refusing to {args.command}: {foreign}")
        return EXIT_REFUSED
    s = spec or get_spec()
    try:
        if args.command == "list":
            return run_list(Path(root), s)
        return run_draw(Path(root), s)
    except Refusal as exc:
        logger.error(f"Refusing to {args.command}: {exc}")
        return exc.code
    except gitrepo.GitError as exc:
        logger.error(f"Refusing to {args.command}: {exc}")
        return EXIT_REFUSED


def main(argv: Optional[Sequence[str]] = None, **seams: Any) -> int:
    return run(parse_args(argv), **seams)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        raise SystemExit(130)
