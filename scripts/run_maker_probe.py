#!/usr/bin/env python3
"""
The maker-execution probe, run once (pre-registration §11).

For each of the 14 days in results/maker_probe/manifest_archives.json:

  1. every archive file is hashed and compared with the committed manifest;
  2. the order book and the trades are parsed from the files dated D and D+1
     (research.maker_probe.read_book, read_trades);
  3. the parsed day is checked: the rebuilt book never holds more than 400
     levels a side (max_depth); it is compared with every snapshot of the
     archive (check_snapshots); each trade is matched with the book decrease
     it causes (trade_book_lags);
  4. the 288 decisions are simulated twice. ``parser_gaps`` keeps only the
     gaps the parsers found (missing data). ``touch_gaps`` also drops the
     interval since a side's previous snapshot wherever a snapshot's best
     level differs from the rebuilt book (touch_mismatch_gaps). Both are
     reported (§12, 2026-11-24).

Then the daily summaries and the mean and SE across days of each run.

It writes, each file new (never overwritten), into results/maker_probe/:

    orders.csv    one row per run, day, decision, side and fill model
    daily.csv     research.maker_probe.daily_summary, per run
    summary.csv   research.maker_probe.across_days, per run
    checks.csv    per day: what the parsers read and found, the depth, the
                  snapshot check by rank, the clock and size-unit check, and
                  each run's gaps

Every row carries the commit of the code that ran (``code_commit``) and the
SHA-256 of the archive manifest. Before reading any file the script checks
that the manifest is committed once and unchanged, that the code it runs
and Appendix A have no uncommitted change, and that no output exists or was
ever committed.

``--verify`` is for readers. It needs the archive files (download them with
scripts/fetch_maker_probe.py, which checks each against the manifest), runs
the same computation, stamped with the committed outputs' code_commit, and
compares each output with its committed text: equal bytes, or equal keys
and text and numbers equal to 1e-9 (run_oos.results_differences). It writes
nothing, and exits 0 when everything matches.

Usage:
    python scripts/run_maker_probe.py              # the one run
    python scripts/run_maker_probe.py --verify     # recompute and compare
"""
from __future__ import annotations

import argparse
import atexit
import dataclasses
import io
import json
import shutil
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
# Startup, as in the other study scripts: this checkout's packages first on
# sys.path, compiled from source into a fresh bytecode cache.
sys.path[:] = [str(REPO_ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]
if __name__ == "__main__":
    sys.pycache_prefix = tempfile.mkdtemp(prefix=f"{Path(__file__).stem}-pycache-")
    atexit.register(shutil.rmtree, sys.pycache_prefix, True)

import polars as pl
from loguru import logger

from research import gitrepo, integrity
from research import maker_probe as mp
from research.io import sha256_file, write_once_csv
from research.spec import get_spec

# results_differences: the comparison verify_results.py and run_oos.py's guard 9 use.
run_oos = integrity.load_script(REPO_ROOT, "run_oos")

EXIT_OK = 0
EXIT_DIFFERS = 1
EXIT_REFUSED = 3
EXIT_EXISTS = 4

ARCHIVE_MANIFEST = "results/maker_probe/manifest_archives.json"
OUT_DIR = "results/maker_probe"
OUTPUTS = ("orders.csv", "daily.csv", "summary.csv", "checks.csv")
RUNS = ("parser_gaps", "touch_gaps")
# The code a run executes and the constants it reads: none may differ from HEAD.
CODE = ("research/__init__.py", "research/maker_probe.py", "research/spec.py", "research/io.py",
        "scripts/run_maker_probe.py", "docs/preregistration.md")


class Refusal(Exception):
    def __init__(self, message: str, code: int = EXIT_REFUSED) -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def check_before_run(root: Path) -> Dict[str, Any]:
    """Everything that must hold before a file is read; returns the committed archive manifest."""
    for name in OUTPUTS:
        rel = f"{OUT_DIR}/{name}"
        if (root / rel).exists():
            raise Refusal(f"{rel} exists; the probe runs once", EXIT_EXISTS)
        if gitrepo.commits_touching(root, rel, reflog=True):
            raise Refusal(f"{rel} was committed before; the probe runs once", EXIT_EXISTS)
    if not gitrepo.is_tracked(root, ARCHIVE_MANIFEST):
        raise Refusal(f"{ARCHIVE_MANIFEST} is not committed; commit it before the files are parsed")
    if gitrepo.path_status(root, ARCHIVE_MANIFEST):
        raise Refusal(f"{ARCHIVE_MANIFEST} has uncommitted changes")
    if len(gitrepo.commits_touching(root, ARCHIVE_MANIFEST)) != 1:
        raise Refusal(f"{ARCHIVE_MANIFEST} was changed after its first commit")
    dirty = [p for p in CODE if gitrepo.path_status(root, p)]
    if dirty:
        raise Refusal(f"uncommitted changes in {', '.join(dirty)}; the probe runs committed code")
    return json.loads(gitrepo.show_file(root, "HEAD", ARCHIVE_MANIFEST))


def check_files(root: Path, manifest: Dict[str, Any]) -> Dict[str, Path]:
    """Each archive file by name, after its SHA-256 has matched the manifest's."""
    folder = root / manifest["archive_dir"]
    out = {}
    for f in manifest["files"]:
        path = folder / f["filename"]
        if not path.is_file():
            raise Refusal(f"{path} is missing; run scripts/fetch_maker_probe.py")
        got = sha256_file(path)
        if got != f["sha256"]:
            raise Refusal(f"{f['filename']}: SHA-256 {got}, the manifest says {f['sha256']}")
        out[f["filename"]] = path
    return out


def day_paths(manifest: Dict[str, Any], files: Dict[str, Path], day: str) -> Dict[str, List[Path]]:
    """The book files and trade files of ``day``, dated D then D+1, as the manifest lists them."""
    nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    out: Dict[str, List[Path]] = {}
    for kind in ("orderbook", "trades"):
        paths = []
        for d in (day, nxt):
            names = [f["filename"] for f in manifest["files"] if f["kind"] == kind and f["file_date"] == d]
            if len(names) != 1:
                raise Refusal(f"{day}: the manifest lists {len(names)} {kind} files dated {d}")
            paths.append(files[names[0]])
        out[kind] = paths
    return out


# --------------------------------------------------------------------------
# one day
# --------------------------------------------------------------------------

def _gap_ms(gaps: Any) -> int:
    return int(sum(hi - lo for lo, hi in (gaps.tolist() if hasattr(gaps, "tolist") else gaps)))


def run_day(day: str, paths: Dict[str, List[Path]], cfg: mp.ProbeConfig) -> Dict[str, Any]:
    """Parse, check and simulate one day; returns the orders of each run and the day's checks."""
    t = time.time()
    book_rep: Dict[str, Any] = {}
    trade_rep: Dict[str, Any] = {}
    book = mp.read_book(paths["orderbook"], day, cfg, report=book_rep)
    trades = mp.read_trades(paths["trades"], day, cfg, report=trade_rep)
    depth = mp.max_depth(book)
    if max(depth) > mp.BOOK_DEPTH:
        raise ValueError(f"{day}: the rebuilt book holds {max(depth)} levels; levels leaving the top "
                         f"{mp.BOOK_DEPTH} are not deleted and would have to be truncated")
    chk = mp.check_snapshots(book, mp.BOOK_DEPTH)
    lags = mp.trade_book_lags(book, trades)
    touch = mp.touch_mismatch_gaps(book, chk)
    books = {"parser_gaps": book, "touch_gaps": dataclasses.replace(book, gaps=touch)}
    orders = {run: mp.simulate_day(b, trades, day, cfg).with_columns(pl.lit(run).alias("run"))
              for run, b in books.items()}
    m = chk.mismatches
    first = m.group_by("row").agg(pl.col("rank").min().alias("first_rank"))
    touch_rows = m.filter(pl.col("rank") == 0)
    checks = {
        "day": day,
        "book_messages": book_rep["messages"], "book_rows": book_rep["rows"], "book_snapshots": book_rep["snapshots"],
        "book_other_instruments": book_rep["other_instruments"], "book_clamped": book_rep["clamped"],
        "book_max_clamp_ms": book_rep["max_clamp_ms"], "book_longest_pause_ms": book_rep["longest_pause_ms"],
        "book_pauses_as_gaps": len(book_rep["pauses_as_gaps"]), "book_ended_early": book_rep["ended_early"],
        "book_start_ms": book.start_ms, "book_end_ms": book.end_ms,
        "max_depth_bid": depth[0], "max_depth_ask": depth[1],
        "trades": trade_rep["trades"], "trades_source_1_left_out": trade_rep["source_1_trades"],
        "trades_other_instruments": trade_rep["other_instruments"],
        "trades_duplicate_rows": trade_rep["duplicate_rows"], "trades_id_jumps": trade_rep["id_jumps"],
        "trades_ended_early": trade_rep["ended_early"], "trades_end_ms": trades.end_ms,
        "trade_gaps": int(trades.gaps.shape[0]), "trade_gap_ms": _gap_ms(trades.gaps),
        "snapshot_sides_checked": chk.n_checked, "snapshot_sides_with_mismatch": chk.n_with_mismatch,
        "snapshot_sides_touch_mismatch": chk.n_touch_mismatch,
        "touch_mismatch_price_differs": int((touch_rows["rebuilt_price"] != touch_rows["reference_price"]).sum()),
        "first_mismatch_rank_1_9": int(first.filter(pl.col("first_rank").is_between(1, 9)).height),
        "first_mismatch_rank_10_99": int(first.filter(pl.col("first_rank").is_between(10, 99)).height),
        "first_mismatch_rank_100_399": int(first.filter(pl.col("first_rank") >= 100).height),
        "parser_gaps": int(book.gaps.shape[0]), "parser_gap_ms": _gap_ms(book.gaps),
        "touch_gaps": len(touch), "touch_gap_ms": _gap_ms(touch),
        **{f"clock_{k}": v for k, v in lags.items() if k != "trades"},
        "seconds": round(time.time() - t, 1),
    }
    return {"orders": orders, "checks": checks}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def compute(manifest: Dict[str, Any], files: Dict[str, Path], cfg: mp.ProbeConfig, commit: str,
            manifest_sha: str) -> Dict[str, pl.DataFrame]:
    """Every day of the manifest, both runs; the four outputs by file name, stamped with ``commit``."""
    orders: List[pl.DataFrame] = []
    checks: List[Dict[str, Any]] = []
    for day in manifest["days"]:
        got = run_day(day, day_paths(manifest, files, day), cfg)
        orders += [got["orders"][r] for r in RUNS]
        checks.append(got["checks"])
        c = got["checks"]
        logger.info(f"{day}: {c['book_rows']:,} book rows, {c['trades']:,} trades, snapshot sides with a touch "
                    f"mismatch {c['snapshot_sides_touch_mismatch']}/{c['snapshot_sides_checked']}, "
                    f"{c['seconds']} s")
    stamp = [pl.lit(commit).alias("code_commit"), pl.lit(manifest_sha).alias("manifest_sha256")]
    all_orders = pl.concat(orders, how="vertical")
    all_orders = all_orders.select(["run"] + [c for c in all_orders.columns if c != "run"])
    daily = pl.concat([mp.daily_summary(all_orders.filter(pl.col("run") == r).drop("run"), cfg)
                       .with_columns(pl.lit(r).alias("run")) for r in RUNS], how="vertical")
    summary = pl.concat([mp.across_days(daily.filter(pl.col("run") == r).drop("run"), cfg)
                         .with_columns(pl.lit(r).alias("run")) for r in RUNS], how="vertical")
    front = lambda df: df.select(["run"] + [c for c in df.columns if c != "run"]).with_columns(stamp)
    return {"orders.csv": front(all_orders), "daily.csv": front(daily), "summary.csv": front(summary),
            "checks.csv": pl.DataFrame(checks).with_columns(stamp)}


# --------------------------------------------------------------------------
# --verify: recompute and compare with the committed outputs
# --------------------------------------------------------------------------

# Wall-clock time per day, in checks.csv: recorded, never compared.
TIMING = "seconds"


def check_before_verify(root: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The committed archive manifest and the committed text of each output, refusing a modified one."""
    if not gitrepo.is_tracked(root, ARCHIVE_MANIFEST) or gitrepo.path_status(root, ARCHIVE_MANIFEST):
        raise Refusal(f"{ARCHIVE_MANIFEST} is not committed as it stands")
    committed = {}
    for name in OUTPUTS:
        rel = f"{OUT_DIR}/{name}"
        if not gitrepo.is_tracked(root, rel):
            raise Refusal(f"{rel} is not committed; there is nothing to verify")
        if gitrepo.path_status(root, rel):
            raise Refusal(f"{rel} differs from its committed version")
        committed[name] = gitrepo.show_file(root, "HEAD", rel)
    return json.loads(gitrepo.show_file(root, "HEAD", ARCHIVE_MANIFEST)), committed


def committed_stamp(committed: Dict[str, str]) -> Tuple[str, str]:
    """The one (code_commit, manifest_sha256) that every committed output carries."""
    seen = set()
    for text in committed.values():
        df = pl.read_csv(io.StringIO(text), infer_schema_length=0, columns=["code_commit", "manifest_sha256"])
        seen |= set(zip(df["code_commit"].to_list(), df["manifest_sha256"].to_list()))
    if len(seen) != 1:
        raise Refusal(f"the committed outputs carry {len(seen)} (code_commit, manifest_sha256) pairs, not one")
    return next(iter(seen))


def compare_outputs(committed: Dict[str, str], frames: Dict[str, pl.DataFrame]) -> Dict[str, List[str]]:
    """
    Per output, where the committed text and the recomputation differ (run_oos.results_differences:
    equal bytes, or equal keys and text and numbers equal to 1e-9). checks.csv's timing column is
    taken from the committed file, since a rerun cannot reproduce it.
    """
    out = {}
    for name, frame in frames.items():
        text = committed[name]
        if TIMING in frame.columns:
            old = pl.read_csv(io.StringIO(text), infer_schema_length=0)
            if TIMING in old.columns and old.height == frame.height:
                frame = frame.with_columns(old[TIMING].cast(frame[TIMING].dtype).alias(TIMING))
        out[name] = run_oos.results_differences(text, frame)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run(*, root: Path = REPO_ROOT, verify: bool = False) -> int:
    action = "verify" if verify else "run"
    foreign = integrity.foreign_code_problem(REPO_ROOT)
    if foreign:
        logger.error(f"Refusing to {action}: {foreign}")
        return EXIT_REFUSED
    root = Path(root)
    try:
        if verify:
            manifest, committed = check_before_verify(root)
            commit, manifest_sha = committed_stamp(committed)
            if manifest_sha != sha256_file(root / ARCHIVE_MANIFEST):
                raise Refusal(f"the outputs name another archive manifest ({manifest_sha[:12]}…)")
        else:
            manifest = check_before_run(root)
            commit, manifest_sha = gitrepo.head_commit(root), sha256_file(root / ARCHIVE_MANIFEST)
        logger.info(f"Hashing {len(manifest['files'])} archive files")
        files = check_files(root, manifest)
    except Refusal as exc:
        logger.error(f"Refusing to {action}: {exc}")
        return exc.code
    except gitrepo.GitError as exc:
        logger.error(f"Refusing to {action}: {exc}")
        return EXIT_REFUSED
    frames = compute(manifest, files, mp.ProbeConfig.from_spec(get_spec()), commit, manifest_sha)
    if verify:
        changed = [p for p in CODE if gitrepo.git(root, "diff", "--quiet", commit, "HEAD", "--", p,
                                                  check=False).returncode]
        if changed:
            logger.info(f"Code changed since the run ({commit[:12]}): {', '.join(changed)}")
        diffs = compare_outputs(committed, frames)
        for name, found in diffs.items():
            logger.info(f"{name}: " + ("identical" if not found else f"{len(found)} differences, e.g. {found[:3]}"))
        if any(diffs.values()):
            logger.error("FAIL: the recomputation differs from the committed outputs")
            return EXIT_DIFFERS
        logger.info(f"PASS: every output recomputes to the committed file ({TIMING!r} in checks.csv not compared)")
        return EXIT_OK
    out = root / OUT_DIR
    for name, frame in frames.items():
        write_once_csv(frame, out / name)
    logger.info(f"Wrote {', '.join(OUTPUTS)} to {OUT_DIR}/. Commit them unedited.")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None, **seams: Any) -> int:
    p = argparse.ArgumentParser(description="Run the maker-execution probe once (§11).")
    p.add_argument("--verify", action="store_true",
                   help="recompute from the archive files and compare with the committed outputs; write nothing")
    return run(verify=p.parse_args(argv).verify, **seams)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        raise SystemExit(130)
