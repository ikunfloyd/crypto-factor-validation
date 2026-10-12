#!/usr/bin/env python3
"""
The in-sample rerun of the pre-registered study (§6.5, §9, §10 step 2).

Computes, on the in-sample year only, each of the 60 cells' direction, flag,
active count, in-sample mean, standard error and minimum detectable effect,
and writes them to results/insample_v2.csv. That file is created exclusively
and never overwritten; the next commit adds it with no other change and is
tagged ``prereg-freeze``. The directions in it are what every test set is
then asked to confirm, so this script is the last point at which the
in-sample year influences anything.

Five refusals, all before any computation:

  * **Code from elsewhere.** A module of this repository's packages loaded
    from another directory (a PYTHONPATH entry ahead of this checkout)
    would make the checks below vouch for code that is not running; so this
    checkout goes first on sys.path, ``python scripts/run_insample_v2.py``
    compiles it from source into a fresh bytecode cache, and any foreign
    module stops the run.
  * **A dirty tree.** §10 step 2 asks for "the IS rerun, from a clean
    checkout": the file must be what committed code produces. Any
    uncommitted change stops the run, and so does any untracked file, since
    code that is in no commit (a new research/ module, say) would otherwise
    decide the directions unseen; git status runs with no file-system
    monitor to answer for it. So does what git status cannot see: an index
    entry marked skip-worktree or assume-unchanged, a file whose bytes are
    not HEAD's (a local clean filter), and an ignored file Python could
    import (gitrepo.checkout_problems). Other ignored files (data/history)
    do not count.
  * **An uncommitted supplement manifest.** §10 step 2 commits and pushes
    results/manifest_is_supp.json before the rerun, so it must be tracked.
  * **Unverified inputs.** The in-sample candles must still have Appendix
    A's SHA-256, and the supplementary spot leg and funding must match that
    manifest (§4.2).
  * **An existing output.** Write-once (§2, item 6), against the history as
    well as the working tree: a results/insample_v2.csv that was ever
    committed and then removed still counts, in any ref, on a branch merged
    after removing it, or in a commit only the reflog still holds (one
    undone with ``git reset``). One that was never committed and then
    deleted leaves no trace in git.

There is no flag to skip any of this; the test suite reaches the clean-tree
check through a keyword-only argument of run() that the CLI cannot set.
Like run_oos.py's guards (whose docstring states the threat model), these
refusals stop accidents and make any deviation a deliberate act; they
cannot stop someone who controls the machine.

It records nothing else: no log file, no second artefact, and the summary
it prints holds counts and flagged cells only, never factor performance.

Usage:
    python scripts/run_insample_v2.py
"""
from __future__ import annotations

import argparse
import atexit
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

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

from research import gitrepo, integrity, pipeline
from research.io import write_once_csv

EXIT_OK = 0
EXIT_REFUSED = 3
EXIT_EXISTS = 4


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write results/insample_v2.csv: the 60 in-sample directions (§6.5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return p.parse_args(argv)


def _summary(df: pl.DataFrame, out: Path) -> None:
    flagged = df.filter(pl.col("flagged"))
    logger.info(f"{df.height} cells -> {out}")
    by_reason = {r: int((flagged["flag_reason"] == r).sum()) for r in ("few_active", "missing_input")}
    logger.info(
        f"flagged: {flagged.height} (few_active {by_reason['few_active']}, "
        f"missing_input {by_reason['missing_input']})"
    )
    for r in flagged.iter_rows(named=True):
        logger.info(f"  {r['bar']:>3} h={r['horizon']:<2} {r['factor']:<22} {r['flag_reason']}")


def run(
    args: argparse.Namespace,
    *,
    root: Path = REPO_ROOT,
    spec: Optional[Dict[str, Any]] = None,
    compute: Optional[Callable[[Path], pl.DataFrame]] = None,
    allow_dirty: bool = False,
) -> int:
    """
    The CLI's body. ``root``, ``spec`` and ``compute`` are seams for tests:
    a throwaway repository, a spec whose in-sample hashes match synthetic
    files, and a cheaper computation. ``allow_dirty`` lets a test run outside
    a repository, skipping every git check (not the foreign-module one); it
    is keyword-only and no CLI flag sets it, so the real run cannot. The
    real run passes none of them.
    """
    root = Path(root)
    out = root / pipeline.INSAMPLE_RESULTS
    rel = pipeline.INSAMPLE_RESULTS.as_posix()

    foreign = integrity.foreign_code_problem(REPO_ROOT)
    if foreign:
        logger.error(f"Refusing the in-sample rerun: {foreign}")
        return EXIT_REFUSED
    if out.exists():
        logger.error(f"Refusing to overwrite {out} (write-once; a rerun needs a new path)")
        return EXIT_EXISTS
    if not allow_dirty:
        if not gitrepo.is_work_tree(root):
            logger.error(f"{root} is not a git work tree; the rerun must come from a clean checkout")
            return EXIT_REFUSED
        problems = gitrepo.checkout_problems(root)
        if problems:
            logger.error("The checkout is not exactly HEAD, so the rerun would not be what committed code "
                         "produces (§10 step 2 runs it from a clean checkout): " + "; ".join(problems))
            return EXIT_REFUSED
        manifest = pipeline.IS_SUPP_MANIFEST.as_posix()
        if not gitrepo.is_tracked(root, manifest):
            logger.error(f"{manifest} is not committed; §10 step 2 commits and pushes it before the rerun")
            return EXIT_REFUSED
        # Write-once against the history too: a committed results file that
        # was later removed, or whose commit was undone, still was the
        # in-sample rerun.
        if gitrepo.commits_touching(root, rel, all_refs=True, reflog=True):
            logger.error(f"{rel} was committed before; the in-sample rerun happens once (write-once, §10 step 2)")
            return EXIT_EXISTS

    problems = pipeline.verify_inputs("IS", root, spec=spec)
    if problems:
        logger.error("In-sample inputs do not match what was committed:\n  " + "\n  ".join(problems))
        return EXIT_REFUSED

    df = compute(root) if compute is not None else pipeline.run_insample(root)
    if tuple(df.columns) != pipeline.INSAMPLE_COLUMNS:
        raise AssertionError(f"computed columns {df.columns} are not the §6.5 columns")
    try:
        write_once_csv(df, out)
    except FileExistsError:
        logger.error(f"Refusing to overwrite {out}")
        return EXIT_EXISTS
    _summary(df, out)
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None, **seams: Any) -> int:
    return run(parse_args(argv), **seams)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    raise SystemExit(main())
