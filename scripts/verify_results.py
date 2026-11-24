#!/usr/bin/env python3
"""
Recompute a committed results file from its inputs and compare, writing nothing.

Each results file (results/insample_v2.csv, results/oos/<set>.csv and the
primary set's robustness file) was written once, by
scripts/run_insample_v2.py or scripts/run_oos.py, and both refuse to run
again: a second run of a set would be a second look at its test data (§10
steps 5-6). A reader who wants to check those numbers (the report on them is
docs/factor-research-v2.md) needs a path that recomputes without being a
run, and this is it. It takes the run's compute path unchanged, imported
rather than copied (research.pipeline.run_insample and run_oos_set, given
the directions and MDEs from where the run read them), and none of its
write path: no output file, no log file, no bytecode cache, no temporary
file.

What it checks, for one set (``--set``; ``all`` takes every set with
committed results, the primary set after the two whose statuses it reads):

  * **Inputs.** Every input file has the SHA-256 its committed manifest
    records (the in-sample candles, which doge_fwd's warm-up reads too:
    Appendix A's), and no unlisted file would be read
    (research.pipeline.verify_inputs, which both run scripts call). For a
    test set, the rest of run_oos's guard 8 too, about the set's own
    manifest: it was committed once, after the freeze amendment; it records
    the public repository and a window covering the set's; and each series
    it lists covers that window. A failure here stops the set before any
    computation, since a recomputation from other inputs proves nothing.
  * **Pushed.** Guard 8 also wants the commit that added a test set's
    manifest on origin/main, the author's record of the push (§10 step 4).
    In a reader's clone origin may be a fork, or the public repository
    under another name, so this asks the main branch of every remote that
    points at the public repository. When none here holds the commit, the
    answer is about the clone, not the results: NOT CHECKED, not FAIL.
  * **Constants and directions.** Appendix A, as this process loads it, is
    the one at prereg-freeze (run_oos's guard 5). The directions are
    results/insample_v2.csv's at the tag, and the MDEs those of the freeze
    amendment in the run's own commit; the two must agree.
  * **Provenance.** The results file was committed once, and the copy on
    disk is that commit's, as are the manifests and (for the primary set)
    the secondary results that the computation reads from disk. A test
    set's code_commit and manifest_sha256 columns are checked for what they
    must equal, with run_oos's own check for the secondary sets (guard 9):
    one commit on every row, which descends from the tag, has the tag's code
    and Appendix A, and comes before the commit that added the file; and the
    SHA-256 of the set's manifest in that commit, which must also be the
    manifest on disk. The recomputation is handed those two values, so in
    the comparison they match by construction; this check is what vouches
    for them. A failure here, too, stops the set.
  * **Code.** Every module of this checkout that the process loaded is byte
    for byte the file at prereg-freeze: research/, quant_core/factor_engine.py
    (whose MACD the factor calls), scripts/run_oos.py and, on the primary
    set, scripts/factor_research.py (whose statistic R2 reports). Each must
    be among the files checked, and a file that is a symbolic link is
    checked by what it points to, wherever that is. No module of the
    repository's packages may come from another directory.
  * **Numbers.** Every column of the committed file (and of the robustness
    file, for the primary set) against the recomputation. Keys, integers,
    flags and text must be equal; floats must agree to a relative 1e-9,
    which forgives formatting and summation order, never a changed value.
    run_oos's guard 9 also accepted two things more: an absolute 1e-9, for
    values that should be zero, and the text NaN for a blank. Here a pair
    accepted only that way is TOLERATED, shown with both values, and the
    verdict is then TOLERATED, not PASS: on a p-value of 1e-12 that floor
    would accept anything from -1e-9 to 1e-9, and the run never writes NaN.

Left out are the guards that only stop a second committed run: a clean
checkout of main (replaced by the narrower checks above: the files the
computation reads are the committed ones, the modules it runs are the
tag's), outputs that must not exist yet, and one run per set. So is guard 6,
which only stops a run of doge_fwd before its window has closed. Also left
out are research.freeze.verify_freeze's questions about the author's remote
(origin is the public repository, the amendment is on its main branch); a
reader answers those by cloning from it.

Without git history (an archive download, a shallow clone, replace refs
or grafts, or no prereg-freeze tag) the provenance, constants and code
checks cannot be made. They are reported as NOT CHECKED, with what would
make them possible, and the overall verdict is then INCOMPLETE, never PASS.
FAIL is kept for a check that was made and found something wrong.

A reader's download of the in-sample year (``--insample-from``). Appendix A's
in-sample candles came from an earlier downloader: a download of the same
year by fetch_v2.py matches them value for value but not byte for byte, and
the file is not in the repository, since OKX's API Agreement (§9.4) does not
allow its market data to be redistributed. With --insample-from DIR, a
download in fetch_v2's layout (DIR/<instId>/<file>) is read wherever
insample_v2, or doge_fwd's warm-up, would read an in-sample file: the
candles, and the supplement's spot leg and funding, each supplement file
only over the span its manifest records for the run's file. Each must be
there, the candles of both legs must cover the in-sample year's warm-up and
window, and the funding must hold as many settlements over that span as the
run's file did, or the set FAILs before any computation. A file with the
bytes the run read (Appendix A's SHA-256 for the candles,
results/manifest_is_supp.json's for the others) passes as usual. One with
other bytes stands in by value: if every column matches, the committed
results follow from that download through the tag's code, but nothing shows
that the run read it, so its check is NOT CHECKED and the set INCOMPLETE at
best.

Usage:
    python scripts/verify_results.py --set insample_v2
    python scripts/verify_results.py --set doge_2023
    python scripts/verify_results.py --set all
    python scripts/verify_results.py --set insample_v2 --insample-from data/history/insample_fresh

Exit status: 0 PASS, 1 FAIL, 2 usage error, 3 INCOMPLETE, 4 TOLERATED.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock


def _find_repo_root(start: Path) -> Path:
    """
    The nearest ancestor holding research/pipeline.py and the pre-registration.

    Walked up to, rather than fixed as parent.parent, so the same file works
    from scripts/ and from wherever a reader keeps a copy inside the checkout.
    """
    for p in start.parents:
        if (p / "research" / "pipeline.py").is_file() and (p / "docs" / "preregistration.md").is_file():
            return p
    raise SystemExit(f"{start} is not inside a checkout of the study (no research/pipeline.py above it)")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
# This file under both of its names, since it is left out of the code check.
THIS_FILE = {Path(__file__).resolve(), Path(os.path.abspath(__file__))}

# Third-party packages load first, from their usual bytecode caches; they are
# not this checkout's code (numba, pandas and yaml are needed by research and
# quant_core.factor_engine). Then, as in the run scripts, this checkout
# goes first on sys.path, and its modules are compiled from source: the
# bytecode cache is pointed below os.devnull, a file, so no cached .pyc is
# found there and, with dont_write_bytecode, none is written anywhere.
import numba  # noqa: E402
import numba.core.caching  # noqa: E402
import numpy as np  # noqa: E402
import pandas  # noqa: E402,F401
import polars as pl  # noqa: E402
import yaml  # noqa: E402,F401
from loguru import logger  # noqa: E402

sys.path[:] = [str(REPO_ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]
sys.dont_write_bytecode = True
sys.pycache_prefix = os.path.join(os.devnull, "verify_results-pycache")

# The MACD factor calls quant_core.factor_engine's calculate_macd_hist
# unchanged (Appendix A), and that module jits one function with cache=True.
# Decorating it makes numba probe its cache directory: create
# quant_core/__pycache__ if it is missing, then create and delete a temporary
# file there. The function is never called, so no cache is ever read or
# written; the module is imported here, ahead of research.factors' own lazy
# import, with the probe turned off and restored afterwards. (Without the
# private hook, in another numba version, the probe simply happens.)
_probe = getattr(getattr(numba.core.caching, "_CacheLocator", None), "ensure_cache_path", None)
with (mock.patch.object(numba.core.caching._CacheLocator, "ensure_cache_path", lambda self: None)
      if _probe is not None else nullcontext()):
    import quant_core.factor_engine  # noqa: E402,F401

from research import freeze, gitrepo, integrity, pipeline, stats  # noqa: E402
from research import io as research_io  # noqa: E402
from research.io import sha256_file  # noqa: E402
from research.spec import MINUTE_MS, get_spec  # noqa: E402

# The run script's guards and comparison rule, loaded from this checkout.
run_oos = integrity.load_script(REPO_ROOT, "run_oos")

PASS = "PASS"
FAIL = "FAIL"
NOT_CHECKED = "NOT CHECKED"
INCOMPLETE = "INCOMPLETE"
TOLERATED = "TOLERATED"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_INCOMPLETE = 3
EXIT_TOLERATED = 4

INSAMPLE = "insample_v2"           # the in-sample results file's name, which readers know it by
PROVENANCE_COLUMNS = ("code_commit", "manifest_sha256")
REL_TOL = 1e-9                     # guard 9's relative tolerance, without its absolute floor

# The modules every recomputation runs, which must therefore be among the
# files the code check hashes; the primary set adds v1's script (R2).
REQUIRED_CODE = ("research/pipeline.py", "quant_core/factor_engine.py", "scripts/run_oos.py")
REQUIRED_CODE_PRIMARY = ("scripts/factor_research.py",)

NO_HISTORY = "no usable git history (see the git check)"


@dataclass
class Check:
    """One named check and its outcome; ``detail`` says what was compared, or why it failed."""

    name: str
    verdict: str
    detail: str


@dataclass
class ColumnVerdict:
    """One column of a results file against the recomputation."""

    column: str
    kind: str          # key, integer, flag, text, number or provenance
    verdict: str
    detail: str


@dataclass
class SetReport:
    """Everything checked for one set, printed as one block."""

    name: str
    role: str
    checks: List[Check] = field(default_factory=list)
    tables: List[Tuple[str, str, List[ColumnVerdict]]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    seconds: float = 0.0

    def verdicts(self) -> List[str]:
        return [c.verdict for c in self.checks] + [v.verdict for _, _, t in self.tables for v in t]

    def verdict(self) -> str:
        return overall(self.verdicts())


class Stop(Exception):
    """A check failed that makes the rest of the set meaningless; it is already in the report."""


def overall(verdicts: Sequence[str]) -> str:
    """
    FAIL if anything failed, else INCOMPLETE if something could not be
    checked, else TOLERATED if some number passed only by guard 9's looser
    rule, else PASS.
    """
    if FAIL in verdicts:
        return FAIL
    if NOT_CHECKED in verdicts:
        return INCOMPLETE
    if TOLERATED in verdicts:
        return TOLERATED
    return PASS


# ---------------------------------------------------------------- sets

def committed_sets(root: Path) -> List[str]:
    """
    The sets whose results file exists, in the order they can be verified:
    the in-sample year, then the test sets in run_oos's order (the primary
    set reads the replication and pooled sets' statuses, so it comes after
    them).
    """
    order = [pipeline.ROLE_REPLICATION, pipeline.ROLE_POOLED, pipeline.ROLE_PRIMARY, pipeline.ROLE_DESCRIPTIVE]
    names = [INSAMPLE] + [pipeline.set_with_role(r) for r in order]
    return [n for n in names if (root / results_paths(n)[0]).exists()]


def known_sets() -> List[str]:
    return [INSAMPLE] + list(run_oos.oos_sets())


def results_paths(set_name: str) -> Tuple[Path, ...]:
    """The set's committed results files, repo-relative: the main file, then (primary set) the robustness file."""
    if set_name == INSAMPLE:
        return (pipeline.INSAMPLE_RESULTS,)
    return run_oos.outputs_for(set_name)


def pipeline_set(set_name: str) -> str:
    """The name Appendix A gives the set ("IS" for insample_v2)."""
    return pipeline.set_with_role(pipeline.ROLE_IS) if set_name == INSAMPLE else set_name


# ---------------------------------------------------------------- git helpers

def _out(proc: Any) -> str:
    return proc.stdout.decode("utf-8", "replace").strip()


def _short(rev: str) -> str:
    """A commit hash abbreviated for display; a ref name (HEAD) as it is."""
    return rev[:12] if gitrepo.FULL_SHA1.fullmatch(rev) else rev


def worktree_problem(root: Path, rel: str, rev: str) -> Optional[str]:
    """
    Why the file on disk at ``rel`` is not the blob ``rev`` holds there; None if it is.

    The bytes on disk are hashed with ``git hash-object --no-filters``, so no
    clean filter, attribute or stat cache answers for them.
    """
    if not (root / rel).is_file():
        return f"{rel} is missing"
    want = gitrepo.git(root, "rev-parse", "--verify", "--quiet", f"{rev}:{rel}", check=False)
    if want.returncode != 0:
        return f"{rel} is not in {_short(rev)}"
    disk = _out(gitrepo.git(root, "hash-object", "--no-filters", "--", rel))
    if disk != _out(want):
        return f"{rel} on disk differs from {_short(rev)}'s"
    return None


def git_state(root: Path) -> Tuple[Optional[str], List[Check]]:
    """
    The commit prereg-freeze resolves to, or None when provenance cannot be
    checked, with the checks that decided it: a work tree, a whole history
    with no substituted objects (run_oos's guard 2), and the tag.

    Each of those is about the clone, not the results, so a missing one is
    NOT CHECKED (with how to supply it) and the sets are then checked as
    without git: the verdict can be INCOMPLETE at best, never PASS.
    """
    tag = get_spec()["freeze"]["tag"]
    cannot = "the files' history, the code_commit column and the code cannot be checked against the tag"
    if not gitrepo.is_work_tree(root):
        return None, [Check("git", NOT_CHECKED, f"{root} is not a git work tree (an archive download?), so {cannot}")]
    try:
        run_oos.check_history(root)
    except run_oos.Refusal as exc:
        return None, [Check("git", NOT_CHECKED, f"{exc}; until then {cannot}")]
    commit = gitrepo.tag_commit(root, tag)
    if commit is None:
        return None, [Check("git", NOT_CHECKED, f"there is no tag {tag} here (fetch the tags: git fetch --tags), "
                                                f"so {cannot}")]
    return commit, [Check("git", PASS, f"whole history, no replace refs or grafts; {tag} is {commit[:12]}")]


def spec_check(root: Path, tag_commit: Optional[str]) -> Check:
    """
    run_oos's guard 5: Appendix A as loaded, and at HEAD, is the tag's, and
    the files frozen before the tag are unchanged.
    """
    if tag_commit is None:
        return Check("constants", NOT_CHECKED, f"Appendix A cannot be compared with the tag's: {NO_HISTORY}")
    try:
        run_oos.check_spec_frozen(root, tag_commit)
    except run_oos.Refusal as exc:
        return Check("constants", FAIL, str(exc))
    return Check("constants", PASS, "Appendix A as loaded and at HEAD equals prereg-freeze's; the directions "
                                    "file and the in-sample supplement's manifest are unchanged since the tag")


# ---------------------------------------------------------------- the checks of one set

def public_main_refs(root: Path) -> List[str]:
    """
    refs/remotes/<remote>/main for every remote whose fetch URL is the
    public repository (research.freeze.is_public_remote, which knows its
    spellings), whatever the remote is called.
    """
    out = []
    for remote in _out(gitrepo.git(root, "remote")).split():
        proc = gitrepo.git(root, "remote", "get-url", remote, check=False)
        if proc.returncode == 0 and freeze.is_public_remote(_out(proc) or None, root):
            out.append(f"refs/remotes/{remote}/main")
    return out


def _pushed_check(rel: str, added: Optional[str], holders: Sequence[str], public: Sequence[str]) -> Check:
    """Whether a public main branch here holds the commit that added the manifest, as a check of its own."""
    if holders:
        return Check("pushed", PASS, f"{rel} was committed in {(added or '?')[:12]}, which {holders[0]} (the "
                                     "public repository's main branch) holds (§10 step 4)")
    found = (f"{', '.join(public)} does not hold it" if public else
             f"no remote here points at {freeze.PUBLIC_REMOTE}")
    return Check("pushed", NOT_CHECKED,
                 f"{rel} was committed in {(added or '?')[:12]}, but {found}, so this clone cannot show that the "
                 "manifest reached the public main branch before any analysis (§10 step 4). Clone from the "
                 "public repository, or add it as a remote and fetch it")


def inputs_check(root: Path, set_name: str, git_ok: bool) -> List[Check]:
    """
    The input files against the committed hashes: verify_inputs for the
    in-sample year (run_insample_v2's check); run_oos's guard 8 for a test
    set, without its git questions when there is no git history.

    Guard 8 asks freeze.REMOTE_MAIN (origin/main) whether it holds the
    commit that added the manifest. Here it is asked of the main branch of
    any remote that points at the public repository, with the answer as a
    check of its own ("pushed"); when none holds the commit, guard 8 runs
    with that one question answered by the commit itself, and "pushed" is
    NOT CHECKED. The rest of guard 8 runs unchanged.
    """
    name = pipeline_set(set_name)
    present = [f for f in pipeline.input_files(name, root) if f.path.exists()]
    listed = ", ".join(f.path.relative_to(root).as_posix() for f in present) or "none"
    required = [m.as_posix() for m in pipeline.manifests_required(name)]
    manifests = ", ".join(required)
    if set_name == INSAMPLE:
        problems = pipeline.verify_inputs(name, root)
        if problems:
            return [Check("inputs", FAIL, "; ".join(problems))]
        return [Check("inputs", PASS, f"{len(present)} files ({listed}); the candles have Appendix A's SHA-256, "
                                      f"the others {manifests}'s, and no unlisted file is read")]
    rel = pipeline.oos_manifest_path(name).as_posix()
    # Guard 8's questions are about the set's own manifest. doge_fwd reads the
    # in-sample files too, for its warm-up (§4.1), and those have Appendix A's
    # SHA-256 (the candles) or the in-sample supplement's.
    if required == [rel]:
        hashes = "its SHA-256s and cover the set's window"
    else:
        hashes = (f"the SHA-256s that Appendix A (the in-sample candles), {', '.join(required[:-1])} and "
                  f"{required[-1]} record, and those {rel} lists cover the set's window")
    extra: List[Check] = []
    try:
        if git_ok:
            added = run_oos._first_added(root, rel)
            public = public_main_refs(root)
            holders = []
            for ref in public if added is not None else ():
                tip = gitrepo.commit_of(root, ref)
                if tip is not None and gitrepo.is_ancestor(root, added, tip):
                    holders.append(ref)
            stand_in = holders[0] if holders else (added or freeze.REMOTE_MAIN)
            with mock.patch.object(freeze, "REMOTE_MAIN", stand_in):
                run_oos.check_manifest(root, name)
            extra.append(_pushed_check(rel, added, holders, public))
            how = (f"{rel} was committed once, after the freeze amendment, records the public repository "
                   "and covers the window; ")
        else:
            run_oos._inputs_match_and_cover(root, name, run_oos._manifest_window_and_unlock(root, name, rel))
            how = f"{rel} covers the window (its history not checked: {NO_HISTORY}); "
    except run_oos.Refusal as exc:
        return [Check("inputs", FAIL, str(exc))]
    return [Check("inputs", PASS, f"{how}{len(present)} files ({listed}) have {hashes}; no unlisted file is "
                                  "read")] + extra


def frozen_directions(root: Path, tag_commit: Optional[str], amendment_rev: Optional[str]) -> Tuple[
        Dict[freeze.CellKey, Tuple[int, bool]], Dict[freeze.CellKey, float], Check]:
    """
    The directions and MDEs a test set's run used, and the check that they agree.

    As in run_oos: the directions from results/insample_v2.csv in the tagged
    commit, the MDEs from the freeze amendment in the commit the run checked
    out (``amendment_rev``), whose freeze-commit block must name the tag's
    commit. Without git history, both files as they are on disk.
    """
    f = get_spec()["freeze"]
    if tag_commit is not None and amendment_rev is not None:
        insample_text = gitrepo.show_file(root, tag_commit, f["directions_file"])
        amendment_text = gitrepo.show_file(root, amendment_rev, f["amendment_file"])
        where = f"{f['directions_file']} at {f['tag']} and {f['amendment_file']} at {amendment_rev[:12]}"
    else:
        insample_text = (root / f["directions_file"]).read_text(encoding="utf-8")
        amendment_text = (root / f["amendment_file"]).read_text(encoding="utf-8")
        where = f"{f['directions_file']} and {f['amendment_file']} on disk ({NO_HISTORY})"
    try:
        record = freeze.parse_freeze_amendment(amendment_text)
        insample = freeze.parse_insample_csv(insample_text)
    except freeze.FreezeFormatError as exc:
        return {}, {}, Check("directions", FAIL, f"{where}: {exc}")
    problems = freeze.compare_with_insample(record, insample)
    if tag_commit is not None and record.commit != tag_commit:
        problems.insert(0, f"the amendment records {record.commit[:12]}, but {f['tag']} is {tag_commit[:12]}")
    directions = pipeline.directions_from_insample(insample)
    mde = {k: c.mde_bps for k, c in record.cells.items()}
    if problems:
        return directions, mde, Check("directions", FAIL, f"{where} disagree: " + "; ".join(problems[:5]))
    flips = sum(1 for d, _ in directions.values() if d == -1)
    return directions, mde, Check("directions", PASS, f"{len(directions)} directions ({flips} of them -1) and "
                                                      f"MDEs from {where}, which agree")


def insample_provenance(root: Path, tag_commit: Optional[str]) -> Tuple[str, List[Check]]:
    """
    results/insample_v2.csv's committed text: added by the tagged commit and
    never changed (§10 step 2), and the copy on disk is that one. Without
    git, the file on disk, unvouched for.
    """
    rel = pipeline.INSAMPLE_RESULTS.as_posix()
    if tag_commit is None:
        return (root / rel).read_text(encoding="utf-8"), [
            Check("provenance", NOT_CHECKED, f"{rel} is compared as it is on disk; with {NO_HISTORY}, "
                                             "nothing shows it is the committed file")]
    commits = gitrepo.commits_touching(root, rel, all_refs=True)
    if commits != [tag_commit]:
        return "", [Check("provenance", FAIL, f"{rel} was committed in {[c[:12] for c in commits]}; §10 step 2 "
                                              f"adds it once, in the tagged commit {tag_commit[:12]}")]
    problems = [p for p in (worktree_problem(root, rel, tag_commit),
                            worktree_problem(root, pipeline.IS_SUPP_MANIFEST.as_posix(), "HEAD")) if p]
    text = gitrepo.show_file(root, tag_commit, rel)
    if problems:
        return text, [Check("provenance", FAIL, "; ".join(problems))]
    return text, [Check("provenance", PASS, f"{rel} was committed once, by {tag_commit[:12]} (prereg-freeze); the "
                                            f"copy on disk and {pipeline.IS_SUPP_MANIFEST.as_posix()} are the "
                                            "committed ones")]


@dataclass
class OosProvenance:
    """What a test set's committed results say about their own run, once checked."""

    texts: Tuple[str, ...]           # when the check passes: the committed main file, then the robustness file
    code_commit: str
    manifest_sha256: str
    checks: List[Check]
    column_notes: Dict[str, Tuple[str, str]]   # provenance column -> (verdict, what it was checked to equal)


def oos_provenance(root: Path, set_name: str, tag_commit: Optional[str]) -> OosProvenance:
    """
    The committed results of a test set and the run they record.

    With git, run_oos._secondary_provenance (guard 9's metadata check, which
    applies to any test set): committed once, one code_commit descending
    from the tag with its code and Appendix A, strictly before the commit
    that added the file, and manifest_sha256 the SHA-256 of the set's
    manifest in that commit. Then: the robustness file (primary set) came in
    the same commit, the copies on disk are the committed ones, and so are
    the manifests and, for the primary set, the secondary results its labels
    are read from. Without git history, the files on disk; code_commit cannot be
    checked, manifest_sha256 is compared with the manifest on disk.
    """
    rels = [p.as_posix() for p in results_paths(set_name)]
    manifest_rel = pipeline.oos_manifest_path(set_name).as_posix()
    disk_digest = sha256_file(root / manifest_rel)
    if tag_commit is None:
        texts = tuple((root / r).read_text(encoding="utf-8") for r in rels)
        df = pl.read_csv(root / rels[0], infer_schema_length=0)
        codes, shas = set(df["code_commit"].to_list()), set(df["manifest_sha256"].to_list())
        code = next(iter(codes)) if len(codes) == 1 else ""
        sha = next(iter(shas)) if len(shas) == 1 else ""
        sha_verdict = PASS if shas == {disk_digest} else FAIL
        notes = {
            "code_commit": (NOT_CHECKED, f"{len(codes)} value(s) {sorted(map(str, codes))[:2]}; with {NO_HISTORY} "
                                         "it cannot be compared with the tag"),
            "manifest_sha256": (sha_verdict, f"{len(shas)} value(s); must be the SHA-256 of {manifest_rel} on "
                                             f"disk, {disk_digest[:16]}..."),
        }
        checks = [Check("provenance", NOT_CHECKED, f"{', '.join(rels)} compared as they are on disk; with "
                                                   f"{NO_HISTORY}, nothing shows they are the committed files")]
        return OosProvenance(texts, code, sha, checks, notes)

    frozen = run_oos.frozen_spec(root, tag_commit)
    try:
        text, code, digest = run_oos._secondary_provenance(root, rels[0], set_name, tag_commit, frozen)
    except run_oos.Refusal as exc:
        fail = (FAIL, f"see provenance: {exc}")
        return OosProvenance((), "", "", [Check("provenance", FAIL, str(exc))],
                             {"code_commit": fail, "manifest_sha256": fail})
    added = gitrepo.commits_touching(root, rels[0], all_refs=True)[0]
    problems: List[str] = []
    texts = [text]
    for rel in rels[1:]:
        commits = gitrepo.commits_touching(root, rel, all_refs=True)
        if commits != [added]:
            # Its text is not read: that commit may not hold the file (git show
            # would fail before this is reported), and the set stops here.
            problems.append(f"{rel} was committed in {[c[:12] for c in commits]}, not once with {rels[0]} "
                            f"in {added[:12]}")
            continue
        texts.append(gitrepo.show_file(root, added, rel))
    problems += [p for p in (worktree_problem(root, r, added) for r in rels) if p]
    if digest != disk_digest:
        problems.append(f"{manifest_rel} on disk hashes to {disk_digest}, not to its version in {code[:12]} "
                        f"({digest})")
    reads = [m.as_posix() for m in pipeline.manifests_required(set_name)]
    if pipeline.set_role(set_name) == pipeline.ROLE_PRIMARY:
        # run_oos_set reads the two secondary sets' statuses from disk.
        reads += [pipeline.oos_results_path(pipeline.set_with_role(r)).as_posix()
                  for r in (pipeline.ROLE_REPLICATION, pipeline.ROLE_POOLED)]
    problems += [p for p in (worktree_problem(root, r, "HEAD") for r in reads) if p]

    manifest_commit = (gitrepo.commits_touching(root, manifest_rel, added_only=True) or [""])[-1]
    then = "the commit that added the set's manifest" if manifest_commit == code else "a later commit"
    after = "the parent" if gitrepo.parents(root, added) == [code] else "an ancestor"
    notes = {
        "code_commit": (PASS, f"{code[:12]} on every row: {then}; descends from prereg-freeze with only docs/ and "
                              f"results/ changed and the same Appendix A; {after} of {added[:12]}, which added "
                              "the file"),
        "manifest_sha256": (PASS if digest == disk_digest else FAIL,
                            f"{digest[:16]}... on every row: the SHA-256 of {manifest_rel} in {code[:12]}, and of "
                            "the file on disk"),
    }
    if problems:
        return OosProvenance(tuple(texts), code, digest, [Check("provenance", FAIL, "; ".join(problems))], notes)
    return OosProvenance(tuple(texts), code, digest, [Check(
        "provenance", PASS, f"{', '.join(rels)} committed once, in {added[:12]}; the copies on disk, and "
                            f"{', '.join(reads)}, are the committed ones")], notes)


def loaded_checkout_files(root: Path, required: Sequence[str] = ()) -> List[str]:
    """
    Repo-relative paths of the modules this process loaded from the
    checkout, this file left out (it is not part of the frozen code).

    A module is placed by the path it was loaded under, not by where a
    symbolic link leads: a module file of the checkout linked to a file
    elsewhere keeps its name here, and git hash-object, given that name,
    hashes the file the link leads to. (Resolving first would drop such a
    file from the list without a word.) integrity.load_script, which loads
    scripts/run_oos.py, records the resolved path instead, so a module
    whose path is where a ``required`` file of the checkout leads takes
    that file's name. Only then, or for a path outside the checkout that
    leads into it, is a module placed by where it leads.
    """
    loaded = list(sys.modules.values())
    # R2 runs v1's own script, which research.stats loads by path and keeps
    # out of sys.modules; its cached module is asked for directly.
    if stats._v1_module.cache_info().currsize:
        loaded.append(stats._v1_module())
    leads_to = {(root / r).resolve(): r for r in required}
    files = set()
    for mod in loaded:
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        as_loaded, resolved = Path(os.path.abspath(f)), Path(f).resolve()
        if as_loaded in THIS_FILE or resolved in THIS_FILE:
            continue
        if resolved in leads_to:
            files.add(leads_to[resolved])
            continue
        for p in (as_loaded, resolved):
            if p.is_relative_to(root):
                files.add(p.relative_to(root).as_posix())
                break
    return sorted(files)


def code_check(root: Path, tag_commit: Optional[str], required: Sequence[str] = REQUIRED_CODE) -> Check:
    """
    The modules this process loaded from the checkout are the tag's files,
    byte for byte, ``required`` among them, and none of the repository's
    packages came from elsewhere (research.integrity, run_oos's guard 1).
    """
    foreign = integrity.foreign_code_problem(root)
    if foreign:
        return Check("code", FAIL, foreign)
    rels = loaded_checkout_files(root, required)
    missing = [r for r in required if r not in rels]
    if missing:
        return Check("code", FAIL, f"{', '.join(missing)} did not run from this checkout (another copy ran, or "
                                   "none), so nothing here vouches for the code that did; loaded from the "
                                   f"checkout: {', '.join(rels)}")
    if tag_commit is None:
        return Check("code", NOT_CHECKED, f"{len(rels)} modules loaded from this checkout ({', '.join(rels)}); "
                                          f"with {NO_HISTORY} they cannot be compared with the tag")
    listing = gitrepo.git(root, "ls-tree", "-z", "--full-tree", tag_commit, "--", *rels).stdout.decode("utf-8")
    at_tag = {}
    for record in filter(None, listing.split("\0")):
        meta, path = record.split("\t", 1)
        at_tag[path] = meta.split()[2]
    disk = _out(gitrepo.git(root, "hash-object", "--no-filters", "--stdin-paths",
                            input="".join(r + "\n" for r in rels).encode("utf-8"))).split()
    if len(disk) != len(rels):
        raise gitrepo.GitError(f"git hash-object returned {len(disk)} hashes for {len(rels)} files")
    differ = [r for r, h in zip(rels, disk) if at_tag.get(r) != h]
    if differ:
        return Check("code", FAIL, f"loaded modules that are not prereg-freeze's: {', '.join(differ)}. The "
                                   "recomputation would not be the frozen code's (check those files out at the tag)")
    return Check("code", PASS, f"{len(rels)} modules loaded from this checkout, each identical to prereg-freeze's: "
                               + ", ".join(rels))


# ---------------------------------------------------------------- comparison

def csv_text(df: pl.DataFrame) -> str:
    """
    The text the run wrote for ``df``, produced without writing it.

    research.io.write_once_csv decides the format (NaN as a blank field) and
    hands the bytes to write_once_bytes, which creates the file. Standing in
    for that last call keeps the run's serialisation, unchanged, and drops
    only the write.
    """
    captured: List[bytes] = []

    def keep(data: bytes, path: Any) -> Path:
        captured.append(data)
        return Path(path)

    with mock.patch.object(research_io, "write_once_bytes", keep):
        research_io.write_once_csv(df, "not-written.csv")
    return captured[0].decode("utf-8")


def _kind(col: str, dtype: Any, keys: Sequence[str]) -> str:
    if col in keys:
        return "key"
    if col in PROVENANCE_COLUMNS:
        return "provenance"
    if dtype in (pl.Float32, pl.Float64):
        return "number"
    if dtype == pl.Boolean:
        return "flag"
    if dtype in (pl.Int64, pl.Int32):
        return "integer"
    return "text"


# How two number fields compare (number_pair).
SAME = "same"              # equal as text
CLOSE = "close"            # equal to a relative 1e-9
FLOOR = "floor"            # accepted by guard 9 only through its absolute 1e-9 floor
NAN_TEXT = "nan-text"      # both undefined, one written as NaN rather than blank
DIFFERENT = "different"    # refused by guard 9's rule


def _as_number(t: Optional[str]) -> float:
    return math.nan if t is None or t == "" else float(t)


def number_pair(x: Optional[str], y: Optional[str]) -> str:
    """
    How two CSV fields compare as numbers.

    run_oos._same_csv_number, guard 9's rule, decides what is refused
    (DIFFERENT). Of what it accepts, only equal text and a relative 1e-9
    count as agreement; the rest it accepts through its absolute 1e-9 floor
    (FLOOR), meant for values that should be zero, or because it reads the
    text NaN as a blank (NAN_TEXT).
    """
    if x == y:
        return SAME
    if not run_oos._same_csv_number(x, y):
        return DIFFERENT
    u, v = _as_number(x), _as_number(y)
    if math.isnan(u) and math.isnan(v):
        return NAN_TEXT
    if math.isclose(u, v, rel_tol=REL_TOL, abs_tol=0.0):
        return CLOSE
    return FLOOR


def _relative(u: float, v: float) -> float:
    scale = max(abs(u), abs(v))
    return abs(u - v) / scale if scale > 0 and math.isfinite(scale) else 0.0


def _number_verdict(xs: Sequence[Optional[str]], ys: Sequence[Optional[str]]) -> Tuple[str, str]:
    """A numeric column's verdict and detail, given each row's committed and recomputed text."""
    kinds = [number_pair(x, y) for x, y in zip(xs, ys)]
    rows = len(kinds)

    def first(kind: str) -> str:
        i = kinds.index(kind)
        return f"first at line {i + 2}: {xs[i]!r} committed, {ys[i]!r} recomputed"

    bad = kinds.count(DIFFERENT)
    if bad:
        return FAIL, f"{bad} of {rows} rows differ; {first(DIFFERENT)}"
    loose = []
    if FLOOR in kinds:
        worst = max(_relative(_as_number(x), _as_number(y)) for x, y, k in zip(xs, ys, kinds) if k == FLOOR)
        loose.append(f"{kinds.count(FLOOR)} row(s) agree only within guard 9's absolute 1e-9, not to a relative "
                     f"1e-9 (max relative difference {worst:.1e}); {first(FLOOR)}")
    if NAN_TEXT in kinds:
        loose.append(f"{kinds.count(NAN_TEXT)} row(s) hold NaN where the other file has a blank, which the run "
                     f"never writes; {first(NAN_TEXT)}")
    if loose:
        return TOLERATED, "; ".join(loose)
    close = [(_as_number(x), _as_number(y)) for x, y, k in zip(xs, ys, kinds) if k == CLOSE]
    if not close:
        return PASS, "identical"
    finite = [(u, v) for u, v in close if math.isfinite(u) and math.isfinite(v)]
    max_abs = max((abs(u - v) for u, v in finite), default=0.0)
    max_rel = max((_relative(u, v) for u, v in finite), default=0.0)
    return PASS, (f"{len(close)} row{'s' if len(close) > 1 else ''} differ{'' if len(close) > 1 else 's'} only in "
                  f"the last digits (max |Δ| {max_abs:.1e}, max relative {max_rel:.1e})")


def compare(committed: str, recomputed: pl.DataFrame, keys: Sequence[str],
            provenance: Optional[Dict[str, Tuple[str, str]]] = None) -> Tuple[str, List[ColumnVerdict]]:
    """
    A committed results file against a recomputation, column by column.

    Returns a one-line summary and a verdict per column. Numbers must agree
    to a relative 1e-9 (number_pair; what guard 9 accepted beyond that is
    TOLERATED); everything else must be equal as text. Provenance columns
    take their verdict from the provenance check: the recomputation was
    handed the checked values, so their equality here proves nothing by
    itself.
    """
    text = csv_text(recomputed)
    a = pl.read_csv(committed.encode("utf-8"), infer_schema_length=0)
    b = pl.read_csv(text.encode("utf-8"), infer_schema_length=0)
    shape = f"{a.height} rows x {a.width} columns"
    if a.columns != b.columns or a.height != b.height:
        return shape, [ColumnVerdict("(shape)", "-", FAIL,
                                     f"committed {a.height} x {a.width}, recomputed {b.height} x {b.width}"
                                     + ("" if a.columns == b.columns else "; the columns differ"))]
    dtypes = dict(zip(recomputed.columns, recomputed.dtypes))
    out: List[ColumnVerdict] = []
    for c in a.columns:
        kind = _kind(c, dtypes[c], keys)
        xs, ys = a[c].to_list(), b[c].to_list()
        if kind == "number":
            verdict, detail = _number_verdict(xs, ys)
            out.append(ColumnVerdict(c, kind, verdict, detail))
            continue
        bad = [(i, x, y) for i, (x, y) in enumerate(zip(xs, ys)) if x != y]
        if bad:
            i, x, y = bad[0]
            out.append(ColumnVerdict(c, kind, FAIL, f"{len(bad)} of {a.height} rows differ; first at line {i + 2}: "
                                                    f"{x!r} committed, {y!r} recomputed"))
        elif kind == "provenance" and provenance and c in provenance:
            out.append(ColumnVerdict(c, kind, *provenance[c]))
        else:
            out.append(ColumnVerdict(c, kind, PASS, "identical"))
    summary = shape + (", byte-identical to the recomputation" if text == committed else
                       ", not byte-identical; see the columns")
    return summary, out


# ---------------------------------------------------------------- a reader's download of the in-sample year

FRESH_CHECK = "in-sample files"    # the check of a reader's download (--insample-from)


def is_insample_file(f: Any, root: Path, spec: Optional[Dict[str, Any]] = None) -> bool:
    """Whether ``f``, a pipeline.InputFile, is an in-sample file: Appendix A's candles or the supplement's."""
    root = Path(root)
    return (f.path.parent == root / (spec or get_spec())["insample_dir"]
            or f.path.parent.parent == root / pipeline.IS_SUPP_DIR)


def fresh_path(f: Any, fresh: Path) -> Path:
    """Where fetch_v2 puts the series of ``f`` under ``fresh``: <out-dir>/<instId>/<file>."""
    return Path(fresh) / f.path.parent.name / f.path.name


def reads_insample(set_name: str, root: Path) -> bool:
    """Whether the set reads in-sample files: the in-sample year, and doge_fwd for its warm-up."""
    return any(is_insample_file(f, root) for f in pipeline.input_files(pipeline_set(set_name), root))


def supplement_records(root: Path) -> Dict[Path, Dict[str, Any]]:
    """The in-sample supplement's manifest by file: what the run read of each (SHA-256, rows, first and last)."""
    series = json.loads((Path(root) / pipeline.IS_SUPP_MANIFEST).read_text(encoding="utf-8"))["series"]
    return {Path(root) / e["path"]: e for e in series}


def spanned(f: Any, record: Optional[Dict[str, Any]]) -> Any:
    """``f`` read only over the span its manifest ``record`` says the run's file covered (all of it without one)."""
    if record is None:
        return f
    lo, hi = int(record["first_ts"]), int(record["last_ts"]) + 1
    return replace(f, lo_ms=lo if f.lo_ms is None else max(f.lo_ms, lo),
                   hi_ms=hi if f.hi_ms is None else min(f.hi_ms, hi))


@contextmanager
def fresh_insample(fresh: Optional[Path], mode: str) -> Any:
    """
    research.pipeline as the run had it when ``fresh`` is None. Otherwise
    every in-sample file a set would read comes from ``fresh`` instead (mode
    "read"), a supplement file only over the span its manifest records, so
    that a download reaching further is read as the run read its file. Or
    the in-sample files are left out of the checks, with the supplement's
    manifest (mode "check"), so that the checks of the set's other files run
    unchanged. Both replace functions the pipeline itself calls, so the code
    that loads and checks the files is the run's.
    """
    if fresh is None:
        yield
        return
    files, manifests = pipeline.input_files, pipeline.manifests_required

    def input_files(set_name, root=pipeline.REPO_ROOT, *, spec=None, is_end_ms=None):
        out, records = [], None
        for f in files(set_name, root, spec=spec, is_end_ms=is_end_ms):
            if is_insample_file(f, root, spec):
                if mode == "check":
                    continue
                records = supplement_records(root) if records is None else records
                f = spanned(replace(f, path=fresh_path(f, fresh)), records.get(f.path))
            out.append(f)
        return out

    def manifests_required(set_name, spec=None):
        return [m for m in manifests(set_name, spec) if m != pipeline.IS_SUPP_MANIFEST]

    with mock.patch.object(pipeline, "input_files", input_files), \
            (mock.patch.object(pipeline, "manifests_required", manifests_required) if mode == "check"
             else nullcontext()):
        yield


def _utc(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


def candles_problem(ts: Any, window: Any) -> Optional[str]:
    """Why candles at the minutes ``ts`` (a polars Series) cannot stand in for the in-sample year's, or None."""
    need = (window.warmup_start_ms, window.end_ms - MINUTE_MS)
    if ts.len() and ts.min() <= need[0] and ts.max() >= need[1]:
        return None
    have = f"{_utc(ts.min())} to {_utc(ts.max())}" if ts.len() else "no rows"
    return f"covers {have}, and the in-sample year needs {_utc(need[0])} to {_utc(need[1])} UTC"


def funding_problem(ts: Any, record: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why settlements at ``ts`` cannot stand in for the run's funding file as its manifest records it, or None."""
    if record is None:
        return "has no counterpart in the supplement's manifest"
    lo, hi, rows = int(record["first_ts"]), int(record["last_ts"]), int(record["rows"])
    held = int(((ts >= lo) & (ts <= hi)).sum())
    if held == rows:
        return None
    return (f"holds {held} settlements from {_utc(lo)} to {_utc(hi)} UTC, where the run's file held {rows} (OKX's "
            "REST funding history reaches back about three months; fetch_v2.py --allow-test-data takes the "
            "months before that from OKX's archive)")


def fresh_check(root: Path, fresh: Path) -> Check:
    """
    A reader's download against the in-sample files it stands in for. Each
    must be there; the candles of both legs must cover the in-sample year's
    warm-up and window; and the funding must hold as many settlements, over
    the span the supplement's manifest records, as the run's file did (FAIL
    otherwise). Each file's SHA-256 is then compared with the one the run
    read, Appendix A's for the candles and the supplement manifest's for the
    rest: a file with other bytes stands in by value, which is NOT CHECKED.
    """
    sp = get_spec()
    window = pipeline.window_of(pipeline_set(INSAMPLE), sp)
    supp = pipeline.IS_SUPP_MANIFEST.as_posix()
    try:
        records = supplement_records(root)
        known = {p: (str(e["sha256"]), f"{supp}'s") for p, e in records.items()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return Check(FRESH_CHECK, FAIL, f"{supp} cannot be read: {exc}")
    known.update({root / sp["insample_dir"] / name: (str(digest), "Appendix A's")
                  for name, digest in sp["insample_sha256"].items()})
    same: List[str] = []
    other: List[str] = []
    problems: List[str] = []
    for f in pipeline.input_files(pipeline_set(INSAMPLE), root):
        if not is_insample_file(f, root, sp):
            continue
        src, rel = fresh_path(f, fresh), f.path.relative_to(root).as_posix()
        if not src.is_file():
            problems.append(f"{src} does not exist; it would stand in for {rel}")
            continue
        try:
            ts = pl.read_parquet(src, columns=["timestamp"])["timestamp"].cast(pl.Int64)
            problem = (funding_problem(ts, records.get(f.path)) if f.kind == "funding"
                       else candles_problem(ts, window))
        except Exception as exc:  # not a file fetch_v2 writes, whatever polars calls the failure
            problem = f"cannot be read: {exc}"
        if problem:
            problems.append(f"{src} {problem}; it would stand in for {rel}")
            continue
        digest, whose = known.get(f.path, (None, "no record's"))
        (same if sha256_file(src) == digest else other).append(f"{rel} ({whose} SHA-256)")
    if problems:
        return Check(FRESH_CHECK, FAIL, "; ".join(problems))
    head = f"read from {fresh}"
    if not other:
        return Check(FRESH_CHECK, PASS, f"{head}, each with the bytes the run read: {', '.join(same)}")
    return Check(FRESH_CHECK, NOT_CHECKED,
                 f"{head}. " + (f"With the bytes the run read: {', '.join(same)}. " if same else "")
                 + f"With other bytes: {', '.join(other)}. Those stand in by value: matching columns show that the "
                   "committed results follow from this download through the tag's code, not that the run read it")


def fresh_inputs_check(root: Path, set_name: str, git_ok: bool, fresh: Path) -> List[Check]:
    """
    inputs_check with the in-sample files read from ``fresh``: the set's own
    files as usual (doge_fwd's, guard 8 and all; the in-sample year has no
    others), then the download.
    """
    own: List[Check] = []
    if set_name != INSAMPLE:
        with fresh_insample(fresh, "check"):
            own = inputs_check(root, set_name, git_ok)
        if any(c.verdict == FAIL for c in own):
            return own
    return own + [fresh_check(root, fresh)]


# ---------------------------------------------------------------- one set

def verify_set(root: Path, set_name: str, tag_commit: Optional[str], git_ok: bool,
               fresh: Optional[Path] = None) -> SetReport:
    """
    Every check of one set, in order; a failed input or provenance check
    stops before the computation. ``fresh``, a reader's download of the
    in-sample year (--insample-from), is read in place of the in-sample files.
    """
    started = time.monotonic()
    name = pipeline_set(set_name)
    report = SetReport(set_name, pipeline.set_role(name))
    try:
        _verify_into(report, root, set_name, tag_commit, git_ok, fresh)
    except Stop:
        pass
    except (run_oos.Refusal, pipeline.PipelineError, gitrepo.GitError, OSError) as exc:
        # A missing or unreadable file, or git failing, outside the checks
        # that word their own refusal: the set cannot be verified.
        report.checks.append(Check("error", FAIL, f"{type(exc).__name__}: {exc}"))
    unchecked = [c.name for c in report.checks if c.verdict == NOT_CHECKED]
    if report.verdict() == INCOMPLETE and unchecked == [FRESH_CHECK]:
        report.notes.append(f"INCOMPLETE only because in-sample files stood in by value (see {FRESH_CHECK})")
    report.seconds = time.monotonic() - started
    return report


def _require(report: SetReport, checks: Sequence[Check]) -> None:
    report.checks += checks
    if any(c.verdict == FAIL for c in checks):
        raise Stop()


def _verify_into(report: SetReport, root: Path, set_name: str, tag_commit: Optional[str], git_ok: bool,
                 fresh: Optional[Path] = None) -> None:
    name = pipeline_set(set_name)
    rels = [p.as_posix() for p in results_paths(set_name)]
    if not (root / rels[0]).exists():
        _require(report, [Check("results", FAIL, f"{rels[0]} does not exist; this set has no committed results yet")])
    _require(report, inputs_check(root, set_name, git_ok) if fresh is None
             else fresh_inputs_check(root, set_name, git_ok, fresh))

    if set_name == INSAMPLE:
        text, checks = insample_provenance(root, tag_commit)
        _require(report, checks)
        logger.info(f"{set_name}: recomputing (research.pipeline.run_insample)")
        with fresh_insample(fresh, "read"):
            df = pipeline.run_insample(root)
        report.checks.append(code_check(root, tag_commit, REQUIRED_CODE))
        summary, table = compare(text, df, ("bar", "horizon", "factor"))
        report.tables.append((rels[0], summary, table))
        return

    prov = oos_provenance(root, name, tag_commit)
    _require(report, prov.checks)
    directions, mde, check = frozen_directions(root, tag_commit, prov.code_commit if tag_commit else None)
    _require(report, [check])
    logger.info(f"{set_name}: recomputing (research.pipeline.run_oos_set)")
    with fresh_insample(fresh, "read"):
        main_df, robust_df = pipeline.run_oos_set(name, directions, root=root, mde_bps=mde,
                                                  code_commit=prov.code_commit,
                                                  manifest_sha256=prov.manifest_sha256)
    primary = pipeline.set_role(name) == pipeline.ROLE_PRIMARY
    report.checks.append(code_check(root, tag_commit, REQUIRED_CODE + (REQUIRED_CODE_PRIMARY if primary else ())))
    summary, table = compare(prov.texts[0], main_df, pipeline.KEY_COLUMNS, prov.column_notes)
    report.tables.append((rels[0], summary, table))
    if len(rels) > 1:
        if robust_df is None:
            report.tables.append((rels[1], "not recomputed",
                                  [ColumnVerdict("(file)", "-", FAIL, "run_oos_set returned no robustness rows")]))
        else:
            summary, table = compare(prov.texts[1], robust_df, ("bar", "horizon", "factor", "row", "parameter"))
            report.tables.append((rels[1], summary, table))
    if primary:
        others = ", ".join(pipeline.set_with_role(r) for r in (pipeline.ROLE_REPLICATION, pipeline.ROLE_POOLED))
        report.notes.append(f"replicated and consistent_on_majors were computed from the statuses in the {others} "
                            "results on disk, as in the run; those files are verified by their own --set")


# ---------------------------------------------------------------- output

def print_common(checks: Sequence[Check]) -> None:
    print("== common")
    print(f"  python {platform.python_version()}, numpy {np.__version__}, polars {pl.__version__}, "
          f"numba {numba.__version__}")
    for c in checks:
        print(f"  [{c.verdict}] {c.name}: {c.detail}")


def print_report(r: SetReport) -> None:
    print(f"\n== {r.name} ({r.role})")
    for c in r.checks:
        print(f"  [{c.verdict}] {c.name}: {c.detail}")
    for rel, summary, table in r.tables:
        print(f"  {rel}: {summary}")
        width = max([len(v.column) for v in table] + [6])
        print(f"    {'column':<{width}}  {'kind':<10}  {'verdict':<11}  detail")
        for v in table:
            print(f"    {v.column:<{width}}  {v.kind:<10}  {v.verdict:<11}  {v.detail}")
    for n in r.notes:
        print(f"  note: {n}")
    print(f"  {r.name}: {r.verdict()} ({r.seconds:.0f} s)")


# ---------------------------------------------------------------- CLI

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute committed results from their inputs and compare every column; writes nothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--set", required=True, dest="set_name",
                   help="one of: " + ", ".join(known_sets()) + ", or all (every set with committed results)")
    p.add_argument("--insample-from", type=Path, metavar="DIR", dest="insample_from",
                   help="a download of the in-sample year by fetch_v2.py, read in place of the in-sample files "
                        "(insample_v2, and doge_fwd's warm-up); a file whose bytes differ from the run's is "
                        "checked by value, and the verdict is then INCOMPLETE at best")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None, *, root: Path = REPO_ROOT) -> int:
    args = parse_args(argv)
    if args.set_name == "all":
        names = committed_sets(root)
    elif args.set_name in known_sets():
        names = [args.set_name]
    else:
        print(f"--set must be one of {', '.join(known_sets())} or all", file=sys.stderr)
        return EXIT_USAGE
    fresh: Optional[Path] = args.insample_from
    readers = [n for n in names if reads_insample(n, root)] if fresh is not None else []
    if fresh is not None:
        if not fresh.is_dir():
            print(f"--insample-from: {fresh} is not a directory", file=sys.stderr)
            return EXIT_USAGE
        if not readers:
            print(f"--insample-from: --set {args.set_name} reads no in-sample file "
                  f"(only {', '.join(n for n in known_sets() if reads_insample(n, root))} do)", file=sys.stderr)
            return EXIT_USAGE
        fresh = fresh.resolve()

    tag_commit, common = git_state(root)
    git_ok = tag_commit is not None
    common.append(spec_check(root, tag_commit))
    print_common(common)
    if any(c.verdict == FAIL for c in common):
        print(f"\noverall: {FAIL} (before any set)")
        return EXIT_FAIL

    reports = []
    for n in names:
        r = verify_set(root, n, tag_commit, git_ok, fresh=fresh if n in readers else None)
        print_report(r)
        reports.append(r)

    verdict = overall([c.verdict for c in common] + [v for r in reports for v in r.verdicts()])
    print(f"\noverall: {verdict} (" + ", ".join(f"{r.name} {r.verdict()}" for r in reports) + ")")
    return {PASS: EXIT_PASS, FAIL: EXIT_FAIL, INCOMPLETE: EXIT_INCOMPLETE, TOLERATED: EXIT_TOLERATED}[verdict]


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    raise SystemExit(main())
