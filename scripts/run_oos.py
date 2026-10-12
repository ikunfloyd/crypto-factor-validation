#!/usr/bin/env python3
"""
The one out-of-sample run per test set (pre-registration §10, step 5).

A confirmatory test is only as good as the proof that its rules were fixed
before its data was seen. This script therefore spends most of its lines
refusing to run. Each guard below is one sentence of §10 step 5 (or of the
protocol around it), checked literally, and the first one that fails stops
the run before the set's data is analysed (guard 8 only hashes the input
files and reads their timestamps; guard 9 recomputes the two secondary sets,
whose data was analysed by their own runs):

  1. What runs is this checkout's committed code ("from a clean checkout of
     main"). Every loaded module of the repository's packages comes from
     this checkout: REPO_ROOT goes first on sys.path, and ``python
     scripts/run_oos.py`` compiles them from source into a fresh bytecode
     cache, so no __pycache__ file of theirs is read (under ``python -m``,
     runpy reads this script's own cached bytecode first; see the startup
     block). The checkout is the branch ``main`` with no uncommitted change
     and no untracked file, as git sees it with no file-system monitor to
     answer for it; no index entry is marked skip-worktree or
     assume-unchanged; every tracked file's bytes hash to HEAD's blob,
     whatever filters or stat caches say; and no ignored file sits where
     Python could import it.
  2. The history is whole: not a shallow clone, no ``refs/replace`` entry and
     no grafts, any of which would let git answer the questions below about
     substituted objects or a cut-off history.
  3. The freeze happened as §10 steps 2 and 3 say (research.freeze.verify_freeze):
     the amendment docs/preregistration-freeze.md is committed, unmodified
     and pushed to origin, which is the public repository; ``prereg-freeze``
     resolves to the commit it records and is an ancestor of HEAD; that
     commit only adds results/insample_v2.csv; and the directions in
     ``git show prereg-freeze:results/insample_v2.csv`` equal the
     amendment's (flags and MDEs too: the amendment copies them, and the
     MDE column of the primary results is taken from it).
  4. Every path in ``git diff --name-only prereg-freeze HEAD`` is under
     docs/ or results/, so the code is that of the tag.
  5. The constants are those of the tag. Appendix A lives under docs/, which
     guard 4 lets change, yet every constant of the run is read from it; so
     its parsed block at HEAD, and the one this process uses, must equal the
     block at the tag (prose and §12 rows may still be added). The directions
     file and the in-sample supplement's manifest must not have changed
     since the tag either.
  6. doge_fwd only: its window has closed (§10: steps 4-6 for doge_fwd
     happen after 2026-11-23 20:00 UTC).
  7. None of the outputs exists, or was ever committed: in any ref, on a
     branch merged after the output was removed from it again, or in a
     commit only the reflog still holds (write-once; a rerun goes to
     results/oos/rerun-N/ and the first run stays primary). An output that
     was never committed and then deleted leaves no trace in git.
  8. The set's manifest was committed once and never edited, after the
     freeze amendment, and pushed (§10 step 4); it records that test data
     was unlocked, the public repository, and that repository's main branch
     at fetch time, which holds the amendment's freeze blocks as they are
     now (its prose may still change); its window covers the set's whole
     window including its warm-up; each perpetual's candles, read from the
     files, reach to within a day of both ends of it (a spot leg or funding
     series that does not is logged, since §4.2 and §6.8 say what its cells
     become); and every input file's SHA-256 equals the manifest's
     (doge_fwd: also Appendix A's and the in-sample supplement's, since its
     warm-up is in-sample data).
  9. doge_2023 only: results/oos/doge_2024.csv and majors_2023.csv are
     committed, once and unedited, by a run of this script on the frozen
     code (their code_commit and manifest_sha256 columns are checked against
     git), their sets pass guard 8, and recomputing each from its inputs
     gives the committed file. The primary set's labels are read from them,
     so it runs last; this guard runs last because it is the slow one.

Then it computes the set (research.pipeline.run_oos_set) and creates
results/oos/<set>.csv, plus results/oos/robustness_doge_2023.csv on the
primary set, exclusively. The outputs are to be committed unedited.

Threat model. The guards stop every accident and every single-step mistake
(a forgotten commit or push, an edit git was told to ignore, a stray file,
a PYTHONPATH pointing at another checkout, a manifest re-hashed after its
data changed, a hand-tidied results file), and they make any
deliberate deviation require a visible, deliberate act. They cannot stop a
determined author who controls the machine: refs, clock and interpreter are
theirs. That is covered outside this script, by the public, timestamped
pre-registration (GitHub Release and Software Heritage) and by every result
being reproducible from public data.

Usage, in this order:
    python scripts/run_oos.py --set doge_2024
    python scripts/run_oos.py --set majors_2023
    python scripts/run_oos.py --set doge_2023
    python scripts/run_oos.py --set doge_fwd        # after its window closes
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

from research import freeze, gitrepo, integrity, pipeline
from research.io import sha256_file, write_once_csv
from research.spec import MINUTE_MS, PREREG_PATH, REPO_ROOT as SPEC_ROOT, get_spec, parse_spec_text

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_EXISTS = 4

MAIN_BRANCH = "main"
ALLOWED_PREFIXES = ("docs/", "results/")
PREREG_REL = PREREG_PATH.relative_to(SPEC_ROOT).as_posix()     # docs/preregistration.md

# How far a required series (a perpetual's candles) may stop short of either
# end of the set's window and still cover it: one day, or two sampling steps
# where that is longer. The coverage check is there to catch a truncated
# download (a wrong clock, an interrupted walk). fetch_v2 walks every window to
# its start and checks OKX's clock, so a series it wrote that stops short within
# a day means OKX has no bars there: an outage or a hole, whose bars §4.2's
# rules drop like any other gap and which costs a negligible share of the
# set's bars. Since this code is frozen at the tag (guard 4), refusing real data
# over it could not be undone without a public deviation. A day or more is not
# an outage to shrug off, and refuses.
COVERAGE_SLACK_MS = pipeline.DAY_MS

# pipeline.run_oos_set's signature: (set_name, directions, *, root, mde_bps,
# code_commit, manifest_sha256) -> (main rows, robustness rows or None).
Compute = Callable[..., Tuple[pl.DataFrame, Optional[pl.DataFrame]]]


class Refusal(Exception):
    """A guard failed; the message says which sentence of §10 was not met."""

    def __init__(self, message: str, code: int = EXIT_REFUSED) -> None:
        super().__init__(message)
        self.code = code


def _freeze_spec() -> Dict[str, str]:
    f = get_spec()["freeze"]
    return {"tag": f["tag"], "amendment": f["amendment_file"], "directions": f["directions_file"]}


def oos_sets() -> Tuple[str, ...]:
    """Every set except the in-sample year, in Appendix A order."""
    return tuple(n for n in get_spec()["sets"] if pipeline.set_role(n) != pipeline.ROLE_IS)


# ---------------------------------------------------------------- guards

def check_imports(repo_root: Path = REPO_ROOT) -> None:
    """
    Guard 1, first half: the git guards below inspect this checkout, so every
    loaded module of its packages must come from it (research.integrity).
    """
    foreign = integrity.foreign_code_problem(repo_root)
    if foreign:
        raise Refusal(foreign)


def check_checkout(root: Path) -> None:
    """
    Guard 1: the branch main, nothing uncommitted, nothing untracked, and the
    files on disk are HEAD's even where git status is told not to look
    (gitrepo.checkout_problems: index flags, filters, ignored importable files).
    """
    if not gitrepo.is_work_tree(root):
        raise Refusal(f"{root} is not a git work tree")
    branch = gitrepo.current_branch(root)
    if branch != MAIN_BRANCH:
        raise Refusal(f"HEAD is {branch or 'detached'}, not the branch {MAIN_BRANCH} (§10 step 5)")
    problems = gitrepo.checkout_problems(root)
    if problems:
        raise Refusal("the checkout is not exactly HEAD (§10 step 5 runs from a clean checkout): "
                      + "; ".join(problems))


def check_history(root: Path) -> None:
    """Guard 2: git answers about the real objects and the whole history."""
    if gitrepo.is_shallow(root):
        raise Refusal("the repository is a shallow clone; the guards below need its whole history "
                      "(fetch it with git fetch --unshallow)")
    replaced = gitrepo.replace_refs(root)
    if replaced:
        raise Refusal(f"the repository has refs/replace entries ({', '.join(replaced[:3])}); they "
                      "substitute objects behind recorded hashes. Remove them (git replace -d) first")
    if gitrepo.has_grafts(root):
        raise Refusal("the repository has an info/grafts file, which rewrites commit parents; remove it first")


def check_freeze(root: Path) -> Tuple[freeze.FreezeRecord, Dict[freeze.CellKey, Dict[str, Any]]]:
    """
    Guard 3: research.freeze.verify_freeze (the tag, the amendment, the tagged
    commit's shape, origin and the push). Returns the record and the tag's
    directions file, parsed.
    """
    try:
        record = freeze.verify_freeze(root)
    except freeze.FreezeNotVerified as exc:
        raise Refusal(str(exc)) from None
    # verify_freeze has just parsed and compared this file; parse it again
    # rather than widen its return value, since the run takes its directions
    # from here.
    text = gitrepo.show_file(root, record.commit, _freeze_spec()["directions"])
    return record, freeze.parse_insample_csv(text)


def check_diff(root: Path, tag_commit: str) -> None:
    """Guard 4: since the tag, only docs/ and results/ changed."""
    bad = [p for p in gitrepo.diff_names(root, tag_commit, "HEAD") if not p.startswith(ALLOWED_PREFIXES)]
    if bad:
        raise Refusal(
            f"paths outside {', '.join(ALLOWED_PREFIXES)} changed since {_freeze_spec()['tag']}: "
            + ", ".join(bad[:10]) + (" ..." if len(bad) > 10 else "")
        )


def _appendix(text: str, where: str) -> Dict[str, Any]:
    try:
        return parse_spec_text(text, where)
    except ValueError as exc:
        raise Refusal(str(exc)) from None


def frozen_spec(root: Path, commit: str) -> Dict[str, Any]:
    """Appendix A as it is in ``commit``."""
    try:
        text = gitrepo.show_file(root, commit, PREREG_REL)
    except gitrepo.GitError:
        raise Refusal(f"{PREREG_REL} is not in commit {commit[:12]}") from None
    return _appendix(text, f"{PREREG_REL} at {commit[:12]}")


def check_spec_frozen(root: Path, tag_commit: str) -> None:
    """
    Guard 5: Appendix A and the files frozen before the tag are those of the tag.

    The parsed blocks are compared, not the files, so the §12 deviations log
    and the prose can still grow after the tag. Two comparisons: with the
    file at HEAD in ``root``, and with get_spec(), the dict the pipeline
    actually computes with (in the real run both read the same file; in
    tests ``root`` is a throwaway repository).
    """
    tag = _freeze_spec()["tag"]
    frozen = frozen_spec(root, tag_commit)
    head_path = root / PREREG_REL
    if not head_path.exists():
        raise Refusal(f"{PREREG_REL} does not exist at HEAD")
    if _appendix(head_path.read_text(encoding="utf-8"), f"{PREREG_REL} at HEAD") != frozen:
        raise Refusal(
            f"Appendix A of {PREREG_REL} differs from its text at {tag}. Every constant of the "
            "run is read from it, so after the tag it is never edited (prose and §12 rows may be)"
        )
    if get_spec() != frozen:
        raise Refusal(f"the specification this process loaded from {PREREG_PATH} differs from Appendix A at {tag}")
    changed = set(gitrepo.diff_names(root, tag_commit, "HEAD"))
    touched = [p for p in (_freeze_spec()["directions"], pipeline.IS_SUPP_MANIFEST.as_posix()) if p in changed]
    if touched:
        raise Refusal(
            f"{', '.join(touched)} changed since {tag}; both are fixed before the tag (§10 step 2) "
            "and the frozen directions depend on them"
        )


def check_window_closed(set_name: str, now_ms: int) -> None:
    """Guard 6: a descriptive set that is still running cannot be analysed (§10)."""
    if pipeline.set_role(set_name) != pipeline.ROLE_DESCRIPTIVE:
        return
    end = int(get_spec()["sets"][set_name]["end_ms"])
    if now_ms < end:
        raise Refusal(f"{set_name}'s window ends at {end} ms; it is analysed once, after it closes")


def outputs_for(set_name: str) -> Tuple[Path, ...]:
    outs = [pipeline.oos_results_path(set_name)]
    if pipeline.set_role(set_name) == pipeline.ROLE_PRIMARY:
        outs.append(pipeline.robustness_results_path(set_name))
    return tuple(outs)


def check_outputs_absent(root: Path, set_name: str) -> None:
    """
    Guard 7: write-once, against the history as well as the working tree.

    A first run's output removed with ``git rm`` is still the first run
    (§10 step 6: it "stays primary"). So a path that was ever committed
    counts as existing: in any ref, on a branch merged after the output was
    removed from it again, or in a commit only the reflog still holds (one
    undone with ``git reset``). The reflog is local and expires, and an
    output that was never committed leaves no trace in git at all.
    """
    existing = [p.as_posix() for p in outputs_for(set_name) if (root / p).exists()]
    if existing:
        raise Refusal(
            f"refusing to overwrite {', '.join(existing)}; a rerun goes to results/oos/rerun-N/ (§10 step 6)",
            EXIT_EXISTS,
        )
    committed = [p.as_posix() for p in outputs_for(set_name)
                 if gitrepo.commits_touching(root, p.as_posix(), all_refs=True, reflog=True)]
    if committed:
        raise Refusal(
            f"{', '.join(committed)} was committed before; the first run stays primary and a rerun "
            "goes to results/oos/rerun-N/ (§10 step 6)",
            EXIT_EXISTS,
        )


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def manifest_window_required(set_name: str) -> Tuple[int, int]:
    """
    The [start, end) ms the set's own manifest must cover: warm-up and evaluation.

    doge_fwd's warm-up lies in the in-sample year and is read from the
    in-sample files (§4.1), so its manifest starts at the end of IS instead.
    The rule is the one pipeline.input_files splits on: a warm-up that begins
    before IS ends, for a window that begins after.
    """
    w = pipeline.window_of(set_name)
    is_end = int(get_spec()["sets"][pipeline.set_with_role(pipeline.ROLE_IS)]["end_ms"])
    lo = is_end if w.warmup_start_ms < is_end <= w.start_ms else w.warmup_start_ms
    return lo, w.end_ms


def series_step_ms() -> Dict[str, int]:
    """The sampling interval of each series kind fetch_v2 writes: 1 min candles, Appendix A's funding spacing."""
    hours = int(get_spec()["data"]["funding"]["expected_spacing_hours"])
    return {"candles": MINUTE_MS, "funding": hours * pipeline.HOUR_MS}


def coverage_problems(root: Path, set_name: str, series: List[Dict[str, Any]]) -> List[str]:
    """
    Each listed series' own first and last timestamp against the window the set needs.

    The manifest's window is what was asked for, and a download that stopped
    early (a wrong clock, an interrupted walk) records the same request; so
    the data itself is read. Its first row must come before the window's
    start plus the slack, and its last at or after the window's end minus
    it; the slack is COVERAGE_SLACK_MS or two of the series' steps,
    whichever is longer. A series of no known kind is refused, since its step is unknown.

    Only a required series (the perpetuals' candles, pipeline.input_files)
    that falls short is refused. A spot leg or funding series that stops
    short shows what the exchange still has, and §4.2 and §6.8 say what its
    factor's cells become (undefined on the bars it does not reach, or
    untestable); refusing it would refuse the whole set, and with a
    secondary set the primary set's labels, over cells the pre-registration
    expects to lose. It is logged instead.
    """
    lo, hi = manifest_window_required(set_name)
    steps = series_step_ms()
    required = {f.path.resolve() for f in pipeline.input_files(set_name, root) if f.required}
    out = []
    for e in series:
        rel, kind = str(e.get("path")), e.get("kind")
        if kind not in steps:
            out.append(f"{rel}: kind {kind!r} is not one of {sorted(steps)}, so its coverage cannot be judged")
            continue
        path = Path(rel) if Path(rel).is_absolute() else root / rel
        try:
            ts = pl.read_parquet(path, columns=["timestamp"])["timestamp"].cast(pl.Int64)
        except (OSError, pl.exceptions.PolarsError) as exc:
            out.append(f"{rel}: cannot read its timestamps ({type(exc).__name__}: {exc})")
            continue
        first, last = ts.min(), ts.max()
        slack = max(2 * steps[kind], COVERAGE_SLACK_MS)
        problem = None
        if first is None:
            problem = f"{rel} holds no rows"
        elif first >= lo + slack or last < hi - slack:
            problem = (f"{rel} runs from {_utc(first)} to {_utc(last)} UTC, but {set_name} needs "
                       f"[{_utc(lo)}, {_utc(hi)}) to within {slack / pipeline.HOUR_MS:g} h at either end")
        if problem and path.resolve() in required:
            out.append(problem)
        elif problem:
            logger.warning(f"{problem}; an optional input, so its cells follow §4.2 and §6.8")
    return out


def _first_added(root: Path, rel: str) -> Optional[str]:
    adds = gitrepo.commits_touching(root, rel, added_only=True)
    return adds[-1] if adds else None


def check_manifest(root: Path, set_name: str) -> str:
    """
    Guard 8: the set's data is what fetch_v2 downloaded under the verified
    freeze, committed once and pushed before any analysis (§10 steps 3-4),
    and every input file matches it. Returns the set manifest's SHA-256.
    """
    rel = pipeline.oos_manifest_path(set_name).as_posix()
    added = _manifest_committed_once(root, set_name, rel)
    manifest = _manifest_window_and_unlock(root, set_name, rel)
    _manifest_after_the_freeze_and_pushed(root, rel, added)
    _manifest_remote_record(root, rel, manifest)
    _inputs_match_and_cover(root, set_name, manifest)
    return sha256_file(root / rel)


def _manifest_committed_once(root: Path, set_name: str, rel: str) -> str:
    """Every manifest the set reads is tracked, and its own was committed once, in every ref. Returns that commit."""
    for m in pipeline.manifests_required(set_name):
        if not (root / m).exists() or not gitrepo.is_tracked(root, m.as_posix()):
            raise Refusal(f"manifest {m.as_posix()} is not committed (§10 step 4 commits it before any analysis)")
    # A manifest re-hashed after its data was changed (a week dropped, say) is
    # a second commit; _inputs_match_and_cover compares the data with the
    # current manifest only.
    added = _first_added(root, rel)
    touching = gitrepo.commits_touching(root, rel, all_refs=True)
    if added is None or touching != [added]:
        raise Refusal(f"{rel} was committed {len(touching)} times; a manifest is committed once, as fetched "
                      "(§10 step 4). A genuine re-download goes to a new path, with a §12 entry saying why")
    return added


def _manifest_window_and_unlock(root: Path, set_name: str, rel: str) -> Dict[str, Any]:
    """The manifest, once its window covers the set's (warm-up included) and it records that test data was unlocked."""
    try:
        manifest = json.loads((root / rel).read_text(encoding="utf-8"))
        window = manifest["window"]
        got = (int(window["start_ms"]), int(window["end_ms"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise Refusal(f"{rel} has no readable window ({type(exc).__name__}: {exc}); "
                      "scripts/fetch_v2.py writes one") from None
    lo, hi = manifest_window_required(set_name)
    if got[0] > lo or got[1] < hi:
        raise Refusal(
            f"{rel} covers [{_utc(got[0])}, {_utc(got[1])}) UTC, but {set_name} needs "
            f"[{_utc(lo)}, {_utc(hi)}) (its warm-up and its whole evaluation window, §4.1)"
        )
    if manifest.get("test_data_unlocked") is not True:
        raise Refusal(f"{rel} does not record test_data_unlocked: true; its data was not fetched "
                      "under a verified freeze (§10 step 4 follows step 3)")
    return manifest


def _manifest_after_the_freeze_and_pushed(root: Path, rel: str, added: str) -> None:
    """The commit that added the manifest comes after the one that added the amendment, and is on REMOTE_MAIN."""
    amendment = _freeze_spec()["amendment"]
    first = _first_added(root, amendment)
    if first is None or first == added or not gitrepo.is_ancestor(root, first, added):
        raise Refusal(
            f"{rel} was first committed in {added[:12]}, which does not come after the commit that "
            f"added {amendment} ({(first or '?')[:12]}); §10 step 3 comes before step 4"
        )
    pushed = gitrepo.commit_of(root, freeze.REMOTE_MAIN)
    if pushed is None or not gitrepo.is_ancestor(root, added, pushed):
        raise Refusal(f"{rel} was committed in {added[:12]}, which is not on {freeze.REMOTE_MAIN}; "
                      "§10 step 4 commits and pushes the manifest before any analysis")


def _manifest_remote_record(root: Path, rel: str, manifest: Dict[str, Any]) -> None:
    """
    What fetch_v2 saw online, since offline the push above is only a local
    ref: the public repository, whose main branch held the amendment in force
    when the data was fetched. In force means its machine-readable blocks,
    which verify_freeze has just checked against the tag; its prose (a
    Software Heritage identifier filled in, a typo) may change after the
    download, since the run reads nothing from it.
    """
    url, seen = manifest.get("remote_url"), manifest.get("remote_main_commit")
    if not isinstance(url, str) or not freeze.is_public_remote(url):
        raise Refusal(f"{rel} records remote_url {url!r}, not {freeze.PUBLIC_REMOTE}; scripts/fetch_v2.py "
                      "records the public repository it found the amendment on")
    if not (isinstance(seen, str) and gitrepo.FULL_SHA1.fullmatch(seen) and gitrepo.commit_of(root, seen) == seen):
        raise Refusal(f"{rel} records remote_main_commit {seen!r}, which is not the full hash of a commit here; "
                      "scripts/fetch_v2.py records the main branch it found the amendment on")
    amendment = _freeze_spec()["amendment"]
    try:
        then = freeze.freeze_blocks(gitrepo.show_file(root, seen, amendment))
    except gitrepo.GitError:
        then = {}
    if not then or then != freeze.freeze_blocks(gitrepo.show_file(root, "HEAD", amendment)):
        raise Refusal(f"{rel} records remote_main_commit {seen[:12]}, whose {amendment} does not hold the "
                      "freeze blocks in force; the public main branch must have had them when the data was "
                      "fetched (§10 step 3 before step 4)")


def _inputs_match_and_cover(root: Path, set_name: str, manifest: Dict[str, Any]) -> None:
    """Every input file has the SHA-256 its manifest records, and each series covers the set's window."""
    problems = pipeline.verify_inputs(set_name, root)
    if problems:
        raise Refusal("inputs differ from the committed manifest:\n  " + "\n  ".join(problems))
    problems = coverage_problems(root, set_name, manifest["series"])
    if problems:
        raise Refusal("the data does not cover the set's window:\n  " + "\n  ".join(problems))


def _secondary_provenance(root: Path, rel: str, other: str, tag_commit: str,
                          frozen: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    That ``rel`` is what one run of this script on the frozen code wrote,
    unedited, as far as git can tell. Returns its committed text, code_commit
    and manifest_sha256.

    * Committed exactly once, in any ref: an edit would be a second commit.
    * One code_commit on every row (the run writes HEAD there). It descends
      from the tag, differs from it only under docs/ or results/ with the
      tag's Appendix A, and is a strict ancestor of the commit that added
      the file, because the output is committed after the run.
    * One manifest_sha256 on every row, equal to the SHA-256 of the raw bytes
      of that set's manifest in code_commit, which is what the run hashed.

    All of this is metadata anyone can copy into a hand-written file, which
    is why check_prerequisites then recomputes the content.
    """
    commits = gitrepo.commits_touching(root, rel, all_refs=True)
    if len(commits) != 1:
        raise Refusal(f"{rel} was committed {len(commits)} times; §10 step 5 commits a run's output once, unedited")
    added = commits[0]
    text = gitrepo.show_file(root, added, rel)
    df = pl.read_csv(io.StringIO(text), infer_schema_length=0)
    missing = [c for c in ("code_commit", "manifest_sha256") if c not in df.columns]
    if missing or df.height == 0:
        raise Refusal(f"{rel} lacks {missing or 'rows'}; it is not an output of this script")
    codes = set(df["code_commit"].to_list())
    code = next(iter(codes)) if len(codes) == 1 else None
    if code is None or not gitrepo.FULL_SHA1.fullmatch(code) or gitrepo.commit_of(root, code) != code:
        raise Refusal(f"{rel}: every row must carry the same existing 40-hex code_commit, "
                      f"found {sorted(map(str, codes))[:3]}")
    tag = _freeze_spec()["tag"]
    if not gitrepo.is_ancestor(root, tag_commit, code):
        raise Refusal(f"{rel} was computed at {code[:12]}, which does not descend from {tag}")
    outside = [p for p in gitrepo.diff_names(root, tag_commit, code) if not p.startswith(ALLOWED_PREFIXES)]
    if outside:
        raise Refusal(f"{rel} was computed at {code[:12]}, whose code differs from {tag}: {', '.join(outside[:5])}")
    if frozen_spec(root, code) != frozen:
        raise Refusal(f"{rel} was computed at {code[:12]}, whose Appendix A differs from {tag}'s")
    if code == added or not gitrepo.is_ancestor(root, code, added):
        raise Refusal(f"{rel} was added in {added[:12]}, which does not come after its run's commit {code[:12]}")
    manifest_rel = pipeline.oos_manifest_path(other).as_posix()
    try:
        digest = hashlib.sha256(gitrepo.show_bytes(root, code, manifest_rel)).hexdigest()
    except gitrepo.GitError:
        raise Refusal(f"{manifest_rel} is not in {rel}'s code_commit {code[:12]}") from None
    shas = set(df["manifest_sha256"].to_list())
    if shas != {digest}:
        raise Refusal(f"{rel}: manifest_sha256 {sorted(map(str, shas))[:2]} is not the SHA-256 of "
                      f"{manifest_rel} at {code[:12]} ({digest})")
    return text, code, digest


def _same_csv_number(x: Optional[str], y: Optional[str]) -> bool:
    """
    Two CSV fields as numbers: equal to a relative 1e-9 (or an absolute 1e-9,
    for values that should be zero), a blank or NaN equal only to a blank or
    NaN. A multi-threaded sum may add in another order on the second run; a
    purely relative test would then refuse a mean of 1e-18 against -3e-19.
    """
    try:
        u, v = (math.nan if t is None or t == "" else float(t) for t in (x, y))
    except ValueError:
        return False
    return (math.isnan(u) and math.isnan(v)) or math.isclose(u, v, rel_tol=1e-9, abs_tol=1e-9)


def results_differences(committed: str, recomputed: pl.DataFrame) -> List[str]:
    """
    Where a committed results file differs from a recomputation; empty if nowhere.

    The recomputation is written exactly as the run writes it, so equal
    bytes are the normal case. Otherwise the columns and row order must be
    the same, every key and text column (status and labels included) equal,
    and the numeric columns equal to 1e-9 (_same_csv_number): enough to
    forgive float formatting and summation order, never a changed value.
    """
    with tempfile.TemporaryDirectory() as tmp:
        text = write_once_csv(recomputed, Path(tmp) / "recomputed.csv").read_text(encoding="utf-8")
    if text == committed:
        return []
    a = pl.read_csv(io.StringIO(committed), infer_schema_length=0)
    b = pl.read_csv(io.StringIO(text), infer_schema_length=0)
    if a.columns != b.columns or a.height != b.height:
        return [f"{a.height} rows of {a.width} columns committed, {b.height} rows of {b.width} recomputed"
                + ("" if a.columns == b.columns else ", with other columns")]
    numeric = {c for c, t in zip(recomputed.columns, recomputed.dtypes) if t in (pl.Float32, pl.Float64)}
    out = []
    for c in a.columns:
        same = _same_csv_number if c in numeric else (lambda x, y: x == y)
        out += [f"line {i + 2} {c}: {x!r} committed, {y!r} recomputed"
                for i, (x, y) in enumerate(zip(a[c].to_list(), b[c].to_list())) if not same(x, y)]
    return out


def check_prerequisites(root: Path, set_name: str, tag_commit: str,
                        directions: Dict[freeze.CellKey, Tuple[int, bool]],
                        mde: Dict[freeze.CellKey, float], compute: Compute) -> None:
    """
    Guard 9: the primary set's label sources are committed, once, by frozen
    runs, and are what the frozen code computes from their committed inputs.

    The provenance columns can be copied into a hand-written file, so each
    secondary set is recomputed: its manifest and inputs must pass guard 8,
    and ``compute`` (run_oos_set in the real run) with the tag's directions,
    the amendment's MDEs and the file's own code_commit and manifest_sha256
    must return the committed file. The computation is deterministic.
    """
    if pipeline.set_role(set_name) != pipeline.ROLE_PRIMARY:
        return
    frozen = frozen_spec(root, tag_commit)
    checked = []
    for role in (pipeline.ROLE_REPLICATION, pipeline.ROLE_POOLED):
        other = pipeline.set_with_role(role)
        rel = pipeline.oos_results_path(other).as_posix()
        if not (root / rel).exists() or not gitrepo.is_tracked(root, rel):
            raise Refusal(
                f"{rel} is not committed. {set_name}'s labels 'replicated' and "
                f"'consistent_on_majors' come from it, so run --set {other} first and commit its output."
            )
        checked.append((other, rel, *_secondary_provenance(root, rel, other, tag_commit, frozen)))
        check_manifest(root, other)
    # Only then the slow part.
    for other, rel, committed, code, digest in checked:
        logger.info(f"{other}: recomputing, to check {rel} against its inputs")
        try:
            again, _ = compute(other, directions, root=root, mde_bps=mde, code_commit=code, manifest_sha256=digest)
        except pipeline.PipelineError as exc:
            raise Refusal(f"recomputing {other} failed: {exc}") from None
        diffs = results_differences(committed, again)
        if diffs:
            raise Refusal(f"{rel} is not what the frozen code computes from {other}'s committed inputs "
                          f"({len(diffs)} difference(s)): " + "; ".join(diffs[:5]))


# ---------------------------------------------------------------- CLI

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one pre-registered out-of-sample set (§10 step 5), once.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--set", required=True, dest="set_name", help="one of: " + ", ".join(oos_sets()))
    return p.parse_args(argv)


def run(args: argparse.Namespace, *, root: Path = REPO_ROOT, compute: Optional[Compute] = None,
        now_ms: Optional[int] = None) -> int:
    """
    The CLI's body. ``root``, ``compute`` and ``now_ms`` are seams for tests
    (a throwaway repository, a stub computation, a fixed clock); the real
    run passes none of them, and none of them can loosen a guard. The stub
    stands in for the recomputation of guard 9 as well as for the run.
    """
    root = Path(root)
    set_name = args.set_name
    if set_name not in oos_sets():
        logger.error(f"--set must be one of {', '.join(oos_sets())}")
        return EXIT_USAGE
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    fn = compute if compute is not None else pipeline.run_oos_set

    try:
        check_imports()
        check_checkout(root)
        check_history(root)
        record, insample = check_freeze(root)
        tag_commit = record.commit
        directions = pipeline.directions_from_insample(insample)
        mde = {k: c.mde_bps for k, c in record.cells.items()}
        check_diff(root, tag_commit)
        check_spec_frozen(root, tag_commit)
        check_window_closed(set_name, now)
        check_outputs_absent(root, set_name)
        manifest_sha = check_manifest(root, set_name)
        check_prerequisites(root, set_name, tag_commit, directions, mde, fn)
    except Refusal as exc:
        logger.error(f"Refusing to run {set_name}: {exc}")
        return exc.code
    except gitrepo.GitError as exc:
        logger.error(f"Refusing to run {set_name}: {exc}")
        return EXIT_REFUSED

    head = gitrepo.head_commit(root)
    logger.info(f"{set_name}: guards passed at {head} (code of {tag_commit}); computing")
    main_df, robust_df = fn(
        set_name, directions, root=root, mde_bps=mde,
        code_commit=head, manifest_sha256=manifest_sha,
    )
    if tuple(main_df.columns) != pipeline.OOS_COLUMNS:
        raise AssertionError("computed columns are not the §10 columns")
    primary = pipeline.set_role(set_name) == pipeline.ROLE_PRIMARY
    if primary != (robust_df is not None):
        raise AssertionError("a robustness table belongs to the primary set, and only to it")

    outs = outputs_for(set_name)
    try:
        write_once_csv(main_df, root / outs[0])
        if robust_df is not None:
            write_once_csv(robust_df, root / outs[1])
    except FileExistsError as exc:
        logger.error(f"Refusing to overwrite {exc.filename}")
        return EXIT_EXISTS

    counts = main_df.group_by("status").len().sort("status")
    logger.info(
        f"{main_df.height} rows -> {outs[0].as_posix()}; status counts: "
        + ", ".join(f"{r['status']} {r['len']}" for r in counts.iter_rows(named=True))
    )
    if robust_df is not None:
        logger.info(f"{robust_df.height} rows -> {outs[1].as_posix()}")
    logger.info("Commit these outputs unedited (§10 step 5).")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None, **seams: Any) -> int:
    return run(parse_args(argv), **seams)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    raise SystemExit(main())
