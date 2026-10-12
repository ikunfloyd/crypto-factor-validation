#!/usr/bin/env python3
"""
Write the freeze amendment, docs/preregistration-freeze.md (pre-registration §10, step 3).

The amendment is what the one out-of-sample run checks itself against
(research.freeze.verify_freeze), and it is written once: so nothing in it is
typed by hand. Each entry comes from where §10 says it lives, and is checked
before the file exists:

  * **Freeze commit.** The commit ``prereg-freeze`` resolves to. The tag must
    exist, be an ancestor of HEAD, and tag a commit that adds
    results/insample_v2.csv and nothing else, since the amendment says so.
  * **Directions table.** research.freeze.render_amendment_blocks from
    ``git show <tag>:results/insample_v2.csv``, never the working tree.
  * **Publication commit.** --publication-commit: the commit the GitHub
    Release is cut from (§10 step 1). A full 40-hex commit of this repository
    that holds docs/preregistration.md and is an ancestor of the tag. It is
    not guessed from the log: which commit was released is a fact about
    GitHub that only the author can state.
  * **Software Heritage.** --swh-publication and --swh-freeze, each
    ``swh:1:(rev|dir|snp|rel):<40 hex>`` with optional ``;qualifiers``.
    Software Heritage computes revision, directory and release identifiers
    as git does, so a rev must be the commit itself, a dir its tree, and a
    rel an annotated tag object of this repository pointing to it; a mistyped
    identifier is refused instead of being frozen. A snapshot (snp) depends
    on the archived origin's state and cannot be recomputed here, so it is
    taken as given.
  * **Maker-probe days (§11).** scripts/maker_probe_days.py's draw from the
    committed candidate list, with the same refusals (list committed once and
    unmodified, numpy as pinned). They go in a fenced block with the info
    string ``maker-probe-days``, one date per line, for the probe to read
    (maker_probe_days_from).

Before writing, the text is parsed back with research.freeze.parse_freeze_amendment
and compared with the tag's directions file (compare_with_insample), as the
run will; any difference stops it. The file is created exclusively, and an
amendment that was ever committed counts as existing. The script does not
commit or push: it prints the commands that come next.

Usage:
    python scripts/write_freeze_amendment.py \\
        --publication-commit <40-hex commit of the GitHub Release> \\
        --swh-publication 'swh:1:rev:<40 hex>' --swh-freeze 'swh:1:rev:<40 hex>'
"""
from __future__ import annotations

import argparse
import atexit
import re
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

from research import freeze, gitrepo, integrity
from research.io import write_once_bytes
from research.spec import PREREG_PATH, REPO_ROOT as SPEC_ROOT, get_spec


# The draw comes from scripts/maker_probe_days.py, so the amendment records exactly what it prints.
mp = integrity.load_script(REPO_ROOT, "maker_probe_days")
Refusal = mp.Refusal

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_EXISTS = 4

PREREG_REL = PREREG_PATH.relative_to(SPEC_ROOT).as_posix()        # docs/preregistration.md
DAYS_BLOCK = "maker-probe-days"
# A Software Heritage identifier (SWHID v1) of one of the four kinds §10 can
# name, then qualifiers such as ;origin=...;visit=...;anchor=... (no spaces,
# no ';' inside a value; values are percent-encoded).
SWHID = re.compile(r"swh:1:(rev|dir|snp|rel):([0-9a-f]{40})((?:;[a-z]+=[^;\s]+)*)")
# The fence research/freeze.py reads: ``` plus an info string, closed by a bare ```.
_FENCE = re.compile(r"^```[ \t]*([^\n`]*)\n(.*?)^```[ \t]*$", re.M | re.S)
_WRAP = 100


# ---------------------------------------------------------------- the checks

def freeze_commit(root: Path, spec: Dict[str, Any]) -> str:
    """
    The commit the tag resolves to, once it is the commit the amendment will
    say it is: an ancestor of HEAD that adds the directions file and nothing
    else (§10 step 2). verify_freeze asks the same later.
    """
    f = spec["freeze"]
    tag, directions = f["tag"], f["directions_file"]
    commit = gitrepo.tag_commit(root, tag)
    if commit is None:
        raise Refusal(f"there is no tag {tag}; §10 step 2 tags the commit that adds {directions} first")
    if not gitrepo.is_ancestor(root, commit, "HEAD"):
        raise Refusal(f"tag {tag} ({commit[:12]}) is not an ancestor of HEAD; the amendment is committed after it")
    ups = gitrepo.parents(root, commit)
    changes = gitrepo.name_status(root, ups[0], commit) if len(ups) == 1 else None
    if changes != [("A", directions)]:
        what = (f"{len(ups)} parents" if changes is None
                else ", ".join(f"{st} {p}" for st, p in changes[:6]) + (" ..." if len(changes) > 6 else ""))
        raise Refusal(f"{tag} must tag the commit that adds {directions} and nothing else (§10 step 2); "
                      f"{commit[:12]} has {what}")
    return commit


def check_publication(root: Path, commit: str, tag_commit: str, tag: str) -> None:
    """The publication commit exists, holds the pre-registration, and precedes the freeze."""
    if gitrepo.commit_of(root, commit) != commit:
        raise Refusal(f"--publication-commit {commit} is not a commit of this repository")
    if gitrepo.git(root, "cat-file", "-e", f"{commit}:{PREREG_REL}", check=False).returncode != 0:
        raise Refusal(f"--publication-commit {commit[:12]} has no {PREREG_REL}; the Release is cut from the "
                      "commit that holds the pre-registration (§10 step 1)")
    if not gitrepo.is_ancestor(root, commit, tag_commit):
        raise Refusal(f"--publication-commit {commit[:12]} is not an ancestor of {tag} ({tag_commit[:12]}); "
                      "the pre-registration is published before the implementation and the freeze (§10)")


def swh_problem(root: Path, swhid: str, commit: str) -> Optional[str]:
    """
    Why ``swhid`` does not identify ``commit``; None if it does or cannot be
    checked (a snapshot).
    """
    m = SWHID.fullmatch(swhid)
    if m is None:
        return f"{swhid!r} is not swh:1:(rev|dir|snp|rel):<40 hex>[;qualifiers]"
    kind, oid = m.group(1), m.group(2)
    if kind == "rev":
        if oid != commit:
            return f"names revision {oid[:12]}, but the commit is {commit[:12]} (a revision id is the commit id)"
    elif kind == "dir":
        proc = gitrepo.git(root, "rev-parse", "--verify", "--quiet", f"{commit}^{{tree}}", check=False)
        tree = proc.stdout.decode("utf-8", "replace").strip()
        if oid != tree:
            return f"names directory {oid[:12]}, but the tree of {commit[:12]} is {tree[:12] or '?'}"
    elif kind == "rel":
        proc = gitrepo.git(root, "cat-file", "-t", oid, check=False)
        if proc.returncode != 0 or proc.stdout.decode("utf-8", "replace").strip() != "tag":
            return f"names release {oid[:12]}, which is not a tag object of this repository (fetch the tags?)"
        peeled = gitrepo.commit_of(root, oid)
        if peeled != commit:
            return f"names release {oid[:12]}, which points to {(peeled or '?')[:12]}, not {commit[:12]}"
    return None


# ---------------------------------------------------------------- the text

def _bullet(text: str) -> str:
    return textwrap.fill(text, _WRAP, initial_indent="- ", subsequent_indent="  ",
                         break_long_words=False, break_on_hyphens=False)


def render_amendment(
    *, spec: Dict[str, Any], publication: str, swh_publication: str, swh_freeze: str, draw: Any, blocks: str,
) -> str:
    """The amendment's text: the prose, the maker-probe-days block, then freeze.render_amendment_blocks' two blocks."""
    f, probe = spec["freeze"], spec["maker_probe"]
    tag, directions = f["tag"], f["directions_file"]
    n_cells = len(freeze.expected_cells(spec))
    n_days = int(probe["n_days"])
    if len(draw.candidates) < n_days:
        how = f"{len(draw.candidates)} days qualified, fewer than {n_days}, so all are used"
    else:
        how = f"{len(draw.days)} of {len(draw.candidates)} candidate days"
    days_block = f"```{DAYS_BLOCK}\n" + "".join(d + "\n" for d in draw.days) + "```\n"
    parts = [
        "# Freeze amendment to the pre-registration",
        "",
        f"This file records the freeze of the out-of-sample study ({PREREG_REL} §10, step 3).",
        "It changes nothing in the pre-registration. It is committed and pushed before any test-set",
        "data is downloaded (§10, step 4).",
        "",
        _bullet(f"**Freeze commit.** The tag `{tag}` points to the commit below, which adds "
                f"`{directions}` and nothing else."),
        _bullet(f"**Publication commit.** `{publication}` (the GitHub Release)."),
        _bullet(f"**Software Heritage.** Publication commit: `{swh_publication}`. `{tag}`: `{swh_freeze}`."),
        _bullet(f"**Maker-probe days (§11).** Drawn from `{probe['candidates_file']}` as §11 specifies "
                f"({how}): {', '.join(draw.days)}."),
        "",
        textwrap.fill(
            f"The table copies the {n_cells} directions, flags and minimum detectable effects (§6.5, §9) from "
            f"`{directions}` at the tag, as text. An MDE is `undefined` where the in-sample cell had no "
            "active bars.", _WRAP, break_long_words=False, break_on_hyphens=False),
        "",
        days_block,
        blocks,
    ]
    return "\n".join(parts)


def maker_probe_days_from(text: str) -> List[str]:
    """
    The maker-probe days recorded in an amendment's text: the one
    ``maker-probe-days`` block, one YYYY-MM-DD date per line, ascending.
    Raises ValueError otherwise. The probe reads its days with this.
    """
    bodies = [m.group(2) for m in _FENCE.finditer(text) if DAYS_BLOCK in m.group(1).split()]
    if len(bodies) != 1:
        raise ValueError(f"the amendment must hold exactly one ```{DAYS_BLOCK} block, found {len(bodies)}")
    days = [line.strip() for line in bodies[0].splitlines() if line.strip()]
    if not days:
        raise ValueError(f"the ```{DAYS_BLOCK} block is empty")
    for d in days:
        mp.parse_day(d)                              # ValueError unless YYYY-MM-DD
    if days != sorted(set(days)):
        raise ValueError(f"the ```{DAYS_BLOCK} block is not ascending without repeats")
    return days


def self_check(text: str, tag_commit: str, insample_text: str, days: List[str], spec: Dict[str, Any]) -> List[str]:
    """Every way ``text`` would not say what the tag and the draw say, read back as the run reads it."""
    try:
        record = freeze.parse_freeze_amendment(text, spec)
    except freeze.FreezeFormatError as exc:
        return [f"it does not parse: {exc}"]
    problems = []
    if record.commit != tag_commit:
        problems.append(f"it records commit {record.commit}, not {tag_commit}")
    problems += freeze.compare_with_insample(record, freeze.parse_insample_csv(insample_text, spec))
    try:
        got = maker_probe_days_from(text)
    except ValueError as exc:
        problems.append(str(exc))
    else:
        if got != days:
            problems.append(f"its maker-probe days {got} are not the draw {days}")
    return problems


# ---------------------------------------------------------------- CLI

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write docs/preregistration-freeze.md, the freeze amendment (§10 step 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--publication-commit", required=True,
                   help="full 40-hex commit the GitHub Release of the pre-registration is cut from")
    p.add_argument("--swh-publication", required=True, help="Software Heritage identifier of the publication commit")
    p.add_argument("--swh-freeze", required=True, help="Software Heritage identifier of prereg-freeze")
    return p.parse_args(argv)


def run(args: argparse.Namespace, *, root: Path = REPO_ROOT, spec: Optional[Dict[str, Any]] = None) -> int:
    """
    The CLI's body. ``root`` and ``spec`` are seams for tests (a throwaway
    repository); the real run passes neither.
    """
    root = Path(root)
    s = spec or get_spec()
    f = s["freeze"]
    tag, amendment, directions = f["tag"], f["amendment_file"], f["directions_file"]

    foreign = integrity.foreign_code_problem(REPO_ROOT)
    if foreign:
        logger.error(f"Refusing to write the amendment: {foreign}")
        return EXIT_REFUSED
    publication = args.publication_commit.strip()
    if not gitrepo.FULL_SHA1.fullmatch(publication):
        logger.error(f"--publication-commit must be a full 40-hex commit, got {args.publication_commit!r}")
        return EXIT_USAGE
    swh = {"--swh-publication": args.swh_publication.strip(), "--swh-freeze": args.swh_freeze.strip()}
    for opt, value in swh.items():
        if not SWHID.fullmatch(value):
            logger.error(f"{opt} must be swh:1:(rev|dir|snp|rel):<40 hex>, optionally followed by ;qualifiers; "
                         f"got {value!r}")
            return EXIT_USAGE

    out = root / amendment
    try:
        if out.exists():
            raise Refusal(f"{amendment} exists; the freeze amendment is written once", EXIT_EXISTS)
        if not gitrepo.is_work_tree(root):
            raise Refusal(f"{root} is not a git work tree")
        if gitrepo.commits_touching(root, amendment, all_refs=True, reflog=True):
            raise Refusal(f"{amendment} was committed before; the freeze amendment is written once", EXIT_EXISTS)
        tag_commit = freeze_commit(root, s)
        check_publication(root, publication, tag_commit, tag)
        for opt, commit in (("--swh-publication", publication), ("--swh-freeze", tag_commit)):
            problem = swh_problem(root, swh[opt], commit)
            if problem:
                raise Refusal(f"{opt} {problem}")
            if SWHID.fullmatch(swh[opt]).group(1) == "snp":
                logger.warning(f"{opt} is a snapshot identifier, which cannot be checked against git here")
        insample = gitrepo.show_file(root, tag_commit, directions)
        try:
            blocks = freeze.render_amendment_blocks(tag_commit, insample, s)
        except freeze.FreezeFormatError as exc:
            raise Refusal(f"{directions} at {tag}: {exc}") from None
        draw = mp.committed_draw(root, s)
        text = render_amendment(spec=s, publication=publication,
                                swh_publication=swh["--swh-publication"], swh_freeze=swh["--swh-freeze"],
                                draw=draw, blocks=blocks)
        problems = self_check(text, tag_commit, insample, draw.days, s)
        if problems:
            raise Refusal(f"the amendment would not read back as the tag and the draw say "
                          f"({len(problems)} problem(s)): " + "; ".join(problems[:5]))
        write_once_bytes(text.encode("utf-8"), out)
    except Refusal as exc:
        logger.error(f"Refusing to write the amendment: {exc}")
        return exc.code
    except FileExistsError:
        logger.error(f"Refusing to overwrite {amendment}")
        return EXIT_EXISTS
    except gitrepo.GitError as exc:
        logger.error(f"Refusing to write the amendment: {exc}")
        return EXIT_REFUSED

    logger.info(f"{amendment}: freeze commit {tag_commit}, {len(freeze.expected_cells(s))} cells, "
                f"maker-probe days {' '.join(draw.days)}")
    logger.info("Next (§10 step 3: commit and push before any test-set download); this script runs none of it:")
    logger.info(f"  git add {amendment}")
    logger.info('  git commit -m "Record the freeze amendment"')
    logger.info("  git push origin main")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None, **seams: Any) -> int:
    return run(parse_args(argv), **seams)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")
    raise SystemExit(main())
