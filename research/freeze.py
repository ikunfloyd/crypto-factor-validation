"""
The freeze amendment's machine-readable part, and the directions file it copies.

§10 step 3 has the freeze amendment (docs/preregistration-freeze.md) record
the commit that the ``prereg-freeze`` tag points to, and the 60 directions,
flags and minimum detectable effects "as they appear in
results/insample_v2.csv". Before the one out-of-sample run, scripts/run_oos.py
must check those against the tag (§10 step 5), so a human-readable amendment
is not enough: it also carries two fenced blocks that this module parses.
Everything else in the file is prose for readers and is ignored here.

Format. The info string after the opening fence names the block:

    ```freeze-commit
    <the full 40-hex commit that prereg-freeze points to>
    ```

    ```csv freeze-directions
    bar,horizon,factor,direction,flagged,flag_reason,mde_bps
    15m,1,rsi_14,1,false,,4.8123
    15m,1,macd_hist_norm,-1,true,few_active,undefined
    ...
    ```

  * exactly one block of each kind;
  * the table has exactly that header and one row per cell (60), in any order;
  * direction is 1 or -1; flagged is true or false; flag_reason is
    few_active, missing_input or blank; mde_bps is a number, or ``undefined``
    (or blank) where the in-sample cell had no active bars (§6.5).

:func:`render_amendment_blocks` writes both blocks from the bytes of
results/insample_v2.csv, so the amendment copies the file instead of
retyping it. A retyped number that differs in the ninth digit would make the
run refuse, and that is the point: the amendment must say what the tag says.

:func:`verify_freeze` asks git whether the freeze happened the way §10
steps 2 and 3 describe. scripts/run_oos.py needs that before the one run,
and scripts/fetch_v2.py before it may download test data: a file that merely
exists at docs/preregistration-freeze.md (a draft, or an empty ``touch``)
must unlock nothing.
"""
from __future__ import annotations

import csv
import io
import math
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from research import gitrepo
from research.spec import get_spec

# (bar, horizon, factor): the key of a cell everywhere in the study.
CellKey = Tuple[str, int, str]

# The columns of results/insample_v2.csv (§6.5). Restated here rather than
# read from Appendix A so a drift between the two is a test failure
# (tests/test_spec_constants.py), not a silent change of the file format.
INSAMPLE_COLUMNS: Tuple[str, ...] = (
    "bar", "horizon", "factor", "direction", "flagged", "flag_reason",
    "n_active", "mean_s_fwd_bps", "se_bps", "se_method", "mde_bps",
)

FREEZE_TABLE_COLUMNS: Tuple[str, ...] = (
    "bar", "horizon", "factor", "direction", "flagged", "flag_reason", "mde_bps",
)
FLAG_REASONS = ("few_active", "missing_input")
COMMIT_BLOCK = "freeze-commit"
TABLE_BLOCK = "freeze-directions"
UNDEFINED = "undefined"

# A fence opens with ``` plus an info string and closes with a bare ```.
_FENCE = re.compile(r"^```[ \t]*([^\n`]*)\n(.*?)^```[ \t]*$", re.M | re.S)
# The tolerance §4.2 uses to call two numbers the same; text copied from the
# CSV round-trips exactly, so this only forgives float formatting noise.
_REL_TOL = 1e-9


# Where the amendment must have been pushed to (§10 step 3: "It is pushed
# before step 4"). Only local evidence: the remote-tracking ref says what the
# last fetch or push saw on the remote, which is as much as can be checked
# without the network. One ``git update-ref`` can write it, so it counts only
# while origin is PUBLIC_REMOTE, and scripts/run_oos.py also wants the main
# branch that fetch_v2 read from the public repository itself.
REMOTE_MAIN = "refs/remotes/origin/main"

# The public repository the pre-registration is published in (§10 step 1).
# A remote-tracking ref is only as good as the remote it tracks, so origin
# must be this repository; scripts/fetch_v2.py, which is online anyway, also
# asks it directly what its main branch holds.
PUBLIC_REMOTE = "https://github.com/ikunfloyd/crypto-factor-validation.git"

# Network spellings of one repository: scheme://[user@]host[:port]/path and
# git's scp-like [user@]host:path.
_URL_REMOTE = re.compile(r"(?:https?|ssh|git)://(?:[^@/]+@)?([^/:]+)(?::\d+)?/(.+)", re.I)
_SCP_REMOTE = re.compile(r"(?:[^@/:]+@)?([^/:]+):(?!/)(.+)")


class FreezeFormatError(ValueError):
    """The amendment or the directions file does not have the documented shape."""


class FreezeNotVerified(RuntimeError):
    """The freeze (§10 steps 2-3) cannot be shown to have happened as written."""


@dataclass(frozen=True)
class FrozenCell:
    """One cell's frozen in-sample outcome (§6.5) and its MDE (§9)."""

    direction: int
    flagged: bool
    flag_reason: str      # "" when the cell is not flagged
    mde_bps: float        # NaN where the MDE is undefined


@dataclass(frozen=True)
class FreezeRecord:
    """What the freeze amendment records, as run_oos.py needs it."""

    commit: str
    cells: Dict[CellKey, FrozenCell]


def expected_cells(spec: Optional[Dict[str, Any]] = None) -> List[CellKey]:
    """The 60 cells in the study's order: bar, then horizon, then factor (Appendix A order)."""
    s = spec or get_spec()
    return [(b, int(h), f) for b in s["bars"] for h in s["horizons"] for f in s["factors"]]


# ---------------------------------------------------------------- field parsers

def _parse_direction(text: str, where: str) -> int:
    t = text.strip()
    if t in ("1", "+1"):
        return 1
    if t == "-1":
        return -1
    raise FreezeFormatError(f"{where}: direction must be 1 or -1, got {text!r}")


def _parse_bool(text: str, where: str) -> bool:
    t = text.strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    raise FreezeFormatError(f"{where}: flagged must be true or false, got {text!r}")


def _parse_reason(text: str, flagged: bool, where: str) -> str:
    t = text.strip()
    if t and t not in FLAG_REASONS:
        raise FreezeFormatError(f"{where}: flag_reason must be one of {FLAG_REASONS} or blank, got {text!r}")
    if bool(t) != flagged:
        raise FreezeFormatError(f"{where}: flag_reason {t!r} does not match flagged={flagged}")
    return t


def _parse_mde(text: str, where: str) -> float:
    t = text.strip()
    if t == "" or t.lower() == UNDEFINED:
        return math.nan
    try:
        v = float(t)
    except ValueError:
        raise FreezeFormatError(f"{where}: mde_bps must be a number or {UNDEFINED!r}, got {text!r}") from None
    if not math.isfinite(v):
        raise FreezeFormatError(f"{where}: mde_bps must be finite, got {text!r}")
    return v


def _rows(text: str, columns: Tuple[str, ...], what: str) -> List[Dict[str, str]]:
    reader = csv.reader(io.StringIO(text.strip("\n") + "\n"))
    lines = [r for r in reader if r]
    if not lines:
        raise FreezeFormatError(f"{what} is empty")
    header = tuple(c.strip() for c in lines[0])
    if header != columns:
        raise FreezeFormatError(f"{what} header is {list(header)}, expected {list(columns)}")
    out = []
    for i, r in enumerate(lines[1:], start=2):
        if len(r) != len(columns):
            raise FreezeFormatError(f"{what} line {i} has {len(r)} fields, expected {len(columns)}")
        out.append(dict(zip(columns, r)))
    return out


def _cells_from_rows(
    rows: Iterable[Dict[str, str]], what: str, spec: Optional[Dict[str, Any]]
) -> Dict[CellKey, Dict[str, Any]]:
    """Key every row by its cell, refusing unknown, duplicate or missing cells."""
    want = expected_cells(spec)
    known = set(want)
    out: Dict[CellKey, Dict[str, Any]] = {}
    for r in rows:
        try:
            key = (r["bar"].strip(), int(r["horizon"]), r["factor"].strip())
        except ValueError:
            raise FreezeFormatError(f"{what}: bad horizon {r['horizon']!r}") from None
        where = f"{what} cell {key}"
        if key not in known:
            raise FreezeFormatError(f"{where} is not one of the study's cells")
        if key in out:
            raise FreezeFormatError(f"{where} appears twice")
        flagged = _parse_bool(r["flagged"], where)
        out[key] = {
            "direction": _parse_direction(r["direction"], where),
            "flagged": flagged,
            "flag_reason": _parse_reason(r["flag_reason"], flagged, where),
            "mde_bps": _parse_mde(r["mde_bps"], where),
            "raw": r,
        }
    missing = [k for k in want if k not in out]
    if missing:
        raise FreezeFormatError(f"{what} lacks {len(missing)} cell(s), e.g. {missing[:3]}")
    return out


# ---------------------------------------------------------------- public API

def parse_insample_csv(text: str, spec: Optional[Dict[str, Any]] = None) -> Dict[CellKey, Dict[str, Any]]:
    """
    results/insample_v2.csv -> {cell: {direction, flagged, flag_reason, mde_bps, raw}}.

    The header must be exactly INSAMPLE_COLUMNS and every cell must appear
    once: the out-of-sample run reads its directions from this file at the
    tag, and a partial or reordered file must stop it, not be half-used.
    """
    return _cells_from_rows(_rows(text, INSAMPLE_COLUMNS, "insample_v2.csv"), "insample_v2.csv", spec)


def _blocks(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for m in _FENCE.finditer(text):
        for token in m.group(1).split():
            if token in (COMMIT_BLOCK, TABLE_BLOCK):
                out.setdefault(token, []).append(m.group(2))
    return out


def freeze_blocks(text: str) -> Dict[str, List[str]]:
    """
    The amendment's machine-readable blocks by name, as text: everything a run
    reads from it. Two versions with the same blocks record the same freeze,
    whatever their prose says.
    """
    return _blocks(text)


def parse_freeze_amendment(text: str, spec: Optional[Dict[str, Any]] = None) -> FreezeRecord:
    """
    The freeze commit and the 60 frozen cells from the amendment's text.

    Raises FreezeFormatError unless there is exactly one ``freeze-commit``
    block holding a full 40-hex commit and exactly one ``freeze-directions``
    block holding the whole table. A short or symbolic commit is refused: the
    guard compares it with what the tag resolves to, and an abbreviation
    could match more than one commit.
    """
    blocks = _blocks(text)
    for name in (COMMIT_BLOCK, TABLE_BLOCK):
        n = len(blocks.get(name, []))
        if n != 1:
            raise FreezeFormatError(f"the amendment must hold exactly one ```{name} block, found {n}")
    commit = blocks[COMMIT_BLOCK][0].strip()
    if not gitrepo.FULL_SHA1.fullmatch(commit):
        raise FreezeFormatError(f"```{COMMIT_BLOCK} must hold a full 40-hex commit, got {commit!r}")
    table = _cells_from_rows(
        _rows(blocks[TABLE_BLOCK][0], FREEZE_TABLE_COLUMNS, "freeze-directions table"),
        "freeze-directions table", spec,
    )
    cells = {
        k: FrozenCell(v["direction"], v["flagged"], v["flag_reason"], v["mde_bps"])
        for k, v in table.items()
    }
    return FreezeRecord(commit=commit, cells=cells)


def _same_number(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=0.0)


def compare_with_insample(record: FreezeRecord, insample: Dict[CellKey, Dict[str, Any]]) -> List[str]:
    """
    Every difference between the amendment and the tag's insample_v2.csv.

    §10 step 5 requires the directions to agree. Flags and MDEs are compared
    too: step 3 defines the amendment as recording them "as they appear in
    results/insample_v2.csv", and the run takes its mde_bps column from the
    amendment, so a disagreement means that column would not be the frozen one.
    """
    problems: List[str] = []
    for key in sorted(set(record.cells) | set(insample)):
        a, b = record.cells.get(key), insample.get(key)
        if a is None or b is None:
            problems.append(f"{key}: present in only one of amendment / insample_v2.csv")
            continue
        if a.direction != b["direction"]:
            problems.append(f"{key}: direction {a.direction:+d} in the amendment, {b['direction']:+d} at the tag")
        if a.flagged != b["flagged"] or a.flag_reason != b["flag_reason"]:
            problems.append(
                f"{key}: flag {a.flagged}/{a.flag_reason or '-'} in the amendment, "
                f"{b['flagged']}/{b['flag_reason'] or '-'} at the tag"
            )
        if not _same_number(a.mde_bps, b["mde_bps"]):
            problems.append(f"{key}: mde_bps {a.mde_bps!r} in the amendment, {b['mde_bps']!r} at the tag")
    return problems


def render_amendment_blocks(commit: str, insample_csv_text: str, spec: Optional[Dict[str, Any]] = None) -> str:
    """
    The two fenced blocks for docs/preregistration-freeze.md, from the tag's file.

    The values are copied as text, so the amendment matches the file exactly;
    a blank MDE is written as ``undefined``, the word §6.5 uses.
    """
    if not gitrepo.FULL_SHA1.fullmatch(commit):
        raise FreezeFormatError(f"commit must be a full 40-hex hash, got {commit!r}")
    cells = parse_insample_csv(insample_csv_text, spec)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FREEZE_TABLE_COLUMNS)
    for key in expected_cells(spec):
        raw = cells[key]["raw"]
        mde = raw["mde_bps"].strip() or UNDEFINED
        w.writerow([raw["bar"], raw["horizon"], raw["factor"], raw["direction"],
                    raw["flagged"], raw["flag_reason"], mde])
    return (
        f"```{COMMIT_BLOCK}\n{commit}\n```\n\n"
        f"```csv {TABLE_BLOCK}\n{buf.getvalue()}```\n"
    )


def normalize_remote(url: str, base: Optional[Path] = None) -> str:
    """
    One spelling per repository, so that the forms git accepts for the same
    remote compare equal.

    ``https://github.com/o/r``, ``https://github.com/o/r.git``,
    ``git@github.com:o/r.git`` and ``ssh://git@github.com/o/r`` all become
    ``github.com/o/r`` (lowercased: GitHub's names are case-insensitive).
    Anything else is a local path or ``file://`` URL and becomes the resolved
    path, relative paths taken from ``base`` (git resolves them from the
    work tree).
    """
    u = url.strip()
    m = _URL_REMOTE.fullmatch(u) or (None if u.startswith((".", "/", "~")) else _SCP_REMOTE.fullmatch(u))
    if m:
        path = m.group(2).strip("/")
        if path.lower().endswith(".git"):
            path = path[:-4]
        return f"{m.group(1)}/{path}".lower()
    if u.lower().startswith("file://"):
        u = urllib.parse.unquote(u[len("file://"):])
    p = Path(u).expanduser()
    return str((p if p.is_absolute() or base is None else Path(base) / p).resolve())


def is_public_remote(url: Optional[str], base: Optional[Path] = None) -> bool:
    """True if ``url`` names PUBLIC_REMOTE (read at call time, so tests can point it at a bare repository)."""
    return url is not None and normalize_remote(url, base) == normalize_remote(PUBLIC_REMOTE)


def amendment_missing_from(root: Path, commit: str, where: str, spec: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    None if the commit that last changed the amendment is an ancestor of
    ``commit``; otherwise why not, naming ``commit`` as ``where``.

    Two checks ask this of a different ``commit``: verify_freeze of
    REMOTE_MAIN, and fetch_v2 of the main branch the public repository
    reports. The last change rather than the first is asked for, so an edit
    made after a push does not ride on the earlier push. (run_oos asks less
    of the main branch a manifest says fetch_v2 saw: only that it held the
    freeze blocks in force, freeze_blocks, so the amendment's prose may
    change after a download.)
    """
    amendment = (spec or get_spec())["freeze"]["amendment_file"]
    last = gitrepo.commits_touching(root, amendment)[:1]
    if last and gitrepo.is_ancestor(root, last[0], commit):
        return None
    return f"the commit that last changed {amendment} ({last[0][:12] if last else '?'}) is not on {where}"


def verify_freeze(root: Path, spec: Optional[Dict[str, Any]] = None) -> FreezeRecord:
    """
    The freeze record, once git shows that §10 steps 2 and 3 happened as written.

    Raises FreezeNotVerified naming the first thing that does not hold (and
    lets gitrepo.GitError through if git itself fails):

      * the amendment is committed and has no uncommitted change, so what is
        checked is what was committed (a draft cannot unlock anything);
      * its machine-readable blocks parse;
      * the tag resolves to the commit the amendment records, and that
        commit is an ancestor of HEAD;
      * the tagged commit holds the directions file, which parses, and
        differs from its single parent only by adding that file (§10 step 2:
        "added in the next commit with no other change, and that commit is
        tagged"), so no code change can hide inside the tagged commit;
      * the amendment agrees with that file (directions, flags and MDEs);
      * origin is PUBLIC_REMOTE, for fetching and for pushing, since
        REMOTE_MAIN only says something about the remote it tracks;
      * the commit that last changed the amendment is an ancestor of
        REMOTE_MAIN: local evidence that the amendment in force was pushed
        (§10 step 3). The last change rather than the first is asked for, so
        an edit made after the push does not ride on the earlier push.

    The checks stop accidents (a draft, a forgotten push, a clone of a fork)
    and make any deviation a deliberate act; they cannot stop someone who
    controls the machine from forging local refs. The public, timestamped
    pre-registration covers that.

    ``spec`` defaults to the checked-in Appendix A; its ``freeze`` block
    names the tag and both files.
    """
    s = spec or get_spec()
    f = s["freeze"]
    tag, amendment, directions = f["tag"], f["amendment_file"], f["directions_file"]
    root = Path(root)

    if not gitrepo.is_work_tree(root):
        raise FreezeNotVerified(f"{root} is not a git work tree, so the freeze cannot be checked")
    if not (root / amendment).exists():
        raise FreezeNotVerified(f"{amendment} does not exist; the freeze amendment (§10 step 3) comes first")
    if not gitrepo.is_tracked(root, amendment):
        raise FreezeNotVerified(f"{amendment} is not committed; §10 step 3 commits and pushes it first")
    dirty = gitrepo.path_status(root, amendment)
    if dirty:
        raise FreezeNotVerified(f"{amendment} has uncommitted changes ({dirty[0][:2].strip()}); "
                                "only the committed amendment counts")
    try:
        record = parse_freeze_amendment(gitrepo.show_file(root, "HEAD", amendment), s)
    except FreezeFormatError as exc:
        raise FreezeNotVerified(f"{amendment}: {exc}") from None

    commit = gitrepo.tag_commit(root, tag)
    if commit is None:
        raise FreezeNotVerified(f"there is no tag {tag}")
    if commit != record.commit:
        raise FreezeNotVerified(f"tag {tag} resolves to {commit}, but the freeze amendment records {record.commit}")
    if not gitrepo.is_ancestor(root, commit, "HEAD"):
        raise FreezeNotVerified(f"tag {tag} ({commit}) is not an ancestor of HEAD")

    try:
        text = gitrepo.show_file(root, commit, directions)
    except gitrepo.GitError:
        raise FreezeNotVerified(f"{directions} is not in the commit {tag} points to") from None
    try:
        insample = parse_insample_csv(text, s)
    except FreezeFormatError as exc:
        raise FreezeNotVerified(f"{directions} at {tag}: {exc}") from None
    ups = gitrepo.parents(root, commit)
    changes = gitrepo.name_status(root, ups[0], commit) if len(ups) == 1 else None
    if changes != [("A", directions)]:
        what = (f"{len(ups)} parents" if changes is None
                else ", ".join(f"{st} {p}" for st, p in changes[:6]) + (" ..." if len(changes) > 6 else ""))
        raise FreezeNotVerified(
            f"{tag} must tag the commit that adds {directions} and nothing else (§10 step 2); "
            f"{commit[:12]} has {what}"
        )

    problems = compare_with_insample(record, insample)
    if problems:
        raise FreezeNotVerified(
            f"the freeze amendment differs from {directions} at {tag} "
            f"({len(problems)} difference(s)): " + "; ".join(problems[:5])
        )

    for push in (False, True):
        url = gitrepo.origin_url(root, push=push)
        if not is_public_remote(url, root):
            raise FreezeNotVerified(
                f"the remote origin {'pushes to' if push else 'is'} {url or 'not set'}, not {PUBLIC_REMOTE}; "
                f"{REMOTE_MAIN} shows a push only if it tracks the public repository (§10 steps 1 and 3)"
            )
    remote = gitrepo.commit_of(root, REMOTE_MAIN)
    if remote is None:
        raise FreezeNotVerified(f"there is no {REMOTE_MAIN}, so nothing shows that {amendment} "
                                "was pushed (§10 step 3 pushes it before any download)")
    problem = amendment_missing_from(root, remote, REMOTE_MAIN, s)
    if problem:
        raise FreezeNotVerified(f"{problem}; §10 step 3 pushes the amendment before any download or run")
    return record


__all__ = [
    "CellKey", "INSAMPLE_COLUMNS", "FREEZE_TABLE_COLUMNS", "REMOTE_MAIN", "PUBLIC_REMOTE",
    "normalize_remote", "is_public_remote", "FreezeFormatError",
    "FreezeNotVerified", "FrozenCell", "FreezeRecord", "expected_cells", "parse_insample_csv",
    "freeze_blocks", "parse_freeze_amendment", "compare_with_insample", "render_amendment_blocks",
    "amendment_missing_from", "verify_freeze",
]
