"""
The few git questions the study's run scripts must answer (§10 steps 2 and 5).

Each function runs read-only git commands against the repository at
``root``; none of them changes the repository, its refs or its config. They
exist so that scripts/run_insample_v2.py and scripts/run_oos.py ask git the
same way, and so the guards can be tested against throwaway repositories.

Every command passes ``-C root`` explicitly: the scripts must inspect the
checkout they read their inputs from, not whatever directory they were
started in. Two more options keep local settings from changing git's
answers:

  * ``--no-replace-objects``: a local ``git replace`` entry would otherwise
    change what ``git show prereg-freeze:results/insample_v2.csv`` and
    ``git diff`` return while ``rev-parse`` still prints the recorded hash.
    (scripts/run_oos.py refuses a repository holding such entries, or
    grafts, or a shallow history, outright.)
  * ``-c core.fsmonitor=false``: with a file-system monitor configured,
    ``git status`` asks a hook which paths changed instead of looking, and
    a hook that always answers "nothing" hides a new untracked module (a
    ``yaml.py`` that would shadow PyYAML) from the clean-checkout check.
    Without it, git scans the disk itself.
"""
from __future__ import annotations

import importlib.machinery
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


# A full commit hash. Wherever a recorded commit is compared with git's
# answer, an abbreviation is refused: it could name more than one commit.
FULL_SHA1 = re.compile(r"[0-9a-f]{40}")


class GitError(RuntimeError):
    """git failed, or the directory is not a git work tree."""


def git(root: Path, *args: str, check: bool = True, input: Optional[bytes] = None) -> subprocess.CompletedProcess:
    """
    Run ``git --no-replace-objects -c core.fsmonitor=false -C root <args>``
    (``input`` on stdin) and capture its output as bytes.
    """
    try:
        proc = subprocess.run(["git", "--no-replace-objects", "-c", "core.fsmonitor=false", "-C", str(root), *args],
                              capture_output=True, input=input)
    except FileNotFoundError as exc:
        raise GitError("git is not installed or not on PATH") from exc
    if check and proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise GitError(f"git {' '.join(args)} failed: {msg}")
    return proc


def _text(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout.decode("utf-8", "replace").strip()


def is_work_tree(root: Path) -> bool:
    proc = git(root, "rev-parse", "--is-inside-work-tree", check=False)
    return proc.returncode == 0 and _text(proc) == "true"


def status_entries(root: Path, include_untracked: bool = True) -> List[str]:
    """
    ``git status --porcelain`` entries, empty for a clean tree.

    Ignored files (the gitignored data/history) never appear. With
    ``include_untracked`` False only changes to tracked files count, which is
    what "uncommitted changes to tracked files" means.
    """
    mode = "--untracked-files=normal" if include_untracked else "--untracked-files=no"
    out = git(root, "status", "--porcelain=v1", "-z", mode).stdout.decode("utf-8", "replace")
    return [e for e in out.split("\0") if e]


def head_commit(root: Path) -> str:
    return _text(git(root, "rev-parse", "--verify", "HEAD^{commit}"))


def current_branch(root: Path) -> Optional[str]:
    """The checked-out branch's short name, or None on a detached HEAD."""
    proc = git(root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    return _text(proc) if proc.returncode == 0 else None


def tag_commit(root: Path, tag: str) -> Optional[str]:
    """
    The commit a tag resolves to, or None if there is no such tag.

    Asked as refs/tags/<tag> so that a branch of the same name cannot stand
    in for the tag.
    """
    proc = git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}", check=False)
    return _text(proc) if proc.returncode == 0 else None


def is_ancestor(root: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    """True if ``ancestor`` is reachable from ``descendant`` (a commit is its own ancestor)."""
    proc = git(root, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if proc.returncode not in (0, 1):
        raise GitError(_text(proc) or proc.stderr.decode("utf-8", "replace"))
    return proc.returncode == 0


def diff_names(root: Path, a: str, b: str = "HEAD") -> List[str]:
    """
    Paths that differ between two commits (``git diff --name-only a b``).

    ``--no-renames`` lists both sides of a rename, so moving a file out of
    research/ into docs/ still shows the research/ path; ``-z`` keeps
    unusual file names unquoted.
    """
    out = git(root, "diff", "--name-only", "--no-renames", "-z", a, b).stdout.decode("utf-8", "replace")
    return [p for p in out.split("\0") if p]


def show_file(root: Path, rev: str, path: str) -> str:
    """A file's content at a revision (``git show rev:path``)."""
    return git(root, "show", f"{rev}:{path}").stdout.decode("utf-8")


def show_bytes(root: Path, rev: str, path: str) -> bytes:
    """A file's raw bytes at a revision, for hashing (no decoding, no newline changes)."""
    return git(root, "show", f"{rev}:{path}").stdout


def is_tracked(root: Path, path: str) -> bool:
    proc = git(root, "ls-files", "--error-unmatch", "--", path, check=False)
    return proc.returncode == 0


def path_status(root: Path, path: str) -> List[str]:
    """``git status --porcelain`` entries for one path: empty when it is as committed."""
    out = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=normal", "--", path).stdout
    return [e for e in out.decode("utf-8", "replace").split("\0") if e]


def commit_of(root: Path, ref: str) -> Optional[str]:
    """The commit ``ref`` (a full ref name such as refs/remotes/origin/main) points to, or None."""
    proc = git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    return _text(proc) if proc.returncode == 0 else None


def parents(root: Path, commit: str) -> List[str]:
    """The parents of ``commit`` (none for a root commit, two or more for a merge)."""
    return _text(git(root, "rev-list", "--parents", "-n", "1", commit)).split()[1:]


def name_status(root: Path, a: str, b: str) -> List[Tuple[str, str]]:
    """(status letter, path) for each path that differs between two commits, renames split."""
    out = git(root, "diff", "--name-status", "--no-renames", "-z", a, b).stdout.decode("utf-8", "replace")
    fields = [f for f in out.split("\0") if f]
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields) - 1, 2)]


def commits_touching(root: Path, path: str, *, all_refs: bool = False, reflog: bool = False,
                     added_only: bool = False) -> List[str]:
    """
    Commits that changed ``path``, newest first (``git log --format=%H -- path``).

    By default only HEAD's history is searched, simplified as ``git log``
    does by default. ``added_only`` keeps the commits that added the path
    (``--diff-filter=A``).

    ``all_refs`` searches every branch, tag and remote-tracking ref
    (``--all``) and turns off the default simplification, which prunes a
    side branch whose merge leaves the path as it was: a file committed on
    a PR branch, removed again there and merged would otherwise leave no
    trace. ``--full-history`` alone would go too far the other way and list
    an innocent merge (one that keeps one parent's version) as a commit of
    its own; ``--simplify-merges`` drops those and keeps a merge that edits
    the path itself.

    ``reflog`` (implies ``all_refs``) also walks the commits that only the
    reflog still reaches, such as one undone with ``git reset --hard
    HEAD~1``. It is for "was this output ever committed": an amend or a
    rebase leaves reflog copies of innocent commits too, so a "committed
    exactly once" question must not ask it. The reflog is local and
    expires (by default after 30 days for such commits), and a file that
    was never committed leaves no trace in git at all.
    """
    args = ["log", "--format=%H"]
    if all_refs or reflog:
        args += ["--all", "--full-history", "--simplify-merges"]
    if reflog:
        args.append("--reflog")
    if added_only:
        args.append("--diff-filter=A")
    return _text(git(root, *args, "--", path)).split()


def is_shallow(root: Path) -> bool:
    return _text(git(root, "rev-parse", "--is-shallow-repository")) == "true"


def replace_refs(root: Path) -> List[str]:
    """Every refs/replace/* entry; each one substitutes one object for another."""
    return _text(git(root, "for-each-ref", "--format=%(refname)", "refs/replace/")).split()


def has_grafts(root: Path) -> bool:
    """True if the repository has an info/grafts file, the older way to rewrite parents."""
    rel = _text(git(root, "rev-parse", "--git-path", "info/grafts"))
    p = Path(rel)
    return (p if p.is_absolute() else Path(root) / p).exists()


def origin_url(root: Path, *, push: bool = False) -> Optional[str]:
    """
    The URL git fetches origin from (``push``: pushes to), with any
    ``insteadOf`` rewriting applied, so it is where the requests really go;
    None if there is no remote called origin.
    """
    proc = git(root, "remote", "get-url", *(["--push"] if push else []), "origin", check=False)
    return (_text(proc) or None) if proc.returncode == 0 else None


# ---------------------------------------------------------------- is the working tree HEAD?
#
# ``git status`` answers from the index, and the index can be told not to
# look: ``update-index --skip-worktree`` / ``--assume-unchanged`` hide any
# edit, and a clean filter set in the untracked .git/info/attributes and the
# local config can make an edited file hash like the committed one. Python,
# for its part, imports files git does not track at all. The three helpers
# below ask the questions the run scripts need answered instead, and
# tree_problems() puts them together. None of this stops someone who
# controls the machine; it makes a leftover flag or a stray file loud.

# File suffixes Python imports a module from, including extension modules'
# ABI-tagged ones (foo.cpython-310-darwin.so is the module foo).
_IMPORTABLE_SUFFIXES = tuple(sorted(set(importlib.machinery.all_suffixes()) | {".pyo", ".pyd"},
                                    key=len, reverse=True))


def index_flagged(root: Path) -> List[str]:
    """``"path (skip-worktree)"`` / ``"path (assume-unchanged)"`` for each index entry git is told to trust blindly."""
    out = git(root, "ls-files", "-v", "-z").stdout.decode("utf-8", "replace")
    flagged = []
    for entry in filter(None, out.split("\0")):
        tag, path = entry[0], entry[2:]
        if tag in "Ss":
            flagged.append(f"{path} (skip-worktree)")
        elif tag.islower():
            flagged.append(f"{path} (assume-unchanged)")
    return flagged


def worktree_mismatches(root: Path) -> List[str]:
    """
    ``"path (why)"`` for each path of HEAD's tree whose working-tree bytes are not HEAD's.

    Every file is hashed as a blob with ``git hash-object --no-filters``,
    which reads the bytes on disk and ignores clean filters, attributes,
    fsmonitor and the index's stat cache, and is compared with the blob id
    in ``git ls-tree HEAD``. A symlink is compared by its target, a missing
    or retyped file is a mismatch, and a submodule is refused outright: its
    content is another repository's, which this one cannot vouch for.
    """
    root = Path(root)
    out = git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD").stdout.decode("utf-8", "replace")
    problems: List[str] = []
    blobs: List[Tuple[str, str]] = []
    for record in filter(None, out.split("\0")):
        meta, path = record.split("\t", 1)
        mode, _, oid = meta.split()
        full = root / path
        if mode == "160000":
            problems.append(f"{path} (a submodule, which cannot be checked)")
        elif mode == "120000":
            target = os.readlink(os.fsencode(full)) if full.is_symlink() else None
            if target is None or _text(git(root, "hash-object", "--no-filters", "--stdin", input=target)) != oid:
                problems.append(f"{path} (symlink differs from HEAD)")
        elif full.is_symlink() or not full.is_file() or "\n" in path:
            # --stdin-paths reads one path per line, so a name holding a
            # newline cannot be hashed; it is refused like a missing file.
            problems.append(f"{path} (missing or not a regular file)")
        else:
            blobs.append((path, oid))
    if blobs:
        names = "".join(p + "\n" for p, _ in blobs).encode("utf-8")
        got = _text(git(root, "hash-object", "--no-filters", "--stdin-paths", input=names)).split()
        if len(got) != len(blobs):          # zip() below would quietly skip the rest
            raise GitError(f"git hash-object returned {len(got)} hashes for {len(blobs)} files")
        problems += [f"{p} (differs from HEAD)" for (p, oid), h in zip(blobs, got) if h != oid]
    return problems


def ignored_importable(root: Path) -> List[str]:
    """
    Ignored files Python could import from the checkout (``.py``, ``.pyc``, extension modules...).

    git status never shows them, yet a sourceless ``scripts/yaml.pyc``
    shadows PyYAML for ``python scripts/run_oos.py``. Excluded: anything in
    a ``__pycache__`` directory (``python scripts/<name>.py`` points Python's
    bytecode cache elsewhere, so it is not read; the scripts' startup block
    says what ``python -m`` reads first), and paths no import statement can
    reach because a directory or the module name is not a Python identifier
    (docs/dev-notes/*.py, for the hyphen).
    """
    out = git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z").stdout
    hits = []
    for rel in filter(None, out.decode("utf-8", "replace").split("\0")):
        *dirs, name = rel.split("/")
        suffix = next((s for s in _IMPORTABLE_SUFFIXES if name.endswith(s)), None)
        if suffix is None or "__pycache__" in dirs:
            continue
        if all(part.isidentifier() for part in [*dirs, name[: -len(suffix)]]):
            hits.append(rel)
    return hits


def _listed(items: List[str], limit: int = 5) -> str:
    return ", ".join(items[:limit]) + (f" and {len(items) - limit} more" if len(items) > limit else "")


def tree_problems(root: Path) -> List[str]:
    """Why the checkout may not be exactly HEAD's code although ``git status`` is clean; empty if it is."""
    problems = []
    flagged = index_flagged(root)
    if flagged:
        problems.append("the index tells git not to look at " + _listed(flagged)
                        + " (undo with git update-index --no-skip-worktree / --no-assume-unchanged)")
    differ = worktree_mismatches(root)
    if differ:
        problems.append("the working tree is not HEAD's, whatever git status says: " + _listed(differ))
    stray = ignored_importable(root)
    if stray:
        problems.append("ignored files Python could import: " + _listed(stray) + " (delete them)")
    return problems


def checkout_problems(root: Path) -> List[str]:
    """
    Why the checkout is not exactly HEAD; empty if it is.

    First what ``git status`` reports, untracked files included (code that
    is in no commit, a new research/ module say, would run unseen); once
    that is clean, what it cannot see (tree_problems). Ignored files that
    Python cannot import, such as data/history, do not count.
    """
    dirty = status_entries(root, include_untracked=True)
    if dirty:
        return ["the working tree is not clean: " + _listed(dirty, 10)]
    return tree_problems(root)


__all__ = [
    "FULL_SHA1", "GitError", "git", "is_work_tree", "status_entries", "head_commit", "current_branch",
    "tag_commit", "is_ancestor", "diff_names", "show_file", "show_bytes", "is_tracked",
    "path_status", "commit_of", "parents", "name_status", "commits_touching", "is_shallow",
    "replace_refs", "has_grafts", "origin_url", "index_flagged", "worktree_mismatches",
    "ignored_importable", "tree_problems", "checkout_problems",
]
