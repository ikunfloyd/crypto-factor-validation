"""
research/gitrepo.py: the questions that ``git status`` does not answer.

The run scripts must know that the files on disk are HEAD's and that Python
can import nothing git does not track. These tests build throwaway
repositories and use the ways git can be told not to look (index flags, a
local clean filter) or not to list (ignored files), and check that each
helper still sees the difference, and that what cannot matter does not trip
it. git runs with the user's global and system config disabled.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from research import gitrepo


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(k, raising=False)
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")


class Repo:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args):
        proc = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def write(self, rel, content):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        return p

    def commit(self, msg="c"):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    r = Repo(tmp_path / "repo")
    r.write(".gitignore", "__pycache__/\n*.py[cod]\n*.so\nbuild/\n")
    r.write("research/a.py", "A = 1\n")
    r.write("research/b.py", "B = 1\n")
    (r.path / "research" / "link.py").symlink_to("a.py")
    r.commit()
    return r


def test_a_clean_checkout_has_no_problems(repo):
    assert gitrepo.tree_problems(repo.path) == []


def test_index_flags(repo):
    repo.git("update-index", "--skip-worktree", "research/a.py")
    repo.git("update-index", "--assume-unchanged", "research/b.py")
    assert gitrepo.index_flagged(repo.path) == ["research/a.py (skip-worktree)", "research/b.py (assume-unchanged)"]
    repo.git("update-index", "--assume-unchanged", "research/a.py")        # both flags: 's'
    assert gitrepo.index_flagged(repo.path)[0] == "research/a.py (skip-worktree)"


def test_same_size_edit_under_a_clean_filter(repo):
    repo.write(".git/info/attributes", "research/a.py filter=tidy\n")
    repo.git("config", "filter.tidy.clean", "sed 's/^A = 2$/A = 1/'")
    repo.write("research/a.py", "A = 2\n")
    assert repo.git("status", "--porcelain") == ""                       # git is fooled
    assert gitrepo.worktree_mismatches(repo.path) == ["research/a.py (differs from HEAD)"]


def test_missing_retyped_and_retargeted_files(repo):
    (repo.path / "research/b.py").unlink()
    (repo.path / "research/link.py").unlink()
    (repo.path / "research/link.py").symlink_to("b.py")
    (repo.path / "research/a.py").unlink()
    (repo.path / "research/a.py").symlink_to("b.py")
    assert sorted(gitrepo.worktree_mismatches(repo.path)) == [
        "research/a.py (missing or not a regular file)",
        "research/b.py (missing or not a regular file)",
        "research/link.py (symlink differs from HEAD)",
    ]


def test_a_file_replaced_by_a_symlink_to_identical_bytes(repo):
    # git hash-object follows the link and would hash HEAD's blob; only git
    # status's typechange would show the swap otherwise.
    repo.write("data/a.txt", "A = 1\n")
    (repo.path / "research/a.py").unlink()
    (repo.path / "research/a.py").symlink_to("../data/a.txt")
    assert gitrepo.worktree_mismatches(repo.path) == ["research/a.py (missing or not a regular file)"]


def test_a_symlink_replaced_by_a_regular_file(repo):
    (repo.path / "research/link.py").unlink()
    repo.write("research/link.py", "A = 2\n")
    assert gitrepo.worktree_mismatches(repo.path) == ["research/link.py (symlink differs from HEAD)"]


def test_a_submodule_is_refused(repo):
    commit = repo.git("rev-parse", "HEAD")
    repo.git("update-index", "--add", "--cacheinfo", f"160000,{commit},vendored")
    repo.git("commit", "-q", "-m", "gitlink")
    assert gitrepo.worktree_mismatches(repo.path) == ["vendored (a submodule, which cannot be checked)"]


def test_ignored_files_python_could_import(repo):
    for rel in ("yaml.pyc", "scripts/yaml.pyc", "research/fast.cpython-310-darwin.so", "build/lib/research/a.py",
                # unreachable: __pycache__ is never read (a stray helper.pyc there included), and no
                # import names a hyphen or another Python's ABI tag
                "research/__pycache__/a.cpython-310.pyc", "research/__pycache__/helper.pyc",
                "docs/dev-notes/debug.py", "my-tool.pyc", "research/old.cpython-39-darwin.so"):
        repo.write(rel, b"x")
    repo.write(".git/info/exclude", "docs/dev-notes/\n")
    assert repo.git("status", "--porcelain") == ""
    assert sorted(gitrepo.ignored_importable(repo.path)) == [
        "build/lib/research/a.py", "research/fast.cpython-310-darwin.so", "scripts/yaml.pyc", "yaml.pyc"]


def test_tree_problems_names_each_kind(repo):
    repo.git("update-index", "--skip-worktree", "research/a.py")
    repo.write("research/b.py", "B = 22\n")
    repo.git("update-index", "--assume-unchanged", "research/b.py")
    repo.write("scripts/yaml.pyc", b"x")
    text = "; ".join(gitrepo.tree_problems(repo.path))
    for needle in ("research/a.py (skip-worktree)", "research/b.py (differs from HEAD)", "scripts/yaml.pyc"):
        assert needle in text


def test_which_commits_count_as_touching_a_path(repo):
    # A branch that adds out.csv, removes it again and is merged: the merge
    # leaves main as it was, so default history simplification hides both.
    main = repo.git("rev-parse", "HEAD")
    repo.git("checkout", "-q", "-b", "pr")
    repo.write("out.csv", "x\n")
    added = repo.commit("add")
    repo.git("rm", "-q", "out.csv")
    removed = repo.commit("remove")
    repo.git("checkout", "-q", "main")
    repo.write("research/a.py", "A = 2\n")
    repo.commit("main moves on")
    repo.git("merge", "-q", "--no-ff", "-m", "merge", "pr")
    repo.git("branch", "-q", "-D", "pr")
    assert repo.git("log", "--all", "--format=%H", "--", "out.csv") == ""
    assert gitrepo.commits_touching(repo.path, "out.csv", all_refs=True) == [removed, added]
    # A commit undone with a reset is in no ref; only the reflog has it.
    repo.write("undone.csv", "y\n")
    undone = repo.commit("to be undone")
    repo.git("reset", "-q", "--hard", "HEAD~1")
    assert gitrepo.commits_touching(repo.path, "undone.csv", all_refs=True) == []
    assert gitrepo.commits_touching(repo.path, "undone.csv", reflog=True) == [undone]
    # Reflog entries of reachable commits add nothing: b.py was committed once.
    assert gitrepo.commits_touching(repo.path, "research/b.py", reflog=True) == [main]


def test_a_file_system_monitor_does_not_answer_for_git_status(repo):
    # A hook that always says "nothing changed" hides a new untracked file
    # from plain git status, but not from the helpers.
    hook = repo.write(".git/quiet-fsmonitor", "#!/bin/sh\nprintf 'token\\0'\n")
    hook.chmod(0o755)
    repo.git("config", "core.untrackedCache", "true")
    repo.git("config", "core.fsmonitor", str(hook))
    repo.git("status", "--porcelain")
    repo.git("status", "--porcelain")
    repo.write("yaml.py", "def safe_load(text):\n    return {}\n")
    assert repo.git("status", "--porcelain") == ""
    assert gitrepo.status_entries(repo.path) == ["?? yaml.py"]
    assert gitrepo.checkout_problems(repo.path) == ["the working tree is not clean: ?? yaml.py"]


def test_origin_url(repo, tmp_path):
    assert gitrepo.origin_url(repo.path) is None
    repo.git("remote", "add", "origin", "https://example.invalid/o/r.git")
    repo.git("remote", "set-url", "--push", "origin", str(tmp_path / "mirror.git"))
    assert gitrepo.origin_url(repo.path) == "https://example.invalid/o/r.git"
    assert gitrepo.origin_url(repo.path, push=True) == str(tmp_path / "mirror.git")
