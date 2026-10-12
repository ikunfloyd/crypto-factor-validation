"""
research/freeze.py: the freeze amendment's machine-readable blocks and the
directions file they copy.

The format is what scripts/run_oos.py trusts to decide whether the one
out-of-sample run may start, so these tests check that a well-formed
amendment round-trips exactly and that every malformation is refused rather
than half-read. verify_freeze, which run_oos.py and scripts/fetch_v2.py
both rely on, is checked against throwaway git repositories, one broken
step of §10 steps 2-3 at a time; each pushes to a bare 'origin' that
freeze.PUBLIC_REMOTE is pointed at, standing in for the public repository.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pytest

from research import freeze, gitrepo

SHA = "0123456789abcdef0123456789abcdef01234567"


def insample_text(mutate=None):
    lines = [",".join(freeze.INSAMPLE_COLUMNS)]
    for i, (b, h, f) in enumerate(freeze.expected_cells()):
        row = [b, str(h), f, "-1" if i % 2 else "1", "false", "", "800", "0.5", "1.25", "hh", f"{2.5 + i / 3:.12g}"]
        if (b, h, f) == ("1H", 15, "basis_z_score"):
            row = [b, str(h), f, "1", "true", "missing_input", "0", "", "", "", ""]
        if mutate:
            row = mutate((b, h, f), row)
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def test_render_then_parse_round_trips():
    text = insample_text()
    blocks = freeze.render_amendment_blocks(SHA, text)
    doc = "# Freeze amendment\n\nSome prose, and a code sample:\n\n```python\nx = 1\n```\n\n" + blocks
    rec = freeze.parse_freeze_amendment(doc)
    assert rec.commit == SHA
    assert list(rec.cells) == freeze.expected_cells()
    ins = freeze.parse_insample_csv(text)
    assert freeze.compare_with_insample(rec, ins) == []
    undefined = rec.cells[("1H", 15, "basis_z_score")]
    assert undefined.flagged and undefined.flag_reason == "missing_input" and math.isnan(undefined.mde_bps)
    assert "undefined" in blocks
    assert rec.cells[("15m", 1, "rsi_14")] == freeze.FrozenCell(1, False, "", 2.5)


def test_rows_may_come_in_any_order():
    text = insample_text()
    head, *rows = freeze.render_amendment_blocks(SHA, text).split("```csv freeze-directions\n")[1].split("```")[0].splitlines()
    shuffled = f"```freeze-commit\n{SHA}\n```\n```csv freeze-directions\n" + "\n".join([head] + rows[::-1]) + "\n```\n"
    rec = freeze.parse_freeze_amendment(shuffled)
    assert freeze.compare_with_insample(rec, freeze.parse_insample_csv(text)) == []


def test_differences_are_all_reported():
    text = insample_text()
    rec = freeze.parse_freeze_amendment(freeze.render_amendment_blocks(SHA, text))

    def flip(key, row):
        if key == ("15m", 5, "hurst_regime"):
            row[3] = "1" if row[3] == "-1" else "-1"
        if key == ("1H", 1, "rsi_14"):
            row[10] = "9.75"
        return row

    problems = freeze.compare_with_insample(rec, freeze.parse_insample_csv(insample_text(flip)))
    assert len(problems) == 2
    assert any("hurst_regime" in p and "direction" in p for p in problems)
    assert any("rsi_14" in p and "mde_bps" in p for p in problems)


@pytest.mark.parametrize("doc,needle", [
    ("no blocks at all", "exactly one ```freeze-commit"),
    (f"```freeze-commit\n{SHA}\n```\n```freeze-commit\n{SHA}\n```\n", "found 2"),
    (f"```freeze-commit\n{SHA}\n```\n", "exactly one ```freeze-directions"),
])
def test_bad_amendments_are_refused(doc, needle):
    with pytest.raises(freeze.FreezeFormatError, match=needle.replace("(", r"\(")):
        freeze.parse_freeze_amendment(doc)


def _table(mutate=None, drop=None, extra=None, header=None):
    text = insample_text()
    blocks = freeze.render_amendment_blocks(SHA, text)
    table = blocks.split("```csv freeze-directions\n")[1].split("```")[0].splitlines()
    rows = table[1:]
    if mutate:
        rows = [mutate(r) for r in rows]
    if drop is not None:
        rows = rows[:drop] + rows[drop + 1:]
    if extra:
        rows.append(extra)
    return f"```freeze-commit\n{SHA}\n```\n```csv freeze-directions\n" + "\n".join([header or table[0]] + rows) + "\n```\n"


@pytest.mark.parametrize("doc,needle", [
    (_table().replace(SHA, SHA[:12], 1), "full 40-hex"),
    (_table().replace(SHA, "HEAD", 1), "full 40-hex"),
    (_table(drop=7), "lacks 1 cell"),
    (_table(extra="15m,1,rsi_14,1,false,,3.0"), "appears twice"),
    (_table(extra="5m,1,rsi_14,1,false,,3.0"), "not one of the study's cells"),
    (_table(mutate=lambda r: r.replace(",1,false,", ",0,false,", 1)), "direction must be 1 or -1"),
    (_table(mutate=lambda r: r.replace(",false,", ",no,", 1)), "flagged must be true or false"),
    (_table(mutate=lambda r: r.replace(",false,,", ",false,few_active,", 1)), "does not match flagged"),
    (_table(mutate=lambda r: r.rsplit(",", 1)[0] + ",n/a" if "rsi_14" in r else r), "mde_bps must be a number"),
    (_table(header="bar,horizon,factor,direction,flagged,mde_bps"), "header"),
])
def test_bad_tables_are_refused(doc, needle):
    with pytest.raises(freeze.FreezeFormatError, match=needle):
        freeze.parse_freeze_amendment(doc)


def test_a_flag_alone_differing_is_reported():
    # Directions and MDEs equal, only flagged/flag_reason differ: run_oos
    # takes the flags from the tag, so the amendment must agree on them too.
    text = insample_text()
    rec = freeze.parse_freeze_amendment(freeze.render_amendment_blocks(SHA, text))

    def flag(key, row):
        if key == ("15m", 15, "rsi_14"):
            row[4], row[5] = "true", "few_active"
        return row

    problems = freeze.compare_with_insample(rec, freeze.parse_insample_csv(insample_text(flag)))
    assert len(problems) == 1 and "rsi_14" in problems[0] and "flag" in problems[0]


def test_insample_file_must_have_exact_header_and_all_cells():
    with pytest.raises(freeze.FreezeFormatError, match="header"):
        freeze.parse_insample_csv(insample_text().replace("mean_s_fwd_bps", "mean_bps"))
    short = "\n".join(insample_text().splitlines()[:-1]) + "\n"
    with pytest.raises(freeze.FreezeFormatError, match="lacks 1 cell"):
        freeze.parse_insample_csv(short)


def test_render_refuses_a_short_commit():
    with pytest.raises(freeze.FreezeFormatError):
        freeze.render_amendment_blocks("abc123", insample_text())


# ---------------------------------------------------------------- verify_freeze

@pytest.fixture
def isolated_git(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(k, raising=False)
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")


@pytest.fixture
def public_remote(monkeypatch, tmp_path):
    """The bare 'origin' Repo.push creates beside tmp_path/'repo' plays the public repository."""
    monkeypatch.setattr(freeze, "PUBLIC_REMOTE", str(tmp_path / "repo-origin.git"))


AMENDMENT = "docs/preregistration-freeze.md"
DIRECTIONS = "results/insample_v2.csv"


class Repo:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args):
        proc = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def write(self, rel, text):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def add_origin(self):
        origin = self.path.parent / f"{self.path.name}-origin.git"
        if not origin.exists():
            subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True, capture_output=True)
            self.git("remote", "add", "origin", str(origin))

    def push(self):
        self.add_origin()
        self.git("push", "-q", "origin", "main")


def frozen(tmp_path, *, tag_extra=None, amend=True, push=True):
    """c1 code -> c2 adds insample_v2.csv [tag] -> c3 amendment -> push."""
    repo = Repo(tmp_path / "repo")
    repo.write("research/code.py", "X = 1\n")
    repo.c1 = repo.commit("code")
    repo.write(DIRECTIONS, insample_text())
    if tag_extra:
        repo.write(*tag_extra)
    repo.c2 = repo.commit("directions")
    repo.git("tag", "-a", "prereg-freeze", "-m", "freeze")
    if amend:
        repo.write(AMENDMENT, "# Freeze\n\n" + freeze.render_amendment_blocks(repo.c2, insample_text()))
        repo.c3 = repo.commit("amendment")
        if push:
            repo.push()
    return repo


@pytest.mark.usefixtures("isolated_git", "public_remote")
class TestVerifyFreeze:
    def refused(self, repo_or_path, needle):
        root = repo_or_path.path if isinstance(repo_or_path, Repo) else repo_or_path
        with pytest.raises(freeze.FreezeNotVerified, match=needle):
            freeze.verify_freeze(root)

    def test_a_freeze_done_as_written_verifies(self, tmp_path):
        repo = frozen(tmp_path)
        rec = freeze.verify_freeze(repo.path)
        assert rec.commit == repo.c2 == gitrepo.tag_commit(repo.path, "prereg-freeze")
        assert freeze.compare_with_insample(rec, freeze.parse_insample_csv(insample_text())) == []
        # Later commits (manifests, results) do not disturb it.
        repo.write("results/oos/manifest_doge_2024.json", "{}\n")
        repo.commit("manifest")
        assert freeze.verify_freeze(repo.path) == rec

    def test_not_a_repository(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / AMENDMENT).write_text("")
        self.refused(tmp_path, "not a git work tree")

    def test_no_amendment(self, tmp_path):
        self.refused(frozen(tmp_path, amend=False), "does not exist")

    def test_draft_amendment_is_not_committed(self, tmp_path):
        repo = frozen(tmp_path, amend=False)
        repo.write(AMENDMENT, freeze.render_amendment_blocks(repo.c2, insample_text()))
        self.refused(repo, "is not committed")

    @pytest.mark.parametrize("staged", [False, True])
    def test_amendment_with_uncommitted_changes(self, tmp_path, staged):
        repo = frozen(tmp_path)
        (repo.path / AMENDMENT).write_text("rewritten\n")
        if staged:
            repo.git("add", AMENDMENT)
        self.refused(repo, "has uncommitted changes")

    def test_amendment_without_blocks(self, tmp_path):
        repo = frozen(tmp_path, amend=False)
        repo.write(AMENDMENT, "The directions are as in the file.\n")
        repo.commit("amendment, prose only")
        repo.push()
        self.refused(repo, "freeze-commit")

    def test_no_tag(self, tmp_path):
        repo = frozen(tmp_path)
        repo.git("tag", "-d", "prereg-freeze")
        self.refused(repo, "there is no tag prereg-freeze")

    def test_tag_on_another_commit(self, tmp_path):
        repo = frozen(tmp_path)
        repo.git("tag", "-f", "-a", "prereg-freeze", "-m", "moved", repo.c1)
        self.refused(repo, "but the freeze amendment records")

    def test_tag_not_an_ancestor_of_head(self, tmp_path):
        repo = frozen(tmp_path)
        repo.git("checkout", "-q", "--orphan", "other")
        repo.git("rm", "-rq", "--cached", ".")
        repo.write("research/code.py", "X = 9\n")
        repo.commit("unrelated history")
        self.refused(repo, "is not an ancestor of HEAD")

    def test_directions_file_missing_at_the_tag(self, tmp_path):
        repo = Repo(tmp_path / "repo")
        repo.write("research/code.py", "X = 1\n")
        c1 = repo.commit("code")
        repo.git("tag", "prereg-freeze", c1)
        repo.write(AMENDMENT, freeze.render_amendment_blocks(c1, insample_text()))
        repo.commit("amendment")
        self.refused(repo, f"{DIRECTIONS} is not in the commit")

    def test_tagged_commit_changes_more_than_the_directions(self, tmp_path):
        repo = frozen(tmp_path, tag_extra=("research/code.py", "X = 2\n"))
        self.refused(repo, "must tag the commit that adds results/insample_v2.csv and nothing else")

    def test_tagged_commit_is_a_merge(self, tmp_path):
        repo = Repo(tmp_path / "repo")
        repo.write("research/code.py", "X = 1\n")
        repo.commit("code")
        repo.git("checkout", "-q", "-b", "side")
        repo.write("docs/note.md", "n\n")
        repo.commit("side")
        repo.git("checkout", "-q", "main")
        repo.write(DIRECTIONS, insample_text())
        repo.commit("directions")
        repo.git("merge", "-q", "--no-ff", "-m", "merge", "side")
        merge = repo.git("rev-parse", "HEAD")
        repo.git("tag", "prereg-freeze", merge)
        repo.write(AMENDMENT, freeze.render_amendment_blocks(merge, insample_text()))
        repo.commit("amendment")
        self.refused(repo, "2 parents")

    def test_amendment_disagrees_with_the_tag(self, tmp_path):
        repo = frozen(tmp_path, amend=False)
        flipped = insample_text(lambda key, row: [*row[:3], "-1" if row[3] == "1" else "1", *row[4:]]
                                if key == ("1H", 5, "ou_reversion") else row)
        repo.write(AMENDMENT, freeze.render_amendment_blocks(repo.c2, flipped))
        repo.commit("amendment")
        repo.push()
        self.refused(repo, r"\('1H', 5, 'ou_reversion'\): direction")

    def test_never_pushed(self, tmp_path):
        repo = frozen(tmp_path, push=False)
        self.refused(repo, "the remote origin is not set")
        repo.add_origin()
        self.refused(repo, "there is no refs/remotes/origin/main")

    def test_a_hand_written_remote_ref_is_not_a_push(self, tmp_path):
        # One `git update-ref` writes refs/remotes/origin/main; without the public origin it means nothing.
        repo = frozen(tmp_path, push=False)
        repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.refused(repo, "the remote origin is not set")

    def test_origin_is_not_the_public_repository(self, tmp_path, monkeypatch):
        repo = frozen(tmp_path)
        monkeypatch.setattr(freeze, "PUBLIC_REMOTE", "https://github.com/ikunfloyd/crypto-factor-validation.git")
        self.refused(repo, "the remote origin is .*, not https://github.com/ikunfloyd")

    def test_origin_pushes_elsewhere(self, tmp_path):
        # A push to a mirror also moves refs/remotes/origin/main.
        repo = frozen(tmp_path)
        repo.git("remote", "set-url", "--push", "origin", str(tmp_path / "mirror.git"))
        self.refused(repo, "the remote origin pushes to")

    def test_pushed_before_the_amendment_was_committed(self, tmp_path):
        repo = frozen(tmp_path, amend=False)
        repo.push()                                           # origin has the tag's commit only
        repo.write(AMENDMENT, freeze.render_amendment_blocks(repo.c2, insample_text()))
        repo.commit("amendment")
        self.refused(repo, "is not on refs/remotes/origin/main")

    def test_amendment_edited_after_the_push(self, tmp_path):
        repo = frozen(tmp_path)
        text = (repo.path / AMENDMENT).read_text()
        (repo.path / AMENDMENT).write_text(text.replace("# Freeze", "# Freeze amendment"))
        repo.commit("retitle, not pushed")
        self.refused(repo, "is not on refs/remotes/origin/main")
        repo.push()
        assert freeze.verify_freeze(repo.path).commit == repo.c2

    def test_amendment_is_read_from_the_commit_not_the_working_tree(self, tmp_path):
        # With --skip-worktree, git status calls the amendment clean whatever the file holds.
        repo = frozen(tmp_path)
        committed = freeze.verify_freeze(repo.path)
        amend = repo.path / AMENDMENT
        repo.git("update-index", "--skip-worktree", AMENDMENT)
        amend.write_text(amend.read_text().replace("15m,1,rsi_14,1,", "15m,1,rsi_14,-1,", 1))
        assert "15m,1,rsi_14,-1," in amend.read_text()                  # a direction flipped on disk
        assert gitrepo.path_status(repo.path, AMENDMENT) == []
        assert freeze.verify_freeze(repo.path) == committed


@pytest.mark.parametrize("url", [
    "https://github.com/ikunfloyd/crypto-factor-validation.git",
    "https://github.com/ikunfloyd/crypto-factor-validation",
    "https://github.com/IkunFloyd/crypto-factor-validation/",
    "git@github.com:ikunfloyd/crypto-factor-validation.git",
    "git@github.com:ikunfloyd/crypto-factor-validation",
    "ssh://git@github.com/ikunfloyd/crypto-factor-validation.git",
    "ssh://git@github.com:22/ikunfloyd/crypto-factor-validation",
])
def test_spellings_of_the_public_repository_compare_equal(url):
    assert freeze.is_public_remote(url)


@pytest.mark.parametrize("url", [
    "https://github.com/someone/crypto-factor-validation.git",        # a fork
    "https://github.com/ikunfloyd/crypto-factor-validation-mirror",
    "https://gitlab.com/ikunfloyd/crypto-factor-validation.git",
    "/srv/git/crypto-factor-validation.git",
])
def test_other_repositories_do_not(url):
    assert not freeze.is_public_remote(url)


def test_local_remotes_compare_by_resolved_path(tmp_path, monkeypatch):
    (tmp_path / "origin.git").mkdir()
    (tmp_path / "link.git").symlink_to(tmp_path / "origin.git")
    monkeypatch.setattr(freeze, "PUBLIC_REMOTE", str(tmp_path / "origin.git"))
    for url in (str(tmp_path / "link.git"), f"file://{tmp_path}/origin.git", "../origin.git"):
        assert freeze.is_public_remote(url, base=tmp_path / "repo"), url
    assert not freeze.is_public_remote(str(tmp_path / "other.git"))
