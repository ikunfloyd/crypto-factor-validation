"""
The guards of scripts/run_oos.py (§10 step 5) and scripts/run_insample_v2.py
(§10 step 2), each against a throwaway git repository.

Every test builds a small repository under tmp_path that has the shape the
real one will have at run time: code and the real docs/preregistration.md,
then a commit adding only results/insample_v2.csv (tagged prereg-freeze),
the freeze amendment in a later commit pushed to a bare 'origin' (which
stands in for the public repository: freeze.PUBLIC_REMOTE is pointed at it),
and the set's manifest (window, unlock flag, remote record, hashes),
committed and pushed, with its (gitignored) data file, a two-row parquet
spanning the window; for the primary set, also the two secondary outputs,
written by this script with its computation stubbed. Each test then breaks
exactly one thing and checks that the run stops for that reason, before
computing the set and without writing an output. The heavy computation is a
stub, which also stands in for guard 9's recomputation of the secondary
sets: what is tested here is the refusing, not the statistics
(tests/test_research_pipeline.py covers those).

git runs with the user's global and system config disabled, and with a fixed
identity in the environment, so nothing outside tmp_path is read or changed.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import marshal
import math
import os
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import polars as pl
import pytest
from loguru import logger

from research import freeze, gitrepo, integrity
from research import pipeline as P
from research.io import sha256_file
from research.spec import MINUTE_MS, PREREG_PATH, REPO_ROOT, get_spec


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ro = _load("run_oos")
ri = _load("run_insample_v2")
SPEC = get_spec()
FWD_END = SPEC["sets"]["doge_fwd"]["end_ms"]
# The pre-registration as this process reads it: the throwaway repositories
# commit it, because run_oos compares its Appendix A at the tag with get_spec().
PREREG_REL = ro.PREREG_REL
PREREG_TEXT = PREREG_PATH.read_text(encoding="utf-8")
DAY_MS = 86_400_000
# What the real .gitignore says about Python's own files, so a stray .pyc is ignored, as it would be.
GITIGNORE = "data/history/\n__pycache__/\n*.py[cod]\n*.so\n"


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(k, raising=False)
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")


@pytest.fixture(autouse=True)
def public_remote(monkeypatch, tmp_path):
    """The bare 'origin' Repo.push creates beside tmp_path/'repo' plays the public repository."""
    monkeypatch.setattr(freeze, "PUBLIC_REMOTE", str(tmp_path / "repo-origin.git"))


@pytest.fixture
def messages():
    out = []
    hid = logger.add(lambda m: out.append(m.record["message"]), level="INFO")
    yield out
    logger.remove(hid)


def joined(messages):
    return "\n".join(messages)


# ---------------------------------------------------------------- fixtures

def insample_text(flip=None):
    """A well-formed 60-row results/insample_v2.csv."""
    lines = [",".join(freeze.INSAMPLE_COLUMNS)]
    for i, key in enumerate(freeze.expected_cells()):
        b, h, f = key
        d = -1 if i % 3 == 0 else 1
        if key == flip:
            d = -d
        if key == ("1H", 15, "funding_cost_pressure"):           # no active bars: blanks
            lines.append(f"{b},{h},{f},1,true,missing_input,0,,,,")
        elif b == "1H" and f == "triple_screen":
            lines.append(f"{b},{h},{f},1,true,few_active,120,0.5,2.0,hh,{7.972 * 1.1:.10g}")
        else:
            lines.append(f"{b},{h},{f},{d},false,,900,{d * 0.25 * (i + 1)},{1 + i / 10},hh,{3.986 * (1 + i / 10):.10g}")
    return "\n".join(lines) + "\n"


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self.origin = None
        path.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")

    def git(self, *args):
        proc = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def write(self, rel, content):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)
        return p

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def add_origin(self):
        """A bare 'origin' beside the repository, the one freeze.PUBLIC_REMOTE names in these tests."""
        if self.origin is None:
            self.origin = self.path.parent / f"{self.path.name}-origin.git"
            subprocess.run(["git", "init", "-q", "--bare", str(self.origin)], check=True, capture_output=True)
            self.git("remote", "add", "origin", str(self.origin))

    def push(self):
        """Push main to origin (created on first use)."""
        self.add_origin()
        self.git("push", "-q", "origin", "main")


def parquet(**columns):
    buf = io.BytesIO()
    pl.DataFrame(columns).write_parquet(buf)
    return buf.getvalue()


def add_set_inputs(repo: Repo, set_name: str, commit=True, *, window=None, unlocked=True,
                   data=None, funding=None, edit=None, push=True):
    """
    The set's (fake) data, gitignored like data/history, and its manifest
    with the keys fetch_v2 writes, committed and (if the repository has an
    origin and ``push``) pushed, as §10 step 4 has it. By default the
    manifest's window is exactly as wide as run_oos requires, test data is
    unlocked, and the remote record names origin and the main it held.

    Each perpetual gets a candle file of two rows, its first and last
    timestamps (``data``, default: the window's first and last minute);
    ``funding`` adds a funding file per perpetual likewise; ``edit`` may
    change the manifest dict before it is written.
    """
    lo, hi = ro.manifest_window_required(set_name)
    files = [("candles", "candles_1m.parquet", data or (lo, hi - MINUTE_MS))]
    if funding:
        files.append(("funding", "funding_rates.parquet", funding))
    series = []
    for inst in SPEC["sets"][set_name]["inst"]:
        for kind, name, (first, last) in files:
            rel = f"data/history/oos/{set_name}/{inst}/{name}"
            path = repo.write(rel, parquet(timestamp=[first, last]))
            series.append({"path": rel, "sha256": sha256_file(path), "kind": kind})
    lo, hi = (lo, hi) if window is None else window
    seen = "refs/remotes/origin/main" if repo.origin is not None else "HEAD"
    manifest = {"manifest_version": 1, "window": {"start_ms": lo, "end_ms": hi, "half_open": True},
                "test_data_unlocked": unlocked, "remote_url": freeze.PUBLIC_REMOTE,
                "remote_main_commit": repo.git("rev-parse", seen), "series": series}
    if edit is not None:
        edit(manifest)
    repo.write(P.oos_manifest_path(set_name).as_posix(), json.dumps(manifest, indent=2))
    if commit:
        repo.commit(f"manifest {set_name}")
        if push and repo.origin is not None:
            repo.push()


def run_secondaries(repo: Repo, tamper=None):
    """
    Run doge_2024 and majors_2023 through run_oos (stub computation) and
    commit each output, as §10 step 5 does before the primary set. ``tamper``
    may rewrite an output's text before it is committed.
    """
    for other in ("doge_2024", "majors_2023"):
        add_set_inputs(repo, other)
        code, _ = run_oos(repo, other)
        assert code == ro.EXIT_OK
        out = repo.path / P.oos_results_path(other)
        if tamper is not None:
            out.write_text(tamper(other, out.read_text()))
        repo.commit(f"{other} results")


def build(tmp_path, set_name="doge_2024", *, amendment=True, amendment_text=None, record=None,
          insample=None, manifest=True, prerequisites=False, push=True, prereg=None):
    """
    The repository at run time:
      c1 code + docs/preregistration.md -> c2 results/insample_v2.csv [tag prereg-freeze]
      -> c3 amendment (pushed) [-> secondary manifests and outputs] -> manifest.
    """
    repo = Repo(tmp_path / "repo")
    repo.write(".gitignore", GITIGNORE)
    repo.write("research/code.py", "X = 1\n")
    repo.write(PREREG_REL, PREREG_TEXT if prereg is None else prereg)
    repo.c1 = repo.commit("code")
    text = insample_text() if insample is None else insample
    repo.write("results/insample_v2.csv", text)
    repo.c2 = repo.commit("in-sample directions")
    repo.git("tag", "-a", "prereg-freeze", "-m", "freeze")
    if amendment:
        body = amendment_text if amendment_text is not None else \
            freeze.render_amendment_blocks(record or repo.c2, text)
        repo.write("docs/preregistration-freeze.md", "# Freeze amendment\n\nProse.\n\n" + body)
        repo.commit("freeze amendment")
        if push:
            repo.push()
    if prerequisites:
        run_secondaries(repo)
    if manifest:
        add_set_inputs(repo, set_name)
    return repo


class Stub:
    def __init__(self):
        self.calls = []

    def __call__(self, set_name, directions, *, root, mde_bps, code_commit, manifest_sha256):
        self.calls.append({"set": set_name, "directions": directions, "root": root, "mde": mde_bps,
                           "code_commit": code_commit, "manifest_sha256": manifest_sha256})
        # Provenance columns as the real run fills them; the rest is filler.
        main = pl.DataFrame({c: pl.Series(c, ["x"], dtype=pl.Utf8) for c in P.OOS_COLUMNS}).with_columns(
            pl.lit(code_commit).alias("code_commit"), pl.lit(manifest_sha256).alias("manifest_sha256"))
        robust = None
        if P.set_role(set_name) == P.ROLE_PRIMARY:
            robust = pl.DataFrame({c: pl.Series(c, ["y"], dtype=pl.Utf8) for c in P.ROBUSTNESS_COLUMNS})
        return main, robust


def run_oos(repo, set_name="doge_2024", now_ms=FWD_END + 1):
    stub = Stub()
    code = ro.main(["--set", set_name], root=repo.path, compute=stub, now_ms=now_ms)
    return code, stub


def outputs(repo, set_name):
    return [p for p in ro.outputs_for(set_name) if (repo.path / p).exists()]


def move_untracked_to(repo, branch, msg):
    """Commit the untracked files on ``branch`` instead of main, and go back to main without them."""
    repo.git("stash", "-u", "-q")
    repo.git("checkout", "-q", branch)
    repo.git("stash", "pop", "-q")
    repo.commit(msg)
    repo.git("checkout", "-q", "main")


def commit_out_of_sight(repo, outs, how):
    """
    Commit a run's (untracked) outputs ``outs`` where plain ``git log --all`` no longer finds them.

    ``merged_branch``, the PR workflow: committed on a branch, removed again
    there ("outputs do not belong in this PR"), merged with --no-ff, branch
    deleted. The merge leaves main as it was, so git's default history
    simplification prunes the branch. The reflog is cleared, as in a clone
    made after the merge, so that only the history can tell. ``reset``:
    committed on main, then "undo that commit" with git reset --hard
    HEAD~1; only the reflog still reaches the commit.
    """
    if how == "merged_branch":
        repo.git("stash", "-u", "-q")
        repo.git("checkout", "-q", "-b", "pr")
        repo.git("stash", "pop", "-q")
        repo.commit("first run's output, on a PR branch")
        repo.git("rm", "-q", *outs)
        repo.commit("outputs do not belong in this PR")
        repo.git("checkout", "-q", "main")
        repo.git("merge", "-q", "--no-ff", "-m", "merge the PR", "pr")
        repo.git("branch", "-q", "-D", "pr")
        repo.git("reflog", "expire", "--expire=now", "--all")
    else:
        repo.commit("first run's output")
        repo.git("reset", "-q", "--hard", "HEAD~1")
    assert repo.git("log", "--all", "--format=%H", "--", *outs) == ""          # plain git finds nothing


def assert_refused(repo, set_name, code, stub, messages, needle, exit_code=ro.EXIT_REFUSED):
    """Refused for the reason ``needle``, before ``set_name`` was computed (guard 9 may have recomputed others)."""
    assert code == exit_code, joined(messages)
    assert [c["set"] for c in stub.calls if c["set"] == set_name] == []
    assert needle in joined(messages), joined(messages)


# ---------------------------------------------------------------- passing runs

class TestPasses:
    def test_secondary_set_runs_and_writes_once(self, tmp_path, messages):
        repo = build(tmp_path)
        code, stub = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)
        (call,) = stub.calls
        assert call["code_commit"] == repo.git("rev-parse", "HEAD")
        assert call["manifest_sha256"] == sha256_file(repo.path / P.oos_manifest_path("doge_2024"))
        parsed = freeze.parse_insample_csv(insample_text())
        assert call["directions"] == P.directions_from_insample(parsed)
        assert call["mde"][("15m", 1, "rsi_14")] == parsed[("15m", 1, "rsi_14")]["mde_bps"]
        assert outputs(repo, "doge_2024") == [P.oos_results_path("doge_2024")]
        assert pl.read_csv(repo.path / P.oos_results_path("doge_2024")).columns == list(P.OOS_COLUMNS)

    def test_primary_set_runs_last_and_writes_robustness(self, tmp_path, messages):
        repo = build(tmp_path, "doge_2023", prerequisites=True)
        code, stub = run_oos(repo, "doge_2023")
        assert code == ro.EXIT_OK, joined(messages)
        assert outputs(repo, "doge_2023") == [P.oos_results_path("doge_2023"),
                                              P.robustness_results_path("doge_2023")]

    def test_later_docs_and_results_commits_are_allowed(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.write("docs/notes.md", "a note\n")
        repo.write("results/oos/rerun-1/README.md", "why\n")
        repo.commit("docs and results only")
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)

    def test_prose_and_deviation_rows_may_follow_the_tag(self, tmp_path, messages):
        # Only Appendix A is frozen; the §12 log may grow and prose may change.
        repo = build(tmp_path)
        row = "| 2026-12-01 | §10 | A dated note. | Why. | Yes: doge_2024's results. |\n"
        text = PREREG_TEXT.replace("\n## References", row + "\n## References", 1)
        assert text != PREREG_TEXT
        repo.write(PREREG_REL, text.replace("**Summary.**", "**Summary (amended prose).**", 1))
        repo.commit("docs: log a deviation")
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)

    def test_doge_fwd_manifest_starts_at_the_end_of_is(self, tmp_path, messages):
        # Its warm-up is in-sample data, read from the in-sample files (§4.1).
        is_end, fwd = SPEC["sets"]["IS"]["end_ms"], SPEC["sets"]["doge_fwd"]
        assert ro.manifest_window_required("doge_fwd") == (is_end, fwd["end_ms"])
        assert ro.manifest_window_required("doge_2023") == (SPEC["sets"]["doge_2023"]["warmup_start_ms"],
                                                            SPEC["sets"]["doge_2023"]["end_ms"])


# ---------------------------------------------------------------- refusals

class TestRefuses:
    def test_no_amendment(self, tmp_path, messages):
        repo = build(tmp_path, amendment=False)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "preregistration-freeze.md does not exist")
        assert outputs(repo, "doge_2024") == []

    def test_amendment_without_machine_readable_blocks(self, tmp_path, messages):
        repo = build(tmp_path, amendment_text="The directions are as in the file.\n")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "freeze-commit")

    def test_tag_resolves_to_another_commit_than_recorded(self, tmp_path, messages):
        repo = Repo(tmp_path / "pre")  # only to get a valid-looking foreign hash
        repo.write("x", "x")
        foreign = repo.commit("x")
        repo = build(tmp_path, record=foreign)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "but the freeze amendment records")

    @pytest.mark.parametrize("origin", [False, True])
    def test_amendment_not_pushed(self, tmp_path, messages, origin):
        repo = build(tmp_path, push=False)
        if origin:
            repo.add_origin()                                   # the public repository, never pushed to
        code, stub = run_oos(repo)
        needle = "there is no refs/remotes/origin/main" if origin else "the remote origin is not set"
        assert_refused(repo, "doge_2024", code, stub, messages, needle)

    def test_hand_written_remote_ref_without_a_remote(self, tmp_path, messages):
        # One `git update-ref` makes a push appear; it means nothing without the public origin.
        repo = build(tmp_path, push=False, manifest=False)
        repo.git("update-ref", "refs/remotes/origin/main", "HEAD")
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "the remote origin is not set")

    @pytest.mark.parametrize("which", ["fetch", "push"])
    def test_origin_is_not_the_public_repository(self, tmp_path, messages, monkeypatch, which):
        repo = build(tmp_path)
        if which == "fetch":
            monkeypatch.setattr(freeze, "PUBLIC_REMOTE", "https://github.com/ikunfloyd/crypto-factor-validation.git")
        else:
            repo.git("remote", "set-url", "--push", "origin", str(tmp_path / "mirror.git"))
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages,
                       "the remote origin " + ("is " if which == "fetch" else "pushes to"))

    def test_amendment_changed_after_the_push(self, tmp_path, messages):
        # The push must cover the amendment in force, not an earlier version.
        repo = build(tmp_path, manifest=False)
        path = repo.path / "docs/preregistration-freeze.md"
        path.write_text(path.read_text().replace("Prose.", "Prose, edited."))
        repo.commit("edit the amendment, unpushed")
        add_set_inputs(repo, "doge_2024", push=False)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "is not on refs/remotes/origin/main")

    def test_tagged_commit_also_changes_code(self, tmp_path, messages):
        # §10 step 2: insample_v2.csv "is added in the next commit with no
        # other change". A code change inside the tagged commit is invisible
        # to the tag..HEAD diff, so the tagged commit's own change is checked.
        repo = Repo(tmp_path / "repo")
        repo.write(".gitignore", "data/history/\n")
        repo.write("research/code.py", "X = 1\n")
        repo.write(PREREG_REL, PREREG_TEXT)
        repo.commit("code that produced the directions")
        repo.write("results/insample_v2.csv", insample_text())
        repo.write("research/code.py", "X = 2  # changed after seeing the in-sample file\n")
        c2 = repo.commit("directions + code change")
        repo.git("tag", "-a", "prereg-freeze", "-m", "freeze")
        repo.write("docs/preregistration-freeze.md", freeze.render_amendment_blocks(c2, insample_text()))
        repo.commit("freeze amendment")
        repo.push()
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "must tag the commit that adds results/insample_v2.csv")
        assert "M research/code.py" in joined(messages)

    def test_tag_missing(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.git("tag", "-d", "prereg-freeze")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "there is no tag prereg-freeze")

    def test_branch_named_like_the_tag_is_not_the_tag(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.git("tag", "-d", "prereg-freeze")
        repo.git("branch", "prereg-freeze", repo.c2)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "there is no tag prereg-freeze")

    def test_tag_not_an_ancestor_of_head(self, tmp_path, messages):
        repo = Repo(tmp_path / "repo")
        repo.write(".gitignore", "data/history/\n")
        repo.write("research/code.py", "X = 1\n")
        repo.commit("code")
        repo.git("checkout", "-q", "-b", "side")
        repo.write("results/insample_v2.csv", insample_text())
        side = repo.commit("directions on a side branch")
        repo.git("tag", "prereg-freeze", side)
        repo.git("checkout", "-q", "main")
        repo.write("results/insample_v2.csv", insample_text())
        repo.write("docs/preregistration-freeze.md", freeze.render_amendment_blocks(side, insample_text()))
        repo.commit("directions and amendment on main")
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "is not an ancestor of HEAD")

    def test_code_change_after_the_tag(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.write("research/code.py", "X = 2\n")
        repo.commit("a 'harmless' fix")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "research/code.py")
        assert outputs(repo, "doge_2024") == []

    def test_moving_code_into_docs_still_counts_as_a_code_change(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.git("mv", "research/code.py", "docs/code.py")
        repo.commit("move")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "research/code.py")

    def test_appendix_a_edited_after_the_tag(self, tmp_path, messages):
        # Every constant of the run is read from Appendix A, under docs/,
        # which guard 4 allows to change. A post-tag edit is refused.
        repo = build(tmp_path, manifest=False)
        edited = PREREG_TEXT.replace("cost_round_trip_bps: 10.0", "cost_round_trip_bps: 0.0", 1)
        edited = edited.replace("min_active_trades: 300", "min_active_trades: 30", 1)
        assert edited != PREREG_TEXT
        repo.write(PREREG_REL, edited)
        repo.commit("docs: wording")
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "Appendix A of docs/preregistration.md differs")
        assert outputs(repo, "doge_2024") == []

    def test_spec_in_use_differs_from_the_tag(self, tmp_path, messages):
        # The repository is consistent with itself, but its Appendix A is not
        # the one this process computes with (get_spec()).
        other = PREREG_TEXT.replace("cost_round_trip_bps: 10.0", "cost_round_trip_bps: 12.5", 1)
        repo = build(tmp_path, prereg=other)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "the specification this process loaded")

    def test_prereg_missing_at_the_tag(self, tmp_path, messages):
        repo = build(tmp_path, manifest=False)
        repo.git("rm", "-q", PREREG_REL)
        repo.commit("docs: remove")
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "does not exist at HEAD")

    @pytest.mark.parametrize("rel", ["results/insample_v2.csv", "results/manifest_is_supp.json"])
    def test_files_frozen_before_the_tag_changed_after_it(self, tmp_path, messages, rel):
        repo = build(tmp_path, manifest=False)
        repo.write(rel, insample_text(flip=("1H", 5, "ou_reversion")) if rel.endswith(".csv") else "{}\n")
        repo.commit(f"results: {rel}")
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, f"{rel} changed since prereg-freeze")

    def test_direction_differs_from_the_amendment(self, tmp_path, messages):
        key = ("1H", 5, "ou_reversion")
        amended = freeze.render_amendment_blocks("0" * 40, insample_text(flip=key))
        repo = build(tmp_path, amendment=False, manifest=False)
        repo.write("docs/preregistration-freeze.md", amended.replace("0" * 40, repo.c2))
        repo.commit("amendment with one direction flipped")
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "('1H', 5, 'ou_reversion'): direction")

    def test_mde_differs_from_the_amendment(self, tmp_path, messages):
        repo = build(tmp_path, amendment=False, manifest=False)
        good = freeze.render_amendment_blocks(repo.c2, insample_text())
        line = next(l for l in good.splitlines() if l.startswith("15m,1,rsi_14,"))
        repo.write("docs/preregistration-freeze.md", good.replace(line, line.rsplit(",", 1)[0] + ",4.0"))
        repo.commit("amendment with a retyped MDE")
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "mde_bps")

    def test_directions_file_missing_at_the_tag(self, tmp_path, messages):
        repo = Repo(tmp_path / "repo")
        repo.write(".gitignore", "data/history/\n")
        repo.write("research/code.py", "X = 1\n")
        c1 = repo.commit("code")
        repo.git("tag", "prereg-freeze", c1)
        repo.write("docs/preregistration-freeze.md", freeze.render_amendment_blocks(c1, insample_text()))
        repo.commit("amendment")
        add_set_inputs(repo, "doge_2024")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "results/insample_v2.csv is not in the commit")

    def test_manifest_hash_mismatch(self, tmp_path, messages):
        repo = build(tmp_path)
        data = repo.path / "data/history/oos/doge_2024/DOGE-USDT-SWAP/candles_1m.parquet"
        data.write_bytes(b"re-downloaded after the manifest was committed")
        assert repo.git("status", "--porcelain") == ""             # data/history is ignored
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "differs from")

    def test_input_not_listed_in_the_manifest(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.write("data/history/oos/doge_2024/DOGE-USDT/candles_1m.parquet", b"an unlisted spot leg")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "no manifest lists it")

    def test_manifest_not_committed(self, tmp_path, messages):
        repo = build(tmp_path, manifest=False)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "manifest_doge_2024.json is not committed")

    @pytest.mark.parametrize("which", [
        "short",          # stops before the evaluation window ends
        "no_warmup",      # starts at the evaluation window, without its warm-up
        "missing",        # a manifest without any window
    ])
    def test_manifest_window_does_not_cover_the_set(self, tmp_path, messages, which):
        repo = build(tmp_path, manifest=False)
        lo, hi = ro.manifest_window_required("doge_2024")
        window = {"short": (lo, hi - 86_400_000),
                  "no_warmup": (SPEC["sets"]["doge_2024"]["start_ms"], hi)}.get(which, (lo, hi))
        add_set_inputs(repo, "doge_2024", commit=False, window=window)
        if which == "missing":
            mp = repo.path / P.oos_manifest_path("doge_2024")
            m = json.loads(mp.read_text())
            del m["window"]
            mp.write_text(json.dumps(m))
        repo.commit("manifest")
        code, stub = run_oos(repo)
        needle = "has no readable window" if which == "missing" else "but doge_2024 needs"
        assert_refused(repo, "doge_2024", code, stub, messages, needle)

    def test_doge_fwd_short_manifest_window(self, tmp_path, messages):
        # A partial doge_fwd download (§4.1: the stop date is fixed), with the
        # supplement manifest committed before the tag as §10 step 2 has it.
        repo = Repo(tmp_path / "repo")
        repo.write(".gitignore", "data/history/\n")
        repo.write("research/code.py", "X = 1\n")
        repo.write(PREREG_REL, PREREG_TEXT)
        repo.write(P.IS_SUPP_MANIFEST.as_posix(), json.dumps({"series": []}))
        repo.commit("code, prereg, supplement manifest")
        repo.write("results/insample_v2.csv", insample_text())
        c2 = repo.commit("in-sample directions")
        repo.git("tag", "-a", "prereg-freeze", "-m", "freeze")
        repo.write("docs/preregistration-freeze.md", freeze.render_amendment_blocks(c2, insample_text()))
        repo.commit("freeze amendment")
        repo.push()
        lo, hi = ro.manifest_window_required("doge_fwd")
        add_set_inputs(repo, "doge_fwd", window=(lo, hi - 3_600_000))   # a partial download
        code, stub = run_oos(repo, "doge_fwd", now_ms=FWD_END)
        assert_refused(repo, "doge_fwd", code, stub, messages, "but doge_fwd needs")

    def test_manifest_not_fetched_under_a_verified_freeze(self, tmp_path, messages):
        repo = build(tmp_path, manifest=False)
        add_set_inputs(repo, "doge_2024", unlocked=False)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "test_data_unlocked")

    @pytest.mark.parametrize("order", ["before", "together"])
    def test_manifest_committed_before_the_amendment(self, tmp_path, messages, order):
        # §10: step 3 (amendment, pushed) comes before step 4 (download, manifest).
        repo = build(tmp_path, amendment=False, manifest=False)
        add_set_inputs(repo, "doge_2024", commit=(order == "before"))
        repo.write("docs/preregistration-freeze.md", freeze.render_amendment_blocks(repo.c2, insample_text()))
        repo.commit("freeze amendment" + (" and manifest" if order == "together" else ""))
        repo.push()
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "does not come after the commit that added")

    @pytest.mark.parametrize("how", ["modified", "untracked", "staged"])
    def test_dirty_tree(self, tmp_path, messages, how):
        repo = build(tmp_path)
        if how == "modified":
            repo.write("research/code.py", "X = 3\n")
        elif how == "untracked":
            repo.write("research/new_helper.py", "Y = 1\n")
        else:
            repo.write("docs/notes.md", "staged\n")
            repo.git("add", "docs/notes.md")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "working tree is not clean")

    def test_not_on_main(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.git("checkout", "-q", "-b", "experiment")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "not the branch main")
        repo.git("checkout", "-q", "--detach", "main")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "HEAD is detached")

    def test_output_exists(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.write(P.oos_results_path("doge_2024").as_posix(), "first run\n")
        repo.commit("first run's output")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "refusing to overwrite", ro.EXIT_EXISTS)
        assert (repo.path / P.oos_results_path("doge_2024")).read_text() == "first run\n"

    def test_output_removed_in_git_then_rerun(self, tmp_path, messages):
        # The first run stays primary (§10 step 6) even after a `git rm`.
        repo = build(tmp_path)
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)
        repo.commit("first run's output")
        repo.git("rm", "-q", P.oos_results_path("doge_2024").as_posix())
        repo.commit("remove it")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "was committed before", ro.EXIT_EXISTS)
        assert outputs(repo, "doge_2024") == []

    def test_first_run_committed_on_another_branch_only(self, tmp_path, messages):
        # Write-once asks every ref, not only main's history.
        repo = build(tmp_path)
        repo.git("branch", "side")
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)
        move_untracked_to(repo, "side", "first run's output, on a side branch")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "was committed before", ro.EXIT_EXISTS)

    @pytest.mark.parametrize("how", ["merged_branch", "reset"])
    def test_first_run_committed_out_of_sight_still_counts(self, tmp_path, messages, how):
        # Two ordinary ways to lose sight of a first run's commit (commit_out_of_sight).
        repo = build(tmp_path)
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)
        commit_out_of_sight(repo, [P.oos_results_path("doge_2024").as_posix()], how)
        assert outputs(repo, "doge_2024") == []
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "was committed before", ro.EXIT_EXISTS)

    def test_primary_before_its_secondaries(self, tmp_path, messages):
        repo = build(tmp_path, "doge_2023")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages, "run --set doge_2024 first")

    def test_hand_made_secondary_file(self, tmp_path, messages):
        # Tracked, with the §10 columns, but written by hand: no provenance.
        repo = build(tmp_path, "doge_2023", manifest=False)
        for other in ("doge_2024", "majors_2023"):
            df = pl.DataFrame({c: pl.Series(c, ["rejected" if c == "status" else ""], dtype=pl.Utf8)
                               for c in P.OOS_COLUMNS})
            repo.write(P.oos_results_path(other).as_posix(), df.write_csv())
        repo.commit("secondary results")
        add_set_inputs(repo, "doge_2023")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages, "existing 40-hex code_commit")

    def test_secondary_file_edited_after_its_commit(self, tmp_path, messages):
        repo = build(tmp_path, "doge_2023", manifest=False, prerequisites=True)
        out = repo.path / P.oos_results_path("doge_2024")
        out.write_text(out.read_text().replace(",x,", ",rejected,", 1))
        repo.commit("results: tidy doge_2024")
        add_set_inputs(repo, "doge_2023")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages, "doge_2024.csv was committed 2 times")

    def test_secondary_edited_on_another_branch_counts(self, tmp_path, messages):
        # A tidied copy kept on a side branch (left over from a squash-merge, say) is a second commit.
        repo = build(tmp_path, "doge_2023", manifest=False, prerequisites=True)
        repo.git("checkout", "-q", "-b", "side")
        out = repo.path / P.oos_results_path("doge_2024")
        out.write_text(out.read_text().replace(",x,", ",rejected,", 1))
        repo.commit("an edited doge_2024.csv, kept on a side branch")
        repo.git("checkout", "-q", "main")
        add_set_inputs(repo, "doge_2023")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages, "doge_2024.csv was committed 2 times")

    def test_secondary_manifest_hash_is_not_its_manifest(self, tmp_path, messages):
        repo = build(tmp_path, "doge_2023", manifest=False, amendment=True)

        def wrong_sha(other, text):
            df = pl.read_csv(io.StringIO(text), infer_schema_length=0)
            return df.with_columns(pl.lit("0" * 64).alias("manifest_sha256")).write_csv()

        run_secondaries(repo, tamper=wrong_sha)
        add_set_inputs(repo, "doge_2023")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages,
                       "is not the SHA-256 of results/oos/manifest_doge_2024.json")

    def test_secondary_computed_on_other_code(self, tmp_path, messages):
        # code_commit names a commit whose code is not the tag's.
        repo = build(tmp_path, "doge_2023", manifest=False)
        repo.git("checkout", "-q", "-b", "tweak")
        repo.write("research/code.py", "X = 3\n")
        tweak = repo.commit("tweaked code")
        repo.git("checkout", "-q", "main")

        def other_code(other, text):
            df = pl.read_csv(io.StringIO(text), infer_schema_length=0)
            return df.with_columns(pl.lit(tweak).alias("code_commit")).write_csv()

        run_secondaries(repo, tamper=other_code)
        add_set_inputs(repo, "doge_2023")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages, "whose code differs from prereg-freeze")

    def test_secondary_computed_under_another_appendix_a(self, tmp_path, messages):
        # code_commit changes only docs/ after the tag, but its Appendix A is not the tag's
        # (edited, used, then reverted on main).
        repo = build(tmp_path, "doge_2023", manifest=False)
        add_set_inputs(repo, "doge_2024")
        add_set_inputs(repo, "majors_2023")
        repo.write(PREREG_REL, PREREG_TEXT.replace("cost_round_trip_bps: 10.0", "cost_round_trip_bps: 4.0", 1))
        edited = repo.commit("docs: Appendix A edited")
        repo.write(PREREG_REL, PREREG_TEXT)
        repo.commit("docs: edit reverted")
        for other in ("doge_2024", "majors_2023"):
            sha = hashlib.sha256(gitrepo.show_bytes(repo.path, edited, P.oos_manifest_path(other).as_posix())).hexdigest()
            main, _ = Stub()(other, {}, root=repo.path, mde_bps={}, code_commit=edited, manifest_sha256=sha)
            repo.write(P.oos_results_path(other).as_posix(), main.write_csv())
        repo.commit("secondary results")
        add_set_inputs(repo, "doge_2023")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages, "whose Appendix A differs")

    def test_doge_fwd_before_its_window_closes(self, tmp_path, messages):
        repo = build(tmp_path, "doge_fwd")
        code, stub = run_oos(repo, "doge_fwd", now_ms=FWD_END - 1)
        assert_refused(repo, "doge_fwd", code, stub, messages, "after it closes")
        # Once closed, it gets past that guard and is stopped by the next one
        # it fails: its warm-up is in-sample data, whose manifest this repo lacks.
        code, stub = run_oos(repo, "doge_fwd", now_ms=FWD_END)
        assert code == ro.EXIT_REFUSED and stub.calls == []
        assert "manifest_is_supp.json is not committed" in joined(messages)

    @pytest.mark.parametrize("name", ["IS", "doge_2025"])
    def test_unknown_or_insample_set(self, tmp_path, messages, name):
        repo = build(tmp_path)
        code, stub = run_oos(repo, name)
        assert code == ro.EXIT_USAGE and stub.calls == []


# ---------------------------------------------------------------- what git is asked about

class TestHistoryIsWhole:
    def test_replace_ref_is_refused(self, tmp_path, messages):
        # A local `git replace` can swap the tagged commit's tree for one with
        # other directions while rev-parse still prints the recorded hash.
        repo = build(tmp_path, manifest=False)
        key = ("1H", 5, "ou_reversion")
        repo.git("checkout", "-q", "-b", "alt", repo.c1)
        repo.write("results/insample_v2.csv", insample_text(flip=key))
        alt = repo.commit("alt directions")
        repo.git("checkout", "-q", "main")
        add_set_inputs(repo, "doge_2024")
        repo.git("replace", repo.c2, alt)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "refs/replace")

    def test_git_reads_ignore_replace_refs(self, tmp_path):
        repo = build(tmp_path, manifest=False)
        repo.git("checkout", "-q", "-b", "alt", repo.c1)
        repo.write("results/insample_v2.csv", "swapped\n")
        alt = repo.commit("alt")
        repo.git("checkout", "-q", "main")
        repo.git("replace", repo.c2, alt)
        assert repo.git("show", f"{repo.c2}:results/insample_v2.csv") == "swapped"     # plain git is fooled
        assert gitrepo.show_file(repo.path, repo.c2, "results/insample_v2.csv") == insample_text()
        assert gitrepo.replace_refs(repo.path) == [f"refs/replace/{repo.c2}"]

    def test_shallow_clone_is_refused(self, tmp_path, messages):
        repo = build(tmp_path)
        clone = tmp_path / "clone"
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--no-local", f"file://{repo.path}", str(clone)],
                       check=True, capture_output=True)
        assert gitrepo.is_shallow(clone)
        stub = Stub()
        code = ro.main(["--set", "doge_2024"], root=clone, compute=stub, now_ms=FWD_END + 1)
        assert_refused(None, "doge_2024", code, stub, messages, "shallow clone")

    def test_grafts_are_refused(self, tmp_path, messages):
        repo = build(tmp_path)
        grafts = repo.path / ".git" / "info" / "grafts"
        grafts.parent.mkdir(parents=True, exist_ok=True)
        grafts.write_text(f"{repo.c2}\n")                  # would make the tagged commit a root
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "info/grafts")


# ---------------------------------------------------------------- what runs is the committed code

def hide_change(repo, how):
    """Change what would run while ``git status`` stays clean, one bypass per ``how``."""
    code = "research/code.py"
    if how in ("skip_worktree", "assume_unchanged"):
        repo.git("update-index", f"--{how.replace('_', '-')}", code)
        repo.write(code, "X = 2  # git was told not to look\n")
    elif how == "clean_filter":
        # A same-size edit that a clean filter, set in the untracked
        # .git/info/attributes and the local config, maps back to the commit.
        repo.write(".git/info/attributes", f"{code} filter=tidy\n")
        repo.git("config", "filter.tidy.clean", "sed 's/^X = 2$/X = 1/'")
        repo.git("config", "filter.tidy.smudge", "cat")
        repo.write(code, "X = 2\n")
    else:
        # A sourceless module, ignored by *.py[cod], that shadows PyYAML for the script.
        repo.write(HIDDEN[how], b"not really bytecode")
    assert repo.git("status", "--porcelain") == ""


# how -> what the refusal names
HIDDEN = {
    "skip_worktree": "research/code.py (skip-worktree)",
    "assume_unchanged": "research/code.py (assume-unchanged)",
    "clean_filter": "research/code.py (differs from HEAD)",
    "root_pyc": "yaml.pyc",
    "scripts_pyc": "scripts/yaml.pyc",
}


class TestWhatRunsIsTheCommittedCode:
    @pytest.mark.parametrize("how", list(HIDDEN))
    def test_a_change_git_status_does_not_show(self, tmp_path, messages, how):
        repo = build(tmp_path)
        hide_change(repo, how)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, HIDDEN[how])

    def test_a_hidden_edit_outside_research_is_refused(self, tmp_path, messages):
        # Every tracked file is hashed, not only research/: the same-size
        # clean-filter edit as hide_change's, on the pre-registration's prose.
        repo = build(tmp_path)
        text = PREREG_TEXT.replace("**Summary.**", "**Sumxary.**", 1)
        assert text != PREREG_TEXT
        repo.write(".git/info/attributes", f"{PREREG_REL} filter=tidy\n")
        repo.git("config", "filter.tidy.clean", "sed 's/[*][*]Sumxary[.][*][*]/**Summary.**/'")
        repo.git("config", "filter.tidy.smudge", "cat")
        repo.write(PREREG_REL, text)
        assert repo.git("status", "--porcelain") == ""
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, f"{PREREG_REL} (differs from HEAD)")

    def test_a_file_system_monitor_cannot_hide_an_untracked_module(self, tmp_path, messages):
        # A local core.fsmonitor hook that always answers "nothing changed",
        # with the untracked cache on: plain git status no longer lists a new,
        # untracked (not ignored) yaml.py at the root, which comes first on the
        # script's sys.path and so would shadow PyYAML.
        repo = build(tmp_path)
        hook = repo.write(".git/quiet-fsmonitor", "#!/bin/sh\nprintf 'token\\0'\n")
        hook.chmod(0o755)
        repo.git("config", "core.untrackedCache", "true")
        repo.git("config", "core.fsmonitor", str(hook))
        repo.git("status", "--porcelain")                       # twice, so the hook's token is in use
        repo.git("status", "--porcelain")
        repo.write("yaml.py", "def safe_load(text):\n    return {}\n")
        assert repo.git("status", "--porcelain") == ""          # plain git is fooled
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "?? yaml.py")

    def test_unreachable_and_cached_ignored_files_are_fine(self, tmp_path, messages):
        # No import statement reaches docs/dev-notes/*.py (the hyphen), and the
        # CLI never reads __pycache__; the dev checkout has both.
        repo = build(tmp_path)
        repo.write(".git/info/exclude", "docs/dev-notes/\n")
        repo.write("docs/dev-notes/debug_candles.py", "print('a note')\n")
        repo.write("research/__pycache__/code.cpython-310.pyc", b"stale bytecode")
        assert repo.git("status", "--porcelain") == ""
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)

    @pytest.mark.parametrize("script", ["run_oos", "run_insample_v2"])
    def test_a_module_loaded_from_elsewhere(self, tmp_path, messages, monkeypatch, script):
        stray = types.ModuleType("research.elsewhere")
        stray.__file__ = str(tmp_path / "other-checkout" / "research" / "elsewhere.py")
        monkeypatch.setitem(sys.modules, "research.elsewhere", stray)
        if script == "run_oos":
            code, stub = run_oos(build(tmp_path))
            assert_refused(None, "doge_2024", code, stub, messages, "loaded from elsewhere")
        else:
            repo, spec = insample_root(tmp_path / "repo", repo=True)
            code, calls = TestInsampleRerun().run(repo.path, spec)
            assert code == ri.EXIT_REFUSED and calls == []
            assert "loaded from elsewhere" in joined(messages)

    def test_a_module_path_is_judged_after_resolving_it(self, monkeypatch):
        # <checkout>/../other-checkout/research/x.py is not inside the checkout.
        stray = types.ModuleType("research.climbed")
        stray.__file__ = str(REPO_ROOT / ".." / "other-checkout" / "research" / "climbed.py")
        monkeypatch.setitem(sys.modules, "research.climbed", stray)
        assert [m.split(" ")[0] for m in integrity.foreign_modules(REPO_ROOT)] == ["research.climbed"]

    def test_a_namespace_package_counts_only_if_all_of_it_is_inside(self, monkeypatch, tmp_path):
        # config/ has no __init__.py, so it is a namespace package, whose
        # __path__ may span other directories too (or none at all).
        inside = types.ModuleType("config")
        inside.__path__ = [str(REPO_ROOT / "config")]
        spread = types.ModuleType("config.sub")
        spread.__path__ = [str(REPO_ROOT / "config" / "sub"), str(tmp_path / "other-checkout" / "config" / "sub")]
        empty = types.ModuleType("config.empty")
        empty.__path__ = []
        for m in (inside, spread, empty):
            monkeypatch.setitem(sys.modules, m.__name__, m)
        assert [m.split(" ")[0] for m in integrity.foreign_modules(REPO_ROOT)] == ["config.empty", "config.sub"]

    @pytest.mark.parametrize("script", ["run_oos.py", "run_insample_v2.py", "fetch_v2.py"])
    def test_only_the_checkouts_own_source_is_imported(self, tmp_path, script):
        # Two ways other code could run under a clean git status, at once:
        # PYTHONPATH=<another checkout>:<this one> (the form the project's .env
        # uses), and a __pycache__ .pyc forged with the source's mtime and size.
        # Each writes a marker when imported. A copy of the checkout keeps the
        # real __pycache__ untouched; --help exits right after the imports.
        # fetch_v2 shares the scripts' startup block, so it is checked here too,
        # and so is the block's promise to remove its fresh cache at exit.
        checkout, shadow = tmp_path / "checkout", tmp_path / "shadow"
        shutil.copytree(REPO_ROOT / "research", checkout / "research", ignore=shutil.ignore_patterns("__pycache__"))
        for rel in (f"scripts/{script}", PREREG_REL):
            (checkout / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO_ROOT / rel, checkout / rel)
        (shadow / "research").mkdir(parents=True)
        (shadow / "research" / "__init__.py").write_text(f"open({str(tmp_path / 'shadow-ran')!r}, 'w').close()\n")
        src = checkout / "research" / "gitrepo.py"
        forged = compile(src.read_text() + f"\nopen({str(tmp_path / 'pyc-ran')!r}, 'w').close()\n", str(src), "exec")
        st = src.stat()
        cache = Path(importlib.util.cache_from_source(str(src)))
        cache.parent.mkdir()
        cache.write_bytes(importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, int(st.st_mtime), st.st_size)
                          + marshal.dumps(forged))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(shadow), str(checkout)]), TMPDIR=str(tmp_path))
        proc = subprocess.run([sys.executable, str(checkout / "scripts" / script), "--help"],
                              capture_output=True, text=True, env=env, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert not (tmp_path / "shadow-ran").exists()
        assert not (tmp_path / "pyc-ran").exists()
        assert list(tmp_path.glob("*-pycache-*")) == []          # TMPDIR is tmp_path


# ---------------------------------------------------------------- the set's manifest and data

class TestManifest:
    def test_manifest_rehashed_after_the_data_changed(self, tmp_path, messages):
        # Drop an inconvenient day, re-hash, commit: the inputs then match the
        # current manifest, so only the manifest's history shows it.
        repo = build(tmp_path)
        lo, hi = ro.manifest_window_required("doge_2024")
        path = repo.write("data/history/oos/doge_2024/DOGE-USDT-SWAP/candles_1m.parquet",
                          parquet(timestamp=[lo, lo + 2 * DAY_MS, hi - MINUTE_MS]))
        mp = repo.path / P.oos_manifest_path("doge_2024")
        m = json.loads(mp.read_text())
        m["series"][0]["sha256"] = sha256_file(path)
        mp.write_text(json.dumps(m, indent=2))
        repo.commit("manifest doge_2024: re-hash")
        repo.push()
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "manifest_doge_2024.json was committed 2 times")
        assert "a new path, with a §12 entry" in joined(messages)

    def test_manifest_edited_on_another_branch_counts(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.git("checkout", "-q", "-b", "side")
        mp = repo.path / P.oos_manifest_path("doge_2024")
        mp.write_text(mp.read_text().replace('"manifest_version": 1', '"manifest_version": 1, "note": "tidied"', 1))
        repo.commit("manifest edited on a side branch")
        repo.git("checkout", "-q", "main")
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "manifest_doge_2024.json was committed 2 times")

    @pytest.mark.parametrize("which", ["ends_a_month_early", "ends_a_day_early", "starts_a_day_late",
                                       "unknown_kind"])
    def test_data_does_not_cover_the_window(self, tmp_path, messages, which):
        # The manifest's window records what was asked for; the files show what
        # arrived. A full day of a perpetual's bars missing at an edge is a
        # truncated download, not an outage (the window's last bar starts at hi - 1 min).
        repo = build(tmp_path, manifest=False)
        lo, hi = ro.manifest_window_required("doge_2024")
        kw = {"ends_a_month_early": dict(data=(lo, hi - 30 * DAY_MS)),
              "ends_a_day_early": dict(data=(lo, hi - MINUTE_MS - DAY_MS)),
              "starts_a_day_late": dict(data=(lo + DAY_MS, hi - MINUTE_MS)),
              "unknown_kind": dict(edit=lambda m: m["series"][0].pop("kind"))}[which]
        add_set_inputs(repo, "doge_2024", **kw)
        code, stub = run_oos(repo)
        needle = "is not one of" if which == "unknown_kind" else "does not cover the set's window"
        assert_refused(repo, "doge_2024", code, stub, messages, needle)

    @pytest.mark.parametrize("funding", ["ends_two_days_early", "ends_a_month_early", "starts_a_week_late"])
    def test_a_short_optional_series_is_logged_not_refused(self, tmp_path, messages, funding):
        # §4.2 / §6.8: funding (or a spot leg) the exchange no longer has leaves
        # its cells undefined there or untestable; it does not refuse the set.
        repo = build(tmp_path, manifest=False)
        lo, hi = ro.manifest_window_required("doge_2024")
        span = {"ends_two_days_early": (lo + 4 * P.HOUR_MS, hi - 2 * DAY_MS - 4 * P.HOUR_MS),
                "ends_a_month_early": (lo + 4 * P.HOUR_MS, hi - 30 * DAY_MS),
                "starts_a_week_late": (lo + 7 * DAY_MS, hi - 4 * P.HOUR_MS)}[funding]
        add_set_inputs(repo, "doge_2024", funding=span)
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)
        assert "an optional input" in joined(messages)

    @pytest.mark.parametrize("missing", [5, 59, 6 * 60, 24 * 60 - 1])
    def test_bars_missing_at_either_edge_for_less_than_a_day_are_accepted(self, tmp_path, messages, missing):
        # Missing bars are gaps, which §4.2 handles (the incomplete bucket is
        # dropped), even hours of them where OKX had an outage on a window's
        # edge; refusing them in code frozen at the tag could not be undone.
        repo = build(tmp_path, manifest=False)
        lo, hi = ro.manifest_window_required("doge_2024")
        add_set_inputs(repo, "doge_2024", data=(lo + missing * MINUTE_MS, hi - MINUTE_MS - missing * MINUTE_MS))
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)

    def test_funding_coverage_is_judged_at_its_own_spacing(self, tmp_path, messages):
        # Settlements come every 8 h (Appendix A) and the windows' edges fall 4 h
        # from one, so one settlement missing at each edge is still a gap.
        repo = build(tmp_path, manifest=False)
        lo, hi = ro.manifest_window_required("doge_2024")
        eight = 8 * P.HOUR_MS
        add_set_inputs(repo, "doge_2024", funding=(lo + 4 * P.HOUR_MS + eight, hi - 4 * P.HOUR_MS - eight),
                       edit=lambda m: m.update(remote_url="file://" + freeze.PUBLIC_REMOTE))   # another spelling
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)
        assert "an optional input" not in joined(messages)

    @pytest.mark.parametrize("case", ["no_remote_url", "fork", "no_main_commit", "main_before_amendment",
                                      "unknown_commit"])
    def test_manifest_does_not_record_the_public_main(self, tmp_path, messages, case):
        # fetch_v2 asks the public repository for its main at fetch time; it must hold the amendment.
        repo = build(tmp_path, manifest=False)
        edit = {
            "no_remote_url": lambda m: m.pop("remote_url"),
            "fork": lambda m: m.update(remote_url="https://github.com/someone/crypto-factor-validation.git"),
            "no_main_commit": lambda m: m.pop("remote_main_commit"),
            "main_before_amendment": lambda m: m.update(remote_main_commit=repo.c2),
            "unknown_commit": lambda m: m.update(remote_main_commit="1" * 40),
        }[case]
        add_set_inputs(repo, "doge_2024", edit=edit)
        code, stub = run_oos(repo)
        needle = "records remote_url" if case in ("no_remote_url", "fork") else "records remote_main_commit"
        assert_refused(repo, "doge_2024", code, stub, messages, needle)

    def test_the_amendments_prose_may_change_after_the_data_was_fetched(self, tmp_path, messages):
        # A Software Heritage id filled in (§10 step 3) or a typo fixed after the
        # download: the main branch fetch_v2 saw held the same freeze blocks.
        repo = build(tmp_path, manifest=False)
        at_fetch = repo.git("rev-parse", "refs/remotes/origin/main")
        path = repo.path / "docs/preregistration-freeze.md"
        path.write_text(path.read_text().replace("Prose.", "Prose, edited after the data was fetched."))
        repo.commit("edit the amendment's prose")
        repo.push()                                                       # verify_freeze is satisfied
        add_set_inputs(repo, "doge_2024", edit=lambda m: m.update(remote_main_commit=at_fetch))
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)

    def test_the_public_main_at_fetch_time_holds_the_freeze_blocks_in_force(self, tmp_path, messages):
        # A block rewritten after the download, even to a value verify_freeze
        # still accepts (an MDE re-typed within its tolerance), is not what was
        # public when the data was fetched.
        repo = build(tmp_path, manifest=False)
        at_fetch = repo.git("rev-parse", "refs/remotes/origin/main")
        path = repo.path / "docs/preregistration-freeze.md"
        text = path.read_text()
        assert ",3.986\n" in text
        path.write_text(text.replace(",3.986\n", ",3.9860\n", 1))
        repo.commit("re-type an MDE in the amendment")
        repo.push()
        add_set_inputs(repo, "doge_2024", edit=lambda m: m.update(remote_main_commit=at_fetch))
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "does not hold the freeze blocks in force")

    def test_manifest_committed_but_not_pushed(self, tmp_path, messages):
        repo = build(tmp_path, manifest=False)
        add_set_inputs(repo, "doge_2024", push=False)
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "commits and pushes the manifest")

    def test_a_pr_merged_after_the_manifest_is_not_a_second_commit(self, tmp_path, messages):
        # A branch forked before the manifest's commit and merged after it: the
        # merge keeps main's manifest, so it is no edit (plain --full-history
        # would count it).
        repo = build(tmp_path, manifest=False)
        repo.git("branch", "pr")
        add_set_inputs(repo, "doge_2024", push=False)
        repo.git("checkout", "-q", "pr")
        repo.write("docs/notes.md", "PR work\n")
        repo.commit("PR work")
        repo.git("checkout", "-q", "main")
        repo.git("merge", "-q", "--no-ff", "-m", "merge the PR", "pr")
        repo.push()
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)

    def test_an_evil_merge_that_edits_the_manifest_is_a_second_commit(self, tmp_path, messages):
        repo = build(tmp_path, manifest=False)
        repo.git("branch", "pr")
        add_set_inputs(repo, "doge_2024", push=False)
        repo.git("checkout", "-q", "pr")
        repo.write("docs/notes.md", "PR work\n")
        repo.commit("PR work")
        repo.git("checkout", "-q", "main")
        repo.git("merge", "-q", "--no-ff", "--no-commit", "pr")
        mp = repo.path / P.oos_manifest_path("doge_2024")
        mp.write_text(mp.read_text().replace('"manifest_version": 1', '"manifest_version": 1, "note": "tidied"', 1))
        repo.commit("merge the PR")
        repo.push()
        code, stub = run_oos(repo)
        assert_refused(repo, "doge_2024", code, stub, messages, "manifest_doge_2024.json was committed 2 times")

    def test_a_manifest_commit_amended_before_the_push_is_committed_once(self, tmp_path, messages):
        # The reflog keeps the commit as it was before the amend; "committed
        # once" asks the refs, not the reflog, so this is not a second commit.
        repo = build(tmp_path, manifest=False)
        add_set_inputs(repo, "doge_2024", push=False)
        repo.git("commit", "-q", "--amend", "-m", "manifest doge_2024, as fetched")
        repo.push()
        code, _ = run_oos(repo)
        assert code == ro.EXIT_OK, joined(messages)


# ---------------------------------------------------------------- the secondary sets, recomputed

class TestSecondariesRecomputed:
    def test_forged_secondary_with_copied_provenance(self, tmp_path, messages):
        # Never produced by run_oos: every cell 'rejected', code_commit an
        # existing post-tag commit, manifest_sha256 copied from git, committed
        # once. Every provenance check passes; the recomputation does not.
        repo = build(tmp_path, "doge_2023", manifest=False)
        add_set_inputs(repo, "doge_2024")
        add_set_inputs(repo, "majors_2023")
        code_commit = repo.git("rev-parse", "HEAD")
        for other, inst in (("doge_2024", "DOGE-USDT-SWAP"), ("majors_2023", P.POOLED)):
            sha = hashlib.sha256(gitrepo.show_bytes(repo.path, code_commit,
                                                    P.oos_manifest_path(other).as_posix())).hexdigest()
            rows = [{**{c: "" for c in P.OOS_COLUMNS}, "instrument": inst, "bar": b, "horizon": str(h),
                     "factor": f, "status": "rejected", "code_commit": code_commit, "manifest_sha256": sha}
                    for b, h, f in P.StudyParams.from_spec().cells()]
            repo.write(P.oos_results_path(other).as_posix(),
                       pl.DataFrame(rows).select(list(P.OOS_COLUMNS)).write_csv())
        repo.commit("secondary results")
        add_set_inputs(repo, "doge_2023")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages, "doge_2024.csv is not what the frozen code computes")
        assert [c["set"] for c in stub.calls] == ["doge_2024"]           # recomputed, then refused
        assert outputs(repo, "doge_2023") == []

    def test_secondary_inputs_changed_since_its_run(self, tmp_path, messages):
        repo = build(tmp_path, "doge_2023", prerequisites=True)
        inst = SPEC["sets"]["majors_2023"]["inst"][0]
        (repo.path / f"data/history/oos/majors_2023/{inst}/candles_1m.parquet").write_bytes(b"re-downloaded")
        code, stub = run_oos(repo, "doge_2023")
        assert_refused(repo, "doge_2023", code, stub, messages, "inputs differ from the committed manifest")
        assert stub.calls == []

    def test_secondaries_are_recomputed_with_the_frozen_directions_and_mdes(self, tmp_path, messages):
        # The tag's directions and the amendment's MDEs, for each recomputation and for the run itself.
        repo = build(tmp_path, "doge_2023", prerequisites=True)
        code, stub = run_oos(repo, "doge_2023")
        assert code == ro.EXIT_OK, joined(messages)
        record = freeze.verify_freeze(repo.path)
        directions = P.directions_from_insample(
            freeze.parse_insample_csv(gitrepo.show_file(repo.path, record.commit, "results/insample_v2.csv")))
        mde = {k: c.mde_bps for k, c in record.cells.items()}
        assert any(math.isnan(v) for v in mde.values())                   # an undefined MDE is passed on too
        assert [c["set"] for c in stub.calls] == ["doge_2024", "majors_2023", "doge_2023"]
        for c in stub.calls:
            assert c["directions"] == directions, c["set"]
            assert c["mde"].keys() == mde.keys(), c["set"]
            assert all(c["mde"][k] == v or (math.isnan(c["mde"][k]) and math.isnan(v)) for k, v in mde.items())

    def test_what_counts_as_the_same_table(self):
        recomputed = pl.DataFrame({"status": ["rejected", "not_rejected"], "net_bps": [1.25, float("nan")],
                                   "n_eval": [10, 11]})
        committed = "status,net_bps,n_eval\nrejected,1.2500000000001,10\nnot_rejected,,11\n"
        assert ro.results_differences(committed, recomputed) == []        # float noise, NaN as blank
        assert ro.results_differences(committed.replace("1.2500000000001", "1.2500000005"), recomputed) == []
        for changed in (committed.replace("not_rejected", "rejected"),       # text: exact
                        committed.replace("1.2500000000001", "1.250000005"),  # rel 4e-9: beyond 1e-9
                        committed.replace(",,11", ",0.0,11"),               # a blank is only a blank
                        committed.replace(",10\n", ",10.0\n"),              # integers are compared as text
                        "status,net_bps,n_eval\nnot_rejected,,11\nrejected,1.25,10\n",     # row order
                        "status,net_bps,n_eval\nrejected,1.25,10\n",                       # a row missing
                        "status,net_bps\nrejected,1.25\nnot_rejected,\n"):                 # a column missing
            assert ro.results_differences(changed, recomputed), changed


# ---------------------------------------------------------------- the in-sample rerun

def insample_frame():
    df = pl.read_csv(io.StringIO(insample_text()), infer_schema_length=0)
    return df.with_columns(pl.col("flagged") == "true")


def insample_root(path: Path, repo: bool):
    """Synthetic in-sample inputs, the supplement's manifest, and a spec whose hashes match them."""
    r = Repo(path) if repo else None
    root = path
    (root).mkdir(parents=True, exist_ok=True)

    def put(rel, content):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    put(".gitignore", GITIGNORE.encode())
    candles = put(f"{SPEC['insample_dir']}/candles_1m.parquet", b"in-sample candles")
    supp = [put("data/history/is_supp/DOGE-USDT/candles_1m.parquet", b"spot"),
            put("data/history/is_supp/DOGE-USDT-SWAP/funding_rates.parquet", b"funding")]
    manifest = {"series": [{"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)} for p in supp]}
    put(P.IS_SUPP_MANIFEST.as_posix(), json.dumps(manifest).encode())
    put("research/code.py", b"X = 1\n")
    if r is not None:
        r.commit("inputs")
    spec = copy.deepcopy(SPEC)
    spec["insample_sha256"]["candles_1m.parquet"] = sha256_file(candles)
    return r, spec


class TestInsampleRerun:
    def run(self, root, spec, *flags, **seams):
        calls = []

        def compute(r):
            calls.append(r)
            return insample_frame()

        return ri.main(list(flags), root=root, spec=spec, compute=compute, **seams), calls

    def test_writes_once_from_a_clean_checkout(self, tmp_path, messages):
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_OK, joined(messages)
        out = repo.path / P.INSAMPLE_RESULTS
        assert out.read_text().splitlines()[0] == ",".join(freeze.INSAMPLE_COLUMNS)
        assert "flagged: 4" in joined(messages)
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_EXISTS and calls == []

    def test_refuses_uncommitted_change_to_a_tracked_file(self, tmp_path, messages):
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        repo.write("research/code.py", "X = 2\n")
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_REFUSED and calls == []
        assert "the working tree is not clean" in joined(messages)
        assert not (repo.path / P.INSAMPLE_RESULTS).exists()

    @pytest.mark.parametrize("rel", ["research/new_module.py", "scratch.txt"])
    def test_untracked_files_count_as_changes(self, tmp_path, messages, rel):
        # Code in no commit (an untracked research/ module) would decide the
        # directions unseen; any untracked file stops the rerun.
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        repo.write(rel, "Y = 1\n")
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_REFUSED and calls == []
        assert rel in joined(messages)
        assert not (repo.path / P.INSAMPLE_RESULTS).exists()

    def test_refuses_an_untracked_supplement_manifest(self, tmp_path, messages):
        # Ignored, so the clean-tree check cannot see it; §10 step 2 commits it first.
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        repo.git("rm", "-q", "--cached", P.IS_SUPP_MANIFEST.as_posix())
        repo.write(".gitignore", f"data/history/\n{P.IS_SUPP_MANIFEST.as_posix()}\n")
        repo.commit("forget the manifest")
        assert repo.git("status", "--porcelain") == ""
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_REFUSED and calls == []
        assert "manifest_is_supp.json is not committed" in joined(messages)

    def test_refuses_a_rerun_after_the_output_was_committed_and_removed(self, tmp_path, messages):
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        code, _ = self.run(repo.path, spec)
        assert code == ri.EXIT_OK, joined(messages)
        repo.commit("in-sample directions")
        repo.git("rm", "-q", P.INSAMPLE_RESULTS.as_posix())
        repo.commit("remove them")
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_EXISTS and calls == []
        assert "was committed before" in joined(messages)

    def test_refuses_a_rerun_after_the_output_was_committed_on_another_branch(self, tmp_path, messages):
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        repo.git("branch", "side")
        code, _ = self.run(repo.path, spec)
        assert code == ri.EXIT_OK, joined(messages)
        move_untracked_to(repo, "side", "in-sample directions, on a side branch")
        assert not (repo.path / P.INSAMPLE_RESULTS).exists()
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_EXISTS and calls == [], joined(messages)

    @pytest.mark.parametrize("how", ["merged_branch", "reset"])
    def test_refuses_a_rerun_after_the_output_was_committed_out_of_sight(self, tmp_path, messages, how):
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        code, _ = self.run(repo.path, spec)
        assert code == ri.EXIT_OK, joined(messages)
        commit_out_of_sight(repo, [P.INSAMPLE_RESULTS.as_posix()], how)
        assert not (repo.path / P.INSAMPLE_RESULTS).exists()
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_EXISTS and calls == [], joined(messages)
        assert "was committed before" in joined(messages)

    @pytest.mark.parametrize("how", list(HIDDEN))
    def test_refuses_a_change_git_status_does_not_show(self, tmp_path, messages, how):
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        hide_change(repo, how)
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_REFUSED and calls == []
        assert HIDDEN[how] in joined(messages)
        assert not (repo.path / P.INSAMPLE_RESULTS).exists()

    def test_refuses_outside_a_repository_unless_allow_dirty(self, tmp_path, messages):
        _, spec = insample_root(tmp_path / "plain", repo=False)
        code, calls = self.run(tmp_path / "plain", spec)
        assert code == ri.EXIT_REFUSED and calls == []
        code, calls = self.run(tmp_path / "plain", spec, allow_dirty=True)
        assert code == ri.EXIT_OK and len(calls) == 1

    def test_allow_dirty_is_not_a_command_line_flag(self, tmp_path):
        _, spec = insample_root(tmp_path / "plain", repo=False)
        with pytest.raises(SystemExit) as exc:
            self.run(tmp_path / "plain", spec, "--allow-dirty")
        assert exc.value.code == 2

    def test_refuses_modified_insample_candles(self, tmp_path, messages):
        repo, _ = insample_root(tmp_path / "repo", repo=True)
        code, calls = self.run(repo.path, SPEC)                     # the real Appendix A hashes
        assert code == ri.EXIT_REFUSED and calls == []
        assert "Appendix A" in joined(messages)

    def test_refuses_supplement_that_differs_from_its_manifest(self, tmp_path, messages):
        repo, spec = insample_root(tmp_path / "repo", repo=True)
        (repo.path / "data/history/is_supp/DOGE-USDT/candles_1m.parquet").write_bytes(b"changed")
        code, calls = self.run(repo.path, spec)
        assert code == ri.EXIT_REFUSED and calls == []
        assert "manifest_is_supp.json" in joined(messages)
