"""
scripts/verify_results.py, the reader's recomputation of the committed results.

CI has neither data/ nor this repository's history (its clone is shallow,
without tags), so almost everything here does without both: how a committed
file is compared with a recomputation, column by column, and which
differences each verdict stands for; how verdicts combine and map to exit
statuses; the command line; the order of ``--set all``; that a set without
its data stops at its inputs, before any computation; that a clone which
cannot show the history (an archive, no tag, a shallow clone, replace refs,
grafts) is NOT CHECKED, never FAIL; what the inputs line says with and
without the history, doge_fwd's included (it reads two manifests and the
in-sample candles, but only its own manifest came after the freeze
amendment); which remotes count as the public repository, and that guard 8
asks their main branches about the push; which loaded modules the code check
hashes; and that the script finds its checkout from scripts/.

What needs the history is checked against throwaway repositories tagged
prereg-freeze: a file on disk against a commit, whatever git status is told;
results committed once, by the right commit, and on disk as committed, and
each way of failing that (committed twice, a robustness file from another
commit, a manifest that is not the one the run hashed); the directions and
MDEs a run used; the constants; and the code. The steps of one set run with
their checks stubbed: which failure stops it before the computation, and
that the code check comes after it. The committed results/ files stand in
for a reader's archive download. The one test that recomputes a committed
set is skipped without every file that set reads; in a clone none of whose
remotes is the public repository (a fork's), it accepts INCOMPLETE for the
push alone.

git runs with the user's global and system config disabled, and with a fixed
identity in the environment, so nothing outside tmp_path is read or changed.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import polars as pl
import pytest

from research import freeze, gitrepo, pipeline
from research.io import sha256_file, write_once_csv
from research.spec import REPO_ROOT, get_spec


def _load(name):
    """
    scripts/<name>.py as a module, registered as _<name> (its dataclasses look
    their module up in sys.modules). Importing verify_results puts this
    checkout first on sys.path and turns bytecode caching off for the whole
    process, as its command line needs; that is undone here, so the test
    files collected after this one import as usual.
    """
    path, prefix, no_write = list(sys.path), sys.pycache_prefix, sys.dont_write_bytecode
    try:
        found = importlib.util.spec_from_file_location(f"_{name}", REPO_ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(found)
        sys.modules[found.name] = mod
        found.loader.exec_module(mod)
    finally:
        sys.path[:] = path
        sys.pycache_prefix, sys.dont_write_bytecode = prefix, no_write
    return mod


vr = _load("verify_results")
PASS, FAIL, NOT_CHECKED, TOLERATED, INCOMPLETE = vr.PASS, vr.FAIL, vr.NOT_CHECKED, vr.TOLERATED, vr.INCOMPLETE
SPEC = get_spec()
TAG = SPEC["freeze"]["tag"]
SCRIPT = REPO_ROOT / "scripts" / "verify_results.py"
# The order --set all verifies the sets in: the primary set, doge_2023, after
# the two whose statuses its labels are read from. Appendix A lists it first.
ALL = ["insample_v2", "doge_2024", "majors_2023", "doge_2023", "doge_fwd"]
PREREG = "docs/preregistration.md"
IS_RESULTS = pipeline.INSAMPLE_RESULTS.as_posix()        # results/insample_v2.csv
IS_SUPP = pipeline.IS_SUPP_MANIFEST.as_posix()           # results/manifest_is_supp.json


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    # tmp_path is no checkout, whatever directory holds it.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(k, raising=False)
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")


# ---------------------------------------------------------------- helpers

def git(cwd, *args):
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def commit_files(root, files, msg="change"):
    """Write ``files`` (path -> text) under ``root`` and commit everything. Returns the commit."""
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", msg)
    return git(root, "rev-parse", "HEAD")


def repo(path, files=None, tag=True):
    """A throwaway repository with one commit of ``files`` (path -> text), tagged prereg-freeze if ``tag``."""
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    head = commit_files(path, files or {"README.md": "x\n"}, "start")
    if tag:
        git(path, "tag", "-a", TAG, "-m", "freeze")
    return head


def history(path):
    """Two commits, the second tagged prereg-freeze. Returns both."""
    first = repo(path, tag=False)
    (path / "README.md").write_text("y\n")
    git(path, "commit", "-q", "-am", "second")
    git(path, "tag", "-a", TAG, "-m", "freeze")
    return first, git(path, "rev-parse", "HEAD")


def clone(tmp_path, how):
    """A checkout that cannot show the history the provenance checks need, one way per ``how``."""
    root = tmp_path / "clone"
    if how == "archive":
        root.mkdir()
    elif how == "no tag":
        repo(root, tag=False)
    elif how == "shallow":
        history(tmp_path / "public")
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--no-local", f"file://{tmp_path / 'public'}",
                        str(root)], check=True, capture_output=True)
    elif how == "replace ref":
        first, second = history(root)
        git(root, "replace", first, second)
    elif how == "grafts":
        first, second = history(root)
        (root / ".git" / "info").mkdir(exist_ok=True)
        (root / ".git" / "info" / "grafts").write_text(second + "\n")
    return root


CLONES = [
    ("archive", "is not a git work tree (an archive download?)"),
    ("no tag", f"there is no tag {TAG} here (fetch the tags: git fetch --tags)"),
    ("shallow", "the repository is a shallow clone"),
    ("replace ref", "refs/replace"),
    ("grafts", "info/grafts"),
]


def archive(root):
    """What a reader has without git or data: the committed results/ and the freeze amendment."""
    shutil.copytree(REPO_ROOT / "results", root / "results")
    amendment = SPEC["freeze"]["amendment_file"]
    (root / amendment).parent.mkdir(parents=True)
    shutil.copy(REPO_ROOT / amendment, root / amendment)
    return root


def with_results(root, names):
    """Empty results files for ``names`` under ``root``, which is all --set all looks for."""
    for name in names:
        for rel in vr.results_paths(name):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("")
    return root


def with_inputs(root, name):
    """Every input file of ``name`` under ``root``, created empty: present, their bytes the stubs' business."""
    for f in pipeline.input_files(vr.pipeline_set(name), root):
        f.path.parent.mkdir(parents=True, exist_ok=True)
        f.path.write_bytes(b"")
    return root


def run_history(root, name, robustness="with the results"):
    """
    A throwaway history of one run of test set ``name``, as §10 steps 4-5
    leave it: prereg-freeze holding Appendix A and the other files the set
    reads, a commit adding the set's manifest (the HEAD the run computed
    at), and its child adding the results, whose one row names that commit
    and the manifest's SHA-256. The primary set's robustness file comes with
    the results, or ``earlier`` (with the manifest) or ``later`` (in a commit
    of its own). Returns (tag, code, added).
    """
    rels = [p.as_posix() for p in vr.results_paths(name)]
    manifest = pipeline.oos_manifest_path(name).as_posix()
    others = [p for p in (IS_SUPP, "results/oos/doge_2024.csv", "results/oos/majors_2023.csv") if p not in rels]
    tag = repo(root, files={PREREG: (REPO_ROOT / PREREG).read_text(encoding="utf-8"),
                            **{p: "x\n" for p in others}})
    earlier = {rels[1]: "x\n"} if robustness == "earlier" else {}
    code = commit_files(root, {manifest: '{"series": []}\n', **earlier})
    row = f"factor,code_commit,manifest_sha256\nrsi_14,{code},{sha256_file(root / manifest)}\n"
    added = commit_files(root, {r: row for r in rels if r == rels[0] or robustness == "with the results"})
    if robustness == "later":
        commit_files(root, {rels[1]: row})
    return tag, code, added


def stub_checks(monkeypatch, sets=None, common=PASS):
    """
    main() with its checks stubbed: git and constants answer ``common`` (None
    keeps the real ones), and each set answers the verdicts ``sets`` gives it,
    PASS by default. Returns the sets main() verified, in order.
    """
    seen = []
    if common is not None:
        tag_commit = None if common == NOT_CHECKED else "0" * 40
        monkeypatch.setattr(vr, "git_state", lambda root: (tag_commit, [vr.Check("git", common, "stub")]))
        monkeypatch.setattr(vr, "spec_check", lambda root, tag: vr.Check("constants", common, "stub"))

    def verify_set(root, name, tag_commit, git_ok):
        seen.append(name)
        verdicts = (sets or {}).get(name, [PASS])
        return vr.SetReport(name, "stub", [vr.Check(f"check {i}", v, "stub") for i, v in enumerate(verdicts)])

    monkeypatch.setattr(vr, "verify_set", verify_set)
    return seen


def stub_guard_8(monkeypatch, rel, added, tips, held):
    """
    inputs_check's git questions answered without a history: ``rel`` was
    first added by ``added``; the public main branches here are ``tips``
    (ref -> commit), and they hold ``added`` if ``held``. run_oos.check_manifest
    only records the set and the ref it would ask whether the push happened.
    Returns those records.
    """
    asked = []
    monkeypatch.setattr(vr.run_oos, "_first_added", lambda root, path: added if path == rel else None)
    monkeypatch.setattr(vr, "public_main_refs", lambda root: list(tips))
    monkeypatch.setattr(gitrepo, "commit_of", lambda root, ref: tips.get(ref))
    monkeypatch.setattr(gitrepo, "is_ancestor",
                        lambda root, a, d="HEAD": held and a == added and d in tips.values())
    monkeypatch.setattr(vr.run_oos, "check_manifest",
                        lambda root, set_name: asked.append((set_name, freeze.REMOTE_MAIN)))
    return asked


def module(monkeypatch, name, file):
    """A module that says it was loaded from ``file``, in sys.modules for one test."""
    m = types.ModuleType(name)
    m.__file__ = str(file)
    monkeypatch.setitem(sys.modules, name, m)
    return m


# ---------------------------------------------------------------- where the script runs

class TestCheckout:
    def test_from_scripts_it_finds_this_checkout(self):
        assert vr.REPO_ROOT == REPO_ROOT
        assert SCRIPT.resolve() in vr.THIS_FILE
        # As it would from a copy kept anywhere else inside the checkout.
        assert vr._find_repo_root(REPO_ROOT / "docs" / "tools" / "verify_results.py") == REPO_ROOT

    def test_the_nearest_ancestor_holding_the_study_is_the_checkout(self, tmp_path):
        outer, inner = tmp_path / "outer", tmp_path / "outer" / "vendor" / "copy"
        for root in (outer, inner):
            for rel in ("research/pipeline.py", "docs/preregistration.md"):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("")
        assert vr._find_repo_root(outer / "scripts" / "verify_results.py") == outer
        assert vr._find_repo_root(outer / "a" / "b" / "verify_results.py") == outer
        assert vr._find_repo_root(inner / "scripts" / "verify_results.py") == inner

    @pytest.mark.parametrize("present", [(), ("research/pipeline.py",), ("docs/preregistration.md",)])
    def test_outside_a_checkout_it_stops(self, tmp_path, present):
        for rel in present:
            (tmp_path / rel).parent.mkdir(parents=True)
            (tmp_path / rel).write_text("")
        with pytest.raises(SystemExit, match="is not inside a checkout of the study"):
            vr._find_repo_root(tmp_path / "scripts" / "verify_results.py")

    def test_the_command_runs_from_any_directory_and_writes_nothing(self, tmp_path):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        helped = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=tmp_path, env=env,
                                capture_output=True, text=True)
        assert helped.returncode == 0, helped.stderr
        assert all(name in helped.stdout for name in ALL)
        refused = subprocess.run([sys.executable, str(SCRIPT), "--set", "nope"], cwd=tmp_path, env=env,
                                 capture_output=True, text=True)
        assert refused.returncode == 2, refused.stderr
        assert "--set must be one of" in refused.stderr and refused.stdout == ""
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- verdicts and exit statuses

class TestVerdicts:
    @pytest.mark.parametrize("verdicts,want", [
        ([PASS, PASS], PASS),
        ([PASS, TOLERATED], TOLERATED),
        ([TOLERATED, NOT_CHECKED, PASS], INCOMPLETE),
        ([NOT_CHECKED, TOLERATED, FAIL, PASS], FAIL),
    ])
    def test_the_worst_verdict_decides(self, verdicts, want):
        assert vr.overall(verdicts) == want
        assert vr.overall(verdicts[::-1]) == want

    def test_a_set_takes_the_worst_of_its_checks_and_columns(self):
        r = vr.SetReport("doge_fwd", pipeline.ROLE_DESCRIPTIVE, [vr.Check("inputs", PASS, "")])
        r.tables.append(("results/oos/doge_fwd.csv", "", [vr.ColumnVerdict("ic_p", "number", TOLERATED, "")]))
        assert r.verdicts() == [PASS, TOLERATED] and r.verdict() == TOLERATED
        r.checks.append(vr.Check("code", NOT_CHECKED, ""))
        assert r.verdict() == INCOMPLETE

    @pytest.mark.parametrize("verdicts,code,word", [
        ([PASS], 0, PASS),
        ([PASS, TOLERATED], 4, TOLERATED),
        ([TOLERATED, NOT_CHECKED], 3, INCOMPLETE),
        ([NOT_CHECKED, FAIL], 1, FAIL),
    ])
    def test_the_exit_status_is_the_overall_verdicts(self, monkeypatch, capsys, tmp_path, verdicts, code, word):
        stub_checks(monkeypatch, {"doge_fwd": verdicts})
        assert vr.main(["--set", "doge_fwd"], root=tmp_path) == code
        out = capsys.readouterr().out.splitlines()
        assert f"  doge_fwd: {word} (0 s)" in out and out[-1] == f"overall: {word} (doge_fwd {word})"

    def test_a_common_check_counts_toward_the_overall_verdict(self, monkeypatch, capsys, tmp_path):
        stub_checks(monkeypatch, common=NOT_CHECKED)
        assert vr.main(["--set", "doge_2024"], root=tmp_path) == 3
        assert capsys.readouterr().out.splitlines()[-1] == "overall: INCOMPLETE (doge_2024 PASS)"

    def test_a_failed_common_check_stops_before_any_set(self, monkeypatch, capsys, tmp_path):
        seen = stub_checks(monkeypatch, common=FAIL)
        assert vr.main(["--set", "doge_fwd"], root=tmp_path) == 1
        assert seen == [] and capsys.readouterr().out.splitlines()[-1] == "overall: FAIL (before any set)"

    @pytest.mark.parametrize("exc", [vr.run_oos.Refusal("broken"), pipeline.PipelineError("broken"),
                                     gitrepo.GitError("broken"), OSError("broken")])
    def test_an_error_inside_a_set_is_that_sets_failure(self, monkeypatch, tmp_path, exc):
        def stop(report, *args):
            report.checks.append(vr.Check("inputs", PASS, "stub"))
            raise exc

        monkeypatch.setattr(vr, "_verify_into", stop)
        r = vr.verify_set(tmp_path, "doge_2024", None, False)
        assert [(c.name, c.verdict) for c in r.checks] == [("inputs", PASS), ("error", FAIL)]
        assert r.checks[-1].detail == f"{type(exc).__name__}: broken" and r.verdict() == FAIL

    def test_a_set_without_results_fails_before_its_inputs(self, tmp_path):
        r = vr.verify_set(tmp_path, "doge_fwd", None, False)
        assert [(c.name, c.verdict) for c in r.checks] == [("results", FAIL)]
        assert r.checks[0].detail == "results/oos/doge_fwd.csv does not exist; this set has no committed results yet"


# ---------------------------------------------------------------- the command line

class TestCommandLine:
    def test_the_sets_are_appendix_as_with_the_in_sample_year_named_for_its_file(self):
        assert vr.known_sets() == ["insample_v2", "doge_2023", "doge_2024", "majors_2023", "doge_fwd"]
        assert vr.pipeline_set("insample_v2") == "IS" and vr.pipeline_set("doge_fwd") == "doge_fwd"
        assert vr.results_paths("insample_v2") == (Path("results/insample_v2.csv"),)
        assert vr.results_paths("doge_2023") == (Path("results/oos/doge_2023.csv"),
                                                 Path("results/oos/robustness_doge_2023.csv"))
        assert vr.results_paths("doge_fwd") == (Path("results/oos/doge_fwd.csv"),)

    @pytest.mark.parametrize("name", ["nope", "IS", "doge_fwd.csv", "ALL", ""])
    def test_an_unknown_set_is_a_usage_error_before_any_check(self, monkeypatch, capsys, tmp_path, name):
        monkeypatch.setattr(vr, "git_state", lambda root: pytest.fail("a clone was checked for an unknown set"))
        assert vr.main(["--set", name], root=tmp_path) == 2
        out, err = capsys.readouterr()
        assert out == ""
        assert err == "--set must be one of insample_v2, doge_2023, doge_2024, majors_2023, doge_fwd or all\n"

    @pytest.mark.parametrize("argv", [[], ["--set"], ["doge_fwd"], ["--set", "doge_fwd", "--extra"]])
    def test_a_malformed_command_line_exits_with_2(self, capsys, argv):
        with pytest.raises(SystemExit) as exc:
            vr.main(argv)
        assert exc.value.code == 2


# ---------------------------------------------------------------- --set all

class TestSetOrder:
    def test_all_takes_the_primary_set_after_the_two_it_reads(self, tmp_path):
        assert vr.committed_sets(with_results(tmp_path, vr.known_sets())) == ALL

    @pytest.mark.parametrize("names", [[], ["doge_fwd"], ["doge_2023", "insample_v2"],
                                       ["majors_2023", "doge_fwd", "doge_2024"]])
    def test_all_takes_only_the_sets_with_results(self, tmp_path, names):
        assert vr.committed_sets(with_results(tmp_path, names)) == [n for n in ALL if n in names]

    def test_a_robustness_file_alone_is_no_results(self, tmp_path):
        rel = pipeline.robustness_results_path("doge_2023")
        (tmp_path / rel).parent.mkdir(parents=True)
        (tmp_path / rel).write_text("")
        assert vr.committed_sets(tmp_path) == []

    def test_main_verifies_them_in_that_order(self, monkeypatch, capsys, tmp_path):
        with_results(tmp_path, ["doge_fwd", "doge_2023", "majors_2023", "doge_2024"])
        seen = stub_checks(monkeypatch)
        assert vr.main(["--set", "all"], root=tmp_path) == 0
        assert seen == ["doge_2024", "majors_2023", "doge_2023", "doge_fwd"]
        assert capsys.readouterr().out.splitlines()[-1] == \
            "overall: PASS (doge_2024 PASS, majors_2023 PASS, doge_2023 PASS, doge_fwd PASS)"

    def test_this_repository_has_results_for_every_set(self):
        # results/ is committed, so --set all verifies all five in any clone, doge_fwd included.
        assert vr.committed_sets(REPO_ROOT) == ALL


# ---------------------------------------------------------------- comparison

class TestNumbers:
    @pytest.mark.parametrize("x,y,kind", [
        ("1.5", "1.5", vr.SAME),
        (None, None, vr.SAME),                                        # two blanks
        ("-10.306465364642463", "-10.30646536464247", vr.CLOSE),     # the last digits
        ("0.0", "-0.0", vr.CLOSE),
        ("3.2e-12", "9e-10", vr.FLOOR),           # equal only within guard 9's absolute 1e-9
        ("8.5e-14", "0.0", vr.FLOOR),
        ("1e-18", "-3e-19", vr.FLOOR),            # run_oos's own example of what the floor is for
        ("0.001", "0.0010000005", vr.FLOOR),      # a relative 5e-7: within 1e-6, not within 1e-9
        (None, "NaN", vr.NAN_TEXT),               # guard 9 reads NaN as a blank; the run never writes it
        ("nan", None, vr.NAN_TEXT),
        ("1.25", "1.250000005", vr.DIFFERENT),    # a relative 4e-9
        (None, "0", vr.DIFFERENT),                # a blank is only a blank
        ("0.5", None, vr.DIFFERENT),
        ("0.5", "half", vr.DIFFERENT),
    ])
    def test_how_two_number_fields_compare(self, x, y, kind):
        assert vr.number_pair(x, y) == kind
        assert vr.number_pair(y, x) == kind


KEYS = pipeline.KEY_COLUMNS


def recomputed():
    """Three rows of a test set's results, with each kind of column compare() tells apart."""
    return pl.DataFrame([
        pl.Series("instrument", ["DOGE-USDT-SWAP"] * 3, dtype=pl.Utf8),
        pl.Series("bar", ["15m", "15m", "1H"], dtype=pl.Utf8),
        pl.Series("horizon", [1, 5, 15], dtype=pl.Int64),
        pl.Series("factor", ["rsi_14", "rsi_14", "macd_hist"], dtype=pl.Utf8),
        pl.Series("direction_flagged", [False, True, False], dtype=pl.Boolean),
        pl.Series("n_active", [2390, 17, 0], dtype=pl.Int64),
        pl.Series("net_bps", [-10.306465364642463, 3.25, float("nan")], dtype=pl.Float64),
        pl.Series("ic_p", [3.2e-12, 0.5, None], dtype=pl.Float64),
        pl.Series("status", ["descriptive"] * 3, dtype=pl.Utf8),
        pl.Series("code_commit", ["a" * 40] * 3, dtype=pl.Utf8),
        pl.Series("manifest_sha256", ["b" * 64] * 3, dtype=pl.Utf8),
    ])


def robustness_rows():
    """Two rows in the shape of the primary set's robustness file."""
    return pl.DataFrame([
        pl.Series("bar", ["15m", "15m"], dtype=pl.Utf8),
        pl.Series("horizon", [1, 1], dtype=pl.Int64),
        pl.Series("factor", ["rsi_14", "rsi_14"], dtype=pl.Utf8),
        pl.Series("row", ["R3", "R3"], dtype=pl.Utf8),
        pl.Series("parameter", ["Q1", "Q2"], dtype=pl.Utf8),
        pl.Series("value_bps", [-11.0256974510836, -9.820877713919415], dtype=pl.Float64),
        pl.Series("n_active", [3761, 3715], dtype=pl.Int64),
    ])


def cell(text, line, col):
    """The field of ``col`` on ``line`` of a CSV text, the header being line 1 (as the tool counts)."""
    rows = [r.split(",") for r in text.splitlines()]
    return rows[line - 1][rows[0].index(col)]


def edit(text, line, col, value):
    """``text`` with that field replaced by ``value``."""
    rows = [r.split(",") for r in text.splitlines()]
    rows[line - 1][rows[0].index(col)] = value
    return "".join(",".join(r) + "\n" for r in rows)


def verdict_of(table, col):
    (v,) = [v for v in table if v.column == col]
    return v


class TestCompare:
    def test_the_runs_own_text_is_byte_identical(self):
        df = recomputed()
        summary, table = vr.compare(vr.csv_text(df), df, KEYS)
        assert summary == "3 rows x 11 columns, byte-identical to the recomputation"
        assert {v.column: v.kind for v in table} == {
            "instrument": "key", "bar": "key", "horizon": "key", "factor": "key",
            "direction_flagged": "flag", "n_active": "integer", "net_bps": "number", "ic_p": "number",
            "status": "text", "code_commit": "provenance", "manifest_sha256": "provenance",
        }
        assert {(v.verdict, v.detail) for v in table} == {(PASS, "identical")}

    @pytest.mark.parametrize("col,line,change,verdict,detail", [
        # Summation order, not a changed value.
        ("net_bps", 2, lambda x: repr(float(x) * (1 + 1e-12)), PASS,
         "1 row differs only in the last digits (max |Δ| 1.0e-11, max relative 1.0e-12)"),
        ("net_bps", 2, lambda x: repr(float(x) * (1 + 1e-6)), FAIL, "1 of 3 rows differ; first at line 2"),
        ("net_bps", 2, lambda x: "", FAIL, "1 of 3 rows differ; first at line 2"),
        ("net_bps", 4, lambda x: "0", FAIL, "1 of 3 rows differ; first at line 4: '0' committed, None recomputed"),
        ("net_bps", 4, lambda x: "NaN", TOLERATED,
         "1 row(s) hold NaN where the other file has a blank, which the run never writes; "
         "first at line 4: 'NaN' committed, None recomputed"),
        ("ic_p", 2, lambda x: "9e-10", TOLERATED,
         "1 row(s) agree only within guard 9's absolute 1e-9, not to a relative 1e-9 (max relative "
         "difference 1.0e+00); first at line 2: '9e-10' committed, '3.2e-12' recomputed"),
        ("ic_p", 2, lambda x: "0.0", TOLERATED, "first at line 2: '0.0' committed, '3.2e-12' recomputed"),
        ("ic_p", 2, lambda x: "-5e-10", TOLERATED, "first at line 2: '-5e-10' committed, '3.2e-12' recomputed"),
        ("n_active", 2, lambda x: str(int(x) + 1), FAIL, "first at line 2: '2391' committed, '2390' recomputed"),
        ("direction_flagged", 3, lambda x: "false", FAIL, "first at line 3: 'false' committed, 'true' recomputed"),
        ("status", 4, lambda x: "rejected", FAIL, "first at line 4: 'rejected' committed, 'descriptive' recomputed"),
        ("factor", 3, lambda x: "macd_hist", FAIL, "first at line 3: 'macd_hist' committed, 'rsi_14' recomputed"),
    ])
    def test_one_changed_field(self, col, line, change, verdict, detail):
        df = recomputed()
        text = vr.csv_text(df)
        summary, table = vr.compare(edit(text, line, col, change(cell(text, line, col))), df, KEYS)
        assert summary == "3 rows x 11 columns, not byte-identical; see the columns"
        v = verdict_of(table, col)
        assert v.verdict == verdict and detail in v.detail, v.detail
        assert {(o.verdict, o.detail) for o in table if o.column != col} == {(PASS, "identical")}
        assert vr.overall([o.verdict for o in table]) == verdict

    def test_a_relative_difference_above_1e_9_is_tolerated_however_small_in_absolute_terms(self):
        # Guard 9 accepted it through its absolute 1e-9; a relative tolerance of 1e-6 would pass it here.
        (v,) = vr.compare("ic_p\n0.001\n", pl.DataFrame({"ic_p": [0.0010000005]}), ())[1]
        assert (v.kind, v.verdict) == ("number", TOLERATED)
        assert v.detail == ("1 row(s) agree only within guard 9's absolute 1e-9, not to a relative 1e-9 (max "
                            "relative difference 5.0e-07); first at line 2: '0.001' committed, '0.0010000005' "
                            "recomputed")

    def test_a_refused_number_outranks_a_tolerated_one(self):
        df = recomputed()
        text = edit(edit(vr.csv_text(df), 2, "ic_p", "9e-10"), 3, "ic_p", "0.6")
        v = verdict_of(vr.compare(text, df, KEYS)[1], "ic_p")
        assert v.verdict == FAIL
        assert v.detail == "1 of 3 rows differ; first at line 3: '0.6' committed, '0.5' recomputed"

    def test_provenance_columns_take_the_provenance_checks_verdict(self):
        df = recomputed()
        notes = {"code_commit": (PASS, "checked against git"), "manifest_sha256": (NOT_CHECKED, "no history")}
        table = vr.compare(vr.csv_text(df), df, KEYS, notes)[1]
        assert (verdict_of(table, "code_commit").verdict, verdict_of(table, "code_commit").detail) == \
            (PASS, "checked against git")
        assert verdict_of(table, "manifest_sha256").verdict == NOT_CHECKED
        # The in-sample file is compared without such notes: equal text passes.
        assert verdict_of(vr.compare(vr.csv_text(df), df, KEYS)[1], "manifest_sha256").verdict == PASS

    def test_a_changed_provenance_value_fails_whatever_its_check_said(self):
        df = recomputed()
        text = edit(vr.csv_text(df), 3, "code_commit", "c" * 40)
        notes = {"code_commit": (PASS, "checked"), "manifest_sha256": (PASS, "checked")}
        v = verdict_of(vr.compare(text, df, KEYS, notes)[1], "code_commit")
        assert v.verdict == FAIL and v.detail.startswith("1 of 3 rows differ; first at line 3")

    def test_rows_in_another_order_fail(self):
        df = recomputed()
        lines = vr.csv_text(df).splitlines(True)
        table = vr.compare("".join([lines[0], lines[2], lines[1], lines[3]]), df, KEYS)[1]
        assert verdict_of(table, "horizon").detail == \
            "2 of 3 rows differ; first at line 2: '5' committed, '1' recomputed"
        assert verdict_of(table, "bar").verdict == PASS             # equal on both rows

    @pytest.mark.parametrize("change,detail", [
        (lambda t: "".join(t.splitlines(True)[:-1]), "committed 2 x 11, recomputed 3 x 11"),
        (lambda t: t + t.splitlines(True)[-1], "committed 4 x 11, recomputed 3 x 11"),
        (lambda t: t.replace("ic_p", "ic_q", 1), "committed 3 x 11, recomputed 3 x 11; the columns differ"),
    ])
    def test_another_shape_fails_as_a_whole(self, change, detail):
        df = recomputed()
        summary, table = vr.compare(change(vr.csv_text(df)), df, KEYS)
        assert [(v.column, v.verdict, v.detail) for v in table] == [("(shape)", FAIL, detail)]

    def test_the_recomputation_is_serialised_as_the_run_writes_it_without_writing(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        df = recomputed()
        text = vr.csv_text(df)
        assert list(tmp_path.iterdir()) == []
        assert write_once_csv(df, tmp_path / "run.csv").read_text(encoding="utf-8") == text
        assert cell(text, 4, "net_bps") == cell(text, 4, "ic_p") == ""        # NaN and null both blank (§10)


# ---------------------------------------------------------------- a clone without the history

class TestWithoutHistory:
    def test_a_whole_history_with_the_tag(self, tmp_path):
        head = repo(tmp_path / "r")
        commit, checks = vr.git_state(tmp_path / "r")
        assert commit == head and [(c.name, c.verdict) for c in checks] == [("git", PASS)]

    @pytest.mark.parametrize("how,needle", CLONES)
    def test_what_the_clone_lacks_is_not_checked_rather_than_failed(self, tmp_path, how, needle):
        root = clone(tmp_path, how)
        commit, checks = vr.git_state(root)
        assert commit is None and [(c.name, c.verdict) for c in checks] == [("git", NOT_CHECKED)]
        assert needle in checks[0].detail
        assert checks[0].detail.endswith("the code_commit column and the code cannot be checked against the tag")
        assert vr.spec_check(root, None).verdict == NOT_CHECKED

    @pytest.mark.parametrize("how", [c[0] for c in CLONES])
    def test_such_a_clone_is_incomplete_even_if_every_set_passes(self, monkeypatch, capsys, tmp_path, how):
        root = clone(tmp_path, how)
        stub_checks(monkeypatch, common=None)
        assert vr.main(["--set", "doge_fwd"], root=root) == 3
        out = capsys.readouterr().out
        assert "[NOT CHECKED] git: " in out and "[NOT CHECKED] constants: " in out
        assert out.splitlines()[-1] == "overall: INCOMPLETE (doge_fwd PASS)"

    @pytest.mark.parametrize("name", ALL)
    def test_without_the_data_a_set_fails_at_its_inputs_before_any_computation(self, monkeypatch, capsys,
                                                                                 tmp_path, name):
        root = archive(tmp_path / "archive")
        for fn in ("run_insample", "run_oos_set"):
            monkeypatch.setattr(pipeline, fn, lambda *a, **k: pytest.fail("computed without its inputs"))
        assert vr.main(["--set", name], root=root) == 1
        out = capsys.readouterr().out
        assert f"\n  [FAIL] inputs: " in out and "does not exist" in out
        assert out.splitlines()[-1] == f"overall: FAIL ({name} FAIL)"

    def test_all_on_an_archive_takes_every_set_in_order(self, monkeypatch, capsys, tmp_path):
        root = archive(tmp_path / "archive")
        for fn in ("run_insample", "run_oos_set"):
            monkeypatch.setattr(pipeline, fn, lambda *a, **k: pytest.fail("computed without its inputs"))
        assert vr.main(["--set", "all"], root=root) == 1
        out = capsys.readouterr().out.splitlines()
        assert [line.split()[1] for line in out if line.startswith("== ") and line != "== common"] == ALL

    def test_the_files_on_disk_are_compared_unvouched_for(self, tmp_path):
        root = archive(tmp_path / "archive")
        text, checks = vr.insample_provenance(root, None)
        assert text == (root / "results" / "insample_v2.csv").read_text(encoding="utf-8")
        assert [(c.name, c.verdict) for c in checks] == [("provenance", NOT_CHECKED)]
        for name in ("doge_2023", "doge_fwd"):
            prov = vr.oos_provenance(root, name, None)
            assert prov.texts == tuple((root / p).read_text(encoding="utf-8") for p in vr.results_paths(name))
            assert [(c.name, c.verdict) for c in prov.checks] == [("provenance", NOT_CHECKED)]
            assert prov.column_notes["code_commit"][0] == NOT_CHECKED
            # manifest_sha256 can still be checked: against the manifest on disk.
            assert prov.column_notes["manifest_sha256"][0] == PASS
            assert prov.manifest_sha256 == sha256_file(root / pipeline.oos_manifest_path(name))
            assert gitrepo.FULL_SHA1.fullmatch(prov.code_commit)

    def test_a_manifest_that_is_not_the_one_the_run_hashed_fails(self, tmp_path):
        root = archive(tmp_path / "archive")
        with open(root / pipeline.oos_manifest_path("doge_fwd"), "a", encoding="utf-8") as fh:
            fh.write("\n")
        assert vr.oos_provenance(root, "doge_fwd", None).column_notes["manifest_sha256"][0] == FAIL

    def test_the_directions_are_read_from_disk_and_must_agree_with_the_amendment(self, tmp_path):
        root = archive(tmp_path / "archive")
        directions, mde, check = vr.frozen_directions(root, None, None)
        assert check.verdict == PASS and len(directions) == len(mde) == 60
        assert check.detail.startswith("60 directions") and "on disk" in check.detail
        path = root / SPEC["freeze"]["directions_file"]
        lines = path.read_text(encoding="utf-8").splitlines(True)
        fields = lines[1].split(",")
        assert lines[0].split(",")[3] == "direction"
        fields[3] = str(-int(fields[3]))
        lines[1] = ",".join(fields)
        path.write_text("".join(lines), encoding="utf-8")
        check = vr.frozen_directions(root, None, None)[2]
        assert check.verdict == FAIL and "disagree" in check.detail


# ---------------------------------------------------------------- a clone with the history

class TestWithHistory:
    @pytest.mark.parametrize("by_name", [False, True])
    def test_a_file_on_disk_against_a_commit(self, tmp_path, by_name):
        root = tmp_path / "r"
        head = repo(root, files={"results/a.csv": "1\n"})
        # A commit is shown abbreviated, a ref name as it is.
        rev, shown = ("HEAD", "HEAD") if by_name else (head, head[:12])
        assert vr.worktree_problem(root, "results/a.csv", rev) is None
        (root / "results" / "b.csv").write_text("2\n")
        assert vr.worktree_problem(root, "results/b.csv", rev) == f"results/b.csv is not in {shown}"
        (root / "results" / "a.csv").write_text("edited\n")
        assert vr.worktree_problem(root, "results/a.csv", rev) == f"results/a.csv on disk differs from {shown}'s"
        (root / "results" / "a.csv").unlink()
        assert vr.worktree_problem(root, "results/a.csv", rev) == "results/a.csv is missing"

    @pytest.mark.parametrize("hide", ["skip-worktree", "assume-unchanged", "clean filter"])
    def test_an_edit_hidden_from_git_status_still_differs(self, tmp_path, hide):
        root = tmp_path / "r"
        head = repo(root, files={"results/a.csv": "1\n"})
        if hide == "clean filter":
            (root / ".git" / "info").mkdir(exist_ok=True)
            (root / ".git" / "info" / "attributes").write_text("results/a.csv filter=tidy\n")
            git(root, "config", "filter.tidy.clean", "sed 's/^2$/1/'")
        else:
            git(root, "update-index", f"--{hide}", "results/a.csv")
        (root / "results" / "a.csv").write_text("2\n")
        assert git(root, "status", "--porcelain") == ""                     # git status is fooled
        assert vr.worktree_problem(root, "results/a.csv", head) == f"results/a.csv on disk differs from {head[:12]}'s"

    def test_the_in_sample_results_added_once_by_the_tagged_commit(self, tmp_path):
        root = tmp_path / "r"
        tag = repo(root, files={IS_RESULTS: "bar\n15m\n", IS_SUPP: "{}\n"})
        assert vr.insample_provenance(root, tag) == ("bar\n15m\n", [vr.Check(
            "provenance", PASS, f"results/insample_v2.csv was committed once, by {tag[:12]} (prereg-freeze); the "
                                "copy on disk and results/manifest_is_supp.json are the committed ones")])

    @pytest.mark.parametrize("when", ["before the tag", "after it too"])
    def test_in_sample_results_not_added_once_by_the_tagged_commit_fail(self, tmp_path, when):
        root = tmp_path / "r"
        first = repo(root, files={IS_RESULTS: "bar\n15m\n", IS_SUPP: "{}\n"}, tag=(when == "after it too"))
        if when == "before the tag":
            tag = commit_files(root, {"README.md": "x\n"})
            git(root, "tag", "-a", TAG, "-m", "freeze")
            commits = [first]
        else:
            tag = first
            commits = [commit_files(root, {IS_RESULTS: "bar\n1H\n"}), tag]
        assert vr.insample_provenance(root, tag) == ("", [vr.Check(
            "provenance", FAIL, f"results/insample_v2.csv was committed in {[c[:12] for c in commits]}; §10 step 2 "
                                f"adds it once, in the tagged commit {tag[:12]}")])

    @pytest.mark.parametrize("rel,against", [(IS_RESULTS, "the tag"), (IS_SUPP, "HEAD")])
    def test_in_sample_copies_on_disk_must_be_the_committed_ones(self, tmp_path, rel, against):
        root = tmp_path / "r"
        tag = repo(root, files={IS_RESULTS: "bar\n15m\n", IS_SUPP: "{}\n"})
        (root / rel).write_text("edited\n")
        shown = tag[:12] if against == "the tag" else "HEAD"
        assert vr.insample_provenance(root, tag)[1] == [
            vr.Check("provenance", FAIL, f"{rel} on disk differs from {shown}'s")]

    @pytest.mark.parametrize("name,reads", [
        ("doge_2023", "results/oos/manifest_doge_2023.json, results/oos/doge_2024.csv, results/oos/majors_2023.csv"),
        ("doge_fwd", "results/manifest_is_supp.json, results/oos/manifest_doge_fwd.json"),
    ])
    def test_test_set_results_committed_once_after_their_run(self, tmp_path, name, reads):
        root = tmp_path / "r"
        tag, code, added = run_history(root, name)
        rels = [p.as_posix() for p in vr.results_paths(name)]
        manifest = pipeline.oos_manifest_path(name).as_posix()
        digest = sha256_file(root / manifest)
        prov = vr.oos_provenance(root, name, tag)
        assert prov.checks == [vr.Check("provenance", PASS, f"{', '.join(rels)} committed once, in {added[:12]}; "
                                                            f"the copies on disk, and {reads}, are the committed ones")]
        assert prov.texts == tuple((root / r).read_text(encoding="utf-8") for r in rels)
        assert (prov.code_commit, prov.manifest_sha256) == (code, digest)
        assert prov.column_notes == {
            "code_commit": (PASS, f"{code[:12]} on every row: the commit that added the set's manifest; descends "
                                  "from prereg-freeze with only docs/ and results/ changed and the same Appendix A; "
                                  f"the parent of {added[:12]}, which added the file"),
            "manifest_sha256": (PASS, f"{digest[:16]}... on every row: the SHA-256 of {manifest} in {code[:12]}, "
                                      "and of the file on disk"),
        }

    def test_test_set_results_committed_twice_fail(self, tmp_path):
        root = tmp_path / "r"
        tag, _, _ = run_history(root, "doge_fwd")
        commit_files(root, {"results/oos/doge_fwd.csv": "tidied\n"})
        message = "results/oos/doge_fwd.csv was committed 2 times; §10 step 5 commits a run's output once, unedited"
        prov = vr.oos_provenance(root, "doge_fwd", tag)
        assert prov.checks == [vr.Check("provenance", FAIL, message)]
        assert prov.column_notes == {c: (FAIL, f"see provenance: {message}") for c in vr.PROVENANCE_COLUMNS}

    @pytest.mark.parametrize("when", ["earlier", "later"])
    def test_a_robustness_file_that_did_not_come_with_the_results_fails(self, tmp_path, when):
        root = tmp_path / "r"
        tag, code, added = run_history(root, "doge_2023", robustness=when)
        rel = "results/oos/robustness_doge_2023.csv"
        by = code if when == "earlier" else git(root, "rev-parse", "HEAD")
        problems = [f"{rel} was committed in {[by[:12]]}, not once with results/oos/doge_2023.csv in {added[:12]}"]
        if when == "later":
            problems.append(f"{rel} is not in {added[:12]}")
        assert vr.oos_provenance(root, "doge_2023", tag).checks == [
            vr.Check("provenance", FAIL, "; ".join(problems))]

    @pytest.mark.parametrize("rel,against", [
        ("results/oos/doge_2023.csv", "the results' commit"),
        ("results/oos/robustness_doge_2023.csv", "the results' commit"),
        ("results/oos/doge_2024.csv", "HEAD"),        # a secondary set's, which the primary set's labels come from
    ])
    def test_a_copy_on_disk_that_is_not_the_committed_one_fails(self, tmp_path, rel, against):
        root = tmp_path / "r"
        tag, _, added = run_history(root, "doge_2023")
        (root / rel).write_text("edited\n")
        shown = "HEAD" if against == "HEAD" else added[:12]
        assert vr.oos_provenance(root, "doge_2023", tag).checks == [
            vr.Check("provenance", FAIL, f"{rel} on disk differs from {shown}'s")]

    def test_a_manifest_on_disk_that_is_not_the_one_the_run_hashed_fails(self, tmp_path):
        root = tmp_path / "r"
        tag, code, _ = run_history(root, "doge_2023")
        rel = "results/oos/manifest_doge_2023.json"
        hashed = sha256_file(root / rel)
        (root / rel).write_text('{"series": [{}]}\n')
        now = sha256_file(root / rel)
        prov = vr.oos_provenance(root, "doge_2023", tag)
        assert prov.checks == [vr.Check("provenance", FAIL, f"{rel} on disk hashes to {now}, not to its version "
                                                            f"in {code[:12]} ({hashed}); {rel} on disk differs from "
                                                            "HEAD's")]
        assert prov.column_notes["manifest_sha256"][0] == FAIL

    def test_the_directions_are_the_tags_and_the_mdes_the_amendment_in_the_runs_commit(self, tmp_path):
        f = SPEC["freeze"]
        root = tmp_path / "r"
        tag = repo(root, files={f["directions_file"]: (REPO_ROOT / f["directions_file"]).read_text(encoding="utf-8")})
        amendment = (REPO_ROOT / f["amendment_file"]).read_text(encoding="utf-8")
        recorded = freeze.parse_freeze_amendment(amendment).commit
        naming_the_tag = commit_files(root, {f["amendment_file"]: amendment.replace(recorded, tag)})
        naming_another = commit_files(root, {f["amendment_file"]: amendment})
        # Both files are read from those commits, whatever is on disk.
        for rel in (f["directions_file"], f["amendment_file"]):
            (root / rel).write_text("not what the run read\n")
        directions, mde, check = vr.frozen_directions(root, tag, naming_the_tag)
        assert len(directions) == len(mde) == 60
        assert check == vr.Check("directions", PASS, "60 directions (29 of them -1) and MDEs from "
                                                     "results/insample_v2.csv at prereg-freeze and "
                                                     f"docs/preregistration-freeze.md at {naming_the_tag[:12]}, "
                                                     "which agree")
        assert vr.frozen_directions(root, tag, naming_another)[2] == vr.Check(
            "directions", FAIL, f"results/insample_v2.csv at prereg-freeze and docs/preregistration-freeze.md at "
                                f"{naming_another[:12]} disagree: the amendment records {recorded[:12]}, but "
                                f"prereg-freeze is {tag[:12]}")

    def test_the_constants_check_is_run_oos_guard_5(self, tmp_path):
        root = tmp_path / "r"
        tag = repo(root, files={PREREG: (REPO_ROOT / PREREG).read_text(encoding="utf-8"), IS_RESULTS: "x\n"})
        assert vr.spec_check(root, tag) == vr.Check(
            "constants", PASS, "Appendix A as loaded and at HEAD equals prereg-freeze's; the directions file and "
                               "the in-sample supplement's manifest are unchanged since the tag")
        commit_files(root, {IS_RESULTS: "y\n"})
        assert vr.spec_check(root, tag) == vr.Check(
            "constants", FAIL, "results/insample_v2.csv changed since prereg-freeze; both are fixed before the tag "
                               "(§10 step 2) and the frozen directions depend on them")


# ---------------------------------------------------------------- the steps of one set

def stub_steps(monkeypatch, fail=(), robust=True):
    """
    The steps of one set with their checks stubbed: each passes unless named
    in ``fail``; the computation returns recomputed() (a test set's also
    robustness_rows(), unless not ``robust``), and the committed files are
    the run's text of those rows. Returns the steps in the order they ran,
    the code check with the files it was told to require.
    """
    ran = []

    def check(name):
        ran.append(name)
        return vr.Check(name, FAIL if name in fail else PASS, "stub")

    def oos_provenance(root, set_name, tag_commit):
        return vr.OosProvenance((vr.csv_text(recomputed()), vr.csv_text(robustness_rows())), "c" * 40, "d" * 64,
                                [check("provenance")], {c: (PASS, "stub") for c in vr.PROVENANCE_COLUMNS})

    def run_insample(root):
        ran.append("computation")
        return recomputed()

    def run_oos_set(set_name, directions, **kwargs):
        ran.append("computation")
        return recomputed(), robustness_rows() if robust else None

    def code_check(root, tag_commit, required):
        ran.append(("code", required))
        return vr.Check("code", FAIL if "code" in fail else PASS, "stub")

    monkeypatch.setattr(vr, "inputs_check", lambda root, set_name, git_ok: [check("inputs")])
    monkeypatch.setattr(vr, "insample_provenance",
                        lambda root, tag_commit: (vr.csv_text(recomputed()), [check("provenance")]))
    monkeypatch.setattr(vr, "oos_provenance", oos_provenance)
    monkeypatch.setattr(vr, "frozen_directions", lambda root, tag_commit, rev: ({}, {}, check("directions")))
    monkeypatch.setattr(pipeline, "run_insample", run_insample)
    monkeypatch.setattr(pipeline, "run_oos_set", run_oos_set)
    monkeypatch.setattr(vr, "code_check", code_check)
    return ran


class TestOneSet:
    @pytest.mark.parametrize("name,failing,ran", [
        ("insample_v2", "provenance", ["inputs", "provenance"]),
        ("doge_2024", "provenance", ["inputs", "provenance"]),
        ("doge_2024", "directions", ["inputs", "provenance", "directions"]),
    ])
    def test_a_failed_provenance_or_directions_check_stops_it_before_the_computation(
            self, monkeypatch, tmp_path, name, failing, ran):
        steps = stub_steps(monkeypatch, fail={failing})
        r = vr.verify_set(with_results(tmp_path, [name]), name, "0" * 40, True)
        assert steps == ran and r.tables == []
        assert [(c.name, c.verdict) for c in r.checks] == [(s, FAIL if s == failing else PASS) for s in ran]
        assert r.verdict() == FAIL

    @pytest.mark.parametrize("name,ran", [
        ("insample_v2", ["inputs", "provenance", "computation"]),
        ("doge_2024", ["inputs", "provenance", "directions", "computation"]),
    ])
    def test_the_code_check_follows_the_computation_and_its_failure_keeps_the_comparison(
            self, monkeypatch, tmp_path, name, ran):
        # The modules it hashes are the ones the computation ran.
        steps = stub_steps(monkeypatch, fail={"code"})
        r = vr.verify_set(with_results(tmp_path, [name]), name, "0" * 40, True)
        assert steps == ran + [("code", vr.REQUIRED_CODE)]
        assert r.checks[-1] == vr.Check("code", FAIL, "stub") and r.verdict() == FAIL
        assert [(rel, summary) for rel, summary, _ in r.tables] == [
            (vr.results_paths(name)[0].as_posix(), "3 rows x 11 columns, byte-identical to the recomputation")]

    def test_the_primary_set_compares_its_robustness_file_too(self, monkeypatch, tmp_path):
        steps = stub_steps(monkeypatch)
        r = vr.verify_set(with_results(tmp_path, ["doge_2023"]), "doge_2023", "0" * 40, True)
        # R2 runs v1's script, so the code check must find it among the modules that ran.
        assert steps[-1] == ("code", vr.REQUIRED_CODE + ("scripts/factor_research.py",))
        assert [(rel, summary) for rel, summary, _ in r.tables] == [
            ("results/oos/doge_2023.csv", "3 rows x 11 columns, byte-identical to the recomputation"),
            ("results/oos/robustness_doge_2023.csv", "2 rows x 7 columns, byte-identical to the recomputation")]
        assert r.notes == ["replicated and consistent_on_majors were computed from the statuses in the doge_2024, "
                           "majors_2023 results on disk, as in the run; those files are verified by their own --set"]
        assert r.verdict() == PASS

    def test_a_primary_set_without_robustness_rows_fails(self, monkeypatch, tmp_path):
        stub_steps(monkeypatch, robust=False)
        r = vr.verify_set(with_results(tmp_path, ["doge_2023"]), "doge_2023", "0" * 40, True)
        assert r.tables[1] == ("results/oos/robustness_doge_2023.csv", "not recomputed",
                               [vr.ColumnVerdict("(file)", "-", FAIL, "run_oos_set returned no robustness rows")])
        assert r.verdict() == FAIL


# ---------------------------------------------------------------- what the inputs line says

# The input files of a test set, as the inputs line lists them and says what
# was checked of them. doge_fwd's warm-up is read from the in-sample files:
# Appendix A has the candles' SHA-256, the in-sample supplement's manifest the
# others'.
INPUT_FILES = {
    "doge_2024": ("3 files (data/history/oos/doge_2024/DOGE-USDT-SWAP/candles_1m.parquet, "
                  "data/history/oos/doge_2024/DOGE-USDT/candles_1m.parquet, "
                  "data/history/oos/doge_2024/DOGE-USDT-SWAP/funding_rates.parquet) have its SHA-256s and cover "
                  "the set's window; no unlisted file is read"),
    "doge_fwd": ("6 files (data/history/DOGE-USDT-SWAP/candles_1m.parquet, "
                 "data/history/is_supp/DOGE-USDT/candles_1m.parquet, "
                 "data/history/is_supp/DOGE-USDT-SWAP/funding_rates.parquet, "
                 "data/history/oos/doge_fwd/DOGE-USDT-SWAP/candles_1m.parquet, "
                 "data/history/oos/doge_fwd/DOGE-USDT/candles_1m.parquet, "
                 "data/history/oos/doge_fwd/DOGE-USDT-SWAP/funding_rates.parquet) have the SHA-256s that Appendix A "
                 "(the in-sample candles), results/manifest_is_supp.json and results/oos/manifest_doge_fwd.json "
                 "record, and those results/oos/manifest_doge_fwd.json lists cover the set's window; no unlisted "
                 "file is read"),
}


class TestInputsLine:
    @pytest.mark.parametrize("name", ["doge_2024", "doge_fwd"])
    def test_a_test_set_without_its_history(self, monkeypatch, tmp_path, name):
        root = with_inputs(archive(tmp_path / "archive"), name)
        calls = []
        monkeypatch.setattr(vr.run_oos, "_inputs_match_and_cover", lambda root, set_name, m: calls.append(set_name))
        (check,) = vr.inputs_check(root, name, git_ok=False)
        assert calls == [name] and check.verdict == PASS
        assert check.detail == (f"results/oos/manifest_{name}.json covers the window (its history not checked: "
                                f"{vr.NO_HISTORY}); " + INPUT_FILES[name])

    @pytest.mark.parametrize("name", ["doge_2024", "doge_fwd"])
    def test_a_test_set_with_its_history(self, monkeypatch, tmp_path, name):
        root = with_inputs(tmp_path / "tree", name)
        rel, ref = f"results/oos/manifest_{name}.json", "refs/remotes/upstream/main"
        asked = stub_guard_8(monkeypatch, rel, "7" * 40, {ref: "8" * 40}, held=True)
        inputs, pushed = vr.inputs_check(root, name, git_ok=True)
        # Guard 8 ran whole, with the public main branch standing in for origin/main.
        assert asked == [(name, ref)] and freeze.REMOTE_MAIN == "refs/remotes/origin/main"
        # Its history questions concern the set's own manifest alone: doge_fwd's
        # other one, the in-sample supplement's, was committed before the tag.
        assert (inputs.name, inputs.verdict) == ("inputs", PASS)
        assert inputs.detail == (f"{rel} was committed once, after the freeze amendment, records the public "
                                 "repository and covers the window; " + INPUT_FILES[name])
        assert pushed == vr.Check("pushed", PASS, f"{rel} was committed in 777777777777, which {ref} (the public "
                                                  "repository's main branch) holds (§10 step 4)")

    @pytest.mark.parametrize("tips,why", [
        ({"refs/remotes/upstream/main": "8" * 40}, "refs/remotes/upstream/main does not hold it"),
        ({}, f"no remote here points at {freeze.PUBLIC_REMOTE}"),
    ])
    def test_without_the_push_on_a_public_main_branch_the_rest_of_guard_8_still_runs(
            self, monkeypatch, tmp_path, tips, why):
        root = with_inputs(tmp_path / "tree", "doge_fwd")
        rel, added = "results/oos/manifest_doge_fwd.json", "7" * 40
        asked = stub_guard_8(monkeypatch, rel, added, tips, held=False)
        inputs, pushed = vr.inputs_check(root, "doge_fwd", git_ok=True)
        # The push question is answered by the commit itself; "pushed" says what this clone cannot show.
        assert asked == [("doge_fwd", added)]
        assert inputs.verdict == PASS and inputs.detail.startswith(f"{rel} was committed once, after the freeze ")
        assert (pushed.name, pushed.verdict) == ("pushed", NOT_CHECKED)
        assert pushed.detail.startswith(f"{rel} was committed in 777777777777, but {why}, so this clone cannot show")

    def test_a_refusal_of_guard_8_fails_the_inputs_alone(self, monkeypatch, tmp_path):
        root = with_inputs(tmp_path / "tree", "doge_fwd")
        rel = "results/oos/manifest_doge_fwd.json"
        stub_guard_8(monkeypatch, rel, "7" * 40, {"refs/remotes/upstream/main": "8" * 40}, held=True)
        message = f"{rel} was committed 2 times; a manifest is committed once, as fetched (§10 step 4)"

        def refuse(root, set_name):
            raise vr.run_oos.Refusal(message)

        monkeypatch.setattr(vr.run_oos, "check_manifest", refuse)
        assert vr.inputs_check(root, "doge_fwd", git_ok=True) == [vr.Check("inputs", FAIL, message)]
        assert freeze.REMOTE_MAIN == "refs/remotes/origin/main"

    def test_the_in_sample_year(self, monkeypatch, tmp_path):
        root = with_inputs(archive(tmp_path / "archive"), "insample_v2")
        monkeypatch.setattr(pipeline, "verify_inputs", lambda set_name, root: [])
        (check,) = vr.inputs_check(root, "insample_v2", git_ok=False)
        assert check.verdict == PASS
        assert check.detail == ("3 files (data/history/DOGE-USDT-SWAP/candles_1m.parquet, "
                                "data/history/is_supp/DOGE-USDT/candles_1m.parquet, "
                                "data/history/is_supp/DOGE-USDT-SWAP/funding_rates.parquet); the candles have "
                                "Appendix A's SHA-256, the others results/manifest_is_supp.json's, and no unlisted "
                                "file is read")


# ---------------------------------------------------------------- pushed

class TestPushed:
    def test_every_remote_naming_the_public_repository_is_asked(self, monkeypatch, tmp_path):
        public = tmp_path / "public.git"
        monkeypatch.setattr(freeze, "PUBLIC_REMOTE", str(public))
        root = tmp_path / "r"
        repo(root)
        git(root, "remote", "add", "origin", str(tmp_path / "fork.git"))
        git(root, "remote", "add", "upstream", str(public))
        git(root, "remote", "add", "mirror", f"file://{public}")
        assert sorted(vr.public_main_refs(root)) == ["refs/remotes/mirror/main", "refs/remotes/upstream/main"]

    def test_the_answer_is_a_check_of_its_own(self):
        rel, added, ref = "results/oos/manifest_doge_fwd.json", "7" * 40, "refs/remotes/upstream/main"
        held = vr._pushed_check(rel, added, [ref], [ref])
        assert held.verdict == PASS and f"in {'7' * 12}, which {ref}" in held.detail
        unheld = vr._pushed_check(rel, added, [], [ref])
        assert unheld.verdict == NOT_CHECKED and f"{ref} does not hold it" in unheld.detail
        nowhere = vr._pushed_check(rel, added, [], [])
        assert nowhere.verdict == NOT_CHECKED and f"no remote here points at {freeze.PUBLIC_REMOTE}" in nowhere.detail
        assert nowhere.detail.endswith("Clone from the public repository, or add it as a remote and fetch it")


# ---------------------------------------------------------------- the code check

class TestCode:
    def test_modules_are_placed_by_the_path_they_were_loaded_under(self, monkeypatch, tmp_path):
        root, elsewhere = tmp_path / "checkout", tmp_path / "elsewhere"
        for rel in ("research/own.py", "elsewhere/real.py", "elsewhere/other.py", "outside.py"):
            path = root / rel if rel.startswith("research") else tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        (root / "research" / "linked.py").symlink_to(elsewhere / "real.py")
        (root / "scripts").mkdir()
        (root / "scripts" / "run_oos.py").symlink_to(elsewhere / "other.py")
        module(monkeypatch, "_vr_test_own", root / "research" / "own.py")
        module(monkeypatch, "_vr_test_linked", root / "research" / "linked.py")   # under the link's own name
        module(monkeypatch, "_vr_test_resolved", elsewhere / "other.py")         # as integrity.load_script records it
        module(monkeypatch, "_vr_test_outside", tmp_path / "outside.py")
        assert vr.loaded_checkout_files(root, ["scripts/run_oos.py"]) == \
            ["research/linked.py", "research/own.py", "scripts/run_oos.py"]
        # Without that required name, a path outside the checkout says nothing about it.
        assert vr.loaded_checkout_files(root) == ["research/linked.py", "research/own.py"]

    def test_this_script_is_left_out_and_v1s_script_counts(self, monkeypatch):
        # This file is not frozen code; R2's v1 script is, though research.stats keeps it out of sys.modules.
        module(monkeypatch, "_vr_test_self", SCRIPT)
        vr.stats._v1_module()
        files = vr.loaded_checkout_files(REPO_ROOT)
        assert "scripts/verify_results.py" not in files
        assert {"scripts/factor_research.py", "scripts/run_oos.py", "research/pipeline.py"} <= set(files)

    def test_the_modules_must_be_the_tags_byte_for_byte(self, monkeypatch, tmp_path):
        root = tmp_path / "checkout"
        tag = repo(root, files={"research/pipeline.py": "X = 1\n", "research/other.py": "Y = 2\n"})
        # This process's own research package is the real checkout's, foreign to the throwaway one.
        monkeypatch.setattr(vr.integrity, "foreign_code_problem", lambda root: None)
        module(monkeypatch, "_vr_test_pipeline", root / "research" / "pipeline.py")
        module(monkeypatch, "_vr_test_other", root / "research" / "other.py")
        required = ("research/pipeline.py",)
        check = vr.code_check(root, tag, required)
        assert check.verdict == PASS and check.detail.endswith(": research/other.py, research/pipeline.py")
        assert vr.code_check(root, None, required).verdict == NOT_CHECKED
        missing = vr.code_check(root, tag, required + ("scripts/run_oos.py",))
        assert missing.verdict == FAIL
        assert missing.detail.startswith("scripts/run_oos.py did not run from this checkout")
        (root / "research" / "other.py").write_text("Y = 3\n")
        check = vr.code_check(root, tag, required)
        assert check.verdict == FAIL and "that are not prereg-freeze's: research/other.py." in check.detail

    @pytest.mark.parametrize("same", [True, False])
    def test_a_linked_module_is_judged_by_what_it_points_to(self, monkeypatch, tmp_path, same):
        root = tmp_path / "checkout"
        tag = repo(root, files={"research/pipeline.py": "X = 1\n"})
        copy = tmp_path / "elsewhere" / "pipeline.py"
        copy.parent.mkdir()
        copy.write_text("X = 1\n" if same else "X = 2\n")
        (root / "research" / "pipeline.py").unlink()
        (root / "research" / "pipeline.py").symlink_to(copy)
        monkeypatch.setattr(vr.integrity, "foreign_code_problem", lambda root: None)
        module(monkeypatch, "_vr_test_pipeline", root / "research" / "pipeline.py")
        assert vr.code_check(root, tag, ("research/pipeline.py",)).verdict == (PASS if same else FAIL)

    def test_a_module_of_the_study_from_another_directory_fails(self, tmp_path):
        root = tmp_path / "checkout"
        tag = repo(root)
        # The real research package, loaded in this process, is foreign to the throwaway checkout.
        check = vr.code_check(root, tag, ())
        assert check.verdict == FAIL and "loaded from elsewhere" in check.detail


# ---------------------------------------------------------------- with the data

# Every file doge_fwd reads, the optional spot leg and funding included: their
# manifests list them, so without one the inputs check fails and nothing is
# recomputed.
FWD_INPUTS = [f.path for f in pipeline.input_files("doge_fwd", REPO_ROOT)]
FWD_INPUTS_LINE = ("  [PASS] inputs: results/oos/manifest_doge_fwd.json was committed once, after the freeze "
                   "amendment, records the public repository and covers the window; " + INPUT_FILES["doge_fwd"])


@pytest.mark.skipif(not all(p.exists() for p in FWD_INPUTS),
                    reason="needs every file doge_fwd reads, its own and the in-sample files of its warm-up "
                           "(data/ is not in the repository)")
class TestWithTheData:
    def test_doge_fwd_recomputes_byte_for_byte(self, tmp_path):
        if vr.git_state(REPO_ROOT)[0] is None:
            pytest.skip("needs the whole history and the prereg-freeze tag as well as the data")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run([sys.executable, str(SCRIPT), "--set", "doge_fwd"], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=600)
        out, shown = proc.stdout.splitlines(), proc.stdout + proc.stderr
        assert [line for line in out if line.startswith("  [FAIL] ")] == [], shown
        assert FWD_INPUTS_LINE in out, shown
        assert "  results/oos/doge_fwd.csv: 60 rows x 38 columns, byte-identical to the recomputation" in out, shown
        unchecked = [line[len("  [NOT CHECKED] "):].split(":")[0] for line in out
                     if line.startswith("  [NOT CHECKED] ")]
        if unchecked == ["pushed"]:
            # A clone none of whose remotes is the public repository (a fork's)
            # cannot show the push; everything else was checked.
            assert (proc.returncode, out[-1]) == (3, "overall: INCOMPLETE (doge_fwd INCOMPLETE)"), shown
        else:
            assert (proc.returncode, out[-1]) == (0, "overall: PASS (doge_fwd PASS)"), shown
        assert list(tmp_path.iterdir()) == []
