"""
scripts/write_freeze_amendment.py (§10 step 3) against throwaway git repositories.

Each test builds the repository as it will stand when the amendment is
written: the published pre-registration, code and the committed maker-probe
candidate list, then a commit adding only results/insample_v2.csv, tagged
prereg-freeze. The amendment the script writes must then read back exactly as
scripts/run_oos.py will read it: parse_freeze_amendment, compare_with_insample
against the tag's file and, once committed and pushed to a bare 'origin'
standing in for the public repository, verify_freeze itself. The other tests
break one input at a time and check that nothing is written.

git runs with the user's global and system config disabled and a fixed
identity, so nothing outside tmp_path is read or changed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from research import freeze
from research.spec import REPO_ROOT, get_spec


def _load(name):
    found = importlib.util.spec_from_file_location(f"_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(found)
    found.loader.exec_module(mod)
    return mod


wfa = _load("write_freeze_amendment")
mp = wfa.mp

SPEC = get_spec()
TAG = SPEC["freeze"]["tag"]
AMENDMENT = SPEC["freeze"]["amendment_file"]
DIRECTIONS = SPEC["freeze"]["directions_file"]
CANDIDATES = SPEC["maker_probe"]["candidates_file"]
ALL_DAYS = [d.isoformat() for d in mp.population(SPEC)]
CANDS = ALL_DAYS[::5]                                          # 73 candidates
HEX = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(k, raising=False)
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")
    # The bare 'origin' Repo.push creates beside tmp_path/'repo' plays the public repository.
    monkeypatch.setattr(freeze, "PUBLIC_REMOTE", str(tmp_path / "repo-origin.git"))


@pytest.fixture
def messages():
    out = []
    hid = logger.add(lambda m: out.append(m.record["message"]), level="INFO")
    yield out
    logger.remove(hid)


def joined(messages):
    return "\n".join(messages)


def insample_text(flip=None):
    """A well-formed 60-row results/insample_v2.csv, with one cell whose MDE is undefined."""
    lines = [",".join(freeze.INSAMPLE_COLUMNS)]
    for i, key in enumerate(freeze.expected_cells()):
        b, h, f = key
        d = -1 if i % 3 == 0 else 1
        if key == flip:
            d = -d
        if key == ("1H", 15, "funding_cost_pressure"):
            lines.append(f"{b},{h},{f},1,true,missing_input,0,,,,")
        else:
            lines.append(f"{b},{h},{f},{d},false,,900,{d * 0.25 * (i + 1)},{1 + i / 10},hh,{3.986 * (1 + i / 10):.10g}")
    return "\n".join(lines) + "\n"


def candidate_manifest(cands):
    return json.dumps({"manifest_version": 1, "inst_id": "DOGE-USDT-SWAP", "candidates": cands,
                       "days": {d: {"qualifies": True, "missing": []} for d in cands}}, indent=2) + "\n"


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
        return p

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def push(self):
        origin = self.path.parent / f"{self.path.name}-origin.git"
        if not origin.exists():
            subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True, capture_output=True)
            self.git("remote", "add", "origin", str(origin))
        self.git("push", "-q", "origin", "main")


def build(tmp_path, *, cands=CANDS, commit_cands=True, tag_extra=None, tag=True):
    """
    c0 README -> pub docs/preregistration.md -> c1 code + candidate list
    -> c2 adds results/insample_v2.csv [tag prereg-freeze].
    """
    repo = Repo(tmp_path / "repo")
    repo.write("README.md", "x\n")
    repo.c0 = repo.commit("start")
    repo.write("docs/preregistration.md", "# Pre-registration\n")
    repo.pub = repo.commit("publish the pre-registration")
    repo.write("research/code.py", "X = 1\n")
    if cands is not None and commit_cands:
        repo.write(CANDIDATES, candidate_manifest(cands))
    repo.c1 = repo.commit("code and candidate days")
    repo.write(DIRECTIONS, insample_text())
    if tag_extra:
        repo.write(*tag_extra)
    repo.c2 = repo.commit("in-sample directions")
    if tag:
        repo.git("tag", "-a", TAG, "-m", "freeze")
    if cands is not None and not commit_cands:
        repo.write(CANDIDATES, candidate_manifest(cands))
    return repo


def write(repo, pub=None, swh_pub=None, swh_freeze=None):
    return wfa.main(["--publication-commit", pub or repo.pub,
                     "--swh-publication", swh_pub or f"swh:1:rev:{repo.pub}",
                     "--swh-freeze", swh_freeze or f"swh:1:rev:{repo.c2}"], root=repo.path)


def amendment(repo):
    return (repo.path / AMENDMENT).read_text()


def numpy_days(cands, n=14):
    return sorted(str(x) for x in np.random.default_rng(20261008).choice(cands, size=min(n, len(cands)),
                                                                          replace=False))


def refused(repo, code, messages, needle, exit_code=wfa.EXIT_REFUSED):
    assert code == exit_code, joined(messages)
    assert needle in joined(messages), joined(messages)
    assert not (repo.path / AMENDMENT).exists() or exit_code == wfa.EXIT_EXISTS


# ---------------------------------------------------------------- the amendment written

class TestWrites:
    def test_the_amendment_reads_back_and_verifies(self, tmp_path, messages):
        repo = build(tmp_path)
        assert write(repo) == wfa.EXIT_OK, joined(messages)
        text = amendment(repo)
        rec = freeze.parse_freeze_amendment(text)
        assert rec.commit == repo.c2 == repo.git("rev-parse", f"{TAG}^{{commit}}")
        at_tag = freeze.parse_insample_csv(repo.git("show", f"{TAG}:{DIRECTIONS}") + "\n")
        assert freeze.compare_with_insample(rec, at_tag) == []
        assert set(freeze.freeze_blocks(text)) == {"freeze-commit", "freeze-directions"}
        assert all(len(v) == 1 for v in freeze.freeze_blocks(text).values())
        days = numpy_days(CANDS)
        assert wfa.maker_probe_days_from(text) == days == mp.committed_draw(repo.path, SPEC).days
        assert "```maker-probe-days\n" + "\n".join(days) + "\n```\n" in text
        flat = " ".join(text.split())                                    # the prose, unwrapped
        for item in (", ".join(days), repo.pub, f"swh:1:rev:{repo.pub}", f"swh:1:rev:{repo.c2}", "undefined",
                     "# Freeze amendment to the pre-registration", f"(14 of {len(CANDS)} candidate days)"):
            assert item in flat
        # Written, not committed: the next commands are printed instead.
        assert repo.git("status", "--porcelain") == f"?? {AMENDMENT}"
        for cmd in (f"git add {AMENDMENT}", "git commit -m", "git push origin main"):
            assert cmd in joined(messages)
        # Committed and pushed as printed, it is what the one run accepts.
        repo.commit("Record the freeze amendment")
        repo.push()
        assert freeze.verify_freeze(repo.path) == rec

    def test_directions_come_from_the_tag_not_the_working_tree(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.write(DIRECTIONS, insample_text(flip=("15m", 5, "ou_reversion")))
        assert write(repo) == wfa.EXIT_OK, joined(messages)
        rec = freeze.parse_freeze_amendment(amendment(repo))
        assert freeze.compare_with_insample(rec, freeze.parse_insample_csv(insample_text())) == []

    def test_fewer_than_n_days_uses_them_all(self, tmp_path, messages):
        few = CANDS[:6]
        repo = build(tmp_path, cands=few)
        assert write(repo) == wfa.EXIT_OK, joined(messages)
        text = amendment(repo)
        assert wfa.maker_probe_days_from(text) == few
        assert "6 days qualified, fewer than 14, so all are used" in " ".join(text.split())

    def test_the_candidate_list_may_be_committed_after_the_tag(self, tmp_path, messages):
        repo = build(tmp_path, commit_cands=False)
        repo.commit("candidate days")
        assert write(repo) == wfa.EXIT_OK, joined(messages)
        assert wfa.maker_probe_days_from(amendment(repo)) == numpy_days(CANDS)

    @pytest.mark.parametrize("kinds", [("rev", "rel"), ("dir", "dir"), ("rev", "snp")])
    def test_other_identifier_kinds_and_qualifiers(self, tmp_path, messages, kinds):
        repo = build(tmp_path)
        ids = {
            "rev": lambda c: f"swh:1:rev:{c};origin=https://github.com/ikunfloyd/crypto-factor-validation",
            "dir": lambda c: f"swh:1:dir:{repo.git('rev-parse', c + '^{tree}')}",
            "rel": lambda c: f"swh:1:rel:{repo.git('rev-parse', 'refs/tags/' + TAG)}",
            "snp": lambda c: f"swh:1:snp:{HEX};origin=https://github.com/ikunfloyd/crypto-factor-validation",
        }
        swh_pub, swh_freeze = ids[kinds[0]](repo.pub), ids[kinds[1]](repo.c2)
        assert write(repo, swh_pub=swh_pub, swh_freeze=swh_freeze) == wfa.EXIT_OK, joined(messages)
        assert f"Publication commit: `{swh_pub}`" in " ".join(amendment(repo).split())
        assert swh_freeze in amendment(repo)
        assert ("snapshot identifier" in joined(messages)) is (kinds[1] == "snp")


# ---------------------------------------------------------------- refusals

class TestRefuses:
    def test_an_existing_amendment(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.write(AMENDMENT, "draft\n")
        refused(repo, write(repo), messages, "is written once", wfa.EXIT_EXISTS)
        assert amendment(repo) == "draft\n"

    def test_an_amendment_committed_before(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.write(AMENDMENT, "draft\n")
        repo.commit("draft amendment")
        repo.git("rm", "-q", AMENDMENT)
        repo.commit("remove it")
        refused(repo, write(repo), messages, "was committed before", wfa.EXIT_EXISTS)
        assert not (repo.path / AMENDMENT).exists()

    def test_no_tag(self, tmp_path, messages):
        repo = build(tmp_path, tag=False)
        refused(repo, write(repo), messages, f"there is no tag {TAG}")

    def test_a_branch_named_like_the_tag_is_not_the_tag(self, tmp_path, messages):
        repo = build(tmp_path, tag=False)
        repo.git("branch", TAG)
        refused(repo, write(repo), messages, f"there is no tag {TAG}")

    def test_a_tagged_commit_that_changes_more(self, tmp_path, messages):
        repo = build(tmp_path, tag_extra=("research/code.py", "X = 2\n"))
        refused(repo, write(repo), messages, f"must tag the commit that adds {DIRECTIONS} and nothing else")

    def test_a_tag_that_is_not_an_ancestor_of_head(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.git("checkout", "-q", "-b", "other", repo.c1)
        refused(repo, write(repo), messages, "is not an ancestor of HEAD")

    def test_a_publication_commit_that_is_not_an_ancestor_of_the_tag(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.git("checkout", "-q", "-b", "side", repo.pub)
        repo.write("docs/preregistration.md", "# Pre-registration, edited\n")
        side = repo.commit("a later version, never released")
        repo.git("checkout", "-q", "main")
        refused(repo, write(repo, pub=side, swh_pub=f"swh:1:rev:{side}"), messages,
                f"is not an ancestor of {TAG}")

    def test_a_publication_commit_without_the_preregistration(self, tmp_path, messages):
        repo = build(tmp_path)
        refused(repo, write(repo, pub=repo.c0, swh_pub=f"swh:1:rev:{repo.c0}"), messages,
                "has no docs/preregistration.md")

    def test_a_publication_commit_that_does_not_exist(self, tmp_path, messages):
        repo = build(tmp_path)
        refused(repo, write(repo, pub=HEX, swh_pub=f"swh:1:rev:{HEX}"), messages, "is not a commit of this repository")

    @pytest.mark.parametrize("pub", ["abc1234", HEX.upper(), "HEAD", HEX + "0"])
    def test_a_publication_commit_that_is_not_a_full_hash(self, tmp_path, messages, pub):
        repo = build(tmp_path)
        refused(repo, write(repo, pub=pub), messages, "must be a full 40-hex commit", wfa.EXIT_USAGE)

    @pytest.mark.parametrize("which", ["--swh-publication", "--swh-freeze"])
    @pytest.mark.parametrize("bad", [
        f"swh:1:cnt:{HEX}",                              # a file, not one of the four kinds
        f"swh:1:rev:{HEX[:-1]}",                         # 39 hex
        f"swh:1:rev:{HEX.upper()}",
        f"swh:2:rev:{HEX}",
        f"rev:{HEX}",
        f"swh:1:rev:{HEX};origin",                       # a qualifier without a value
        f"swh:1:rev:{HEX};origin=https://example.org/a b",
        f"swh:1:rev:{HEX}:extra",
    ])
    def test_malformed_software_heritage_ids(self, tmp_path, messages, which, bad):
        repo = build(tmp_path)
        args = {"swh_pub": bad} if which == "--swh-publication" else {"swh_freeze": bad}
        refused(repo, write(repo, **args), messages, f"{which} must be swh:1:", wfa.EXIT_USAGE)

    @pytest.mark.parametrize("case", ["rev_of_tag", "rev_of_pub", "dir_of_other", "rel_of_commit", "rel_elsewhere"])
    def test_software_heritage_ids_must_name_their_commits(self, tmp_path, messages, case):
        repo = build(tmp_path)
        repo.git("tag", "-a", "v1", "-m", "release", repo.pub)
        swh_pub, swh_freeze = f"swh:1:rev:{repo.pub}", f"swh:1:rev:{repo.c2}"
        if case == "rev_of_tag":
            swh_pub, needle = f"swh:1:rev:{repo.c2}", "--swh-publication names revision"
        elif case == "rev_of_pub":
            swh_freeze, needle = f"swh:1:rev:{repo.pub}", "--swh-freeze names revision"
        elif case == "dir_of_other":
            swh_freeze, needle = f"swh:1:dir:{repo.git('rev-parse', repo.pub + '^{tree}')}", "names directory"
        elif case == "rel_of_commit":
            swh_freeze, needle = f"swh:1:rel:{repo.c2}", "is not a tag object"
        else:
            swh_freeze, needle = f"swh:1:rel:{repo.git('rev-parse', 'refs/tags/v1')}", "which points to"
        refused(repo, write(repo, swh_pub=swh_pub, swh_freeze=swh_freeze), messages, needle)

    def test_an_uncommitted_candidate_list(self, tmp_path, messages):
        repo = build(tmp_path, commit_cands=False)
        refused(repo, write(repo), messages, f"{CANDIDATES} is not committed")

    def test_a_candidate_list_with_uncommitted_changes(self, tmp_path, messages):
        repo = build(tmp_path)
        repo.write(CANDIDATES, candidate_manifest(CANDS[1:]))
        refused(repo, write(repo), messages, "has uncommitted changes")

    def test_another_numpy(self, tmp_path, messages, monkeypatch):
        repo = build(tmp_path)
        monkeypatch.setattr(np, "__version__", "2.0.0")
        refused(repo, write(repo), messages, "numpy is 2.0.0")

    def test_a_rendering_that_does_not_read_back(self, tmp_path, messages, monkeypatch):
        repo = build(tmp_path)
        real = freeze.render_amendment_blocks
        flipped = insample_text(flip=("1H", 5, "rsi_14"))
        monkeypatch.setattr(freeze, "render_amendment_blocks", lambda commit, text, spec=None: real(commit, flipped, spec))
        refused(repo, write(repo), messages, "('1H', 5, 'rsi_14'): direction")

    def test_foreign_code(self, tmp_path, messages, monkeypatch):
        repo = build(tmp_path)
        monkeypatch.setattr(wfa.integrity, "foreign_code_problem", lambda root: "research from elsewhere")
        refused(repo, write(repo), messages, "research from elsewhere")


# ---------------------------------------------------------------- reading the days back

@pytest.mark.parametrize("text,needle", [
    ("no block\n", "found 0"),
    ("```maker-probe-days\n2025-10-01\n```\n```maker-probe-days\n2025-10-02\n```\n", "found 2"),
    ("```maker-probe-days\n```\n", "empty"),
    ("```maker-probe-days\n2025-10-02\n2025-10-01\n```\n", "not ascending"),
    ("```maker-probe-days\n2025-10-01\n2025-10-01\n```\n", "not ascending"),
    ("```maker-probe-days\n2025-13-01\n```\n", "does not match format"),
])
def test_bad_days_blocks_are_refused(text, needle):
    with pytest.raises(ValueError, match=needle):
        wfa.maker_probe_days_from(text)


def test_the_days_block_is_not_a_freeze_block():
    text = "```maker-probe-days\n2025-10-01\n```\n"
    assert freeze.freeze_blocks(text) == {}
    assert wfa.maker_probe_days_from(text) == ["2025-10-01"]
