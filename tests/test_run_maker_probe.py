"""
scripts/run_maker_probe.py (§11): the guards before any file is read, the
choice of each day's files, and one small synthetic day taken through
parsing, the day checks and both runs. No real archive file is read.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import tarfile
import zipfile
from types import SimpleNamespace

import polars as pl
import pytest

from research import maker_probe as probe
from research.spec import REPO_ROOT


def _load(name):
    found = importlib.util.spec_from_file_location(f"_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(found)
    found.loader.exec_module(mod)
    return mod


rmp = _load("run_maker_probe")

DAY, NEXT = "2026-01-05", "2026-01-06"          # inside the in-sample year, as every probe day is
T0 = probe._day_start_ms(DAY)
CFG = probe.ProbeConfig.from_spec()


def manifest_doc(files):
    return {"days": [DAY], "archive_dir": "data/history/maker_probe",
            "files": [{"kind": k, "file_date": d, "filename": n, "sha256": "0" * 64} for k, d, n in files]}


FILES = [("orderbook", DAY, "book-d.tar.gz"), ("orderbook", NEXT, "book-d1.tar.gz"),
         ("trades", DAY, "trades-d.zip"), ("trades", NEXT, "trades-d1.zip")]


def test_day_paths_takes_the_files_dated_d_and_d_plus_one():
    paths = {n: f"/x/{n}" for _, _, n in FILES}
    got = rmp.day_paths(manifest_doc(FILES), paths, DAY)
    assert got == {"orderbook": ["/x/book-d.tar.gz", "/x/book-d1.tar.gz"], "trades": ["/x/trades-d.zip", "/x/trades-d1.zip"]}
    with pytest.raises(rmp.Refusal, match="lists 0 trades files dated 2026-01-06"):
        rmp.day_paths(manifest_doc(FILES[:3]), paths, DAY)


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True,
                              text=True).stdout.strip()

    def commit(rel, text, message="c"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
        git("add", rel)
        git("commit", "-q", "-m", message)

    git("init", "-q", "-b", "main")
    commit("research/maker_probe.py", "# engine\n")
    return SimpleNamespace(path=tmp_path, git=git, commit=commit)


def test_refuses_before_the_archive_manifest_is_committed(repo):
    with pytest.raises(rmp.Refusal, match="not committed"):
        rmp.check_before_run(repo.path)


def test_reads_the_manifest_as_committed(repo):
    repo.commit(rmp.ARCHIVE_MANIFEST, json.dumps(manifest_doc(FILES)))
    assert rmp.check_before_run(repo.path)["days"] == [DAY]


def test_refuses_an_edited_manifest(repo):
    repo.commit(rmp.ARCHIVE_MANIFEST, json.dumps(manifest_doc(FILES)))
    (repo.path / rmp.ARCHIVE_MANIFEST).write_text("{}")
    with pytest.raises(rmp.Refusal, match="uncommitted changes"):
        rmp.check_before_run(repo.path)
    repo.commit(rmp.ARCHIVE_MANIFEST, json.dumps(manifest_doc(FILES[:2])))
    with pytest.raises(rmp.Refusal, match="changed after its first commit"):
        rmp.check_before_run(repo.path)


def test_refuses_uncommitted_code(repo):
    repo.commit(rmp.ARCHIVE_MANIFEST, json.dumps(manifest_doc(FILES)))
    (repo.path / "research" / "maker_probe.py").write_text("# edited\n")
    with pytest.raises(rmp.Refusal, match="research/maker_probe.py"):
        rmp.check_before_run(repo.path)


@pytest.mark.parametrize("committed", [False, True])
def test_runs_once(repo, committed):
    repo.commit(rmp.ARCHIVE_MANIFEST, json.dumps(manifest_doc(FILES)))
    out = f"{rmp.OUT_DIR}/summary.csv"
    if committed:
        repo.commit(out, "x\n")
        repo.git("rm", "-q", out)
        repo.git("commit", "-q", "-m", "removed again")
    else:
        (repo.path / out).write_text("x\n")
    with pytest.raises(rmp.Refusal, match="runs once") as exc:
        rmp.check_before_run(repo.path)
    assert exc.value.code == rmp.EXIT_EXISTS


def test_refuses_a_file_whose_hash_differs(repo):
    doc = manifest_doc(FILES[:1])
    folder = repo.path / doc["archive_dir"]
    folder.mkdir(parents=True)
    (folder / "book-d.tar.gz").write_bytes(b"x")
    with pytest.raises(rmp.Refusal, match="SHA-256"):
        rmp.check_files(repo.path, doc)


# --------------------------------------------------------------------------
# one small day, end to end
# --------------------------------------------------------------------------

def _lv(price, size):
    return [f"{price:.5f}", f"{size:.2f}", "1"]


def _book(path, messages):
    data = "".join(json.dumps(m) + "\n" for m in messages).encode()
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(path.name.replace(".tar.gz", ".data"))
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return path


def _trades(path, rows):
    text = "instrument_name,trade_id,side,price,size,created_time\n" + "".join(
        f"DOGE-USDT-SWAP,{i},{s},{p},{z},{t}\n" for i, s, p, z, t in rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.name.replace(".zip", ".csv"), text)
    return path


def _snapshot(ts, bid_size=5.0):
    return {"instId": "DOGE-USDT-SWAP", "action": "snapshot", "ts": str(ts),
            "bids": [_lv(0.20000, bid_size), _lv(0.19999, 9)], "asks": [_lv(0.20001, 5), _lv(0.20002, 9)]}


def _update(ts, bids=(), asks=()):
    return {"instId": "DOGE-USDT-SWAP", "action": "update", "ts": str(ts), "bids": list(bids), "asks": list(asks)}


def test_run_day_parses_checks_and_runs_both_ways(tmp_path):
    # Three minutes of book, a snapshot each minute; the third snapshot's bid
    # size differs from the rebuilt book (7 against 6), so touch_gaps drops
    # the minute before it.
    msgs = [_snapshot(T0 + 4)]
    msgs += [_update(T0 + k * 1000 + 4, bids=[_lv(0.20000, 5 + (k % 2))]) for k in range(1, 180)]
    msgs[60:60] = [_snapshot(T0 + 60_004, bid_size=6.0)]
    msgs[121:121] = [_snapshot(T0 + 120_004, bid_size=7.0)]
    book_d = _book(tmp_path / "book-d.tar.gz", msgs)
    book_d1 = _book(tmp_path / "book-d1.tar.gz", [])           # the data ends inside day D
    rows = [(1, "sell", "0.20000", "1.00", T0 + 30_000), (2, "buy", "0.20002", "2.00", T0 + 90_000),
            (3, "buy", "0.20001", "1.00", T0 + 86_400_000 + 900_000)]
    tr_d = _trades(tmp_path / "trades-d.zip", rows[:2])
    tr_d1 = _trades(tmp_path / "trades-d1.zip", rows[2:])
    got = rmp.run_day(DAY, {"orderbook": [book_d, book_d1], "trades": [tr_d, tr_d1]}, CFG)
    for run in rmp.RUNS:
        o = got["orders"][run]
        assert o.height == 288 * 2 * 2 and set(o["run"]) == {run}
    c = got["checks"]
    assert c["snapshot_sides_checked"] == 4 and c["snapshot_sides_touch_mismatch"] == 1
    assert c["max_depth_bid"] == 2 and c["parser_gaps"] == 0 and c["touch_gaps"] == 1
    assert c["touch_gap_ms"] == 60_000 - 1 and c["trades"] == 2 and c["trade_gaps"] == 0
    assert c["book_ended_early"] is True and c["book_end_ms"] == T0 + 179_005
    # the 00:00 decision has no book yet; decisions after the data are no_data, in both runs
    first = got["orders"]["parser_gaps"].filter(pl.col("decision_ts") == T0)
    assert set(first["status"]) == {"no_data"}


# --------------------------------------------------------------------------
# --verify
# --------------------------------------------------------------------------

def _csv(df):
    return df.write_csv()


def test_compare_outputs_ignores_only_the_timing_column():
    committed = pl.DataFrame({"day": ["d"], "rows": [5], "seconds": [12.5], "code_commit": ["c"]})
    same_but_slower = committed.with_columns(pl.lit(99.0).alias("seconds"))
    assert rmp.compare_outputs({"checks.csv": _csv(committed)}, {"checks.csv": same_but_slower}) == {"checks.csv": []}
    other_rows = same_but_slower.with_columns(pl.lit(6).alias("rows"))
    found = rmp.compare_outputs({"checks.csv": _csv(committed)}, {"checks.csv": other_rows})["checks.csv"]
    assert len(found) == 1 and "rows" in found[0]


def test_compare_outputs_forgives_float_formatting_but_not_values():
    committed = pl.DataFrame({"run": ["a"], "mean": [0.1 + 0.2]})
    assert rmp.compare_outputs({"summary.csv": _csv(committed)},
                               {"summary.csv": pl.DataFrame({"run": ["a"], "mean": [0.3]})}) == {"summary.csv": []}
    assert rmp.compare_outputs({"summary.csv": _csv(committed)},
                               {"summary.csv": pl.DataFrame({"run": ["a"], "mean": [0.31]})})["summary.csv"]


def test_committed_stamp_must_be_one_pair():
    a = _csv(pl.DataFrame({"x": [1], "code_commit": ["c1"], "manifest_sha256": ["m"]}))
    b = _csv(pl.DataFrame({"x": [2], "code_commit": ["c2"], "manifest_sha256": ["m"]}))
    assert rmp.committed_stamp({"one.csv": a, "two.csv": a}) == ("c1", "m")
    with pytest.raises(rmp.Refusal, match="2 \\(code_commit, manifest_sha256\\) pairs"):
        rmp.committed_stamp({"one.csv": a, "two.csv": b})


def test_verify_wants_every_output_committed_as_it_stands(repo):
    repo.commit(rmp.ARCHIVE_MANIFEST, json.dumps(manifest_doc(FILES)))
    with pytest.raises(rmp.Refusal, match="not committed; there is nothing to verify"):
        rmp.check_before_verify(repo.path)
    for name in rmp.OUTPUTS:
        repo.commit(f"{rmp.OUT_DIR}/{name}", "x\n")
    manifest, committed = rmp.check_before_verify(repo.path)
    assert manifest["days"] == [DAY] and set(committed) == set(rmp.OUTPUTS)
    (repo.path / rmp.OUT_DIR / "daily.csv").write_text("edited\n")
    with pytest.raises(rmp.Refusal, match="differs from its committed version"):
        rmp.check_before_verify(repo.path)
