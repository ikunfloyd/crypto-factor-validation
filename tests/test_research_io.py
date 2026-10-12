"""
research.io: write-once artefacts and manifest records.

Write-once is the fix for v1's overwritten results file (pre-registration
§2 item 6, §10); the manifest is what the one analysis run checks its inputs
against (§4.2, §10 step 5). Synthetic data only.
"""
import hashlib
import json

import numpy as np
import polars as pl
import pytest

from research.io import (
    find_gaps,
    manifest_entry,
    sha256_file,
    write_once_bytes,
    write_once_csv,
    write_once_json,
    write_once_parquet,
)


class TestSha256:
    def test_known_vector(self, tmp_path):
        p = tmp_path / "abc.txt"
        p.write_bytes(b"abc")
        assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    def test_streams_files_larger_than_one_chunk(self, tmp_path):
        data = np.random.default_rng(0).bytes(3 * (1 << 20) + 17)
        p = tmp_path / "big.bin"
        p.write_bytes(data)
        assert sha256_file(p) == hashlib.sha256(data).hexdigest()


class TestWriteOnce:
    def test_csv_creates_parents_and_refuses_second_write(self, tmp_path):
        p = tmp_path / "a" / "b" / "out.csv"
        write_once_csv(pl.DataFrame({"x": [1, 2]}), p)
        before = p.read_bytes()
        with pytest.raises(FileExistsError):
            write_once_csv(pl.DataFrame({"x": [9]}), p)
        assert p.read_bytes() == before

    def test_csv_writes_nan_as_blank(self, tmp_path):
        # The results files leave undefined values blank (§6.5, §10).
        p = tmp_path / "r.csv"
        write_once_csv(pl.DataFrame({"f": ["a", "b"], "se_bps": [1.5, float("nan")]}), p)
        assert p.read_text().splitlines() == ["f,se_bps", "a,1.5", "b,"]

    def test_csv_can_keep_nan(self, tmp_path):
        p = tmp_path / "r.csv"
        write_once_csv(pl.DataFrame({"v": [float("nan")]}), p, nan_as_blank=False)
        assert p.read_text().splitlines() == ["v", "NaN"]

    def test_json_round_trips_and_refuses_overwrite(self, tmp_path):
        p = tmp_path / "m.json"
        obj = {"rows": np.int64(3), "x": np.float64(0.5), "ok": np.bool_(True), "a": np.arange(2)}
        write_once_json(obj, p)
        assert json.loads(p.read_text()) == {"rows": 3, "x": 0.5, "ok": True, "a": [0, 1]}
        with pytest.raises(FileExistsError):
            write_once_json({}, p)
        assert json.loads(p.read_text())["rows"] == 3

    def test_json_refuses_nan_and_leaves_no_file(self, tmp_path):
        p = tmp_path / "bad.json"
        with pytest.raises(ValueError):
            write_once_json({"x": float("nan")}, p)
        assert not p.exists()

    def test_existing_file_is_never_touched(self, tmp_path):
        p = tmp_path / "keep.bin"
        p.write_bytes(b"original")
        for writer in (
            lambda: write_once_bytes(b"new", p),
            lambda: write_once_csv(pl.DataFrame({"x": [1]}), p),
            lambda: write_once_json({"x": 1}, p),
            lambda: write_once_parquet(pl.DataFrame({"x": [1]}), p),
        ):
            with pytest.raises(FileExistsError):
                writer()
        assert p.read_bytes() == b"original"

    def test_parquet_round_trip(self, tmp_path):
        df = pl.DataFrame({"timestamp": [1, 2], "close": [0.1, 0.2]})
        p = write_once_parquet(df, tmp_path / "x" / "c.parquet")
        assert pl.read_parquet(p).equals(df)


class TestManifest:
    STEP = 60_000

    def test_gaps_are_half_open_first_missing_to_next_present(self):
        s = self.STEP
        ts = np.array([0, s, 2 * s, 5 * s, 6 * s, 10 * s], dtype=np.int64)
        # missing 3s,4s and 7s..9s
        assert find_gaps(ts, s) == [[3 * s, 5 * s], [7 * s, 10 * s]]

    def test_no_gaps_and_short_series(self):
        s = self.STEP
        assert find_gaps(np.arange(5, dtype=np.int64) * s, s) == []
        assert find_gaps(np.array([7], dtype=np.int64), s) == []
        assert find_gaps(np.array([], dtype=np.int64), s) == []

    def test_off_grid_overshoot_counts_as_gap(self):
        s = self.STEP
        assert find_gaps(np.array([0, s + 1], dtype=np.int64), s) == [[s, s + 1]]

    def test_unsorted_or_duplicate_timestamps_rejected(self):
        with pytest.raises(ValueError):
            find_gaps(np.array([0, 120_000, 60_000]), self.STEP)
        with pytest.raises(ValueError):
            find_gaps(np.array([0, 0]), self.STEP)

    def test_entry_fields(self, tmp_path):
        s = self.STEP
        p = tmp_path / "c.parquet"
        p.write_bytes(b"not really parquet")
        ts = np.array([0, s, 4 * s], dtype=np.int64)
        src = {"1970-01-02": "missing", "1970-01-01": "rest_history_candles"}
        e = manifest_entry(p, src, ts, s)
        assert e["rows"] == 3
        assert e["first_ts"] == 0 and e["last_ts"] == 4 * s
        assert e["gaps"] == [[2 * s, 4 * s]]
        assert e["step_ms"] == s
        assert e["sha256"] == sha256_file(p)
        assert list(e["source_by_day"]) == ["1970-01-01", "1970-01-02"]
        assert e["path"].endswith("c.parquet")
        # Plain Python types only: the record must survive JSON unchanged.
        assert json.loads(json.dumps(e, allow_nan=False)) == e
        assert all(type(v) is int for v in (e["rows"], e["first_ts"], e["last_ts"]))

    def test_entry_for_empty_series(self, tmp_path):
        p = tmp_path / "empty.parquet"
        p.write_bytes(b"")
        e = manifest_entry(p, {}, np.array([], dtype=np.int64), self.STEP)
        assert e["rows"] == 0 and e["first_ts"] is None and e["last_ts"] is None
        assert e["gaps"] == []

    def test_path_inside_repo_is_relative(self):
        from research.spec import REPO_ROOT

        e = manifest_entry(REPO_ROOT / "requirements.txt", {}, np.array([1]), self.STEP)
        assert e["path"] == "requirements.txt"
