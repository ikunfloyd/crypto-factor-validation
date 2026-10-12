"""
Write-once outputs and data manifests for the out-of-sample study.

v1's results table was lost because a later partial run overwrote it
(pre-registration §2, item 6). Every artefact here is therefore created
exclusively: if the path exists, the write fails instead of replacing it,
and a rerun has to choose a new path (§10, step 6: ``rerun-N/``).

The manifest records what a downloaded series contains, so that the one
analysis run can refuse inputs that differ from what was committed (§4.2,
§10 step 5). Gaps are listed, never filled.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import polars as pl

from research.spec import REPO_ROOT

PathLike = Union[str, Path]

_CHUNK = 1 << 20


def sha256_file(path: PathLike) -> str:
    """Hex SHA-256 of a file's bytes, streamed so large parquet files fit."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def write_once_bytes(data: bytes, path: PathLike) -> Path:
    """
    Create ``path`` holding ``data``; raise FileExistsError if it exists.

    Mode 'x' makes existence check and creation one atomic step, so two runs
    racing for the same path cannot both succeed. If the write itself fails
    (disk full), the half-written file is ours and is removed, so it cannot
    later pass for a complete artefact.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "xb") as fh:
        try:
            fh.write(data)
            fh.flush()
        except BaseException:
            fh.close()
            p.unlink(missing_ok=True)
            raise
    return p


def write_once_csv(df: pl.DataFrame, path: PathLike, *, nan_as_blank: bool = True) -> Path:
    """
    Write ``df`` as CSV to a new file; never overwrite.

    The pre-registration's results files leave undefined values blank (§6.5,
    §10). Polars writes a float NaN as the text "NaN", so by default NaN is
    turned into a null first and comes out as an empty field.
    """
    if nan_as_blank:
        floats = [c for c, t in zip(df.columns, df.dtypes) if t in (pl.Float32, pl.Float64)]
        if floats:
            df = df.with_columns([pl.col(c).fill_nan(None) for c in floats])
    # Serialise before creating the file, so a bad frame leaves nothing behind.
    return write_once_bytes(df.write_csv().encode("utf-8"), path)


def write_once_parquet(df: pl.DataFrame, path: PathLike) -> Path:
    """Write ``df`` as parquet to a new file; never overwrite."""
    buf = io.BytesIO()
    df.write_parquet(buf)
    return write_once_bytes(buf.getvalue(), path)


def _json_default(obj: Any) -> Any:
    # numpy scalars sneak in from array arithmetic; json cannot serialise them.
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_once_json(obj: Any, path: PathLike) -> Path:
    """
    Write ``obj`` as indented JSON to a new file; never overwrite.

    NaN and infinity are refused (allow_nan=False): they are not JSON, and a
    committed manifest that some parsers reject is worse than an error now.
    """
    text = json.dumps(obj, indent=2, allow_nan=False, default=_json_default) + "\n"
    return write_once_bytes(text.encode("utf-8"), path)


def _display_path(path: PathLike) -> str:
    """Repo-relative POSIX path when possible, so a committed manifest is portable."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def find_gaps(ts: np.ndarray, step_ms: int) -> List[List[int]]:
    """
    Missing stretches of a regular series, as half-open [from_ts, to_ts).

    ``from_ts`` is the first missing timestamp (previous present + step) and
    ``to_ts`` the next present one, matching the half-open windows used
    throughout the pre-registration. A step that overshoots without landing
    on the grid still counts as a gap: something is missing either way.
    """
    if step_ms <= 0:
        raise ValueError("step_ms must be positive")
    t = np.asarray(ts, dtype=np.int64)
    if t.size < 2:
        return []
    d = np.diff(t)
    if (d <= 0).any():
        raise ValueError("timestamps must be strictly increasing")
    idx = np.flatnonzero(d > step_ms)
    return [[int(t[i] + step_ms), int(t[i + 1])] for i in idx]


def manifest_entry(
    path: PathLike,
    source_by_day: Dict[str, str],
    ts: np.ndarray,
    step_ms: int,
) -> Dict[str, Any]:
    """
    One series' manifest record (§4.2).

    Keys: path, rows, first_ts, last_ts, step_ms, gaps, sha256, source_by_day.
    ``ts`` must be the series' timestamps in ms, strictly increasing; ``gaps``
    are the interior holes from :func:`find_gaps`. ``step_ms`` is kept in the
    record so the gap list can be re-derived from the series. It does not
    set the funding staleness limit, which §4.2 fixes at 8 h
    (factors.funding_fresh). Every value is a plain Python type so the
    record round-trips through JSON unchanged.
    """
    t = np.asarray(ts, dtype=np.int64)
    gaps = find_gaps(t, step_ms)
    return {
        "path": _display_path(path),
        "rows": int(t.size),
        "first_ts": int(t[0]) if t.size else None,
        "last_ts": int(t[-1]) if t.size else None,
        "step_ms": int(step_ms),
        "gaps": gaps,
        "sha256": sha256_file(path),
        "source_by_day": {str(k): str(v) for k, v in sorted(source_by_day.items())},
    }


__all__ = [
    "sha256_file",
    "write_once_bytes",
    "write_once_csv",
    "write_once_parquet",
    "write_once_json",
    "find_gaps",
    "manifest_entry",
]
