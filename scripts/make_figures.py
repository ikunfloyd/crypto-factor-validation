#!/usr/bin/env python3
"""
The four figures of the factor-research report (docs/factor-research-v2.md).

    fig1_cost.png       1-minute absolute returns of DOGE against the 10 bp round trip (§1)
    fig2_tstats.png     the 60 primary cells' t-statistics, in and out of sample,
                        against Holm's thresholds (the README figure)
    fig3_corr.png       Spearman correlation of the ten factors, in sample, 15m and 1H (§8)
    fig4_is_vs_oos.png  in-sample against out-of-sample net mean per trade, per cell (§5)

A figure in a pre-registered report is a claim like any number in it, so it
must be re-derivable by anyone from what the repository holds. This script
therefore reads only

  * the committed results: results/insample_v2.csv and the primary set's
    results/oos/<set>.csv;
  * the 1-minute candles of the in-sample year, the primary and the
    replication set, and (for fig3) the in-sample spot leg and funding, all
    through research.pipeline.load_set_data and only after verify_inputs
    has matched every file against Appendix A's or its committed manifest's
    SHA-256. Without --insample-from, a figure cannot come from a different
    download than the results did.

Appendix A's in-sample candles are not in the repository, since OKX's API
Agreement (§9.4) does not allow its market data to be redistributed, and a
new download of that year matches them value for value but not byte for
byte. With --insample-from DIR, a reader's own download of the in-sample
year by fetch_v2.py (DIR/<instId>/<file>) is read in place of the in-sample
files, as scripts/verify_results.py reads it: each supplement file only
over the span its manifest records, and nothing is drawn unless every file
is there, the candles of both legs cover the in-sample year's warm-up and
window, and the funding holds as many settlements over its span as the
run's file did. The script names the files whose bytes differ from those
the results were computed from; their figures are checked by value. Figures
drawn from such a download that are byte-identical to the committed ones
show that the figures follow from OKX's public data.

The factor values behind fig3 come from research.pipeline.prepare_frames,
the code path of the in-sample run itself: same bars, same warm-up, same
factor code. The test's parameters (set names, windows, the cost, alpha
and the upper bound's z) are read from Appendix A; the code restates none
of them.

Three definitions that are easy to get wrong:

  * **In-sample net mean.** insample_v2.csv's mean_s_fwd_bps is the
    *undirected* mean of s·fwd, whose sign set the cell's direction d
    (§6.5). The in-sample mean net return in the traded direction is
    therefore d·mean_s_fwd_bps − cost, and its t divides that by se_bps
    (the SE does not depend on d, §6.7). Dropping d would turn every short
    cell's in-sample edge into a loss. Because d is that sign, the in-sample
    net mean can never fall below −cost.
  * **The test year's interval.** For the primary set the pre-registration
    reports the one-sided 95% upper bound ub95 = μ̂ + 1.645·SE (§9); the
    two-sided 1.960 interval belongs to doge_fwd. fig4 draws each
    cell's committed ub95_net_bps, after checking it against μ̂ + z·SE with
    Appendix A's z.
  * **The §1 windows.** The table in §1 and fig1 use each set's evaluation
    window: the two test years (23 September 20:00 UTC to 23 September
    20:00 UTC) and the in-sample run's own window (2025-12-23 00:00 to
    2026-09-23 20:00 UTC), which fixed the directions and the frozen SEs.
    The in-sample warm-up before it is left out.

It writes the four PNG files into --out and nothing else: imported modules
compile into a temporary bytecode cache removed at exit (the same startup
as scripts/run_oos.py), and so would any Numba kernel declared with an
on-disk cache. The output is deterministic: defaults are reset (no
matplotlibrc is read), the font is the DejaVu Sans that ships with
matplotlib, nothing is random, and the PNG metadata matplotlib would add
(its version, as 'Software') is dropped. Byte-identical output across
machines also needs the same rasteriser and encoder: the committed figures
were drawn by matplotlib 3.9.2 from PyPI, whose wheel bundles FreeType
2.6.1, and encoded by Pillow 10.4.0. A conda-forge matplotlib links a newer
FreeType and draws text with different pixels. The script prints the three
versions it ran with.

Usage:
    python scripts/make_figures.py --out docs/figures
    python scripts/make_figures.py --out /tmp/figures --insample-from data/history/insample_fresh
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import shutil
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock


def _repo_root() -> Path:
    # The first ancestor holding the research package: the checkout's root,
    # from scripts/ and from wherever else in the checkout a copy of this file
    # is run.
    for p in Path(__file__).resolve().parents:
        if (p / "research" / "spec.py").is_file():
            return p
    raise SystemExit(f"{Path(__file__).name}: no research/ package above {Path(__file__).parent}")


REPO_ROOT = _repo_root()
sys.path[:] = [str(REPO_ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]
if __name__ == "__main__":
    # Nothing but the figures may be written: bytecode of this checkout's
    # modules and any Numba disk cache go to a temporary directory instead of
    # __pycache__ folders inside the repository. NUMBA_CACHE_DIR must be set
    # before numba is first imported (by research.factors, below).
    _tmp = tempfile.mkdtemp(prefix=f"{Path(__file__).stem}-cache-")
    atexit.register(shutil.rmtree, _tmp, True)
    sys.pycache_prefix = _tmp
    os.environ["NUMBA_CACHE_DIR"] = _tmp

import matplotlib

matplotlib.use("Agg")
import numpy as np
import PIL
import polars as pl
from matplotlib import ft2font, patheffects
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from research import pipeline
from research import stats as S
from research.bars import bar_ms
from research.freeze import INSAMPLE_COLUMNS
from research.io import sha256_file
from research.pipeline import (
    INSAMPLE_RESULTS, OOS_COLUMNS, ROLE_IS, ROLE_PRIMARY, ROLE_REPLICATION, StudyParams,
    Window, load_set_data, oos_results_path, prepare_frames, set_with_role, verify_inputs,
    window_of,
)
from research.spec import get_spec

MINUTE_MS = 60_000
BPS = 1e4
FIGURE_NAMES = ("fig1_cost.png", "fig2_tstats.png", "fig3_corr.png", "fig4_is_vs_oos.png")

# ---------------------------------------------------------------- style

# Ink and chrome, light surface only: a PNG cannot follow a reader's theme,
# so it carries its own opaque white background and is legible on GitHub's
# dark theme as a white card.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f3f2ee"
# Three categorical hues, validated as a set for colour-vision deficiency
# (worst pair under deutan simulation ΔE_OKLab 9.2) on white. The third sits
# below 3:1 contrast, so every series that uses it is also labelled in text.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Holding horizon in fig4: ordinal, so one hue from light to dark, and a hue
# the year colours above do not use (validated as an ordinal ramp on white:
# monotone lightness, adjacent ΔL >= 0.06, light end 2.8:1).
HORIZON_RAMP = ("#9d8fe8", "#6a58d6", "#3a2a96")
# Diverging pair for correlations: blue (negative) and red (positive) arms
# of equal lightness steps around a neutral grey, so zero reads as nothing.
DIVERGING = ("#104281", "#5598e7", "#f0efec", "#e34948", "#8c1d1d")

DPI = 200            # 9 in -> 1800 px: sharp at GitHub's ~900 px column on high-density screens
WIDTH_IN = 9.0
# DejaVu Sans averages about 14 characters per inch at 10 pt; wrapping scales
# that inversely with the point size.
CHARS_PER_IN_10PT = 13.5
SUBTITLE_PT, SUBTITLE_SPACING = 10.0, 1.45
FOOTER_PT, FOOTER_SPACING = 9.0, 1.4
# Height below an axes taken by its tick labels and x label, inches.
XLABEL_IN = 0.55

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SURFACE,
    "axes.unicode_minus": True,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "path.simplify": False,
}


def _apply_style() -> None:
    """Matplotlib's built-in defaults plus STYLE, ignoring any matplotlibrc."""
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(STYLE)


def _minus(text: str) -> str:
    return text.replace("-", "−")


def _num(x: float, fmt: str) -> str:
    """x formatted with a typographic minus, and no minus on a value that rounds to zero."""
    s = format(x, fmt)
    if float(s.replace("%", "")) == 0:
        s = format(0.0, fmt)
    return _minus(s)


def _date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _datetime(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _count(n: int, noun: str) -> str:
    """'no cell', '1 cell', '3 cells'."""
    return f"no {noun}" if n == 0 else f"{n} {noun}" + ("" if n == 1 else "s")


def _bar_words(bar: str) -> str:
    """'15m' -> '15-minute', '1H' -> '1-hour'."""
    minutes = bar_ms(bar) // MINUTE_MS
    return f"{minutes // 60}-hour" if minutes % 60 == 0 else f"{minutes}-minute"


def _season(lo_ms: int, hi_ms: int) -> str:
    """'2023–24' for a window from 2023-09-23 to 2024-09-23."""
    a = datetime.fromtimestamp(lo_ms / 1000, tz=timezone.utc).year
    b = datetime.fromtimestamp(hi_ms / 1000, tz=timezone.utc).year
    return f"{a}–{b % 100:02d}"


def _new_figure(height_in: float) -> Figure:
    # A bare Figure (no pyplot): no global figure registry, no GUI backend.
    return Figure(figsize=(WIDTH_IN, height_in), dpi=DPI)


def _place(fig: Figure, left: float, right: float, top: float, bottom: float) -> Axes:
    """An axes with the given margins in inches."""
    w, h = fig.get_size_inches()
    return fig.add_axes((left / w, bottom / h, 1 - (left + right) / w, 1 - (top + bottom) / h))


def _wrap(text: str, left_in: float, pt: float) -> str:
    return textwrap.fill(text, int((WIDTH_IN - left_in - 0.35) * CHARS_PER_IN_10PT * 10.0 / pt))


def _text_height_in(text: str, left_in: float, pt: float, spacing: float) -> float:
    lines = _wrap(text, left_in, pt).count("\n") + 1
    return (1 + (lines - 1) * spacing) * pt / 72.0


def header_height_in(subtitle: str, left_in: float) -> float:
    """Inches from the top of the figure to the bottom of the subtitle _header draws."""
    return 0.58 + _text_height_in(subtitle, left_in, SUBTITLE_PT, SUBTITLE_SPACING)


def footer_height_in(text: str, left_in: float) -> float:
    """Inches from the bottom of the figure to the top of the footer _footer draws."""
    return 0.16 + _text_height_in(text, left_in, FOOTER_PT, FOOTER_SPACING)


def _header(fig: Figure, title: str, subtitle: str, left_in: float) -> None:
    w, h = fig.get_size_inches()
    x = left_in / w
    fig.text(x, 1 - 0.22 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(x, 1 - 0.58 / h, _wrap(subtitle, left_in, SUBTITLE_PT), ha="left", va="top",
             fontsize=SUBTITLE_PT, color=INK_2, linespacing=SUBTITLE_SPACING)


def _footer(fig: Figure, text: str, left_in: float) -> None:
    # Secondary ink, not the muted grey: the footer carries definitions a
    # reader needs (sources, what an interval is), at 7.9:1 contrast.
    w, h = fig.get_size_inches()
    fig.text(left_in / w, 0.16 / h, _wrap(text, left_in, FOOTER_PT), ha="left", va="bottom",
             fontsize=FOOTER_PT, color=INK_2, linespacing=FOOTER_SPACING)


def _halo(width: float = 3.0) -> List[patheffects.AbstractPathEffect]:
    """A white outline that keeps text legible where it crosses a line."""
    return [patheffects.withStroke(linewidth=width, foreground=SURFACE)]


def _hgrid(ax: Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: Figure, path: Path) -> None:
    # metadata: matplotlib otherwise writes its version string as 'Software'.
    fig.savefig(path, dpi=DPI, format="png", metadata={"Software": None})


# ---------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Cell:
    bar: str
    horizon: int
    factor: str
    direction: int
    is_net_bps: float     # d·mean_s_fwd − cost, in sample
    is_t: float
    oos_net_bps: float    # primary set
    oos_t: float
    oos_ub95_bps: float   # one-sided 95% upper bound, as committed

    @property
    def label(self) -> str:
        return f"{self.factor}, {self.bar} bars, {_held(self.horizon)}"


def _held(horizon: int) -> str:
    """'held 1 bar', 'held 5 bars': the figures' one name for the horizon h."""
    return f"held {horizon} bar" + ("" if horizon == 1 else "s")


def _read_csv(path: Path, columns: Sequence[str]) -> pl.DataFrame:
    df = pl.read_csv(path)
    if tuple(df.columns) != tuple(columns):
        raise SystemExit(f"{path}: columns {df.columns} are not the committed layout {list(columns)}")
    return df


def read_cells(params: StudyParams, primary: str, ub95_z: float) -> List[Cell]:
    """
    The 60 primary cells from the committed results files, in the study's order.

    Refuses a file that does not hold exactly the study's cells, whose
    directions differ between the in-sample file and the primary results
    (the primary run traded the in-sample directions, so a mismatch means
    the files are not the ones the study produced), or whose ub95_net_bps is
    not net_bps + ub95_z·se_bps (then fig4 would draw some other interval
    than the one the pre-registration reports).
    """
    cost = params.cost_bps
    ins = _read_csv(REPO_ROOT / INSAMPLE_RESULTS, INSAMPLE_COLUMNS)
    oos = _read_csv(REPO_ROOT / oos_results_path(primary), OOS_COLUMNS)
    keys = ["bar", "horizon", "factor"]
    for name, df in (("in-sample", ins), (primary, oos)):
        got = [tuple(r) for r in df.select(keys).iter_rows()]
        if sorted(got) != sorted(params.cells()) or len(got) != len(set(got)):
            raise SystemExit(f"the {name} results do not hold exactly the study's {len(params.cells())} cells")
    i_rows = {(r["bar"], r["horizon"], r["factor"]): r for r in ins.iter_rows(named=True)}
    o_rows = {(r["bar"], r["horizon"], r["factor"]): r for r in oos.iter_rows(named=True)}
    cells: List[Cell] = []
    for key in params.cells():
        i, o = i_rows[key], o_rows[key]
        d = int(i["direction"])
        if d != int(o["direction"]):
            raise SystemExit(f"cell {key}: direction {d} in sample but {o['direction']} in {primary}")
        is_net = d * float(i["mean_s_fwd_bps"]) - cost
        net, se, ub = float(o["net_bps"]), float(o["se_bps"]), float(o["ub95_net_bps"])
        if not math.isclose(ub, net + ub95_z * se, rel_tol=0.0, abs_tol=1e-6):
            raise SystemExit(f"cell {key}: ub95_net_bps {ub} is not net_bps + {ub95_z}·se_bps")
        cells.append(Cell(
            bar=key[0], horizon=key[1], factor=key[2], direction=d,
            is_net_bps=is_net, is_t=is_net / float(i["se_bps"]),
            oos_net_bps=net, oos_t=float(o["t_hh"]), oos_ub95_bps=ub,
        ))
    return cells


def require_verified(set_names: Sequence[str]) -> None:
    for name in set_names:
        problems = verify_inputs(name)
        if problems:
            raise SystemExit(f"inputs of {name} do not match their committed hashes:\n  " + "\n  ".join(problems))


def insample_file(f: Any, root: Path, spec: Optional[Dict[str, Any]] = None) -> bool:
    """Whether ``f``, a pipeline.InputFile, is Appendix A's in-sample candles or a file of the supplement."""
    root = Path(root)
    return (f.path.parent == root / (spec or get_spec())["insample_dir"]
            or f.path.parent.parent == root / pipeline.IS_SUPP_DIR)


def supplement_records(root: Path) -> Dict[Path, Dict[str, Any]]:
    """The in-sample supplement's manifest by file: what the run read of each (SHA-256, rows, first and last)."""
    series = json.loads((Path(root) / pipeline.IS_SUPP_MANIFEST).read_text(encoding="utf-8"))["series"]
    return {Path(root) / e["path"]: e for e in series}


def spanned(f: Any, record: Optional[Dict[str, Any]]) -> Any:
    """``f`` read only over the span its manifest ``record`` says the run's file covered (all of it without one)."""
    if record is None:
        return f
    lo, hi = int(record["first_ts"]), int(record["last_ts"]) + 1
    return replace(f, lo_ms=lo if f.lo_ms is None else max(f.lo_ms, lo),
                   hi_ms=hi if f.hi_ms is None else min(f.hi_ms, hi))


@contextmanager
def reading_fresh(fresh: Optional[Path]) -> Any:
    """
    research.pipeline as the results had it when ``fresh`` is None. Otherwise
    every in-sample file is read from ``fresh`` (fetch_v2's
    <out-dir>/<instId>/<file>) instead, a supplement file only over the span
    its manifest records, through the pipeline's own loader.
    """
    if fresh is None:
        yield
        return
    files = pipeline.input_files

    def input_files(set_name, root=pipeline.REPO_ROOT, *, spec=None, is_end_ms=None):
        out, records = [], None
        for f in files(set_name, root, spec=spec, is_end_ms=is_end_ms):
            if insample_file(f, root, spec):
                records = supplement_records(root) if records is None else records
                f = spanned(replace(f, path=fresh / f.path.parent.name / f.path.name), records.get(f.path))
            out.append(f)
        return out

    with mock.patch.object(pipeline, "input_files", input_files):
        yield


def check_fresh(fresh: Path, is_name: str, window: Window) -> List[str]:
    """
    Stop unless ``fresh`` holds every in-sample file the in-sample year
    reads: the candles of both legs covering its warm-up and window, and the
    funding holding as many settlements, over the span the supplement's
    manifest records, as the run's file did. Returns the files whose bytes
    differ from those the results were computed from: Appendix A's SHA-256
    for the candles, the supplement manifest's for the rest.
    """
    root, sp = pipeline.REPO_ROOT, get_spec()
    try:
        records = supplement_records(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"--insample-from: {pipeline.IS_SUPP_MANIFEST.as_posix()} cannot be read: {exc}")
    known = {p: str(e["sha256"]) for p, e in records.items()}
    known.update({root / sp["insample_dir"] / name: str(digest) for name, digest in sp["insample_sha256"].items()})
    other: List[str] = []
    for f in pipeline.input_files(is_name, root):
        if not insample_file(f, root, sp):
            continue
        src = fresh / f.path.parent.name / f.path.name
        if not src.is_file():
            raise SystemExit(f"--insample-from: {src} does not exist")
        try:
            ts = pl.read_parquet(src, columns=["timestamp"])["timestamp"].cast(pl.Int64)
        except Exception as exc:  # not a file fetch_v2 writes, whatever polars calls the failure
            raise SystemExit(f"--insample-from: {src} cannot be read: {exc}")
        if f.kind == "funding":
            record = records.get(f.path)
            if record is None:
                raise SystemExit(f"--insample-from: {src} has no counterpart in the supplement's manifest")
            lo, hi, rows = (int(record[k]) for k in ("first_ts", "last_ts", "rows"))
            held = int(((ts >= lo) & (ts <= hi)).sum())
            if held != rows:
                raise SystemExit(f"--insample-from: {src} holds {held} settlements from {_datetime(lo)} to "
                                 f"{_datetime(hi)} UTC, where the run's file held {rows} (OKX's REST funding "
                                 "history reaches back about three months; fetch_v2.py --allow-test-data takes "
                                 "the months before that from OKX's archive)")
        elif not (ts.len() and ts.min() <= window.warmup_start_ms and ts.max() >= window.end_ms - MINUTE_MS):
            raise SystemExit(f"--insample-from: {src} does not cover the in-sample year's warm-up and window, "
                             f"{_datetime(window.warmup_start_ms)} to {_datetime(window.end_ms)} UTC")
        if sha256_file(src) != known.get(f.path):
            other.append(f.path.relative_to(root).as_posix())
    return other


def minute_returns(candles: pl.DataFrame, lo_ms: int, hi_ms: int) -> Tuple[np.ndarray, int]:
    """
    C_t / C_(t−1) − 1 in bp for consecutive minutes with both candles in [lo, hi),
    and the number of gaps (neighbouring rows more than a minute apart), whose
    returns are left out rather than spread over the gap.
    """
    d = candles.filter((pl.col("timestamp") >= lo_ms) & (pl.col("timestamp") < hi_ms)).sort("timestamp")
    ts = d["timestamp"].to_numpy()
    close = d["close"].to_numpy()
    step = np.diff(ts)
    if np.any(step <= 0):
        raise SystemExit("minute candles are not strictly increasing in time")
    ok = step == MINUTE_MS
    return (close[1:][ok] / close[:-1][ok] - 1.0) * BPS, int((~ok).sum())


# ---------------------------------------------------------------- figure 1

@dataclass(frozen=True)
class Year:
    label: str          # '2023–24'
    role: str           # what the year is in the study
    lo_ms: int
    hi_ms: int
    abs_bps: np.ndarray
    gaps: int
    std_bps: float      # of the signed returns


def year_of(role: str, candles: pl.DataFrame, lo_ms: int, hi_ms: int) -> Year:
    r, gaps = minute_returns(candles, lo_ms, hi_ms)
    return Year(_season(lo_ms, hi_ms), role, lo_ms, hi_ms, np.abs(r), gaps, float(np.std(r, ddof=1)))


def print_cost_table(years: Sequence[Year], cost: float) -> None:
    """The §1 table: one column per evaluation window."""
    cols = list(years)
    head = ["1-minute returns (bp)"] + [f"{y.label} {y.role}" for y in cols]
    rows = [
        ["Standard deviation"] + [f"{y.std_bps:.1f}" for y in cols],
        ["Median |r|"] + [f"{np.median(y.abs_bps):.1f}" for y in cols],
        ["90th percentile of |r|"] + [f"{np.quantile(y.abs_bps, 0.9):.1f}" for y in cols],
        [f"Share of minutes with |r| > {cost:g} bp"] + [f"{100 * np.mean(y.abs_bps > cost):.1f}%" for y in cols],
        ["Returns (gaps skipped)"] + [f"{y.abs_bps.size} ({y.gaps})" for y in cols],
        ["Window, UTC [start, end)"] + [f"{_datetime(y.lo_ms)} -> {_datetime(y.hi_ms)}" for y in cols],
    ]
    widths = [max(len(r[i]) for r in [head] + rows) for i in range(len(head))]
    print("\n§1 table: close-to-close returns of consecutive 1-minute candles")
    for r in [head] + rows:
        print("  " + "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))))


def _or_list(items: Sequence[Any]) -> str:
    """'1, 5 or 15'."""
    s = [str(x) for x in items]
    return s[0] if len(s) == 1 else ", ".join(s[:-1]) + f" or {s[-1]}"


def fig_cost(years: Sequence[Year], colors: Sequence[str], params: StudyParams, path: Path) -> None:
    """
    Share of minutes whose |return| exceeds x, one curve per year, against the cost.

    Each of ``years`` is a set's evaluation window: the two test years and the
    in-sample period that fixed the directions (nine months, after its warm-up).
    The legend carries each window's dates.
    """
    cost = params.cost_bps
    shares = [100 * float(np.mean(y.abs_bps > cost)) for y in years]
    lo, hi = min(shares), max(shares)
    left = 0.85
    subtitle = (
        f"Share of minutes whose absolute close-to-close return exceeds x, OKX DOGE-USDT perpetual. "
        f"A trader who caught the whole move of every minute, in the right direction, would cover a "
        f"{cost:g} bp round trip in only {lo:.1f}–{hi:.1f}% of minutes. So the study trades "
        f"{' and '.join(_bar_words(b) for b in params.bars)} bars, held {_or_list(params.horizons)} bars."
    )
    footer = (
        f"Each line is a set's evaluation window (dates in UTC): the two test years, and the "
        f"in-sample period that fixed each trade direction, without the warm-up before it. Simple "
        f"returns between consecutive 1-minute candles, in basis points (1 bp = 0.01%)."
    )
    top = header_height_in(subtitle, left) + 0.22
    bottom = footer_height_in(footer, left) + XLABEL_IN
    fig = _new_figure(top + 3.25 + bottom)
    ax = _place(fig, left=left, right=0.3, top=top, bottom=bottom)

    x = np.linspace(0.0, 40.0, 1601)
    ax.axvspan(0, cost, color=BAND, zorder=0, linewidth=0)
    for y, c, share in zip(years, colors, shares):
        srt = np.sort(y.abs_bps)
        surv = 1.0 - np.searchsorted(srt, x, side="right") / srt.size
        ax.plot(x, 100 * surv, color=c, linewidth=2.0, solid_capstyle="round", zorder=3,
                label=f"{share:.1f}%   {y.role}, {_date(y.lo_ms)} → {_date(y.hi_ms)}")
    ax.axvline(cost, color=INK, linewidth=1.2, zorder=4)
    ax.text(cost + 0.4, 98, f"{cost:g} bp: one taker round trip", color=INK, fontsize=9, va="top", ha="left")
    ax.text(cost / 2, 3, "a whole minute's move\nis smaller than the fee", color=INK_2, fontsize=8.5,
            ha="center", va="bottom", linespacing=1.3)
    # The curves are too steep and close at the crossing for labels there:
    # the values sit in the legend, the dots mark where they are read.
    for s, c in zip(shares, colors):
        ax.plot([cost], [s], marker="o", markersize=6.5, color=c, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), handlelength=1.6, handletextpad=0.6,
              labelspacing=0.6, title=f"Share of minutes with |return| > {cost:g} bp",
              title_fontsize=9, alignment="left")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Absolute 1-minute close-to-close return, x (basis points)")
    ax.set_ylabel("Share of minutes whose |return| exceeds x")
    _hgrid(ax)
    _header(fig, "Most 1-minute moves of DOGE are smaller than the cost of one trade", subtitle, left)
    _footer(fig, footer, left)
    _save(fig, path)


# ---------------------------------------------------------------- figure 2

def holm_thresholds(m: int, alpha: float) -> np.ndarray:
    """t that the k-th largest of m one-sided statistics must reach: Φ⁻¹(1 − α/(m − k + 1))."""
    return np.array([S.norm_ppf(1.0 - alpha / (m - k + 1)) for k in range(1, m + 1)])


def holm_rejections(t_desc: np.ndarray, thr: np.ndarray) -> int:
    """How many of the statistics, sorted largest first, Holm's step-down rejects: it stops at the first miss."""
    n = 0
    while n < len(t_desc) and t_desc[n] >= thr[n]:
        n += 1
    return n


def _break_between(values: np.ndarray) -> Tuple[float, float]:
    """The widest empty interval between neighbouring values: where an axis break hides nothing."""
    v = np.sort(values)
    i = int(np.argmax(np.diff(v)))
    return float(v[i]), float(v[i + 1])


def fig_tstats(cells: Sequence[Cell], thr: np.ndarray, alpha: float, params: StudyParams,
               primary_span: str, is_span: str, primary_before_is: bool, path: Path) -> None:
    """
    Sorted t-statistics against Holm's step-down thresholds, on a broken axis.

    The ten 15m cells held 1 bar sit near t = −20 in both sets, far below
    the rest. On one linear axis they would squeeze the region that matters,
    between the largest t's and the threshold, into a sliver; so the axis is
    split at the widest gap between any two t's, where no point is hidden,
    and the lower panel is drawn at a smaller scale. A note at the break
    marks says both, with the scale ratio computed from the two axes.
    """
    m = len(cells)
    cost = params.cost_bps
    k = np.arange(1, m + 1)
    t_oos = np.array(sorted((c.oos_t for c in cells), reverse=True))
    t_is = np.array(sorted((c.is_t for c in cells), reverse=True))
    top = max(cells, key=lambda c: c.oos_t)
    top_is = max(cells, key=lambda c: c.is_t)
    rejected = holm_rejections(t_oos, thr)
    gap_lo, gap_hi = _break_between(np.concatenate([t_oos, t_is]))
    low = [c for c in cells if c.oos_t <= gap_lo or c.is_t <= gap_lo]

    left = 0.85
    if rejected == 0:
        title = (f"No cell reached its threshold: the largest t was {_num(top.oos_t, '.2f')}, "
                 f"short of {_num(thr[0], '.2f')}")
    else:
        title = f"{_count(rejected, 'cell').capitalize()} reached {'its' if rejected == 1 else 'their'} Holm threshold"
    subtitle = (
        f"Each dot is one of {m} cells ({len(params.factor_names)} factors × {len(params.bars)} bar sizes × "
        f"{len(params.horizons)} holding horizons), sorted by the t-statistic of its mean return per trade "
        f"after a {cost:g} bp round-trip taker cost. A cell would have counted as profitable only if its "
        f"test-year t reached the Holm threshold (one-sided, {alpha:.0%} family-wise error over the {m} "
        f"cells). The in-sample t's were not tested; they are shown for comparison."
    )
    footer = (
        f"DOGE-USDT perpetual (OKX). Standard errors: Hansen–Hodrick, for overlapping holding periods. "
        f"In-sample t = (mean return in the traded direction − {cost:g} bp) / SE; it is biased upward, "
        f"since each direction is the sign of the in-sample mean. "
        f"Sources: {INSAMPLE_RESULTS.as_posix()}, {primary_results_rel()}."
    )
    top_in = header_height_in(subtitle, left) + 0.33
    up_h, gap_in, low_h = 2.83, 0.12, 1.2
    bottom = footer_height_in(footer, left) + XLABEL_IN
    fig = _new_figure(top_in + up_h + gap_in + low_h + bottom)
    w_in, h_in = fig.get_size_inches()
    width = (w_in - left - 0.3) / w_in
    ax = fig.add_axes((left / w_in, (bottom + low_h + gap_in) / h_in, width, up_h / h_in))
    axl = fig.add_axes((left / w_in, bottom / h_in, width, low_h / h_in), sharex=ax)

    y_top = math.ceil(thr[0]) + 2.0
    y_up_lo = math.floor(gap_hi) - 1.0
    y_lo_hi = math.ceil(gap_lo) + 1.5
    y_lo_lo = math.floor(min(t_oos.min(), t_is.min())) - 1.5
    scale = (up_h / (y_top - y_up_lo)) / (low_h / (y_lo_hi - y_lo_lo))
    for a in (ax, axl):
        a.scatter(k, t_is, s=30, color=AQUA, edgecolors=SURFACE, linewidths=0.8, zorder=5)
        a.scatter(k, t_oos, s=30, color=BLUE, edgecolors=SURFACE, linewidths=0.8, zorder=6)
        _hgrid(a)
    ax.axhline(0, color=AXIS, linewidth=0.9, zorder=1)
    ax.step(k, thr, where="mid", color=INK, linewidth=1.6, zorder=4)
    ax.fill_between(k, thr, y_top, step="mid", color=BAND, linewidth=0, zorder=0)
    ax.set_ylim(y_up_lo, y_top)
    axl.set_ylim(y_lo_lo, y_lo_hi)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    axl.yaxis.set_major_locator(MultipleLocator(10))
    for a in (ax, axl):
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, ".0f")))
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # Break marks on the left spine, the same physical size in both panels.
    for a, y in ((ax, 0.0), (axl, 1.0)):
        a.plot([0], [y], transform=a.transAxes, marker=[(-1, -0.6), (1, 0.6)], markersize=9,
               linestyle="none", color=INK_2, markeredgecolor=INK_2, markeredgewidth=1.0, clip_on=False)
    axl.text(2.0, y_lo_hi - 0.06 * (y_lo_hi - y_lo_lo),
             # Rounded inward, so that no cell lies inside the stated interval.
             f"Axis broken: no cell has a t between {_num(math.ceil(gap_lo * 10) / 10, '.1f')} and "
             f"{_num(math.floor(gap_hi * 10) / 10, '.1f')}. "
             f"Below the break the scale is {scale:.1f}× smaller.",
             color=INK, fontsize=8.5, ha="left", va="top")

    ax.text(2.0, thr[0] + 0.35, f"Holm threshold: the t a cell needs at its rank, from "
            f"{_num(thr[0], '.2f')} (rank 1) to {_num(thr[-1], '.2f')} (rank {m})",
            color=INK, fontsize=8.5, ha="left", va="bottom")
    ax.annotate(
        f"Largest test-year t: {_num(top.oos_t, '.2f')}  ({top.label})",
        xy=(1, top.oos_t), xytext=(3.2, top.oos_t), color=INK, fontsize=8.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.7, shrinkA=0, shrinkB=4),
    )
    # An elbow straight down from the dot, then right: the rank-1 column is
    # empty below it, so the leader crosses no other point.
    ax.annotate(
        f"Largest in-sample t: {_num(top_is.is_t, '.2f')}", xy=(1, top_is.is_t), xytext=(3.2, -3.0),
        color=INK, fontsize=8.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.7, shrinkA=2, shrinkB=4,
                        relpos=(0.0, 0.5), connectionstyle="angle,angleA=180,angleB=90,rad=0"),
    )
    low_oos = {c for c in cells if c.oos_t <= gap_lo}
    low_is = {c for c in cells if c.is_t <= gap_lo}
    groups = sorted({(c.bar, c.horizon) for c in low})
    what = ", ".join(f"{b} bars {_held(h)}" for b, h in groups)
    where = "in both sets" if low_oos == low_is else "in either set"
    gross = max(max(abs(c.oos_net_bps + cost), abs(c.is_net_bps + cost)) for c in low)
    axl.text(2.0, y_lo_lo + 0.36 * (y_lo_hi - y_lo_lo),
             f"The {len(low)} cells at {what}, {where}: before costs their mean is within "
             f"{math.ceil(gross * 10) / 10:.1f} bp of zero,\nso each cell loses about the {cost:g} bp cost per trade, with a "
             f"small standard error.",
             color=INK_2, fontsize=8.5, ha="left", va="center", linespacing=1.35)

    when = "before" if primary_before_is else "after"
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=7, color=BLUE, markeredgecolor=SURFACE,
               label=f"Held-out test year, {primary_span} ({when} the in-sample data)"),
        Line2D([], [], linestyle="none", marker="o", markersize=7, color=AQUA, markeredgecolor=SURFACE,
               label=f"In sample, where each direction was chosen: {is_span}"),
    ]
    ax.legend(handles=handles, loc="lower left", handletextpad=0.3, borderaxespad=0.5)
    ax.set_xlim(0, m + 1)
    axl.xaxis.set_major_locator(MultipleLocator(10))
    axl.set_xlabel(f"Rank of the cell's t-statistic within its set (1 = largest; {m} cells)")
    fig.text(0.22 / w_in, (bottom + (low_h + gap_in + up_h) / 2) / h_in,
             "t-statistic of mean net return per trade", rotation=90, ha="left", va="center",
             color=INK_2, fontsize=10)
    _header(fig, title, subtitle, left)
    _footer(fig, footer, left)
    _save(fig, path)


# ---------------------------------------------------------------- figure 3

def spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """Spearman's rho with average ranks over the pairs where both are finite, and that count."""
    ok = np.isfinite(x) & np.isfinite(y)
    rx = pl.Series(x[ok]).rank("average").to_numpy().astype(np.float64)
    ry = pl.Series(y[ok]).rank("average").to_numpy().astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def factor_correlations(data: Dict[str, Any], bar: str, window: Window,
                        params: StudyParams) -> Tuple[np.ndarray, int]:
    """
    Spearman matrix of the ten factors over the bars whose start lies in the
    evaluation window, computed as the in-sample run computes them (warm-up
    included, so no factor starts cold at the window's first bar). Also the
    number of bars in the window.
    """
    frames = prepare_frames(data, bar, window, params)
    if len(frames) != 1:
        raise SystemExit(f"expected one instrument in sample, got {sorted(frames)}")
    fr = next(iter(frames.values()))
    inside = (fr.ts >= window.start_ms) & (fr.ts < window.end_ms)
    names = params.factor_names
    n = len(names)
    rho = np.eye(n)
    min_pairs = int(inside.sum())
    for i in range(n):
        for j in range(i):
            r, cnt = spearman(fr.factors[names[i]][inside], fr.factors[names[j]][inside])
            rho[i, j] = rho[j, i] = r
            min_pairs = min(min_pairs, cnt)
    if min_pairs < int(inside.sum()):
        print(f"  {bar}: some factor pairs have only {min_pairs} of {int(inside.sum())} bars with both values")
    return rho, int(inside.sum())


def print_matrix(names: Sequence[str], rho: np.ndarray, title: str) -> None:
    print(f"\n{title}")
    w = max(len(s) for s in names)
    print("  " + " " * w + "".join(f"{i + 1:>7d}" for i in range(len(names))))
    for i, s in enumerate(names):
        print(f"  {s:<{w}}" + "".join(f"{rho[i, j]:>7.2f}" for j in range(len(names))) + f"   ({i + 1})")


def fig_corr(names: Sequence[str], mats: Sequence[Tuple[str, np.ndarray, int]], window: Window,
             path: Path) -> None:
    """
    Two lower-triangle heatmaps, one per (bar, matrix, number of bars) in
    ``mats``; the colour key sits in the right panel's empty upper triangle.
    """
    n = len(names)
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING, N=256)
    cmap.set_bad(SURFACE)
    margin, labels, gap, right = 0.3, 1.45, 0.2, 0.3
    size = (WIDTH_IN - margin - labels - gap - right) / 2
    # The pair the report compares with v1's 0.90 (§8).
    i_r, i_b = names.index("rsi_14"), names.index("bollinger_pctb")
    pair = ", ".join(f"{_num(rho[i_r, i_b], '.2f')} on {_bar_words(b)}" for b, rho, _ in mats)
    subtitle = (
        f"Rank correlation between the factors' values on each bar of the in-sample period "
        f"({_date(window.start_ms)} → {_date(window.end_ms)}), OKX DOGE-USDT perpetual. "
        f"rsi_14 and bollinger_pctb: {pair} bars. Near 0, two factors' rankings of the bars show no "
        f"monotone relation; near ±1, they carry the same information."
    )
    footer = ("Factor values from the in-sample run's code (research/pipeline.py), warm-up from "
              f"{_date(window.warmup_start_ms)}. Each pair is shown once.")
    # Below the panels: the rotated factor names (about 1.45 in), then the footer.
    bottom = footer_height_in(footer, margin) + 1.45
    fig = _new_figure(header_height_in(subtitle, margin) + 0.6 + size + bottom)
    w_in, h_in = fig.get_size_inches()
    x0 = margin + labels
    axes = [fig.add_axes(((x0 + k * (size + gap)) / w_in, bottom / h_in, size / w_in, size / h_in))
            for k in range(2)]
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    im = None
    for k, (ax, (bar, rho, n_bars)) in enumerate(zip(axes, mats)):
        shown = np.where(lower, rho, np.nan)[1:, :-1]
        im = ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest",
                       aspect="equal")
        for i in range(n - 1):
            for j in range(n - 1):
                v = shown[i, j]
                if np.isfinite(v):
                    ax.text(j, i, _num(v, ".2f"), ha="center", va="center", fontsize=7.8,
                            color=SURFACE if abs(v) >= 0.6 else INK)
        ax.set_xticks(range(n - 1))
        ax.set_xticklabels(names[:-1], rotation=50, ha="right", rotation_mode="anchor", fontsize=8.5)
        ax.set_yticks(range(n - 1))
        ax.set_yticklabels(names[1:] if k == 0 else [], fontsize=8.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{_bar_words(bar)} bars ({n_bars:,} bars)", fontsize=10, color=INK, loc="left", pad=6)
    cax = axes[1].inset_axes((0.42, 0.86, 0.55, 0.035))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_xticklabels([_minus(t) for t in ("-1", "-0.5", "0", "0.5", "1")], fontsize=8)
    cb.ax.tick_params(length=2, color=AXIS)
    cb.outline.set_visible(False)
    cb.ax.set_title("Spearman rank correlation", fontsize=8.5, color=INK_2, pad=4)

    _header(fig, "Some of the ten factors measure nearly the same thing", subtitle, margin)
    _footer(fig, footer, margin)
    _save(fig, path)


# ---------------------------------------------------------------- figure 4

def settled_groups(cells: Sequence[Cell], params: StudyParams) -> List[Tuple[str, int]]:
    """The (bar, horizon) groups all of whose cells have a test-year 95% upper bound below zero."""
    return [(b, h) for b in params.bars for h in params.horizons
            if all(c.oos_ub95_bps < 0 for c in cells if (c.bar, c.horizon) == (b, h))]


def _describe_groups(groups: Sequence[Tuple[str, int]], params: StudyParams) -> str:
    """'all 20 held 1 bar and the 10 at 15m held 5 bars'."""
    nf = len(params.factor_names)
    parts: List[str] = []
    for h in params.horizons:
        bars = [b for b in params.bars if (b, h) in groups]
        if len(bars) == len(params.bars):
            parts.append(f"all {nf * len(bars)} {_held(h)}")
        else:
            parts.extend(f"the {nf} at {b} {_held(h)}" for b in bars)
    return " and ".join(parts)


def fig_is_vs_oos(cells: Sequence[Cell], params: StudyParams, ub95_z: float, primary_span: str,
                  is_span: str, path: Path) -> None:
    """
    Test-year against in-sample net mean, with each cell's test-year one-sided 95% upper bound.

    The two axes cannot share a scale (in-sample means span about 20 bp, the
    test year's means and bounds over 100), so y = x is drawn as a reference
    and looks shallow. Shape is the bar size. The holding horizon is ordered,
    so it is one hue from light (shortest) to dark, a hue the other figures
    do not use for their years. Groups are drawn longest holding time first,
    so the tight short-hold groups are not buried under the wider ones.
    """
    cost = params.cost_bps
    if min(c.is_net_bps for c in cells) < -cost - 1e-9:
        raise SystemExit("an in-sample net mean lies below −cost: a direction is not the sign of its mean")
    hcolors = dict(zip(params.horizons, HORIZON_RAMP))
    markers = dict(zip(params.bars, ("o", "s")))
    msize = dict(zip(params.bars, (36, 32)))
    top = max(cells, key=lambda c: c.oos_t)
    both = sum(c.is_net_bps > 0 and c.oos_net_bps > 0 for c in cells)
    n_settled = sum(c.oos_ub95_bps < 0 for c in cells)
    rho, _ = spearman(np.array([c.is_net_bps for c in cells]), np.array([c.oos_net_bps for c in cells]))
    groups = settled_groups(cells, params)
    cl = [c for c in cells if (c.bar, c.horizon) in groups]

    left = 1.0
    subtitle = (
        f"Each point is one of {len(cells)} cells: its mean return per trade after a {cost:g} bp "
        f"round-trip cost in the period that fixed its trade direction (x) and in the test year (y). "
        f"{n_settled} cells have a test-year 95% upper bound below zero: charged a round trip on every "
        f"signal bar, as the test charges them, they do not pay at this cost. "
        f"Rank correlation of x and y across the cells: {_num(rho, '.2f')}."
    )
    footer = (
        f"DOGE-USDT perpetual (OKX). x cannot fall below {_num(-cost, 'g')} bp: each direction is the sign "
        f"of the cell's in-sample mean before costs. Vertical lines run from the test-year mean up to its "
        f"one-sided 95% upper bound (mean + {ub95_z:g} Hansen–Hodrick standard errors, as pre-registered). "
        f"Colour: holding horizon; shape: bar size. Sources: {INSAMPLE_RESULTS.as_posix()}, "
        f"{primary_results_rel()}."
    )
    top_in = header_height_in(subtitle, left) + 0.35
    bottom = footer_height_in(footer, left) + XLABEL_IN
    fig = _new_figure(top_in + 4.1 + bottom)
    ax = _place(fig, left=left, right=0.3, top=top_in, bottom=bottom)

    order = sorted(((b, h) for b in params.bars for h in params.horizons),
                   key=lambda g: bar_ms(g[0]) * g[1], reverse=True)
    for i, (bar, h) in enumerate(order):
        sel = [c for c in cells if c.bar == bar and c.horizon == h]
        xs_g = [c.is_net_bps for c in sel]
        for c in sel:
            ax.plot([c.is_net_bps] * 2, [c.oos_net_bps, c.oos_ub95_bps], color=hcolors[h], alpha=0.7,
                    linewidth=1.1, solid_capstyle="butt", zorder=2)
        ax.scatter(xs_g, [c.oos_ub95_bps for c in sel], marker="_", s=30, linewidths=1.1,
                   color=hcolors[h], alpha=0.7, zorder=2)
        ax.scatter(xs_g, [c.oos_net_bps for c in sel], s=msize[bar], marker=markers[bar], color=hcolors[h],
                   edgecolors=SURFACE, linewidths=0.8, zorder=4 + 0.01 * i)
    xs = [c.is_net_bps for c in cells]
    lo_y = min(c.oos_net_bps for c in cells)
    hi_y = max(c.oos_ub95_bps for c in cells)
    x0, x1 = math.floor(min(xs)) - 1.0, math.ceil(max(xs)) + 1.0
    # Headroom above the highest bound holds the largest-t label clear of every line.
    y0, y1 = math.floor(lo_y / 10) * 10 - 5, math.ceil(hi_y / 10) * 10 + 15
    ax.axhline(0, color=INK_2, linewidth=0.9, zorder=1)
    ax.axvline(0, color=INK_2, linewidth=0.9, zorder=1)
    d = np.array([max(x0, y0), min(x1, y1)])
    ax.plot(d, d, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.text(d[1] - 0.15, d[1] + 2.0, "y = x", color=INK_2, fontsize=8.5, ha="right", va="bottom")
    ax.text(x1 - 0.15, y1 - 2.5, f"net positive in both periods:\n{_count(both, 'cell')}",
            color=INK_2, fontsize=8.5, ha="right", va="top", linespacing=1.3)

    ax.scatter([top.is_net_bps], [top.oos_net_bps], s=170, facecolors="none", edgecolors=INK,
               linewidths=1.1, zorder=5)
    ax.annotate(
        f"Largest test-year t ({_num(top.oos_t, '.2f')}):\n{top.label}",
        xy=(top.is_net_bps, top.oos_net_bps), xytext=(top.is_net_bps + 2.3, y1 - 8),
        color=INK, fontsize=8.5, ha="left", va="center", linespacing=1.35, path_effects=_halo(),
        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.7, shrinkA=2, shrinkB=7),
    )
    if cl:
        # The note's bounds hold for every one of these cells in both periods, so
        # each is rounded outward: |gross| up, the smallest loss down, the largest up.
        gross = max(max(abs(c.is_net_bps + cost), abs(c.oos_net_bps + cost)) for c in cl)
        nets = [x for c in cl for x in (c.is_net_bps, c.oos_net_bps)]
        if max(nets) >= 0:
            raise SystemExit("fig4: a settled cell is not net negative in both periods; the note would be false")
        least, most = math.floor(-max(nets) * 10) / 10, math.ceil(-min(nets) * 10) / 10
        cx = float(np.mean([c.is_net_bps for c in cl]))
        cy = float(np.mean([c.oos_net_bps for c in cl]))
        ax.annotate(
            f"{len(cl)} cells pile up here: {_describe_groups(groups, params)}.\n"
            f"Before costs each earns within {math.ceil(gross * 10) / 10:.1f} bp of zero in both periods,\n"
            f"so each cell loses {least:.1f} to {most:.1f} bp per trade on average; every upper bound is below zero.",
            xy=(cx, cy), xytext=(cx + 1.6, (y0 + cy) / 2), color=INK_2, fontsize=8.5, ha="left",
            va="center", linespacing=1.3, path_effects=_halo(),
            arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.7, shrinkA=1, shrinkB=6,
                            relpos=(0.0, 1.0)),
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, ".0f")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, ".0f")))
    ax.set_xlabel(f"In sample ({is_span}): mean net return per trade (bp)")
    ax.set_ylabel(f"Test year ({primary_span}):\nmean net return per trade (bp)")
    _hgrid(ax)
    hl = [Line2D([], [], linestyle="none", marker="o", markersize=7, color=hcolors[h], markeredgecolor=SURFACE,
                 label=_held(h)) for h in params.horizons]
    bl = [Line2D([], [], linestyle="none", marker=markers[b], markersize=7, color=INK_2,
                 markeredgecolor=SURFACE, label=f"{b} bars") for b in params.bars]
    el = [Line2D([], [], color=INK_2, alpha=0.7, linewidth=1.1, label="up to the 95% upper bound")]
    ax.legend(handles=hl + bl + el, loc="lower right", ncol=2, columnspacing=1.4, handletextpad=0.3,
              borderaxespad=0.5)
    _header(
        fig,
        f"{_count(both, 'cell').capitalize()} {'was' if both < 2 else 'were'} net positive in both periods",
        subtitle, left,
    )
    _footer(fig, footer, left)
    _save(fig, path)


# ---------------------------------------------------------------- main

def primary_results_rel() -> str:
    return oos_results_path(set_with_role(ROLE_PRIMARY)).as_posix()


def renderer_versions() -> str:
    """What decides the PNG bytes besides this script: the rasteriser's and the encoder's versions."""
    return (f"matplotlib {matplotlib.__version__} (FreeType {ft2font.__freetype_version__}), "
            f"Pillow {PIL.__version__}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw the report's four figures from committed results.")
    p.add_argument("--out", required=True, type=Path, help="directory for the PNG files (created if absent)")
    p.add_argument("--insample-from", type=Path, metavar="DIR", dest="insample_from",
                   help="a download of the in-sample year by fetch_v2.py, read in place of the in-sample files")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    spec = get_spec()
    params = StudyParams.from_spec(spec)
    is_name, primary, replication = (set_with_role(r, spec) for r in (ROLE_IS, ROLE_PRIMARY, ROLE_REPLICATION))
    w_is, w_p, w_r = (window_of(s, spec) for s in (is_name, primary, replication))
    fresh = args.insample_from.resolve() if args.insample_from is not None else None
    require_verified([primary, replication] if fresh is not None else [is_name, primary, replication])
    if fresh is not None:
        other = check_fresh(fresh, is_name, w_is)
        print(f"in-sample files read from {fresh}" + (
            f"; these have other bytes than the results were computed from, so the figures check them by value: "
            f"{', '.join(other)}" if other else ", each with the bytes the results were computed from"))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    _apply_style()
    print(f"rendering with {renderer_versions()}")
    cost = params.cost_bps
    ub95_z = float(spec["intervals"]["ub95"]["z"])

    cells = read_cells(params, primary, ub95_z)
    is_eval = f"{_date(w_is.start_ms)} → {_date(w_is.end_ms)}"
    primary_span = f"{_date(w_p.start_ms)} → {_date(w_p.end_ms)}"

    with reading_fresh(fresh):
        data_is = load_set_data(is_name)
    [inst] = data_is["instruments"]
    years = [
        year_of("primary test year", load_set_data(primary)["perp"][inst], w_p.start_ms, w_p.end_ms),
        year_of("replication year", load_set_data(replication)["perp"][inst], w_r.start_ms, w_r.end_ms),
        # The evaluation window that fixed the directions and the frozen SEs (§4.1), as each
        # test year is its set's evaluation window; the warm-up before it is left out.
        year_of("in-sample period", data_is["perp"][inst], w_is.start_ms, w_is.end_ms),
    ]
    print_cost_table(years, cost)
    fig_cost(years, (BLUE, ORANGE, AQUA), params, out / FIGURE_NAMES[0])

    alpha = float(spec["test"]["alpha"])
    thr = holm_thresholds(len(cells), alpha)
    if abs(thr[0] - float(spec["first_holm_threshold_t"])) > 5e-4:
        raise SystemExit(f"first Holm threshold {thr[0]:.4f} differs from Appendix A's "
                         f"{spec['first_holm_threshold_t']}")
    top_is = max(cells, key=lambda c: c.is_t)
    top_oos = max(cells, key=lambda c: c.oos_t)
    print(f"\nHolm thresholds: rank 1 needs t >= {thr[0]:.3f}, rank {len(cells)} needs t >= {thr[-1]:.3f}")
    print(f"largest in-sample net t {top_is.is_t:.3f} ({top_is.label}); "
          f"{sum(c.is_net_bps + cost > cost for c in cells)} cells with in-sample gross mean > {cost:g} bp")
    print(f"largest out-of-sample t {top_oos.oos_t:.3f} ({top_oos.label}); Holm rejects "
          f"{holm_rejections(np.array(sorted((c.oos_t for c in cells), reverse=True)), thr)}")
    fig_tstats(cells, thr, alpha, params, primary_span, is_eval, w_p.end_ms <= w_is.start_ms,
               out / FIGURE_NAMES[1])

    names = list(params.factor_names)
    mats: List[Tuple[str, np.ndarray, int]] = []
    for bar in params.bars:
        rho, n_bars = factor_correlations(data_is, bar, w_is, params)
        print_matrix(names, rho, f"Spearman correlation of the factors, in sample, {bar} bars "
                                 f"({n_bars} bars, {is_eval})")
        mats.append((bar, rho, n_bars))
    i_r, i_b = names.index("rsi_14"), names.index("bollinger_pctb")
    print("rsi_14 vs bollinger_pctb: " + ", ".join(f"{b} {m[1][i_r, i_b]:.3f}" for b, m in zip(params.bars, mats)))
    fig_corr(names, mats, w_is, out / FIGURE_NAMES[2])

    groups = settled_groups(cells, params)
    print(f"\ntest-year ub95 < 0: {sum(c.oos_ub95_bps < 0 for c in cells)} cells; groups with every "
          f"cell's ub95 < 0: {', '.join(f'{b} h={h}' for b, h in groups)}")
    fig_is_vs_oos(cells, params, ub95_z, primary_span, is_eval, out / FIGURE_NAMES[3])
    print("\nwrote " + ", ".join(str(out / f) for f in FIGURE_NAMES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
