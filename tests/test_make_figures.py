"""
scripts/make_figures.py without the candles: its helpers, the numbers it
writes into titles and notes, its determinism settings, each figure drawn
from small synthetic inputs, fig2 and fig4 also from the committed results
(results/ is in every checkout; data/ is not), and main() itself, with
verify_inputs, load_set_data and prepare_frames replaced by fakes that hand
it synthetic candles and factor values inside each set's window.

No test reads data/history, and none compares a PNG with the report's
figures: a PNG's bytes depend on the FreeType that rasterises its text,
which differs between platforms. What a figure says is checked through its
text instead. The drawing functions save through the module's _save, which
the ``drawn`` fixture replaces so that each Figure is kept rather than
written. The smoke tests write real files and check only that they are
PNGs of the page's width without the version metadata, and that the same
inputs drawn twice on one machine give the same bytes.

A reader's own download of the in-sample year (--insample-from) is checked on
small files in fetch_v2's layout: that it is read for the in-sample files
only, which of its files are named as differing from the run's bytes, that a
missing file or a short year stops the script, and that main() checks it, in
place of the in-sample hashes, before anything is drawn.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import PIL
import polars as pl
import pytest
from matplotlib import ft2font
from matplotlib.style.core import STYLE_BLACKLIST
from matplotlib.text import Text
from PIL import Image

from research.spec import REPO_ROOT, get_spec


def _load(name):
    found = importlib.util.spec_from_file_location(f"_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(found)
    # The script declares dataclasses under postponed annotations, which
    # Python 3.10's dataclasses module resolves through sys.modules.
    sys.modules[found.name] = mod
    found.loader.exec_module(mod)
    return mod


mf = _load("make_figures")

SPEC = get_spec()
PARAMS = mf.StudyParams.from_spec(SPEC)
COST = PARAMS.cost_bps
UB95_Z = float(SPEC["intervals"]["ub95"]["z"])
PRIMARY = mf.set_with_role(mf.ROLE_PRIMARY)
IS_SET, REPLICATION = mf.set_with_role(mf.ROLE_IS), mf.set_with_role(mf.ROLE_REPLICATION)
(INST,) = SPEC["sets"][IS_SET]["inst"]
NAMES = list(PARAMS.factor_names)
MINUTE = 60_000
FIG1, FIG2, FIG3, FIG4 = mf.FIGURE_NAMES


def utc_ms(*args):
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


# Windows like the study's, stated here so that the expected texts are literal.
Y2023, Y2024, Y2025 = utc_ms(2023, 9, 23, 20), utc_ms(2024, 9, 23, 20), utc_ms(2025, 9, 23, 20)
IS_WINDOW = mf.Window(utc_ms(2025, 9, 24), utc_ms(2025, 12, 23), utc_ms(2026, 9, 23, 20))
PRIMARY_SPAN, IS_SPAN = "2023-09-23 → 2024-09-23", "2025-12-23 → 2026-09-23"


@pytest.fixture(autouse=True)
def style():
    """The script's style in every test, and matplotlib's settings as they were afterwards."""
    with matplotlib.rc_context():
        mf._apply_style()
        yield


@pytest.fixture
def drawn(monkeypatch):
    """{file name: Figure} for each figure the drawing functions finish; nothing is written."""
    figs = {}
    monkeypatch.setattr(mf, "_save", lambda fig, path: figs.__setitem__(Path(path).name, fig))
    return figs


def texts(fig):
    """Every non-empty text of a figure (tick labels exist only once it has been drawn)."""
    return [t.get_text() for t in fig.findobj(Text) if t.get_text()]


def text_with(fig, phrase):
    """The one text of ``fig`` that contains ``phrase``."""
    (found,) = [t for t in texts(fig) if phrase in t]
    return found


def squash(s):
    """``s`` without whitespace: subtitles and footers are wrapped to the page, so line breaks are layout."""
    return "".join(s.split())


def page(fig):
    """(title, subtitle, footer): the last three figure-level texts, as _header and _footer add them."""
    title, subtitle, footer = (t.get_text() for t in fig.texts[-3:])
    return title, subtitle, footer


def minus(x, fmt):
    """``x`` formatted with a typographic minus, written out here rather than taken from the script."""
    return format(x, fmt).replace("-", "−")


# ---------------------------------------------------------------- synthetic inputs

def cells(changes=None):
    """
    The 60 cells in the study's order, shaped like the study's results: the
    ten at 15m held 1 bar have a gross mean within 0.45 bp of zero and an SE
    of 0.5 bp in both periods (t from -20.5 to -19.1, far below the rest);
    the other 50 have in-sample net means from -9.6 to +10 bp (SE 10) and
    lose 3.35 to 8.85 bp in the test year (SE 20). No cell is net positive
    in both periods, and only the 15m h=1 group has every test-year upper
    bound below zero. ``changes`` maps a cell key to the fields to replace.
    """
    out = []
    n = len(NAMES)
    for i, (bar, h, factor) in enumerate(PARAMS.cells()):
        k, g = i % n, i // n
        if (bar, h) == ("15m", 1):
            is_net, is_se, net, se = -COST + 0.05 * k, 0.5, -COST - 0.03 * k, 0.5
        else:
            is_net, is_se, net, se = -COST + 2.0 * k + 0.4 * g, 10.0, -3.1 - 0.5 * k - 0.25 * g, 20.0
        c = mf.Cell(bar=bar, horizon=h, factor=factor, direction=1 if k % 2 else -1,
                    is_net_bps=is_net, is_t=is_net / is_se, oos_net_bps=net, oos_t=net / se,
                    oos_ub95_bps=net + UB95_Z * se)
        out.append(dataclasses.replace(c, **(changes or {}).get((bar, h, factor), {})))
    return out


def year(role, lo_ms, hi_ms, pattern):
    """A Year whose consecutive 1-minute returns repeat ``pattern`` (bp) 50 times from ``lo_ms``."""
    r = np.tile(np.asarray(pattern, dtype=np.float64), 50) / 1e4
    close = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    ts = lo_ms + MINUTE * np.arange(close.size, dtype=np.int64)
    return mf.year_of(role, pl.DataFrame({"timestamp": ts, "close": close}), lo_ms, hi_ms)


def years():
    """Three years in which 50%, 25% and 12.5% of the minutes move more than 10 bp."""
    return [
        year("primary test year", Y2023, Y2024, [5, -12, 20, -3]),
        year("replication year", Y2024, Y2025, [5, -3, 2, -11]),
        year("in-sample period", IS_WINDOW.start_ms, IS_WINDOW.end_ms, [1, -2, 3, -4, 5, -6, 7, -12]),
    ]


def corr_matrix(pair, shift):
    """Symmetric with a unit diagonal; rsi_14 against bollinger_pctb at ``pair``, one pair at -0.004."""
    n = len(NAMES)
    i, j = np.indices((n, n))
    rho = ((3 * (i + j) + shift) % 19 - 9) / 10.0
    np.fill_diagonal(rho, 1.0)
    r, b, v = NAMES.index("rsi_14"), NAMES.index("bollinger_pctb"), NAMES.index("rolling_vol_ratio")
    rho[r, b] = rho[b, r] = pair
    rho[r, v] = rho[v, r] = -0.004
    return rho


def corr_mats():
    return [("15m", corr_matrix(0.904, 0), 26384), ("1H", corr_matrix(0.906, 5), 6596)]


def draw_fig2(cells_, path, before=True):
    thr = mf.holm_thresholds(len(cells_), PARAMS.alpha)
    mf.fig_tstats(cells_, thr, PARAMS.alpha, PARAMS, PRIMARY_SPAN, IS_SPAN, before, path)
    return thr


def draw_fig4(cells_, path):
    mf.fig_is_vs_oos(cells_, PARAMS, UB95_Z, PRIMARY_SPAN, IS_SPAN, path)


def png_chunks(path):
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, out = 8, []
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        out.append((kind, data[pos + 8:pos + 8 + n]))
        pos += 12 + n
    return out


def png_text_keys(path):
    """The keywords of a PNG's text chunks (tEXt, zTXt, iTXt)."""
    return [body.split(b"\0", 1)[0] for kind, body in png_chunks(path) if kind in (b"tEXt", b"zTXt", b"iTXt")]


def png_size(path):
    ((_, ihdr),) = [c for c in png_chunks(path) if c[0] == b"IHDR"]
    return struct.unpack(">II", ihdr[:8])


# ---------------------------------------------------------------- where the script runs from

# `python <script> --help`, run through runpy so that, once --help has exited,
# the launcher can print (as JSON, on the last line) where the script sent
# bytecode and Numba's cache: sys.pycache_prefix, NUMBA_CACHE_DIR, the cache
# folder numba read when it was imported, and the bytecode file of one of the
# checkout's modules, with whether it was written.
HELP_THEN_REPORT = """
import importlib.util, json, os, runpy, sys
script = sys.argv[1]
sys.path.insert(0, os.path.dirname(script))      # as `python <script>` puts its folder first
sys.argv = ["make_figures.py", "--help"]
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit:
    pyc = importlib.util.cache_from_source(sys.modules["research.pipeline"].__file__)
    print(json.dumps({"pycache_prefix": sys.pycache_prefix, "NUMBA_CACHE_DIR": os.environ["NUMBA_CACHE_DIR"],
                      "numba": sys.modules["numba"].config.CACHE_DIR, "pyc": pyc,
                      "pyc_written": os.path.isfile(pyc)}))
    raise
"""


class TestCheckout:
    def test_found_from_scripts(self):
        assert mf.REPO_ROOT == REPO_ROOT

    @pytest.mark.parametrize("where", ["scripts", "a/b/c"])
    def test_found_by_walking_up_from_the_file(self, tmp_path, monkeypatch, where):
        (tmp_path / "research").mkdir()
        (tmp_path / "research" / "spec.py").write_text("")
        monkeypatch.setattr(mf, "__file__", str(tmp_path / where / "make_figures.py"))
        assert mf._repo_root() == tmp_path.resolve()

    def test_no_research_package_above_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mf, "__file__", str(tmp_path / "scripts" / "make_figures.py"))
        with pytest.raises(SystemExit, match="no research/ package above"):
            mf._repo_root()

    def test_the_script_starts_from_scripts_and_removes_its_cache(self, tmp_path):
        # Another working directory and no PYTHONPATH: the script finds its
        # checkout from its own path. --help exits right after the imports.
        # Bytecode writing is on and TMPDIR is tmp_path: the checkout's modules
        # compile into one new folder there, which is also the cache folder
        # numba read when imported, and the folder is gone once the script exits.
        env = {k: v for k, v in os.environ.items()
               if k not in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
        env["TMPDIR"] = str(tmp_path)
        proc = subprocess.run([sys.executable, "-c", HELP_THEN_REPORT, str(REPO_ROOT / "scripts" / "make_figures.py")],
                              capture_output=True, text=True, env=env, cwd=tmp_path)
        assert proc.returncode == 0, proc.stderr
        assert "--out" in proc.stdout
        seen = json.loads(proc.stdout.splitlines()[-1])
        assert seen["pycache_prefix"] == seen["NUMBA_CACHE_DIR"] == seen["numba"], seen
        cache = Path(seen["pycache_prefix"])
        assert cache.parent.resolve() == tmp_path.resolve() and cache.name.startswith("make_figures-cache-")
        assert seen["pyc_written"] and Path(seen["pyc"]).is_relative_to(cache)
        assert not cache.exists() and list(tmp_path.glob("make_figures-cache-*")) == []

    def test_out_is_required(self):
        with pytest.raises(SystemExit) as exc:
            mf.parse_args([])
        assert exc.value.code == 2
        assert mf.parse_args(["--out", "docs/figures"]).out == Path("docs/figures")


# ---------------------------------------------------------------- words and numbers

class TestWords:
    def test_numbers_take_a_typographic_minus_but_zero_never_does(self):
        assert mf._num(-1.234, ".2f") == "−1.23"
        assert mf._num(1.234, ".2f") == "1.23"
        assert mf._num(-0.004, ".2f") == "0.00"          # rounds to zero: no "−0.00"
        assert mf._num(-0.0, ".1f") == "0.0"
        assert mf._num(-0.0004, ".1%") == "0.0%"
        assert mf._num(-0.25, ".0%") == "−25%"
        assert mf._num(-COST, "g") == "−10"
        assert mf._minus("-1 to -2") == "−1 to −2"

    def test_counts_horizons_bars_and_lists(self):
        assert [mf._count(n, "cell") for n in (0, 1, 3)] == ["no cell", "1 cell", "3 cells"]
        assert [mf._held(h) for h in (1, 5)] == ["held 1 bar", "held 5 bars"]
        assert [mf._bar_words(b) for b in ("15m", "1H", "4H", "90m")] == [
            "15-minute", "1-hour", "4-hour", "90-minute"]
        assert [mf._or_list(x) for x in ([1, 5, 15], [1, 5], [7])] == ["1, 5 or 15", "1 or 5", "7"]
        assert mf.Cell("1H", 5, "rsi_14", 1, 0.0, 0.0, 0.0, 0.0, 0.0).label == "rsi_14, 1H bars, held 5 bars"

    def test_dates_are_utc(self):
        late = utc_ms(2025, 12, 31, 23, 30)
        assert (mf._date(late), mf._datetime(late)) == ("2025-12-31", "2025-12-31 23:30")
        assert mf._season(Y2023, Y2024) == "2023–24"

    def test_the_renderer_versions_are_reported(self):
        assert mf.renderer_versions() == (f"matplotlib {matplotlib.__version__} "
                                          f"(FreeType {ft2font.__freetype_version__}), Pillow {PIL.__version__}")


# ---------------------------------------------------------------- the committed results, read

def results_rows(z=UB95_Z):
    """
    Rows of results/insample_v2.csv and of the primary set's results, in the
    study's order: cell i is short when i is even, its undirected in-sample
    mean has that sign (|mean| = 3 + 0.25·i bp, SE 1 + 0.1·i), and its
    test-year bound is net + z·SE.
    """
    ins, oos = [], []
    for i, (bar, h, factor) in enumerate(PARAMS.cells()):
        d = 1 if i % 2 else -1
        ins.append({"bar": bar, "horizon": h, "factor": factor, "direction": d, "flagged": "false",
                    "flag_reason": "", "n_active": 900, "mean_s_fwd_bps": d * (3.0 + 0.25 * i),
                    "se_bps": 1.0 + 0.1 * i, "se_method": "hh", "mde_bps": 4.0})
        net, se = -6.0 + 0.1 * i, 2.0 + 0.05 * i
        row = dict.fromkeys(mf.OOS_COLUMNS, "")
        row.update({"instrument": "DOGE-USDT-SWAP", "bar": bar, "horizon": h, "factor": factor, "direction": d,
                    "net_bps": net, "se_bps": se, "t_hh": net / se, "ub95_net_bps": net + z * se})
        oos.append(row)
    return ins, oos


def write_results(root, ins, oos, is_columns=mf.INSAMPLE_COLUMNS):
    for rel, columns, rows in ((mf.INSAMPLE_RESULTS, is_columns, ins),
                               (mf.oos_results_path(PRIMARY), mf.OOS_COLUMNS, oos)):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [",".join(columns)] + [",".join(str(r[c]) for c in columns) for r in rows]
        path.write_text("\n".join(lines) + "\n")


class TestReadCells:
    def test_the_in_sample_net_mean_is_taken_in_the_traded_direction(self, tmp_path, monkeypatch):
        ins, oos = results_rows()
        write_results(tmp_path, ins[::-1], oos[::-1])               # row order does not matter
        monkeypatch.setattr(mf, "REPO_ROOT", tmp_path)
        got = mf.read_cells(PARAMS, PRIMARY, UB95_Z)
        assert [(c.bar, c.horizon, c.factor) for c in got] == PARAMS.cells()
        # mean_s_fwd_bps is undirected and its sign chose d. Cell 0 is short
        # with a mean of -3.0 bp: +3.0 bp before costs, -7.0 after (not -13.0).
        short, long_ = got[0], got[1]
        assert (short.direction, short.is_net_bps, short.is_t) == (-1, -7.0, -7.0)
        assert (long_.direction, long_.is_net_bps) == (1, -6.75)
        assert long_.is_t == pytest.approx(-6.75 / 1.1)
        for c, r in zip(got, oos):
            assert (c.oos_net_bps, c.oos_t, c.oos_ub95_bps) == (r["net_bps"], r["t_hh"], r["ub95_net_bps"])

    @pytest.mark.parametrize("how,needle", [
        ("direction", f"direction -1 in sample but 1 in {PRIMARY}"),
        ("two_sided_bound", "is not net_bps + 1.645·se_bps"),
        ("missing_cell", "the in-sample results do not hold exactly the study's 60 cells"),
        ("repeated_cell", f"the {PRIMARY} results do not hold exactly the study's 60 cells"),
        ("column_order", "are not the committed layout"),
    ])
    def test_files_the_study_did_not_produce_are_refused(self, tmp_path, monkeypatch, how, needle):
        ins, oos = results_rows(z=1.96 if how == "two_sided_bound" else UB95_Z)
        columns = mf.INSAMPLE_COLUMNS
        if how == "direction":
            oos[0]["direction"] = 1                                 # cell 0 is short in sample
        elif how == "missing_cell":
            ins = ins[1:]
        elif how == "repeated_cell":
            oos[1] = dict(oos[0])
        elif how == "column_order":
            columns = columns[::-1]
        write_results(tmp_path, ins, oos, is_columns=columns)
        monkeypatch.setattr(mf, "REPO_ROOT", tmp_path)
        with pytest.raises(SystemExit, match=re.escape(needle)):
            mf.read_cells(PARAMS, PRIMARY, UB95_Z)

    def test_the_committed_results_pass_and_agree(self):
        # results/ is committed, so this runs in every checkout.
        got = {(c.bar, c.horizon, c.factor): c for c in mf.read_cells(PARAMS, PRIMARY, UB95_Z)}
        assert len(got) == len(PARAMS.cells())
        for r in pl.read_csv(REPO_ROOT / mf.INSAMPLE_RESULTS).iter_rows(named=True):
            c = got[(r["bar"], r["horizon"], r["factor"])]
            net = r["direction"] * r["mean_s_fwd_bps"] - COST
            assert c.is_net_bps == pytest.approx(net, abs=1e-12)
            assert c.is_t == pytest.approx(net / r["se_bps"], abs=1e-12)
        # d is the sign of the in-sample mean, so no in-sample net mean lies below -cost.
        assert min(c.is_net_bps for c in got.values()) >= -COST


# ---------------------------------------------------------------- 1-minute returns and fig1

def frame(minutes, closes, start=Y2023):
    return pl.DataFrame({"timestamp": [start + m * MINUTE for m in minutes], "close": closes})


class TestMinuteReturns:
    def test_returns_skip_gaps_and_keep_to_the_window(self):
        df = frame([0, 1, 2, 4, 5], [100.0, 101.0, 100.0, 102.0, 102.0])
        r, gaps = mf.minute_returns(df, Y2023, Y2023 + 10 * MINUTE)
        assert r == pytest.approx([100.0, (100 / 101 - 1) * 1e4, 0.0]) and gaps == 1
        # [lo, hi): minute 0 is left out by lo, minute 5 by hi.
        r, gaps = mf.minute_returns(df, Y2023 + MINUTE, Y2023 + 5 * MINUTE)
        assert r == pytest.approx([(100 / 101 - 1) * 1e4]) and gaps == 1
        r, gaps = mf.minute_returns(df.reverse(), Y2023, Y2023 + 10 * MINUTE)   # sorted first
        assert r == pytest.approx([100.0, (100 / 101 - 1) * 1e4, 0.0]) and gaps == 1

    def test_a_repeated_minute_is_refused(self):
        with pytest.raises(SystemExit, match="not strictly increasing"):
            mf.minute_returns(frame([0, 1, 1, 2], [1.0, 1.1, 1.2, 1.3]), Y2023, Y2024)

    def test_year_of(self):
        y = year("primary test year", Y2023, Y2024, [5, -12, 20, -3])
        signed = np.tile([5.0, -12.0, 20.0, -3.0], 50)
        assert (y.label, y.role, y.gaps, y.abs_bps.size) == ("2023–24", "primary test year", 0, 200)
        assert y.abs_bps == pytest.approx(np.abs(signed))
        assert y.std_bps == pytest.approx(np.std(signed, ddof=1))

    def test_cost_table(self, capsys):
        mf.print_cost_table(years(), COST)
        out = capsys.readouterr().out
        share = next(l for l in out.splitlines() if "Share of minutes with |r| > 10 bp" in l)
        assert share.split()[-3:] == ["50.0%", "25.0%", "12.5%"]
        assert "2023-09-23 20:00 -> 2024-09-23 20:00" in out
        counts = next(l for l in out.splitlines() if "Returns (gaps skipped)" in l)
        assert counts.split()[-6:] == ["200", "(0)", "200", "(0)", "400", "(0)"]


class TestFig1:
    def test_texts(self, drawn):
        mf.fig_cost(years(), (mf.BLUE, mf.ORANGE, mf.AQUA), PARAMS, Path(FIG1))
        fig = drawn[FIG1]
        title, subtitle, footer = page(fig)
        assert title == "Most 1-minute moves of DOGE are smaller than the cost of one trade"
        assert squash("would cover a 10 bp round trip in only 12.5–50.0% of minutes. So the study trades "
                      "15-minute and 1-hour bars, held 1, 5 or 15 bars.") in squash(subtitle)
        everything = squash("\n".join(texts(fig)))
        for label in ("50.0%   primary test year, 2023-09-23 → 2024-09-23",
                      "25.0%   replication year, 2024-09-23 → 2025-09-23",
                      "12.5%   in-sample period, 2025-12-23 → 2026-09-23",
                      "Share of minutes with |return| > 10 bp", "10 bp: one taker round trip"):
            assert squash(label) in everything

    def test_a_move_of_exactly_the_cost_does_not_exceed_it(self, drawn, capsys):
        # Of |r| = 10, 10, 12, 3 and 3 bp, only 12 exceeds the 10 bp cost: 20% of
        # the minutes, where counting |r| >= cost would give 60%.
        exact = mf.Year("2023–24", "primary test year", Y2023, Y2024, np.tile([COST, COST, 12.0, 3.0, 3.0], 40),
                        0, 7.0)
        ys = [exact] + years()[1:]
        mf.print_cost_table(ys, COST)
        share = next(l for l in capsys.readouterr().out.splitlines() if "Share of minutes" in l)
        assert share.split()[-3:] == ["20.0%", "25.0%", "12.5%"]
        mf.fig_cost(ys, (mf.BLUE, mf.ORANGE, mf.AQUA), PARAMS, Path(FIG1))
        fig = drawn[FIG1]
        assert "20.0%   primary test year, 2023-09-23 → 2024-09-23" in texts(fig)
        assert squash("in only 12.5–25.0% of minutes") in squash(page(fig)[1])
        # The curve is the share of minutes whose |return| exceeds x, so at x = cost it reads the same 20%.
        (ax,) = fig.axes
        (curve,) = [l for l in ax.get_lines() if l.get_label().startswith("20.0%")]
        x, y = curve.get_data()
        assert list(y[x == COST]) == pytest.approx([20.0])


# ---------------------------------------------------------------- Holm's step-down and fig2

class TestHolm:
    def test_thresholds(self):
        m, alpha = len(PARAMS.cells()), PARAMS.alpha
        thr = mf.holm_thresholds(m, alpha)
        nd = statistics.NormalDist()
        assert thr == pytest.approx([nd.inv_cdf(1 - alpha / (m - j)) for j in range(m)], abs=1e-9)
        assert abs(thr[0] - float(SPEC["first_holm_threshold_t"])) <= 5e-4      # Appendix A's rank-1 value
        assert thr[-1] == pytest.approx(nd.inv_cdf(1 - alpha))                   # the plain one-sided test
        assert np.all(np.diff(thr) < 0)

    @pytest.mark.parametrize("t,n", [
        ([3.5, 3.2, 1.0], 2),
        ([3.5, 1.0, 3.0], 1),            # stops at the first miss, whatever follows
        ([3.0, 2.5, 2.0], 3),            # reaching the threshold counts
        ([2.9, 2.6, 2.1], 0),
    ])
    def test_rejections_stop_at_the_first_miss(self, t, n):
        assert mf.holm_rejections(np.array(t), np.array([3.0, 2.5, 2.0])) == n

    def test_the_axis_breaks_at_the_widest_gap(self):
        assert mf._break_between(np.array([1.0, -20.0, -19.5, -3.0, -2.0])) == (-19.5, -3.0)

    def test_the_break_label_rounds_inward(self, drawn):
        # The primary year's neighbours of the gap: -16.97 and -7.55. Rounded to
        # the nearest they would read "between -17.0 and -7.5", which both lie in.
        lows = {("15m", 1, f): {"oos_t": -20.0, "is_t": -20.0} for f in NAMES}
        lows[("15m", 1, NAMES[0])] = {"oos_t": -16.97, "is_t": -20.0}
        lows[("1H", 15, NAMES[0])] = {"oos_t": -7.55}
        draw_fig2(cells(lows), Path(FIG2))
        assert text_with(drawn[FIG2], "Axis broken").startswith("Axis broken: no cell has a t between −16.9 and −7.6.")


class TestFig2:
    @pytest.mark.parametrize("top,title", [
        ([1.7], "No cell reached its threshold: the largest t was 1.70, short of 3.14"),
        ([3.5], "1 cell reached its Holm threshold"),
        ([3.6, 3.5, 3.4], "3 cells reached their Holm threshold"),
        ([3.6, 3.13, 3.0], "1 cell reached its Holm threshold"),     # 3.13 misses rank 2's 3.139
    ])
    def test_title_counts_holms_rejections(self, drawn, top, title):
        keys = [("1H", 5, f) for f in NAMES[:len(top)]]
        draw_fig2(cells({k: {"oos_t": t} for k, t in zip(keys, top)}), Path(FIG2))
        fig = drawn[FIG2]
        assert page(fig)[0] == title
        assert text_with(fig, "Largest test-year t") == (
            f"Largest test-year t: {top[0]:.2f}  ({NAMES[0]}, 1H bars, held 5 bars)")

    @pytest.mark.parametrize("period", ["in_sample", "test_year"])
    @pytest.mark.parametrize("gross,shown", [(0.7237, "0.8"), (0.6501, "0.7"), (1.2049, "1.3")])
    def test_the_low_cells_gross_mean_is_bounded_by_rounding_up(self, drawn, period, gross, shown):
        # "within x bp" must hold for every low cell in both periods, so the
        # largest |gross| is rounded up: 0.7237 is "within 0.8", never 0.7.
        field = "is_net_bps" if period == "in_sample" else "oos_net_bps"
        draw_fig2(cells({("15m", 1, NAMES[1]): {field: -COST + gross}}), Path(FIG2))
        assert text_with(drawn[FIG2], "before costs") == (
            f"The 10 cells at 15m bars held 1 bar, in both sets: before costs their mean is within {shown} bp "
            f"of zero,\nso each cell loses about the 10 bp cost per trade, with a small standard error.")

    def test_low_cells_of_one_set_only(self, drawn):
        draw_fig2(cells({("1H", 1, NAMES[0]): {"oos_t": -25.0}}), Path(FIG2))
        note = text_with(drawn[FIG2], "before costs")
        assert note.startswith("The 11 cells at 15m bars held 1 bar, 1H bars held 1 bar, in either set:")

    @pytest.mark.parametrize("before", [True, False])
    def test_axis_break_legend_and_footer(self, drawn, before):
        cs = cells()
        thr = draw_fig2(cs, Path(FIG2), before=before)
        fig = drawn[FIG2]
        up, low = fig.axes[:2]
        t_all = [t for c in cs for t in (c.is_t, c.oos_t)]
        below = max(t for t in t_all if t < -10)                    # the 15m h=1 group, near -20
        above = min(t for t in t_all if t > -10)                    # the rest, from -0.96
        # Every t is inside one of the two panels: the break hides no cell.
        for t in t_all:
            assert up.get_ylim()[0] <= t <= up.get_ylim()[1] or low.get_ylim()[0] <= t <= low.get_ylim()[1]
        assert low.get_ylim()[1] < up.get_ylim()[0]
        # The scale ratio the note states is the one drawn: t per inch, lower over upper panel.
        height = fig.get_size_inches()[1]

        def per_inch(ax):
            lo, hi = ax.get_ylim()
            return (hi - lo) / (ax.get_position().height * height)

        assert text_with(fig, "Axis broken") == (
            f"Axis broken: no cell has a t between {minus(math.ceil(below * 10) / 10, '.1f')} and "
            f"{minus(math.floor(above * 10) / 10, '.1f')}. "
            f"Below the break the scale is {per_inch(low) / per_inch(up):.1f}× smaller.")
        assert text_with(fig, "Holm threshold:") == (
            f"Holm threshold: the t a cell needs at its rank, from {thr[0]:.2f} (rank 1) to {thr[-1]:.2f} (rank 60)")
        when = "before" if before else "after"
        assert f"Held-out test year, {PRIMARY_SPAN} ({when} the in-sample data)" in texts(fig)
        title, subtitle, footer = page(fig)
        assert squash("Each dot is one of 60 cells (10 factors × 2 bar sizes × 3 holding horizons)") in squash(subtitle)
        assert squash("(one-sided, 5% family-wise error over the 60 cells)") in squash(subtitle)
        assert squash("In-sample t = (mean return in the traded direction − 10 bp) / SE") in squash(footer)
        assert squash(footer).endswith(squash(f"Sources: results/insample_v2.csv, results/oos/{PRIMARY}.csv."))


# ---------------------------------------------------------------- fig3

class TestSpearman:
    def test_pairs_with_a_missing_value_are_left_out_and_ties_share_a_rank(self):
        x = np.array([1.0, np.nan, 3.0, 2.0, 2.0, 5.0, np.inf])
        y = np.array([10.0, 7.0, np.nan, 30.0, 20.0, 40.0, 1.0])
        # By hand: the finite pairs are (1, 10), (2, 30), (2, 20) and (5, 40). x ranks
        # 1, 2.5, 2.5, 4 and y ranks 1, 3, 2, 4; about the mean rank 2.5 these are
        # (-1.5, 0, 0, 1.5) and (-1.5, 0.5, -0.5, 1.5), so rho = 4.5 / sqrt(4.5 · 5) = sqrt(0.9).
        for a, b in ((x, y), (y, x)):
            rho, n = mf.spearman(a, b)
            assert n == 4 and rho == pytest.approx(math.sqrt(0.9), abs=1e-12)


def bar_frame(ts, factors):
    """What factor_correlations reads of an InstrumentFrame: its bars' start times and factor values."""
    return SimpleNamespace(ts=np.asarray(ts, dtype=np.int64), factors=factors)


class TestFactorCorrelations:
    def test_only_the_windows_bars_count_and_missing_values_are_left_out(self, monkeypatch, capsys):
        step = 15 * MINUTE
        # Three warm-up bars, the window's first eight, and two from its end.
        ts = np.concatenate([IS_WINDOW.start_ms + step * np.arange(-3, 8), IS_WINDOW.end_ms + step * np.arange(2)])
        inside = np.repeat([False, True, False], [3, 8, 2])
        # In the window each factor rises (sign 1) or falls (-1) with time, so each
        # pair's rho is the product of their signs. Outside it every factor rises,
        # which would pull each pair of opposite signs above -1 if those bars counted.
        signs = np.array([1.0 if k % 2 == 0 else -1.0 for k in range(len(NAMES))])
        t = np.arange(ts.size, dtype=np.float64)
        factors = {f: np.where(inside, s * t, 100.0 + t) for f, s in zip(NAMES, signs)}
        factors[NAMES[2]][3 + 3] = np.nan               # the window's 4th bar
        factors[NAMES[5]][3 + 6] = np.inf               # its 7th: these two keep 6 of 8 bars as a pair
        asked = []

        def fake_prepare_frames(data, bar, window, params):
            asked.append((data, bar, window, params))
            return {INST: bar_frame(ts, factors)}

        monkeypatch.setattr(mf, "prepare_frames", fake_prepare_frames)
        data = {"set": IS_SET}
        rho, n_bars = mf.factor_correlations(data, "15m", IS_WINDOW, PARAMS)
        assert asked == [(data, "15m", IS_WINDOW, PARAMS)]
        assert n_bars == 8
        assert rho == pytest.approx(np.outer(signs, signs))
        assert capsys.readouterr().out == "  15m: some factor pairs have only 6 of 8 bars with both values\n"

    def test_one_instrument_only(self, monkeypatch):
        frame = bar_frame([IS_WINDOW.start_ms], {f: np.zeros(1) for f in NAMES})
        monkeypatch.setattr(mf, "prepare_frames", lambda data, bar, window, params: {"B": frame, "A": frame})
        with pytest.raises(SystemExit, match=re.escape("expected one instrument in sample, got ['A', 'B']")):
            mf.factor_correlations({}, "15m", IS_WINDOW, PARAMS)


class TestFig3:
    def test_texts(self, drawn):
        mf.fig_corr(NAMES, corr_mats(), IS_WINDOW, Path(FIG3))
        fig = drawn[FIG3]
        title, subtitle, footer = page(fig)
        assert title == "Some of the ten factors measure nearly the same thing"
        assert squash("each bar of the in-sample period (2025-12-23 → 2026-09-23)") in squash(subtitle)
        assert squash("rsi_14 and bollinger_pctb: 0.90 on 15-minute, 0.91 on 1-hour bars.") in squash(subtitle)
        assert squash("warm-up from 2025-09-24.") in squash(footer)
        panels = fig.axes[:2]
        assert [ax.get_title(loc="left") for ax in panels] == ["15-minute bars (26,384 bars)",
                                                               "1-hour bars (6,596 bars)"]
        for ax in panels:
            shown = list(ax.texts)
            assert len(shown) == len(NAMES) * (len(NAMES) - 1) // 2          # each pair once
            values = [t.get_text() for t in shown]
            assert "0.00" in values and "−0.00" not in values            # -0.004 is shown as 0.00
            # Dark cells carry white numbers. The shown value decides it here, since every
            # value is a multiple of 0.1 or one of 0.904, 0.906 and -0.004.
            for t in shown:
                v = float(t.get_text().replace("−", "-"))
                assert t.get_color() == (mf.SURFACE if abs(v) >= 0.6 else mf.INK)


# ---------------------------------------------------------------- fig4

class TestFig4:
    def test_settled_groups_need_every_upper_bound_below_zero(self):
        assert mf.settled_groups(cells(), PARAMS) == [("15m", 1)]
        assert mf.settled_groups(cells({("15m", 1, NAMES[4]): {"oos_ub95_bps": 0.0}}), PARAMS) == []

    @pytest.mark.parametrize("groups,words", [
        ([("15m", 1), ("1H", 1), ("15m", 5)], "all 20 held 1 bar and the 10 at 15m held 5 bars"),
        ([("1H", 15)], "the 10 at 1H held 15 bars"),
        ([("15m", 5), ("1H", 5)], "all 20 held 5 bars"),
    ])
    def test_groups_in_words(self, groups, words):
        assert mf._describe_groups(groups, PARAMS) == words

    @pytest.mark.parametrize("n,title", [
        (0, "No cell was net positive in both periods"),
        (1, "1 cell was net positive in both periods"),
        (2, "2 cells were net positive in both periods"),
    ])
    def test_title_counts_cells_positive_in_both_periods(self, drawn, n, title):
        # The last two factors at 1H h=15 are net positive in sample; make n of them positive in the test year.
        keys = [("1H", 15, f) for f in NAMES[-2:]][:n]
        draw_fig4(cells({k: {"oos_net_bps": 4.0} for k in keys}), Path(FIG4))
        fig = drawn[FIG4]
        assert page(fig)[0] == title
        assert text_with(fig, "net positive in both periods:") == (
            f"net positive in both periods:\n{['no cell', '1 cell', '2 cells'][n]}")

    @pytest.mark.parametrize("field,net,within,losses", [
        ("oos_net_bps", -COST + 0.7237, "0.8", "9.2 to 10.3"),
        ("oos_net_bps", -COST + 1.2049, "1.3", "8.7 to 10.3"),
        ("oos_net_bps", -COST - 0.62, "0.7", "9.5 to 10.7"),
        ("is_net_bps", -COST + 1.2049, "1.3", "8.7 to 10.3"),
    ])
    def test_the_settled_cells_note(self, drawn, field, net, within, losses):
        # Without the change, the ten cells' net means run from -10.27 (test year) to
        # -9.55 (in sample). Each bound must hold for every cell in both periods, so
        # each is rounded outward: |gross| 0.7237 is "within 0.8", a smallest loss of
        # 9.2763 bp is stated as 9.2 (not 9.3), a largest of 10.62 as 10.7. An
        # in-sample mean counts as much as a test-year one (the last case).
        draw_fig4(cells({("15m", 1, NAMES[1]): {field: net}}), Path(FIG4))
        assert text_with(drawn[FIG4], "pile up") == (
            f"10 cells pile up here: the 10 at 15m held 1 bar.\nBefore costs each earns within {within} bp "
            f"of zero in both periods,\nso each cell loses {losses} bp per trade on average; every upper bound is below zero.")

    def test_no_note_without_a_settled_group(self, drawn):
        # An upper bound of exactly zero is not below zero.
        draw_fig4(cells({("15m", 1, NAMES[4]): {"oos_ub95_bps": 0.0}}), Path(FIG4))
        fig = drawn[FIG4]
        assert squash("9 cells have a test-year 95% upper bound below zero") in squash(page(fig)[1])
        assert not any("pile up" in t for t in texts(fig))

    @pytest.mark.parametrize("sign,shown", [(1, "1.00"), (-1, "−1.00")])
    def test_subtitle_and_footer(self, drawn, sign, shown):
        # The test-year mean a monotone function of the in-sample one: rank correlation ±1.
        cs = [dataclasses.replace(c, oos_net_bps=sign * c.is_net_bps - 20.0) for c in cells()]
        draw_fig4(cs, Path(FIG4))
        title, subtitle, footer = page(drawn[FIG4])
        assert squash("10 cells have a test-year 95% upper bound below zero") in squash(subtitle)
        assert squash(subtitle).endswith(squash(f"Rank correlation of x and y across the cells: {shown}."))
        assert squash("x cannot fall below −10 bp") in squash(footer)
        assert squash("(mean + 1.645 Hansen–Hodrick standard errors, as pre-registered)") in squash(footer)
        assert squash(footer).endswith(squash(f"Sources: results/insample_v2.csv, results/oos/{PRIMARY}.csv."))

    def test_an_in_sample_mean_below_minus_the_cost_is_refused(self, drawn):
        with pytest.raises(SystemExit, match="below −cost"):
            draw_fig4(cells({("1H", 5, NAMES[0]): {"is_net_bps": -COST - 0.01}}), Path(FIG4))
        assert drawn == {}


# ---------------------------------------------------------------- the committed results, drawn

class TestFromTheCommittedResults:
    def test_fig2_hides_no_cell_and_names_the_largest_t(self, drawn):
        cs = mf.read_cells(PARAMS, PRIMARY, UB95_Z)
        thr = draw_fig2(cs, Path(FIG2))
        fig = drawn[FIG2]
        up, low = fig.axes[:2]
        for t in [t for c in cs for t in (c.is_t, c.oos_t)]:
            assert up.get_ylim()[0] <= t <= up.get_ylim()[1] or low.get_ylim()[0] <= t <= low.get_ylim()[1]
        # The study's result: the largest t falls short of rank 1's threshold, so
        # the step-down rejects nothing; the title says so with the committed numbers.
        largest = pl.read_csv(REPO_ROOT / mf.oos_results_path(PRIMARY))["t_hh"].max()
        assert largest < thr[0]
        assert page(fig)[0] == (f"No cell reached its threshold: the largest t was {minus(largest, '.2f')}, "
                                f"short of {thr[0]:.2f}")

    def test_fig4_holds_every_cell(self, drawn):
        cs = mf.read_cells(PARAMS, PRIMARY, UB95_Z)
        draw_fig4(cs, Path(FIG4))
        (ax,) = drawn[FIG4].axes
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        for c in cs:
            assert x0 <= c.is_net_bps <= x1 and y0 <= c.oos_net_bps <= c.oos_ub95_bps <= y1
        # The note's loss range holds the net mean of every cell in a group whose
        # bounds are all below zero, in both periods, and is no wider than rounding needs.
        unsettled = {(c.bar, c.horizon) for c in cs if c.oos_ub95_bps >= 0}
        losses = [-x for c in cs if (c.bar, c.horizon) not in unsettled for x in (c.is_net_bps, c.oos_net_bps)]
        least, most = map(float, re.search(r"loses (\S+) to (\S+) bp", text_with(drawn[FIG4], "pile up")).groups())
        assert least <= min(losses) < least + 0.1 and most - 0.1 < max(losses) <= most


# ---------------------------------------------------------------- determinism and the files written

class TestDeterminism:
    def test_style_is_matplotlibs_own_defaults_plus_style(self):
        rc = matplotlib.rcParams
        rc["lines.linewidth"], rc["font.size"], rc["axes.facecolor"] = 9.0, 30.0, "black"   # as a matplotlibrc might
        mf._apply_style()
        style = matplotlib.RcParams(mf.STYLE)                  # validated: a font family becomes a list
        for key, default in matplotlib.rcParamsDefault.items():
            if key not in STYLE_BLACKLIST:
                assert rc[key] == (style[key] if key in mf.STYLE else default), key

    def test_a_saved_png_carries_no_software_version(self, tmp_path):
        fig = mf._new_figure(2.0)
        fig.add_axes((0.1, 0.1, 0.8, 0.8)).plot([0, 1], [0, 1])
        fig.savefig(tmp_path / "plain.png", dpi=mf.DPI, format="png")
        mf._save(fig, tmp_path / "ours.png")
        assert b"Software" in png_text_keys(tmp_path / "plain.png")         # what matplotlib adds otherwise
        assert png_text_keys(tmp_path / "ours.png") == []
        assert png_size(tmp_path / "ours.png") == (1800, 400)

    def test_the_same_inputs_give_the_same_bytes(self, tmp_path):
        for name in ("a.png", "b.png"):
            mf.fig_corr(NAMES, corr_mats(), IS_WINDOW, tmp_path / name)
        assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()

    def test_each_figure_is_written_as_a_png_of_the_page_width(self, tmp_path):
        committed = mf.read_cells(PARAMS, PRIMARY, UB95_Z)
        mf.fig_cost(years(), (mf.BLUE, mf.ORANGE, mf.AQUA), PARAMS, tmp_path / FIG1)
        draw_fig2(committed, tmp_path / FIG2)
        mf.fig_corr(NAMES, corr_mats(), IS_WINDOW, tmp_path / FIG3)
        draw_fig4(committed, tmp_path / FIG4)
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(mf.FIGURE_NAMES)
        for name in mf.FIGURE_NAMES:
            path = tmp_path / name
            with Image.open(path) as im:
                im.verify()
            width, height = png_size(path)
            assert width == round(mf.WIDTH_IN * mf.DPI) == 1800 and height > 0
            assert png_text_keys(path) == []


# ---------------------------------------------------------------- main, with fakes for data/

DAY = 86_400_000
# The 1-minute returns (bp) each set's fake candles repeat in its evaluation
# window: 50%, 25% and 12.5% of them exceed the 10 bp cost, as in years().
PATTERNS = {PRIMARY: [5, -12, 20, -3], REPLICATION: [5, -3, 2, -11], IS_SET: [1, -2, 3, -4, 5, -6, 7, -12]}


def minute_candles(window, pattern, edge=10):
    """
    1-minute candles whose returns repeat ``pattern`` (bp) 50 times from the
    window's start. Before them come the last ``edge`` minutes of the warm-up,
    and from the window's end ``edge`` more; those all move 50 bp, and only
    the window's own minutes may reach fig1 and the §1 table.
    """
    inside = np.tile(np.asarray(pattern, dtype=np.float64), 50)
    moves = np.concatenate([np.full(edge, 50.0), inside, np.full(edge, 50.0)]) / 1e4
    ts = np.concatenate([window.start_ms - MINUTE * np.arange(edge, 0, -1),
                         window.start_ms + MINUTE * np.arange(inside.size + 1),
                         window.end_ms + MINUTE * np.arange(edge)])
    close = np.concatenate([[1.0], np.cumprod(1.0 + moves)])
    return pl.DataFrame({"timestamp": ts.astype(np.int64), "close": close})


def factor_frame(bar, window):
    """Ten factors on four warm-up bars and then the window's first day of ``bar`` bars."""
    step = mf.bar_ms(bar)
    ts = window.start_ms + step * np.arange(-4, DAY // step)
    i = np.arange(ts.size, dtype=np.float64)
    return bar_frame(ts, {f: np.sin((k + 1) * i) for k, f in enumerate(NAMES)})


class FakeInputs:
    """verify_inputs, load_set_data and prepare_frames as main() calls them, without data/; each records its calls."""

    def __init__(self, problems=None):
        self.problems = problems or {}
        self.verified, self.loaded, self.prepared = [], [], []

    def install(self, monkeypatch):
        for name in ("verify_inputs", "load_set_data", "prepare_frames"):
            monkeypatch.setattr(mf, name, getattr(self, name))
        return self

    def verify_inputs(self, set_name):
        self.verified.append(set_name)
        return list(self.problems.get(set_name, []))

    def load_set_data(self, set_name):
        self.loaded.append(set_name)
        perp = minute_candles(mf.window_of(set_name), PATTERNS[set_name])
        return {"set": set_name, "instruments": (INST,), "perp": {INST: perp}, "spot": {INST: None},
                "funding": {INST: None}, "files": []}

    def prepare_frames(self, data, bar, window, params):
        self.prepared.append((data["set"], bar, window, params))
        return {INST: factor_frame(bar, window)}


class TestRequireVerified:
    def test_each_set_is_checked(self, monkeypatch):
        fake = FakeInputs().install(monkeypatch)
        assert mf.require_verified([IS_SET, PRIMARY, REPLICATION]) is None
        assert fake.verified == [IS_SET, PRIMARY, REPLICATION]

    def test_a_problem_is_reported_with_its_set(self, monkeypatch):
        problems = ["a.parquet: SHA-256 0 differs from 1 in m.json",
                    "b.parquet would be read but no manifest lists it"]
        FakeInputs({PRIMARY: problems}).install(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            mf.require_verified([IS_SET, PRIMARY, REPLICATION])
        assert exc.value.code == (f"inputs of {PRIMARY} do not match their committed hashes:\n"
                                  "  a.parquet: SHA-256 0 differs from 1 in m.json\n"
                                  "  b.parquet would be read but no manifest lists it")


class TestMain:
    def test_it_draws_the_four_figures_from_each_sets_window(self, tmp_path, monkeypatch, capsys):
        fake = FakeInputs().install(monkeypatch)
        out = tmp_path / "docs" / "figures"                     # created, with its parent
        assert mf.main(["--out", str(out)]) == 0
        printed = capsys.readouterr().out
        assert sorted(p.name for p in out.iterdir()) == sorted(mf.FIGURE_NAMES)
        for name in mf.FIGURE_NAMES:
            with Image.open(out / name) as im:
                im.verify()
            assert png_size(out / name)[0] == 1800 and png_text_keys(out / name) == []
        sets = sorted([IS_SET, PRIMARY, REPLICATION])
        assert sorted(fake.verified) == sorted(fake.loaded) == sets
        assert fake.prepared == [(IS_SET, bar, mf.window_of(IS_SET), PARAMS) for bar in PARAMS.bars]

        # The §1 table counts each set's own evaluation window: none of the
        # 50 bp minutes before or after it, and no gap between its candles.
        def line(label):
            return next(l for l in printed.splitlines() if l.strip().startswith(label))

        assert line("Share of minutes").split()[-3:] == ["50.0%", "25.0%", "12.5%"]
        assert line("Returns (gaps skipped)").split()[-6:] == ["200", "(0)", "200", "(0)", "400", "(0)"]
        assert squash(line("Window, UTC")).endswith(squash(
            "2023-09-23 20:00 -> 2024-09-23 20:00  2024-09-23 20:00 -> 2025-09-23 20:00  "
            "2025-12-23 00:00 -> 2026-09-23 20:00"))
        for bar, n in (("15m", 96), ("1H", 24)):
            assert f"Spearman correlation of the factors, in sample, {bar} bars ({n} bars, {IS_SPAN})" in printed
        assert "some factor pairs" not in printed
        assert printed.rstrip().endswith("wrote " + ", ".join(str(out / name) for name in mf.FIGURE_NAMES))

        # The same inputs drawn again give the same bytes, figure by figure.
        again = tmp_path / "again"
        assert mf.main(["--out", str(again)]) == 0
        for name in mf.FIGURE_NAMES:
            assert (again / name).read_bytes() == (out / name).read_bytes(), name

    @pytest.mark.parametrize("bad", [IS_SET, PRIMARY, REPLICATION])
    def test_inputs_that_fail_their_hashes_stop_it_before_anything_is_written(self, tmp_path, monkeypatch, bad):
        fake = FakeInputs({bad: ["x.parquet: SHA-256 0 differs from 1 in m.json"]}).install(monkeypatch)
        out = tmp_path / "figures"
        with pytest.raises(SystemExit, match=f"inputs of {bad} do not match their committed hashes"):
            mf.main(["--out", str(out)])
        assert not out.exists() and fake.loaded == [] and fake.prepared == []


# ---------------------------------------------------------------- a reader's download of the in-sample year

IS_WINDOW = mf.window_of(IS_SET)
INPUT_FILES = mf.pipeline.input_files                    # the pipeline's own, before any test replaces it
# Each in-sample file with its counterpart in a fetch_v2 download (<out-dir>/<instId>/<file>).
COUNTERPARTS = [("data/history/DOGE-USDT-SWAP/candles_1m.parquet", "DOGE-USDT-SWAP/candles_1m.parquet"),
                ("data/history/is_supp/DOGE-USDT/candles_1m.parquet", "DOGE-USDT/candles_1m.parquet"),
                ("data/history/is_supp/DOGE-USDT-SWAP/funding_rates.parquet", "DOGE-USDT-SWAP/funding_rates.parquet")]


def download(path, first=IS_WINDOW.warmup_start_ms, leave_out=()):
    """Small files in fetch_v2's layout, whose minutes run from ``first`` to the in-sample year's last."""
    for _, rel in COUNTERPARTS:
        if rel not in leave_out:
            (path / rel).parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"timestamp": [first, IS_WINDOW.end_ms - MINUTE], "close": [1.0, 1.0]}).write_parquet(
                path / rel)
    return path


def supplement_manifest(root, fresh):
    """results/manifest_is_supp.json under ``root``, recording the supplement's counterparts in ``fresh`` as they are."""
    series = []
    for rel, twin in COUNTERPARTS[1:]:
        ts = pl.read_parquet(fresh / twin)["timestamp"]
        series.append({"path": rel, "sha256": mf.sha256_file(fresh / twin), "rows": ts.len(),
                       "first_ts": int(ts.min()), "last_ts": int(ts.max())})
    (root / "results").mkdir(parents=True)
    (root / "results" / "manifest_is_supp.json").write_text(json.dumps({"series": series}), encoding="utf-8")
    return root


class TestInSampleDownload:
    def test_it_is_read_for_the_in_sample_files_only(self, tmp_path):
        fresh = download(tmp_path / "fresh")
        root = supplement_manifest(tmp_path / "repo", fresh)
        before = {n: mf.pipeline.input_files(n, root) for n in (IS_SET, PRIMARY, REPLICATION)}
        with mf.reading_fresh(fresh):
            during = {n: mf.pipeline.input_files(n, root) for n in before}
        assert mf.pipeline.input_files is INPUT_FILES
        assert {n: [(b.path.relative_to(root).as_posix(), d.path.relative_to(fresh).as_posix())
                    for b, d in zip(before[n], during[n]) if d.path != b.path] for n in before} == {
            IS_SET: COUNTERPARTS, PRIMARY: [], REPLICATION: []}
        # The supplement's files only over the span its manifest records.
        span = (IS_WINDOW.warmup_start_ms, IS_WINDOW.end_ms - MINUTE + 1)
        assert [(d.lo_ms, d.hi_ms) for d in during[IS_SET]] == [(None, None), span, span]
        with mf.reading_fresh(None):
            assert mf.pipeline.input_files is INPUT_FILES

    def test_the_files_whose_bytes_differ_from_the_runs_are_named(self, monkeypatch, tmp_path):
        fresh = download(tmp_path / "fresh")
        root = supplement_manifest(tmp_path / "repo", fresh)
        monkeypatch.setattr(mf.pipeline, "REPO_ROOT", root)
        # The supplement's two files have the bytes its manifest records; the candles cannot have Appendix A's.
        assert mf.check_fresh(fresh, IS_SET, IS_WINDOW) == [COUNTERPARTS[0][0]]

    @pytest.mark.parametrize("leave_out,first,needle", [
        (("DOGE-USDT/candles_1m.parquet",), IS_WINDOW.warmup_start_ms, "DOGE-USDT/candles_1m.parquet does not exist"),
        ((), IS_WINDOW.warmup_start_ms + MINUTE,
         "does not cover the in-sample year's warm-up and window, 2025-09-24 00:00 to 2026-09-23 20:00 UTC"),
    ])
    def test_a_missing_file_or_a_short_year_stops_it(self, monkeypatch, tmp_path, leave_out, first, needle):
        monkeypatch.setattr(mf.pipeline, "REPO_ROOT", supplement_manifest(tmp_path / "repo", download(tmp_path / "twin")))
        fresh = download(tmp_path / "fresh", first=first, leave_out=leave_out)
        with pytest.raises(SystemExit, match=re.escape(needle)):
            mf.check_fresh(fresh, IS_SET, IS_WINDOW)

    def test_funding_short_of_the_runs_settlements_stops_it(self, monkeypatch, tmp_path):
        fresh = download(tmp_path / "fresh")
        monkeypatch.setattr(mf.pipeline, "REPO_ROOT", supplement_manifest(tmp_path / "repo", fresh))
        pl.DataFrame({"timestamp": [IS_WINDOW.warmup_start_ms]}).write_parquet(fresh / COUNTERPARTS[2][1])
        with pytest.raises(SystemExit, match=re.escape("holds 1 settlements from 2025-09-24 00:00 to 2026-09-23 "
                                                       "19:59 UTC, where the run's file held 2")):
            mf.check_fresh(fresh, IS_SET, IS_WINDOW)

    def test_main_reads_it_in_place_of_the_in_sample_files(self, tmp_path, monkeypatch, capsys):
        fake = FakeInputs().install(monkeypatch)
        fresh, checked, swapped = tmp_path / "fresh", [], []
        fresh.mkdir()
        monkeypatch.setattr(mf, "check_fresh",
                            lambda f, name, window: checked.append((f, name, window)) or ["x.parquet"])

        def load_set_data(set_name):
            swapped.append((set_name, mf.pipeline.input_files is not INPUT_FILES))
            return fake.load_set_data(set_name)

        monkeypatch.setattr(mf, "load_set_data", load_set_data)
        out = tmp_path / "figures"
        assert mf.main(["--out", str(out), "--insample-from", str(fresh)]) == 0
        # The in-sample year's files are checked against the download, not against their hashes.
        assert sorted(fake.verified) == sorted([PRIMARY, REPLICATION])
        assert checked == [(fresh.resolve(), IS_SET, IS_WINDOW)]
        assert sorted(swapped) == sorted([(IS_SET, True), (PRIMARY, False), (REPLICATION, False)])
        assert (f"in-sample files read from {fresh.resolve()}; these have other bytes than the results were "
                "computed from, so the figures check them by value: x.parquet") in capsys.readouterr().out
        assert sorted(p.name for p in out.iterdir()) == sorted(mf.FIGURE_NAMES)

    def test_a_download_that_fails_its_check_stops_it_before_anything_is_written(self, tmp_path, monkeypatch):
        fake = FakeInputs().install(monkeypatch)

        def refuse(f, name, window):
            raise SystemExit("--insample-from: x does not exist")

        monkeypatch.setattr(mf, "check_fresh", refuse)
        out = tmp_path / "figures"
        with pytest.raises(SystemExit, match="x does not exist"):
            mf.main(["--out", str(out), "--insample-from", str(tmp_path)])
        assert not out.exists() and fake.loaded == []
