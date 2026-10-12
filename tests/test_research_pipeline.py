"""
research/pipeline.py end to end, on synthetic data only.

No real candles, no network. One module-scoped study builds small synthetic
sets in a temporary directory laid out exactly as fetch_v2 and the
pre-registration put them (data/history/..., results/...), runs the
in-sample rerun, and then the four test sets in the order §10 requires
(doge_2024, majors_2023, doge_2023 last, doge_fwd). The tests read the files
the pipeline wrote, because those files are the deliverable.

How the synthetic world is built, so the assertions mean something:

  * 1-minute perp candles are a random walk plus a planted effect: a
    standard-normal g_b per 15-minute bucket b moves bucket b+2's return
    by beta * g_b. Prices up to bucket b carry no information about g_b,
    so every price-derived factor is pure noise for the forward return.
  * The spot leg is the perp divided by (1 + kappa * g_b), so the basis on
    the 15m grid is exactly kappa * g_b: basis_z_score at 15m is the one
    factor that predicts, net of the 10 bp cost. At 1H the one-bar entry
    delay is longer than the two-bucket lag, so basis is noise there too.
  * A few 1-minute bars are deleted from each series, so buckets are
    dropped and forward returns across them are undefined.
  * Funding settles every 8 h with noise rates. doge_2024 has no funding
    file (its funding cells must be untestable), and SOL in majors_2023
    has no spot leg (its pooled basis cells must be untestable).
  * The windows have Appendix A's shape where it matters: the primary
    evaluation window is longer than the in-sample one (so the MDE's
    sqrt(N_IS / N_primary) is not 1 and an argument swap shows), and
    doge_fwd's evaluation starts exactly where the in-sample files end.
    The in-sample perp file runs a day past that split with different
    prices, so doge_fwd reading it beyond the split would show.

Look-backs are shortened (as in tests/test_research_factors.py) and the
placebo uses K = 199 with 5-day minimum shifts so the windows can be a few
weeks; the bucket burn-in (500), the 300-trade rule, the 10 bp cost and
alpha are the spec's own.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from research import freeze
from research import pipeline as P
from research import stats as S
from research.io import sha256_file, write_once_csv
from research.spec import get_spec

DAY = 86_400_000
MIN = 60_000
M15 = 900_000
H8 = 8 * 3_600_000
T0 = 1_600_000_000_000 // DAY * DAY
WARM, EVAL = 30, 40
BETA = 20e-4          # planted move per 15m bucket per unit of g
KAPPA = 5e-4          # basis per unit of g
SPEC = get_spec()


def window(t0, warm=WARM, ev=EVAL):
    return P.Window(t0, t0 + warm * DAY, t0 + (warm + ev) * DAY)


WINDOWS = {
    "IS": window(T0),
    # Appendix A: doge_fwd.start_ms == IS.end_ms; its warm-up is in-sample data.
    "doge_fwd": window(T0 + 45 * DAY, 25, 15),
    # Appendix A: the primary window (366 days) is longer than IS (274.8 days),
    # and majors_2023 shares the primary's dates.
    "doge_2023": window(T0 - 400 * DAY, WARM, 50),
    "majors_2023": window(T0 - 400 * DAY, WARM, 50),
    "doge_2024": window(T0 - 300 * DAY),
}
IS_TAIL_DAYS = 1      # in-sample perp rows past the split, with other prices


# ---------------------------------------------------------------- synthetic data

def synthetic_minutes(lo, hi, seed, beta, sigma=5e-4, n_drop=5):
    """(perp, spot, funding) over [lo, hi); see the module docstring."""
    rng = np.random.default_rng(seed)
    ts = np.arange(lo, hi, MIN, dtype=np.int64)
    n = len(ts)
    b = (ts - lo) // M15
    g = rng.standard_normal(int(b[-1]) + 1)
    drift = np.zeros(len(g))
    drift[2:] = beta * g[:-2]
    close = 0.1 * np.exp(np.cumsum(rng.normal(0, sigma, n) + drift[b] / 15))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 2e-4, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 2e-4, n))
    vol = rng.uniform(1e3, 1e4, n)
    perp = pl.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                         "close": close, "volume": vol, "volume_ccy": vol * close})
    k = pl.Series(1.0 + KAPPA * g[b])
    spot = perp.with_columns(*(pl.col(c) / k for c in ("open", "high", "low", "close")))

    def holes(df):
        gone = np.isin(np.arange(n), rng.choice(np.arange(10, n - 10), size=n_drop, replace=False))
        return df.filter(~pl.Series(gone))

    f_ts = np.arange(-(-lo // H8) * H8, hi, H8, dtype=np.int64)
    funding = pl.DataFrame({"timestamp": f_ts, "rate": rng.normal(1e-4, 5e-5, len(f_ts)),
                            "realized_rate": np.zeros(len(f_ts))})
    return holes(perp), holes(spot), funding


def put(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def small_params():
    sp = P.StudyParams.from_spec()
    f = copy.deepcopy(sp.factor_params)
    f["macd_hist_norm"].update(fast=3, slow=6, signal=3)
    f["rolling_vol_ratio"].update(short=5, long=30)
    f["net_taker_flow"]["lookback"] = 10
    f["basis_z_score"]["lookback"] = 30
    f["funding_cost_pressure"]["lookback_settlements"] = 20
    f["ou_reversion"]["window"] = 20
    f["hurst_regime"].update(window=40, q=5)
    f["triple_screen"]["macd"] = [3, 6, 3]
    q0 = WINDOWS["doge_2023"].start_ms
    return replace(sp, factor_params=f, placebo_k=199, placebo_min_shift_days=5,
                   quarter_bounds_ms=(q0 + 10 * DAY, q0 + 20 * DAY, q0 + 30 * DAY))


def write_world(root: Path) -> None:
    hist = root / "data" / "history"
    legs = SPEC["spot_legs"]
    # The in-sample year and doge_fwd are one continuous path, split where
    # the in-sample files end, as on disk for the real study.
    perp, spot, fund = synthetic_minutes(WINDOWS["IS"].warmup_start_ms, WINDOWS["doge_fwd"].end_ms, 1, BETA)
    split = WINDOWS["IS"].end_ms
    before, after = pl.col("timestamp") < split, pl.col("timestamp") >= split
    # The in-sample file holds a day past the split at other prices (as a
    # file fetched after the split would): neither the in-sample run nor
    # doge_fwd may read those rows.
    tail = perp.filter(after & (pl.col("timestamp") < split + IS_TAIL_DAYS * DAY)).with_columns(
        *(pl.col(c) * 1.5 for c in ("open", "high", "low", "close")))
    put(pl.concat([perp.filter(before), tail]), root / SPEC["insample_dir"] / P.CANDLE_FILE)
    put(spot.filter(before), root / P.IS_SUPP_DIR / "DOGE-USDT" / P.CANDLE_FILE)
    put(fund.filter(before), root / P.IS_SUPP_DIR / "DOGE-USDT-SWAP" / P.FUNDING_FILE)
    fwd = hist / "oos" / "doge_fwd"
    put(perp.filter(after), fwd / "DOGE-USDT-SWAP" / P.CANDLE_FILE)
    put(spot.filter(after), fwd / "DOGE-USDT" / P.CANDLE_FILE)
    put(fund.filter(after), fwd / "DOGE-USDT-SWAP" / P.FUNDING_FILE)

    def oos(set_name, inst, seed, beta, spot_ok=True, funding_ok=True):
        w = WINDOWS[set_name]
        p, s, f = synthetic_minutes(w.warmup_start_ms, w.end_ms, seed, beta)
        base = hist / "oos" / set_name
        put(p, base / inst / P.CANDLE_FILE)
        if spot_ok:
            put(s, base / legs[inst] / P.CANDLE_FILE)
        if funding_ok:
            put(f, base / inst / P.FUNDING_FILE)

    oos("doge_2023", "DOGE-USDT-SWAP", 2, BETA)
    oos("doge_2024", "DOGE-USDT-SWAP", 3, BETA, funding_ok=False)
    oos("majors_2023", "BTC-USDT-SWAP", 4, 0.0)
    oos("majors_2023", "ETH-USDT-SWAP", 5, 0.0)
    oos("majors_2023", "SOL-USDT-SWAP", 6, 0.0, spot_ok=False)


class Study:
    def __init__(self, root, params):
        self.root = root
        self.params = params

    def csv(self, name) -> pl.DataFrame:
        return P.read_results_csv(self.root / "results" / name)


def floats(df, col):
    return np.array([math.nan if v is None else float(v) for v in df[col].to_list()])


def p_upper(t):
    """1 - Φ(t) to full precision in the tail (the textbook erfc form)."""
    return 0.5 * math.erfc(t / math.sqrt(2.0))


def independent_cell(frame, factor, h, d, w, params):
    """
    One cell rebuilt from the frame's raw arrays with research.stats alone.

    Bypasses InstrumentFrame.signal and evaluate_cell, and passes every
    parameter explicitly, so a slip in the pipeline's wiring (full-window
    edges, the wrong h, the wrong d or c) cannot hide behind the same slip
    here.
    """
    B = frame.bar_ms
    f = frame.factors[factor]
    lo, hi = S.expanding_edges(f, params.edges_pct[0], params.edges_pct[1], params.min_prior)
    s = S.assign_buckets(f, lo, hi)
    fwd = S.forward_returns(frame.close, frame.ts, B, h)
    net = S.per_trade_net(s, fwd, d, params.cost_bps)
    ev = S.evaluable(frame.ts, w.start_ms, w.end_ms, B, h)
    st = S.cell_statistic(net, frame.ts, w.start_ms, w.end_ms, B, h)
    return {"f": f, "s": s, "fwd": fwd, "net": net, "ev": ev, "st": st,
            "active": ev & (s != 0) & np.isfinite(fwd)}


def n_grid_evaluable(w, B, h):
    """Evaluable bars of a gap-free window, from the window arithmetic of §9."""
    return (w.end_ms - w.start_ms) // B - (h + 1)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    write_world(root)
    params = small_params()

    ins = P.run_insample(root, window=WINDOWS["IS"], primary_window=WINDOWS["doge_2023"], params=params)
    write_once_csv(ins, root / P.INSAMPLE_RESULTS)
    # Directions and MDEs go through the file and the amendment format, as in the real run.
    text = (root / P.INSAMPLE_RESULTS).read_text()
    record = freeze.parse_freeze_amendment(freeze.render_amendment_blocks("a" * 40, text))
    directions = P.directions_from_insample(freeze.parse_insample_csv(text))
    mde = {k: c.mde_bps for k, c in record.cells.items()}

    for set_name in ("doge_2024", "majors_2023", "doge_2023", "doge_fwd"):
        main, robust = P.run_oos_set(
            set_name, directions, root=root, window=WINDOWS[set_name], params=params,
            mde_bps=mde, code_commit="c" * 40, manifest_sha256=f"sha-of-{set_name}",
            is_end_ms=WINDOWS["IS"].end_ms,
        )
        write_once_csv(main, root / P.oos_results_path(set_name))
        if robust is not None:
            write_once_csv(robust, root / P.robustness_results_path(set_name))
    return Study(root, params)


# ---------------------------------------------------------------- (a) the in-sample file

class TestInsample:
    def test_exact_columns_60_rows_and_directions(self, study):
        path = study.root / P.INSAMPLE_RESULTS
        assert path.read_text().splitlines()[0] == ",".join(SPEC["freeze"]["directions_columns"])
        df = study.csv("insample_v2.csv")
        assert df.height == 60
        assert [(r["bar"], int(r["horizon"]), r["factor"]) for r in df.iter_rows(named=True)] == \
            freeze.expected_cells()
        assert set(df["direction"].to_list()) <= {"1", "-1"}
        assert set(df["flagged"].to_list()) <= {"true", "false"}

    def test_direction_rule_and_flags(self, study):
        df = study.csv("insample_v2.csv")
        for r in df.iter_rows(named=True):
            n = int(r["n_active"])
            if r["flagged"] == "true":
                assert r["direction"] == "1"
                assert r["flag_reason"] in ("few_active", "missing_input")
                if r["flag_reason"] == "few_active":
                    assert n < study.params.min_active_is
            else:
                assert r["flag_reason"] is None and n >= study.params.min_active_is
                mean = float(r["mean_s_fwd_bps"])
                assert int(r["direction"]) == (1 if mean >= 0 else -1)
        # The planted factor's sign is found in sample.
        basis15 = df.filter((pl.col("factor") == "basis_z_score") & (pl.col("bar") == "15m"))
        assert basis15["direction"].to_list() == ["1", "1", "1"]

    def test_se_and_mde(self, study):
        df = study.csv("insample_v2.csv")
        w_is, w_pr = WINDOWS["IS"], WINDOWS["doge_2023"]
        z = SPEC["power"]["z_alpha"] + SPEC["power"]["z_power"]
        for r in df.iter_rows(named=True):
            if int(r["n_active"]) == 0:
                assert r["se_bps"] is None and r["mde_bps"] is None and r["se_method"] is None
                continue
            assert r["se_method"] in ("hh", "bartlett_fallback")
            B = P.bar_ms(r["bar"])
            h = int(r["horizon"])
            n_is = (w_is.end_ms - w_is.start_ms) // B - (h + 1)
            n_pr = (w_pr.end_ms - w_pr.start_ms) // B - (h + 1)
            # Unequal, as in Appendix A: with N_IS == N_primary the scale is
            # 1 and swapped scale_se arguments would go unnoticed.
            assert n_is != n_pr
            assert float(r["mde_bps"]) == pytest.approx(z * float(r["se_bps"]) * math.sqrt(n_is / n_pr), rel=1e-12)

    def test_rows_are_recomputed_independently(self, study):
        """direction, flag, n_active, mean, SE and MDE of every row from raw arrays (§6.5, §9)."""
        df = study.csv("insample_v2.csv")
        params, w_is, w_pr = study.params, WINDOWS["IS"], WINDOWS["doge_2023"]
        z = SPEC["power"]["z_alpha"] + SPEC["power"]["z_power"]
        data = P.load_set_data("IS", study.root)
        checked = 0
        for bar in params.bars:
            frame = P.prepare_frames(data, bar, w_is, params)["DOGE-USDT-SWAP"]
            for r in df.filter(pl.col("bar") == bar).iter_rows(named=True):
                h, factor = int(r["horizon"]), r["factor"]
                c = independent_cell(frame, factor, h, 1, w_is, params)
                act, s, fwd = c["active"], c["s"], c["fwd"]
                n = int(act.sum())
                assert int(r["n_active"]) == n
                assert r["flag_reason"] != "missing_input"           # every IS input exists here
                mean = float(np.mean(s[act] * fwd[act])) if n else math.nan
                if n < params.min_active_is:
                    want_d, want_flag, want_reason = 1, "true", "few_active"
                else:
                    want_d, want_flag, want_reason = (1 if mean >= 0 else -1), "false", None
                assert (int(r["direction"]), r["flagged"], r["flag_reason"]) == (want_d, want_flag, want_reason)
                if n == 0:
                    assert r["se_bps"] is None and r["mde_bps"] is None
                    continue
                assert float(r["mean_s_fwd_bps"]) == pytest.approx(mean * 1e4, rel=1e-9)
                st = S.cell_statistic(S.per_trade_net(s, fwd, want_d, params.cost_bps), frame.ts,
                                      w_is.start_ms, w_is.end_ms, frame.bar_ms, h)
                assert float(r["se_bps"]) == pytest.approx(st["se"] * 1e4, rel=1e-9)
                assert r["se_method"] == st["se_method"]
                scale = math.sqrt(n_grid_evaluable(w_is, frame.bar_ms, h) / n_grid_evaluable(w_pr, frame.bar_ms, h))
                assert float(r["mde_bps"]) == pytest.approx(z * st["se"] * scale * 1e4, rel=1e-9)
                checked += 1
        assert checked >= 40

    def test_basis_cells_are_flagged_missing_input_without_the_spot_file(self, study, tmp_path):
        # The in-sample world without data/history/is_supp/DOGE-USDT/: the
        # basis cells must be frozen as flagged for the missing input (§6.5,
        # §6.8), not as "few_active" because an all-NaN factor never trades.
        for rel in (Path(SPEC["insample_dir"]) / P.CANDLE_FILE,
                    P.IS_SUPP_DIR / "DOGE-USDT-SWAP" / P.FUNDING_FILE):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes((study.root / rel).read_bytes())
        assert not (tmp_path / P.IS_SUPP_DIR / "DOGE-USDT" / P.CANDLE_FILE).exists()
        params = replace(study.params, bars=("15m",))
        ins = P.run_insample(tmp_path, window=WINDOWS["IS"], primary_window=WINDOWS["doge_2023"], params=params)
        full = study.csv("insample_v2.csv").filter(pl.col("bar") == "15m")
        assert ins.height == 30
        for r, ref in zip(ins.iter_rows(named=True), full.iter_rows(named=True)):
            assert (r["bar"], r["horizon"], r["factor"]) == (ref["bar"], int(ref["horizon"]), ref["factor"])
            if r["factor"] == "basis_z_score":
                assert (r["direction"], r["flagged"], r["flag_reason"]) == (1, True, "missing_input")
                assert r["n_active"] == 0
            else:
                # Every other factor is untouched by the missing spot leg.
                assert r["flag_reason"] != "missing_input"
                assert str(r["direction"]) == ref["direction"] and r["n_active"] == int(ref["n_active"])


# ---------------------------------------------------------------- (b) the planted factor

class TestPrimaryDecision:
    def test_planted_factor_confirmed_and_noise_not(self, study):
        df = study.csv("oos/doge_2023.csv")
        assert df.height == 60 and set(df["instrument"].to_list()) == {"DOGE-USDT-SWAP"}
        confirmed = {(r["bar"], int(r["horizon"]), r["factor"])
                     for r in df.iter_rows(named=True) if r["status"] == "confirmed"}
        assert ("15m", 1, "basis_z_score") in confirmed
        assert confirmed <= {("15m", h, "basis_z_score") for h in (1, 5, 15)}
        # Decisions use Holm-adjusted p at alpha.
        for r in df.iter_rows(named=True):
            if r["status"] == "confirmed":
                assert float(r["p_holm"]) <= study.params.alpha
            elif r["status"] == "not_rejected":
                assert float(r["p_holm"]) > study.params.alpha

    def test_holm_runs_over_all_60_primary_cells(self, study):
        df = study.csv("oos/doge_2023.csv")
        p = floats(df, "p_one_sided")
        np.testing.assert_allclose(floats(df, "p_holm"), S.holm(p), rtol=1e-12)
        np.testing.assert_allclose(floats(df, "q_bh"), S.bh(p), rtol=1e-12)
        np.testing.assert_allclose(floats(df, "q_by"), S.by(p), rtol=1e-12)

    def test_labels_come_from_the_secondary_files(self, study):
        prim = study.csv("oos/doge_2023.csv")
        rep = {(r["bar"], r["horizon"], r["factor"]): r["status"]
               for r in study.csv("oos/doge_2024.csv").iter_rows(named=True)}
        maj = {(r["bar"], r["horizon"], r["factor"]): r["status"]
               for r in study.csv("oos/majors_2023.csv").filter(pl.col("instrument") == P.POOLED).iter_rows(named=True)}
        n_replicated = 0
        for r in prim.iter_rows(named=True):
            k = (r["bar"], r["horizon"], r["factor"])
            conf = r["status"] == "confirmed"
            assert r["replicated"] == ("true" if conf and rep[k] == "rejected" else "false")
            assert r["consistent_on_majors"] == ("true" if conf and maj[k] == "rejected" else "false")
            n_replicated += r["replicated"] == "true"
        # doge_2024 carries the same planted effect; the majors do not.
        assert n_replicated >= 1
        assert set(prim["consistent_on_majors"].to_list()) == {"false"}

    def test_primary_refuses_before_its_secondaries(self, tmp_path, study):
        dirs = {k: (1, False) for k in study.params.cells()}
        mde = {k: 1.0 for k in study.params.cells()}
        with pytest.raises(P.PipelineError, match="must run after"):
            P.run_oos_set("doge_2023", dirs, root=tmp_path, params=study.params, mde_bps=mde,
                          window=WINDOWS["doge_2023"])

    def test_robustness_rows(self, study):
        rb = study.csv("oos/robustness_doge_2023.csv")
        assert tuple(rb.columns) == P.ROBUSTNESS_COLUMNS
        per_cell = {"R3": 4, "R4": 3, "R5": 1, "R6": 4, "R7": 1, "R8": 1, "R10": 5, "RL": 1}
        assert rb.height == 60 * sum(per_cell.values())
        counts = rb.group_by("row").len()
        assert dict(zip(counts["row"].to_list(), counts["len"].to_list())) == \
            {k: 60 * v for k, v in per_cell.items()}
        r5 = rb.filter(pl.col("row") == "R5")
        assert r5["se_bps"].null_count() == r5["t_hh"].null_count() == 60
        assert r5["value_bps"].null_count() == 0
        # RL (§10): value_bps is a length in bars, n_active the number of
        # runs, se and t blank; written for every cell, confirmed or not.
        rl = rb.filter(pl.col("row") == "RL")
        assert set(rl["parameter"].to_list()) == {"mean_run_length"}
        assert rl["se_bps"].null_count() == rl["t_hh"].null_count() == 60
        with_runs = rl.filter(pl.col("n_active").cast(pl.Int64) > 0)
        assert with_runs.height >= 40 and with_runs["value_bps"].null_count() == 0
        assert (with_runs["value_bps"].cast(pl.Float64) >= 1.0).all()
        assert rl.filter(pl.col("n_active") == "0")["value_bps"].null_count() == \
            rl.filter(pl.col("n_active") == "0").height
        others = rb.filter(~pl.col("row").is_in(["R5", "RL"]) & (pl.col("n_active").cast(pl.Int64) > 0))
        assert others["value_bps"].null_count() == 0 and others["se_bps"].null_count() == 0

        main = study.csv("oos/doge_2023.csv")
        for r in main.iter_rows(named=True):
            cell = rb.filter((pl.col("bar") == r["bar"]) & (pl.col("horizon") == r["horizon"])
                             & (pl.col("factor") == r["factor"]))
            k0 = cell.filter((pl.col("row") == "R4") & (pl.col("parameter") == "k=0"))
            assert k0["n_active"].item() == r["n_active"]
            if r["net_bps"] is not None:
                assert float(k0["value_bps"].item()) == pytest.approx(float(r["net_bps"]), rel=1e-12)
            q = cell.filter(pl.col("row") == "R3")
            assert q["n_active"].cast(pl.Int64).sum() == int(r["n_active"])
            r8 = cell.filter(pl.col("row") == "R8")
            if r["net_funding_bps"] is not None:
                assert float(r8["value_bps"].item()) == pytest.approx(float(r["net_funding_bps"]), rel=1e-12)

    @pytest.mark.parametrize("set_name", ["doge_2023", "doge_2024"])
    def test_main_statistics_are_recomputed_independently(self, study, set_name):
        """Every row's |A|, shares, gross, net, SE, t and one-sided p from raw arrays (§6.4-§6.8)."""
        w, params = WINDOWS[set_name], study.params
        df = study.csv(f"oos/{set_name}.csv")
        data = P.load_set_data(set_name, study.root)
        checked = 0
        for bar in params.bars:
            frame = P.prepare_frames(data, bar, w, params)["DOGE-USDT-SWAP"]
            for r in df.filter(pl.col("bar") == bar).iter_rows(named=True):
                h, factor, d = int(r["horizon"]), r["factor"], int(r["direction"])
                c = independent_cell(frame, factor, h, d, w, params)
                st, act, s, fwd = c["st"], c["active"], c["s"], c["fwd"]
                assert int(r["n_active"]) == st["n_active"] == int(act.sum()), (bar, h, factor)
                m = c["ev"] & np.isfinite(fwd)
                assert int(r["n_eval"]) == int(m.sum())
                if m.any():
                    assert float(r["share_top"]) == pytest.approx((s[m] == 1).mean(), rel=1e-12)
                    assert float(r["share_bottom"]) == pytest.approx((s[m] == -1).mean(), rel=1e-12)
                if st["n_active"] == 0:
                    assert r["net_bps"] is None and r["status"] == "untestable"
                    assert r["p_one_sided"] == "1.0"
                    continue
                assert float(r["gross_bps"]) == pytest.approx(np.mean(d * s[act] * fwd[act]) * 1e4, rel=1e-9, abs=1e-9)
                assert float(r["net_bps"]) == pytest.approx(st["mu"] * 1e4, rel=1e-9)
                assert float(r["se_bps"]) == pytest.approx(st["se"] * 1e4, rel=1e-9)
                assert r["se_method"] == st["se_method"]
                assert float(r["t_hh"]) == pytest.approx(st["t"], rel=1e-9)
                testable = factor not in frame.missing_inputs and st["n_active"] >= params.min_active_trades
                assert (r["status"] != "untestable") == testable
                if testable:
                    assert float(r["p_one_sided"]) == pytest.approx(p_upper(st["t"]), rel=1e-9, abs=1e-300)
                else:
                    assert r["p_one_sided"] == "1.0"
                checked += 1
        assert checked >= 40

    def test_r6_is_tied_to_the_main_file(self, study):
        # R6 at c = 0 is the gross mean, at c it is net + 10 - c, and the SE
        # does not depend on c: the robustness rows use the cell's own s, fwd
        # and d (a d = +1 here would flip every short cell's sign).
        rb = study.csv("oos/robustness_doge_2023.csv").filter(pl.col("row") == "R6")
        main = study.csv("oos/doge_2023.csv")
        r6 = {(r["bar"], r["horizon"], r["factor"], r["parameter"]): r for r in rb.iter_rows(named=True)}
        checked = n_short = 0
        for r in main.iter_rows(named=True):
            if r["net_bps"] is None or r["se_bps"] is None:
                continue
            key = (r["bar"], r["horizon"], r["factor"])
            net, se = float(r["net_bps"]), float(r["se_bps"])
            assert float(r6[key + ("c=0",)]["value_bps"]) == pytest.approx(float(r["gross_bps"]), abs=1e-9)
            for c in (0, 4, 7, 11):
                row = r6[key + (f"c={c}",)]
                assert float(row["value_bps"]) == pytest.approx(net + 10 - c, abs=1e-9)
                assert float(row["se_bps"]) == pytest.approx(se, rel=1e-9)
                assert row["n_active"] == r["n_active"]
            checked += 1
            n_short += r["direction"] == "-1"
        assert checked >= 40 and n_short >= 1

    def test_robustness_rows_are_recomputed_for_a_short_cell(self, study):
        """R1 p, R2, R3, R4 k=5, R5, R7, R8, R10 and RL of one d = -1, h = 5 cell from raw arrays."""
        params, w = study.params, WINDOWS["doge_2023"]
        main = study.csv("oos/doge_2023.csv")
        cand = main.filter((pl.col("horizon") == "5") & (pl.col("direction") == "-1")
                           & (pl.col("factor") != "funding_cost_pressure")
                           & (pl.col("n_active").cast(pl.Int64) >= 300)).sort("bar")   # 15m first
        assert cand.height >= 1, "no short h = 5 cell to check"
        r = cand.row(0, named=True)
        bar, h, factor, d = r["bar"], 5, r["factor"], -1
        frame = P.prepare_frames(P.load_set_data("doge_2023", study.root), bar, w, params)["DOGE-USDT-SWAP"]
        c = independent_cell(frame, factor, h, d, w, params)
        f, s, fwd, net, ev, st = c["f"], c["s"], c["fwd"], c["net"], c["ev"], c["st"]
        ts, B, cost = frame.ts, frame.bar_ms, params.cost_bps
        start, end = w.start_ms, w.end_ms
        rb = study.csv("oos/robustness_doge_2023.csv").filter(
            (pl.col("bar") == bar) & (pl.col("horizon") == "5") & (pl.col("factor") == factor))
        rows = {(x["row"], x["parameter"]): x for x in rb.iter_rows(named=True)}

        def same(row, x, value_key="mu"):
            assert float(row["value_bps"]) == pytest.approx(x[value_key] * 1e4, rel=1e-9, abs=1e-12)
            assert float(row["se_bps"]) == pytest.approx(x["se"] * 1e4, rel=1e-9)
            assert float(row["t_hh"]) == pytest.approx(x["t"], rel=1e-9)
            assert int(row["n_active"]) == x["n_active"]

        # R1: the cell's own t against its own rotations, with d = -1 and c.
        offsets = P.placebo_offset_table(w, params)[bar]
        placebo = S.circular_placebo_t(s, fwd, d, cost, ts, start, end, B, h, offsets)
        assert float(r["p_placebo"]) == pytest.approx(S.placebo_p(st["t"], placebo), rel=1e-12)
        # R2: v1's statistic on the evaluation bars only.
        assert float(r["t_sqrt_h"]) == pytest.approx(
            S.v1_t_sqrt_h(f, fwd, h, quantiles=params.quantiles, mask=ev), rel=1e-12)
        # R3: quarters of the net at the cell's own h.
        for q in S.by_quarter(net, ts, start, end, B, h, params.quarter_bounds_ms):
            same(rows[("R3", f"Q{q['quarter']}")], q)
        # R4 k = 5: the five most volatile days, not the 6th-10th.
        worst5 = S.volatile_days(frame.minute_ts, frame.minute_close, start, end, 5)
        same(rows[("R4", "k=5")], S.leave_out_volatile_days(net, ts, start, end, B, h, worst5))
        # R5: the staggered portfolio pays the cell's cost.
        tw = S.turnover_aware(s, d, frame.close, ts, start, end, B, h, cost_bps=cost)
        r5 = rows[("R5", f"c={cost:g}")]
        assert float(r5["value_bps"]) == pytest.approx(tw["value_bps"], rel=1e-9)
        assert int(r5["n_active"]) == int((ev & (s != 0)).sum())
        # R7: winsorised forward returns, not the raw ones.
        wf = S.winsorized_fwd(fwd, ts, start, end, B, h, params.winsorize_pct)
        r7 = S.cell_statistic(S.per_trade_net(s, wf, d, cost), ts, start, end, B, h)
        assert r7["mu"] != st["mu"]
        same(rows[("R7", "p1_p99")], r7)
        # R8: funding on the net (not the gross), on Appendix A's fixed 8 h grid.
        spacing = SPEC["data"]["funding"]["expected_spacing_hours"] * 3_600_000
        adj = S.funding_adjusted_net(net, s, d, ts, B, h, frame.funding_ts, frame.funding_rate, spacing_ms=spacing)
        r8 = S.cell_statistic(adj, ts, start, end, B, h)
        same(rows[("R8", "funding_included")], r8)
        assert float(r["net_funding_bps"]) == pytest.approx(r8["mu"] * 1e4, rel=1e-9)
        # R10: expanding quintiles with the study's burn-in.
        for q in S.quintile_means(f, fwd, ts, start, end, B, h, params.min_prior, params.monotonicity_buckets):
            row = rows[("R10", f"Q{q['quintile']}")]
            assert int(row["n_active"]) == q["n_active"]
            if q["n_active"]:
                same(row, q)
        # RL: §7's mean run length of the cell's own s on the evaluable bars.
        rl_len, rl_runs = S.mean_run_length(s, ts, start, end, B, h)
        rl = rows[("RL", "mean_run_length")]
        assert int(rl["n_active"]) == rl_runs > 0
        assert float(rl["value_bps"]) == pytest.approx(rl_len, rel=1e-12)
        assert rl["se_bps"] is None and rl["t_hh"] is None

    def test_r10_uses_the_studys_burn_in(self, study):
        # On the fixture every evaluation bar has more than 500 finite prior
        # values, so R10's min_prior cannot show there. With a burn-in longer
        # than the warm-up, the first evaluation bars have no quintile edges.
        w = WINDOWS["doge_2023"]
        params = replace(study.params, min_prior=WARM * 96 + 300)
        frame = P.prepare_frames(P.load_set_data("doge_2023", study.root), "15m", w, params)["DOGE-USDT-SWAP"]
        out = P.evaluate_cell(frame, "rsi_14", 1, w, params, d=1, robustness=True,
                              placebo_offsets=np.array([500, 1000]))
        r10 = [x for x in out["robust_rows"] if x["row"] == "R10"]
        want = S.quintile_means(frame.factors["rsi_14"], frame.fwd(1), frame.ts, w.start_ms, w.end_ms,
                                frame.bar_ms, 1, params.min_prior, params.monotonicity_buckets)
        assert [x["n_active"] for x in r10] == [q["n_active"] for q in want]
        loose = S.quintile_means(frame.factors["rsi_14"], frame.fwd(1), frame.ts, w.start_ms, w.end_ms,
                                 frame.bar_ms, 1, 0, params.monotonicity_buckets)
        assert sum(q["n_active"] for q in loose) > sum(q["n_active"] for q in want)

    def test_r9_full_window_ic_and_its_own_holm_family(self, study):
        df = study.csv("oos/doge_2023.csv")
        ic_p = floats(df, "ic_p")
        # Undefined IC p-values enter R9's family of 60 as 1.
        np.testing.assert_allclose(floats(df, "ic_p_holm"), S.holm(np.where(np.isfinite(ic_p), ic_p, 1.0)),
                                   rtol=1e-12)
        assert np.all(np.isfinite(ic_p[floats(df, "n_eval") > 300]))
        # The planted factor predicts: a strongly positive IC at 15m, h = 1.
        row = df.filter((pl.col("bar") == "15m") & (pl.col("horizon") == "1")
                        & (pl.col("factor") == "basis_z_score")).row(0, named=True)
        assert float(row["ic"]) > 0.1 and float(row["ic_p_holm"]) < 1e-6

        # The file's IC is stats.full_window_ic on the cell's own bars.
        w = WINDOWS["doge_2023"]
        frame = P.prepare_frames(P.load_set_data("doge_2023", study.root), "15m", w, study.params)["DOGE-USDT-SWAP"]
        for h, factor in ((1, "basis_z_score"), (5, "rsi_14")):
            want = S.full_window_ic(frame.factors[factor], frame.fwd(h), frame.ts, w.start_ms, w.end_ms,
                                    frame.bar_ms, h)
            got = df.filter((pl.col("bar") == "15m") & (pl.col("horizon") == str(h))
                            & (pl.col("factor") == factor)).row(0, named=True)
            assert float(got["ic"]) == pytest.approx(want["ic"], rel=1e-12)
            assert float(got["ic_t"]) == pytest.approx(want["t"], rel=1e-12)
            assert float(got["ic_p"]) == pytest.approx(want["p"], rel=1e-9, abs=1e-300)

    def test_r8_count_goes_to_the_robustness_file(self, study):
        rb = study.csv("oos/robustness_doge_2023.csv").filter(pl.col("row") == "R8")
        main = study.csv("oos/doge_2023.csv")
        for r, m in zip(rb.iter_rows(named=True), main.iter_rows(named=True)):
            assert (r["bar"], r["horizon"], r["factor"]) == (m["bar"], m["horizon"], m["factor"])
            assert r["parameter"] == "funding_included"
            assert 0 <= int(r["n_active"]) <= int(m["n_active"])
            if int(r["n_active"]) > 1:
                assert r["se_bps"] is not None

    def test_placebo_resolution_and_family_p(self, study):
        df = study.csv("oos/doge_2023.csv")
        k_prime = study.params.placebo_k  # no duplicate offsets on these grids
        p = floats(df, "p_placebo")
        fam = floats(df, "p_placebo_family")
        ok = np.isfinite(p)
        assert ok.any()
        assert np.all(p[ok] >= 1 / (1 + k_prime)) and np.all(p[ok] <= 1)
        # A family-wise p is never smaller than the cell's own.
        both = ok & np.isfinite(fam)
        assert np.all(fam[both] >= p[both] - 1e-12)
        untestable = (df["status"] == "untestable").to_numpy()
        assert np.all(np.isnan(fam[untestable]))


# ---------------------------------------------------------------- (c) untestable cells

class TestUntestable:
    def test_missing_funding_cells_are_untestable_with_p_one(self, study):
        df = study.csv("oos/doge_2024.csv")
        assert df.height == 60
        fc = df.filter(pl.col("factor") == "funding_cost_pressure")
        assert fc.height == 6
        assert set(fc["status"].to_list()) == {"untestable"}
        assert set(fc["p_one_sided"].to_list()) == {"1.0"}
        assert set(fc["n_active"].to_list()) == {"0"}

    def test_untestable_cells_stay_in_holms_m(self, study):
        df = study.csv("oos/doge_2024.csv")
        p = floats(df, "p_one_sided")
        holm = floats(df, "p_holm")
        assert len(p) == 60
        np.testing.assert_allclose(holm, S.holm(p), rtol=1e-12)
        testable = (df["status"] != "untestable").to_numpy()
        assert (~testable).sum() >= 6
        # Dropping the untestable cells would shrink m and change some adjusted p.
        assert not np.allclose(holm[testable], S.holm(p[testable]), rtol=1e-12, atol=0)
        assert set(df["status"].to_list()) <= {"rejected", "not_rejected", "untestable"}

    def test_family_function_puts_p_one_for_untestable_and_nan_t(self):
        params = P.StudyParams.from_spec()
        cells = [
            {"p": 0.001, "testable": True},
            {"p": 0.002, "testable": False},     # |A| < 300 or missing input
            {"p": math.nan, "testable": True},   # SE = 0, t undefined
            {"p": 0.5, "testable": True},
        ]
        fam = P._family(cells, params)
        np.testing.assert_array_equal(fam["p"], [0.001, 1.0, 1.0, 0.5])
        np.testing.assert_allclose(fam["holm"], S.holm([0.001, 1.0, 1.0, 0.5]))


# ---------------------------------------------------------------- (d) pooled majors

class TestPooledMajors:
    def test_rows_and_pooled_statistics(self, study):
        df = study.csv("oos/majors_2023.csv")
        assert df.height == 4 * 60
        insts = df["instrument"].to_list()
        for i in ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", P.POOLED):
            assert insts.count(i) == 60
        pooled = df.filter(pl.col("instrument") == P.POOLED)
        per = df.filter(pl.col("instrument") != P.POOLED)
        assert set(per["status"].to_list()) == {"descriptive"}
        assert set(pooled["status"].to_list()) <= {"rejected", "not_rejected", "untestable"}
        p = floats(pooled, "p_one_sided")
        np.testing.assert_allclose(floats(pooled, "p_holm"), S.holm(p), rtol=1e-12)

        nets = {}
        for r in per.iter_rows(named=True):
            nets.setdefault((r["bar"], r["horizon"], r["factor"]), []).append(r["net_bps"])
        checked = 0
        for r in pooled.iter_rows(named=True):
            vals = nets[(r["bar"], r["horizon"], r["factor"])]
            if r["net_bps"] is not None and all(v is not None for v in vals):
                assert float(r["net_bps"]) == pytest.approx(np.mean([float(v) for v in vals]), rel=1e-9)
                checked += 1
        assert checked >= 40

    def test_pooled_statistics_are_recomputed_independently(self, study):
        """Every POOLED row's SE, t and p from the instruments' raw arrays (§7 F_majors)."""
        w, params = WINDOWS["majors_2023"], study.params
        df = study.csv("oos/majors_2023.csv").filter(pl.col("instrument") == P.POOLED)
        data = P.load_set_data("majors_2023", study.root)
        insts = SPEC["sets"]["majors_2023"]["inst"]
        checked = 0
        for bar in params.bars:
            frames = P.prepare_frames(data, bar, w, params)
            for r in df.filter(pl.col("bar") == bar).iter_rows(named=True):
                h, factor, d = int(r["horizon"]), r["factor"], int(r["direction"])
                cells = [independent_cell(frames[i], factor, h, d, w, params) for i in insts]
                st = S.pooled_statistic([(c["net"], frames[i].ts) for c, i in zip(cells, insts)],
                                        w.start_ms, w.end_ms, P.bar_ms(bar), h)
                testable = all(factor not in frames[i].missing_inputs
                               and c["st"]["n_active"] >= params.min_active_trades for c, i in zip(cells, insts))
                assert (r["status"] != "untestable") == testable
                if not testable:
                    assert r["p_one_sided"] == "1.0"
                    continue
                assert float(r["net_bps"]) == pytest.approx(np.mean([c["st"]["mu"] for c in cells]) * 1e4, rel=1e-9)
                assert float(r["se_bps"]) == pytest.approx(st["se"] * 1e4, rel=1e-9)
                assert float(r["t_hh"]) == pytest.approx(st["t"], rel=1e-9)
                assert float(r["p_one_sided"]) == pytest.approx(st["p"], rel=1e-9, abs=1e-300)
                assert float(r["p_one_sided"]) == pytest.approx(p_upper(float(r["t_hh"])), rel=1e-9, abs=1e-300)
                checked += 1
        assert checked >= 40

    def test_pooled_cell_untestable_if_any_instrument_is(self, study):
        df = study.csv("oos/majors_2023.csv")
        basis = df.filter(pl.col("factor") == "basis_z_score")
        sol = basis.filter(pl.col("instrument") == "SOL-USDT-SWAP")
        assert set(sol["n_active"].to_list()) == {"0"}                 # no spot leg
        btc = basis.filter(pl.col("instrument") == "BTC-USDT-SWAP")
        assert min(int(v) for v in btc["n_active"].to_list()) >= 300   # testable on its own
        pooled = basis.filter(pl.col("instrument") == P.POOLED)
        assert set(pooled["status"].to_list()) == {"untestable"}
        assert set(pooled["p_one_sided"].to_list()) == {"1.0"}


# ---------------------------------------------------------------- (e) §10 fill rules

# Restated from §10 independently of research/pipeline.py, so a slip in the
# pipeline's table cannot hide behind the same slip here.
EVERY_ROW = {"direction", "direction_flagged", "gross_bps", "net_bps", "se_bps", "se_method",
             "status", "code_commit", "manifest_sha256"}
NOT_POOLED = {"n_eval", "n_active", "share_top", "share_bottom"}
TESTED = {"t_hh", "p_one_sided", "p_holm", "q_bh", "q_by"}
FWD_ONLY = {"ci95_lo_bps", "ci95_hi_bps"}
PRIMARY_ONLY = {"ub95_net_bps", "ub95_family_net_bps", "mde_bps", "mde_realised_bps",
                "p_placebo", "p_placebo_family", "t_sqrt_h", "net_funding_bps",
                "ic", "ic_t", "ic_p", "ic_p_holm", "replicated", "consistent_on_majors"}
KEYS = {"instrument", "bar", "horizon", "factor"}


def expected_filled(set_name, inst):
    cols = set(EVERY_ROW) | KEYS
    if inst != "POOLED":
        cols |= NOT_POOLED
    if set_name in ("doge_2023", "doge_2024") or (set_name == "majors_2023" and inst == "POOLED"):
        cols |= TESTED
    if set_name == "doge_fwd":
        cols |= FWD_ONLY
    if set_name == "doge_2023":
        cols |= PRIMARY_ONLY
    return cols


class TestFillRules:
    @pytest.mark.parametrize("set_name", ["doge_2023", "doge_2024", "majors_2023", "doge_fwd"])
    def test_blank_exactly_where_section_10_says(self, study, set_name):
        df = study.csv(f"oos/{set_name}.csv")
        assert tuple(df.columns) == P.OOS_COLUMNS
        strict = 0
        for r in df.iter_rows(named=True):
            want = expected_filled(set_name, r["instrument"])
            blank = {c for c, v in r.items() if v is None}
            # Never filled outside the rules ...
            assert blank >= set(df.columns) - want, (set_name, r["instrument"], r["factor"])
            # ... and always filled within them where the cell is defined.
            assert not blank & (EVERY_ROW - {"gross_bps", "net_bps", "se_bps", "se_method"})
            if r["instrument"] == "POOLED":
                defined = r["status"] != "untestable"
            else:
                # The primary's R2 and R9 need a few hundred pairs; elsewhere
                # every filled column is defined once a trade exists.
                defined = int(r["n_active"]) >= (300 if set_name == "doge_2023" else 2)
            if defined:
                assert blank == set(df.columns) - want, (set_name, r["instrument"], r["factor"], blank & want)
                strict += 1
        assert strict >= 30

    def test_values_are_consistent(self, study):
        for set_name in ("doge_2023", "doge_2024", "majors_2023", "doge_fwd"):
            df = study.csv(f"oos/{set_name}.csv")
            g, n = floats(df, "gross_bps"), floats(df, "net_bps")
            ok = np.isfinite(g)
            np.testing.assert_allclose(g[ok] - n[ok], SPEC["cost_round_trip_bps"], atol=1e-9)
            assert set(df["code_commit"].to_list()) == {"c" * 40}
            assert set(df["manifest_sha256"].to_list()) == {f"sha-of-{set_name}"}
        fwd = study.csv("oos/doge_fwd.csv")
        lo, hi, n, se = (floats(fwd, c) for c in ("ci95_lo_bps", "ci95_hi_bps", "net_bps", "se_bps"))
        ok = np.isfinite(se)
        np.testing.assert_allclose(lo[ok], n[ok] - 1.96 * se[ok], rtol=1e-12)
        np.testing.assert_allclose(hi[ok], n[ok] + 1.96 * se[ok], rtol=1e-12)
        prim = study.csv("oos/doge_2023.csv")
        n, se = floats(prim, "net_bps"), floats(prim, "se_bps")
        np.testing.assert_allclose(floats(prim, "ub95_net_bps"), n + 1.645 * se, rtol=1e-12)
        np.testing.assert_allclose(floats(prim, "ub95_family_net_bps"), n + 3.144 * se, rtol=1e-12)
        np.testing.assert_allclose(floats(prim, "mde_realised_bps"), (3.144 + 0.842) * se, rtol=1e-12)
        mde_is = floats(study.csv("insample_v2.csv"), "mde_bps")
        np.testing.assert_allclose(floats(prim, "mde_bps"), mde_is, rtol=1e-12)
        # Directions are the in-sample ones on every set.
        d_is = study.csv("insample_v2.csv")["direction"].to_list()
        assert prim["direction"].to_list() == d_is
        maj = study.csv("oos/majors_2023.csv")
        for inst in ("BTC-USDT-SWAP", P.POOLED):
            assert maj.filter(pl.col("instrument") == inst)["direction"].to_list() == d_is


# ---------------------------------------------------------------- inputs

class TestInputs:
    def test_doge_fwd_reads_insample_warmup_then_its_own_files(self, study):
        data = P.load_set_data("doge_fwd", study.root, is_end_ms=WINDOWS["IS"].end_ms)
        perp = data["perp"]["DOGE-USDT-SWAP"]
        split = WINDOWS["IS"].end_ms
        assert WINDOWS["doge_fwd"].start_ms == split          # Appendix A's shape
        assert perp["timestamp"].min() < split <= perp["timestamp"].max()
        assert np.all(np.diff(perp["timestamp"].to_numpy()) > 0)
        # The in-sample file runs past the split at other prices; every row
        # doge_fwd evaluates must come from its own files nonetheless.
        own = pl.read_parquet(study.root / "data/history/oos/doge_fwd/DOGE-USDT-SWAP" / P.CANDLE_FILE)
        insample = pl.read_parquet(study.root / SPEC["insample_dir"] / P.CANDLE_FILE)
        tail = insample.filter(pl.col("timestamp") >= split)
        assert tail.height > 1000
        overlap = tail.join(own, on="timestamp", suffix="_own")
        assert overlap.height == tail.height and (overlap["close"] != overlap["close_own"]).all()
        evaluated = perp.filter(pl.col("timestamp") >= split)
        assert evaluated["timestamp"].to_list() == own["timestamp"].to_list()
        np.testing.assert_array_equal(evaluated["close"].to_numpy(), own["close"].to_numpy())
        warm = perp.filter(pl.col("timestamp") < split)
        np.testing.assert_array_equal(warm["close"].to_numpy(),
                                      insample.filter(pl.col("timestamp") < split)["close"].to_numpy())
        for kind, rel in (("spot", "DOGE-USDT/" + P.CANDLE_FILE), ("funding", "DOGE-USDT-SWAP/" + P.FUNDING_FILE)):
            got = data[kind]["DOGE-USDT-SWAP"].filter(pl.col("timestamp") >= split)
            want = pl.read_parquet(study.root / "data/history/oos/doge_fwd" / rel)
            assert got["timestamp"].to_list() == want["timestamp"].to_list(), kind
        rel = {p.relative_to(study.root).as_posix() for p in data["files"]}
        assert f"{SPEC['insample_dir']}/candles_1m.parquet" in rel
        assert "data/history/oos/doge_fwd/DOGE-USDT-SWAP/candles_1m.parquet" in rel
        assert "data/history/is_supp/DOGE-USDT/candles_1m.parquet" in rel

    def test_missing_series_are_none_and_perp_is_required(self, study, tmp_path):
        data = P.load_set_data("doge_2024", study.root)
        assert data["funding"]["DOGE-USDT-SWAP"] is None
        assert data["spot"]["DOGE-USDT-SWAP"] is not None
        with pytest.raises(P.PipelineError, match="required input"):
            P.load_set_data("doge_2024", tmp_path)

    def test_verify_inputs(self, tmp_path):
        root = tmp_path
        base = root / "data/history/oos/doge_2024"
        perp = base / "DOGE-USDT-SWAP" / P.CANDLE_FILE
        spot = base / "DOGE-USDT" / P.CANDLE_FILE
        for p in (perp, spot):
            p.parent.mkdir(parents=True)
            p.write_bytes(p.name.encode() + p.parent.name.encode())
        assert any("manifest" in m and "does not exist" in m for m in P.verify_inputs("doge_2024", root))

        def manifest(entries):
            m = root / P.oos_manifest_path("doge_2024")
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(json.dumps({"series": entries}))

        good = [{"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)} for p in (perp, spot)]
        manifest(good)
        assert P.verify_inputs("doge_2024", root) == []

        manifest(good[:1])                                        # spot read but not listed
        assert any("no manifest lists it" in m for m in P.verify_inputs("doge_2024", root))

        manifest(good)
        spot.write_bytes(b"changed")                              # hash mismatch
        assert any("differs" in m for m in P.verify_inputs("doge_2024", root))

    def test_verify_insample_against_appendix_a(self, study):
        problems = P.verify_inputs("IS", study.root)
        assert any("Appendix A" in m for m in problems)           # synthetic candles, real hashes
        assert any("manifest_is_supp.json does not exist" in m for m in problems)

    def test_insample_rerun_needs_one_instrument(self, study):
        data = P.load_set_data("majors_2023", study.root)
        with pytest.raises(P.PipelineError):
            P.run_insample(study.root, data=data, window=WINDOWS["majors_2023"], params=study.params)


def test_placebo_family_uses_offset_indices_new_on_both_grids():
    # 70-day window, 30-day minimum shift, K = 2000: the 1H grid has only 241
    # distinct offsets and the 15m grid 961, so most j repeat on one grid.
    t0 = T0
    w = P.Window(t0, t0 + 10 * DAY, t0 + 80 * DAY)
    params = replace(P.StudyParams.from_spec(), placebo_k=2000, placebo_min_shift_days=30)
    table = P.placebo_offset_table(w, params)
    cols = P.placebo_family_columns(w, params)
    j = np.arange(2000)
    raw, new = {}, {}
    for bar in ("15m", "1H"):
        B = P.bar_ms(bar)
        n_grid, d30 = 70 * DAY // B, 30 * DAY // B
        np.testing.assert_array_equal(table[bar], S.placebo_offsets(n_grid, d30, 2000))   # per-cell K'
        raw[bar] = d30 + np.round(j * (n_grid - 2 * d30) / 1999).astype(np.int64)
        new[bar] = np.r_[True, raw[bar][1:] != raw[bar][:-1]]
    assert (len(table["1H"]), len(table["15m"])) == (241, 961)
    kept = np.flatnonzero(new["15m"] & new["1H"])
    for bar in ("15m", "1H"):
        assert len(cols[bar]) == len(kept)
        # Column m on every grid is the rotation by the same j.
        np.testing.assert_array_equal(table[bar][cols[bar]], raw[bar][kept])


def test_funding_limits_are_appendix_a_constants_whatever_the_series_spacing(study, monkeypatch):
    # The synthetic funding is 8-hourly, where a spacing measured from the
    # series and Appendix A's constant agree. With 4-hourly settlements (one
    # missing) R8's grid must stay 8 h, the staleness mask must stay 8 h, and
    # the other spacing must be reported (§4.2).
    w = WINDOWS["doge_2023"]
    data = P.load_set_data("doge_2023", study.root)
    f8 = data["funding"]["DOGE-USDT-SWAP"]
    f4 = pl.concat([f8, f8.with_columns(pl.col("timestamp") + H8 // 2)]).sort("timestamp")
    gone = int(f4["timestamp"][f4.height // 2])
    f4 = f4.filter(pl.col("timestamp") != gone)
    data4 = dict(data, funding={"DOGE-USDT-SWAP": f4})
    reported = []
    real = P.irregular_settlements
    monkeypatch.setattr(P, "irregular_settlements",
                        lambda ts, label="funding": reported.append(real(ts, label)) or reported[-1])
    frame = P.prepare_frames(data4, "1H", w, study.params)["DOGE-USDT-SWAP"]
    assert frame.funding_spacing_ms == study.params.funding_spacing_ms == H8
    assert reported and all(sp != H8 for _, sp in reported[-1]) and len(reported[-1]) > 100
    fcp = frame.factors["funding_cost_pressure"]
    after_gap = (frame.ts >= gone) & (frame.ts < gone + H8 // 2)
    assert after_gap.any() and np.isfinite(fcp[after_gap]).all()   # 4-8 h old: still fresh
    # No bar is 8 h past its latest settlement, so nothing in the window is masked.
    assert np.isfinite(fcp[frame.ts >= w.start_ms]).all()


def test_build_bars_refuses_a_window_off_the_htf_grid(study):
    data = P.load_set_data("doge_2023", study.root)
    w = WINDOWS["doge_2023"]
    with pytest.raises(ValueError, match="boundary"):
        P.build_bars(data, "1H", P.Window(w.warmup_start_ms + 3_600_000, w.start_ms, w.end_ms))


# ---------------------------------------------------------------- boundaries

def _hand_frame(s, window, missing=frozenset()):
    """An InstrumentFrame on a gap-free 15m grid whose signal is ``s`` (cached, so edges are not under test)."""
    rng = np.random.default_rng(41)
    ts = np.arange(window.warmup_start_ms, window.end_ms, M15, dtype=np.int64)
    close = 0.1 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(ts))))
    return P.InstrumentFrame(
        instrument="DOGE-USDT-SWAP", bar="15m", ts=ts, close=close, factors={"f": np.zeros(len(ts))},
        missing_inputs=frozenset(missing), minute_ts=ts, minute_close=close, funding_ts=None,
        funding_rate=None, funding_spacing_ms=None, min_prior=500, edges_pct=(20.0, 80.0),
        _signal={"f": np.asarray(s, dtype=np.int8)},
    )


@pytest.mark.parametrize("n_signals, testable", [(300, True), (299, False)])
def test_three_hundred_active_bars_make_a_cell_testable(n_signals, testable):
    # §6.8: a cell is testable with |A| >= 300, so exactly 300 is enough and
    # 299 is not. Signals in the warm-up and on the last, non-evaluable bars
    # must not count towards |A|.
    w = P.Window(T0, T0 + DAY, T0 + 6 * DAY)
    params = P.StudyParams.from_spec()
    ts = np.arange(w.warmup_start_ms, w.end_ms, M15, dtype=np.int64)
    ev = np.flatnonzero(S.evaluable(ts, w.start_ms, w.end_ms, M15, 1))
    s = np.zeros(len(ts), dtype=np.int8)
    s[ev[:n_signals]] = np.where(np.arange(n_signals) % 2 == 0, 1, -1)
    s[: ev[0]] = 1                       # warm-up
    s[ev[-1] + 1:] = 1                   # exit past the window's end
    c = P.evaluate_cell(_hand_frame(s, w), "f", 1, w, params, d=1)
    assert c["n_active"] == n_signals and c["testable"] is testable
    assert params.min_active_trades == 300
    # A missing input makes the cell untestable whatever |A| is.
    c = P.evaluate_cell(_hand_frame(s, w, missing={"f"}), "f", 1, w, params, d=1)
    assert c["n_active"] == n_signals and c["testable"] is False


def test_holm_p_exactly_at_alpha_is_rejected(monkeypatch, tmp_path):
    # §7 rejects where the Holm-adjusted p <= alpha. p = 2^-10 makes Holm's
    # first step 60 * 2^-10 exactly, so with alpha set to that value the cell
    # sits on the boundary; the next cell is clearly above it. The cells are
    # canned so that only the decision rule is under test.
    alpha = 60 * 2.0 ** -10
    params = replace(P.StudyParams.from_spec(), alpha=alpha)
    cells = params.cells()
    p_of = {cells[0]: 2.0 ** -10, cells[1]: 2.0 ** -9}

    def fake_prepare(data, bar, window, params_):
        return {i: bar for i in data["instruments"]}

    def fake_evaluate(frame, factor, h, window, params_, d=None, **kw):
        return {"direction": d, "n_eval": 1000, "n_active": 500, "share_top": 0.2, "share_bottom": 0.2,
                "gross": 0.002, "mu": 0.001, "se": 0.0003, "se_method": "hh", "t": 3.3,
                "p": p_of.get((frame, h, factor), 0.5), "testable": True, "ub95": 0.0, "ub95_family": 0.0,
                "ci95_lo": 0.0, "ci95_hi": 0.0, "net": None, "ts": None}

    monkeypatch.setattr(P, "prepare_frames", fake_prepare)
    monkeypatch.setattr(P, "evaluate_cell", fake_evaluate)
    main, _ = P.run_oos_set("doge_2024", {k: (1, False) for k in cells}, root=tmp_path,
                            data={"instruments": ("DOGE-USDT-SWAP",)}, window=WINDOWS["doge_2024"], params=params)
    rows = main.to_dicts()
    assert rows[0]["p_holm"] == alpha and rows[0]["status"] == "rejected"
    assert rows[1]["p_holm"] > alpha and rows[1]["status"] == "not_rejected"
    assert {r["status"] for r in rows[2:]} == {"not_rejected"}


class TestSecondaryStatuses:
    """read_secondary_statuses refuses a file that would mislabel the primary set (§7, §10)."""

    @staticmethod
    def _copy(study, tmp_path, edit=None, which=None):
        for name in ("doge_2024", "majors_2023"):
            df = study.csv(f"oos/{name}.csv")
            if name == which:
                df = edit(df)
            (tmp_path / f"{name}.csv").write_text(df.write_csv())
        return tmp_path

    def test_reads_the_written_files(self, study, tmp_path):
        out = P.read_secondary_statuses(self._copy(study, tmp_path), study.params)
        rep = {(r["bar"], int(r["horizon"]), r["factor"]): r["status"]
               for r in study.csv("oos/doge_2024.csv").iter_rows(named=True)}
        assert out[P.ROLE_REPLICATION] == rep
        assert set(out[P.ROLE_POOLED]) == set(study.params.cells())

    def test_duplicate_cell(self, study, tmp_path):
        root = self._copy(study, tmp_path, lambda df: pl.concat([df, df[:1]]), "doge_2024")
        with pytest.raises(P.PipelineError, match="appears twice"):
            P.read_secondary_statuses(root, study.params)

    def test_missing_cell(self, study, tmp_path):
        root = self._copy(study, tmp_path, lambda df: df[1:], "doge_2024")
        with pytest.raises(P.PipelineError, match="cover exactly"):
            P.read_secondary_statuses(root, study.params)
        pooled_gone = lambda df: df.filter(~((pl.col("instrument") == P.POOLED)  # noqa: E731
                                             & (pl.col("factor") == "rsi_14") & (pl.col("bar") == "1H")))
        root = self._copy(study, tmp_path, pooled_gone, "majors_2023")
        with pytest.raises(P.PipelineError, match="cover exactly"):
            P.read_secondary_statuses(root, study.params)

    def test_descriptive_status_on_a_pooled_row(self, study, tmp_path):
        def edit(df):
            first = (pl.int_range(pl.len()) == df["instrument"].to_list().index(P.POOLED))
            return df.with_columns(
                pl.when(first).then(pl.lit("descriptive")).otherwise(pl.col("status")).alias("status"))
        root = self._copy(study, tmp_path, edit, "majors_2023")
        with pytest.raises(P.PipelineError, match="has status 'descriptive'"):
            P.read_secondary_statuses(root, study.params)


def test_r9_is_the_full_window_ic():
    # R9's mean of daily ICs was replaced before publication (§12, 2026-10-09);
    # research.stats keeps daily_ic only for the test that shows its bias.
    import inspect
    src = inspect.getsource(P)
    assert "full_window_ic" in src and "daily_ic" not in src
