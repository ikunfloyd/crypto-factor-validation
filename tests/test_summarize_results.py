"""
scripts/summarize_results.py on the committed results.

The script reads only committed files (results/, Appendix A of
docs/preregistration.md and the freeze amendment's list of maker-probe
days), so nothing here needs data/ or the network. It is run as a reader
runs it, in a fresh interpreter with PYTHONPATH unset, and its output is
checked against facts read from the committed files with the csv and json
modules rather than the script's own loader: its sections are the
comparison's own headings, in the comparison's order; the primary family's
60 cells, none rejected by Holm, and its largest t; every doge_fwd line the
comparison quotes, and each cell's μ̂ and 95% interval; every maker-probe
line, the table row by row; the in-sample counts the report quotes. The
helpers are checked on small inputs: finding the checkout from the script's
own path, rebuilding funding settlement times from a manifest entry (and
from every committed one), reading the freeze amendment's days, recomputing
summary.csv from daily.csv, and the formatting. Importing the module leaves
the interpreter's bytecode setting alone.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from research.spec import REPO_ROOT, get_spec


def _load(name):
    found = importlib.util.spec_from_file_location(f"_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(found)
    found.loader.exec_module(mod)
    return mod


sr = _load("summarize_results")
SCRIPT = REPO_ROOT / "scripts" / "summarize_results.py"
DOCUMENT = REPO_ROOT / "docs" / "preregistration-results.md"
STEP = sr.FUNDING_STEP_MS
T0 = 58_872 * STEP                      # 2023-09-24 00:00 UTC, a time on the 8-hour funding grid
LARGEST = "The largest primary cell, descriptively"
DOGE_FWD = "`doge_fwd`, descriptively (§4.1, §7)"
PROBE = "Maker probe (§11, exploratory)"
IN_SAMPLE = "In sample (results/insample_v2.csv; quoted by docs/factor-research-v2.md, §4)"
RUNS = ("parser_gaps", "touch_gaps")
# Every committed manifest with a funding series, named rather than globbed:
# the maker probe's download adds manifests of its own under results/.
FUNDING_MANIFESTS = ["results/manifest_is_supp.json", "results/oos/manifest_doge_2023.json",
                     "results/oos/manifest_doge_2024.json", "results/oos/manifest_majors_2023.json",
                     "results/oos/manifest_doge_fwd.json"]


def committed_rows(rel):
    """A committed results file as text fields, read without the script's loader."""
    with open(REPO_ROOT / rel, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def committed_json(rel):
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def values(rows, column):
    return [float(r[column]) for r in rows]


def by_cell(rel, column):
    """A results file's column by (bar, horizon, factor)."""
    return {(r["bar"], r["horizon"], r["factor"]): float(r[column]) for r in committed_rows(rel)}


def probe_summary():
    """The maker probe's summary.csv by (run, side, model, metric), in the file's order."""
    return {(r["run"], r["side"], r["model"], r["metric"]): r for r in committed_rows(sr.PROBE_SUMMARY)}


def probe_metric(summary, metric, field="mean"):
    """One metric of summary.csv by (run, side, model)."""
    return {k[:3]: float(r[field]) for k, r in summary.items() if k[3] == metric}


def run_script(cwd, script):
    """The script as a reader runs it: a fresh interpreter, with PYTHONPATH unset."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, script], cwd=cwd, env=env, capture_output=True, encoding="utf-8",
                          timeout=300)


def sections(text):
    """The output's non-empty lines under each '## ' heading; those before the first heading under ''."""
    out, key = {"": []}, ""
    for line in text.splitlines():
        if line.startswith("## "):
            key = line[3:]
            out[key] = []
        elif line:
            out[key].append(line)
    return out


@pytest.fixture(scope="module")
def run():
    """One run from the repository root, as the docstring's usage line has it."""
    return run_script(REPO_ROOT, "scripts/summarize_results.py")


@pytest.fixture(scope="module")
def out(run):
    assert run.returncode == 0, run.stderr
    return sections(run.stdout)


# ---------------------------------------------------------------- the run

class TestRun:
    def test_it_exits_with_0(self, run):
        assert run.returncode == 0, run.stderr
        assert run.stdout.startswith("# Results compared with the pre-registration\n")

    def test_its_sections_are_the_comparisons_headings_in_order(self, out):
        headings = [line[3:] for line in DOCUMENT.read_text(encoding="utf-8").splitlines() if line.startswith("## ")]
        assert headings == ["Record", "Headline (§7)", "Cells that do not pay at taker cost (§9)", "Power (§9)",
                            "Secondary families (§7)", "Robustness (§8: reported, never decisive)", LARGEST,
                            DOGE_FWD, "Data", PROBE, "Not done"]
        assert list(out) == [""] + headings + [IN_SAMPLE]

    def test_nothing_is_still_to_come(self, out):
        assert "Still to come" not in out
        assert out["Not done"] == ["- numbers this section quotes: none"]

    def test_the_working_directory_does_not_matter(self, run, tmp_path):
        other = run_script(tmp_path, str(SCRIPT))
        assert other.returncode == 0, other.stderr
        assert other.stdout == run.stdout


class TestPrimary:
    def test_sixty_cells_none_rejected_by_holm(self, out):
        rows = committed_rows(sr.PRIMARY)
        assert len(rows) == 60
        assert {r["status"] for r in rows} == {"not_rejected"}
        assert min(float(r["p_holm"]) for r in rows) > 0.05
        assert "-   cells: 60 (10 factors x bars ['15m', '1H'] x horizons [1, 5, 15])" in out[""]
        headline = out["Headline (§7)"]
        assert "- status counts: not_rejected 60" in headline
        assert "- smallest Holm p / BH q / BY q: 1.000 / 1.000 / 1.000" in headline
        assert "- confirmed / replicated / consistent_on_majors: 0 / 0 / 0" in headline

    def test_the_largest_t(self, out):
        rows = committed_rows(sr.PRIMARY)
        top = max(rows, key=lambda r: float(r["t_hh"]))
        assert (top["bar"], top["horizon"], top["factor"]) == ("1H", "5", "rolling_vol_ratio")
        assert float(top["t_hh"]) == pytest.approx(1.709, abs=5e-4)
        assert ("- largest t: +1.71 at 1H h=5 rolling_vol_ratio (μ̂ +17.6, SE 10.3); Holm's first step needs 3.144"
                in out["Headline (§7)"])
        assert "- cells with one-sided p < 0.05: 1" in out["Headline (§7)"]

    def test_the_constants_read_back_from_the_rows(self, out):
        terms = out[""]
        assert "-   gross - net on every row (the round-trip cost): 10.0 bp" in terms
        assert "-   ub95 = μ̂ + z·SE with z: 1.645" in terms
        assert "-   ub95_family = μ̂ + z·SE with z (Holm's first step for 60 cells): 3.144" in terms
        assert "-   MDE realised = (z_alpha + z_power)·SE with z_power: 0.842" in terms


class TestDogeFwd:
    def test_the_counts_and_the_interval(self, out):
        rows = committed_rows(sr.FORWARD)
        assert len(rows) == 60 and {r["status"] for r in rows} == {"descriptive"}
        # Both ends of every interval lie 1.960 SE from μ̂.
        z = ({round((float(r["ci95_hi_bps"]) - float(r["net_bps"])) / float(r["se_bps"]), 9) for r in rows}
             | {round((float(r["net_bps"]) - float(r["ci95_lo_bps"])) / float(r["se_bps"]), 9) for r in rows})
        assert z == {1.96}
        above = sum(float(r["ci95_lo_bps"]) > 0 for r in rows)
        below = sum(float(r["ci95_hi_bps"]) < 0 for r in rows)
        assert (above, below) == (0, 35)
        section = out[DOGE_FWD]
        assert "- 95% interval = μ̂ ± z·SE with z: 1.960" in section
        assert "- status counts: descriptive 60" in section
        assert f"- 95% interval above 0 / entirely below 0: {above} / {below} of 60" in section

    def test_its_window_record_and_data(self, out):
        sets = get_spec()["sets"]
        win, ins = sets["doge_fwd"], sets["IS"]
        assert win["start_ms"] == ins["end_ms"] and ins["warmup_start_ms"] <= win["warmup_start_ms"]
        manifest = committed_json(sr.MANIFESTS["doge_fwd"])
        added = committed_json(sr.MANIFESTS["maker_probe_archives"])["doge_fwd_results_commit"]
        (code,) = {r["code_commit"] for r in committed_rows(sr.FORWARD)}
        assert (code[:7], added[:7]) == ("7a8e5c6", "70c2596")          # the comparison's table
        section = out[DOGE_FWD]
        assert ("- evaluation window (Appendix A): 2026-09-23 20:00:00 to 2026-11-23 20:00:00 UTC, 61 days, "
                "from the in-sample year's end: True") in section
        assert f"- code_commit of every row: {code[:7]}" in section
        assert f"- manifest created (UTC): {manifest['created_utc']}" in section
        assert (f"- public main when its data was fetched (manifest remote_main_commit): "
                f"{manifest['remote_main_commit'][:7]}") in section
        assert (f"- the commit that added the results (doge_fwd_results_commit, the maker probe's archive "
                f"manifest): {added[:7]}") in section
        assert ("- warm-up from (Appendix A): 2026-06-25 20:00:00 UTC, inside the in-sample data "
                "(2025-09-24 00:00:00 to 2026-09-23 20:00:00): True") in section
        minutes = (win["end_ms"] - win["start_ms"]) // sr.MINUTE_MS
        slots = (win["end_ms"] - win["start_ms"]) // STEP
        candles = {e["inst_id"]: e for e in manifest["series"] if e["kind"] == "candles"}
        assert sorted(candles) == ["DOGE-USDT", "DOGE-USDT-SWAP"]        # the perpetual and its spot leg
        for inst, e in candles.items():
            assert e["rows"] == minutes == 87_840 and not e["gaps"]
            assert f"- doge_fwd {inst} candles: {minutes:,} rows, {minutes:,} minutes in the window, gaps 0" in section
        (funding,) = [e for e in manifest["series"] if e["kind"] == "funding"]
        assert funding["rows"] == slots == 183 and not funding["spacing_other_than_expected"]
        assert (f"- doge_fwd DOGE-USDT-SWAP funding: {slots} settlements, {slots} 8-hour slots in the window, "
                "one in each") in section
        assert f"- doge_fwd DOGE-USDT-SWAP off-grid stamps: none of {slots}" in section

    def test_the_cells_the_comparison_reads(self, out):
        net, lo, hi = (by_cell(sr.FORWARD, c) for c in ("net_bps", "ci95_lo_bps", "ci95_hi_bps"))
        pos = [k for k in net if net[k] > 0]
        assert len(pos) == 6 and {k[:2] for k in pos} == {("1H", "5"), ("1H", "15")}
        widths = [hi[k] - lo[k] for k in pos]
        assert min(widths) >= 36
        top = max(net, key=net.get)
        assert top == ("1H", "15", "basis_z_score")
        fast = [k for k in net if k[:2] == ("15m", "1")]
        gross = by_cell(sr.FORWARD, "gross_bps")
        assert len(fast) == 10 and max(hi[k] for k in fast) < -8.0
        section = out[DOGE_FWD]
        per_h = ", ".join(f"1H h={h} ({sum(k[1] == h for k in pos)})" for h in ("5", "15"))
        assert f"- cells with μ̂ > 0: 6: {per_h}" in section
        assert f"- their 95% intervals' width, narrowest / widest: {min(widths):.1f} / {max(widths):.1f} bp" in section
        assert (f"- largest μ̂: {net[top]:+.1f} at 1H h=15 basis_z_score, 95% interval "
                f"[{lo[top]:+.1f}, {hi[top]:+.1f}]") in section
        assert (f"- 15m, h = 1 (10 cells): gross means from / to: {min(gross[k] for k in fast):+.1f} / "
                f"{max(gross[k] for k in fast):+.1f}") in section
        assert (f"- 15m, h = 1: μ̂ from / to: {min(net[k] for k in fast):+.1f} / "
                f"{max(net[k] for k in fast):+.1f}") in section
        assert f"- 15m, h = 1: highest upper end of a 95% interval: {max(hi[k] for k in fast):+.1f}" in section

    def test_against_the_other_sets(self, out):
        fwd, primary, replication = (by_cell(rel, "net_bps") for rel in (sr.FORWARD, sr.PRIMARY, sr.REPLICATION))
        assert set(fwd) == set(primary)
        key = ("1H", "5", "rolling_vol_ratio")
        assert (f"- the largest primary cell, 1H h=5 rolling_vol_ratio: μ̂ {fwd[key]:+.1f}, against "
                f"{primary[key]:+.1f} on doge_2023 and {replication[key]:+.1f} on doge_2024") in out[DOGE_FWD]
        same = sum(np.sign(fwd[k]) == np.sign(primary[k]) for k in fwd)
        negative = sum(fwd[k] < 0 and primary[k] < 0 for k in fwd)
        positive = sum(fwd[k] > 0 and primary[k] > 0 for k in fwd)
        assert (same, negative, positive) == (55, 53, 2)
        assert (f"- cells whose μ̂ has the sign it has on doge_2023: {same} of 60 (both negative {negative}, "
                f"both positive {positive})") in out[DOGE_FWD]
        # Neither set has ties, so each value's rank is its place in the sorted order.
        a, b = (np.array([d[k] for k in sorted(fwd)]) for d in (fwd, primary))
        assert len(set(a)) == len(set(b)) == 60
        rho = np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
        assert f"{rho:.2f}" == "0.21"
        assert f"- rank correlation of the cells' μ̂ with doge_2023's (Spearman): {rho:.2f}" in out[DOGE_FWD]

    def test_the_largest_primary_cell_points_ahead_to_it(self, out):
        net, lo, hi = (by_cell(sr.FORWARD, c)[("1H", "5", "rolling_vol_ratio")]
                       for c in ("net_bps", "ci95_lo_bps", "ci95_hi_bps"))
        assert (f"{net:+.1f}", f"{lo:+.1f}", f"{hi:+.1f}") == ("+5.9", "-13.6", "+25.3")
        assert f"- doge_fwd (next section): μ̂ {net:+.1f}, 95% interval [{lo:+.1f}, {hi:+.1f}]" in out[LARGEST]

    def test_every_cell_is_listed_in_the_files_order(self, out):
        section = out[DOGE_FWD]
        first = section.index("- every cell (bar, h, factor, d, μ̂, 95% interval):") + 1
        rows = committed_rows(sr.FORWARD)
        listed = section[first:first + len(rows)]
        assert len(rows) == 60 and section[first + len(rows)] == "- not computed here:"
        for line, r in zip(listed, rows):
            bar, h, factor, d = line.split()[:4]
            assert (bar, h, factor, int(d)) == (r["bar"], r["horizon"], r["factor"], int(r["direction"]))
            net, lo, hi = (float(r[c]) for c in ("net_bps", "ci95_lo_bps", "ci95_hi_bps"))
            assert line.split()[4:] == [f"{net:+.1f}", f"[{lo:+.1f},", f"{hi:+.1f}]"]


class TestMakerProbe:
    def test_its_record(self, out):
        archives = committed_json(sr.MANIFESTS["maker_probe_archives"])
        files = archives["files"]
        kinds = [f["kind"] for f in files]
        size = sum(f["bytes"] for f in files)
        assert (len(files), f"{size / 1e9:.1f}") == (56, "5.2")
        fetched = sorted(f["downloaded_utc"] for f in files)
        (code,) = {r["code_commit"] for rel in (sr.PROBE_SUMMARY, sr.PROBE_DAILY, sr.PROBE_CHECKS)
                   for r in committed_rows(rel)}
        section = out[PROBE]
        assert "- c, which the probe never replaces (Terms): 10 bp" in section
        assert f"- code_commit of every row of summary.csv, daily.csv and checks.csv: {code[:7]}" in section
        assert (f"- archive files (manifest_archives.json): 56 ({kinds.count('orderbook')} orderbook, "
                f"{kinds.count('trades')} trades), {size:,} bytes (5.2 GB)") in section
        assert f"- downloaded (UTC), first / last: {fetched[0]} / {fetched[-1]}" in section
        assert f"- archive manifest created (UTC): {archives['created_utc']}" in section
        assert (f"- doge_fwd's results commit, required on the public main before the download "
                f"(doge_fwd_results_commit): {archives['doge_fwd_results_commit'][:7]}") in section

    def test_its_days_and_design(self, out):
        spec = get_spec()
        plan = spec["maker_probe"]
        days = [r["day"] for r in committed_rows(sr.PROBE_CHECKS)]
        candidates = committed_json(sr.MANIFESTS["maker_probe"])
        assert committed_json(sr.MANIFESTS["maker_probe_archives"])["days"] == days
        assert sr.maker_probe_days(REPO_ROOT / spec["freeze"]["amendment_file"]) == days
        assert len(days) == plan["n_days"] == 14
        assert candidates["n_candidates"] == 364 and set(days) <= set(candidates["candidates"])
        per_day = 24 * 60 // plan["decision_every_minutes"]
        skips = ("n_skipped_one_sided", "n_skipped_crossed", "n_skipped_no_data", "n_skipped_gap")
        assert {int(r["n_decisions"]) + sum(int(r[c]) for c in skips)
                for r in committed_rows(sr.PROBE_DAILY)} == {per_day} == {288}
        section = out[PROBE]
        assert ("- days (Appendix A): 14; the freeze amendment's, the archive manifest's and checks.csv's are the "
                "same 14: True") in section
        assert ("- candidate days they were drawn from (results/manifest_maker_probe.json): 364; every day among "
                "them: True") in section
        assert ("- decisions a day, 5 minutes apart (Appendix A): 288; decisions and skips add up to it in every row "
                "of daily.csv: True") in section
        assert "- each order (Appendix A): 1 contract for 60 s" in section
        fees = {(r["maker_fee_bps"], r["taker_fee_bps"], r["fees_verified"]) for r in committed_rows(sr.PROBE_SUMMARY)}
        assert fees == {("2.0", "5.0", "false")}
        assert "- fees in every row of summary.csv: maker / taker, verified: 2 / 5 bp, False" in section

    def test_the_checks_of_each_day(self, out):
        rows = committed_rows(sr.PROBE_CHECKS)
        missing = ("book_pauses_as_gaps", "trades_id_jumps", "book_clamped", "parser_gaps", "trade_gaps")
        total = {c: int(sum(values(rows, c))) for c in missing + (
            "snapshot_sides_checked", "snapshot_sides_touch_mismatch", "touch_mismatch_price_differs", "touch_gap_ms",
            "trades_source_1_left_out")}
        assert [total[c] for c in missing] == [0] * 5
        longest = max(rows, key=lambda r: int(r["book_longest_pause_ms"]))
        pause = int(longest["book_longest_pause_ms"])
        n, k = total["snapshot_sides_checked"], total["snapshot_sides_touch_mismatch"]
        price = total["touch_mismatch_price_differs"]
        assert (f"{pause / 1000:.1f}", n, k, price) == ("1.6", 16_210, 2_531, 145)
        hours = total["touch_gap_ms"] / 3_600_000
        assert round(hours) == 62
        left_out = [r["day"] for r in rows if int(r["trades_source_1_left_out"])]
        assert (left_out, total["trades_source_1_left_out"]) == (["2026-09-11"], 1_790)
        section = out[PROBE]
        assert "- days checked (checks.csv): 14" in section
        assert "- book pauses of 5 s or more, counted as missing data (book_pauses_as_gaps), all days: 0" in section
        assert f"- longest pause: {pause / 1000:.1f} s ({pause:,} ms, {longest['day']})" in section
        assert ("- trade id jumps / book messages stamped before the one before them (book_clamped), all days: "
                "0 / 0") in section
        assert "- gaps from missing data, book / trades (parser_gaps, trade_gaps), all days: 0 / 0" in section
        assert (f"- snapshot sides that differ from the rebuilt book at the touch: {k:,} of {n:,} ({k / n:.1%}), "
                f"{price} in price, {k - price:,} in size only") in section
        assert (f"- touch_gaps: hours dropped (touch_gap_ms), of the days' hours: {hours:.1f} of "
                f"{24 * len(rows)}") in section
        assert f"- trades marked source 1, left out: {total['trades_source_1_left_out']:,}, on 2026-09-11" in section

    def test_the_clocks_and_the_depth(self, out):
        rows = committed_rows(sr.PROBE_CHECKS)
        matched, within = values(rows, "clock_matched_share"), values(rows, "clock_lag_within_10ms_share")
        lag, ratio = values(rows, "clock_lag_ms_median"), values(rows, "clock_decrease_over_size_median")
        matched_span = f"{min(matched):.0%} / {max(matched):.0%}"
        within_span = f"{min(within):.0%} / {max(within):.0%}"
        assert (matched_span, within_span, min(lag), max(lag), set(ratio)) == ("98% / 99%", "95% / 98%", 5, 6, {1})
        bid, ask = int(max(values(rows, "max_depth_bid"))), int(max(values(rows, "max_depth_ask")))
        assert (bid, ask) == (400, 400)
        section = out[PROBE]
        assert f"- most levels a side of the rebuilt book held, bid / ask, any day: {bid} / {ask}" in section
        assert ("- share of trades followed within 100 ms by a decrease of the level they hit (clock_matched_share), "
                f"by day, from / to: {matched_span}") in section
        assert f"- median lag (clock_lag_ms_median), by day, from / to: {min(lag):g} / {max(lag):g} ms" in section
        assert ("- share of those lags within 10 ms (clock_lag_within_10ms_share), by day, from / to: "
                f"{within_span}") in section
        assert ("- median ratio of the decrease to the trade's size, trades alone at their price in their ms "
                f"(clock_decrease_over_size_median), by day, from / to: {min(ratio):.1f} / {max(ratio):.1f}") in section

    def test_the_decisions_and_the_skips(self, out):
        summary = probe_summary()
        totals = {(run, metric): {int(summary[(run, side, model, metric)]["total"])
                                  for side in ("buy", "sell") for model in ("M1", "M2")}
                  for run in RUNS for metric in ("n_decisions", "n_skipped_no_data", "n_skipped_gap")}
        assert totals == {("parser_gaps", "n_decisions"): {4_018}, ("parser_gaps", "n_skipped_no_data"): {14},
                          ("parser_gaps", "n_skipped_gap"): {0}, ("touch_gaps", "n_decisions"): {2_953},
                          ("touch_gaps", "n_skipped_no_data"): {14}, ("touch_gaps", "n_skipped_gap"): {1_065}}
        assert {r["n_skipped_no_data"] for r in committed_rows(sr.PROBE_DAILY)} == {"1"}
        skipped = [int(r["decision_ts"]) for r in committed_rows(sr.PROBE_ORDERS) if r["status"] == "no_data"]
        assert len(skipped) == 112 and {ts % sr.DAY_MS for ts in skipped} == {0}     # 14 days x 2 runs x 4
        section = out[PROBE]
        assert "- parser_gaps: decisions skipped for a gap: 0" in section
        assert ("- decisions per side and model: 4,032 (14 days x 288); decisions and skips add up to it in both "
                "runs: True") in section
        assert ("- skipped for no data (n_skipped_no_data), each run, side and model: 14 of 4,032; one on each day: "
                "True; each at 00:00 (orders.csv): 112 of 112") in section
        assert ("- touch_gaps: decisions skipped for a gap, per side and model, of those parser_gaps simulates: "
                "1,065 of 4,018") in section

    def test_summary_is_daily_averaged_over_the_days(self, out):
        """Each mean in summary.csv is the mean of daily.csv's values, every day weighted alike; its SE, SD/√days."""
        daily = committed_rows(sr.PROBE_DAILY)
        summary = probe_summary()
        assert len(summary) == 8 * 27
        for (run, side, model, metric), r in summary.items():
            days = [d for d in daily if (d["run"], d["side"], d["model"]) == (run, side, model)]
            assert len(days) == int(r["n_days"]) == 14
            if metric.startswith("n_"):
                assert float(r["total"]) == sum(int(d[metric]) for d in days)
            else:
                v = np.array(values(days, metric))
                assert float(r["mean"]) == pytest.approx(v.mean(), rel=1e-12, abs=1e-12)
                assert float(r["se"]) == pytest.approx(v.std(ddof=1) / math.sqrt(v.size), rel=1e-12, abs=1e-12)
        section = out[PROBE]
        assert ("- summary.csv recomputed from daily.csv (totals; means over the days, each weighted equally; "
                "SE = SD/√days), to 1e-9: True") in section
        assert "- days in every mean and total (n_days): 14" in section

    def test_the_table_row_by_row(self, out):
        summary = probe_summary()
        keys = list(dict.fromkeys(k[:3] for k in summary))
        assert len(keys) == 8
        section = out[PROBE]
        first = section.index("- mean (SE) of each run, side and model: fill rate, markout 1 / 5 / 15 min, "
                              "post then cross, difference:") + 1
        columns = (("fill_rate", ""), ("markout_1m_bps_mean", "+"), ("markout_5m_bps_mean", "+"),
                   ("markout_15m_bps_mean", "+"), ("shortfall_post_then_cross_bps_mean", ""),
                   ("shortfall_diff_bps_mean", "+"))
        for line, key in zip(section[first:first + len(keys)], keys):
            want = list(key)
            for metric, sign in columns:
                r = summary[key + (metric,)]
                want += [f"{float(r['mean']):{sign}.2f}", f"({float(r['se']):.2f})"]
            assert line.split() == want
        assert section[first + len(keys)].startswith("- ")
        # The comparison's first row, as it prints it.
        assert section[first].split()[3:] == ["0.78", "(0.02)", "-1.69", "(0.21)", "-1.06", "(0.45)", "-0.71",
                                              "(1.12)", "3.72", "(0.10)", "-1.75", "(0.08)"]

    def test_crossing_at_once(self, out):
        summary = probe_summary()
        cross, se = (probe_metric(summary, "shortfall_cross_now_bps_mean", f) for f in ("mean", "se"))
        now = {}
        for run in RUNS:
            means, ses = ({v for k, v in d.items() if k[0] == run} for d in (cross, se))
            assert len(means) == len(ses) == 1                      # the same for either side and either model
            now[run] = (means.pop(), ses.pop())
        assert {(f"{m:.2f}", f"{s:.2f}") for m, s in now.values()} == {("5.48", "0.04")}
        section = out[PROBE]
        assert ("- crossing at once, mean (SE), the same for each side and model: "
                + "; ".join(f"{run} {m:.2f} ({s:.2f})" for run, (m, s) in now.items())) in section
        assert ("- crossing at once less the 5 bp taker fee (half the spread): "
                + "; ".join(f"{run} {m - 5:.2f}" for run, (m, _) in now.items())) in section
        assert ("- a round trip crossing twice (2 x crossing at once): "
                + "; ".join(f"{run} {2 * m:.2f}" for run, (m, _) in now.items())) in section
        assert {f"{2 * m:.1f}" for m, _ in now.values()} == {"11.0"}

    def test_the_ranges_and_sums_the_reading_quotes(self, out):
        summary = probe_summary()
        fill = probe_metric(summary, "fill_rate")
        one = probe_metric(summary, "markout_1m_bps_mean").values()
        late_se = probe_metric(summary, "markout_15m_bps_mean", "se").values()
        post = probe_metric(summary, "shortfall_post_then_cross_bps_mean")
        diff, diff_se = (probe_metric(summary, "shortfall_diff_bps_mean", f) for f in ("mean", "se"))
        m1, m2 = ([v for k, v in fill.items() if k[2] == model] for model in ("M1", "M2"))
        assert (f"{min(m1):.0%}", f"{max(m1):.0%}") == ("75%", "78%")
        assert max(one) < 0                                          # every fill adversely selected after a minute
        assert (f"{min(post.values()):.1f}", f"{max(post.values()):.1f}") == ("2.9", "4.0")
        assert (f"{-max(diff.values()):.1f}", f"{-min(diff.values()):.1f}") == ("1.5", "2.5")
        assert (f"{min(diff_se.values()):.2f}", f"{max(diff_se.values()):.2f}") == ("0.07", "0.14")
        gap = max(abs(diff[("parser_gaps",) + k[1:]] - diff[("touch_gaps",) + k[1:]]) for k in diff)
        assert gap < 0.02
        trips = {(run, model): post[(run, "buy", model)] + post[(run, "sell", model)]
                 for run in RUNS for model in ("M1", "M2")}
        assert {f"{trips[(run, 'M1')]:.1f}" for run in RUNS} == {"7.7"}
        section = out[PROBE]
        assert (f"- fill rate from / to, both runs and sides: M1 {min(m1):.0%} / {max(m1):.0%}; "
                f"M2 {min(m2):.0%} / {max(m2):.0%}") in section
        assert f"- 1-minute markout from / to: {min(one):+.2f} / {max(one):+.2f} bp" in section
        assert f"- 15-minute markout's SE from / to: {min(late_se):.2f} / {max(late_se):.2f}" in section
        assert f"- post, then cross, from / to: {min(post.values()):.2f} / {max(post.values()):.2f} bp" in section
        assert (f"- difference from / to; its SE from / to: {min(diff.values()):+.2f} / {max(diff.values()):+.2f} bp; "
                f"{min(diff_se.values()):.2f} / {max(diff_se.values()):.2f}") in section
        assert f"- largest gap between the two runs' differences, same side and model: {gap:.3f} bp" in section
        assert ("- a round trip posting first (buy + sell post then cross), M2 / M1: "
                + "; ".join(f"{run} {trips[(run, 'M2')]:.2f} / {trips[(run, 'M1')]:.2f}" for run in RUNS)) in section


class TestInSampleAndData:
    def test_the_in_sample_counts_the_report_quotes(self, out):
        rows = committed_rows(sr.INSAMPLE)
        assert [sum(r["direction"] == d for r in rows) for d in ("1", "-1")] == [31, 29]
        section = out[IN_SAMPLE]
        assert "- directions +1 / -1: 31 / 29 of 60" in section
        assert "- flagged directions: 0" in section
        assert "- largest net t, (d·mean_s_fwd - 10)/SE: +0.64 at 1H h=15 funding_cost_pressure" in section
        assert "- cells with net t above 1.645: 0" in section
        assert any(line.startswith("- cells whose gross mean in the traded direction exceeds the 10 bp cost: 6: ")
                   for line in section)

    def test_the_funding_stamps_from_the_manifests(self, out):
        section = out["Data"]
        assert not [line for line in section if "NOT " in line]           # every series complete
        assert "- last grid time with an off-grid stamp (any file): 2024-01-17 08:00:00" in section
        assert ("- doge_2023 DOGE-USDT-SWAP off-grid stamps: 617 of 1,368 (617 of the 617 up to the cut-over), "
                "late by 3 to 23 s, none after 2024-01-17 08:00:00") in section
        assert ("- majors_2023 ETH-USDT-SWAP funding_cost_pressure on those bars (each bar grid): "
                "60 take the previous value, 82 are undefined") in section


# ---------------------------------------------------------------- helpers

class TestFindRepoRoot:
    def test_from_scripts_and_from_deeper_down(self):
        assert sr.REPO_ROOT == REPO_ROOT
        assert sr._find_repo_root(SCRIPT) == REPO_ROOT
        assert sr._find_repo_root(REPO_ROOT / "docs" / "a" / "b" / "c" / "summarize_results.py") == REPO_ROOT

    def test_the_nearest_ancestor_with_both_files(self, tmp_path):
        study, inner = tmp_path / "study", tmp_path / "study" / "vendor" / "copy"
        for rel in ("results/insample_v2.csv", "docs/preregistration.md", "scripts/results/insample_v2.csv"):
            for root in (study, inner):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("")
        # scripts/ holds a results file but no pre-registration, so the walk goes on past it.
        assert sr._find_repo_root(study / "scripts" / "summarize_results.py") == study
        assert sr._find_repo_root(study / "docs" / "a" / "b" / "c" / "summarize_results.py") == study
        assert sr._find_repo_root(inner / "scripts" / "summarize_results.py") == inner

    def test_outside_a_checkout_it_stops(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "insample_v2.csv").write_text("")         # no pre-registration beside it
        with pytest.raises(SystemExit, match="is not inside a checkout of the study"):
            sr._find_repo_root(tmp_path / "scripts" / "summarize_results.py")


def funding_entry(ts):
    """A manifest's funding entry for these settlement times, listing each spacing other than 8 h."""
    odd = [[a, b] for a, b in zip(ts[:-1], ts[1:]) if b - a != STEP]
    return {"path": "funding.parquet", "rows": len(ts), "first_ts": ts[0], "last_ts": ts[-1],
            "spacing_other_than_expected": odd}


LATE = [T0, T0 + STEP + 20_000, T0 + 2 * STEP + 3_000, T0 + 3 * STEP, T0 + 4 * STEP]


class TestSettlementTimes:
    def test_on_the_grid(self):
        ts = [T0 + i * STEP for i in range(4)]
        got = sr.settlement_times(funding_entry(ts))
        assert got.dtype == np.int64 and got.tolist() == ts
        assert sr.grid_offset_ms(got).tolist() == [0, 0, 0, 0]

    def test_late_stamps_are_rebuilt_from_the_listed_spacings(self):
        entry = funding_entry(LATE)
        assert len(entry["spacing_other_than_expected"]) == 3
        got = sr.settlement_times(entry)
        assert got.tolist() == LATE
        assert sr.grid_offset_ms(got).tolist() == [0, 20_000, 3_000, 0, 0]

    def test_the_offset_is_signed(self):
        ts = np.array([T0 - 5_000, T0, T0 + 5_000], dtype=np.int64)
        assert sr.grid_offset_ms(ts).tolist() == [-5_000, 0, 5_000]

    @pytest.mark.parametrize("change", [
        {"last_ts": T0 + 5 * STEP},
        {"rows": 4},
        {"rows": 6},
        {"spacing_other_than_expected": funding_entry(LATE)["spacing_other_than_expected"]
         + [[T0 + 9 * STEP, T0 + 10 * STEP + 1_000]]},
    ], ids=["last_ts", "fewer rows", "more rows", "unused spacing"])
    def test_an_entry_that_does_not_rebuild_stops(self, change):
        entry = {**funding_entry(LATE), **change}
        with pytest.raises(SystemExit, match="funding.parquet: its manifest's spacings do not rebuild"):
            sr.settlement_times(entry)

    def test_every_committed_funding_series_rebuilds(self):
        seen = 0
        for rel in FUNDING_MANIFESTS:
            for e in json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))["series"]:
                if e["kind"] != "funding":
                    continue
                seen += 1
                ts = sr.settlement_times(e)
                assert (len(ts), ts[0], ts[-1]) == (e["rows"], e["first_ts"], e["last_ts"]), (rel, e["inst_id"])
                spacings = np.diff(ts)
                assert (spacings > 0).all()
                assert int((spacings > STEP).sum()) == len(e["gaps"]), (rel, e["inst_id"])
                off = sr.grid_offset_ms(ts)
                assert 0 <= off.min() and off.max() <= 24_000, (rel, e["inst_id"])    # late by at most 24 s
        assert seen == 7          # DOGE in each of four sets, the three majors


class TestMakerProbeDays:
    def test_the_block_and_nothing_else(self, tmp_path):
        doc = tmp_path / "freeze.md"
        doc.write_text("- **Maker-probe days.** 2025-10-19, 2025-11-02.\n\n```maker-probe-days\n2025-10-19\n"
                       "2025-11-02\n```\n\n```freeze-commit\n4a4e4b0\n```\n", encoding="utf-8")
        assert sr.maker_probe_days(doc) == ["2025-10-19", "2025-11-02"]

    @pytest.mark.parametrize("text", ["2025-10-19\n", "```maker-probe-days\n2025-10-19\n"], ids=["none", "unclosed"])
    def test_without_a_closed_block_it_stops(self, tmp_path, text):
        doc = tmp_path / "freeze.md"
        doc.write_text(text, encoding="utf-8")
        with pytest.raises(SystemExit, match="no closed maker-probe-days block"):
            sr.maker_probe_days(doc)


class TestSummaryDifferences:
    def test_the_committed_files_agree(self):
        assert sr.summary_differences(sr.load(sr.PROBE_SUMMARY), sr.load(sr.PROBE_DAILY)) <= 1e-9

    def test_a_changed_number_a_blank_or_a_missing_row_shows(self):
        summary, daily = sr.load(sr.PROBE_SUMMARY), sr.load(sr.PROBE_DAILY)
        assert summary.row(0, named=True)["metric"] == "fill_rate"
        mean, se = summary["mean"].to_list(), summary["se"].to_list()
        mean[0] += 0.01
        se[0] = None
        moved = summary.with_columns(pl.Series("mean", mean))
        assert sr.summary_differences(moved, daily) == pytest.approx(0.01)
        assert sr.summary_differences(summary.with_columns(pl.Series("se", se, dtype=pl.Float64)), daily) == math.inf
        assert sr.summary_differences(summary.slice(1), daily) == math.inf
        # A day counted twice changes n_days, the totals and the means.
        assert sr.summary_differences(summary, pl.concat([daily, daily.head(1)])) >= 1


class TestFormatting:
    @pytest.mark.parametrize("x", [None, math.nan, math.inf, -math.inf])
    def test_undefined(self, x):
        assert sr.bp(x) == sr.tt(x) == sr.pv(x) == "undefined"

    def test_rounding_and_sign(self):
        assert (sr.bp(17.597), sr.bp(-87.14), sr.bp(0.0)) == ("+17.6", "-87.1", "+0.0")
        assert (sr.tt(1.70887), sr.tt(-2.8249)) == ("+1.71", "-2.82")
        assert sr.pv(0.04374) == "0.044"

    def test_cells(self):
        df = pl.DataFrame({"bar": ["15m", "1H"], "horizon": [1, 5], "factor": ["rsi_14", "rolling_vol_ratio"]})
        assert sr.cells(df) == "15m h=1 rsi_14; 1H h=5 rolling_vol_ratio"
        assert sr.cells(df.head(0)) == "none"

    def test_utc(self):
        assert sr.utc(0) == "1970-01-01 00:00:00"
        assert sr.utc(get_spec()["sets"]["doge_fwd"]["end_ms"]) == "2026-11-23 20:00:00"

    def test_spearman_shares_a_tied_rank(self):
        assert sr.spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 30.0, 20.0])) == pytest.approx(0.5)
        # Ties take their mean rank: [1, 2.5, 2.5] against [1, 2, 3].
        assert sr.spearman(np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])) == pytest.approx(math.sqrt(0.75))


class TestReading:
    def test_one_value(self):
        assert sr.one_value(pl.Series([10.0, 10.0 + 1e-12, None, math.nan]), "x") == 10.0
        with pytest.raises(SystemExit, match="x is not constant across rows"):
            sr.one_value(pl.Series([10.0, 10.1]), "x")

    def test_by_bar_h_puts_15m_first_then_the_horizon(self):
        df = pl.DataFrame({"bar": ["1H", "15m", "1H", "15m", "15m"], "horizon": [1, 5, 5, 1, 1]})
        got = sr.by_bar_h(df, pl.len().alias("n"))
        assert [(g["bar"], g["horizon"], g["n"]) for g in got] == [("15m", 1, 2), ("15m", 5, 1), ("1H", 1, 1),
                                                                   ("1H", 5, 1)]

    def test_load_types_each_column_and_reads_a_blank_as_null(self):
        f = sr.load(sr.FORWARD)
        assert f.height == 60
        assert (f.schema["horizon"], f.schema["direction"], f.schema["n_active"]) == (pl.Int64,) * 3
        assert (f.schema["direction_flagged"], f.schema["replicated"]) == (pl.Boolean,) * 2
        assert (f.schema["status"], f.schema["code_commit"]) == (pl.Utf8,) * 2
        assert f.schema["net_bps"] == pl.Float64
        # doge_fwd decides nothing: its t and its labels are blank on every row.
        assert f["t_hh"].null_count() == f["replicated"].null_count() == 60
        assert f["net_bps"].null_count() == 0
        assert sr.load(sr.PRIMARY)["t_hh"].max() == pytest.approx(1.709, abs=5e-4)

    def test_load_types_the_maker_probe_files(self):
        s, c = sr.load(sr.PROBE_SUMMARY), sr.load(sr.PROBE_CHECKS)
        assert (s.schema["run"], s.schema["side"], s.schema["metric"], c.schema["day"]) == (pl.Utf8,) * 4
        assert (s.schema["fees_verified"], c.schema["book_ended_early"]) == (pl.Boolean,) * 2
        assert (s.schema["total"], s.schema["se"], c.schema["touch_gap_ms"]) == (pl.Float64,) * 3
        # A mean has no total and a count no SE: 7 means and 20 counts for each of 8 run, side and model.
        assert (s["total"].null_count(), s["se"].null_count()) == (7 * 8, 20 * 8)


class TestImport:
    @pytest.mark.parametrize("before", [False, True])
    def test_it_leaves_dont_write_bytecode_alone(self, before, monkeypatch, tmp_path):
        # Run as a script it writes no bytecode; imported, the setting is the importing
        # process's. Each value is set here, so PYTHONDONTWRITEBYTECODE does not decide the
        # outcome, and whatever bytecode the import writes goes under tmp_path.
        monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path))
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(sys, "dont_write_bytecode", before)
        _load("summarize_results")
        assert sys.dont_write_bytecode is before
