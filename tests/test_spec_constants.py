"""
Constants that live in code, checked against the pre-registration.

Appendix A is the source of truth (research/spec.py). A module that restates
a value, or a default argument that would silently stand in for one, must
agree with it; so must the results-file layout, which §10 gives in prose
rather than YAML and which is therefore parsed from the text here. A failure
means the code drifted from the document, and the fix belongs in the code
(or, after publication, in a §12 deviation), never in this test.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import re
from statistics import NormalDist

import numpy as np
import pytest

from research import factors, freeze
from research import pipeline as P
from research import stats as S
from research.spec import PREREG_PATH, REPO_ROOT, get_spec

SPEC = get_spec()
TEXT = PREREG_PATH.read_text(encoding="utf-8")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}_consts", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- §10 text parsing

def _section_10_results():
    body = TEXT.split("**Results files.**", 1)[1].split("## 11.", 1)[0]
    return body


def _bullets(block):
    out, cur = [], None
    for line in block.splitlines():
        if line.startswith("- "):
            if cur is not None:
                out.append(cur)
            cur = line[2:].strip()
        elif cur is not None and line.startswith("  "):
            cur += " " + line.strip()
    if cur is not None:
        out.append(cur)
    return out


def _column_bullets():
    body = _section_10_results()
    return _bullets(body.split("The other columns, in order:", 1)[1].split("Columns are filled as follows", 1)[0])


def section_10_columns():
    """The §10 column list in order, and the columns each R-number tags."""
    cols, tags = [], {}
    for b in _column_bullets():
        if b.startswith("status:"):
            cols.append("status")
            cols += re.findall(r"`(\w+)`", b.split("Plus", 1)[1])
            continue
        pending = []  # a tag such as "(R1)" covers the names since the previous tag in its bullet
        for item in re.split(r",(?![^()]*\))", b):
            m = re.match(r"\s*(\w+)\s*(?:\(([^)]*)\))?", item)
            cols.append(m.group(1))
            pending.append(m.group(1))
            note = (m.group(2) or "").strip()
            if re.fullmatch(r"R\d+", note):
                tags[note] = pending
                pending = []
    return cols, tags


def section_10_fill_rules():
    body = _section_10_results()
    block = body.split("Columns are filled as follows, and are blank otherwise:", 1)[1]
    block = block.split("The multi-valued rows", 1)[0]
    rules = {}
    for b in _bullets(block):
        m = re.match(r"\*\*(.+?):\*\*\s*(.*)$", b)
        rules[m.group(1)] = m.group(2).rstrip(".")
    return rules


def _names(text, tags):
    """Column names in a fill-rule sentence, expanding "the R1, R2, R8 and R9 columns"."""
    out = set()
    for r in re.findall(r"R\d+", text):
        out |= set(tags[r])
    text = re.sub(r"the (?:R\d+(?:,\s*|\s+and\s+))*R\d+ columns", "", text)
    for part in re.split(r",\s*|\s+and\s+", text):
        if re.fullmatch(r"[a-z][a-z0-9_]*", part.strip()):
            out.add(part.strip())
    return out


class TestResultsLayout:
    def test_key_columns(self):
        m = re.search(r"has one row per \(([^)]*)\); those\s+four key columns come first\.", _section_10_results())
        assert tuple(x.strip() for x in m.group(1).split(",")) == P.KEY_COLUMNS
        assert P.OOS_COLUMNS[:len(P.KEY_COLUMNS)] == P.KEY_COLUMNS

    def test_value_columns_in_order(self):
        cols, _ = section_10_columns()
        assert tuple(cols) == P.OOS_VALUE_COLUMNS
        assert P.OOS_COLUMNS == P.KEY_COLUMNS + P.OOS_VALUE_COLUMNS
        assert len(set(P.OOS_COLUMNS)) == len(P.OOS_COLUMNS)

    def test_r_tags(self):
        _, tags = section_10_columns()
        assert tags == {
            "R1": ["p_placebo", "p_placebo_family"],
            "R2": ["t_sqrt_h"],
            "R8": ["net_funding_bps"],
            "R9": ["ic", "ic_t", "ic_p", "ic_p_holm"],
        }

    def test_fill_rules(self):
        _, tags = section_10_columns()
        rules = section_10_fill_rules()
        assert _names(rules["Every row"], tags) == P.FILL_EVERY_ROW
        assert _names(rules["Every row except `POOLED`"], tags) == P.FILL_UNLESS_POOLED
        assert _names(rules["`doge_2023` rows, `doge_2024` rows and `majors_2023` `POOLED` rows"], tags) == P.FILL_TESTED
        assert _names(rules["`doge_fwd` rows"], tags) == P.FILL_DESCRIPTIVE
        assert _names(rules["`doge_2023` rows only"], tags) == P.FILL_PRIMARY
        assert "no t or p" in rules["Per-instrument `majors_2023` rows"]
        assert "`descriptive`" in rules["Per-instrument `majors_2023` rows"]

    def test_fill_sets_are_disjoint_and_cover_every_column(self):
        sets = [P.FILL_EVERY_ROW, P.FILL_UNLESS_POOLED, P.FILL_TESTED, P.FILL_DESCRIPTIVE, P.FILL_PRIMARY]
        union = set().union(*sets)
        assert sum(len(s) for s in sets) == len(union)
        assert union == set(P.OOS_VALUE_COLUMNS)

    def test_filled_columns_by_role(self):
        f = P.filled_columns
        assert f(P.ROLE_POOLED, pooled=False) == P.FILL_EVERY_ROW | P.FILL_UNLESS_POOLED
        assert f(P.ROLE_POOLED, pooled=True) == P.FILL_EVERY_ROW | P.FILL_TESTED
        assert f(P.ROLE_REPLICATION) == P.FILL_EVERY_ROW | P.FILL_UNLESS_POOLED | P.FILL_TESTED
        assert f(P.ROLE_DESCRIPTIVE) == P.FILL_EVERY_ROW | P.FILL_UNLESS_POOLED | P.FILL_DESCRIPTIVE
        assert f(P.ROLE_PRIMARY) == set(P.OOS_VALUE_COLUMNS) - P.FILL_DESCRIPTIVE

    def test_statuses(self):
        status = next(b for b in _column_bullets() if b.startswith("status:"))
        primary, secondary, elsewhere = status.split(";")[:3]
        assert re.findall(r"`(\w+)`", primary) == [P.STATUS_CONFIRMED, P.STATUS_NOT_REJECTED, P.STATUS_UNTESTABLE]
        assert re.findall(r"`(\w+)`", secondary) == [P.STATUS_REJECTED, P.STATUS_NOT_REJECTED, P.STATUS_UNTESTABLE]
        assert re.findall(r"`(\w+)`", elsewhere.split(".")[0]) == [P.STATUS_DESCRIPTIVE]
        # The labels are false, not blank, on a primary row that is not confirmed.
        assert "(true/false; false for cells that are not confirmed) on primary rows" in status
        assert "se_method (hh | bartlett_fallback)" in " ".join(_column_bullets())

    def test_robustness_columns(self):
        m = re.search(r"one row per\s+\(([^)]*)\),\s*with columns ([\w, ]+?) and (\w+)\.", _section_10_results())
        keys = [x.strip() for x in m.group(1).split(",")]
        values = [x.strip() for x in m.group(2).split(",")] + [m.group(3)]
        assert tuple(keys + values) == P.ROBUSTNESS_COLUMNS

    def test_robustness_rows_are_the_ones_section_10_describes(self):
        # The bullets under "What those columns hold depends on the row" name
        # every row of the robustness file except R8, whose count and SE §8
        # sends there; the pipeline must write exactly those rows, and leave
        # se_bps and t_hh blank exactly where a bullet says so.
        block = _section_10_results().split("What\nthose columns hold depends on the row:", 1)[1]
        described, blank = set(), set()
        for b in _bullets(block):
            m = re.match(r"\*\*(.+?):\*\*\s*(.*)$", b)
            names = set(re.findall(r"R(?:\d+|L)\b", m.group(1)))
            described |= names
            if "se_bps and t_hh are blank" in m.group(2):
                blank |= names
        assert described == {"R3", "R4", "R5", "R6", "R7", "RL", "R10"}
        assert blank == {"R5", "RL"}
        src = inspect.getsource(P._robustness)
        written = set(re.findall(r'add(?:_stat)?\("(R\w+)"', src))
        assert written == described | {"R8"}
        assert set(re.findall(r'add\("(R\w+)",[^\n]*math\.nan, math\.nan', src)) == blank

    def test_rl_row_is_what_section_10_says(self):
        rl = next(b for b in _bullets(_section_10_results()) if b.startswith("**RL:**"))
        rl = " ".join(rl.split())
        assert "the realised mean run length of s_t in bars that §7 requires for a confirmed cell" in rl
        assert "over the evaluable bars (a sign change, s = 0 or a dropped bucket ends a run)" in rl
        assert "value_bps holds the length in bars, n_active the number of runs, and se_bps and t_hh are blank" in rl
        assert "written for every cell" in rl
        assert list(inspect.signature(S.mean_run_length).parameters) == \
            ["s", "ts", "start_ms", "end_ms", "bar_ms", "h"]

    def test_insample_columns(self):
        assert freeze.INSAMPLE_COLUMNS == tuple(SPEC["freeze"]["directions_columns"])
        assert P.INSAMPLE_COLUMNS is freeze.INSAMPLE_COLUMNS
        # And the §6.5 prose list, in the same order.
        sec = TEXT.split("`results/insample_v2.csv` has one row per (bar, horizon, factor)", 1)[1]
        sec = sec.split("se_bps and mde_bps are blank", 1)[0]
        names = ["bar", "horizon", "factor"]
        for b in _bullets(sec):
            b = re.sub(r"\([^)]*\)", "", b).strip(" ;.")
            names += [x.strip() for x in b.split(" and ")]
        assert names == list(freeze.INSAMPLE_COLUMNS)


# ---------------------------------------------------------------- parameters

class TestParameters:
    def test_study_params_are_appendix_a(self):
        p = P.StudyParams.from_spec()
        rb = SPEC["robustness"]
        assert p.bars == tuple(SPEC["bars"]) and p.horizons == tuple(SPEC["horizons"])
        assert p.factor_names == tuple(SPEC["factors"]) == factors.FACTOR_NAMES
        assert p.factor_params == SPEC["factors"]
        assert p.cost_bps == SPEC["cost_round_trip_bps"] == 10.0
        assert list(p.edges_pct) == SPEC["buckets"]["edges_pct"] == [20, 80]
        assert p.min_prior == SPEC["buckets"]["min_prior_values"] == 500
        assert p.quantiles == SPEC["buckets"]["quantiles"] == 5
        assert p.min_active_is == SPEC["direction_rule"]["min_active_IS"] == 300
        assert p.min_active_trades == SPEC["untestable"]["min_active_trades"] == 300
        assert p.untestable_p == SPEC["untestable"]["p_value"] == 1.0
        assert p.alpha == SPEC["test"]["alpha"] == 0.05
        assert p.placebo_k == rb["placebo"]["k"] == 2000
        assert p.placebo_min_shift_days == rb["placebo"]["min_shift_days"] == 30
        assert list(p.quarter_bounds_ms) == rb["quarter_bounds_ms"]
        assert list(p.volatile_days_k) == rb["leave_out_volatile_days"]["k"] == [0, 5, 10]
        assert list(p.cost_sensitivity_bps) == rb["cost_sensitivity_bps"] == [0.0, 4.0, 7.0, 11.0]
        assert list(p.winsorize_pct) == rb["winsorize_pct"] == [1, 99]
        assert p.monotonicity_buckets == rb["monotonicity"]["buckets"] == 5
        assert p.funding_spacing_ms == SPEC["data"]["funding"]["expected_spacing_hours"] * 3_600_000 == 28_800_000
        assert len(p.cells()) == SPEC["families"]["F_primary"]["cells"] == 60
        assert p.cells() == freeze.expected_cells()

    def test_factor_parameters_the_engine_cannot_vary(self):
        factors._check_params(SPEC["factors"])  # raises on drift
        p = P.StudyParams.from_spec()
        factors._check_params(p.factor_params)

    def test_funding_staleness_is_a_fixed_8h(self):
        # §4.2 (2026-10-11): a fixed limit, the regular interval; nothing in
        # the code derives it from the series any more.
        fcp = SPEC["factors"]["funding_cost_pressure"]
        assert fcp["max_staleness_hours"] == SPEC["data"]["funding"]["expected_spacing_hours"] == 8
        assert fcp["staleness_test"] == "ge"
        assert factors._max_staleness_ms(fcp) == 8 * 3_600_000
        assert not hasattr(factors, "settlement_spacing_ms")
        assert "funding_spacing_ms" not in inspect.signature(factors.compute_factors).parameters
        assert "is a fixed 8 h, the regular interval" in " ".join(TEXT.split())

    def test_stats_defaults(self):
        sig = lambda f: inspect.signature(f).parameters  # noqa: E731
        assert [sig(S.expanding_edges)["lo_pct"].default, sig(S.expanding_edges)["hi_pct"].default] == \
            SPEC["buckets"]["edges_pct"]
        for f in (S.expanding_edges, S.expanding_percentiles, S.quintile_means):
            assert sig(f)["min_prior"].default == SPEC["buckets"]["min_prior_values"]
        assert sig(S.direction)["min_active"].default == SPEC["direction_rule"]["min_active_IS"]
        assert sig(S.turnover_aware)["cost_bps"].default == SPEC["cost_round_trip_bps"]
        assert sig(S.quintile_means)["quantiles"].default == SPEC["robustness"]["monotonicity"]["buckets"]
        assert sig(S.v1_t_sqrt_h)["quantiles"].default == SPEC["buckets"]["quantiles"]

    def test_thresholds(self):
        b = S.bounds(0.0, 1.0)
        iv = SPEC["intervals"]
        assert b["ub95"] == iv["ub95"]["z"] == 1.645
        assert b["ub95_family"] == iv["ub95_family"]["z"] == SPEC["first_holm_threshold_t"] == 3.144
        assert b["ci95_hi"] == -b["ci95_lo"] == iv["doge_fwd_ci"]["z"] == 1.96
        assert S.mde(1.0) == pytest.approx(SPEC["power"]["z_alpha"] + SPEC["power"]["z_power"]) == 3.986
        m, a = SPEC["families"]["F_primary"]["cells"], SPEC["test"]["alpha"]
        # Holm's first step is p <= alpha/m, i.e. t >= the registered 3.144.
        assert round(NormalDist().inv_cdf(1 - a / m), 3) == SPEC["first_holm_threshold_t"]
        p_edge = a / m
        assert S.holm(np.r_[p_edge, np.ones(m - 1)])[0] == pytest.approx(a)

    def test_placebo_offsets_for_the_primary_window(self):
        p = P.StudyParams.from_spec()
        w = P.window_of("doge_2023")
        table = P.placebo_offset_table(w, p)
        cols = P.placebo_family_columns(w, p)
        for bar, off in table.items():
            B = P.bar_ms(bar)
            ref = S.placebo_offsets((w.end_ms - w.start_ms) // B, 30 * 86_400_000 // B, 2000)
            np.testing.assert_array_equal(off, ref)
            assert len(off) == 2000                       # K' = K: no duplicates on either grid
            np.testing.assert_array_equal(cols[bar], np.arange(2000))   # every j enters the family max

    def test_n_evaluable_matches_section_9_example(self):
        w_is, w_pr = P.window_of("IS"), P.window_of("doge_2023")
        H = 3_600_000
        assert S.n_evaluable(w_is.start_ms, w_is.end_ms, H, 1) == 6596 - 1 - 1
        assert S.n_evaluable(w_pr.start_ms, w_pr.end_ms, H, 15) == 8784 - 15 - 1

    def test_check_spec_refuses_unimplemented_variants(self):
        P.check_spec(SPEC)
        for path, value in ((("test", "correction"), "bonferroni"),
                            (("families", "F_majors", "pooling"), "inverse_variance"),
                            (("untestable", "counts_in_family"), False),
                            (("se", "lag"), "h"),
                            (("robustness", "ic", "method"), "spearman"),
                            (("robustness", "ic", "lag"), "1_day"),
                            (("robustness", "ic", "min_pairs"), 20)):     # an extra key: refused too
            bad = copy.deepcopy(SPEC)
            d = bad
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = value
            with pytest.raises(ValueError):
                P.StudyParams.from_spec(bad)



# ---------------------------------------------------------------- R9 and the other clarifications

def _r_row(name):
    """The text of one row of the §8 robustness table."""
    m = re.search(rf"^\| {name} \| (.*) \|$", TEXT, re.M)
    assert m, f"§8 has no row {name}"
    return m.group(1)


class TestR9:
    def test_appendix_a_is_the_full_window_ic(self):
        assert SPEC["robustness"]["ic"] == {
            "method": "spearman_full_window", "ties": "average",
            "statistic": "mean_of_standardised_rank_products", "se": "hansen_hodrick",
            "lag": "h_minus_1", "family_correction": "holm", "sidedness": "two_sided",
        }

    def test_pipeline_pins_every_ic_key(self):
        pinned = {p[-1]: v for p, v in P._SPEC_FIXED if p[:-1] == ("robustness", "ic")}
        assert pinned == SPEC["robustness"]["ic"]

    def test_prose_matches_the_spec_and_the_code(self):
        r9 = _r_row("R9")
        assert "over the evaluable bars where both are defined" in r9
        assert "IC = mean of u_t·v_t" in r9                                 # statistic
        assert "(average ranks for ties)" in r9                              # ties: average
        assert "standardised to mean 0 and variance 1" in r9
        assert "§6.7 Hansen–Hodrick SE of that mean (ℓ = h − 1" in r9        # se, lag
        assert "products indexed on the grid, zero where undefined" in r9
        assert "The test is two-sided" in r9                                 # sidedness
        assert "Holm at 0.05 is applied within this separate family of 60" in r9
        assert "A mean of daily ICs was registered first and replaced before publication" in r9
        assert list(inspect.signature(S.full_window_ic).parameters) == \
            ["f", "fwd", "ts", "start_ms", "end_ms", "bar_ms", "h"]

    def test_deviation_is_logged(self):
        log = TEXT.split("## 12. Deviations log", 1)[1].split("## References", 1)[0]
        row = next(line for line in log.splitlines() if line.startswith("| 2026-10-09 | §8 R9 |"))
        assert "full-window Spearman IC with a Hansen–Hodrick SE" in row
        assert row.rstrip(" |").endswith("no in-sample v2 or test-set statistic had been computed.")


# Each sentence the 2026-10-09 clarification added, and the code that does
# what it says (behaviour is tested in tests/test_research_*.py).
CLARIFIED = {
    "§4.2 fixed funding staleness (factors.funding_fresh, _max_staleness_ms; 2026-10-11)":
        ("text", "The limit is a\n    fixed 8 h, the regular interval, so no bar's mask depends on later settlements."),
    "§10 RL row (stats.mean_run_length, pipeline._robustness row RL; 2026-10-11)":
        ("text", "over\n  the evaluable bars (a sign change, s = 0 or a dropped bucket ends a run)."),
    "§5 macd warm-up (factors.compute_factors, triple_screen)":
        ("text", "defines index slow + signal − 2 only when at least slow + signal bars\n  exist"),
    "§5 basis warm-up (factors._basis)":
        ("text", "returns nothing unless the whole series has ≥ 1440 finite basis\n  values"),
    "§6.7 SE = 0 (stats._stat_on_grid, pooled_statistic, pipeline._family)":
        ("text", "If SE = 0, t_HH is undefined and the\ncell's p is 1."),
    "R1 per-cell t (pipeline._robustness)": ("R1", "using that cell's own t_HH"),
    "R1 offset index (pipeline.placebo_family_columns)":
        ("R1", "with the same offset index j for every cell; an index is kept only if its offset is new on both bar grids"),
    "R1 below 300 (stats.circular_placebo_t)":
        ("R1", "A rotated cell is evaluated even if it has fewer than 300 active bars."),
    "R4 day of a return (stats.volatile_days)":
        ("R4", "A 1-minute return belongs to the UTC day of its later minute."),
    "R4 overlap (stats.leave_out_volatile_days)":
        ("R4", "[close of t+1, close of t+1+h] shares a stretch of positive length with one of those days"),
    "R5 trading (stats.turnover_aware)":
        ("R5", "Only evaluable bars' signals are traded, and a return across a dropped bucket contributes nothing."),
    "R5 per day (stats.turnover_aware)":
        ("R5", "the sum divided by the window's length in days"),
    "R8 count (pipeline._robustness, row R8)":
        ("R8", "that count, with the §6.7 SE, goes to the robustness file"),
    "R10 quintiles (stats.quintile_means)":
        ("R10", "Q1 = f < e1, Q2 = [e1, e2), Q3 = [e2, e3], Q4 = (e3, e4], Q5 = f > e4"),
    "R10 folding (stats.quintile_means)":
        ("R10", "Where e1 = e2, Q2 folds into Q3; where e3 = e4, Q4 folds into Q3."),
}


@pytest.mark.parametrize("what", sorted(CLARIFIED))
def test_clarified_sentence_is_in_the_document(what):
    where, phrase = CLARIFIED[what]
    assert phrase in (TEXT if where == "text" else _r_row(where)), what

# ---------------------------------------------------------------- sets, roles, paths

class TestSetsAndPaths:
    def test_roles_and_families(self):
        assert P.set_with_role(P.ROLE_PRIMARY) == SPEC["families"]["F_primary"]["set"] == "doge_2023"
        assert P.set_with_role(P.ROLE_REPLICATION) == SPEC["families"]["F_doge2024"]["set"] == "doge_2024"
        assert P.set_with_role(P.ROLE_POOLED) == SPEC["families"]["F_majors"]["set"] == "majors_2023"
        assert P.set_with_role(P.ROLE_DESCRIPTIVE) == SPEC["descriptive_sets"][0] == "doge_fwd"
        assert P.set_with_role(P.ROLE_IS) == "IS"
        assert SPEC["robustness"]["sets"] == [P.set_with_role(P.ROLE_PRIMARY)]

    def test_windows(self):
        for name, s in SPEC["sets"].items():
            w = P.window_of(name)
            assert (w.warmup_start_ms, w.start_ms, w.end_ms) == (s["warmup_start_ms"], s["start_ms"], s["end_ms"])

    def test_paths(self):
        assert P.INSAMPLE_RESULTS.as_posix() == SPEC["freeze"]["directions_file"]
        assert P.oos_manifest_path("doge_2023").as_posix() == "results/oos/manifest_doge_2023.json"
        assert "`results/oos/manifest_<set>.json`" in TEXT
        assert P.IS_SUPP_MANIFEST.as_posix() == "results/manifest_is_supp.json"
        assert "`results/manifest_is_supp.json`" in TEXT
        assert P.IS_SUPP_DIR.as_posix() == "data/history/is_supp" and "`data/history/is_supp/`" in TEXT
        assert P.OOS_DATA_DIR.as_posix() == "data/history/oos" and "`data/history/oos/<set>/<instrument>/`" in TEXT
        assert P.oos_results_path("doge_2024").as_posix() == "results/oos/doge_2024.csv"
        assert P.robustness_results_path("doge_2023").as_posix() == "results/oos/robustness_doge_2023.csv"
        assert "`results/oos/robustness_doge_2023.csv`" in TEXT

    def test_input_layout_matches_fetch_v2(self):
        fv = _load("fetch_v2")
        assert fv.OUTPUT_FILE == {"candles": P.CANDLE_FILE, "funding": P.FUNDING_FILE}
        assert fv.FUNDING_STEP_MS == SPEC["data"]["funding"]["expected_spacing_hours"] * 3_600_000
        # Every instrument's spot leg is what Appendix A says it is.
        files = P.input_files("majors_2023")
        spots = {f.instrument: f.path.parent.name for f in files if f.kind == "spot"}
        assert spots == {i: SPEC["spot_legs"][i] for i in SPEC["sets"]["majors_2023"]["inst"]}

    def test_run_oos_reads_the_freeze_from_appendix_a(self):
        ro = _load("run_oos")
        assert ro._freeze_spec() == {"tag": SPEC["freeze"]["tag"], "amendment": SPEC["freeze"]["amendment_file"],
                                     "directions": SPEC["freeze"]["directions_file"]}
        assert SPEC["freeze"]["tag"] == "prereg-freeze"
        assert ro.oos_sets() == ("doge_2023", "doge_2024", "majors_2023", "doge_fwd")
        assert ro.ALLOWED_PREFIXES == ("docs/", "results/")
        assert re.search(r"is under\s+`docs/` or `results/`", TEXT)
