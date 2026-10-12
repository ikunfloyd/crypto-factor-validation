"""
The code against the pre-registration (docs/preregistration.md, §10 step 2).

Three things are pinned here:

  * Appendix A parses and is internally consistent: set boundaries, the
    90-day warm-up, the thresholds derived from Φ⁻¹, the 60 cells.
  * The §5.1 factor functions, and the helpers they call, are byte-for-byte
    the code of the pre-registration commit (2e14ee1). So are the two v1
    functions that §8 R2 defines as "v1's statistic, as in the commit
    containing this document". The hashes below were taken from that
    commit; CI clones with depth 1, so the test compares against constants
    rather than git history. A failure here is a deviation that must be
    logged in §12, not a hash to update.
  * The download guard's protected windows are the test sets' ranges.
  * The §12 deviations log is well formed (dated rows, reason, whether
    test-set data was seen) and records the replacement of R9's daily IC.
"""
import ast
import hashlib
import re
from datetime import datetime, timezone
from statistics import NormalDist

import pytest

from research import spec

DAY_MS = 86_400_000
H20_MS = 20 * 3_600_000


def utc_ms(text):
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@pytest.fixture(scope="module")
def s():
    return spec.load_spec()


class TestAppendixA:
    REQUIRED = [
        "version", "written", "timezone", "windows_half_open", "sets", "spot_legs", "data",
        "insample_dir", "insample_sha256", "bars", "horizons", "bar_aggregation",
        "forward_return", "gaps", "factors", "excluded_factors", "buckets", "direction_rule",
        "cost_round_trip_bps", "statistic", "se", "test", "families", "first_holm_threshold_t",
        "untestable", "descriptive_sets", "descriptive_stats", "intervals", "robustness",
        "power", "maker_probe", "freeze",
    ]

    def test_parses_and_has_required_keys(self, s):
        assert isinstance(s, dict)
        missing = [k for k in self.REQUIRED if k not in s]
        assert not missing, f"Appendix A lacks {missing}"
        assert s["timezone"] == "UTC" and s["windows_half_open"] is True
        assert spec.get_spec() == s

    def test_sets(self, s):
        assert set(s["sets"]) == {"IS", "doge_2023", "doge_2024", "majors_2023", "doge_fwd"}
        for name, st in s["sets"].items():
            for key in ("inst", "warmup_start_ms", "start_ms", "end_ms", "role"):
                assert key in st, f"{name} lacks {key}"
            for key in ("warmup_start_ms", "start_ms", "end_ms"):
                assert st[key] % DAY_MS in (0, H20_MS), f"{name}.{key} is not at 00:00 or 20:00 UTC"
            assert st["warmup_start_ms"] < st["start_ms"] < st["end_ms"]
            assert st["start_ms"] - st["warmup_start_ms"] == 90 * DAY_MS, f"{name} warm-up is not 90 days"
            assert set(st["inst"]) <= set(s["spot_legs"]), f"{name} has an instrument without a spot leg"

    def test_sets_match_the_table_in_section_4_1(self, s):
        # The prose table and the YAML were written separately; they must agree.
        text = spec.PREREG_PATH.read_text(encoding="utf-8")
        section = text.split("### 4.1 Sets", 1)[1].split("**Why these sets.**", 1)[0]
        rows = {}
        for line in section.splitlines():
            m = re.match(r"^\|\s*\**`(\w+)`\**\s*\|", line)
            if m:
                rows[m.group(1)] = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", line)
        assert set(rows) == set(s["sets"])
        for name, dts in rows.items():
            st = s["sets"][name]
            assert [utc_ms(d) for d in dts[:3]] == [st["start_ms"], st["end_ms"], st["warmup_start_ms"]], name

    def test_primary_and_secondary_windows(self, s):
        sets = s["sets"]
        assert sets["doge_2023"]["start_ms"] == utc_ms("2023-09-23 20:00")
        assert sets["doge_2023"]["end_ms"] == sets["doge_2024"]["start_ms"] == utc_ms("2024-09-23 20:00")
        assert sets["doge_2024"]["end_ms"] == spec.IS_DATA_START_MS == utc_ms("2025-09-23 20:00")
        assert sets["IS"]["end_ms"] == sets["doge_fwd"]["start_ms"] == utc_ms("2026-09-23 20:00")
        for k in ("warmup_start_ms", "start_ms", "end_ms"):
            assert sets["majors_2023"][k] == sets["doge_2023"][k]

    def test_quarter_bounds_match_r3(self, s):
        q = s["robustness"]["quarter_bounds_ms"]
        assert q == [utc_ms(d) for d in ("2023-12-23 20:00", "2024-03-23 20:00", "2024-06-23 20:00")]
        p = s["sets"]["doge_2023"]
        assert p["start_ms"] < q[0] < q[1] < q[2] < p["end_ms"]

    def test_cells(self, s):
        assert len(s["factors"]) == 10
        assert len(s["bars"]) == 2 and len(s["horizons"]) == 3
        cells = len(s["factors"]) * len(s["bars"]) * len(s["horizons"])
        assert cells == 10 * 2 * 3 == 60
        for fam in s["families"].values():
            assert fam["cells"] == cells
        assert not set(s["excluded_factors"]) & set(s["factors"])

    def test_first_holm_threshold(self, s):
        alpha = s["test"]["alpha"]
        m = s["families"]["F_primary"]["cells"]
        t = round(NormalDist().inv_cdf(1 - alpha / m), 3)
        assert s["first_holm_threshold_t"] == t == 3.144

    def test_interval_and_power_z_values(self, s):
        z = NormalDist().inv_cdf
        iv = s["intervals"]
        assert iv["ub95"] == {"sides": 1, "z": round(z(0.95), 3)} and iv["ub95"]["z"] == 1.645
        assert iv["ub95_family"] == {"sides": 1, "z": s["first_holm_threshold_t"]}
        assert iv["doge_fwd_ci"] == {"sides": 2, "z": round(z(0.975), 3)} and iv["doge_fwd_ci"]["z"] == 1.96
        pw = s["power"]
        assert pw["z_alpha"] == s["first_holm_threshold_t"]
        assert pw["z_power"] == round(z(pw["power"]), 3) == 0.842

    def test_code_constants(self, s):
        assert set(spec.BAR_MS) == set(s["bars"])
        assert spec.BAR_MS == {"15m": 15 * 60_000, "1H": 60 * 60_000}
        assert spec.MINUTE_MS == 60_000
        assert spec.FREEZE_AMENDMENT_PATH.relative_to(spec.REPO_ROOT).as_posix() == s["freeze"]["amendment_file"]

    def test_robustness_ic_is_the_full_window_spearman(self, s):
        # §8 R9 as replaced on 2026-10-09 (§12): one Spearman IC over the
        # window, with §6.7's SE, lag and Holm, tested two-sided.
        ic = s["robustness"]["ic"]
        assert ic == {
            "method": "spearman_full_window", "ties": "average",
            "statistic": "mean_of_standardised_rank_products", "se": "hansen_hodrick",
            "lag": "h_minus_1", "family_correction": "holm", "sidedness": "two_sided",
        }
        assert ic["se"] == s["se"]["method"] and ic["lag"] == s["se"]["lag"]
        assert ic["family_correction"] == s["test"]["correction"]
        # The daily version's keys are gone, not merely unused.
        assert not {"per", "min_pairs", "lag_days", "drop_zero_rank_variance"} & set(ic)

    def test_insample_hashes(self, s):
        for name, digest in s["insample_sha256"].items():
            assert re.fullmatch(r"[0-9a-f]{64}", digest), name

    def test_insample_files_unmodified_when_present(self, s):
        # data/history is gitignored, so CI has nothing to check; locally the
        # in-sample files must still be the ones the pre-registration hashed.
        from research.io import sha256_file

        base = spec.REPO_ROOT / s["insample_dir"]
        present = [n for n in s["insample_sha256"] if (base / n).exists()]
        if not present:
            pytest.skip("in-sample files not present")
        for name in present:
            assert sha256_file(base / name) == s["insample_sha256"][name], name


def _table_cells(line):
    # Split a Markdown table row on pipes that are not escaped (\|t\| stays one cell).
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


class TestDeviationsLog:
    """§12: every change is a dated row giving its reason and whether test data was seen."""

    @pytest.fixture(scope="class")
    def rows(self):
        text = spec.PREREG_PATH.read_text(encoding="utf-8")
        log = text.split("## 12. Deviations log", 1)[1].split("## References", 1)[0]
        lines = [ln for ln in log.splitlines() if ln.startswith("|")]
        assert _table_cells(lines[0]) == ["Date", "Section", "Deviation", "Reason", "Test-set data seen?"]
        assert set("".join(_table_cells(lines[1]))) <= set("-:")
        return [_table_cells(ln) for ln in lines[2:]]

    def test_rows_are_dated_and_complete(self, s, rows):
        assert rows, "the log has no rows"
        for r in rows:
            assert len(r) == 5, r
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", r[0]), r[0]
            assert r[0] >= s["written"], f"deviation dated {r[0]} before the document ({s['written']})"
            assert all(r), f"empty field in {r}"
            assert re.match(r"(No|Yes)\b", r[4]), r[4]

    def test_r9_replacement_is_logged(self, rows):
        r9 = [r for r in rows if r[1] == "§8 R9"]
        assert len(r9) == 1
        date, _, change, reason, seen = r9[0]
        assert date == "2026-10-09"
        assert "full-window Spearman IC with a Hansen–Hodrick SE" in change
        assert "biased for persistent price-derived factors on a pure random walk" in reason
        assert seen.startswith("No. Made before publication")

    def test_fixed_staleness_and_rl_are_logged(self, s, rows):
        # §4.2's fixed 8 h and §10's RL row (2026-10-11) are deviations from
        # the text as first written, so the log must carry them.
        row = [r for r in rows if r[0] == "2026-10-11"]
        assert len(row) == 1
        _, section, change, _, seen = row[0]
        assert "§4.2" in section and "§10" in section
        assert "fixed 8 h" in change and "RL" in change
        assert seen.startswith("No. Made before publication")
        fcp = s["factors"]["funding_cost_pressure"]
        assert fcp["max_staleness_hours"] == s["data"]["funding"]["expected_spacing_hours"] == 8
        assert "max_staleness" not in fcp       # the old "one_settlement_interval" key is gone


# SHA-256 of each method's source text in quant_core/factor_engine.py at the
# pre-registration commit 2e14ee1: its decorators ("@" + expression, one per
# line) followed by ast.get_source_segment of the def. Decorators are included
# because `@jit(..., fastmath=True)` changes what a kernel computes.
# _signed_trade_volume is listed because calculate_net_taker_flow calls it
# when trade-side data is supplied; the study supplies none (§5.1), but §10
# freezes "the helpers they call".
FROZEN_SHA256 = {
    "calculate_rsi": "930a41d8023b5ac8801d18411a75ad992a2d3f9d1779f2e8f218a625c0f96a61",
    "_calculate_rsi_numba": "5e611be8d5c5dfbab70bf5d049dd3a6d479db84d626a113445a86499a2107fde",
    "calculate_macd_hist": "1b3270905ba0c60eb918d166c60172c744c6e73b11346947e91af8021c1a75d4",
    "_calculate_ema_numba": "eec4bb04ae26168294112fa7f3ef8a0565a793b3850748c221bb82a25b9fb6c7",
    "calculate_bollinger_pctb": "d9b7936b05ff8435c1d70208754aa9dcbe5a9a111fadbf9cd57065120ea2df60",
    "calculate_rolling_vol_ratio": "d6a132a371b4e30f7c0540526194616e09530acc4006fff84794ee5467bf6aad",
    "calculate_net_taker_flow": "7c442ef48c6ab28708f442eef43faf7399042ad8b59f533eb651a2dd380f19a1",
    "_signed_trade_volume": "d85dd6539c5a9fe8b9b51a25d0f8f800a073bd50896b023e8f56a8a902c7a2ad",
    "calculate_basis_z_score": "c59ad098ad942c335700a2319aba5bbfe5c7a1cf3be73a270c8fb98d3af647d3",
    "calculate_funding_cost_pressure": "615998618afa9a3ce1bde78d3058ee885991437e8ec29f51af9fdb6329f021ab",
    "_asof_backward": "367ce2bf23e370159d0b625c3431ecd86bdc7100aeb38b8c9fd15a2150e64f98",
}


FACTOR_ENGINE = spec.REPO_ROOT / "quant_core" / "factor_engine.py"


def _method_sources(src=None):
    if src is None:
        src = FACTOR_ENGINE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FactorEngine"]
    assert len(classes) == 1, "FactorEngine class not found exactly once"
    out = {}
    for node in classes[0].body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FROZEN_SHA256:
            assert node.name not in out, f"{node.name} defined twice"
            parts = ["@" + ast.get_source_segment(src, d) for d in node.decorator_list]
            parts.append(ast.get_source_segment(src, node))
            out[node.name] = "\n".join(parts)
    return out


class TestFrozenFactorCode:
    def test_every_frozen_function_exists(self):
        assert set(_method_sources()) == set(FROZEN_SHA256)

    @pytest.mark.parametrize("name", sorted(FROZEN_SHA256))
    def test_unchanged_since_preregistration(self, name):
        digest = hashlib.sha256(_method_sources()[name].encode("utf-8")).hexdigest()
        assert digest == FROZEN_SHA256[name], (
            f"FactorEngine.{name} differs from the pre-registration commit 2e14ee1. "
            f"The §5.1 code is frozen: log the change as a deviation (§12) instead of "
            f"updating this hash."
        )

    def test_hash_covers_decorators(self):
        # Turning fastmath off (or on) changes a kernel's arithmetic, so it
        # must change the hash even though the def body is untouched.
        src = FACTOR_ENGINE.read_text(encoding="utf-8")
        assert src.count("fastmath=True") >= 2
        before = _method_sources(src)["_calculate_rsi_numba"]
        after = _method_sources(src.replace("fastmath=True", "fastmath=False"))["_calculate_rsi_numba"]
        assert hashlib.sha256(after.encode()).hexdigest() != hashlib.sha256(before.encode()).hexdigest()

    def test_no_other_definition_shadows_them(self):
        # A later `FactorEngine.calculate_rsi = ...` would bypass the hashed body.
        src = FACTOR_ENGINE.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Attribute):
                        assert t.attr not in FROZEN_SHA256, f"{t.attr} reassigned at line {node.lineno}"


# SHA-256 of the two module-level functions of scripts/factor_research.py
# that research.stats.v1_t_sqrt_h loads by path for §8 R2 ("v1's statistic,
# as in the commit containing this document"), taken from 2e14ee1 the same
# way as FROZEN_SHA256: decorators (none today) followed by the def's source.
# run_oos only guards the span from prereg-freeze to HEAD, so without this an
# edit between 2e14ee1 and the freeze would silently change R2.
V1_FROZEN_SHA256 = {
    "layered_returns": "f78f84c180f67c407fd9ebaa4fe7188fea809e0675ae37fe8d87d508ad7b795f",
    "_spread_t_stat": "2f3710a8e15e2b459b527de55aa3c1bbc4648fda160c427af48523b8bcc1fd8b",
}

V1_SCRIPT = spec.REPO_ROOT / "scripts" / "factor_research.py"


def _v1_function_sources(src=None):
    if src is None:
        src = V1_SCRIPT.read_text(encoding="utf-8")
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in V1_FROZEN_SHA256:
            assert node.name not in out, f"{node.name} defined twice"
            parts = ["@" + ast.get_source_segment(src, d) for d in node.decorator_list]
            parts.append(ast.get_source_segment(src, node))
            out[node.name] = "\n".join(parts)
    return out


class TestFrozenV1Statistic:
    def test_every_frozen_function_exists(self):
        assert set(_v1_function_sources()) == set(V1_FROZEN_SHA256)

    @pytest.mark.parametrize("name", sorted(V1_FROZEN_SHA256))
    def test_unchanged_since_preregistration(self, name):
        digest = hashlib.sha256(_v1_function_sources()[name].encode("utf-8")).hexdigest()
        assert digest == V1_FROZEN_SHA256[name], (
            f"scripts/factor_research.py:{name} differs from the pre-registration commit 2e14ee1. "
            f"§8 R2 is v1's statistic as in that commit: log the change as a deviation (§12) "
            f"instead of updating this hash."
        )

    def test_hash_covers_decorators(self):
        src = V1_SCRIPT.read_text(encoding="utf-8")
        before = _v1_function_sources(src)["_spread_t_stat"]
        after = _v1_function_sources(src.replace("def _spread_t_stat", "@staticmethod\ndef _spread_t_stat"))
        assert hashlib.sha256(after["_spread_t_stat"].encode()).hexdigest() != \
            hashlib.sha256(before.encode()).hexdigest()

    def test_r2_loads_exactly_these_functions_from_that_file(self):
        # stats.v1_t_sqrt_h must call the hashed module-level functions, not a
        # copy: same file, same names, and nothing rebinds them afterwards.
        import inspect
        from research import stats

        src = inspect.getsource(stats.v1_t_sqrt_h)
        assert "mod.layered_returns(" in src and "mod._spread_t_stat(" in src
        assert '"scripts" / "factor_research.py"' in inspect.getsource(stats._v1_module)
        mod = stats._v1_module()
        for name in V1_FROZEN_SHA256:
            assert inspect.getsource(getattr(mod, name)).rstrip() == _v1_function_sources()[name].rstrip()
        for node in ast.walk(ast.parse(V1_SCRIPT.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    name = t.id if isinstance(t, ast.Name) else getattr(t, "attr", None)
                    assert name not in V1_FROZEN_SHA256, f"{name} reassigned at line {node.lineno}"


class TestProtectedWindows:
    def test_three_ranges(self):
        assert spec.protected_windows() == [
            (utc_ms("2023-06-25 20:00"), utc_ms("2024-09-23 20:00")),  # doge_2023 / majors_2023
            (utc_ms("2024-06-25 20:00"), utc_ms("2025-09-23 20:00")),  # doge_2024
            (utc_ms("2026-09-23 20:00"), utc_ms("2026-11-23 20:00")),  # doge_fwd minus its IS warm-up
        ]

    def test_test_sets_are_refused(self, s):
        for name in ("doge_2023", "doge_2024", "majors_2023", "doge_fwd"):
            st = s["sets"][name]
            assert spec.overlaps_protected(st["start_ms"], st["end_ms"]), name
        d = s["sets"]["doge_2023"]
        assert spec.overlaps_protected(d["warmup_start_ms"], d["start_ms"])

    def test_insample_supplement_is_allowed(self):
        assert not spec.overlaps_protected(utc_ms("2025-09-24 00:00"), utc_ms("2026-09-23 20:00"))

    def test_boundaries_are_half_open(self):
        end_2024 = utc_ms("2025-09-23 20:00")
        assert not spec.overlaps_protected(end_2024, end_2024 + DAY_MS)
        assert spec.overlaps_protected(end_2024 - 1, end_2024 + DAY_MS)
        fwd = utc_ms("2026-09-23 20:00")
        assert not spec.overlaps_protected(fwd - DAY_MS, fwd)
        assert spec.overlaps_protected(fwd - DAY_MS, fwd + 1)
