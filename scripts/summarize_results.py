#!/usr/bin/env python3
"""
Every number docs/preregistration-results.md quotes, from the committed results files.

That document (§10 step 6) reads the out-of-sample results against the
pre-registration, and almost every sentence of it carries a number: a count
of cells, a mean, a t, a bound. A reader should not have to trust its
arithmetic. This script prints those numbers under the document's own
headings, in its order and at its rounding (or one digit more), computed
from the committed files:

    results/oos/doge_2023.csv, results/oos/robustness_doge_2023.csv,
    results/oos/doge_2024.csv, results/oos/majors_2023.csv,
    results/oos/doge_fwd.csv, results/insample_v2.csv,
    results/maker_probe/summary.csv, daily.csv, checks.csv and two columns
    of orders.csv, and the manifests (MANIFESTS) of the four test sets, the
    in-sample supplement, the maker-probe days and the maker probe's
    archive files,

plus Appendix A of the pre-registration for the evaluation windows and the
maker probe's design, and the freeze amendment's list of maker-probe days.
It uses none of the study's code beyond the parser of Appendix A
(research.spec), so it checks the document independently of the pipeline
that wrote the files; whether the files are what the frozen code computes
from the data is scripts/verify_results.py's question, and for the maker
probe scripts/run_maker_probe.py --verify's. The constants the bounds use
(1.645, 3.144, 0.842, 1.960, the 10 bp cost) are read back from the files
as well, as (ub95 - μ̂)/SE and the like, rather than restated.

The funding stamps the Data section describes are not committed, but their
manifests are, and a manifest lists every spacing other than 8 h with its
two timestamps; with the first timestamp and the row count, that rebuilds
every settlement time (settlement_times), so their lateness is computed
here too.

The doge_fwd section ends with every cell's μ̂ and 95% interval, which is
what §7 says that descriptive set reports. The maker-probe section prints
the probe's table from summary.csv, after recomputing summary.csv from the
days in daily.csv, and the checks of each day from checks.csv. What the
document takes from elsewhere (commit times from git, the Release and the
archive visit from GitHub and Software Heritage, §3's overlap with earlier
research and its earlier contact with doge_fwd, the probe's thresholds) is
named at the end of its section, so a number missing here is never a
silent omission. A last section gives the in-sample counts the report,
docs/factor-research-v2.md, quotes in its §4 (directions, the largest net
t), from results/insample_v2.csv.

The checkout is the one this file is in, found by walking up from its own
path, so the output does not depend on the working directory. It reads
only; nothing is written.

Usage:
    python scripts/summarize_results.py
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl


def _find_repo_root(start: Path) -> Path:
    """The nearest ancestor holding results/ and the pre-registration: the checkout this file is in."""
    for p in start.parents:
        if (p / "results" / "insample_v2.csv").is_file() and (p / "docs" / "preregistration.md").is_file():
            return p
    raise SystemExit(f"{start} is not inside a checkout of the study (no results/insample_v2.csv above it)")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
# Appendix A gives the evaluation windows and the maker probe's design;
# research.spec only parses it. Nothing else of the study's code is used.
sys.path[:] = [str(REPO_ROOT)] + [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]
if __name__ == "__main__":
    # Run as a script it writes nothing, bytecode included; imported (by the
    # tests), it leaves that setting to the importing process.
    sys.dont_write_bytecode = True
from research.spec import load_spec  # noqa: E402

PRIMARY = "results/oos/doge_2023.csv"
ROBUSTNESS = "results/oos/robustness_doge_2023.csv"
REPLICATION = "results/oos/doge_2024.csv"
MAJORS = "results/oos/majors_2023.csv"
FORWARD = "results/oos/doge_fwd.csv"
INSAMPLE = "results/insample_v2.csv"
PROBE_SUMMARY = "results/maker_probe/summary.csv"
PROBE_DAILY = "results/maker_probe/daily.csv"
PROBE_CHECKS = "results/maker_probe/checks.csv"
PROBE_ORDERS = "results/maker_probe/orders.csv"
MANIFESTS = {
    "doge_2023": "results/oos/manifest_doge_2023.json",
    "doge_2024": "results/oos/manifest_doge_2024.json",
    "majors_2023": "results/oos/manifest_majors_2023.json",
    "doge_fwd": "results/oos/manifest_doge_fwd.json",
    "is_supp": "results/manifest_is_supp.json",
    "maker_probe": "results/manifest_maker_probe.json",
    "maker_probe_archives": "results/maker_probe/manifest_archives.json",
}

KEY = ["bar", "horizon", "factor"]
POOLED = "POOLED"
PROBE_RUNS = ("parser_gaps", "touch_gaps")
# Columns the run wrote as text or as true/false (research.pipeline, and for
# the maker probe scripts/run_maker_probe.py); every other column is a
# number, a blank being undefined (§6.5, §10).
TEXT_COLUMNS = frozenset({"instrument", "bar", "factor", "se_method", "status", "code_commit",
                          "manifest_sha256", "flag_reason", "row", "parameter",
                          "run", "day", "side", "model", "metric"})
FLAG_COLUMNS = frozenset({"direction_flagged", "flagged", "replicated", "consistent_on_majors",
                          "fees_verified", "book_ended_early", "trades_ended_early"})
INT_COLUMNS = frozenset({"horizon", "direction", "n_eval", "n_active"})

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
FUNDING_STEP_MS = 8 * HOUR_MS
N01 = NormalDist()


# ---------------------------------------------------------------- reading

def load(rel: str) -> pl.DataFrame:
    """A results CSV with each column typed as the run wrote it; a blank field is null."""
    raw = pl.read_csv(REPO_ROOT / rel, infer_schema_length=0)
    cols = []
    for c in raw.columns:
        s = raw[c]
        if c in TEXT_COLUMNS:
            cols.append(s)
        elif c in FLAG_COLUMNS:
            cols.append((s == "true").alias(c))          # a blank stays null
        elif c in INT_COLUMNS:
            cols.append(s.cast(pl.Int64))
        else:
            cols.append(s.cast(pl.Float64))
    return pl.DataFrame(cols)


def load_json(rel: str) -> Dict[str, Any]:
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def maker_probe_days(path: Path) -> List[str]:
    """The days of the freeze amendment's ``maker-probe-days`` block, one ISO date a line."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    try:
        first = lines.index("```maker-probe-days") + 1
        return lines[first:lines.index("```", first)]
    except ValueError:
        raise SystemExit(f"{path}: no closed maker-probe-days block") from None


# ---------------------------------------------------------------- formatting

def bp(x: Optional[float]) -> str:
    """Basis points as the document writes them: signed, one decimal."""
    return "undefined" if x is None or not math.isfinite(x) else f"{x:+.1f}"


def tt(x: Optional[float]) -> str:
    return "undefined" if x is None or not math.isfinite(x) else f"{x:+.2f}"


def pv(x: Optional[float]) -> str:
    """A p- or q-value to three decimals (the document rounds to two or three)."""
    return "undefined" if x is None or not math.isfinite(x) else f"{x:.3f}"


def cell(r: Dict[str, Any]) -> str:
    return f"{r['bar']} h={r['horizon']} {r['factor']}"


def cells(df: pl.DataFrame) -> str:
    return "; ".join(cell(r) for r in df.iter_rows(named=True)) or "none"


def say(label: str, value: Any) -> None:
    print(f"- {label}: {value}")


def heading(title: str) -> None:
    print(f"\n## {title}")


def elsewhere(items: Iterable[str]) -> None:
    """The section's numbers this script cannot compute from results/, and where they come from."""
    items = list(items)
    if items:
        print("- not computed here:")
        for i in items:
            print(f"    {i}")


def by_bar_h(df: pl.DataFrame, *aggs: pl.Expr) -> List[Dict[str, Any]]:
    """Rows grouped by bar and horizon, in the document's order (15m before 1H)."""
    return df.group_by("bar", "horizon").agg(*aggs).sort(pl.col("bar") == "1H", "horizon").to_dicts()


def utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def one_value(s: pl.Series, name: str) -> float:
    """The single value a derived constant takes on every defined row, refusing a spread."""
    v = s.drop_nulls().drop_nans()
    lo, hi = float(v.min()), float(v.max())
    if hi - lo > 1e-9 * max(1.0, abs(hi)):
        raise SystemExit(f"{name} is not constant across rows ({lo} to {hi})")
    return lo


# ---------------------------------------------------------------- sections

def terms(p: pl.DataFrame) -> Dict[str, float]:
    """The constants the document's terms use, read back from the primary file; returned for the other sections."""
    print("- Terms (the constants read back from the primary results):")
    z = {
        "cost": one_value(p["gross_bps"] - p["net_bps"], "gross - net"),
        "z_ub95": one_value((p["ub95_net_bps"] - p["net_bps"]) / p["se_bps"], "(ub95 - net)/SE"),
        "z_family": one_value((p["ub95_family_net_bps"] - p["net_bps"]) / p["se_bps"], "(ub95_family - net)/SE"),
        "z_mde": one_value(p["mde_realised_bps"] / p["se_bps"], "MDE realised/SE"),
    }
    z["z_power"] = z["z_mde"] - z["z_family"]
    say("  cells", f"{p.height} ({p['factor'].n_unique()} factors x bars {sorted(p['bar'].unique().to_list())} "
                 f"x horizons {sorted(p['horizon'].unique().to_list())})")
    say("  gross - net on every row (the round-trip cost)", f"{z['cost']:.1f} bp")
    say("  ub95 = μ̂ + z·SE with z", f"{z['z_ub95']:.3f}")
    say("  ub95_family = μ̂ + z·SE with z (Holm's first step for 60 cells)", f"{z['z_family']:.3f}")
    say("  MDE realised = (z_alpha + z_power)·SE with z_power", f"{z['z_power']:.3f}")
    elsewhere(["the 80th/20th percentile edges and the 0.05 family-wise error: Appendix A"])
    return z


def record(p: pl.DataFrame, s: pl.DataFrame, m: pl.DataFrame) -> None:
    heading("Record")
    for name, df in (("doge_2024", s), ("majors_2023", m), ("doge_2023", p)):
        mf = load_json(MANIFESTS[name])
        codes = df["code_commit"].unique().to_list()
        say(f"{name}: code_commit of every row", ", ".join(c[:7] for c in codes))
        say(f"{name}: manifest created (UTC)", mf["created_utc"])
        say(f"{name}: public main when its data was fetched (manifest remote_main_commit)",
            mf["remote_main_commit"][:7])
    for name in ("is_supp", "maker_probe"):
        say(f"{name} manifest created (UTC)", load_json(MANIFESTS[name])["created_utc"])
    elsewhere([
        "every other commit hash, and every commit time: git (a manifest's created_utc above is when "
        "fetch_v2 wrote it; the commit that added it, whose time the table gives, came shortly after)",
        "the Release time and the Software Heritage visit: GitHub and Software Heritage",
        "the 1e-9 tolerance of the primary run's check of the secondary files: scripts/run_oos.py (guard 9)",
    ])


def headline(p: pl.DataFrame, r: pl.DataFrame, z: Dict[str, float]) -> None:
    heading("Headline (§7)")
    status = dict(p.group_by("status").len().iter_rows())
    say("status counts", ", ".join(f"{k} {v}" for k, v in sorted(status.items())))
    say("untestable cells", status.get("untestable", 0))
    say("flagged directions", int(p["direction_flagged"].sum()))
    say("SE methods", ", ".join(f"{k} {v}" for k, v in sorted(p.group_by("se_method").len().iter_rows())))
    top = p.sort("t_hh", descending=True, nulls_last=True).row(0, named=True)
    say("largest t", f"{tt(top['t_hh'])} at {cell(top)} (μ̂ {bp(top['net_bps'])}, SE {top['se_bps']:.1f}); "
                     f"Holm's first step needs {z['z_family']:.3f}")
    say("its one-sided p", pv(top["p_one_sided"]))
    say("cells with one-sided p < 0.05", int((p["p_one_sided"] < 0.05).sum()))
    say("smallest Holm p / BH q / BY q",
        f"{pv(p['p_holm'].min())} / {pv(p['q_bh'].min())} / {pv(p['q_by'].min())}")
    pos = p.filter(pl.col("net_bps") > 0).sort("t_hh", descending=True, nulls_last=True)
    say("cells with μ̂ > 0", f"{pos.height}: " + "; ".join(
        f"{cell(x)} (μ̂ {bp(x['net_bps'])}, t {tt(x['t_hh'])})" for x in pos.iter_rows(named=True)))
    say("cells with gross mean > 0", int((p["gross_bps"] > 0).sum()))
    gross = r.filter((pl.col("row") == "R6") & (pl.col("parameter") == "c=0"))
    up = gross.filter(pl.col("t_hh") > z["z_ub95"])
    say(f"gross t > {z['z_ub95']:.3f} (R6, c = 0)", f"{up.height}, factors {sorted(set(up['factor'].to_list()))}")
    say(f"gross t < -{z['z_ub95']:.3f} (R6, c = 0)", int((gross["t_hh"] < -z["z_ub95"]).sum()))
    say("confirmed / replicated / consistent_on_majors",
        f"{status.get('confirmed', 0)} / {int(p['replicated'].sum())} / {int(p['consistent_on_majors'].sum())}")


def no_pay(p: pl.DataFrame) -> None:
    heading("Cells that do not pay at taker cost (§9)")
    neg = p.filter(pl.col("ub95_net_bps") < 0)
    say("cells with ub95 < 0", f"{neg.height} of {p.height}")
    for g in by_bar_h(neg, pl.col("factor"), pl.len().alias("n")):
        say(f"  {g['bar']}, h = {g['horizon']} ({g['n']})", ", ".join(g["factor"]))
    fam = p.filter(pl.col("ub95_family_net_bps") < 0)
    say("cells with ub95_family < 0", fam.height)
    gap = p.filter((pl.col("ub95_net_bps") < 0) & (pl.col("ub95_family_net_bps") >= 0))
    say("ub95 < 0 but ub95_family >= 0", f"{gap.height}: {cells(gap)}")
    one = p.filter((pl.col("bar") == "15m") & (pl.col("horizon") == 1))
    say("15m, h = 1: largest ub95_family", f"{one['ub95_family_net_bps'].max():+.2f}")
    say("15m, h = 1: gross means from / to", f"{bp(one['gross_bps'].min())} / {bp(one['gross_bps'].max())}")
    top = p.sort("ub95_family_net_bps", descending=True, nulls_last=True).row(0, named=True)
    say("largest ub95_family", f"{bp(top['ub95_family_net_bps'])} at {cell(top)}")
    rest = p.filter(pl.col("ub95_net_bps") >= 0)
    print(f"- the {rest.height} cells with ub95 >= 0 (bar, h, factor, d, μ̂, SE, t, ub95, ub95_family, "
          "MDE frozen, MDE realised):")
    order = {f: i for i, f in enumerate(p["factor"].unique(maintain_order=True).to_list())}
    rows = sorted(rest.iter_rows(named=True), key=lambda x: (x["bar"] != "15m", x["horizon"], order[x["factor"]]))
    for x in rows:
        print(f"    {x['bar']:>3} {x['horizon']:>2} {x['factor']:<22} {x['direction']:+d} {bp(x['net_bps']):>6} "
              f"{x['se_bps']:5.1f} {tt(x['t_hh']):>6} {bp(x['ub95_net_bps']):>6} {bp(x['ub95_family_net_bps']):>6} "
              f"{x['mde_bps']:5.1f} {x['mde_realised_bps']:5.1f}")


def power(p: pl.DataFrame, z: Dict[str, float]) -> None:
    heading("Power (§9)")
    p = p.with_columns((pl.col("mde_realised_bps") / pl.col("mde_bps")).alias("ratio"))
    larger = p.filter(pl.col("mde_realised_bps") > pl.col("mde_bps"))
    say("cells whose realised MDE (so SE) exceeds the frozen one", f"{larger.height} of {p.height}")
    say("the others", cells(p.filter(pl.col("mde_realised_bps") <= pl.col("mde_bps"))))
    say("realised / frozen MDE: median, min, max",
        f"{p['ratio'].median():.2f}, {p['ratio'].min():.2f}, {p['ratio'].max():.2f}")
    print("- by bar and h (median MDE frozen, median MDE realised, median ratio):")
    for g in by_bar_h(p, pl.col("mde_bps").median(), pl.col("mde_realised_bps").median(), pl.col("ratio").median()):
        print(f"    {g['bar']:>3} {g['horizon']:>2}  {g['mde_bps']:5.1f}  {g['mde_realised_bps']:5.1f}  "
              f"{g['ratio']:.2f}")
    one = p.filter((pl.col("bar") == "15m") & (pl.col("horizon") == 1))["mde_realised_bps"]
    say("15m, h = 1: realised MDE median (gross: + the cost), min, max",
        f"{one.median():.1f} ({one.median() + z['cost']:.1f} gross), {one.min():.1f}, {one.max():.1f}")
    long = p.filter((pl.col("bar") == "1H") & (pl.col("horizon") == 15))["mde_realised_bps"]
    se = long.median() / z["z_mde"]
    say("1H, h = 15: realised MDE min, max, median", f"{long.min():.1f}, {long.max():.1f}, {long.median():.1f}")
    edge = 50.0          # the document's example of a large net edge, in bp per trade
    say(f"1H, h = 15: power of the first Holm step against a {edge:g} bp net edge, at the median MDE",
        f"{N01.cdf(edge / se - z['z_family']):.2f}")
    elsewhere(["80% power: Appendix A (it is the z_power read back under Terms)"])


def _family_lines(name: str, df: pl.DataFrame) -> None:
    status = dict(df.group_by("status").len().iter_rows())
    say(f"{name}: status counts", ", ".join(f"{k} {v}" for k, v in sorted(status.items())))
    top = df.sort("t_hh", descending=True, nulls_last=True).row(0, named=True)
    say(f"{name}: largest t", f"{tt(top['t_hh'])} at {cell(top)} (μ̂ {bp(top['net_bps'])}, SE {top['se_bps']:.1f}, "
                              f"one-sided p {pv(top['p_one_sided'])}, Holm p {pv(top['p_holm'])})")
    low = df.filter(pl.col("p_one_sided") < 0.05).sort("t_hh", descending=True, nulls_last=True)
    say(f"{name}: cells with p < 0.05", str(low.height) + "".join(
        f"{':' if i == 0 else ';'} {cell(x)} (μ̂ {bp(x['net_bps'])}, t {tt(x['t_hh'])})"
        for i, x in enumerate(low.iter_rows(named=True))))
    say(f"{name}: cells with μ̂ > 0 / gross > 0",
        f"{int((df['net_bps'] > 0).sum())} / {int((df['gross_bps'] > 0).sum())}")
    bh = df.sort("q_bh").row(0, named=True)
    say(f"{name}: smallest BH q, smallest BY q", f"{pv(bh['q_bh'])} ({cell(bh)}), {pv(df['q_by'].min())}")
    say(f"{name}: largest BH q, largest BY q", f"{pv(df['q_bh'].max())}, {pv(df['q_by'].max())}")


def secondary(s: pl.DataFrame, m: pl.DataFrame) -> None:
    heading("Secondary families (§7)")
    _family_lines("F_doge2024 (doge_2024)", s)
    pooled = m.filter(pl.col("instrument") == POOLED)
    _family_lines("F_majors (majors_2023, POOLED)", pooled)
    per = m.filter(pl.col("instrument") != POOLED)
    say("majors per instrument: cells with μ̂ > 0", ", ".join(
        f"{i} {n}" for i, n in per.group_by("instrument").agg((pl.col("net_bps") > 0).sum()).sort("instrument").rows()))
    elsewhere(["the share of doge_2024 that earlier research covered: §3 of the pre-registration"])


def _rows(r: pl.DataFrame, row: str, parameter: Optional[str] = None) -> pl.DataFrame:
    out = r.filter(pl.col("row") == row)
    return out if parameter is None else out.filter(pl.col("parameter") == parameter)


def robustness(p: pl.DataFrame, r: pl.DataFrame, z: Dict[str, float]) -> None:
    heading("Robustness (§8: reported, never decisive)")
    plac = p.filter(pl.col("p_placebo") <= 0.05)
    say("R1: cells with placebo p <= 0.05", f"{plac.height}: {cells(plac)}")
    fam = p.sort("p_placebo_family").row(0, named=True)
    say("R1: smallest family-wise placebo p", f"{pv(fam['p_placebo_family'])} ({fam['p_placebo_family']:.5f}) "
                                             f"at {cell(fam)}")
    say("R1: rolling_vol_ratio cells among them", int((plac["factor"] == "rolling_vol_ratio").sum()))
    fast = plac.filter((pl.col("bar") == "15m") & (pl.col("horizon") == 1))
    say("R1: 15m, h = 1 cells among them", fast.height)
    say("R1: their t from / to", f"{tt(fast['t_hh'].min())} / {tt(fast['t_hh'].max())}")
    say("R1: their μ̂ from / to", f"{bp(fast['net_bps'].min())} / {bp(fast['net_bps'].max())}")
    say("R1: their largest |gross|", f"{fast['gross_bps'].abs().max():.2f}")
    rsi = fast.filter(pl.col("factor") == "rsi_14")
    say("R1: rsi_14's gross (traded direction)", ", ".join(bp(g) for g in rsi["gross_bps"]))

    t2 = p["t_sqrt_h"]
    say(f"R2: cells with |t/sqrt(h)| >= {z['z_family']:.3f}", int((t2.abs() >= z["z_family"]).sum()))
    say("R2: largest t/sqrt(h), largest |t/sqrt(h)|", f"{tt(t2.max())}, {t2.abs().max():.2f}")
    say("R3: largest t in any quarter", tt(_rows(r, "R3")["t_hh"].max()))
    r4 = _rows(r, "R4").filter(pl.col("parameter").is_in(["k=5", "k=10"]))
    say("R4: largest t without the 5 or 10 most volatile days", tt(r4["t_hh"].max()))
    say("R7: largest t with winsorised returns", tt(_rows(r, "R7")["t_hh"].max()))

    r5 = _rows(r, "R5")
    say("R5: cells with a positive value / median (bp per day)",
        f"{int((r5['value_bps'] > 0).sum())} / {r5['value_bps'].median():+.1f}")
    top4 = r5.sort("value_bps", descending=True, nulls_last=True).head(4)
    say("R5: four largest", "; ".join(f"{cell(x)} {bp(x['value_bps'])}" for x in top4.iter_rows(named=True)))
    rvr15 = r5.filter((pl.col("factor") == "rolling_vol_ratio") & (pl.col("horizon") == 15))
    say("R5: rolling_vol_ratio at h = 15", "; ".join(f"{x['bar']} {bp(x['value_bps'])}"
                                                    for x in rvr15.iter_rows(named=True)))
    rl = _rows(r, "RL").filter(pl.col("factor") == "rolling_vol_ratio")
    say("RL: rolling_vol_ratio mean run (bars) and number of runs", "; ".join(
        f"{x['bar']} h={x['horizon']} {x['value_bps']:.2f} ({x['n_active']} runs)" for x in rl.iter_rows(named=True)))
    gross = _rows(r, "R6", "c=0").filter((pl.col("factor") == "rolling_vol_ratio") & (pl.col("bar") == "1H"))
    say("R6 c=0: rolling_vol_ratio 1H gross t", "; ".join(f"h={x['horizon']} {tt(x['t_hh'])}"
                                                         for x in gross.iter_rows(named=True)))
    for c in ("c=0", "c=4", "c=7", "c=11"):
        x = _rows(r, "R6", c)
        say(f"R6 {c}: cells with t >= {z['z_family']:.3f} / value > 0 / largest t / smallest t",
            f"{int((x['t_hh'] >= z['z_family']).sum())} / {int((x['value_bps'] > 0).sum())} / {tt(x['t_hh'].max())}"
            f" / {tt(x['t_hh'].min())}")

    r8 = _rows(r, "R8").join(p.select(KEY + [pl.col("n_active").alias("n_cell"), "net_bps"]), on=KEY)
    pos = r8.filter(pl.col("value_bps") > 0)
    say("R8: cells with a positive value", f"{pos.height}: " + "; ".join(
        f"{cell(x)} {bp(x['value_bps'])} (t {tt(x['t_hh'])}, {x['n_active']:,} of {x['n_cell']:,} trades, "
        f"μ̂ {bp(x['net_bps'])})" for x in pos.iter_rows(named=True)))
    r8 = r8.with_columns((pl.col("n_active") / pl.col("n_cell")).alias("share"))
    print("- R8: share of trades with R8 defined, by bar and h (median, min, max):")
    for g in by_bar_h(r8, pl.col("share").median().alias("med"), pl.col("share").min().alias("lo"),
                      pl.col("share").max().alias("hi")):
        print(f"    {g['bar']:>3} {g['horizon']:>2}  {g['med']:.0%} ({g['lo']:.0%}-{g['hi']:.0%})")

    ic = p.filter(pl.col("ic_p_holm") <= 0.05)
    say("R9: cells with an IC significant in its Holm family", ic.height)
    for x in ic.iter_rows(named=True):
        print(f"    {x['bar']:>3} {x['horizon']:>2} {x['factor']:<16} d {x['direction']:+d} IC {x['ic']:+.3f} "
              f"IC t {tt(x['ic_t'])} gross {bp(x['gross_bps'])}")
    opposite = ic.filter(pl.col("ic") * pl.col("direction") < 0)
    say("R9: IC sign opposite to d", f"{opposite.height} ({sorted(set(opposite['factor'].to_list()))})")
    say("R9: largest |gross| among them", f"{ic['gross_bps'].abs().max():.2f}")

    r10 = _rows(r, "R10").sort(KEY + ["parameter"]).group_by(KEY, maintain_order=True).agg(pl.col("value_bps"))
    r10 = r10.join(p.select(KEY + ["direction"]), on=KEY)
    strict = sum(bool(np.all(np.diff(np.array(v) * d) > 0)) for v, d in r10.select("value_bps", "direction").rows())
    weak = sum(bool(np.all(np.diff(np.array(v) * d) >= 0)) for v, d in r10.select("value_bps", "direction").rows())
    say("R10: cells whose quintile means rise in the traded direction (strictly / weakly)", f"{strict} / {weak}")
    allrl = _rows(r, "RL")
    say("RL: mean run length from / to (bars)", f"{allrl['value_bps'].min():.1f} / {allrl['value_bps'].max():.1f}")
    elsewhere(["'c = 4 bp is maker on both sides at §11's fees': Appendix A and §11",
               "R8's left-out trades end at the funding cut-over: see Data"])


def largest_cell(p: pl.DataFrame, s: pl.DataFrame, m: pl.DataFrame, r: pl.DataFrame,
                 ins: pl.DataFrame, fwd: pl.DataFrame) -> None:
    heading("The largest primary cell, descriptively")
    key = (pl.col("bar") == "1H") & (pl.col("horizon") == 5) & (pl.col("factor") == "rolling_vol_ratio")
    c = p.filter(key).row(0, named=True)
    say("primary", f"μ̂ {bp(c['net_bps'])}, t {tt(c['t_hh'])}, one-sided p {pv(c['p_one_sided'])}, "
                   f"Holm p {pv(c['p_holm'])}, family-wise placebo p {pv(c['p_placebo_family'])}")
    say("only cell with p < 0.05", int((p["p_one_sided"] < 0.05).sum()) == 1 and c["p_one_sided"] < 0.05)
    say("smallest family-wise placebo p is this cell's", p["p_placebo_family"].min() == c["p_placebo_family"])
    x = s.filter(key).row(0, named=True)
    say("doge_2024", f"μ̂ {bp(x['net_bps'])} (t {tt(x['t_hh'])})")
    for inst in (POOLED, "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"):
        x = m.filter(key & (pl.col("instrument") == inst)).row(0, named=True)
        say(f"majors_2023 {inst}", f"μ̂ {bp(x['net_bps'])}" + (f" (t {tt(x['t_hh'])})" if inst == POOLED else ""))
    rc = r.filter(key)
    q = _rows(rc, "R3").sort("parameter")
    say("R3 by quarter: μ̂", ", ".join(bp(v) for v in q["value_bps"]))
    say("R3 by quarter: t", ", ".join(tt(v) for v in q["t_hh"]))
    say("R3 by quarter: trades", ", ".join(f"{n:,}" for n in q["n_active"]))
    r4 = {x["parameter"]: x for x in _rows(rc, "R4").iter_rows(named=True)}
    say("R4 k=5, k=10", f"μ̂ {bp(r4['k=5']['value_bps'])} (t {tt(r4['k=5']['t_hh'])}), "
                        f"{bp(r4['k=10']['value_bps'])} (t {tt(r4['k=10']['t_hh'])}); k=0 {bp(r4['k=0']['value_bps'])}")
    dropped = r4["k=0"]["n_active"] - r4["k=5"]["n_active"]
    say("R4: trades touching the 5 days / all trades, and the fall in μ̂",
        f"{dropped:,} / {r4['k=0']['n_active']:,}, "
        f"{1 - r4['k=5']['value_bps'] / r4['k=0']['value_bps']:.0%}")
    r7 = _rows(rc, "R7").row(0, named=True)
    say("R7", f"μ̂ {bp(r7['value_bps'])} (t {tt(r7['t_hh'])})")

    rvr = pl.col("factor") == "rolling_vol_ratio"
    i6 = ins.filter(rvr).with_columns((pl.col("mean_s_fwd_bps") / pl.col("se_bps")).alias("t_is"))
    say("in-sample directions of rolling_vol_ratio", "; ".join(
        f"{x['bar']} h={x['horizon']} {x['direction']:+d}" for x in i6.iter_rows(named=True)))
    say("in-sample t of the means that set them (mean_s_fwd / SE), from / to",
        f"{tt(i6['t_is'].min())} / {tt(i6['t_is'].max())}")
    lowest = p.sort("net_bps").row(0, named=True)
    say("lowest μ̂ in the primary family", f"{bp(lowest['net_bps'])} (t {tt(lowest['t_hh'])}) at {cell(lowest)}")
    for bar in ("1H", "15m"):
        k15 = (pl.col("bar") == bar) & (pl.col("horizon") == 15) & rvr
        a, b = p.filter(k15).row(0, named=True), s.filter(k15).row(0, named=True)
        say(f"rolling_vol_ratio {bar} h=15 (d {a['direction']:+d})",
            f"doge_2023 μ̂ {bp(a['net_bps'])} (t {tt(a['t_hh'])}), doge_2024 μ̂ {bp(b['net_bps'])}")
    six = p.filter(rvr)
    say("rolling_vol_ratio IC > 0 in", f"{int((six['ic'] > 0).sum())} of {six.height} cells; smallest IC Holm p "
                                       f"{pv(six['ic_p_holm'].min())}")
    rl = _rows(r, "RL").sort("value_bps", descending=True, nulls_last=True).row(0, named=True)
    say("most persistent signal (largest RL)", f"{cell(rl)} {rl['value_bps']:.1f} bars")
    x = fwd.filter(key).row(0, named=True)
    say("doge_fwd (next section)", f"μ̂ {bp(x['net_bps'])}, 95% interval [{bp(x['ci95_lo_bps'])}, "
                                   f"{bp(x['ci95_hi_bps'])}]")


def settlement_times(entry: Dict[str, Any]) -> np.ndarray:
    """
    Every settlement time of a funding series, rebuilt from its manifest entry.

    The manifest lists each spacing other than 8 h as [previous, next]
    (spacing_other_than_expected); every other spacing is exactly 8 h. So a
    walk from first_ts gives every timestamp, and it must land on last_ts
    after rows - 1 steps, having used every listed spacing, or the entry is
    not what this assumes and the script stops.
    """
    odd = {int(a): int(b) for a, b in entry["spacing_other_than_expected"]}
    ts = [int(entry["first_ts"])]
    for _ in range(int(entry["rows"]) - 1):
        ts.append(odd.pop(ts[-1], ts[-1] + FUNDING_STEP_MS))
    if odd or ts[-1] != int(entry["last_ts"]):
        raise SystemExit(f"{entry['path']}: its manifest's spacings do not rebuild its {entry['rows']} timestamps")
    return np.array(ts, dtype=np.int64)


def grid_offset_ms(ts: np.ndarray) -> np.ndarray:
    """How far each stamp is from the nearest 8-hour grid time, signed (late > 0)."""
    half = FUNDING_STEP_MS // 2
    return (ts + half) % FUNDING_STEP_MS - half


def completeness(name: str, mf: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Whether each series of a set's manifest fills the manifest's window: a
    candle in every minute, a funding settlement in every 8-hour slot.
    Returns the funding entries as (set, instrument, entry).
    """
    lo, hi = int(mf["window"]["start_ms"]), int(mf["window"]["end_ms"])
    funding: List[Tuple[str, str, Dict[str, Any]]] = []
    for e in mf["series"]:
        inst, rows = e["inst_id"], int(e["rows"])
        if e["kind"] == "candles":
            full = (hi - lo) // MINUTE_MS
            ok = (rows == full and not e["gaps"] and not e["edge_gaps"]
                  and int(e["first_ts"]) == lo and int(e["last_ts"]) == hi - MINUTE_MS)
            say(f"{name} {inst} candles", f"{rows:,} rows, {full:,} minutes in the window, gaps {len(e['gaps'])}"
                                          f"{'' if ok else ' (NOT COMPLETE)'}")
        else:
            funding.append((name, inst, e))
            grid = settlement_times(e) - grid_offset_ms(settlement_times(e))
            slots = (hi - lo) // FUNDING_STEP_MS
            filled = len(np.unique(grid)) == slots and grid.min() >= lo and grid.max() < hi
            say(f"{name} {inst} funding", f"{rows:,} settlements, {slots:,} 8-hour slots in the window, "
                                          f"{'one in each' if filled else 'NOT one in each'}")
    return funding


def data(p: pl.DataFrame, sets: Dict[str, Dict[str, Any]]) -> None:
    heading("Data")
    funding: List[Tuple[str, str, Dict[str, Any]]] = []
    for name in ("doge_2023", "doge_2024", "majors_2023"):
        funding += completeness(name, load_json(MANIFESTS[name]))
    for e in load_json(MANIFESTS["is_supp"])["series"]:
        if e["kind"] == "funding":
            funding.append(("is_supp", e["inst_id"], e))

    # When the late stamps stop: the last grid time with an off-grid stamp, over all files.
    late_grid = [int(t - o) for _, _, e in funding
                 for t, o in zip(settlement_times(e), grid_offset_ms(settlement_times(e))) if o != 0]
    cut = max(late_grid)
    say("last grid time with an off-grid stamp (any file)", utc(cut))
    say("first grid time after it", utc(cut + FUNDING_STEP_MS))
    for name, inst, e in funding:
        ts = settlement_times(e)
        off = grid_offset_ms(ts)
        grid = ts - off
        late = off != 0
        upto = grid <= cut
        label = f"{name} {inst}"
        if not late.any():
            say(f"{label} off-grid stamps", f"none of {len(ts):,}")
            continue
        say(f"{label} off-grid stamps", f"{int(late.sum())} of {len(ts):,} ({int(late[upto].sum())} of the "
                                        f"{int(upto.sum())} up to the cut-over), late by {off[late].min() / 1000:g} "
                                        f"to {off[late].max() / 1000:g} s, none after {utc(int(grid[late].max()))}")
        if name not in sets:
            continue
        start, end = int(sets[name]["start_ms"]), int(sets[name]["end_ms"])
        win = (grid >= start) & (grid < end)
        say(f"{label} in the evaluation window", f"{int(late[win].sum())} of {int(win.sum()):,} off the grid "
                                                 f"({int(late[win].sum())} of the {int((win & upto).sum())} "
                                                 "up to the cut-over)")
        # The bar that starts at a late stamp's grid time sees the previous
        # settlement (as-of at or before the bar's start): fresh if that one
        # is less than 8 h before the bar (itself late), stale if it is
        # exactly 8 h before (on the grid), §4.2's staleness test being >=.
        idx = np.flatnonzero(late & win)
        age = grid[idx] - ts[idx - 1]
        say(f"{label} funding_cost_pressure on those bars (each bar grid)",
            f"{int((age < FUNDING_STEP_MS).sum())} take the previous value, "
            f"{int((age >= FUNDING_STEP_MS).sum())} are undefined")
        spacings = np.diff(ts)
        odd = spacings != FUNDING_STEP_MS
        say(f"{label} spacings other than 8 h (manifest)",
            f"{int(odd.sum())}, off by at most {np.abs(spacings[odd] - FUNDING_STEP_MS).max() / 1000:g} s; "
            f"longer than 8 h (the manifest's gaps): {int((spacings > FUNDING_STEP_MS).sum())} "
            f"(listed: {len(e['gaps'])})")
    primary = sets.get("doge_2023")
    if primary is not None:
        months = (cut - int(primary["start_ms"])) / (30.44 * 86_400_000)
        say("cut-over, in months after the primary window starts", f"{months:.1f}")
    fcp = p.filter(pl.col("factor") == "funding_cost_pressure")
    say("funding_cost_pressure on doge_2023: largest t", tt(fcp["t_hh"].max()))


def in_sample(ins: pl.DataFrame, z: Dict[str, float]) -> None:
    """
    The in-sample year as the report, docs/factor-research-v2.md, sums it up
    (its §4, "In sample"); the results document quotes it only for
    rolling_vol_ratio. The directions are all it fixed, so the question is
    how much support they had there.
    """
    heading("In sample (results/insample_v2.csv; quoted by docs/factor-research-v2.md, §4)")
    d = dict(ins.group_by("direction").len().iter_rows())
    say("directions +1 / -1", f"{d.get(1, 0)} / {d.get(-1, 0)} of {ins.height}")
    say("flagged directions", int(ins["flagged"].sum()))
    # mean_s_fwd is the mean of signal x forward return, before costs, so
    # d * mean_s_fwd is the gross mean in the traded direction.
    ins = ins.with_columns((pl.col("direction") * pl.col("mean_s_fwd_bps")).alias("gross"))
    ins = ins.with_columns(((pl.col("gross") - z["cost"]) / pl.col("se_bps")).alias("t_net"))
    top = ins.sort("t_net", descending=True, nulls_last=True).row(0, named=True)
    say(f"largest net t, (d·mean_s_fwd - {z['cost']:g})/SE", f"{tt(top['t_net'])} at {cell(top)}")
    say(f"cells with net t above {z['z_ub95']:.3f}", int((ins["t_net"] > z["z_ub95"]).sum()))
    say(f"cells whose gross mean in the traded direction exceeds the {z['cost']:g} bp cost",
        f"{int((ins['gross'] > z['cost']).sum())}: {cells(ins.filter(pl.col('gross') > z['cost']))}")
    elsewhere(["the report's §1 table of 1-minute returns: candles, not results/ "
               "(scripts/make_figures.py prints it as '§1 table')"])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rank correlation: Pearson's on the ranks, tied values sharing their mean rank."""
    ra, rb = (pl.Series(v).rank("average").to_numpy() for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def doge_fwd(p: pl.DataFrame, s: pl.DataFrame, f: pl.DataFrame, spec: Dict[str, Any]) -> None:
    """
    The descriptive set, which decides nothing (§7): its window and warm-up
    (Appendix A), its record and data (its manifest, and the commit of its
    results as the maker probe's archive manifest names it), what the
    document reads off its cells, and then every cell's μ̂ and 95% interval,
    which is what §7 says the set reports.
    """
    heading("`doge_fwd`, descriptively (§4.1, §7)")
    win, ins = spec["sets"]["doge_fwd"], spec["sets"]["IS"]
    start, end = int(win["start_ms"]), int(win["end_ms"])
    say("evaluation window (Appendix A)", f"{utc(start)} to {utc(end)} UTC, {(end - start) / DAY_MS:g} days, "
                                          f"from the in-sample year's end: {start == int(ins['end_ms'])}")
    half = pl.concat([f["ci95_hi_bps"] - f["net_bps"], f["net_bps"] - f["ci95_lo_bps"]])
    z_ci = one_value(half / pl.concat([f["se_bps"]] * 2), "(ci95 - net)/SE on either side")
    say("95% interval = μ̂ ± z·SE with z", f"{z_ci:.3f}")
    say("status counts", ", ".join(f"{k} {v}" for k, v in sorted(f.group_by("status").len().iter_rows())))

    mf = load_json(MANIFESTS["doge_fwd"])
    say("code_commit of every row", ", ".join(c[:7] for c in f["code_commit"].unique(maintain_order=True)))
    say("manifest created (UTC)", mf["created_utc"])
    say("public main when its data was fetched (manifest remote_main_commit)", mf["remote_main_commit"][:7])
    say("the commit that added the results (doge_fwd_results_commit, the maker probe's archive manifest)",
        load_json(MANIFESTS["maker_probe_archives"])["doge_fwd_results_commit"][:7])
    warm, lo, hi = int(win["warmup_start_ms"]), int(ins["warmup_start_ms"]), int(ins["end_ms"])
    say("warm-up from (Appendix A)", f"{utc(warm)} UTC, inside the in-sample data ({utc(lo)} to {utc(hi)}): "
                                     f"{lo <= warm and start <= hi}")
    for _, inst, e in completeness("doge_fwd", mf):
        off = grid_offset_ms(settlement_times(e))
        late = int((off != 0).sum())
        say(f"doge_fwd {inst} off-grid stamps", f"{late} of {len(off):,}" if late else f"none of {len(off):,}")

    say("95% interval above 0 / entirely below 0", f"{int((f['ci95_lo_bps'] > 0).sum())} / "
                                                   f"{int((f['ci95_hi_bps'] < 0).sum())} of {f.height}")
    pos = f.filter(pl.col("net_bps") > 0)
    say("cells with μ̂ > 0", f"{pos.height}: " + ", ".join(f"{g['bar']} h={g['horizon']} ({g['n']})"
                                                         for g in by_bar_h(pos, pl.len().alias("n"))))
    width = pos["ci95_hi_bps"] - pos["ci95_lo_bps"]
    say("their 95% intervals' width, narrowest / widest", f"{width.min():.1f} / {width.max():.1f} bp")
    top = f.sort("net_bps", descending=True).row(0, named=True)
    say("largest μ̂", f"{bp(top['net_bps'])} at {cell(top)}, 95% interval [{bp(top['ci95_lo_bps'])}, "
                     f"{bp(top['ci95_hi_bps'])}]")
    one = f.filter((pl.col("bar") == "15m") & (pl.col("horizon") == 1))
    say(f"15m, h = 1 ({one.height} cells): gross means from / to",
        f"{bp(one['gross_bps'].min())} / {bp(one['gross_bps'].max())}")
    say("15m, h = 1: μ̂ from / to", f"{bp(one['net_bps'].min())} / {bp(one['net_bps'].max())}")
    say("15m, h = 1: highest upper end of a 95% interval", bp(one["ci95_hi_bps"].max()))
    best = p.sort("t_hh", descending=True, nulls_last=True).row(0, named=True)
    key = (pl.col("bar") == best["bar"]) & (pl.col("horizon") == best["horizon"]) & (pl.col("factor") == best["factor"])
    x, y = f.filter(key).row(0, named=True), s.filter(key).row(0, named=True)
    say(f"the largest primary cell, {cell(best)}", f"μ̂ {bp(x['net_bps'])}, against {bp(best['net_bps'])} on "
                                                   f"doge_2023 and {bp(y['net_bps'])} on doge_2024")
    both = f.select(KEY + ["net_bps"]).join(p.select(KEY + [pl.col("net_bps").alias("primary")]), on=KEY)
    a, b = both["net_bps"].to_numpy(), both["primary"].to_numpy()
    say("cells whose μ̂ has the sign it has on doge_2023",
        f"{int((np.sign(a) == np.sign(b)).sum())} of {len(a)} (both negative {int(((a < 0) & (b < 0)).sum())}, "
        f"both positive {int(((a > 0) & (b > 0)).sum())})")
    say("rank correlation of the cells' μ̂ with doge_2023's (Spearman)", f"{spearman(a, b):.2f}")
    print("- every cell (bar, h, factor, d, μ̂, 95% interval):")
    for r in f.iter_rows(named=True):
        print(f"    {r['bar']:>3} {r['horizon']:>2} {r['factor']:<22} {r['direction']:+d} {bp(r['net_bps']):>6} "
              f"[{bp(r['ci95_lo_bps'])}, {bp(r['ci95_hi_bps'])}]")
    elsewhere([
        "the commit times, and the paths git diff listed at the run's HEAD: git (the manifest's created_utc "
        "above is when fetch_v2 wrote it; the commit that added it came shortly after)",
        "the check that the window had closed: scripts/fetch_v2.py and scripts/run_oos.py (guard 6)",
        "the earlier contact, 2026-10-03 16:00 -> 2026-10-05 00:00 UTC: §3 of the pre-registration",
    ])


def summary_differences(summary: pl.DataFrame, daily: pl.DataFrame) -> float:
    """
    The largest difference between summary.csv and the same numbers from
    daily.csv, or infinity if a row or a blank differs. Per run, side and
    model: a count's total is its sum over the days; a mean metric's mean
    is the mean of its daily values over the days on which it is defined,
    each day weighted equally, and its SE is their SD (ddof 1) / √days.
    """
    want: Dict[Tuple[str, str, str, str], Dict[str, Optional[float]]] = {}
    for (run, side, model), g in daily.group_by(["run", "side", "model"], maintain_order=True):
        for c in g.columns:
            if c.startswith("n_"):
                want[(run, side, model, c)] = {"n_days": float(g.height), "total": float(g[c].sum())}
            elif c == "fill_rate" or c.endswith("_bps_mean"):
                v = g[c].drop_nulls().to_numpy()
                v = v[np.isfinite(v)]
                want[(run, side, model, c)] = {
                    "n_days": float(v.size), "mean": float(v.mean()) if v.size else None,
                    "se": float(np.std(v, ddof=1) / math.sqrt(v.size)) if v.size >= 2 else None}
    got = {(r["run"], r["side"], r["model"], r["metric"]): r for r in summary.iter_rows(named=True)}
    if set(want) != set(got):
        return math.inf
    worst = 0.0
    for key, fields in want.items():
        for name, value in fields.items():
            other = got[key][name]
            if (value is None) != (other is None):
                return math.inf
            if value is not None:
                worst = max(worst, abs(value - other))
    return worst


def probe_total(summary: pl.DataFrame, run: str, metric: str) -> int:
    """A count's total over the days in summary.csv, the same for each side and model of the run."""
    rows = summary.filter((pl.col("run") == run) & (pl.col("metric") == metric))
    return int(one_value(rows["total"], f"{run} {metric}"))


def maker_probe(z: Dict[str, float], spec: Dict[str, Any]) -> None:
    """
    §11's probe, which informs the discussion of cost only: its record (the
    archive manifest, and the commit every output row names), its design
    (Appendix A, the freeze amendment's days, the candidate list), the
    checks of each day (checks.csv), and the means over the days with their
    SE (summary.csv, recomputed from daily.csv first), as the document's
    table and reading quote them.
    """
    heading("Maker probe (§11, exploratory)")
    summary, daily, checks = load(PROBE_SUMMARY), load(PROBE_DAILY), load(PROBE_CHECKS)
    arch, plan = load_json(MANIFESTS["maker_probe_archives"]), spec["maker_probe"]
    say("c, which the probe never replaces (Terms)", f"{z['cost']:g} bp")

    codes = sorted({c for df in (summary, daily, checks) for c in df["code_commit"].to_list()})
    say("code_commit of every row of summary.csv, daily.csv and checks.csv", ", ".join(c[:7] for c in codes))
    files = arch["files"]
    kinds = ", ".join(f"{sum(x['kind'] == k for x in files)} {k}" for k in sorted({x["kind"] for x in files}))
    size = sum(int(x["bytes"]) for x in files)
    say("archive files (manifest_archives.json)", f"{len(files)} ({kinds}), {size:,} bytes ({size / 1e9:.1f} GB)")
    fetched = sorted(x["downloaded_utc"] for x in files if x.get("downloaded_utc"))
    say("downloaded (UTC), first / last", f"{fetched[0]} / {fetched[-1]}")
    say("archive manifest created (UTC)", arch["created_utc"])
    say("doge_fwd's results commit, required on the public main before the download (doge_fwd_results_commit)",
        arch["doge_fwd_results_commit"][:7])

    days = list(arch["days"])
    same = days == maker_probe_days(REPO_ROOT / spec["freeze"]["amendment_file"]) == checks["day"].to_list()
    say("days (Appendix A)", f"{plan['n_days']}; the freeze amendment's, the archive manifest's and checks.csv's "
                             f"are the same {len(days)}: {same and len(days) == plan['n_days']}")
    cand = load_json(MANIFESTS["maker_probe"])
    say("candidate days they were drawn from (results/manifest_maker_probe.json)",
        f"{cand['n_candidates']}; every day among them: {set(days) <= set(cand['candidates'])}")
    every = int(plan["decision_every_minutes"])
    per_day = DAY_MS // (every * MINUTE_MS)
    skips = [c for c in daily.columns if c.startswith("n_skipped_")]
    tried = daily["n_decisions"] + sum(daily[c] for c in skips)
    say(f"decisions a day, {every} minutes apart (Appendix A)",
        f"{per_day}; decisions and skips add up to it in every row of daily.csv: {bool((tried == per_day).all())}")
    say("each order (Appendix A)", f"{plan['size_contracts']} contract for {plan['order_life_seconds']} s")
    fees = summary.select("maker_fee_bps", "taker_fee_bps", "fees_verified").unique(maintain_order=True).rows()
    say("fees in every row of summary.csv: maker / taker, verified", "; ".join(f"{m:g} / {t:g} bp, {v}"
                                                                              for m, t, v in fees))

    say("days checked (checks.csv)", checks.height)
    say("book pauses of 5 s or more, counted as missing data (book_pauses_as_gaps), all days",
        int(checks["book_pauses_as_gaps"].sum()))
    pause = checks.sort("book_longest_pause_ms", descending=True).row(0, named=True)
    say("longest pause", f"{pause['book_longest_pause_ms'] / 1000:.1f} s ({int(pause['book_longest_pause_ms']):,} ms, "
                         f"{pause['day']})")
    say("trade id jumps / book messages stamped before the one before them (book_clamped), all days",
        f"{int(checks['trades_id_jumps'].sum())} / {int(checks['book_clamped'].sum())}")
    say("gaps from missing data, book / trades (parser_gaps, trade_gaps), all days",
        f"{int(checks['parser_gaps'].sum())} / {int(checks['trade_gaps'].sum())}")
    say("parser_gaps: decisions skipped for a gap", probe_total(summary, "parser_gaps", "n_skipped_gap"))
    scheduled = per_day * len(days)
    adds_up = all(sum(probe_total(summary, run, c) for c in ["n_decisions"] + skips) == scheduled for run in PROBE_RUNS)
    say("decisions per side and model", f"{scheduled:,} ({len(days)} days x {per_day}); decisions and skips add up "
                                        f"to it in both runs: {adds_up}")
    # Which decision each day skipped for want of data: two columns of orders.csv.
    orders = pl.read_csv(REPO_ROOT / PROBE_ORDERS, columns=["decision_ts", "status"], infer_schema_length=0)
    no_data = orders.filter(pl.col("status") == "no_data")["decision_ts"].cast(pl.Int64)
    at_midnight = f"{int((no_data % DAY_MS == 0).sum())} of {no_data.len()}"
    say("skipped for no data (n_skipped_no_data), each run, side and model",
        f"{probe_total(summary, 'parser_gaps', 'n_skipped_no_data')} of {scheduled:,}; one on each day: "
        f"{bool((daily['n_skipped_no_data'] == 1).all())}; each at 00:00 (orders.csv): {at_midnight}")
    say("most levels a side of the rebuilt book held, bid / ask, any day",
        f"{int(checks['max_depth_bid'].max())} / {int(checks['max_depth_ask'].max())}")
    say("share of trades followed within 100 ms by a decrease of the level they hit (clock_matched_share), "
        "by day, from / to", f"{checks['clock_matched_share'].min():.0%} / {checks['clock_matched_share'].max():.0%}")
    say("median lag (clock_lag_ms_median), by day, from / to",
        f"{checks['clock_lag_ms_median'].min():g} / {checks['clock_lag_ms_median'].max():g} ms")
    say("share of those lags within 10 ms (clock_lag_within_10ms_share), by day, from / to",
        f"{checks['clock_lag_within_10ms_share'].min():.0%} / {checks['clock_lag_within_10ms_share'].max():.0%}")
    ratio = checks["clock_decrease_over_size_median"]
    say("median ratio of the decrease to the trade's size, trades alone at their price in their ms "
        "(clock_decrease_over_size_median), by day, from / to", f"{ratio.min():.1f} / {ratio.max():.1f}")
    n, k = int(checks["snapshot_sides_checked"].sum()), int(checks["snapshot_sides_touch_mismatch"].sum())
    price = int(checks["touch_mismatch_price_differs"].sum())
    say("snapshot sides that differ from the rebuilt book at the touch",
        f"{k:,} of {n:,} ({k / n:.1%}), {price} in price, {k - price:,} in size only")
    say("touch_gaps: hours dropped (touch_gap_ms), of the days' hours",
        f"{checks['touch_gap_ms'].sum() / HOUR_MS:.1f} of {24 * checks.height}")
    dropped = probe_total(summary, "touch_gaps", "n_skipped_gap")
    simulated = probe_total(summary, "parser_gaps", "n_decisions")
    say("touch_gaps: decisions skipped for a gap, per side and model, of those parser_gaps simulates",
        f"{dropped:,} of {simulated:,}")
    src = checks.filter(pl.col("trades_source_1_left_out") > 0)
    say("trades marked source 1, left out", f"{int(src['trades_source_1_left_out'].sum()):,}, on "
                                            f"{', '.join(src['day'].to_list())}")

    say("summary.csv recomputed from daily.csv (totals; means over the days, each weighted equally; "
        "SE = SD/√days), to 1e-9", summary_differences(summary, daily) <= 1e-9)
    say("days in every mean and total (n_days)", int(one_value(summary["n_days"], "n_days")))
    cross = summary.filter(pl.col("metric") == "shortfall_cross_now_bps_mean")
    now = {run: (one_value(cross.filter(pl.col("run") == run)["mean"], f"{run} crossing at once"),
                 one_value(cross.filter(pl.col("run") == run)["se"], f"{run} crossing at once, SE"))
           for run in PROBE_RUNS}
    say("crossing at once, mean (SE), the same for each side and model",
        "; ".join(f"{run} {mean:.2f} ({se:.2f})" for run, (mean, se) in now.items()))
    taker = one_value(summary["taker_fee_bps"], "taker fee")
    say(f"crossing at once less the {taker:g} bp taker fee (half the spread)",
        "; ".join(f"{run} {mean - taker:.2f}" for run, (mean, _) in now.items()))
    table = ("fill_rate", "markout_1m_bps_mean", "markout_5m_bps_mean", "markout_15m_bps_mean",
             "shortfall_post_then_cross_bps_mean", "shortfall_diff_bps_mean")
    unsigned = ("fill_rate", "shortfall_post_then_cross_bps_mean")
    rows = {(x["run"], x["side"], x["model"], x["metric"]): x for x in summary.iter_rows(named=True)}
    print("- mean (SE) of each run, side and model: fill rate, markout 1 / 5 / 15 min, post then cross, difference:")
    for run, side, model in dict.fromkeys(summary.select("run", "side", "model").rows()):
        values = []
        for metric in table:
            x = rows[(run, side, model, metric)]
            values.append(f"{x['mean']:{'.2f' if metric in unsigned else '+.2f'}} ({x['se']:.2f})")
        print(f"    {run:<11} {side:<4} {model}  " + "  ".join(values))

    fill = summary.filter(pl.col("metric") == "fill_rate")
    spans = []
    for model in sorted(set(fill["model"])):
        rates = fill.filter(pl.col("model") == model)["mean"]
        spans.append(f"{model} {rates.min():.0%} / {rates.max():.0%}")
    say("fill rate from / to, both runs and sides", "; ".join(spans))
    one = summary.filter(pl.col("metric") == "markout_1m_bps_mean")["mean"]
    say("1-minute markout from / to", f"{one.min():+.2f} / {one.max():+.2f} bp")
    late = summary.filter(pl.col("metric") == "markout_15m_bps_mean")["se"]
    say("15-minute markout's SE from / to", f"{late.min():.2f} / {late.max():.2f}")
    post = summary.filter(pl.col("metric") == "shortfall_post_then_cross_bps_mean")
    say("post, then cross, from / to", f"{post['mean'].min():.2f} / {post['mean'].max():.2f} bp")
    diff = summary.filter(pl.col("metric") == "shortfall_diff_bps_mean")
    say("difference from / to; its SE from / to", f"{diff['mean'].min():+.2f} / {diff['mean'].max():+.2f} bp; "
                                                  f"{diff['se'].min():.2f} / {diff['se'].max():.2f}")
    first, second = (diff.filter(pl.col("run") == run).select("side", "model", "mean") for run in PROBE_RUNS)
    pair = first.join(second, on=["side", "model"])
    say("largest gap between the two runs' differences, same side and model",
        f"{(pair['mean'] - pair['mean_right']).abs().max():.3f} bp")
    trips = []
    for run in PROBE_RUNS:
        legs = post.filter(pl.col("run") == run)          # a buy and a sell for each model
        trips.append(f"{run} " + " / ".join(f"{legs.filter(pl.col('model') == model)['mean'].sum():.2f}"
                                            for model in ("M2", "M1")))
    say("a round trip posting first (buy + sell post then cross), M2 / M1", "; ".join(trips))
    say("a round trip crossing twice (2 x crossing at once)", "; ".join(f"{run} {2 * mean:.2f}"
                                                                        for run, (mean, _) in now.items()))
    elsewhere([
        "the other commit hashes, and every commit time: git",
        "the §12 rows those commits added (three on the files' format, one on source-1 trades), and the first "
        "attempt, which stopped at the first trade file with a source column (the second day's) and wrote "
        "nothing: §12 of the pre-registration (2026-11-24)",
        "the 5 s pause counted as missing data, the 100 ms window of the clock check and the 0.01-contract size "
        "unit: research/maker_probe.py (MAX_SILENCE_MS, trade_book_lags, SIZE_UNIT)",
        "the first snapshot stamped a few ms after 00:00, and the book's updates in batches about 10 ms apart: "
        "the archive files (research/maker_probe.py's notes on their format)",
    ])


def not_done() -> None:
    heading("Not done")
    say("numbers this section quotes", "none")


def main() -> int:
    pl.Config.set_tbl_rows(100)
    p, r, s, m, fwd, ins = (load(x) for x in (PRIMARY, ROBUSTNESS, REPLICATION, MAJORS, FORWARD, INSAMPLE))
    spec = load_spec(REPO_ROOT / "docs" / "preregistration.md")
    print("# Results compared with the pre-registration")
    print("(every number docs/preregistration-results.md quotes, recomputed from the committed results files)")
    z = terms(p)
    record(p, s, m)
    headline(p, r, z)
    no_pay(p)
    power(p, z)
    secondary(s, m)
    robustness(p, r, z)
    largest_cell(p, s, m, r, ins, fwd)
    doge_fwd(p, s, fwd, spec)
    data(p, spec["sets"])
    maker_probe(z, spec)
    not_done()
    in_sample(ins, z)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
