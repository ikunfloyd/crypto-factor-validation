"""
The out-of-sample study end to end (docs/preregistration.md §4-§10).

    load_set_data   1-minute perp and spot candles and funding, per instrument
    build_bars      complete 15m / 1H bars, their spot legs and the 4x HTF bars
    prepare_frames  the ten factors on each instrument's bar sequence
    evaluate_cell   one (instrument, bar, h, factor): buckets, trades, §6.7
                    statistics and, on the primary set, the robustness rows
    run_insample    -> rows of results/insample_v2.csv (§6.5)
    run_oos_set     -> rows of results/oos/<set>.csv (§10) and, on the
                       primary set, of results/oos/robustness_doge_2023.csv

The formulas live in research.stats and the factors in research.factors; this
module only wires them together, so every formula has one home and one set of
tests. What lives here is what those modules cannot know on their own:

  * **Which data a set may read.** Each set reads only its own directory
    (doge_fwd also reads the in-sample files for its warm-up, as §4.1 says),
    and verify_inputs refuses any file whose SHA-256 differs from what was
    committed, or that no committed manifest lists. An input that was never
    downloaded is not an error: its factor's cells are untestable (§6.8).

  * **Families.** A p-value means what §7 says only once the right 60 cells
    share one Holm correction. Untestable cells enter with p = 1 instead of
    being dropped: dropping them would shrink m and loosen every threshold
    in a way the pre-registration did not fix.

  * **The §10 fill rules.** Every statistic is computed for every row, and
    the columns a set's rows may carry are then chosen by one table
    (filled_columns). Blanking by rule rather than by "whatever happened to be
    computed" keeps, say, a per-instrument majors row from ever showing a p.

Nothing here touches the network, and nothing here writes a file: the
scripts do that, write-once.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from research import stats as S
from research.bars import bar_ms, htf_bar, resample_complete
from research.factors import compute_factors, irregular_settlements
from research.freeze import INSAMPLE_COLUMNS, CellKey
from research.io import sha256_file
from research.spec import REPO_ROOT, get_spec

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
BPS = 1e4
POOLED = "POOLED"

# ---------------------------------------------------------------- layout on disk

# File names as scripts/fetch_v2.py writes them: <out-dir>/<instId>/<file>.
CANDLE_FILE = "candles_1m.parquet"
FUNDING_FILE = "funding_rates.parquet"
IS_SUPP_DIR = Path("data/history/is_supp")
OOS_DATA_DIR = Path("data/history/oos")
RESULTS_DIR = Path("results")
OOS_RESULTS_DIR = RESULTS_DIR / "oos"
IS_SUPP_MANIFEST = RESULTS_DIR / "manifest_is_supp.json"
INSAMPLE_RESULTS = Path(get_spec()["freeze"]["directions_file"])


def oos_manifest_path(set_name: str) -> Path:
    """results/oos/manifest_<set>.json, committed before the set is analysed (§10 step 4)."""
    return OOS_RESULTS_DIR / f"manifest_{set_name}.json"


def oos_results_path(set_name: str) -> Path:
    return OOS_RESULTS_DIR / f"{set_name}.csv"


def robustness_results_path(set_name: str) -> Path:
    return OOS_RESULTS_DIR / f"robustness_{set_name}.csv"


# ---------------------------------------------------------------- roles

ROLE_IS = "direction_and_power"
ROLE_PRIMARY = "primary"
ROLE_REPLICATION = "secondary_replication"
ROLE_POOLED = "secondary_pooled"
ROLE_DESCRIPTIVE = "descriptive"
_ROLES = (ROLE_IS, ROLE_PRIMARY, ROLE_REPLICATION, ROLE_POOLED, ROLE_DESCRIPTIVE)


def set_role(set_name: str, spec: Optional[Dict[str, Any]] = None) -> str:
    sets = (spec or get_spec())["sets"]
    if set_name not in sets:
        raise ValueError(f"unknown set {set_name!r}; Appendix A defines {sorted(sets)}")
    role = sets[set_name]["role"]
    if role not in _ROLES:
        raise ValueError(f"set {set_name!r} has role {role!r}, which this pipeline does not implement")
    return role


def set_with_role(role: str, spec: Optional[Dict[str, Any]] = None) -> str:
    """The one set that plays ``role`` (the roles are what §7 assigns, the names are labels)."""
    names = [n for n, s in (spec or get_spec())["sets"].items() if s["role"] == role]
    if len(names) != 1:
        raise ValueError(f"expected exactly one set with role {role!r}, found {names}")
    return names[0]


# ---------------------------------------------------------------- results columns

KEY_COLUMNS: Tuple[str, ...] = ("instrument", "bar", "horizon", "factor")
# §10 "Results files", in the order the pre-registration lists them.
OOS_VALUE_COLUMNS: Tuple[str, ...] = (
    "direction", "direction_flagged", "n_eval", "n_active", "share_top", "share_bottom",
    "gross_bps", "net_bps", "se_bps", "se_method", "t_hh", "p_one_sided", "p_holm",
    "q_bh", "q_by", "ub95_net_bps", "ub95_family_net_bps", "ci95_lo_bps", "ci95_hi_bps",
    "mde_bps", "mde_realised_bps",
    "p_placebo", "p_placebo_family", "t_sqrt_h", "net_funding_bps", "ic", "ic_t",
    "ic_p", "ic_p_holm",
    "status", "replicated", "consistent_on_majors",
    "code_commit", "manifest_sha256",
)
OOS_COLUMNS: Tuple[str, ...] = KEY_COLUMNS + OOS_VALUE_COLUMNS
ROBUSTNESS_COLUMNS: Tuple[str, ...] = (
    "bar", "horizon", "factor", "row", "parameter", "value_bps", "se_bps", "t_hh", "n_active",
)

_INT_COLUMNS = frozenset({"horizon", "direction", "n_eval", "n_active"})
_BOOL_COLUMNS = frozenset({"direction_flagged", "flagged", "replicated", "consistent_on_majors"})
_STR_COLUMNS = frozenset({
    "instrument", "bar", "factor", "se_method", "status", "code_commit", "manifest_sha256",
    "flag_reason", "row", "parameter",
})

# §10 fill rules. A column outside a row's filled set is blank, whatever was computed.
FILL_EVERY_ROW = frozenset({
    "direction", "direction_flagged", "gross_bps", "net_bps", "se_bps", "se_method",
    "status", "code_commit", "manifest_sha256",
})
FILL_UNLESS_POOLED = frozenset({"n_eval", "n_active", "share_top", "share_bottom"})
FILL_TESTED = frozenset({"t_hh", "p_one_sided", "p_holm", "q_bh", "q_by"})
FILL_DESCRIPTIVE = frozenset({"ci95_lo_bps", "ci95_hi_bps"})
FILL_PRIMARY = frozenset({
    "ub95_net_bps", "ub95_family_net_bps", "mde_bps", "mde_realised_bps",
    "p_placebo", "p_placebo_family", "t_sqrt_h", "net_funding_bps",
    "ic", "ic_t", "ic_p", "ic_p_holm", "replicated", "consistent_on_majors",
})

STATUS_CONFIRMED = "confirmed"
STATUS_REJECTED = "rejected"
STATUS_NOT_REJECTED = "not_rejected"
STATUS_UNTESTABLE = "untestable"
STATUS_DESCRIPTIVE = "descriptive"


def filled_columns(role: str, pooled: bool = False) -> FrozenSet[str]:
    """The §10 columns a row of a set with ``role`` carries (instrument POOLED or not)."""
    cols = set(FILL_EVERY_ROW)
    if not pooled:
        cols |= FILL_UNLESS_POOLED
    if role in (ROLE_PRIMARY, ROLE_REPLICATION) or (role == ROLE_POOLED and pooled):
        cols |= FILL_TESTED
    if role == ROLE_DESCRIPTIVE:
        cols |= FILL_DESCRIPTIVE
    if role == ROLE_PRIMARY:
        cols |= FILL_PRIMARY
    return frozenset(cols)


# ---------------------------------------------------------------- windows and parameters

@dataclass(frozen=True)
class Window:
    """A set's warm-up start and evaluation window [start, end), UTC ms (§4.1)."""

    warmup_start_ms: int
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.warmup_start_ms <= self.start_ms < self.end_ms:
            raise ValueError(f"window out of order: {self}")


def window_of(set_name: str, spec: Optional[Dict[str, Any]] = None) -> Window:
    s = (spec or get_spec())["sets"][set_name]
    return Window(int(s["warmup_start_ms"]), int(s["start_ms"]), int(s["end_ms"]))


# Choices Appendix A states that this module (with research.stats) implements
# and cannot vary. If the spec ever says otherwise, refuse rather than compute
# something else (the same stance as research.factors._FIXED).
_SPEC_FIXED: Tuple[Tuple[Tuple[str, ...], Any], ...] = (
    (("statistic",), "per_trade_net_mean"),
    (("se", "method"), "hansen_hodrick"),
    (("se", "lag"), "h_minus_1"),
    (("se", "index"), "bucket_start_grid"),
    (("se", "fallback", "method"), "bartlett"),
    (("se", "fallback", "lag"), "2h"),
    (("test", "sidedness"), "one_sided_greater"),
    (("test", "correction"), "holm"),
    (("untestable", "counts_in_family"), True),
    (("buckets", "top_if"), "gt"),
    (("buckets", "bottom_if"), "lt"),
    (("buckets", "edge_ties"), "middle"),
    (("buckets", "edges"), "expanding_prior_bars_including_warmup"),
    (("buckets", "percentile_method"), "linear"),
    (("direction_rule", "statistic"), "sign_of_IS_mean_s_times_fwd"),
    (("direction_rule", "tie"), 1),
    (("direction_rule", "fallback"), 1),
    (("forward_return", "entry"), "close_t_plus_1"),
    (("forward_return", "exit"), "close_t_plus_1_plus_h"),
    (("families", "F_majors", "pooling"), "equal_weight_mean"),
    (("families", "F_majors", "se", "divide_by_A"), False),
    (("families", "F_majors", "untestable_if_any_instrument_untestable"), True),
    (("robustness", "sets"), ["doge_2023"]),
    (("robustness", "placebo", "method"), "circular_shift_of_signal"),
    (("robustness", "placebo", "family_max_t"), True),
    (("robustness", "v1_statistic", "function"), "_spread_t_stat"),
    (("robustness", "v1_statistic", "edges"), "full_window"),
    (("robustness", "quarter_assign"), "bar_t_start"),
    (("robustness", "leave_out_volatile_days", "drop_if"), "holding_interval_overlaps"),
    (("robustness", "turnover_aware", "hold_bars"), "h"),
    (("robustness", "ic", "method"), "spearman_full_window"),
    (("robustness", "ic", "ties"), "average"),
    (("robustness", "ic", "statistic"), "mean_of_standardised_rank_products"),
    (("robustness", "ic", "se"), "hansen_hodrick"),
    (("robustness", "ic", "lag"), "h_minus_1"),
    (("robustness", "ic", "family_correction"), "holm"),
    (("robustness", "ic", "sidedness"), "two_sided"),
    (("robustness", "monotonicity", "edges"), "expanding"),
    (("robustness", "monotonicity", "merge_coincident_edges"), True),
    (("power", "scaling"), "sqrt_N_IS_over_N_primary"),
)


def _dig(d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    for k in path:
        d = d[k]
    return d


# Spec blocks whose every key is pinned in _SPEC_FIXED: an extra key would be
# a rule this pipeline silently ignores (e.g. the daily IC's ``per`` and
# ``min_pairs`` returning to R9).
_SPEC_EXACT_KEYS: Tuple[Tuple[Tuple[str, ...], FrozenSet[str]], ...] = (
    (("robustness", "ic"), frozenset(p[-1] for p, _ in _SPEC_FIXED if p[:-1] == ("robustness", "ic"))),
)


def check_spec(spec: Optional[Dict[str, Any]] = None) -> None:
    """Raise if Appendix A asks for a variant this pipeline does not implement."""
    sp = spec or get_spec()
    for path, want in _SPEC_FIXED:
        got = _dig(sp, path)
        if got != want:
            raise ValueError(f"spec sets {'.'.join(path)} = {got!r}; this pipeline implements only {want!r}")
    for path, keys in _SPEC_EXACT_KEYS:
        got_keys = set(_dig(sp, path))
        if got_keys != keys:
            raise ValueError(
                f"spec block {'.'.join(path)} has keys {sorted(got_keys)}; this pipeline "
                f"implements exactly {sorted(keys)}"
            )
    for name, fam in sp["families"].items():
        role = set_role(fam["set"], sp)
        if role not in (ROLE_PRIMARY, ROLE_REPLICATION, ROLE_POOLED):
            raise ValueError(f"family {name} is on set {fam['set']} with role {role}")


@dataclass(frozen=True)
class StudyParams:
    """
    Every tunable the pipeline uses, defaulting to Appendix A (from_spec).

    Tests shrink windows and factor look-backs with dataclasses.replace; the
    real runs always use from_spec(), and tests/test_spec_constants.py pins
    that from_spec() equals the spec.
    """

    bars: Tuple[str, ...]
    horizons: Tuple[int, ...]
    factor_names: Tuple[str, ...]
    factor_params: Dict[str, Any]
    cost_bps: float
    edges_pct: Tuple[float, float]
    min_prior: int
    quantiles: int
    min_active_is: int
    min_active_trades: int
    untestable_p: float
    alpha: float
    placebo_k: int
    placebo_min_shift_days: int
    quarter_bounds_ms: Tuple[int, ...]
    volatile_days_k: Tuple[int, ...]
    cost_sensitivity_bps: Tuple[float, ...]
    winsorize_pct: Tuple[float, float]
    monotonicity_buckets: int
    # R8's expected settlement grid (Appendix A's expected_spacing_hours). A
    # constant, like the §4.2 staleness limit in factor_params: a spacing
    # measured from the series would let later settlements decide which
    # earlier ones count as missing.
    funding_spacing_ms: int

    @classmethod
    def from_spec(cls, spec: Optional[Dict[str, Any]] = None) -> "StudyParams":
        sp = spec or get_spec()
        check_spec(sp)
        rb = sp["robustness"]
        return cls(
            bars=tuple(sp["bars"]),
            horizons=tuple(int(h) for h in sp["horizons"]),
            factor_names=tuple(sp["factors"]),
            factor_params=copy.deepcopy(sp["factors"]),
            cost_bps=float(sp["cost_round_trip_bps"]),
            edges_pct=tuple(float(x) for x in sp["buckets"]["edges_pct"]),
            min_prior=int(sp["buckets"]["min_prior_values"]),
            quantiles=int(sp["buckets"]["quantiles"]),
            min_active_is=int(sp["direction_rule"]["min_active_IS"]),
            min_active_trades=int(sp["untestable"]["min_active_trades"]),
            untestable_p=float(sp["untestable"]["p_value"]),
            alpha=float(sp["test"]["alpha"]),
            placebo_k=int(rb["placebo"]["k"]),
            placebo_min_shift_days=int(rb["placebo"]["min_shift_days"]),
            quarter_bounds_ms=tuple(int(x) for x in rb["quarter_bounds_ms"]),
            volatile_days_k=tuple(int(k) for k in rb["leave_out_volatile_days"]["k"]),
            cost_sensitivity_bps=tuple(float(c) for c in rb["cost_sensitivity_bps"]),
            winsorize_pct=tuple(float(x) for x in rb["winsorize_pct"]),
            monotonicity_buckets=int(rb["monotonicity"]["buckets"]),
            funding_spacing_ms=int(sp["data"]["funding"]["expected_spacing_hours"]) * HOUR_MS,
        )

    def cells(self) -> List[CellKey]:
        """The cells in the study's order: bar, horizon, factor."""
        return [(b, h, f) for b in self.bars for h in self.horizons for f in self.factor_names]


class PipelineError(RuntimeError):
    """A run that must not proceed (missing prerequisite, inconsistent inputs)."""


# ---------------------------------------------------------------- inputs

@dataclass(frozen=True)
class InputFile:
    """One file a set may read; [lo_ms, hi_ms) restricts which of its rows it uses."""

    kind: str                 # "perp", "spot" or "funding"
    instrument: str           # the perpetual this series belongs to
    path: Path                # absolute
    required: bool
    lo_ms: Optional[int] = None
    hi_ms: Optional[int] = None


def input_files(set_name: str, root: Path = REPO_ROOT, *, spec: Optional[Dict[str, Any]] = None,
                is_end_ms: Optional[int] = None) -> List[InputFile]:
    """
    Every file load_set_data may read for ``set_name``, whether or not it exists.

    IS: the untouched in-sample candles (Appendix A's insample_dir) plus the
    supplementary spot leg and funding in data/history/is_supp/. A test set:
    data/history/oos/<set>/<instId>/ as fetch_v2 writes it. doge_fwd: the
    in-sample files before the in-sample end, its own files from then on
    (§4.1: its warm-up is not re-downloaded). Only the perp candles are
    required; a missing spot leg or funding series makes its factor's cells
    untestable instead (§4.2, §6.8).
    """
    sp = spec or get_spec()
    role = set_role(set_name, sp)
    insts = list(sp["sets"][set_name]["inst"])
    legs = sp["spot_legs"]
    root = Path(root)
    split = int(sp["sets"]["IS"]["end_ms"] if is_end_ms is None else is_end_ms)

    def is_segment(inst: str, lo: Optional[int], hi: Optional[int]) -> List[InputFile]:
        perp_dir = root / sp["insample_dir"]
        if perp_dir.name != inst:
            raise PipelineError(f"Appendix A's insample_dir {sp['insample_dir']} does not hold {inst}")
        return [
            InputFile("perp", inst, perp_dir / CANDLE_FILE, True, lo, hi),
            InputFile("spot", inst, root / IS_SUPP_DIR / legs[inst] / CANDLE_FILE, False, lo, hi),
            InputFile("funding", inst, root / IS_SUPP_DIR / inst / FUNDING_FILE, False, lo, hi),
        ]

    def oos_segment(inst: str, lo: Optional[int], hi: Optional[int]) -> List[InputFile]:
        base = root / OOS_DATA_DIR / set_name
        return [
            InputFile("perp", inst, base / inst / CANDLE_FILE, True, lo, hi),
            InputFile("spot", inst, base / legs[inst] / CANDLE_FILE, False, lo, hi),
            InputFile("funding", inst, base / inst / FUNDING_FILE, False, lo, hi),
        ]

    out: List[InputFile] = []
    for inst in insts:
        if role == ROLE_IS:
            out += is_segment(inst, None, None)
        elif role == ROLE_DESCRIPTIVE:
            out += is_segment(inst, None, split) + oos_segment(inst, split, None)
        else:
            out += oos_segment(inst, None, None)
    return out


_CANDLE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def _read_candles(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    missing = [c for c in _CANDLE_COLUMNS if c not in df.columns]
    if missing:
        raise PipelineError(f"{path} lacks column(s) {missing}")
    cols = list(_CANDLE_COLUMNS) + (["volume_ccy"] if "volume_ccy" in df.columns else [])
    return df.select([pl.col("timestamp").cast(pl.Int64)] + [pl.col(c).cast(pl.Float64) for c in cols[1:]])


def _read_funding(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    missing = [c for c in ("timestamp", "rate") if c not in df.columns]
    if missing:
        raise PipelineError(f"{path} lacks column(s) {missing}")
    return df.select(pl.col("timestamp").cast(pl.Int64), pl.col("rate").cast(pl.Float64))


def _between(df: pl.DataFrame, lo: Optional[int], hi: Optional[int]) -> pl.DataFrame:
    if lo is not None:
        df = df.filter(pl.col("timestamp") >= lo)
    if hi is not None:
        df = df.filter(pl.col("timestamp") < hi)
    return df


def _concat(parts: List[pl.DataFrame]) -> Optional[pl.DataFrame]:
    if not parts:
        return None
    common = [c for c in parts[0].columns if all(c in p.columns for p in parts)]
    return pl.concat([p.select(common) for p in parts], how="vertical").sort("timestamp")


def load_set_data(set_name: str, root: Path = REPO_ROOT, *, spec: Optional[Dict[str, Any]] = None,
                  is_end_ms: Optional[int] = None) -> Dict[str, Any]:
    """
    A set's raw inputs: {"set", "instruments", "perp", "spot", "funding", "files"}.

    perp / spot / funding map each perpetual to its 1-minute candles, its
    spot leg's 1-minute candles and its funding settlements (timestamp,
    rate); spot and funding are None where the series was not downloaded.
    "files" lists every file read, for the record. Nothing is filtered to the
    window yet (build_bars does that), and nothing is filled: gaps stay gaps.
    """
    sp = spec or get_spec()
    files = input_files(set_name, root, spec=sp, is_end_ms=is_end_ms)
    insts = tuple(sp["sets"][set_name]["inst"])
    parts: Dict[Tuple[str, str], List[pl.DataFrame]] = {}
    read: List[Path] = []
    for f in files:
        if not f.path.exists():
            if f.required:
                raise PipelineError(f"required input {f.path} does not exist")
            continue
        df = _read_funding(f.path) if f.kind == "funding" else _read_candles(f.path)
        parts.setdefault((f.kind, f.instrument), []).append(_between(df, f.lo_ms, f.hi_ms))
        read.append(f.path)
    return {
        "set": set_name,
        "instruments": insts,
        "perp": {i: _concat(parts.get(("perp", i), [])) for i in insts},
        "spot": {i: _concat(parts.get(("spot", i), [])) for i in insts},
        "funding": {i: _concat(parts.get(("funding", i), [])) for i in insts},
        "files": read,
    }


def _rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def manifests_required(set_name: str, spec: Optional[Dict[str, Any]] = None) -> List[Path]:
    """
    The committed manifests a set's inputs must match (repo-relative).

    A test set always needs its own (§10 step 4: committed before analysis).
    The IS run and doge_fwd (whose warm-up is in-sample data, §4.1) always
    need the in-sample supplement's: §10 step 2 commits it before the IS
    rerun, so its absence means the supplement was never fetched, and
    running anyway would freeze the basis and funding cells as flagged
    "missing_input" instead of stopping.
    """
    sp = spec or get_spec()
    role = set_role(set_name, sp)
    out: List[Path] = []
    if role in (ROLE_IS, ROLE_DESCRIPTIVE):
        out.append(IS_SUPP_MANIFEST)
    if role != ROLE_IS:
        out.append(oos_manifest_path(set_name))
    return out


def verify_inputs(set_name: str, root: Path = REPO_ROOT, *, spec: Optional[Dict[str, Any]] = None,
                  is_end_ms: Optional[int] = None) -> List[str]:
    """
    Every reason the set's inputs cannot be trusted; empty when they can.

    * every series a required manifest lists exists with the committed SHA-256;
    * the in-sample candles still have Appendix A's SHA-256 (§4.2: never modified);
    * every file load_set_data would read is covered by one of those, so no
      unlisted file can slip into the analysis.
    """
    sp = spec or get_spec()
    root = Path(root)
    problems: List[str] = []
    files = input_files(set_name, root, spec=sp, is_end_ms=is_end_ms)
    for f in files:
        if f.required and not f.path.exists():
            problems.append(f"required input {_rel(f.path, root)} does not exist")
    present = [f for f in files if f.path.exists()]

    # Appendix A's hashes are checked for the in-sample files whatever any
    # manifest says about them: they are the pre-registered values.
    appendix = {(root / sp["insample_dir"] / name).resolve(): digest
                for name, digest in sp["insample_sha256"].items()}
    listed: Dict[Path, str] = {}
    checked: Dict[Path, str] = {}
    for m in manifests_required(set_name, sp):
        mp = root / m
        if not mp.exists():
            problems.append(f"manifest {m.as_posix()} does not exist")
            continue
        try:
            series = json.loads(mp.read_text(encoding="utf-8"))["series"]
            entries = [(Path(e["path"]), str(e["sha256"])) for e in series]
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"manifest {m.as_posix()} cannot be read: {exc}")
            continue
        for p, digest in entries:
            full = p if p.is_absolute() else root / p
            if not full.exists():
                problems.append(f"{m.as_posix()} lists {p.as_posix()}, which does not exist")
                continue
            actual = sha256_file(full)
            checked[full.resolve()] = actual
            if actual != digest:
                problems.append(f"{p.as_posix()}: SHA-256 {actual} differs from {digest} in {m.as_posix()}")
            listed[full.resolve()] = digest

    for f in present:
        key = f.path.resolve()
        if key in appendix:
            actual = checked.get(key) or sha256_file(f.path)
            if actual != appendix[key]:
                problems.append(f"{_rel(f.path, root)}: SHA-256 {actual} differs from Appendix A's {appendix[key]}")
        elif key not in listed:
            problems.append(f"{_rel(f.path, root)} would be read but no manifest lists it")
    return problems


# ---------------------------------------------------------------- bars and factors

def build_bars(data: Dict[str, Any], bar: str, window: Window) -> Dict[str, Dict[str, Optional[pl.DataFrame]]]:
    """
    Per instrument: complete ``bar`` bars, the spot leg's bars and the 4 x bar HTF bars.

    Rows are restricted to [warmup_start, end) before resampling. The
    window's ends must fall on HTF bucket boundaries (all of Appendix A's
    do), so the restriction never cuts a bucket: a cut bucket would merely
    be dropped as incomplete, but it would mean the window was mistyped.
    The spot leg is resampled on its own; compute_factors puts it on the
    perp grid with a NaN close where a spot bucket was dropped (§5.1).
    """
    htf = htf_bar(bar)
    step = bar_ms(htf)
    for name, v in (("warmup_start", window.warmup_start_ms), ("start", window.start_ms), ("end", window.end_ms)):
        if v % step:
            raise ValueError(f"window {name} {v} is not on a {htf} boundary")

    def cut(df: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
        return None if df is None else _between(df, window.warmup_start_ms, window.end_ms)

    out: Dict[str, Dict[str, Optional[pl.DataFrame]]] = {}
    for inst in data["instruments"]:
        perp = cut(data["perp"][inst])
        if perp is None:
            raise PipelineError(f"no perp candles for {inst}")
        spot = cut(data["spot"].get(inst))
        out[inst] = {
            "bars": resample_complete(perp, bar),
            "htf": resample_complete(perp, htf),
            "spot": None if spot is None else resample_complete(spot, bar),
            "minutes": perp.select("timestamp", "close").sort("timestamp"),
        }
    return out


@dataclass
class InstrumentFrame:
    """One instrument's bars at one size, its factors, and lazily cached derived series."""

    instrument: str
    bar: str
    ts: np.ndarray
    close: np.ndarray
    factors: Dict[str, np.ndarray]
    missing_inputs: FrozenSet[str]
    minute_ts: np.ndarray
    minute_close: np.ndarray
    funding_ts: Optional[np.ndarray]
    funding_rate: Optional[np.ndarray]
    funding_spacing_ms: Optional[int]
    min_prior: int
    edges_pct: Tuple[float, float]
    _fwd: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    _signal: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def bar_ms(self) -> int:
        return bar_ms(self.bar)

    def fwd(self, h: int) -> np.ndarray:
        """fwd_t = C_{t+1+h}/C_{t+1} - 1, NaN unless bars t..t+1+h are consecutive (§6.3)."""
        if h not in self._fwd:
            self._fwd[h] = S.forward_returns(self.close, self.ts, self.bar_ms, h)
        return self._fwd[h]

    def signal(self, factor: str) -> np.ndarray:
        """
        s_t from expanding-window P20/P80 over all earlier bars, warm-up included (§6.4).

        The edges do not depend on h, and they are the O(n^2) step, so they
        are computed once per factor and reused by the three horizons.
        """
        if factor not in self._signal:
            lo, hi = S.expanding_edges(self.factors[factor], self.edges_pct[0], self.edges_pct[1], self.min_prior)
            self._signal[factor] = S.assign_buckets(self.factors[factor], lo, hi)
        return self._signal[factor]


def prepare_frames(data: Dict[str, Any], bar: str, window: Window, params: StudyParams) -> Dict[str, InstrumentFrame]:
    """
    The ten factors on each instrument's ``bar`` bars over [warmup_start, end).

    A factor whose input series is unavailable (no spot leg, no funding) is
    all-NaN from compute_factors and is also named in missing_inputs, so its
    cells are untestable for that reason (§6.8) and not only for |A| < 300.

    Funding settlement spacings other than 8 h are reported (logged, §4.2);
    they change nothing else. The staleness limit (factor_params) and R8's
    settlement grid (funding_spacing_ms) are both fixed by Appendix A.
    """
    built = build_bars(data, bar, window)
    frames: Dict[str, InstrumentFrame] = {}
    for inst in data["instruments"]:
        b = built[inst]
        fund = data["funding"].get(inst)
        if fund is not None:
            fund = _between(fund, window.warmup_start_ms, window.end_ms).sort("timestamp")
            if fund.height == 0:
                fund = None
        spot = b["spot"] if b["spot"] is not None and b["spot"].height > 0 else None
        if fund is not None:
            irregular_settlements(fund["timestamp"].to_numpy(),
                                  label=f"{data.get('set', '?')} {inst} funding")
        fac = compute_factors(
            b["bars"], bar, spot, fund, b["htf"], params=params.factor_params,
        )
        missing = set()
        if spot is None:
            missing.add("basis_z_score")
        if fund is None:
            missing.add("funding_cost_pressure")
        frames[inst] = InstrumentFrame(
            instrument=inst,
            bar=bar,
            ts=b["bars"]["timestamp"].to_numpy().astype(np.int64),
            close=b["bars"]["close"].to_numpy().astype(np.float64),
            factors={name: fac[name].to_numpy().astype(np.float64) for name in params.factor_names},
            missing_inputs=frozenset(missing),
            minute_ts=b["minutes"]["timestamp"].to_numpy().astype(np.int64),
            minute_close=b["minutes"]["close"].to_numpy().astype(np.float64),
            funding_ts=None if fund is None else fund["timestamp"].to_numpy().astype(np.int64),
            funding_rate=None if fund is None else fund["rate"].to_numpy().astype(np.float64),
            funding_spacing_ms=None if fund is None else int(params.funding_spacing_ms),
            min_prior=params.min_prior,
            edges_pct=params.edges_pct,
        )
    return frames


# ---------------------------------------------------------------- one cell

def _raw_placebo_offsets(window: Window, params: StudyParams) -> Dict[str, np.ndarray]:
    """δ_j for j = 0..K-1 on each bar grid, before duplicates are dropped (§8 R1)."""
    k = int(params.placebo_k)
    if k < 2:
        raise ValueError("need placebo_k >= 2")
    j = np.arange(k, dtype=np.int64)
    raw: Dict[str, np.ndarray] = {}
    for b in params.bars:
        B = bar_ms(b)
        n_grid = (window.end_ms - window.start_ms) // B
        d30 = params.placebo_min_shift_days * DAY_MS // B
        span = n_grid - 2 * d30
        if span <= 0:
            raise ValueError(
                f"the {b} evaluation grid ({n_grid} bars) is shorter than two "
                f"{params.placebo_min_shift_days}-day minimum shifts"
            )
        # Same integer half-up rounding as stats.placebo_offsets.
        raw[b] = d30 + (2 * j * span + (k - 1)) // (2 * (k - 1))
    return raw


def _new_offset(raw: np.ndarray) -> np.ndarray:
    """True at each j whose offset has not appeared at a smaller j (offsets are non-decreasing in j)."""
    return np.r_[True, raw[1:] != raw[:-1]]


def placebo_offset_table(window: Window, params: StudyParams) -> Dict[str, np.ndarray]:
    """
    R1's offsets on each bar grid, duplicates dropped: K' per grid (§8 R1).

    These are the rotations each cell's own placebo p uses, so they equal
    stats.placebo_offsets on that grid (a test checks this). The family
    maximum uses the subset placebo_family_columns picks.
    """
    return {b: r[_new_offset(r)] for b, r in _raw_placebo_offsets(window, params).items()}


def placebo_family_columns(window: Window, params: StudyParams) -> Dict[str, np.ndarray]:
    """
    Which of each grid's placebo t's enter R1's family maximum.

    §8 R1 takes the family maximum "with the same offset index j for every
    cell; an index is kept only if its offset is new on both bar grids".
    Cells on the 15m and 1H grids share j (the same fraction of the window,
    so nearly the same shift in time). For each bar size this returns the
    positions, within placebo_offset_table's offsets for that bar, of the
    kept j's, in order of j; every bar gets the same number of them. For
    Appendix A's windows no offset repeats on either grid and every j is
    kept.
    """
    raw = _raw_placebo_offsets(window, params)
    new = {b: _new_offset(r) for b, r in raw.items()}
    keep = np.logical_and.reduce(list(new.values()))
    return {b: (np.cumsum(nb) - 1)[keep] for b, nb in new.items()}


def _nan_stat() -> Dict[str, Any]:
    return {"mu": math.nan, "se": math.nan, "t": math.nan, "p": math.nan, "n_active": 0, "se_method": None}


def evaluate_cell(frame: InstrumentFrame, factor: str, h: int, window: Window, params: StudyParams,
                  d: Optional[int] = None, *, robustness: bool = False,
                  placebo_offsets: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Everything about one (instrument, bar, h, factor) cell on ``window``.

    Always: n_eval and the bucket shares (§6.4), |A|, the undirected mean of
    s·fwd, and with direction d the per-trade gross and net means, the §6.7
    SE, t, one-sided p, the §9 bounds and whether the cell is testable
    (input available and |A| >= 300, §6.8). With d None, d is set from this
    window by the §6.5 rule, which is how the in-sample run uses it.

    With ``robustness`` (the primary set only, §8): R1's placebo t's on the
    bar's own de-duplicated offsets and the cell's placebo p (the caller
    takes the family maximum across cells), R2, R8, R9 (the full-window
    Spearman IC; the caller applies Holm across the 60 cells) and the rows
    of R3-R7, R8's count, R10 and RL (§7's mean run length of s_t) for the
    robustness file.

    Fractions throughout; the row builders convert to basis points.
    """
    B = frame.bar_ms
    ts, start, end = frame.ts, window.start_ms, window.end_ms
    fwd = frame.fwd(h)
    s = frame.signal(factor)
    ev = S.evaluable(ts, start, end, B, h)
    shares = S.bucket_shares(s, fwd, ts, start, end, B, h)
    active = ev & (s != 0) & np.isfinite(fwd)
    n_active = int(active.sum())
    missing = factor in frame.missing_inputs
    out: Dict[str, Any] = {
        "instrument": frame.instrument, "bar": frame.bar, "horizon": h, "factor": factor,
        "missing_input": missing, "n_active": n_active, **shares,
        "mean_s_fwd": float(np.mean(s[active] * fwd[active])) if n_active else math.nan,
        "ts": ts,
    }
    if d is None:
        d, flagged, reason, _ = S.direction(s, fwd, ev, params.min_active_is, missing_input=missing)
        out.update(flagged=flagged, flag_reason=reason)
    if d not in (1, -1):
        raise ValueError(f"direction must be +1 or -1, got {d!r}")

    net = S.per_trade_net(s, fwd, d, params.cost_bps)
    st = S.cell_statistic(net, ts, start, end, B, h)
    if st["n_active"] != n_active:  # both are |A|; a mismatch is a wiring bug
        raise AssertionError(f"|A| mismatch {st['n_active']} != {n_active} in {out['bar']} {h} {factor}")
    gross = float(np.mean(d * s[active] * fwd[active])) if n_active else math.nan
    out.update(
        direction=d, net=net, gross=gross, mu=st["mu"], se=st["se"], t=st["t"], p=st["p"],
        se_method=st["se_method"], testable=(not missing) and n_active >= params.min_active_trades,
        **S.bounds(st["mu"], st["se"]),
    )
    if robustness:
        out.update(_robustness(frame, factor, h, window, params, d, s, fwd, net, ev, st, placebo_offsets))
    return out


def _robustness(frame: InstrumentFrame, factor: str, h: int, window: Window, params: StudyParams,
                d: int, s: np.ndarray, fwd: np.ndarray, net: np.ndarray, ev: np.ndarray,
                st: Dict[str, Any], offsets: Optional[np.ndarray]) -> Dict[str, Any]:
    """R1, R2, R8, R9 for the main file and the R3-R8, R10 and RL rows for the robustness file (§8, §10)."""
    B = frame.bar_ms
    ts, start, end = frame.ts, window.start_ms, window.end_ms
    f = frame.factors[factor]
    c = params.cost_bps
    if offsets is None:
        offsets = placebo_offset_table(window, replace(params, bars=(frame.bar,)))[frame.bar]

    placebo_t = S.circular_placebo_t(s, fwd, d, c, ts, start, end, B, h, offsets)
    t_sqrt_h = S.v1_t_sqrt_h(f, fwd, h, quantiles=params.quantiles, mask=ev)
    if frame.funding_ts is None:
        # Without a funding series the expected settlement times are unknown,
        # so no trade's funding can be called complete: R8 is undefined.
        r8 = _nan_stat()
    else:
        adj = S.funding_adjusted_net(net, s, d, ts, B, h, frame.funding_ts, frame.funding_rate,
                                     spacing_ms=frame.funding_spacing_ms)
        r8 = S.cell_statistic(adj, ts, start, end, B, h)
    ic = S.full_window_ic(f, fwd, ts, start, end, B, h)

    rows: List[Dict[str, Any]] = []

    def add(row: str, parameter: str, value_bps: float, se_bps: float, t: float, n: int) -> None:
        rows.append({"row": row, "parameter": parameter, "value_bps": value_bps,
                     "se_bps": se_bps, "t_hh": t, "n_active": int(n)})

    def add_stat(row: str, parameter: str, x: Dict[str, Any]) -> None:
        add(row, parameter, x["mu"] * BPS, x["se"] * BPS, x["t"], x["n_active"])

    for q in S.by_quarter(net, ts, start, end, B, h, params.quarter_bounds_ms):
        add_stat("R3", f"Q{q['quarter']}", q)
    ks = list(params.volatile_days_k)
    worst = S.volatile_days(frame.minute_ts, frame.minute_close, start, end, max(ks) if ks else 0)
    for k in ks:
        add_stat("R4", f"k={k}", S.leave_out_volatile_days(net, ts, start, end, B, h, worst[:k]))
    tw = S.turnover_aware(s, d, frame.close, ts, start, end, B, h, cost_bps=c)
    # n_active here counts the signals the staggered portfolio trades: every
    # evaluable bar with s != 0, whether or not its own fwd is defined.
    add("R5", f"c={c:g}", tw["value_bps"], math.nan, math.nan, int((ev & (s != 0)).sum()))
    for r in S.cost_sensitivity(s, fwd, d, ts, start, end, B, h, params.cost_sensitivity_bps):
        add_stat("R6", f"c={r['cost_bps']:g}", r)
    wf = S.winsorized_fwd(fwd, ts, start, end, B, h, params.winsorize_pct)
    lo_p, hi_p = params.winsorize_pct
    add_stat("R7", f"p{lo_p:g}_p{hi_p:g}",
             S.cell_statistic(S.per_trade_net(s, wf, d, c), ts, start, end, B, h))
    # R8's main-file column is its mean; §8 also asks for the number of trades
    # it is defined for, which has no main-file column, so it is reported here.
    add_stat("R8", "funding_included", r8)
    for q in S.quintile_means(f, fwd, ts, start, end, B, h, params.min_prior, params.monotonicity_buckets):
        add_stat("R10", f"Q{q['quintile']}", q)
    # §7 puts the realised mean run length of s_t next to every confirmed
    # cell's decision. Which cells are confirmed is known only after Holm
    # over all 60, and the outputs are write-once, so it is written for
    # every cell now rather than computed from test data in a second pass.
    # value_bps holds bars, n_active the number of runs (§10).
    rl, n_runs = S.mean_run_length(s, ts, start, end, B, h)
    add("RL", "mean_run_length", rl, math.nan, math.nan, n_runs)

    return {
        "placebo_t": placebo_t,
        "p_placebo": S.placebo_p(st["t"], placebo_t),
        "t_sqrt_h": t_sqrt_h,
        "net_funding": r8["mu"],
        "n_funding": r8["n_active"],
        "ic": ic["ic"], "ic_t": ic["t"], "ic_p": ic["p"],
        "robust_rows": rows,
    }


# ---------------------------------------------------------------- the in-sample run

def run_insample(root: Path = REPO_ROOT, *, data: Optional[Dict[str, Any]] = None,
                 window: Optional[Window] = None, primary_window: Optional[Window] = None,
                 params: Optional[StudyParams] = None) -> pl.DataFrame:
    """
    The rows of results/insample_v2.csv (§6.5), one per (bar, horizon, factor).

    direction / flagged / flag_reason / mean_s_fwd_bps come from
    stats.direction on the IS evaluation window. se_bps is the §6.7 SE of the
    IS per-trade net mean in that direction (the SE does not depend on d or
    c, only μ̂ does). mde_bps is §9's (3.144 + 0.842) · SE_IS · sqrt(N_IS /
    N_primary), N from n_evaluable on each window. se_bps, se_method and
    mde_bps are blank where the cell has no active bars.
    """
    params = params or StudyParams.from_spec()
    window = window or window_of("IS")
    primary_window = primary_window or window_of(set_with_role(ROLE_PRIMARY))
    data = data if data is not None else load_set_data("IS", root)
    if len(data["instruments"]) != 1:
        raise PipelineError(f"the in-sample set has one instrument, got {data['instruments']}")
    inst = data["instruments"][0]

    rows: List[Dict[str, Any]] = []
    for bar in params.bars:
        frame = prepare_frames(data, bar, window, params)[inst]
        B = bar_ms(bar)
        for h in params.horizons:
            n_is = S.n_evaluable(window.start_ms, window.end_ms, B, h)
            n_primary = S.n_evaluable(primary_window.start_ms, primary_window.end_ms, B, h)
            for factor in params.factor_names:
                c = evaluate_cell(frame, factor, h, window, params)
                has_active = c["n_active"] > 0
                se = c["se"] if has_active else math.nan
                mde = S.mde(S.scale_se(se, n_is, n_primary)) if has_active else math.nan
                rows.append({
                    "bar": bar, "horizon": h, "factor": factor,
                    "direction": c["direction"], "flagged": c["flagged"],
                    "flag_reason": c["flag_reason"],
                    "n_active": c["n_active"],
                    "mean_s_fwd_bps": c["mean_s_fwd"] * BPS,
                    "se_bps": se * BPS,
                    "se_method": c["se_method"] if has_active else None,
                    "mde_bps": float(mde) * BPS,
                })
        del frame
    return _frame(rows, INSAMPLE_COLUMNS)


# ---------------------------------------------------------------- the out-of-sample runs

def directions_from_insample(parsed: Mapping[CellKey, Mapping[str, Any]]) -> Dict[CellKey, Tuple[int, bool]]:
    """{cell: (direction, flagged)} from freeze.parse_insample_csv output."""
    return {k: (int(v["direction"]), bool(v["flagged"])) for k, v in parsed.items()}


def read_results_csv(path: Path) -> pl.DataFrame:
    """A results file with every column as text, so blanks stay blank (null)."""
    return pl.read_csv(path, infer_schema_length=0)


def read_secondary_statuses(results_dir: Path, params: StudyParams,
                            spec: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[CellKey, str]]:
    """
    Each cell's status on the replication set and the pooled majors set.

    The primary set's labels `replicated` and `consistent_on_majors` (§7)
    come from these already-written files, so the primary set must run last;
    without them this refuses, rather than labelling every cell false.
    """
    sp = spec or get_spec()
    out: Dict[str, Dict[CellKey, str]] = {}
    for role in (ROLE_REPLICATION, ROLE_POOLED):
        name = set_with_role(role, sp)
        inst = POOLED if role == ROLE_POOLED else sp["sets"][name]["inst"][0]
        path = Path(results_dir) / f"{name}.csv"
        if not path.exists():
            raise PipelineError(
                f"{path} does not exist. The primary set's labels 'replicated' and "
                f"'consistent_on_majors' are read from the {name} results, so "
                f"{set_with_role(ROLE_PRIMARY, sp)} must run after {name}."
            )
        df = read_results_csv(path)
        if tuple(df.columns) != OOS_COLUMNS:
            raise PipelineError(f"{path} does not have the §10 columns")
        sel = df.filter(pl.col("instrument") == inst)
        statuses: Dict[CellKey, str] = {}
        for r in sel.iter_rows(named=True):
            key = (r["bar"], int(r["horizon"]), r["factor"])
            if key in statuses:
                raise PipelineError(f"{path}: cell {key} appears twice for {inst}")
            if r["status"] not in (STATUS_REJECTED, STATUS_NOT_REJECTED, STATUS_UNTESTABLE):
                raise PipelineError(f"{path}: cell {key} has status {r['status']!r}")
            statuses[key] = r["status"]
        if set(statuses) != set(params.cells()):
            raise PipelineError(f"{path}: {inst} rows do not cover exactly the study's cells")
        out[role] = statuses
    return out


def _family(cells: List[Dict[str, Any]], params: StudyParams) -> Dict[str, np.ndarray]:
    """
    p used in the family, Holm-adjusted p and BH / BY q-values (§7).

    An untestable cell enters with p = 1 (§6.8); so does a testable cell
    whose t is undefined (SE = 0: every trade returned exactly μ̂), since no
    evidence against H0 is not evidence for rejection. m is always the full
    family.
    """
    p = np.array([
        c["p"] if c["testable"] and c["p"] == c["p"] else params.untestable_p for c in cells
    ], dtype=np.float64)
    return {"p": p, "holm": S.holm(p), "bh": S.bh(p), "by": S.by(p)}


def _bps(x: Any) -> float:
    return float(x) * BPS if x is not None and x == x else math.nan


def _clean(v: Any) -> Any:
    """numpy scalars to Python, NaN to None (blank in the CSV)."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _dtype(col: str) -> Any:
    if col in _INT_COLUMNS:
        return pl.Int64
    if col in _BOOL_COLUMNS:
        return pl.Boolean
    if col in _STR_COLUMNS:
        return pl.Utf8
    return pl.Float64


def _frame(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pl.DataFrame:
    return pl.DataFrame([
        pl.Series(c, [_clean(r.get(c)) for r in rows], dtype=_dtype(c)) for c in columns
    ])


def _main_row(role: str, inst: str, key: CellKey, values: Mapping[str, Any]) -> Dict[str, Any]:
    keep = filled_columns(role, pooled=inst == POOLED)
    row: Dict[str, Any] = {"instrument": inst, "bar": key[0], "horizon": key[1], "factor": key[2]}
    for col in OOS_VALUE_COLUMNS:
        row[col] = values.get(col) if col in keep else None
    return row


def run_oos_set(
    set_name: str,
    directions: Mapping[CellKey, Tuple[int, bool]],
    *,
    root: Path = REPO_ROOT,
    data: Optional[Dict[str, Any]] = None,
    window: Optional[Window] = None,
    params: Optional[StudyParams] = None,
    mde_bps: Optional[Mapping[CellKey, float]] = None,
    results_dir: Optional[Path] = None,
    code_commit: str = "",
    manifest_sha256: str = "",
    is_end_ms: Optional[int] = None,
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    """
    The rows of results/oos/<set>.csv and, on the primary set, of the robustness file.

    ``directions`` maps every cell to its frozen (d, flagged) from
    insample_v2.csv at the tag; ``mde_bps`` (primary only) maps every cell to
    the freeze amendment's MDE (NaN where undefined). On the primary set
    ``results_dir`` (default root/results/oos) must already hold the
    replication and pooled-majors results: their statuses give the labels.

    Families (§7): the primary and replication sets' 60 cells; the pooled
    majors' 60 POOLED cells, each untestable if any instrument's cell is.
    Per-instrument majors rows and doge_fwd rows are descriptive.
    """
    sp = get_spec()
    role = set_role(set_name, sp)
    if role == ROLE_IS:
        raise ValueError("the in-sample set is run by run_insample, not run_oos_set")
    params = params or StudyParams.from_spec()
    window = window or window_of(set_name)
    cells = params.cells()
    missing_dirs = [k for k in cells if k not in directions]
    if missing_dirs or len(directions) != len(cells):
        raise PipelineError(f"directions must cover exactly the {len(cells)} cells; missing {missing_dirs[:3]}")
    for k in cells:
        if directions[k][0] not in (1, -1):
            raise PipelineError(f"direction of {k} is {directions[k][0]!r}, not +1 or -1")

    primary = role == ROLE_PRIMARY
    labels = None
    if primary:
        if mde_bps is None or any(k not in mde_bps for k in cells):
            raise PipelineError("the primary set needs the freeze amendment's MDE for every cell")
        labels = read_secondary_statuses(results_dir or Path(root) / OOS_RESULTS_DIR, params, sp)

    data = data if data is not None else load_set_data(set_name, root, is_end_ms=is_end_ms)
    insts = list(data["instruments"])
    offsets = placebo_offset_table(window, params) if primary else None
    family_cols = placebo_family_columns(window, params) if primary else None

    res: Dict[Tuple[str, CellKey], Dict[str, Any]] = {}
    for bar in params.bars:
        frames = prepare_frames(data, bar, window, params)
        for h in params.horizons:
            for factor in params.factor_names:
                key = (bar, h, factor)
                for inst in insts:
                    res[(inst, key)] = evaluate_cell(
                        frames[inst], factor, h, window, params, d=directions[key][0],
                        robustness=primary, placebo_offsets=offsets[bar] if primary else None,
                    )
        del frames

    pooled: Dict[CellKey, Dict[str, Any]] = {}
    if role == ROLE_POOLED:
        for key in cells:
            members = [res[(i, key)] for i in insts]
            st = S.pooled_statistic([(m["net"], m["ts"]) for m in members],
                                    window.start_ms, window.end_ms, bar_ms(key[0]), key[1])
            gross = [m["gross"] for m in members]
            pooled[key] = {
                "direction": directions[key][0], "mu": st["mu"], "se": st["se"], "t": st["t"],
                "p": st["p"], "se_method": st["se_method"],
                "gross": float(np.mean(gross)) if all(g == g for g in gross) else math.nan,
                "testable": all(m["testable"] for m in members),
            }

    # The family: which rows carry the tested statistics.
    if role in (ROLE_PRIMARY, ROLE_REPLICATION):
        fam_cells = [res[(insts[0], k)] for k in cells]
    elif role == ROLE_POOLED:
        fam_cells = [pooled[k] for k in cells]
    else:
        fam_cells = []
    fam = _family(fam_cells, params) if fam_cells else None

    extra: Dict[CellKey, Dict[str, Any]] = {k: {} for k in cells}
    if fam is not None:
        for i, k in enumerate(cells):
            c = fam_cells[i]
            rejected = c["testable"] and fam["holm"][i] <= params.alpha
            if not c["testable"]:
                status = STATUS_UNTESTABLE
            elif rejected:
                status = STATUS_CONFIRMED if primary else STATUS_REJECTED
            else:
                status = STATUS_NOT_REJECTED
            extra[k].update(p_one_sided=fam["p"][i], p_holm=fam["holm"][i], q_bh=fam["bh"][i],
                            q_by=fam["by"][i], status=status)

    if primary:
        doge = [res[(insts[0], k)] for k in cells]
        t_obs = np.array([c["t"] for c in doge], dtype=np.float64)
        # Per-cell p used each grid's own K' rotations; the family maximum
        # uses only the offset indices j kept on both grids (§8 R1).
        T = np.vstack([c["placebo_t"][family_cols[c["bar"]]] for c in doge])
        p_fam_placebo = S.family_placebo_p(t_obs, T, testable=[c["testable"] for c in doge])
        # R9's own family of 60: an undefined IC (no pairs, or every pair tied
        # in f or fwd) enters with p = 1, as an untestable cell does in §6.8.
        ic_p = np.array([c["ic_p"] if c["ic_p"] == c["ic_p"] else params.untestable_p for c in doge])
        ic_holm = S.holm(ic_p)
        for i, k in enumerate(cells):
            confirmed = extra[k]["status"] == STATUS_CONFIRMED
            extra[k].update(
                p_placebo_family=p_fam_placebo[i], ic_p_holm=ic_holm[i],
                replicated=confirmed and labels[ROLE_REPLICATION][k] == STATUS_REJECTED,
                consistent_on_majors=confirmed and labels[ROLE_POOLED][k] == STATUS_REJECTED,
            )

    rows: List[Dict[str, Any]] = []
    for inst in insts:
        for k in cells:
            c = res[(inst, k)]
            v: Dict[str, Any] = {
                "direction": c["direction"], "direction_flagged": directions[k][1],
                "n_eval": c["n_eval"], "n_active": c["n_active"],
                "share_top": c["share_top"], "share_bottom": c["share_bottom"],
                "gross_bps": _bps(c["gross"]), "net_bps": _bps(c["mu"]), "se_bps": _bps(c["se"]),
                "se_method": c["se_method"], "t_hh": c["t"],
                "ub95_net_bps": _bps(c["ub95"]), "ub95_family_net_bps": _bps(c["ub95_family"]),
                "ci95_lo_bps": _bps(c["ci95_lo"]), "ci95_hi_bps": _bps(c["ci95_hi"]),
                "status": STATUS_DESCRIPTIVE,
                "code_commit": code_commit, "manifest_sha256": manifest_sha256,
            }
            if primary:
                v.update(
                    mde_bps=mde_bps[k], mde_realised_bps=_bps(S.mde(c["se"])),
                    p_placebo=c["p_placebo"], t_sqrt_h=c["t_sqrt_h"],
                    net_funding_bps=_bps(c["net_funding"]),
                    ic=c["ic"], ic_t=c["ic_t"], ic_p=c["ic_p"],
                )
            if role in (ROLE_PRIMARY, ROLE_REPLICATION):
                v.update(extra[k])
            rows.append(_main_row(role, inst, k, v))
    if role == ROLE_POOLED:
        for k in cells:
            c = pooled[k]
            v = {
                "direction": c["direction"], "direction_flagged": directions[k][1],
                "gross_bps": _bps(c["gross"]), "net_bps": _bps(c["mu"]), "se_bps": _bps(c["se"]),
                "se_method": c["se_method"], "t_hh": c["t"],
                "code_commit": code_commit, "manifest_sha256": manifest_sha256,
                **extra[k],
            }
            rows.append(_main_row(role, POOLED, k, v))
    main = _frame(rows, OOS_COLUMNS)

    robust = None
    if primary:
        rrows = []
        for k in cells:
            for r in res[(insts[0], k)]["robust_rows"]:
                rrows.append({"bar": k[0], "horizon": k[1], "factor": k[2], **r})
        robust = _frame(rrows, ROBUSTNESS_COLUMNS)
    return main, robust


__all__ = [
    "POOLED", "KEY_COLUMNS", "OOS_VALUE_COLUMNS", "OOS_COLUMNS", "ROBUSTNESS_COLUMNS",
    "INSAMPLE_COLUMNS", "FILL_EVERY_ROW", "FILL_UNLESS_POOLED", "FILL_TESTED",
    "FILL_DESCRIPTIVE", "FILL_PRIMARY", "filled_columns",
    "Window", "window_of", "StudyParams", "PipelineError", "check_spec",
    "set_role", "set_with_role", "InputFile", "input_files", "load_set_data",
    "manifests_required", "verify_inputs", "build_bars", "InstrumentFrame",
    "prepare_frames", "placebo_offset_table", "placebo_family_columns", "evaluate_cell", "run_insample",
    "directions_from_insample", "read_results_csv", "read_secondary_statuses", "run_oos_set",
    "oos_manifest_path", "oos_results_path", "robustness_results_path",
]
