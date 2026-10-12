"""
Load the machine-readable specification from the pre-registration.

Appendix A of docs/preregistration.md is a YAML block. It is the single
source of the study's constants: sets and their boundaries, factor
parameters, bucket rules, the standard-error method, families and
thresholds. Code reads them from here, and tests/test_prereg_spec.py checks
that nothing hard-coded elsewhere drifts from them.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_PATH = REPO_ROOT / "docs" / "preregistration.md"
FREEZE_AMENDMENT_PATH = REPO_ROOT / "docs" / "preregistration-freeze.md"

# The in-sample download starts at 2025-09-23 20:02 UTC; its first full
# 1H bucket and the end of doge_2024 are both 20:00 that day.
IS_DATA_START_MS = 1758657600000

_APPENDIX = re.compile(
    r"^## Appendix A: machine-readable specification\s*\n+```yaml\n(.*?)^```",
    re.S | re.M,
)

BAR_MS = {"15m": 15 * 60_000, "1H": 60 * 60_000}
MINUTE_MS = 60_000


def parse_spec_text(text: str, where: str = "the pre-registration") -> Dict[str, Any]:
    """
    Appendix A of a pre-registration's text, parsed into a dict.

    Separate from :func:`load_spec` so that the text can come from anywhere,
    in particular from ``git show <tag>:docs/preregistration.md``:
    scripts/run_oos.py compares the block at the freeze tag with the one the
    run will use. Raises ValueError (a YAML error included) naming ``where``.
    """
    match = _APPENDIX.search(text)
    if match is None:
        raise ValueError(f"No Appendix A YAML block found in {where}")
    try:
        parsed = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(f"Appendix A in {where} is not valid YAML: {exc}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"Appendix A in {where} is not a mapping")
    return parsed


# The name the review notes use; one function, two names.
parse_appendix = parse_spec_text


def load_spec(path: Path = PREREG_PATH) -> Dict[str, Any]:
    """Parse Appendix A of the pre-registration into a dict."""
    return parse_spec_text(Path(path).read_text(encoding="utf-8"), str(path))


@lru_cache(maxsize=1)
def get_spec() -> Dict[str, Any]:
    """The spec of the checked-in pre-registration, parsed once."""
    return load_spec()


def protected_windows() -> List[Tuple[int, int]]:
    """
    Half-open [start, end) ms ranges that hold test data.

    Each test set's warm-up and evaluation window, minus the in-sample year
    (whose data is already on disk and may be used freely). Downloads into
    these ranges are refused until the freeze is verified
    (research.freeze.verify_freeze, §10), and into one whose end is still in
    the future even then (§10: doge_fwd's steps 4-6 follow its close).
    """
    sets = get_spec()["sets"]
    is_lo, is_hi = IS_DATA_START_MS, sets["IS"]["end_ms"]
    out: List[Tuple[int, int]] = []
    for name, s in sets.items():
        if name == "IS":
            continue
        lo, hi = s["warmup_start_ms"], s["end_ms"]
        # Subtract the in-sample range: doge_fwd's warm-up is in-sample data.
        if lo < is_lo:
            out.append((lo, min(hi, is_lo)))
        if hi > is_hi:
            out.append((max(lo, is_hi), hi))
    return sorted(set(out))


def protected_overlapping(start_ms: int, end_ms: int) -> List[Tuple[int, int]]:
    """The protected windows that [start_ms, end_ms) intersects, in order."""
    return [(lo, hi) for lo, hi in protected_windows() if start_ms < hi and lo < end_ms]


def overlaps_protected(start_ms: int, end_ms: int) -> bool:
    """True if [start_ms, end_ms) intersects any protected window."""
    return bool(protected_overlapping(start_ms, end_ms))
