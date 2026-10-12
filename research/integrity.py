"""
Whether the code a run script is executing is this checkout's.

The guards in scripts/run_oos.py, run_insample_v2.py and fetch_v2.py ask
git whether the checkout holds the frozen code. That answer is about the files
on disk, not about what Python imported. A PYTHONPATH entry that points at
another checkout (the project's .env sets PYTHONPATH) makes ``import
research`` load that checkout's package while every git guard still passes
on this one. The scripts therefore put their own repository first on
sys.path before importing anything from it, and refuse to run if a module
of the repository's own packages was nonetheless loaded from elsewhere
(foreign_code_problem words that refusal once for all of them).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional

# The top-level packages of this repository that a run imports.
OWN_PACKAGES = ("research", "quant_core", "config")


def _under(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def foreign_modules(repo_root: Path, packages: Iterable[str] = OWN_PACKAGES) -> List[str]:
    """
    ``"name (location)"`` for each loaded module of ``packages`` that did not
    come from ``repo_root``; empty when all of them did.

    A module without ``__file__`` is a namespace package, which exists only
    when its ``__init__.py`` is missing; it counts as foreign unless every
    directory it spans is inside ``repo_root``.
    """
    root = Path(repo_root).resolve()
    own = set(packages)
    out: List[str] = []
    for name, mod in sorted(sys.modules.items()):
        if mod is None or name.split(".")[0] not in own:
            continue
        f = getattr(mod, "__file__", None)
        if f is not None:
            if not _under(f, root):
                out.append(f"{name} ({f})")
            continue
        spans = [str(p) for p in getattr(mod, "__path__", [])]
        if not spans or not all(_under(p, root) for p in spans):
            out.append(f"{name} (namespace package over {spans or 'nothing'})")
    return out


def foreign_code_problem(repo_root: Path) -> Optional[str]:
    """Why a script must not run because some of its own code came from elsewhere; None if none did."""
    foreign = foreign_modules(repo_root)
    if not foreign:
        return None
    return ("modules of this repository were loaded from elsewhere (is PYTHONPATH naming another checkout?), "
            "so the checks would vouch for code that is not running: " + ", ".join(foreign[:5])
            + (f" and {len(foreign) - 5} more" if len(foreign) > 5 else ""))


__all__ = ["OWN_PACKAGES", "foreign_modules", "foreign_code_problem"]
