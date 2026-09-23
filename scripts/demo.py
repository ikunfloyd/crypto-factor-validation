#!/usr/bin/env python3
"""
Offline end-to-end demo — no exchange account, API keys, or M3 hardware required.

It drives the full compute path the live system uses:

    synthetic OHLCV candles
        -> DataProcessor        (feature engineering)
        -> FactorEngine         (quant factors, Numba-accelerated)
        -> ModelInference (MoE) (three LightGBM experts + gating network)
        -> trading signal        (direction / confidence / position size)

The expert models are trained on *synthetic* data (see
quant_core/training/train_models.py). This demo therefore showcases the
factor -> scaling -> MoE -> signal *pipeline*, not a profitable strategy.
If the models are missing they are trained automatically on first run.

Usage:
    python -m scripts.demo               # from the repo root
    python scripts/demo.py --candles 2000 --seed 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/demo.py` as well as `python -m scripts.demo`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

# Quiet library logs by default; --verbose re-enables them in main(). Configured
# before importing quant_core modules so their import-time logs stay silent too.
logger.remove()

from quant_core.data_processor import DataProcessor
from quant_core.factor_engine import FactorEngine
from quant_core.model_inference import ModelInference

MODEL_DIR = REPO_ROOT / "data" / "models"
EXPERTS = ("hunter", "sniper", "defender")

CANDLE_DTYPE = np.dtype([
    ("timestamp", "f8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
    ("close", "f8"), ("volume", "f8"), ("volume_ccy", "f8"),
])


def ensure_models() -> None:
    """Train the three expert models on synthetic data if they are absent."""
    missing = [m for m in EXPERTS if not (MODEL_DIR / m / f"{m}_model.txt").exists()]
    if not missing:
        return
    print(f"[setup] models not found ({', '.join(missing)}) -> training on synthetic data ...")
    from quant_core.training.train_models import ModelTrainer

    ModelTrainer(str(MODEL_DIR)).train_all_models(use_synthetic=True)
    print("[setup] training complete.\n")


def make_synthetic_candles(n: int, seed: int) -> np.ndarray:
    """Generate n 1-minute OHLCV candles as a random walk with a regime shift.

    The first ~60% trends gently upward with low volatility; the tail turns
    choppy with higher volatility, so the MoE output visibly changes.
    """
    rng = np.random.default_rng(seed)
    arr = np.zeros(n, dtype=CANDLE_DTYPE)
    t0, price = 1_700_000_000.0, 0.15
    shift = int(n * 0.6)
    for i in range(n):
        if i < shift:
            drift, vol = 0.00012, 0.0018      # calm uptrend
        else:
            drift, vol = 0.0, 0.0055          # choppy / high-vol
        ret = rng.normal(drift, vol)
        o = price
        c = o * (1 + ret)
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0012)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0012)))
        v = abs(rng.normal(10_000, 2_500))
        arr[i] = (t0 + i * 60, o, h, l, c, v, v * c)
        price = c
    return arr


def classify_regime(candles) -> str:
    """Lightweight regime label for the MoE gating network.

    Maps recent realized volatility / trend into one of the gating regimes
    the expert models were trained against: trending / ranging / high_vol.
    """
    close = candles["close"].to_numpy()
    tail = close[-60:] if len(close) >= 60 else close
    returns = np.diff(tail) / tail[:-1]
    vol = float(np.std(returns))
    trend = float((tail[-1] - tail[0]) / tail[0])
    if vol > 0.004:
        return "high_vol"
    if abs(trend) > 0.01:
        return "trending"
    return "ranging"


def fmt_signal(result: dict) -> str:
    sig = result.get("signal", {})
    preds = result.get("predictions", {})
    weights = result.get("weights", {})
    lines = [
        f"  regime          : {result.get('regime')}",
        f"  gating weights  : " + ", ".join(f"{k}={weights.get(k, 0):.2f}" for k in EXPERTS),
        f"  expert outputs  : " + ", ".join(f"{k}={preds.get(k, 0):+.3f}" for k in EXPERTS),
        f"  fused score     : {result.get('final_score', 0):+.3f}",
        f"  -> direction    : {sig.get('direction')}",
        f"  -> confidence   : {sig.get('confidence', 0):.1%}",
        f"  -> position size: {sig.get('position_size', 0):.1%}",
        f"  -> defender risk: {sig.get('defender_risk', 0):.1%}"
        + ("   [CIRCUIT BREAKER]" if result.get("final_score") == -1.0 else ""),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline MoE trading-pipeline demo.")
    parser.add_argument("--candles", type=int, default=2000, help="number of 1m candles to synthesize")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility")
    parser.add_argument("-v", "--verbose", action="store_true", help="show library debug logs")
    args = parser.parse_args()

    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")

    print("=" * 66)
    print("  DOGE-USDT trading system — offline pipeline demo")
    print("=" * 66)

    ensure_models()

    print(f"[1/4] synthesizing {args.candles} 1m candles (seed={args.seed}) ...")
    raw = make_synthetic_candles(args.candles, args.seed)

    print("[2/4] data processing + factor engine ...")
    candles = DataProcessor().process_candle_data(raw)
    factors = FactorEngine().calculate_all_factors({"candles": candles})
    computed = [c for c in factors.columns if c != "timestamp"] if len(factors) else []
    print(f"      candle features : {len(candles.columns)} columns")
    print(f"      factors computed: {len(computed)} -> {', '.join(computed) or '(none)'}")
    print("      (funding-rate / open-interest / order-flow factors need live feeds and are skipped)")

    print("[3/4] loading MoE experts ...")
    engine = ModelInference(model_dir=str(MODEL_DIR))
    loaded = list(engine.models.keys())
    print(f"      experts loaded  : {loaded or '(none — using rule fallback)'}")

    print("[4/4] inference:\n")
    features = engine.prepare_features(factors, candles)
    if features is None:
        print("  no features available for inference.")
        return 1

    regime = classify_regime(candles)
    result = engine.predict_with_moe(features, regime)
    result.setdefault("regime", regime)
    print(fmt_signal(result))

    print("\n" + "-" * 66)
    print("How the same features route through every regime (gating demo):")
    for rg in ("trending", "ranging", "high_vol", "crash"):
        r = engine.predict_with_moe(features, rg)
        s = r.get("signal", {})
        print(f"  {rg:9s} -> {s.get('direction','?'):7s} "
              f"conf={s.get('confidence',0):5.1%}  pos={s.get('position_size',0):5.1%}")

    # Optional: run the execution layer (signal fusion) on the primary signal.
    # Kept defensive so an integration hiccup never breaks the demo.
    try:
        from execution.signal_fusion import SignalFusion

        fusion_regime = {
            "trending": "trending_up" if result.get("final_score", 0) >= 0.5 else "trending_down",
            "ranging": "ranging", "high_vol": "high_vol", "crash": "high_risk",
        }.get(regime, "ranging")
        risk_level = "extreme" if result["signal"].get("defender_risk", 0) > 0.6 else "normal"
        fused = SignalFusion().fuse(
            quant_signal=result["signal"], regime=fusion_regime,
            current_position=0.0, risk_level=risk_level,
        )
        print("\n" + "-" * 66)
        print("Execution layer — signal fusion (quant 60% / regime 30% / risk 10%):")
        print(f"  mapped regime   : {fusion_regime}   risk_level={risk_level}")
        print(f"  final decision  : {fused.get('direction')}  "
              f"confidence={fused.get('confidence', 0):.1%}  "
              f"target position={fused.get('position_size', fused.get('target_position', 0)):.1%}")
    except Exception as exc:  # pragma: no cover - demo-only best effort
        print(f"\n[execution layer skipped: {type(exc).__name__}: {exc}]")

    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
