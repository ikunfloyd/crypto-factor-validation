#!/usr/bin/env python3
"""
Reproducible micro-benchmarks for the hot paths.

Numbers are machine-dependent — run it yourself rather than trusting a README.
Numba kernels are warmed up before timing so JIT compilation isn't measured.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --candles 5000 --iters 2000
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger
logger.remove()

from shared_memory.ring_buffer import ArrowRingBuffer, RingBufferConfig, RING_BUFFER_CONFIGS
from quant_core.data_processor import DataProcessor
from quant_core.factor_engine import FactorEngine

CANDLE_DTYPE = np.dtype([
    ("timestamp", "f8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
    ("close", "f8"), ("volume", "f8"), ("volume_ccy", "f8"),
])


def _synth_candles(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = np.zeros(n, dtype=CANDLE_DTYPE)
    t0, p = 1_700_000_000.0, 0.15
    for i in range(n):
        c = p * (1 + rng.normal(0, 0.002))
        arr[i] = (t0 + i * 60, p, max(p, c) * 1.001, min(p, c) * 0.999, c,
                  abs(rng.normal(1e4, 2e3)), 1.0)
        p = c
    return arr


def _row(label: str, value: str) -> None:
    print(f"  {label:<34} {value}")


def bench_ring_buffer(iters: int) -> None:
    print("\n[ring buffer] tick buffer, single-item writes")
    cfg: RingBufferConfig = RING_BUFFER_CONFIGS["fast_tick"]
    buf = ArrowRingBuffer(cfg, cleanup_on_exit=True)
    try:
        one = np.zeros(1, dtype=cfg.dtype)
        buf.write(one)  # warm
        t0 = time.perf_counter()
        for _ in range(iters):
            buf.write(one)
        dt = time.perf_counter() - t0
        _row("write throughput", f"{iters / dt:,.0f} items/s  ({dt / iters * 1e6:.2f} µs/write)")

        buf.peek_latest(1)
        lat = []
        for _ in range(min(iters, 5000)):
            s = time.perf_counter()
            buf.peek_latest(1)
            lat.append((time.perf_counter() - s) * 1e6)
        _row("peek_latest(1) latency", f"median {statistics.median(lat):.2f} µs")
    finally:
        buf.cleanup()


def bench_data_processor(candles_raw: np.ndarray, reps: int) -> None:
    print("\n[data processor] numpy -> Polars + derived features")
    dp = DataProcessor()
    dp.process_candle_data(candles_raw)  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        dp.process_candle_data(candles_raw)
    dt = time.perf_counter() - t0
    n = len(candles_raw)
    _row("throughput", f"{n * reps / dt:,.0f} candles/s  ({dt / reps * 1e3:.2f} ms/{n}-bar frame)")


def bench_factor_engine(candles_raw: np.ndarray, reps: int) -> None:
    print("\n[factor engine] all candle-derived factors (Numba)")
    dp, fe = DataProcessor(), FactorEngine()
    candles = dp.process_candle_data(candles_raw)
    fe.calculate_all_factors({"candles": candles})  # warm (JIT compile)
    t0 = time.perf_counter()
    for _ in range(reps):
        fe.calculate_all_factors({"candles": candles})
    dt = time.perf_counter() - t0
    n = len(candles_raw)
    _row("throughput", f"{n * reps / dt:,.0f} candles/s  ({dt / reps * 1e3:.2f} ms/{n}-bar frame)")


def bench_inference(candles_raw: np.ndarray, iters: int) -> None:
    print("\n[MoE inference] prepare_features + predict_with_moe")
    from quant_core.model_inference import ModelInference
    model_dir = REPO_ROOT / "data" / "models"
    if not (model_dir / "hunter" / "hunter_model.txt").exists():
        _row("skipped", "no models — run scripts/demo.py or train_models first")
        return
    dp, fe = DataProcessor(), FactorEngine()
    candles = dp.process_candle_data(candles_raw)
    factors = fe.calculate_all_factors({"candles": candles})
    eng = ModelInference(model_dir=str(model_dir))
    feats = eng.prepare_features(factors, candles)
    eng.predict_with_moe(feats, "trending")  # warm
    lat = []
    for _ in range(iters):
        s = time.perf_counter()
        eng.predict_with_moe(feats, "trending")
        lat.append((time.perf_counter() - s) * 1e6)
    _row("predict latency", f"median {statistics.median(lat):.1f} µs  ({1e6 / statistics.median(lat):,.0f}/s)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Hot-path micro-benchmarks.")
    ap.add_argument("--candles", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    print("=" * 60)
    print("  DOGE-USDT trading system — micro-benchmarks")
    print(f"  candles/frame={args.candles}  (numbers are machine-specific)")
    print("=" * 60)

    raw = _synth_candles(args.candles)
    bench_ring_buffer(args.iters)
    bench_data_processor(raw, args.reps)
    bench_factor_engine(raw, args.reps)
    bench_inference(raw, min(args.iters, 20_000))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
