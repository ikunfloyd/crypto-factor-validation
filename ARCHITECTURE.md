# Architecture

> Design notes for the DOGE-USDT trading system.
> [English](ARCHITECTURE.md) · [中文（原始详版）](ARCHITECTURE.zh.md)

This document explains *why* the system is shaped the way it is. For a runnable
overview and quick start, see the [README](README.md).

## 1. System shape

The system is a set of cooperating OS processes rather than a single event loop.
Each stage owns one responsibility and communicates through a shared-memory
transport, so a slow or crashed stage can't block the others and each can be
pinned to the CPU cores that suit it.

```
                         ┌───────────────────────────────────────┐
                         │            Orchestrator               │
                         │  spawn · CPU affinity · health · WAL  │
                         └───────────────────────────────────────┘
                              │ supervises every process below
   ┌──────────────┬───────────┴────────────┬──────────────────────┐
   │  IO Gateway  │       Quant Core        │      Execution       │
   │  (E-cores)   │       (P-cores)         │      (P-cores)       │
   ├──────────────┼─────────────────────────┼──────────────────────┤
   │ WS ingest    │ DataProcessor           │ RegimeDetector       │
   │ WAL writer   │ FactorEngine (Numba)    │ SignalFusion         │
   │ data cleaner │ ModelInference (MoE)    │ RiskManager          │
   │              │ GatingNetwork           │ OrderRouter          │
   └──────┬───────┴───────────┬─────────────┴──────────────────────┘
          │ write             │ read_latest / peek_latest
          ▼                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │      Shared Memory — ring buffers: tick (3600) · candle (5000) │
   └───────────────────────────────────────────────────────────────┘
          │
          ▼
   ┌───────────────────────────────────────────────────────────────┐
   │   Persistence — WAL (msgpack+zlib) · checkpoints · models      │
   └───────────────────────────────────────────────────────────────┘
```

## 2. Design principles

1. **Process isolation over threads.** Python's GIL makes true parallelism
   across CPU-bound stages hard in one process. Separate processes let ingest,
   compute, and execution run genuinely concurrently, and let the orchestrator
   restart any one of them independently.
2. **Zero-copy transport.** Market data moves through shared memory, not a
   serializing queue — see §3.
3. **Durability before visibility.** A tick is written to the WAL *before* it is
   published to shared memory, so a crash can never lose an acknowledged event.
4. **Hardware-aware placement.** I/O-bound ingest is pinned to efficiency cores;
   CPU-bound factor/inference work is pinned to performance cores.

## 3. Shared-memory ring buffer

[`shared_memory/ring_buffer.py`](shared_memory/ring_buffer.py) implements a
fixed-capacity ring over `multiprocessing.shared_memory`, typed with a NumPy
structured dtype. Two configs ship in `RING_BUFFER_CONFIGS`:

- `fast_tick` — top-5 order-book levels, capacity 3600
- `candle` — OHLCV + quote volume, capacity 5000

Producers call `write()`; consumers choose their semantics:

- `read_latest(n)` advances the read index (consume-once).
- `peek_latest(n)` does not (used by the dashboard so monitoring never steals
  data from the compute path).

**Why not a message queue?** For a single-host, high-frequency pipeline the cost
that matters is per-message serialization and copy. A shared-memory ring hands
the consumer the same bytes the producer wrote — no pickling, no socket. The
trade-off is that it's single-host only and requires manual lifecycle care
(unlink on shutdown); both are acceptable here.

## 4. Compute path (Quant Core)

```
raw candles (np structured array)
    → DataProcessor.process_candle_data   # → Polars DF + time/price/vol features
    → FactorEngine.calculate_all_factors  # Numba-JIT kernels, per-factor registry
    → ModelInference.prepare_features      # assemble the 17-feature vector
    → ModelInference.predict_with_moe      # 3 experts × gating weights → score
    → trading signal (direction / confidence / position size)
```

- **DataProcessor** ([`quant_core/data_processor.py`](quant_core/data_processor.py))
  converts the NumPy frame to Polars and derives returns, ranges, multi-horizon
  volatility, ATR, and calendar features.
- **FactorEngine** ([`quant_core/factor_engine.py`](quant_core/factor_engine.py))
  holds a `factor_registry` of 9 factors in three groups (momentum/technical,
  flow/positioning, risk). Hot loops are Numba `@jit(nopython=True)` kernels.
  Only the candle-derived factors compute from candles alone; the flow and risk
  factors need live funding-rate / open-interest / trade feeds and are skipped
  when those are absent.
- **ModelInference** ([`quant_core/model_inference.py`](quant_core/model_inference.py))
  is the MoE — see §5.

## 5. Mixture of Experts

Three LightGBM models each specialize in a market regime:

| Expert | Specialty | Target |
|--------|-----------|--------|
| Hunter | trending markets | up-move probability |
| Sniper | ranging markets | precise entry timing |
| Defender | high-volatility / risk | drawdown risk |

A **gating network** maps the current regime to per-expert weights, the weighted
sum becomes the score, and a **circuit breaker** forces liquidation when the
Defender's risk estimate exceeds 0.8. The same feature vector produces different
signals per regime — that routing is the point of the design.

> **Feature-list contract.** Training and inference share one 17-feature list in
> the same order ([`train_models.py`](quant_core/training/train_models.py) and
> [`model_inference.py`](quant_core/model_inference.py)). If they drift, every
> `predict` raises a dimension error and silently returns 0.0 — a bug that once
> made the MoE emit constant signals. Change both together, then retrain.

> **On the models.** They are trained on *synthetic* data as a pipeline
> demonstration, not a profitable strategy. `data/models/` is generated (and
> git-ignored); `scripts/demo.py` trains on first run.

## 6. Durability & recovery

- **WAL** ([`io_gateway/wal_writer.py`](io_gateway/wal_writer.py)) appends every
  record as msgpack + zlib, date-partitioned under `data/wal/`. Because the WAL
  write precedes the shared-memory publish, no acknowledged event is lost on a
  crash.
- **Checkpoints** periodically snapshot process state.
- **Recovery** ([`utils/disaster_recovery.py`](utils/disaster_recovery.py),
  [`io_gateway/wal_reader.py`](io_gateway/wal_reader.py)) restores from the last
  checkpoint and replays the WAL forward.

## 7. Orchestration

[`orchestrator/main.py`](orchestrator/main.py) spawns each stage as a
`multiprocessing.Process` running the stage's async `main()`, pins CPU affinity
via `psutil` (IO Gateway → efficiency cores; Quant Core / Execution →
performance cores), health-checks each process, and restarts any that die.

> Note: the orchestrator launches `quant_core.main`; `quant_core.main_v2` is the
> model-integrated variant used when running that stage standalone.

## 8. Execution

[`execution/`](execution/) turns a score into an order: `SignalFusion` blends the
quant score with regime and risk adjustments (0.6 / 0.3 / 0.1), `RiskManager`
enforces position and daily-loss limits plus the Defender circuit breaker, and
`OrderRouter` splits large orders with a TWAP schedule. `RegimeDetector`
classifies the market state that feeds both the gating network and fusion rules.

## 9. Known simplifications

- Single symbol (`DOGE-USDT-SWAP`); multi-symbol would need per-symbol buffers.
- Synthetic-trained models (see §5).
- Flow/risk factors depend on live feeds the offline path doesn't provide.
- The live trading path is a reference implementation, not production-hardened.

For the original, more detailed Chinese design document (with sequence diagrams
and per-component rationale), see [`ARCHITECTURE.zh.md`](ARCHITECTURE.zh.md).
