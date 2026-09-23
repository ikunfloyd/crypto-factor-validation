# DOGE-USDT Trading System

> An event-driven, multi-process high-frequency trading engine for the OKX
> DOGE-USDT perpetual swap, built as a systems-engineering study on Apple Silicon.

<!-- Replace OWNER with your GitHub username once pushed. -->
[![CI](https://github.com/OWNER/doge_usdt_trading/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/doge_usdt_trading/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey.svg)](https://www.apple.com/mac/)

**English** · [中文](README.zh.md)

---

This is a personal project exploring how to build a low-latency trading stack in
Python: separate OS processes for I/O, computation, and execution, wired together
through **zero-copy shared memory** instead of a message queue, with a
write-ahead log for crash recovery and a small Mixture-of-Experts model for
decisioning. The interesting part is the plumbing, not the strategy — see
[Honest status](#honest-status) before reading too much into it.

## Highlights

- **Zero-copy IPC** — a hand-rolled ring buffer over `multiprocessing.shared_memory`;
  producer and consumers read the same bytes with no serialization
  ([`shared_memory/ring_buffer.py`](shared_memory/ring_buffer.py)).
- **Crash recovery** — every tick is appended to a write-ahead log *before* it
  hits shared memory, plus periodic checkpoints, so a kill -9 loses nothing
  ([`io_gateway/wal_writer.py`](io_gateway/wal_writer.py), [`utils/disaster_recovery.py`](utils/disaster_recovery.py)).
- **Async market data** — dual OKX WebSocket endpoints (public + business) with
  heartbeat monitoring and exponential-backoff reconnect
  ([`io_gateway/ws_client.py`](io_gateway/ws_client.py)).
- **Process orchestration** — a supervisor spawns each stage as its own process,
  pins it to performance vs. efficiency cores, health-checks it, and restarts it
  on death ([`orchestrator/main.py`](orchestrator/main.py)).
- **Vectorized compute** — Polars for dataframes, Numba-JIT factor kernels, and a
  three-expert LightGBM MoE with a regime-aware gating network
  ([`quant_core/`](quant_core/)).

## Try it in 30 seconds (offline, no keys)

The demo runs the full compute path — synthetic candles → feature engineering →
factors → MoE → trading signal — with **no exchange account, API keys, or
Apple-Silicon hardware required**. Missing models are trained automatically on
first run.

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate doge_trading
python scripts/demo.py
```

```
[2/4] data processing + factor engine ...
      candle features : 29 columns
      factors computed: 5 -> rsi_14, macd_hist, bollinger_pctb, rolling_vol_ratio, basis_z_score
[3/4] loading MoE experts ...
      experts loaded  : ['hunter', 'sniper', 'defender']
[4/4] inference:
  regime          : high_vol
  gating weights  : hunter=0.00, sniper=0.00, defender=1.00
  fused score     : +0.267
  -> direction    : NEUTRAL   confidence: 0.0%   position size: 0.0%

How the same features route through every regime (gating demo):
  trending  -> SHORT   conf=95.8%  pos=47.9%
  ranging   -> SHORT   conf=87.3%  pos=30.6%
  high_vol  -> NEUTRAL conf= 0.0%  pos= 0.0%
```

## Architecture

```
        OKX WebSocket (public + business)
                    │
          ┌─────────▼─────────┐        ┌────────────────────┐
          │    IO Gateway     │──────▶ │  WAL (msgpack+zlib) │  durability
          │  async ingest     │        └────────────────────┘
          └─────────┬─────────┘
                    │ write (zero-copy)
          ┌─────────▼─────────┐
          │   Shared Memory   │  ring buffers: tick (3600) · candle (5000)
          └─────────┬─────────┘
                    │ read_latest / peek_latest
          ┌─────────▼─────────┐
          │    Quant Core     │  DataProcessor → FactorEngine → MoE inference
          └─────────┬─────────┘
                    │ signal
          ┌─────────▼─────────┐
          │    Execution      │  signal fusion → risk manager → order router
          └───────────────────┘

   Orchestrator supervises every process: CPU-affinity pinning, health
   checks, auto-restart, checkpoint + WAL-replay disaster recovery.
```

Full design write-up with rationale and data-flow diagrams:
[`ARCHITECTURE.md`](ARCHITECTURE.md).

| Layer | Responsibility | Key tech |
|-------|----------------|----------|
| IO Gateway | WebSocket ingest, WAL persistence, cleaning | `asyncio`, `websockets`, `msgpack` |
| Shared Memory | zero-copy inter-process transport | `multiprocessing.shared_memory`, `numpy` |
| Quant Core | features, factors, MoE inference | `polars`, `numba`, `lightgbm` |
| Execution | signal fusion, risk, order routing | `numpy` |
| Orchestrator | process lifecycle, affinity, recovery | `multiprocessing`, `psutil` |
| Dashboard | live monitoring (read-only) | `streamlit`, `plotly` |

## Benchmarks

Measured on an Apple Silicon laptop with `python scripts/benchmark.py` (single
run, warm Numba JIT). Numbers are machine-specific — reproduce them yourself
rather than trusting the table:

| Hot path | Result |
|----------|--------|
| Ring-buffer write (shared memory) | ~1.2M items/s · 0.81 µs/write |
| `peek_latest(1)` read latency | 1.4 µs median |
| Data processing (numpy → Polars + features) | ~1.5M candles/s |
| Factor engine (candle factors, Numba) | ~68K candles/s · 30 ms per 2000-bar frame |
| MoE inference (features + predict) | 218 µs median · ~4.6K predictions/s |

```bash
python scripts/benchmark.py               # or --candles 5000 --iters 100000
```

## Running the full system

The live pipeline needs OKX API credentials and is intended for Apple Silicon
(the orchestrator pins P-/E-cores). Each stage is also runnable standalone.

```bash
cp .env.example .env && source .env        # add your OKX_API_KEY / OKX_SECRET / OKX_PASSWORD

python -m orchestrator.main                # supervised: spawns every stage
# — or run stages individually —
python -m io_gateway.main                  # OKX ingest → shared memory + WAL
python -m quant_core.main_v2               # factors + MoE inference
python -m execution.main                   # fusion, risk, routing
streamlit run dashboard/main.py            # monitoring UI
```

Retrain the expert models at any time:

```bash
python -m quant_core.training.train_models  # writes data/models/{hunter,sniper,defender}/
```

## Project layout

```
io_gateway/      WebSocket ingest, WAL read/write, data cleaning
shared_memory/   Arrow/numpy ring buffers over shared memory + IPC manager
quant_core/      data processing, factor engine, MoE inference, training
execution/       regime detection, signal fusion, risk, order routing
orchestrator/    process supervisor, CPU affinity, disaster recovery
dashboard/       Streamlit monitoring app
utils/           logging, health monitor, performance, recovery helpers
config/          settings (flat constants) + trading_config (dataclasses)
scripts/         demo.py (offline pipeline) + benchmark.py (hot-path timings)
tests/           pytest suites (factors, ring buffer, data processor, MoE)
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                    # config in pytest.ini
pytest tests/test_ring_buffer.py -v       # a single suite
```

## Honest status

This is a learning project, not a production trading system. What that means:

- **The models are trained on *synthetic* data** ([`quant_core/training/train_models.py`](quant_core/training/train_models.py)).
  The demo showcases the factor → scaling → MoE → signal *pipeline*, not a
  profitable edge. Don't read alpha into the numbers.
- **Single symbol.** Everything is wired for `DOGE-USDT-SWAP`; multi-symbol
  support would need per-symbol shared-memory segments.
- **Some factors need live feeds.** The funding-rate, open-interest, and
  order-flow factors require data the offline demo doesn't have, so they're
  skipped there; only the candle-derived factors compute.
- **Not battle-tested with real capital.** Treat the live path as a reference
  implementation, not something to point at your account.

## Disclaimer

For education and research only. Cryptocurrency trading carries substantial risk
of loss. Nothing here is financial advice, and the authors accept no liability
for any use of this software.

## License

[MIT](LICENSE).
