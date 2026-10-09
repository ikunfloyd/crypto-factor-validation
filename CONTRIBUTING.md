# Contributing

Each rule below exists because breaking it once produced results that looked
fine. Most are pinned by a test; read this before touching the factors, the
model feature list, or the research scripts.

## Running the checks

```bash
pip install -r requirements-dev.txt    # Python 3.10
# or: conda env create -f environment.yml && conda activate doge_trading
python scripts/demo.py --candles 500   # trains the synthetic expert models
pytest                                  # run after the demo, or one test is skipped
```

Run everything from the repository root: imports are absolute from there
(`from quant_core.factor_engine import FactorEngine`).

## Invariants

### Join coarser series as-of, never by index

Funding rates (8h), open interest (5m) and the spot leg do not share the candle
timeline. Join them with `FactorEngine._asof_backward`, which takes the most
recent observation at or before each bar. Pairing two series by array index
silently mixes future observations into the past, because the series start at
different times and sample at different rates — the failure that makes a
backtest profitable and a live system not.
`tests/test_factor_engine.py::TestAsofAlignment` pins this.

Known exception: `calculate_downside_beta` still pairs the BTC leg with the
candles by array index, and returns a constant 0 when there are too few
samples. Treat its output as unreliable until it is rewritten.

### A factor that cannot be computed returns `None`

Return `None` when the input a factor needs is missing; the engine skips it and
model inference substitutes a default for the missing column. Never return a
constant in its place: a constant looks like a feature, occupies a model slot
and carries nothing. Return a series of the candle frame's length —
`calculate_all_factors` drops a series whose length differs, with only a
warning. `TestBasisZScore` and `TestFundingCostPressure` in
`tests/test_factor_engine.py` pin both halves of this.

### Only closed candles

OKX marks the in-progress bar `confirm="0"` and keeps rewriting it until the
minute ends. `scripts/fetch_history.py` drops those bars at ingest. No test
covers this yet, and the live WebSocket ingest (`io_gateway/ws_client.py`) does
not filter unconfirmed bars.

### Training and inference share one feature list

`quant_core/training/train_models.py` and `quant_core/model_inference.py`
declare the same 17 features in the same order. Change both, then retrain. If
the counts differ, every `predict` raises internally, logs an error and returns
0.0, so the process keeps running on dead models. If only the order or a name
differs, nothing raises at all and the models silently score the wrong columns.
`tests/test_model_inference.py::test_feature_contract_matches_training`
guards it.

## Adding a factor

In `quant_core/factor_engine.py`:

1. Write the kernel as a `@staticmethod` decorated with `@jit(nopython=True)`
   over NumPy arrays.
2. Wrap it in a `calculate_*` method that returns a `pl.Series` (or `None`, see
   above).
3. Register it in `self.factor_registry`, and add a regression test. A
   registered factor is picked up by `scripts/factor_research.py`.
4. To feed it to the models as well, add it to both feature lists, update the
   count in `test_feature_contract_matches_training`, and retrain.

Leave `fastmath` off in any kernel that tests for NaN (`np.isnan`, `x != x`),
for example to skip warm-up windows or gaps. Under `fastmath=True`, Numba may
assume no NaNs exist and drop those checks: a loop counting the NaNs in
`[1, nan, 2, nan, 3]` returns 0 instead of 2.
