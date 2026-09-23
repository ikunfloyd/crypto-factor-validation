#!/usr/bin/env python3
"""DataProcessor 测试"""

import numpy as np
import pytest

from quant_core.data_processor import DataProcessor

CANDLE_DTYPE = np.dtype([
    ("timestamp", "f8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
    ("close", "f8"), ("volume", "f8"), ("volume_ccy", "f8"),
])


def _candles(n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(1)
    arr = np.zeros(n, dtype=CANDLE_DTYPE)
    t0, p = 1_700_000_000.0, 0.15
    for i in range(n):
        c = p * (1 + rng.normal(0, 0.002))
        arr[i] = (t0 + i * 60, p, max(p, c) * 1.001, min(p, c) * 0.999, c,
                  abs(rng.normal(1e4, 2e3)), 1.0)
        p = c
    return arr


class TestDataProcessor:
    def setup_method(self):
        self.dp = DataProcessor()

    def test_process_returns_expected_columns(self):
        df = self.dp.process_candle_data(_candles())
        assert len(df) == 120
        for col in ("close", "returns", "volatility_5m", "atr_pct_14",
                    "hour", "weekday", "market_hours"):
            assert col in df.columns, f"missing derived column: {col}"

    def test_empty_input_returns_empty(self):
        empty = np.zeros(0, dtype=CANDLE_DTYPE)
        df = self.dp.process_candle_data(empty)
        assert len(df) == 0

    def test_returns_are_finite(self):
        df = self.dp.process_candle_data(_candles())
        returns = df["returns"].to_numpy()
        # first bar may be null (no prior); the rest must be finite
        assert np.isfinite(returns[1:]).all()
