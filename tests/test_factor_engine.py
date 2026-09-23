#!/usr/bin/env python3
"""因子引擎测试"""

import pytest
import numpy as np
import polars as pl
from quant_core.factor_engine import FactorEngine

class TestFactorEngine:
    """因子引擎测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.engine = FactorEngine()
    
    def test_create_engine(self):
        """测试创建引擎"""
        assert self.engine is not None
        assert hasattr(self.engine, 'calculate_factors')
    
    def test_calculate_with_valid_data(self):
        """测试有效数据的因子计算"""
        # 创建测试数据
        n = 100
        test_data = {
            'timestamp': np.arange(n, dtype=float),
            'open': np.random.randn(n) * 10 + 100,
            'high': np.random.randn(n) * 10 + 105,
            'low': np.random.randn(n) * 10 + 95,
            'close': np.random.randn(n) * 10 + 100,
            'volume': np.random.rand(n) * 1000000,
        }
        
        candle_df = pl.DataFrame(test_data)

        # 计算因子（calculate_all_factors 接受 {'candles': df, ...} 数据字典）
        factors = self.engine.calculate_factors({'candles': candle_df})

        assert factors is not None
        # 至少 K 线派生因子（rsi / macd / bollinger）应被计算出来
        assert len(factors) > 0
        assert 'rsi_14' in factors.columns
    
    def test_calculate_with_insufficient_data(self):
        """测试数据不足时的处理"""
        # 创建少量数据
        test_data = {
            'timestamp': [1.0, 2.0],
            'close': [100.0, 101.0],
        }
        
        candle_df = pl.DataFrame(test_data)

        # 数据不足时应返回 None 或空结果
        factors = self.engine.calculate_factors({'candles': candle_df})
        assert factors is None or len(factors) == 0
    
    def test_macd_hist_no_broadcast_error(self):
        """回归测试：MACD 曾因 signal_line 对齐错误抛出广播异常。"""
        n = 100
        rng = np.random.default_rng(0)
        close = np.cumsum(rng.normal(0, 0.5, n)) + 100
        candle_df = pl.DataFrame({
            'timestamp': np.arange(n, dtype=float),
            'open': close, 'high': close + 1, 'low': close - 1,
            'close': close, 'volume': np.abs(rng.normal(1e6, 1e5, n)),
        })
        factors = self.engine.calculate_factors({'candles': candle_df})
        assert 'macd_hist' in factors.columns
        macd = factors['macd_hist'].to_numpy()
        assert len(macd) == n
        # 尾部（预热之后）应有有效数值
        assert np.isfinite(macd[-1])

    def test_rsi_calculation(self):
        """测试RSI计算"""
        # RSI应该在0-100范围内
        n = 50
        close_prices = np.random.randn(n).cumsum() + 100
        
        # 这里可以添加更具体的RSI计算测试
        # rsi = self.engine._calculate_rsi(close_prices, period=14)
        # assert all(0 <= r <= 100 for r in rsi if not np.isnan(r))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
