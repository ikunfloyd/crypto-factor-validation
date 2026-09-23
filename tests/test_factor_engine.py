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

class TestAsofAlignment:
    """
    as-of backward 对齐测试。

    这是低频序列（OI、资金费率、现货）接入K线时间轴的唯一入口，一旦它向前
    看，整条回测链路的收益都是假的 —— 所以这里的断言比因子本身更重要。
    """

    def setup_method(self):
        self.engine = FactorEngine()

    def test_picks_most_recent_past_observation(self):
        """每个目标时刻取不晚于它的最近一个观测"""
        source_ts = np.array([0.0, 10.0, 20.0])
        source_val = np.array([1.0, 2.0, 3.0])
        target_ts = np.array([5.0, 15.0, 25.0])

        out = self.engine._asof_backward(target_ts, source_ts, source_val)

        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_exact_timestamp_match_is_inclusive(self):
        """时间戳正好相等时该观测可用（它在那一刻已经发生）"""
        source_ts = np.array([0.0, 10.0, 20.0])
        source_val = np.array([1.0, 2.0, 3.0])

        out = self.engine._asof_backward(np.array([10.0]), source_ts, source_val)

        assert out[0] == 2.0

    def test_never_uses_future_values(self):
        """核心不变量：输出只能来自 <= 目标时刻的观测"""
        rng = np.random.default_rng(0)
        source_ts = np.sort(rng.uniform(0, 1000, 50))
        source_val = rng.normal(size=50)
        target_ts = np.sort(rng.uniform(0, 1000, 200))

        out = self.engine._asof_backward(target_ts, source_ts, source_val)

        for t, v in zip(target_ts, out):
            if np.isnan(v):
                continue
            # 取值必须出现在目标时刻之前（含）的观测集合里
            allowed = source_val[source_ts <= t]
            assert v in allowed, f"t={t} 用到了未来的观测 {v}"
            # 且必须是其中最近的那个
            assert v == allowed[-1]

    def test_target_before_any_source_is_nan(self):
        """目标早于所有观测时填 NaN，不外推"""
        out = self.engine._asof_backward(
            np.array([-5.0, 5.0]), np.array([0.0, 10.0]), np.array([1.0, 2.0])
        )

        assert np.isnan(out[0])
        assert out[1] == 1.0

    def test_empty_source_returns_all_nan(self):
        """没有低频数据时整列 NaN，调用方据此跳过该因子"""
        out = self.engine._asof_backward(
            np.array([1.0, 2.0]), np.array([]), np.array([])
        )

        assert np.isnan(out).all()


class TestBasisZScore:
    """基差因子：缺现货数据时必须跳过，而不是退化成常数。"""

    def setup_method(self):
        self.engine = FactorEngine()

    def _candles(self, n, close):
        return pl.DataFrame({
            "timestamp": np.arange(n, dtype=float),
            "close": close,
            "volume": np.ones(n),
        })

    def test_returns_none_without_spot_data(self):
        """
        回归测试：此前该因子令 spot_price = perp_price，基差恒为 0，
        Z-Score 恒为 0 —— 一个零信息量的特征白占一个特征位。
        """
        n = 1500
        candles = self._candles(n, np.full(n, 100.0))

        result = self.engine.calculate_basis_z_score({"candles": candles})

        assert result is None

    def test_computes_real_basis_when_spot_present(self):
        """有现货腿时输出真实基差的Z-Score，且不恒为零"""
        n = 1500
        rng = np.random.default_rng(1)
        spot_close = 100.0 + rng.normal(0, 1, n).cumsum() * 0.01
        # 永续在现货上叠加一个波动的升贴水
        premium = 0.002 + rng.normal(0, 0.001, n)
        perp_close = spot_close * (1 + premium)

        candles = self._candles(n, perp_close)
        spot = pl.DataFrame({
            "timestamp": np.arange(n, dtype=float),
            "close": spot_close,
        })

        result = self.engine.calculate_basis_z_score(
            {"candles": candles, "spot_candles": spot}
        )

        assert result is not None
        values = result.to_numpy()
        finite = values[np.isfinite(values)]
        assert len(finite) > 0
        assert np.abs(finite).max() > 0, "Z-Score 不应恒为零"


class TestOIChangeRate:
    """持仓量因子：低频OI必须按时间戳而非下标对齐到K线。"""

    def setup_method(self):
        self.engine = FactorEngine()

    def test_aligns_low_frequency_oi_by_timestamp(self):
        """
        回归测试：此前按数组下标配对，5m的OI会被配到1m的K线上，
        时间偏差随下标线性放大，形成随机的前视偏差。
        """
        n = 20
        candles = pl.DataFrame({
            "timestamp": np.arange(n, dtype=float) * 60.0,   # 1分钟
            "close": np.full(n, 100.0),
            "volume": np.full(n, 10.0),
        })
        # OI每5分钟一个点，且起点晚于K线起点
        oi = pl.DataFrame({
            "timestamp": np.array([300.0, 600.0, 900.0]),
            "oi": np.array([1000.0, 1100.0, 1050.0]),
        })

        result = self.engine.calculate_oi_change_rate(
            {"candles": candles, "open_interest": oi}
        )

        assert result is not None
        values = result.to_numpy()

        # 前5根K线（t<300）没有任何OI观测，必须是 NaN
        assert np.isnan(values[:5]).all()

        # t=600 那根K线上，OI 刚从 1000 跳到 1100，变化率 = 100/10 = 10
        idx = 10  # timestamp = 600
        assert values[idx] == pytest.approx(10.0)

    def test_returns_none_when_ranges_do_not_overlap(self):
        """OI与K线时间范围完全不重叠时跳过该因子"""
        candles = pl.DataFrame({
            "timestamp": np.arange(10, dtype=float),
            "close": np.full(10, 100.0),
            "volume": np.full(10, 1.0),
        })
        oi = pl.DataFrame({
            "timestamp": np.array([1000.0, 2000.0]),
            "oi": np.array([1.0, 2.0]),
        })

        result = self.engine.calculate_oi_change_rate(
            {"candles": candles, "open_interest": oi}
        )

        assert result is None


class TestFundingCostPressure:
    """资金费率因子：低频费率必须对齐到K线长度，否则整个因子会被静默丢弃。"""

    def setup_method(self):
        self.engine = FactorEngine()

    def test_output_length_matches_candles(self):
        """
        回归测试：资金费率每8小时一个点，原实现按 min(len(rate), len(candles))
        截断，返回的序列远短于K线，calculate_all_factors 会因长度不匹配把它
        丢掉 —— 该因子从未进入过模型。
        """
        n_candles = 480  # 8小时的1分钟K线
        candles = pl.DataFrame({
            "timestamp": np.arange(n_candles, dtype=float) * 60_000,
            "close": np.full(n_candles, 100.0),
            "volume": np.ones(n_candles),
        })
        # 20次结算，跨度远大于K线，保证有可用历史
        n_fr = 20
        rng = np.random.default_rng(2)
        funding = pl.DataFrame({
            "timestamp": (np.arange(n_fr, dtype=float) - n_fr + 2) * 8 * 3600_000,
            "rate": rng.normal(0.0001, 0.00005, n_fr),
        })

        result = self.engine.calculate_funding_cost_pressure(
            {"candles": candles, "funding_rates": funding}, lookback=10
        )

        assert result is not None
        assert len(result) == n_candles, "长度必须等于K线数，否则会被静默丢弃"

    def test_survives_calculate_all_factors(self):
        """端到端：修复后该因子应真正出现在因子表里"""
        n_candles = 600
        rng = np.random.default_rng(3)
        close = 100.0 + rng.normal(0, 1, n_candles).cumsum() * 0.05
        candles = pl.DataFrame({
            "timestamp": np.arange(n_candles, dtype=float) * 60_000,
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": np.abs(rng.normal(100, 10, n_candles)),
        })
        n_fr = 30
        funding = pl.DataFrame({
            "timestamp": (np.arange(n_fr, dtype=float) - n_fr + 2) * 8 * 3600_000,
            "rate": rng.normal(0.0001, 0.00005, n_fr),
        })

        factors = self.engine.calculate_all_factors(
            {"candles": candles, "funding_rates": funding}
        )

        assert "funding_cost_pressure" in factors.columns


class TestNetTakerFlow:
    """主动买卖流：不得与收益率共线，否则是泄漏源。"""

    def setup_method(self):
        self.engine = FactorEngine()

    def _candles(self, n, seed=4, wick_seed=None):
        """
        构造形状真实的K线：影线长度独立于收盘价序列。

        wick_seed 不同则影线不同、收盘价完全相同 —— 用来隔离"K线内部结构"
        这一个变量。
        """
        rng = np.random.default_rng(seed)
        close = 100.0 + rng.normal(0, 1, n).cumsum() * 0.05
        open_ = np.concatenate([[close[0]], close[:-1]])

        wrng = np.random.default_rng(seed if wick_seed is None else wick_seed)
        upper = np.abs(wrng.normal(0, 0.03, n))
        lower = np.abs(wrng.normal(0, 0.03, n))

        return pl.DataFrame({
            "timestamp": np.arange(n, dtype=float) * 60_000,
            "open": open_,
            "high": np.maximum(open_, close) + upper,
            "low": np.minimum(open_, close) - lower,
            "close": close,
            "volume": np.abs(rng.normal(100, 20, n)),
        })

    def test_depends_on_intrabar_shape_not_just_returns(self):
        """
        回归测试：原实现 net_flow = sign(close.diff())，是收益率符号的确定性
        函数 —— 只要收盘价序列相同，因子值必然逐点相同，K线内部结构完全不
        起作用。新实现用 OHLC 的相对位置，因此同样的收盘价配不同的影线必须
        给出不同的因子值。
        """
        n = 600
        a = self._candles(n, seed=4, wick_seed=100)
        b = self._candles(n, seed=4, wick_seed=200)

        # 前提：两组K线的收盘价完全一致
        np.testing.assert_array_equal(a["close"].to_numpy(), b["close"].to_numpy())

        fa = self.engine.calculate_net_taker_flow({"candles": a}).to_numpy()
        fb = self.engine.calculate_net_taker_flow({"candles": b}).to_numpy()

        mask = np.isfinite(fa) & np.isfinite(fb)
        assert mask.sum() > 0
        assert not np.allclose(fa[mask], fb[mask]), (
            "收盘价相同、K线形状不同却得到相同因子值，"
            "说明仍然只用了收益率信息"
        )

    def test_bounded_range(self):
        """净流量是占总成交量的比例，应落在 [-1, 1]"""
        result = self.engine.calculate_net_taker_flow({"candles": self._candles(400)})

        values = result.to_numpy()
        finite = values[np.isfinite(values)]
        assert len(finite) > 0
        assert finite.min() >= -1.0 - 1e-9
        assert finite.max() <= 1.0 + 1e-9

    def test_uses_real_trades_when_available(self):
        """有带 side 的逐笔数据时优先用真实主动买卖方向"""
        n = 200
        candles = self._candles(n)
        # 全部为主动买入 -> 净流量应接近 +1
        n_trades = 2000
        rng = np.random.default_rng(5)
        trades = pl.DataFrame({
            "timestamp": rng.uniform(0, n * 60_000, n_trades),
            "size": np.abs(rng.normal(10, 2, n_trades)),
            "side": np.array(["buy"] * n_trades),
        })

        result = self.engine.calculate_net_taker_flow(
            {"candles": candles, "trades": trades}
        )

        assert result is not None
        finite = result.to_numpy()[np.isfinite(result.to_numpy())]
        assert len(finite) > 0
        assert finite.min() > 0.9, "全买单时净流量应接近 +1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
