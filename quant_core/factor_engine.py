import numpy as np
import polars as pl
import numba
from numba import jit, prange
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# M3 性能优化配置
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def _parallel_operation(arr):
    """Numba并行化操作的示例模板"""
    result = np.empty_like(arr)
    for i in prange(len(arr)):
        result[i] = arr[i] * 2  # 示例操作
    return result

class FactorEngine:
    """高性能因子计算引擎（M3优化版）"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.factor_cache = {}  # 因子缓存

        # 因子注册表
        self.factor_registry = {
            # A组：动量与技术
            'rsi_14': self.calculate_rsi,
            'macd_hist': self.calculate_macd_hist,
            'bollinger_pctb': self.calculate_bollinger_pctb,
            'rolling_vol_ratio': self.calculate_rolling_vol_ratio,
            
            # B组：资金与博弈
            'basis_z_score': self.calculate_basis_z_score,
            'oi_change_rate': self.calculate_oi_change_rate,
            'net_taker_flow': self.calculate_net_taker_flow,
            
            # C组：风险特征
            'downside_beta': self.calculate_downside_beta,
            'funding_cost_pressure': self.calculate_funding_cost_pressure,
        }
        
        logger.info("Factor Engine initialized with M3 optimizations")
    
    def calculate_all_factors(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        计算所有因子
        
        Args:
            data: 包含各种数据源的字典
                - candles: K线数据
                - ticks: tick数据
                - funding_rates: 资金费率
                - open_interest: 持仓量
                - btc_data: BTC数据（用于计算beta）
        """
        try:
            logger.info("Calculating all factors...")
            
            # 提取数据
            candles = data.get('candles')
            if candles is None or len(candles) == 0:
                logger.warning("No candle data available")
                return pl.DataFrame()
            
            # 添加诊断日志：输入数据状态
            logger.debug(f"Factor engine input - candles: {len(candles)} rows, available_keys: {list(data.keys())}")
            
            # 计算所有注册的因子（能算的就算；缺数据源的因子会被跳过）
            results = []
            factor_names = []
            failed_factors = []
            
            for factor_name, factor_func in self.factor_registry.items():
                try:
                    factor_series = factor_func(data)
                    if factor_series is not None and len(factor_series) > 0:
                        # 对齐时间戳
                        if len(factor_series) == len(candles):
                            results.append(factor_series)
                            factor_names.append(factor_name)
                        else:
                            logger.warning(f"Factor {factor_name} length mismatch: {len(factor_series)} vs {len(candles)}")
                            failed_factors.append(f"{factor_name}(length_mismatch)")
                    else:
                        failed_factors.append(f"{factor_name}(None_or_empty)")
                except Exception as e:
                    logger.error(f"Error calculating factor {factor_name}: {e}")
                    failed_factors.append(f"{factor_name}(error)")
                    continue
            
            # 添加诊断日志：因子计算结果汇总
            logger.info(f"Factor calculation results - attempted: {len(self.factor_registry)}, successful: {len(results)}, failed: {len(failed_factors)}")
            if failed_factors:
                logger.debug(f"Failed factors: {', '.join(failed_factors)}")
            
            # 合并所有因子
            if results:
                # 创建一个包含所有因子的DataFrame
                factor_df = pl.DataFrame({
                    'timestamp': candles['timestamp'],
                    **{name: series for name, series in zip(factor_names, results)}
                })
                
                # 缓存计算结果
                self.factor_cache = {name: series for name, series in zip(factor_names, results)}
                
                logger.info(f"Calculated {len(factor_names)} factors successfully")
                return factor_df
            else:
                logger.warning("No factors calculated")
                return pl.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in calculate_all_factors: {e}")
            import traceback
            traceback.print_exc()
            return pl.DataFrame()
    
    def calculate_factors(self, data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        别名方法，兼容旧API
        向后兼容：calculate_factors() -> calculate_all_factors()
        """
        return self.calculate_all_factors(data)
    
    # ==================== A组：动量与技术因子 ====================
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """使用Numba加速的RSI计算"""
        n = len(prices)
        if n < period + 1:
            return np.full(n, np.nan)
        
        rsi = np.full(n, np.nan)
        gains = np.zeros(n)
        losses = np.zeros(n)
        
        # 计算价格变化
        for i in range(1, n):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains[i] = change
                losses[i] = 0
            else:
                gains[i] = 0
                losses[i] = -change
        
        # 计算初始平均值
        avg_gain = np.mean(gains[1:period+1])
        avg_loss = np.mean(losses[1:period+1])
        
        if avg_loss == 0:
            rsi[period] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))
        
        # 递归计算后续RSI值
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_rsi(self, data: Dict[str, Any], period: int = 14) -> Optional[pl.Series]:
        """计算RSI指标"""
        try:
            candles = data.get('candles')
            if candles is None or len(candles) < period + 1:
                return None
            
            close_prices = candles['close'].to_numpy()
            rsi_values = self._calculate_rsi_numba(close_prices, period)
            
            return pl.Series('rsi_14', rsi_values)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _calculate_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """计算指数移动平均"""
        n = len(prices)
        if n < period:
            return np.full(n, np.nan)
        
        ema = np.full(n, np.nan)
        multiplier = 2.0 / (period + 1)
        
        # 初始EMA为简单移动平均
        ema[period-1] = np.mean(prices[:period])
        
        # 计算后续EMA
        for i in range(period, n):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def calculate_macd_hist(self, data: Dict[str, Any], 
                           fast_period: int = 12, 
                           slow_period: int = 26, 
                           signal_period: int = 9) -> Optional[pl.Series]:
        """计算MACD柱状图"""
        try:
            candles = data.get('candles')
            if candles is None or len(candles) < slow_period + signal_period:
                return None
            
            close_prices = candles['close'].to_numpy()
            
            # 计算EMA
            ema_fast = self._calculate_ema_numba(close_prices, fast_period)
            ema_slow = self._calculate_ema_numba(close_prices, slow_period)
            
            # 计算MACD线
            macd_line = ema_fast - ema_slow
            
            # 计算信号线（MACD的EMA）
            # signal_line 基于 macd_line[slow_period-1:] 计算，其索引 i 对应
            # macd_line[slow_period-1+i]；EMA 从子数组第 signal_period-1 个元素起有效，
            # 即对应 macd_line[slow_period+signal_period-2]。
            signal_line = self._calculate_ema_numba(macd_line[slow_period-1:], signal_period)

            # 计算MACD柱状图（对齐 macd_line 与 signal_line 的有效区间）
            macd_hist = np.full(len(close_prices), np.nan)

            start_idx = slow_period + signal_period - 2
            macd_hist[start_idx:] = macd_line[start_idx:] - signal_line[signal_period-1:]
            
            return pl.Series('macd_hist', macd_hist)
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return None
    
    def calculate_bollinger_pctb(self, data: Dict[str, Any], 
                                period: int = 20, 
                                std_dev: float = 2.0) -> Optional[pl.Series]:
        """计算布林带百分比B"""
        try:
            candles = data.get('candles')
            if candles is None or len(candles) < period:
                return None
            
            close_prices = candles['close'].to_numpy()
            n = len(close_prices)
            
            pct_b = np.full(n, np.nan)
            
            for i in range(period-1, n):
                window = close_prices[i-period+1:i+1]
                sma = np.mean(window)
                std = np.std(window)
                
                upper_band = sma + std_dev * std
                lower_band = sma - std_dev * std
                
                if upper_band != lower_band:
                    pct_b[i] = (close_prices[i] - lower_band) / (upper_band - lower_band)
                else:
                    pct_b[i] = 0.5
            
            return pl.Series('bollinger_pctb', pct_b)
        except Exception as e:
            logger.error(f"Error calculating Bollinger %B: {e}")
            return None
    
    def calculate_rolling_vol_ratio(self, data: Dict[str, Any], 
                                   short_window: int = 60,  # 1小时
                                   long_window: int = 1440) -> Optional[pl.Series]:  # 24小时
        """计算成交量比率（短期/长期）"""
        try:
            candles = data.get('candles')
            if candles is None or len(candles) < long_window:
                return None
            
            volume = candles['volume'].to_numpy()
            n = len(volume)
            
            vol_ratio = np.full(n, np.nan)
            
            for i in range(long_window-1, n):
                short_vol = np.mean(volume[i-short_window+1:i+1])
                long_vol = np.mean(volume[i-long_window+1:i+1])
                
                if long_vol > 0:
                    vol_ratio[i] = short_vol / long_vol
                else:
                    vol_ratio[i] = 1.0
            
            return pl.Series('rolling_vol_ratio', vol_ratio)
        except Exception as e:
            logger.error(f"Error calculating rolling volume ratio: {e}")
            return None
    
    # ==================== B组：资金与博弈因子 ====================
    
    @staticmethod
    def _asof_backward(target_ts: np.ndarray, source_ts: np.ndarray,
                       source_val: np.ndarray) -> np.ndarray:
        """
        把低频序列对齐到高频时间轴，只允许向后看（as-of backward）。

        对每个 target_ts[i]，取 source 中**不晚于**它的最近一个观测。找不到
        （target 早于所有 source）时填 NaN。

        这一步必须按时间戳做，不能按数组下标配对：两个序列的起点和采样频率
        通常不同，按下标对齐会把几小时之后的观测配到当前 K 线上，产生随机的
        前视偏差 —— 回测赚钱、实盘归零的典型原因。
        """
        n = len(target_ts)
        out = np.full(n, np.nan)
        if len(source_ts) == 0:
            return out

        # side='right' - 1 给出最后一个 <= target 的位置；-1 表示无可用历史
        idx = np.searchsorted(source_ts, target_ts, side='right') - 1
        valid = idx >= 0
        out[valid] = source_val[idx[valid]]
        return out

    def calculate_basis_z_score(self, data: Dict[str, Any], 
                               lookback: int = 1440) -> Optional[pl.Series]:  # 24小时
        """
        计算永续-现货基差的滚动Z-Score。

        基差 = (永续价 - 现货价) / 现货价，反映合约相对现货的升贴水，是永续
        市场多空拥挤度的直接度量。需要现货K线（data['spot_candles']）；缺失
        时返回 None 而不是退化成常数，避免给模型喂一个零信息量的特征。

        用 scripts/fetch_history.py --with-spot 拉取现货腿。
        """
        try:
            candles = data.get('candles')
            spot = data.get('spot_candles')

            if candles is None or len(candles) < lookback:
                return None

            if spot is None or len(spot) == 0:
                logger.warning(
                    "basis_z_score: 缺少现货数据(spot_candles)，跳过该因子。"
                    "用 scripts/fetch_history.py --with-spot 拉取现货K线。"
                )
                return None

            if 'timestamp' not in candles.columns or 'timestamp' not in spot.columns:
                logger.warning("basis_z_score: 缺少 timestamp 列，无法按时间对齐")
                return None

            perp_ts = candles['timestamp'].to_numpy().astype(np.float64)
            perp_price = candles['close'].to_numpy().astype(np.float64)
            spot_ts = spot['timestamp'].to_numpy().astype(np.float64)
            spot_close = spot['close'].to_numpy().astype(np.float64)

            # 现货按时间戳对齐到永续的时间轴（只向后看）
            spot_price = self._asof_backward(perp_ts, spot_ts, spot_close)

            with np.errstate(divide='ignore', invalid='ignore'):
                basis = np.where(spot_price > 0,
                                 (perp_price - spot_price) / spot_price,
                                 np.nan)

            covered = int(np.isfinite(basis).sum())
            if covered < lookback:
                logger.warning(
                    f"basis_z_score: 现货覆盖不足({covered} < {lookback})，跳过该因子"
                )
                return None

            n = len(basis)
            basis_z = np.full(n, np.nan)

            for i in range(lookback - 1, n):
                window = basis[i - lookback + 1:i + 1]
                if not np.isfinite(basis[i]):
                    continue

                mean = np.nanmean(window)
                std = np.nanstd(window)

                # std==0 意味着窗口内基差没有任何变化（基本只会出现在数据异常
                # 的情况下）。此时Z-Score无定义，留 NaN 让下游知道这里没信息，
                # 而不是填 0 假装因子是中性的。
                if std > 0:
                    basis_z[i] = (basis[i] - mean) / std

            return pl.Series('basis_z_score', basis_z)
        except Exception as e:
            logger.error(f"Error calculating basis Z-score: {e}")
            return None
    
    def calculate_oi_change_rate(self, data: Dict[str, Any]) -> Optional[pl.Series]:
        """
        计算持仓量变化率（单位成交带来的持仓变化）。

        OI上升伴随大成交 = 新仓在建（趋势可能延续）；OI下降伴随大成交 = 平仓
        了结（趋势可能衰竭）。除以成交量是为了让不同活跃度的时段可比。

        OI序列通常比K线低频（OKX的统计口径是5m），因此按时间戳做 as-of
        backward 对齐，保证每根K线只用到该时刻已经发生的OI观测。
        """
        try:
            candles = data.get('candles')
            oi_data = data.get('open_interest')

            if candles is None or oi_data is None or len(candles) < 2:
                return None

            if 'timestamp' not in candles.columns or 'timestamp' not in oi_data.columns:
                logger.warning("oi_change_rate: 缺少 timestamp 列，无法按时间对齐")
                return None

            candle_ts = candles['timestamp'].to_numpy().astype(np.float64)
            volume = candles['volume'].to_numpy().astype(np.float64)
            oi_ts = oi_data['timestamp'].to_numpy().astype(np.float64)
            oi_values = oi_data['oi'].to_numpy().astype(np.float64)

            # 对齐到K线时间轴：每根K线取当时最新的OI读数
            oi_aligned = self._asof_backward(candle_ts, oi_ts, oi_values)

            n = len(candle_ts)
            oi_change_rate = np.full(n, np.nan)

            for i in range(1, n):
                prev_oi, cur_oi = oi_aligned[i - 1], oi_aligned[i]
                if not (np.isfinite(prev_oi) and np.isfinite(cur_oi)):
                    continue
                if volume[i] > 0:
                    oi_change_rate[i] = (cur_oi - prev_oi) / volume[i]

            covered = int(np.isfinite(oi_change_rate).sum())
            if covered == 0:
                logger.warning("oi_change_rate: OI与K线时间范围无重叠，跳过该因子")
                return None

            return pl.Series('oi_change_rate', oi_change_rate)
        except Exception as e:
            logger.error(f"Error calculating OI change rate: {e}")
            return None
    
    def calculate_net_taker_flow(self, data: Dict[str, Any], 
                                lookback: int = 60) -> Optional[pl.Series]:
        """
        计算主动买卖净流量（成交量加权）。

        有逐笔成交数据（data['trades'] 带 side 字段）时用真实的主动买卖方向；
        没有时退化为 Chaikin Money Flow 式的估计：用收盘价在当根K线区间内的
        相对位置 CLV = ((close-low) - (high-close)) / (high-low) 作为买卖压力
        的代理，再按成交量加权。收盘贴近最高价说明买方吃掉了卖单，反之亦然。

        原实现用 sign(close[i] - close[i-1]) 当作主动买卖方向，有两个问题：
          1. 那是K线**之间**的收益率符号，与"主动成交方向"无关；
          2. 它与收益率严格共线。一旦标签定义稍有偏移（例如误用当期收益而非
             前向收益），模型会直接抄答案，回测夏普虚高、实盘归零 —— 这是
             量化回测最经典的失败模式。
        CLV 用的是K线**内部**的 OHLC 相对位置，不是相邻K线的收益，因此不存在
        这个共线性。
        """
        try:
            candles = data.get('candles')
            if candles is None or len(candles) < lookback:
                return None

            trades = data.get('trades')
            n = len(candles)

            if trades is not None and len(trades) > 0 and 'side' in getattr(trades, 'columns', []):
                # 真实逐笔：买方主动为 +size，卖方主动为 -size，按K线时间桶汇总
                signed = self._signed_trade_volume(candles, trades)
                if signed is None:
                    return None
                flow, weight = signed
            else:
                high = candles['high'].to_numpy().astype(np.float64)
                low = candles['low'].to_numpy().astype(np.float64)
                close = candles['close'].to_numpy().astype(np.float64)
                volume = candles['volume'].to_numpy().astype(np.float64)

                rng = high - low
                with np.errstate(divide='ignore', invalid='ignore'):
                    clv = np.where(rng > 0, ((close - low) - (high - close)) / rng, 0.0)

                flow = clv * volume
                weight = volume

            # 滚动的净流量占总成交量之比，范围约在 [-1, 1]
            net_taker_flow = np.full(n, np.nan)
            for i in range(lookback - 1, n):
                w = weight[i - lookback + 1:i + 1].sum()
                if w > 0:
                    net_taker_flow[i] = flow[i - lookback + 1:i + 1].sum() / w

            return pl.Series('net_taker_flow', net_taker_flow)
        except Exception as e:
            logger.error(f"Error calculating net taker flow: {e}")
            return None

    def _signed_trade_volume(self, candles, trades):
        """把逐笔成交按 side 聚合到K线时间桶，返回 (净流量, 总量) 两个数组。"""
        try:
            if 'timestamp' not in candles.columns or 'timestamp' not in trades.columns:
                return None

            candle_ts = candles['timestamp'].to_numpy().astype(np.float64)
            trade_ts = trades['timestamp'].to_numpy().astype(np.float64)
            size = trades['size'].to_numpy().astype(np.float64)
            side = trades['side'].to_numpy()

            sign = np.where(side == 'buy', 1.0, -1.0)

            # 每笔成交归入不晚于它的那根K线
            idx = np.searchsorted(candle_ts, trade_ts, side='right') - 1
            valid = idx >= 0

            flow = np.zeros(len(candle_ts))
            weight = np.zeros(len(candle_ts))
            np.add.at(flow, idx[valid], (sign * size)[valid])
            np.add.at(weight, idx[valid], size[valid])
            return flow, weight
        except Exception as e:
            logger.error(f"Error aggregating signed trade volume: {e}")
            return None
    
    # ==================== C组：风险特征因子 ====================
    
    def calculate_downside_beta(self, data: Dict[str, Any], 
                               lookback: int = 1440) -> Optional[pl.Series]:
        """计算下行Beta（DOGE相对于BTC在下跌时的相关系数）"""
        try:
            doge_candles = data.get('candles')
            btc_data = data.get('btc_data')
            
            if doge_candles is None or btc_data is None:
                return None
            
            doge_returns = doge_candles['close'].pct_change().to_numpy()
            btc_returns = btc_data['close'].pct_change().to_numpy()
            
            n = min(len(doge_returns), len(btc_returns))
            if n < lookback:
                return None
            
            downside_beta = np.full(n, np.nan)
            
            for i in range(lookback-1, n):
                # 只考虑下跌日
                doge_window = doge_returns[i-lookback+1:i+1]
                btc_window = btc_returns[i-lookback+1:i+1]
                
                # 找出两个都下跌的时段
                downside_mask = (doge_window < 0) & (btc_window < 0)
                
                if np.sum(downside_mask) > 10:  # 至少需要10个样本
                    downside_corr = np.corrcoef(
                        doge_window[downside_mask], 
                        btc_window[downside_mask]
                    )[0, 1]
                    downside_beta[i] = downside_corr
                else:
                    downside_beta[i] = 0
            
            return pl.Series('downside_beta', downside_beta[:len(doge_candles)])
        except Exception as e:
            logger.error(f"Error calculating downside beta: {e}")
            return None
    
    def calculate_funding_cost_pressure(self, data: Dict[str, Any],
                                        lookback: int = 90) -> Optional[pl.Series]:
        """
        计算资金费率压力：当前资金费率相对其近期分布的Z-Score。

        资金费率为正且处于历史高位，说明多头拥挤、持多仓要持续付费，是反转
        风险的前兆；负值极端则相反。用Z-Score而不是原始费率，是因为费率的
        绝对水平会随市场周期漂移，只有"相对自身历史有多极端"才可比。

        两个此前的实现问题已修正：
          1. 原先按数组下标截断，资金费率每8小时才一个点，长度远短于K线，
             整个因子会因长度不匹配被 calculate_all_factors 丢弃 —— 也就是说
             它从未真正进入过模型。现按时间戳 as-of backward 对齐。
          2. 原先把费率乘以常数8.0 当作"压力"。对树模型而言 x 与 8x 完全
             等价（分裂点随之缩放），该变换不产生任何新信息。

        Args:
            lookback: Z-Score的回看窗口，单位是资金费率结算次数（8小时一次，
                      默认90次约等于30天）。
        """
        try:
            candles = data.get('candles')
            funding_rates = data.get('funding_rates')

            if candles is None or funding_rates is None or len(funding_rates) == 0:
                return None

            if 'timestamp' not in candles.columns or 'timestamp' not in funding_rates.columns:
                logger.warning("funding_cost_pressure: 缺少 timestamp 列，无法按时间对齐")
                return None

            fr_ts = funding_rates['timestamp'].to_numpy().astype(np.float64)
            fr_rate = funding_rates['rate'].to_numpy().astype(np.float64)

            # 先在资金费率自己的时间轴上算滚动Z-Score，再对齐到K线。
            # 顺序不能反：在K线时间轴上算会因为同一个费率被重复上百次而
            # 严重低估标准差。
            m = len(fr_rate)
            fr_z = np.full(m, np.nan)
            for i in range(m):
                start = max(0, i - lookback + 1)
                window = fr_rate[start:i + 1]
                if len(window) < min(10, lookback):
                    continue
                mean, std = np.mean(window), np.std(window)
                if std > 0:
                    fr_z[i] = (fr_rate[i] - mean) / std

            candle_ts = candles['timestamp'].to_numpy().astype(np.float64)
            pressure = self._asof_backward(candle_ts, fr_ts, fr_z)

            covered = int(np.isfinite(pressure).sum())
            if covered == 0:
                logger.warning("funding_cost_pressure: 资金费率与K线时间范围无重叠，跳过该因子")
                return None

            return pl.Series('funding_cost_pressure', pressure)
        except Exception as e:
            logger.error(f"Error calculating funding cost pressure: {e}")
            return None
    
    def get_factor_descriptions(self) -> Dict[str, str]:
        """获取因子描述"""
        return {
            'rsi_14': '相对强弱指数（14周期）',
            'macd_hist': 'MACD柱状图',
            'bollinger_pctb': '布林带百分比B',
            'rolling_vol_ratio': '成交量比率（1小时/24小时）',
            'basis_z_score': '基差Z-Score',
            'oi_change_rate': '持仓量变化率',
            'net_taker_flow': '主动买卖净流量',
            'downside_beta': '下行Beta（相对于BTC）',
            'funding_cost_pressure': '资金费率压力值'
        }