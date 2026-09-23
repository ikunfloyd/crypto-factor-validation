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
    
    def calculate_basis_z_score(self, data: Dict[str, Any], 
                               lookback: int = 1440) -> Optional[pl.Series]:  # 24小时
        """计算基差Z-Score"""
        try:
            # 在实际系统中，需要永续合约价格和现货价格
            # 这里使用简化版本：假设永续价格在candles中，现货价格用close近似
            candles = data.get('candles')
            if candles is None or len(candles) < lookback:
                return None
            
            # 假设永续价格就是close（在实际中需要获取永续合约和现货的价差）
            perp_price = candles['close'].to_numpy()
            spot_price = perp_price  # 简化：假设现货价格相同
            
            # 计算基差
            basis = (perp_price - spot_price) / spot_price
            
            # 计算Z-Score
            n = len(basis)
            basis_z = np.full(n, np.nan)
            
            for i in range(lookback-1, n):
                window = basis[i-lookback+1:i+1]
                mean = np.mean(window)
                std = np.std(window)
                
                if std > 0:
                    basis_z[i] = (basis[i] - mean) / std
                else:
                    basis_z[i] = 0
            
            return pl.Series('basis_z_score', basis_z)
        except Exception as e:
            logger.error(f"Error calculating basis Z-score: {e}")
            return None
    
    def calculate_oi_change_rate(self, data: Dict[str, Any]) -> Optional[pl.Series]:
        """计算持仓量变化率（单位成交带来的持仓变化）"""
        try:
            candles = data.get('candles')
            oi_data = data.get('open_interest')
            
            if candles is None or oi_data is None:
                return None
            
            # 对齐时间戳（简化处理）
            oi_values = oi_data['oi'].to_numpy()
            volume = candles['volume'].to_numpy()
            
            n = min(len(oi_values), len(volume))
            if n < 2:
                return None
            
            oi_change_rate = np.full(n, np.nan)
            
            for i in range(1, n):
                oi_change = oi_values[i] - oi_values[i-1]
                if volume[i] > 0:
                    oi_change_rate[i] = oi_change / volume[i]
            
            return pl.Series('oi_change_rate', oi_change_rate[:len(candles)])
        except Exception as e:
            logger.error(f"Error calculating OI change rate: {e}")
            return None
    
    def calculate_net_taker_flow(self, data: Dict[str, Any], 
                                lookback: int = 60) -> Optional[pl.Series]:
        """计算主动买卖净流量"""
        try:
            trades = data.get('trades')
            if trades is None or len(trades) < lookback:
                return None
            
            # 在实际系统中，需要解析trades数据
            # 这里使用简化版本
            candles = data.get('candles')
            if candles is None:
                return None
            
            # 模拟主动买卖流（在实际中需要从tick数据计算）
            close_prices = candles['close'].to_numpy()
            n = len(close_prices)
            
            # 使用价格变化作为主动买卖的代理
            net_flow = np.zeros(n)
            for i in range(1, n):
                if close_prices[i] > close_prices[i-1]:
                    net_flow[i] = 1  # 主动买入
                elif close_prices[i] < close_prices[i-1]:
                    net_flow[i] = -1  # 主动卖出
            
            # 计算滚动平均值
            net_taker_flow = np.full(n, np.nan)
            for i in range(lookback-1, n):
                window = net_flow[i-lookback+1:i+1]
                net_taker_flow[i] = np.mean(window)
            
            return pl.Series('net_taker_flow', net_taker_flow)
        except Exception as e:
            logger.error(f"Error calculating net taker flow: {e}")
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
    
    def calculate_funding_cost_pressure(self, data: Dict[str, Any]) -> Optional[pl.Series]:
        """计算资金费率压力值"""
        try:
            candles = data.get('candles')
            funding_rates = data.get('funding_rates')
            
            if candles is None or funding_rates is None:
                return None
            
            # 对齐数据（简化处理）
            funding_rate_values = funding_rates['rate'].to_numpy()
            
            # 预测持仓时间（简化：假设持仓时间为8小时）
            predicted_hold_time = 8.0
            
            # 计算资金费率压力
            n = min(len(funding_rate_values), len(candles))
            funding_pressure = np.full(n, np.nan)
            
            for i in range(n):
                # 资金费率压力 = 资金费率 × 预测持仓时间
                funding_pressure[i] = funding_rate_values[i] * predicted_hold_time
            
            return pl.Series('funding_cost_pressure', funding_pressure[:len(candles)])
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