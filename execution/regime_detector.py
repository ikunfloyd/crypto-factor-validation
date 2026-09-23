import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger

# 尝试导入TA-Lib，如果失败则使用降级实现
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib is available for technical analysis")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available, using fallback calculations")

class RegimeDetector:
    """市场状态检测器"""
    
    def __init__(self):
        # 状态阈值
        self.thresholds = {
            'high_vol_atr_pct': 0.02,  # ATR百分比 > 2%为高波动
            'low_vol_atr_pct': 0.005,  # ATR百分比 < 0.5%为低波动
            'trending_adx': 25,  # ADX > 25为趋势市
            'ranging_adx': 20,   # ADX < 20为震荡市
            'liquid_spread_pct': 0.0005,  # 价差 < 0.05%为高流动性
            'illiquid_spread_pct': 0.001,  # 价差 > 0.1%为低流动性
        }
        
        # 状态历史
        self.regime_history = []
        self.max_history_size = 100
        
        logger.info(f"RegimeDetector initialized (TA-Lib: {'available' if TALIB_AVAILABLE else 'fallback'})")
    
    def _calculate_atr_fallback(self, high: np.ndarray, low: np.ndarray,
                                close: np.ndarray, period: int = 14) -> np.ndarray:
        """ATR的降级计算（无需TA-Lib）"""
        n = len(close)
        if n < period:
            return np.zeros(n)
        
        # 计算True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # 计算ATR (简单移动平均)
        atr = np.zeros(n)
        atr[period-1] = np.mean(tr[:period])
        
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def _calculate_ema_fallback(self, data: np.ndarray, period: int) -> np.ndarray:
        """EMA的降级计算（无需TA-Lib）"""
        n = len(data)
        if n < period:
            return np.full(n, data[0] if n > 0 else 0.0)
        
        alpha = 2 / (period + 1)
        ema = np.zeros(n)
        ema[0] = data[0]
        
        for i in range(1, n):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_adx_fallback(self, high: np.ndarray, low: np.ndarray,
                                close: np.ndarray, period: int = 14) -> float:
        """ADX的简化降级计算（无需TA-Lib）"""
        # 这是一个简化版本，返回基于价格波动的估算值
        if len(close) < period:
            return 20.0  # 默认中性值
        
        # 计算价格变化
        price_changes = np.diff(close[-period:])
        positive_moves = np.sum(price_changes[price_changes > 0])
        negative_moves = abs(np.sum(price_changes[price_changes < 0]))
        
        # 计算方向性
        total_moves = positive_moves + negative_moves
        if total_moves == 0:
            return 20.0
        
        directional_movement = abs(positive_moves - negative_moves) / total_moves
        
        # 转换为类似ADX的值（0-100）
        adx_estimate = directional_movement * 50 + 10  # 缩放到10-60范围
        
        return float(adx_estimate)
    
    def detect(self, candle_data: np.ndarray, tick_data: np.ndarray) -> str:
        """
        检测市场状态
        
        Returns:
            str: 市场状态，可选值: 'trending_up', 'trending_down', 'ranging', 
                 'high_vol', 'low_vol', 'liquid', 'illiquid', 'normal'
        """
        try:
            if len(candle_data) < 20 or len(tick_data) < 10:
                logger.warning("Insufficient data for regime detection")
                return 'unknown'
            
            # 提取数据
            close_prices = candle_data['close'][-100:]  # 使用最近100个点
            high_prices = candle_data['high'][-100:]
            low_prices = candle_data['low'][-100:]
            
            # 1. 检测波动率状态
            vol_regime = self._detect_volatility_regime(close_prices, high_prices, low_prices)
            
            # 2. 检测趋势状态
            trend_regime = self._detect_trend_regime(close_prices, high_prices, low_prices)
            
            # 3. 检测流动性状态
            liq_regime = self._detect_liquidity_regime(tick_data)
            
            # 4. 综合判断
            final_regime = self._combine_regimes(vol_regime, trend_regime, liq_regime)
            
            # 记录历史
            self.regime_history.append({
                'timestamp': datetime.now(),
                'vol_regime': vol_regime,
                'trend_regime': trend_regime,
                'liq_regime': liq_regime,
                'final_regime': final_regime
            })
            
            # 限制历史大小
            if len(self.regime_history) > self.max_history_size:
                self.regime_history.pop(0)
            
            logger.debug(f"Regime detected: {final_regime} "
                        f"(vol: {vol_regime}, trend: {trend_regime}, liq: {liq_regime})")
            
            return final_regime
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return 'unknown'
    
    def _detect_volatility_regime(self, close_prices: np.ndarray,
                                 high_prices: np.ndarray,
                                 low_prices: np.ndarray) -> str:
        """检测波动率状态"""
        try:
            # 计算ATR
            if len(close_prices) >= 14:
                # 使用talib或降级计算ATR
                if TALIB_AVAILABLE:
                    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                    atr_pct = atr[-1] / close_prices[-1] if close_prices[-1] > 0 else 0
                else:
                    atr = self._calculate_atr_fallback(high_prices, low_prices, close_prices, 14)
                    atr_pct = atr[-1] / close_prices[-1] if close_prices[-1] > 0 and len(atr) > 0 else 0
                
                # 判断波动率状态
                if atr_pct > self.thresholds['high_vol_atr_pct']:
                    return 'high_vol'
                elif atr_pct < self.thresholds['low_vol_atr_pct']:
                    return 'low_vol'
                else:
                    return 'normal_vol'
            else:
                return 'unknown_vol'
                
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return 'unknown_vol'
    
    def _detect_trend_regime(self, close_prices: np.ndarray,
                            high_prices: np.ndarray,
                            low_prices: np.ndarray) -> str:
        """检测趋势状态"""
        try:
            if len(close_prices) < 20:
                return 'unknown_trend'
            
            # 计算ADX
            if TALIB_AVAILABLE:
                adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
                current_adx = adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 20.0
            else:
                current_adx = self._calculate_adx_fallback(high_prices, low_prices, close_prices, 14)
            
            # 计算EMA斜率
            if TALIB_AVAILABLE:
                ema_20 = talib.EMA(close_prices, timeperiod=20)
            else:
                ema_20 = self._calculate_ema_fallback(close_prices, 20)
            
            if len(ema_20) >= 5:
                slope = (ema_20[-1] - ema_20[-5]) / 5
                slope_pct = slope / close_prices[-1] if close_prices[-1] > 0 else 0
            else:
                slope_pct = 0
            
            # 判断趋势状态
            if current_adx > self.thresholds['trending_adx']:
                if slope_pct > 0.001:  # 正向斜率
                    return 'trending_up'
                elif slope_pct < -0.001:  # 负向斜率
                    return 'trending_down'
                else:
                    return 'trending_sideways'
            elif current_adx < self.thresholds['ranging_adx']:
                return 'ranging'
            else:
                return 'normal_trend'
                
        except Exception as e:
            logger.error(f"Error detecting trend regime: {e}")
            return 'unknown_trend'
    
    def _detect_liquidity_regime(self, tick_data: np.ndarray) -> str:
        """检测流动性状态"""
        try:
            if len(tick_data) < 5:
                return 'unknown_liq'
            
            # 计算平均价差
            spreads = []
            depths = []
            
            for tick in tick_data[-10:]:  # 使用最近10个tick
                if tick['ask1_price'] > 0 and tick['bid1_price'] > 0:
                    spread = tick['ask1_price'] - tick['bid1_price']
                    spread_pct = spread / ((tick['ask1_price'] + tick['bid1_price']) / 2)
                    spreads.append(spread_pct)
                    
                    # 计算深度
                    depth = tick['ask1_quantity'] + tick['bid1_quantity']
                    depths.append(depth)
            
            if spreads:
                avg_spread = np.mean(spreads)
                avg_depth = np.mean(depths) if depths else 0
                
                # 判断流动性状态
                if avg_spread < self.thresholds['liquid_spread_pct'] and avg_depth > 1000:
                    return 'liquid'
                elif avg_spread > self.thresholds['illiquid_spread_pct']:
                    return 'illiquid'
                else:
                    return 'normal_liq'
            else:
                return 'unknown_liq'
                
        except Exception as e:
            logger.error(f"Error detecting liquidity regime: {e}")
            return 'unknown_liq'
    
    def _combine_regimes(self, vol_regime: str, trend_regime: str, liq_regime: str) -> str:
        """综合各个状态判断最终市场状态"""
        try:
            # 流动性差时，不管其他状态如何，都视为高风险
            if liq_regime == 'illiquid':
                return 'high_risk'
            
            # 高波动时
            if vol_regime == 'high_vol':
                if 'trending' in trend_regime:
                    return 'trending_high_vol'
                else:
                    return 'high_vol'
            
            # 低波动时
            if vol_regime == 'low_vol':
                if trend_regime == 'ranging':
                    return 'low_vol_ranging'
                else:
                    return 'low_vol'
            
            # 正常波动时，主要看趋势
            if 'trending_up' in trend_regime:
                return 'trending_up'
            elif 'trending_down' in trend_regime:
                return 'trending_down'
            elif trend_regime == 'ranging':
                return 'ranging'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error combining regimes: {e}")
            return 'unknown'
    
    def get_regime_history(self, n: int = 10) -> list:
        """获取最近的市场状态历史"""
        return self.regime_history[-n:] if self.regime_history else []
    
    def get_current_regime_stats(self) -> Dict[str, Any]:
        """获取当前状态统计"""
        if not self.regime_history:
            return {}
        
        current = self.regime_history[-1]
        return {
            'vol_regime': current['vol_regime'],
            'trend_regime': current['trend_regime'],
            'liq_regime': current['liq_regime'],
            'final_regime': current['final_regime'],
            'timestamp': current['timestamp']
        }