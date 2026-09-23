#!/usr/bin/env python3
"""Quant_Core主进程 V2 - 集成模型推理"""

import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
import psutil
from loguru import logger
import polars as pl 

# M3核心绑定优化
def bind_to_p_cores():
    """绑定到M3的性能核（P-Core）"""
    try:
        p = psutil.Process()
        if not hasattr(p, 'cpu_affinity'):
            logger.debug("CPU affinity not supported on this platform, skipping")
            return
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count >= 8:  # M3通常有4个P-core和4个E-core
            # P-cores通常是前几个核心
            p_cores = list(range(0, cpu_count - 4))
            p.cpu_affinity(p_cores)
            logger.info(f"Quant_Core bound to P-cores: {p_cores}")
    except Exception as e:
        logger.warning(f"Could not bind to P-cores: {e}")

from shared_memory.ring_buffer import ArrowRingBuffer, RING_BUFFER_CONFIGS
from quant_core.data_processor import DataProcessor
from quant_core.factor_engine import FactorEngine
from quant_core.model_inference import ModelInference
from quant_core.gating_network import GatingNetwork

class QuantCoreV2:
    """量化计算核心 V2（集成模型推理）"""
    
    def __init__(self):
        # 绑定到P-Core
        bind_to_p_cores()
        
        # 初始化共享内存缓冲区
        self.tick_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['fast_tick'], cleanup_on_exit=False)
        self.candle_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['candle'], cleanup_on_exit=False)
        
        # 初始化处理器
        self.data_processor = DataProcessor()
        self.factor_engine = FactorEngine()
        
        # 初始化模型推理和门控网络
        self.model_inference = ModelInference()
        self.gating_network = GatingNetwork()
        self.gating_network.load_state()  # 加载保存的状态
        
        # 状态变量
        self.running = False
        self.last_calculation_time = None
        self.calculation_interval = 60  # 1分钟计算一次
        
        # 市场状态检测
        self.current_regime = 'trending'  # 默认状态
        self.regime_history = []
        
        # 结果缓存
        self.current_factors = None
        self.model_predictions = None
        self.trading_signals = []
        
        logger.info("Quant_Core V2 initialized with model inference")
    
    async def start(self):
        """启动量化计算核心"""
        self.running = True
        
        logger.info("Starting Quant_Core V2 with MoE model inference...")
        
        # 设置信号处理
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # 主循环
            while self.running:
                current_time = time.time()
                
                # 检查是否需要计算
                if (self.last_calculation_time is None or 
                    current_time - self.last_calculation_time >= self.calculation_interval):
                    
                    logger.debug("Starting calculation cycle...")
                    
                    # 从共享内存读取数据
                    candle_data = self._read_candle_data()
                    tick_data = self._read_tick_data()
                    
                    if len(candle_data) > 0:
                        # 处理数据
                        processed_data = self._process_data(candle_data, tick_data)
                        
                        # 计算因子
                        factors_df = self._calculate_factors(processed_data)
                        
                        if len(factors_df) > 0:
                            # 缓存结果
                            self.current_factors = factors_df
                            
                            # 检测市场状态
                            self._detect_market_regime(factors_df, processed_data)
                            
                            # 执行模型推理
                            signal_result = self._run_model_inference(factors_df, processed_data)
                            
                            if signal_result and 'signal' in signal_result:
                                # 缓存信号
                                self.trading_signals.append(signal_result)
                                if len(self.trading_signals) > 100:  # 限制历史长度
                                    self.trading_signals.pop(0)
                                
                                # 执行交易（模拟或实际）
                                self._execute_trading_signal(signal_result)
                            
                            # 记录计算时间
                            self.last_calculation_time = current_time
                            
                            # 输出状态
                            self._log_status(signal_result)
                        else:
                            logger.warning("No factors calculated")
                    else:
                        logger.debug("No candle data available for calculation")
                
                # 定期保存门控网络状态
                if int(current_time) % 300 == 0:  # 每5分钟保存一次
                    self.gating_network.save_state()
                    logger.debug("Gating network state saved")
                
                # 等待下一次计算
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Quant_Core error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()
    
    def _read_candle_data(self) -> np.ndarray:
        """从共享内存读取K线数据"""
        try:
            # 读取最近的数据
            data = self.candle_buffer.read_latest(5000)  # 读取最多5000条
            if data is not None:
                logger.debug(f"Read {len(data)} candle records from shared memory")
                return data
            else:
                return np.array([])
        except Exception as e:
            logger.error(f"Error reading candle data: {e}")
            return np.array([])
    
    def _read_tick_data(self) -> np.ndarray:
        """从共享内存读取tick数据"""
        try:
            # 读取最近的数据
            data = self.tick_buffer.read_latest(3600)  # 读取最多3600条
            if data is not None:
                logger.debug(f"Read {len(data)} tick records from shared memory")
                return data
            else:
                return np.array([])
        except Exception as e:
            logger.error(f"Error reading tick data: {e}")
            return np.array([])
    
    def _process_data(self, candle_data: np.ndarray, tick_data: np.ndarray) -> Dict[str, Any]:
        """处理原始数据"""
        try:
            processed_data = {}
            
            # 处理K线数据
            if len(candle_data) > 0:
                candle_df = self.data_processor.process_candle_data(candle_data)
                processed_data['candles'] = candle_df
            
            # 处理tick数据
            if len(tick_data) > 0:
                tick_df = self.data_processor.process_tick_data(tick_data)
                processed_data['ticks'] = tick_df
            
            # 对齐数据
            processed_data = self.data_processor.align_dataframes(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return {}
    
    def _calculate_factors(self, processed_data: Dict[str, Any]) -> Optional[pl.DataFrame]:
        """计算所有因子"""
        try:
            if not processed_data or 'candles' not in processed_data:
                logger.warning("No processed data available for factor calculation")
                return None
            
            # 计算因子
            factors_df = self.factor_engine.calculate_all_factors(processed_data)
            
            if len(factors_df) > 0:
                logger.info(f"Calculated {len(factors_df.columns)-1} factors")
                return factors_df
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calculating factors: {e}")
            return None
    
    def _detect_market_regime(self, factors_df: pl.DataFrame, processed_data: Dict[str, Any]):
        """检测市场状态"""
        try:
            if factors_df is None or len(factors_df) < 20:
                return
            
            # 获取最新数据
            latest = factors_df.tail(1)
            
            # 简单规则检测（实际应该使用更复杂的算法）
            # 1. 波动率状态
            atr_pct = latest['atr_pct_14'][0] if 'atr_pct_14' in latest.columns else 0.01
            volatility_regime = 'high_vol' if atr_pct > 0.02 else ('low_vol' if atr_pct < 0.005 else 'normal')
            
            # 2. 趋势状态（简化）
            if 'rsi_14' in latest.columns:
                rsi = latest['rsi_14'][0]
                if rsi > 60:
                    trend = 'up'
                elif rsi < 40:
                    trend = 'down'
                else:
                    trend = 'ranging'
            else:
                trend = 'unknown'
            
            # 3. 确定主状态
            if volatility_regime == 'high_vol':
                self.current_regime = 'high_vol'
            elif trend in ['up', 'down']:
                self.current_regime = 'trending'
            else:
                self.current_regime = 'ranging'
            
            # 记录历史
            self.regime_history.append({
                'timestamp': time.time(),
                'regime': self.current_regime,
                'volatility': atr_pct,
                'trend': trend
            })
            
            if len(self.regime_history) > 100:
                self.regime_history.pop(0)
            
            logger.debug(f"Market regime detected: {self.current_regime} (vol={atr_pct:.3%}, trend={trend})")
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
    
    def _run_model_inference(self, factors_df: pl.DataFrame,
                             processed_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """运行模型推理"""
        try:
            if factors_df is None or len(factors_df) == 0:
                return None

            # 准备特征（传入已处理 K 线，取回 K 线派生特征，避免训练/推理错位）
            candles_df = processed_data.get('candles') if processed_data else None
            features = self.model_inference.prepare_features(factors_df, candles_df)
            
            if features is None:
                logger.warning("Could not prepare features for inference")
                return None
            
            # 使用门控网络获取当前权重
            current_weights = self.gating_network.get_weights(self.current_regime)
            
            # 更新模型推理的门控权重
            self.model_inference.gating_weights[self.current_regime] = current_weights
            
            # 进行MoE预测
            result = self.model_inference.predict_with_moe(features, self.current_regime)
            
            # 缓存预测结果
            self.model_predictions = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in model inference: {e}")
            return None
    
    def _execute_trading_signal(self, signal_result: Dict[str, Any]):
        """执行交易信号（模拟版本）"""
        try:
            if 'signal' not in signal_result:
                return
            
            signal = signal_result['signal']
            
            # 这里应该是实际的订单执行逻辑
            # 现在只是模拟记录
            
            if signal['direction'] != 'NEUTRAL' and signal['confidence'] > 0.3:
                logger.info(f"📈 Trading Signal: {signal['direction']} | "
                          f"Confidence: {signal['confidence']:.1%} | "
                          f"Position: {signal['position_size']:.1%} | "
                          f"Regime: {self.current_regime}")
            
            # 在实际系统中，这里会调用Execution_Guard的订单接口
            
        except Exception as e:
            logger.error(f"Error executing trading signal: {e}")
    
    def _update_gating_network(self, signal_result: Dict[str, Any], actual_return: float = None):
        """根据实际表现更新门控网络"""
        try:
            if 'predictions' not in signal_result or 'regime' not in signal_result:
                return
            
            regime = signal_result['regime']
            predictions = signal_result['predictions']
            
            # 如果没有实际收益数据，使用信号置信度作为代理
            if actual_return is None:
                actual_return = signal_result.get('final_score', 0) * 0.01  # 简单转换
            
            # 评估各个专家的表现
            for expert_name, expert_pred in predictions.items():
                # 这里需要根据实际交易结果评估专家表现
                # 现在使用简化版本
                if expert_name == 'defender':
                    # defender预测的是风险，实际风险越低表现越好
                    performance = 1.0 - abs(expert_pred)  # 风险预测越接近0越好
                else:
                    # hunter和sniper预测的是上涨概率
                    performance = expert_pred  # 预测概率作为表现
                
                # 更新门控网络
                self.gating_network.update_performance(regime, expert_name, performance)
            
            # 更新权重
            self.gating_network.update_weights(regime)
            
            # 定期衰减
            if int(time.time()) % 3600 == 0:  # 每小时衰减一次
                self.gating_network.decay_to_base()
            
            logger.debug(f"Gating network updated for {regime}")
            
        except Exception as e:
            logger.error(f"Error updating gating network: {e}")
    
    def _log_status(self, signal_result: Dict[str, Any] = None):
        """记录状态"""
        try:
            if signal_result and 'signal' in signal_result:
                signal = signal_result['signal']
                
                status_msg = f"Regime: {self.current_regime} | "
                
                if signal['direction'] != 'NEUTRAL':
                    status_msg += f"Signal: {signal['direction']} ({signal['confidence']:.1%}) | "
                    status_msg += f"Position: {signal['position_size']:.1%} | "
                
                if 'predictions' in signal_result:
                    preds = signal_result['predictions']
                    status_msg += f"Experts: H={preds.get('hunter', 0):.2f}, "
                    status_msg += f"S={preds.get('sniper', 0):.2f}, "
                    status_msg += f"D={preds.get('defender', 0):.2f}"
                
                logger.info(status_msg)
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    async def stop(self):
        """停止量化计算核心"""
        self.running = False
        
        # 保存门控网络状态
        self.gating_network.save_state()
        
        # 清理共享内存
        self.tick_buffer.cleanup()
        self.candle_buffer.cleanup()
        
        logger.info("Quant_Core V2 stopped")

async def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>Quant_Core_V2</cyan> | <level>{message}</level>",
        level="INFO"
    )
    
    # 创建并运行量化核心
    quant_core = QuantCoreV2()
    await quant_core.start()

if __name__ == "__main__":
    # 设置事件循环策略为uvloop（如果可用）
    try:
        import uvloop
        uvloop.install()
        logger.info("Using uvloop for better async performance")
    except ImportError:
        logger.warning("uvloop not installed, using default asyncio")
    
    # 运行主程序
    asyncio.run(main())