#!/usr/bin/env python3
"""Execution_Guard主进程 - 绑定到M3 P-Core性能核"""

import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import psutil
from loguru import logger

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
            logger.info(f"Execution_Guard bound to P-cores: {p_cores}")
    except Exception as e:
        logger.warning(f"Could not bind to P-cores: {e}")

from shared_memory.ring_buffer import ArrowRingBuffer, RING_BUFFER_CONFIGS
from execution.regime_detector import RegimeDetector
from execution.signal_fusion import SignalFusion
from execution.order_router import OrderRouter
from execution.risk_manager import RiskManager

class ExecutionGuard:
    """执行守卫 - 负责状态识别、信号融合、订单路由和风控"""
    
    def __init__(self):
        # 绑定到P-Core
        bind_to_p_cores()
        
        # 初始化共享内存缓冲区
        self.tick_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['fast_tick'], cleanup_on_exit=False)
        self.candle_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['candle'], cleanup_on_exit=False)
        
        # 初始化各个模块
        self.regime_detector = RegimeDetector()
        self.signal_fusion = SignalFusion()
        self.order_router = OrderRouter()
        self.risk_manager = RiskManager()
        
        # 状态变量
        self.running = False
        self.last_execution_time = None
        self.execution_interval = 60  # 1分钟执行一次
        
        # 当前状态
        self.current_regime = None
        self.current_signal = None
        self.current_position = 0.0  # 当前仓位，0-1表示百分比
        self.pnl_history = []  # 盈亏历史
        
        # 性能监控
        self.execution_stats = {
            'total_signals': 0,
            'executed_orders': 0,
            'rejected_orders': 0,
            'total_pnl': 0.0
        }
        
        logger.info("Execution_Guard initialized (M3 P-Core optimized)")
    
    async def start(self):
        """启动执行守卫"""
        self.running = True
        
        logger.info("Starting Execution_Guard...")
        
        # 设置信号处理
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # 加载保存的状态
            self._load_state()
            
            # 主循环
            while self.running:
                current_time = time.time()
                
                # 检查是否需要执行
                if (self.last_execution_time is None or 
                    current_time - self.last_execution_time >= self.execution_interval):
                    
                    logger.debug("Starting execution cycle...")
                    
                    # 从共享内存读取数据
                    candle_data = self._read_candle_data()
                    tick_data = self._read_tick_data()
                    
                    if len(candle_data) > 0 and len(tick_data) > 0:
                        # 检测市场状态
                        regime = self._detect_regime(candle_data, tick_data)
                        
                        # 从共享内存读取信号（如果有）
                        signal_data = self._read_signal_data()
                        
                        if signal_data:
                            # 融合信号（考虑市场状态和风险）
                            fused_signal = self._fuse_signal(signal_data, regime)
                            
                            # 风险检查
                            if self._risk_check(fused_signal):
                                # 执行订单
                                execution_result = self._execute_order(fused_signal, tick_data)
                                
                                # 更新状态
                                self._update_state(execution_result)
                            else:
                                logger.warning("Signal rejected by risk manager")
                        
                        # 记录执行时间
                        self.last_execution_time = current_time
                        
                        # 输出状态
                        self._log_status()
                    else:
                        logger.debug("Insufficient data for execution")
                
                # 定期保存状态
                if int(current_time) % 300 == 0:  # 每5分钟保存一次
                    self._save_state()
                    logger.debug("Execution state saved")
                
                # 等待下一次执行
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Execution_Guard error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()
    
    def _read_candle_data(self) -> np.ndarray:
        """从共享内存读取K线数据"""
        try:
            # 读取最近的数据
            data = self.candle_buffer.read_latest(100)  # 读取最近100条
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
            data = self.tick_buffer.read_latest(100)  # 读取最近100条
            if data is not None:
                logger.debug(f"Read {len(data)} tick records from shared memory")
                return data
            else:
                return np.array([])
        except Exception as e:
            logger.error(f"Error reading tick data: {e}")
            return np.array([])
    
    def _read_signal_data(self) -> Optional[Dict[str, Any]]:
        """从共享内存读取信号数据"""
        try:
            # 在实际系统中，这里应该从共享内存读取Quant_Core生成的信号
            # 现在使用模拟信号
            if np.random.random() > 0.7:  # 30%概率生成信号
                signal = {
                    'timestamp': time.time(),
                    'direction': 'LONG' if np.random.random() > 0.5 else 'SHORT',
                    'confidence': np.random.uniform(0.3, 0.9),
                    'position_size': np.random.uniform(0.1, 0.5),
                    'regime': np.random.choice(['trending', 'ranging', 'high_vol']),
                    'score': np.random.uniform(-1, 1)
                }
                return signal
            return None
        except Exception as e:
            logger.error(f"Error reading signal data: {e}")
            return None
    
    def _detect_regime(self, candle_data: np.ndarray, tick_data: np.ndarray) -> str:
        """检测市场状态"""
        try:
            # 使用regime_detector检测状态
            regime = self.regime_detector.detect(candle_data, tick_data)
            self.current_regime = regime
            logger.debug(f"Market regime detected: {regime}")
            return regime
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return 'unknown'
    
    def _fuse_signal(self, signal_data: Dict[str, Any], regime: str) -> Dict[str, Any]:
        """融合信号（考虑市场状态、风险、当前仓位等）"""
        try:
            # 使用signal_fusion模块融合信号
            fused_signal = self.signal_fusion.fuse(
                signal_data, 
                regime, 
                self.current_position,
                self.risk_manager.get_risk_level()
            )
            
            logger.debug(f"Signal fused: {fused_signal.get('direction', 'NEUTRAL')} "
                        f"(confidence: {fused_signal.get('confidence', 0):.1%})")
            
            return fused_signal
        except Exception as e:
            logger.error(f"Error fusing signal: {e}")
            return {'direction': 'NEUTRAL', 'confidence': 0}
    
    def _risk_check(self, signal: Dict[str, Any]) -> bool:
        """风险检查"""
        try:
            # 使用risk_manager进行风险检查
            risk_ok = self.risk_manager.check(
                signal, 
                self.current_position,
                self.pnl_history
            )
            
            if not risk_ok:
                reason = self.risk_manager.get_last_rejection_reason()
                logger.warning(f"Risk check failed: {reason}")
            
            return risk_ok
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False
    
    def _execute_order(self, signal: Dict[str, Any], tick_data: np.ndarray) -> Dict[str, Any]:
        """执行订单"""
        try:
            # 使用order_router执行订单
            if signal['direction'] != 'NEUTRAL' and signal['confidence'] > 0.3:
                # 计算订单详情
                order_details = self._prepare_order(signal, tick_data)
                
                # 执行订单（模拟或实际）
                if order_details['size_usdt'] > 10000:  # 大额订单使用TWAP
                    execution_result = self.order_router.execute_twap(order_details, tick_data)
                else:
                    execution_result = self.order_router.execute_market(order_details, tick_data)
                
                # 更新统计数据
                self.execution_stats['executed_orders'] += 1
                self.execution_stats['total_signals'] += 1
                
                logger.info(f"Order executed: {signal['direction']} "
                          f"${order_details['size_usdt']:.0f} "
                          f"@ ~${order_details['estimated_price']:.4f}")
                
                return execution_result
            else:
                logger.debug("No order to execute (neutral or low confidence)")
                return {'status': 'no_action'}
                
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _prepare_order(self, signal: Dict[str, Any], tick_data: np.ndarray) -> Dict[str, Any]:
        """准备订单详情"""
        try:
            # 获取最新价格
            latest_tick = tick_data[-1]
            current_price = (latest_tick['ask1_price'] + latest_tick['bid1_price']) / 2
            
            # 计算订单大小
            position_size = signal.get('position_size', 0.1)
            max_position = 10000  # 最大仓位$10,000（示例）
            size_usdt = min(max_position * position_size, 50000)  # 限制单笔订单
            
            # 计算滑点估计
            spread = latest_tick['ask1_price'] - latest_tick['bid1_price']
            spread_pct = spread / current_price
            
            # 准备订单详情
            order_details = {
                'direction': signal['direction'],
                'symbol': 'DOGE-USDT',
                'size_usdt': size_usdt,
                'size_coin': size_usdt / current_price if current_price > 0 else 0,
                'current_price': current_price,
                'estimated_price': current_price * (1 + spread_pct/2 if signal['direction'] == 'LONG' else 1 - spread_pct/2),
                'spread_pct': spread_pct,
                'confidence': signal['confidence'],
                'regime': signal.get('regime', 'unknown'),
                'timestamp': time.time()
            }
            
            return order_details
            
        except Exception as e:
            logger.error(f"Error preparing order: {e}")
            return {'direction': 'NEUTRAL', 'size_usdt': 0}
    
    def _update_state(self, execution_result: Dict[str, Any]):
        """更新执行状态"""
        try:
            if execution_result.get('status') == 'executed':
                # 更新仓位
                if execution_result['direction'] == 'LONG':
                    self.current_position += execution_result.get('filled_pct', 0)
                elif execution_result['direction'] == 'SHORT':
                    self.current_position -= execution_result.get('filled_pct', 0)
                
                # 限制仓位在-1到1之间
                self.current_position = max(-1.0, min(1.0, self.current_position))
                
                # 记录PNL（模拟）
                if 'estimated_pnl' in execution_result:
                    self.pnl_history.append({
                        'timestamp': time.time(),
                        'pnl': execution_result['estimated_pnl']
                    })
                    self.execution_stats['total_pnl'] += execution_result['estimated_pnl']
                
                # 限制历史记录长度
                if len(self.pnl_history) > 1000:
                    self.pnl_history.pop(0)
            
        except Exception as e:
            logger.error(f"Error updating state: {e}")
    
    def _log_status(self):
        """记录状态"""
        try:
            status_msg = f"Status: Position={self.current_position:.1%} | "
            status_msg += f"Regime={self.current_regime} | "
            status_msg += f"Orders={self.execution_stats['executed_orders']} | "
            status_msg += f"PNL=${self.execution_stats['total_pnl']:.2f}"
            
            logger.info(status_msg)
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    def _save_state(self):
        """保存状态到文件"""
        try:
            import pickle
            state = {
                'current_position': self.current_position,
                'pnl_history': self.pnl_history[-100:],  # 只保存最近100条
                'execution_stats': self.execution_stats,
                'saved_at': time.time()
            }
            
            with open('./data/execution_state.pkl', 'wb') as f:
                pickle.dump(state, f)
                
            logger.debug("Execution state saved")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        """从文件加载状态"""
        try:
            import pickle
            import os
            
            if os.path.exists('./data/execution_state.pkl'):
                with open('./data/execution_state.pkl', 'rb') as f:
                    state = pickle.load(f)
                
                self.current_position = state.get('current_position', 0)
                self.pnl_history = state.get('pnl_history', [])
                self.execution_stats = state.get('execution_stats', self.execution_stats)
                
                logger.info("Execution state loaded from file")
            else:
                logger.info("No saved execution state found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    async def stop(self):
        """停止执行守卫"""
        self.running = False
        
        # 保存状态
        self._save_state()
        
        # 清理共享内存
        self.tick_buffer.cleanup()
        self.candle_buffer.cleanup()
        
        logger.info("Execution_Guard stopped")

async def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>Execution_Guard</cyan> | <level>{message}</level>",
        level="INFO"
    )
    
    # 创建并运行执行守卫
    guard = ExecutionGuard()
    await guard.start()

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