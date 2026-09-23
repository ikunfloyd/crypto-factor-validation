"""
数据加载器模块 - 统一数据加载接口
从各种数据源加载监控数据
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger_config import get_logger
from shared_memory.ring_buffer import ArrowRingBuffer, RING_BUFFER_CONFIGS

logger = get_logger(__name__)


class DataLoader:
    """统一数据加载接口"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        
    def load_shared_memory_data(self) -> Dict[str, Any]:
        """从共享内存加载最新数据"""
        data = {
            'ticks': None,
            'candles': None,
            'factors': None,
            'signals': None,
            'status': 'unknown'
        }
        
        try:
            # 尝试从共享内存读取实时tick数据
            try:
                tick_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['fast_tick'], cleanup_on_exit=False)
                latest_ticks = tick_buffer.peek_latest(n=100)
                if latest_ticks is not None and len(latest_ticks) > 0:
                    data['ticks'] = latest_ticks
                    data['status'] = 'active'
                    logger.debug(f"Loaded {len(latest_ticks)} ticks from shared memory")
            except Exception as e:
                logger.debug(f"Could not load ticks from shared memory: {e}")
            
            # 尝试从共享内存读取K线数据
            try:
                candle_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['candle'], cleanup_on_exit=False)
                
                # 添加调试信息：显示ring buffer的索引状态
                write_idx = candle_buffer.write_index
                read_idx = candle_buffer.read_index
                logger.info(f"[DEBUG] Candle buffer state: write_index={write_idx}, read_index={read_idx}")
                
                latest_candles = candle_buffer.peek_latest(n=500)
                if latest_candles is not None and len(latest_candles) > 0:
                    data['candles'] = latest_candles
                    logger.info(f"[DEBUG] Loaded {len(latest_candles)} candles from shared memory")
                    
                    # 添加调试：显示最早和最晚的K线时间（转换为本地时区）
                    if len(latest_candles) > 0:
                        first_ts = pd.to_datetime(latest_candles[0]['timestamp'], unit='s', utc=True).tz_convert('America/Los_Angeles')
                        last_ts = pd.to_datetime(latest_candles[-1]['timestamp'], unit='s', utc=True).tz_convert('America/Los_Angeles')
                        logger.info(f"[DEBUG] Candles time range (Pacific): {first_ts} to {last_ts}")
                else:
                    logger.warning(f"[DEBUG] No candles available in shared memory (write_idx={write_idx}, read_idx={read_idx})")
            except Exception as e:
                logger.error(f"[DEBUG] Could not load candles from shared memory: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 尝试读取因子数据
            factors_latest = self.data_dir / "factors_latest.csv"
            if factors_latest.exists():
                df = pd.read_csv(factors_latest)
                if not df.empty:
                    data['factors'] = df.tail(1).to_dict('records')[0]
                    data['status'] = 'active'
            
            # 尝试读取因子历史
            factors_history = self.data_dir / "factors_history.csv"
            if factors_history.exists():
                df_history = pd.read_csv(factors_history)
                if not df_history.empty:
                    data['factors_history'] = df_history.tail(100)
            
            logger.debug(f"Loaded shared memory data with {len(data)} keys")
            
        except Exception as e:
            logger.error(f"Failed to load shared memory data: {e}")
            data['status'] = 'error'
            
        return data
    
    def load_health_metrics(self) -> Dict[str, Any]:
        """加载健康监控数据"""
        metrics = {
            'health_score': 0,
            'uptime': 0,
            'status': 'unknown',
            'processes': {},
            'errors': []
        }
        
        try:
            # 尝试从健康监控器获取数据
            from utils.health_monitor import HealthMonitor
            
            monitor = HealthMonitor(
                module_name="dashboard",
                max_errors=100,
                error_window=3600
            )
            
            # 简单的健康评分计算
            health_data = monitor.perform_health_check()
            metrics.update(health_data)
            metrics['status'] = 'healthy' if health_data.get('health_score', 0) > 80 else 'degraded'
            
        except Exception as e:
            logger.error(f"Failed to load health metrics: {e}")
            metrics['status'] = 'error'
            
        return metrics
    
    def load_process_status(self) -> Dict[str, Any]:
        """加载进程状态"""
        import psutil
        
        processes = {
            'io_gateway': {'status': 'unknown', 'pid': None, 'cpu': 0, 'memory': 0},
            'quant_core': {'status': 'unknown', 'pid': None, 'cpu': 0, 'memory': 0},
            'execution': {'status': 'unknown', 'pid': None, 'cpu': 0, 'memory': 0}
        }
        
        try:
            # 查找Python进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    info = proc.info
                    if 'python' not in info['name'].lower():
                        continue
                    
                    cmdline = ' '.join(info.get('cmdline', []))
                    
                    # 根据命令行参数识别进程
                    if 'io_gateway' in cmdline:
                        processes['io_gateway'] = {
                            'status': 'running',
                            'pid': info['pid'],
                            'cpu': info.get('cpu_percent', 0),
                            'memory': info.get('memory_info').rss / 1024 / 1024 if info.get('memory_info') else 0
                        }
                    elif 'quant_core' in cmdline:
                        processes['quant_core'] = {
                            'status': 'running',
                            'pid': info['pid'],
                            'cpu': info.get('cpu_percent', 0),
                            'memory': info.get('memory_info').rss / 1024 / 1024 if info.get('memory_info') else 0
                        }
                    elif 'execution' in cmdline:
                        processes['execution'] = {
                            'status': 'running',
                            'pid': info['pid'],
                            'cpu': info.get('cpu_percent', 0),
                            'memory': info.get('memory_info').rss / 1024 / 1024 if info.get('memory_info') else 0
                        }
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load process status: {e}")
            
        return processes
    
    def load_factor_history(self, hours: int = 1) -> pd.DataFrame:
        """加载因子历史数据"""
        try:
            factors_history = self.data_dir / "factors_history.csv"
            if factors_history.exists():
                df = pd.read_csv(factors_history)
                
                # 如果有时间戳列，过滤最近N小时
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    cutoff_time = datetime.now() - timedelta(hours=hours)
                    df = df[df['timestamp'] >= cutoff_time]
                else:
                    # 否则取最后N条记录
                    df = df.tail(hours * 60)  # 假设每分钟一条
                
                return df
            else:
                logger.warning(f"Factor history file not found: {factors_history}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load factor history: {e}")
            return pd.DataFrame()
    
    def load_execution_state(self) -> Dict[str, Any]:
        """加载交易执行状态"""
        state = {
            'position': None,
            'orders': [],
            'trades': [],
            'pnl': 0.0,
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'trade_count': 0,
            'risk_metrics': {},
            'status': 'unknown'
        }
        
        try:
            # 尝试加载执行状态文件
            state_file = self.data_dir / "execution_state.pkl"
            logger.info(f"[DEBUG] Looking for execution state file: {state_file}")
            
            if state_file.exists():
                with open(state_file, 'rb') as f:
                    loaded_state = pickle.load(f)
                    state.update(loaded_state)
                    state['status'] = 'active'
                    
                    # 添加调试信息
                    logger.info(f"[DEBUG] Execution state loaded successfully")
                    logger.info(f"[DEBUG] Position: {state.get('position')}")
                    logger.info(f"[DEBUG] Active orders count: {len(state.get('orders', []))}")
                    logger.info(f"[DEBUG] Trades count: {len(state.get('trades', []))}")
            else:
                logger.warning(f"[DEBUG] Execution state file not found at {state_file}")
                state['status'] = 'no_data'
                
        except Exception as e:
            logger.error(f"[DEBUG] Failed to load execution state: {e}")
            import traceback
            logger.error(traceback.format_exc())
            state['status'] = 'error'
            
        return state
    
    def load_recent_logs(self, level: str = "INFO", max_lines: int = 100) -> List[Dict[str, str]]:
        """加载最近的日志"""
        logs = []
        
        try:
            # 查找日志文件
            log_files = list(self.project_root.glob("*.log"))
            log_files.extend(list((self.project_root / "logs").glob("*.log")) if (self.project_root / "logs").exists() else [])
            
            if not log_files:
                logger.warning("No log files found")
                return logs
            
            # 读取最新的日志文件
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 解析最后max_lines行
            for line in lines[-max_lines:]:
                # 简单解析 (假设格式: timestamp [level] [module] message)
                try:
                    parts = line.strip().split(']', 2)
                    if len(parts) >= 3:
                        timestamp = parts[0].replace('[', '').strip()
                        level_module = parts[1].replace('[', '').strip()
                        message = parts[2].strip()
                        
                        log_entry = {
                            'timestamp': timestamp,
                            'level': level_module.split()[0] if level_module else 'INFO',
                            'message': message
                        }
                        
                        # 过滤日志级别
                        if level == "ALL" or log_entry['level'] == level:
                            logs.append(log_entry)
                            
                except Exception:
                    # 如果解析失败，添加原始行
                    logs.append({
                        'timestamp': 'N/A',
                        'level': 'INFO',
                        'message': line.strip()
                    })
                    
        except Exception as e:
            logger.error(f"Failed to load recent logs: {e}")
            
        return logs
    
    def load_wal_stats(self) -> Dict[str, Any]:
        """加载WAL日志统计"""
        stats = {
            'total_size': 0,
            'file_count': 0,
            'latest_date': None,
            'status': 'unknown'
        }
        
        try:
            wal_dir = self.data_dir / "wal"
            if wal_dir.exists():
                wal_files = list(wal_dir.rglob("*.bin"))
                stats['file_count'] = len(wal_files)
                stats['total_size'] = sum(f.stat().st_size for f in wal_files) / (1024 ** 3)  # GB
                
                if wal_files:
                    latest_file = max(wal_files, key=lambda p: p.stat().st_mtime)
                    stats['latest_date'] = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    stats['status'] = 'active'
                else:
                    stats['status'] = 'no_data'
            else:
                logger.warning(f"WAL directory not found: {wal_dir}")
                stats['status'] = 'not_found'
                
        except Exception as e:
            logger.error(f"Failed to load WAL stats: {e}")
            stats['status'] = 'error'
            
        return stats
    
    def load_checkpoint_info(self) -> Dict[str, Any]:
        """加载检查点信息"""
        info = {
            'latest_checkpoint': None,
            'checkpoint_count': 0,
            'total_size': 0,
            'status': 'unknown'
        }
        
        try:
            checkpoint_dir = self.data_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pkl"))
                info['checkpoint_count'] = len(checkpoint_files)
                info['total_size'] = sum(f.stat().st_size for f in checkpoint_files) / (1024 ** 2)  # MB
                
                # 读取最新检查点文件
                latest_file = checkpoint_dir / "latest_checkpoint.txt"
                if latest_file.exists():
                    with open(latest_file, 'r') as f:
                        info['latest_checkpoint'] = f.read().strip()
                        info['status'] = 'active'
                else:
                    info['status'] = 'no_latest'
            else:
                logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
                info['status'] = 'not_found'
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint info: {e}")
            info['status'] = 'error'
            
        return info
    
    def load_market_data(self) -> Dict[str, Any]:
        """加载实时市场数据"""
        market_data = {
            'price': 0.0,
            'change_24h': 0.0,
            'volume_24h': 0.0,
            'funding_rate': 0.0,
            'high_24h': 0.0,
            'low_24h': 0.0,
            'status': 'unknown'
        }
        
        try:
            # 从共享内存获取最新tick数据
            shm_data = self.load_shared_memory_data()
            ticks = shm_data.get('ticks')
            
            if ticks is not None and len(ticks) > 0:
                latest_tick = ticks[-1]
                market_data['price'] = float(latest_tick['bid1_price'])
                market_data['status'] = 'active'
                
                # 计算24h统计（使用candle数据）
                candles = shm_data.get('candles')
                if candles is not None and len(candles) > 0:
                    # 假设是1分钟K线，取最近1440条（24小时）
                    recent_candles = candles[-min(1440, len(candles)):]
                    if len(recent_candles) > 0:
                        first_close = float(recent_candles[0]['close'])
                        current_price = float(recent_candles[-1]['close'])
                        market_data['change_24h'] = ((current_price - first_close) / first_close) * 100
                        market_data['high_24h'] = float(np.max([c['high'] for c in recent_candles]))
                        market_data['low_24h'] = float(np.min([c['low'] for c in recent_candles]))
                        market_data['volume_24h'] = float(np.sum([c['volume'] for c in recent_candles]))
                
                logger.debug(f"Loaded market data: price={market_data['price']:.5f}")
            else:
                market_data['status'] = 'no_data'
                
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            market_data['status'] = 'error'
            
        return market_data
    
    def load_model_predictions(self) -> Dict[str, Any]:
        """加载模型预测数据"""
        predictions = {
            'Hunter': 0.0,
            'Sniper': 0.0,
            'Defender': 0.0,
            'fusion_score': 0.0,
            'signal': 'FLAT',
            'confidence': 0.0,
            'regime': 'unknown',
            'status': 'unknown'
        }
        
        try:
            # 尝试从因子数据中获取模型预测
            factors_latest = self.data_dir / "factors_latest.csv"
            if factors_latest.exists():
                df = pd.read_csv(factors_latest)
                if not df.empty:
                    latest = df.tail(1).iloc[0]
                    
                    # 提取模型预测（如果有）
                    if 'model_hunter' in latest:
                        predictions['Hunter'] = float(latest['model_hunter'])
                    if 'model_sniper' in latest:
                        predictions['Sniper'] = float(latest['model_sniper'])
                    if 'model_defender' in latest:
                        predictions['Defender'] = float(latest['model_defender'])
                    if 'fusion_score' in latest:
                        predictions['fusion_score'] = float(latest['fusion_score'])
                    if 'signal' in latest:
                        predictions['signal'] = str(latest['signal'])
                    if 'confidence' in latest:
                        predictions['confidence'] = float(latest['confidence'])
                    if 'regime' in latest:
                        predictions['regime'] = str(latest['regime'])
                    
                    predictions['status'] = 'active'
                    logger.debug(f"Loaded model predictions: signal={predictions['signal']}")
            else:
                predictions['status'] = 'no_data'
                
        except Exception as e:
            logger.error(f"Failed to load model predictions: {e}")
            predictions['status'] = 'error'
            
        return predictions


# 创建全局实例
_data_loader = None

def get_data_loader() -> DataLoader:
    """获取数据加载器单例"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
    return _data_loader
