#!/usr/bin/env python3
"""系统健康监控模块"""

import time
import psutil
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

@dataclass
class HealthMetrics:
    """健康指标数据类"""
    timestamp: float
    websocket_latency: Optional[float] = None
    websocket_connected: bool = False
    shared_memory_ok: bool = False
    model_loaded: bool = False
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    process_uptime: float = 0.0
    data_flow_rate: float = 0.0  # 数据流速率（条/秒）
    error_count: int = 0
    warning_count: int = 0

class HealthMonitor:
    """系统健康监控器"""
    
    def __init__(self, check_interval: int = 60):
        """
        初始化健康监控器
        
        Args:
            check_interval: 健康检查间隔（秒）
        """
        self.check_interval = check_interval
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        
        # 历史指标
        self.metrics_history: List[HealthMetrics] = []
        self.max_history_size = 1000
        
        # 阈值配置
        self.thresholds = {
            'max_latency_ms': 1000,  # 最大延迟（毫秒）
            'max_cpu_percent': 80,  # CPU使用率
            'max_memory_percent': 80,  # 内存使用率
            'max_disk_percent': 90,  # 磁盘使用率
            'min_data_flow_rate': 0.1,  # 最小数据流速率（条/秒）
        }
        
        # 状态跟踪
        self.last_data_timestamp = time.time()
        self.data_count = 0
        self.error_count = 0
        self.warning_count = 0
        
        logger.info("HealthMonitor initialized")
    
    def collect_metrics(self) -> HealthMetrics:
        """收集当前健康指标"""
        try:
            metrics = HealthMetrics(
                timestamp=time.time(),
                cpu_percent=self.process.cpu_percent(interval=0.1),
                memory_percent=self.process.memory_percent(),
                process_uptime=time.time() - self.start_time,
                error_count=self.error_count,
                warning_count=self.warning_count
            )
            
            # 磁盘使用率
            try:
                disk = psutil.disk_usage('/')
                metrics.disk_usage_percent = disk.percent
            except Exception as e:
                logger.warning(f"Failed to get disk usage: {e}")
            
            # 数据流速率
            current_time = time.time()
            time_delta = current_time - self.last_data_timestamp
            if time_delta > 0:
                metrics.data_flow_rate = self.data_count / time_delta
            
            # 重置数据计数
            self.data_count = 0
            self.last_data_timestamp = current_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return HealthMetrics(timestamp=time.time())
    
    def check_websocket_health(self, ws_client=None) -> bool:
        """
        检查WebSocket连接健康状态
        
        Args:
            ws_client: WebSocket客户端实例
            
        Returns:
            连接是否健康
        """
        try:
            if ws_client is None:
                return False
            
            # 检查连接状态
            if not hasattr(ws_client, 'running') or not ws_client.running:
                logger.warning("WebSocket not running")
                return False
            
            # 检查心跳时间
            if hasattr(ws_client, 'last_heartbeat_public'):
                latency = time.time() - ws_client.last_heartbeat_public
                if latency > self.thresholds['max_latency_ms'] / 1000:
                    logger.warning(f"WebSocket latency too high: {latency:.2f}s")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking WebSocket health: {e}")
            return False
    
    def check_shared_memory_status(self) -> bool:
        """
        检查共享内存状态
        
        Returns:
            共享内存是否正常
        """
        try:
            from shared_memory.ring_buffer import _shared_memory_objects
            
            if not _shared_memory_objects:
                logger.debug("No shared memory objects registered")
                return True  # 没有共享内存对象也算正常
            
            # 尝试访问共享内存
            try:
                from multiprocessing import shared_memory
                for shm_name in _shared_memory_objects[:1]:  # 检查第一个即可
                    shm = shared_memory.SharedMemory(name=shm_name)
                    shm.close()
                return True
            except FileNotFoundError:
                logger.warning("Shared memory objects not found")
                return False
            
        except Exception as e:
            logger.error(f"Error checking shared memory: {e}")
            return False
    
    def check_model_performance(self, model_inference=None) -> bool:
        """
        检查模型性能
        
        Args:
            model_inference: 模型推理实例
            
        Returns:
            模型是否正常
        """
        try:
            if model_inference is None:
                return False
            
            # 检查模型是否加载
            if not hasattr(model_inference, 'models'):
                return False
            
            if len(model_inference.models) == 0:
                if hasattr(model_inference, 'fallback_mode') and model_inference.fallback_mode:
                    logger.debug("Model in fallback mode")
                    return True  # 降级模式也算正常
                else:
                    logger.warning("No models loaded and fallback disabled")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return False
    
    def check_data_freshness(self, max_age_seconds: float = 60) -> bool:
        """
        检查数据新鲜度
        
        Args:
            max_age_seconds: 最大数据年龄（秒）
            
        Returns:
            数据是否新鲜
        """
        try:
            age = time.time() - self.last_data_timestamp
            if age > max_age_seconds:
                logger.warning(f"Data is stale: {age:.1f}s old")
                return False
            return True
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return False
    
    def perform_health_check(self, 
                            ws_client=None,
                            model_inference=None) -> Dict[str, Any]:
        """
        执行完整的健康检查
        
        Args:
            ws_client: WebSocket客户端实例
            model_inference: 模型推理实例
            
        Returns:
            健康检查结果
        """
        try:
            metrics = self.collect_metrics()
            
            # 各项检查
            checks = {
                'websocket': self.check_websocket_health(ws_client),
                'shared_memory': self.check_shared_memory_status(),
                'model': self.check_model_performance(model_inference),
                'data_freshness': self.check_data_freshness(),
                'cpu_ok': metrics.cpu_percent < self.thresholds['max_cpu_percent'],
                'memory_ok': metrics.memory_percent < self.thresholds['max_memory_percent'],
                'disk_ok': metrics.disk_usage_percent < self.thresholds['max_disk_percent'],
            }
            
            # 计算总体健康状态
            all_ok = all(checks.values())
            health_score = sum(checks.values()) / len(checks) * 100
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy' if all_ok else 'degraded',
                'health_score': health_score,
                'checks': checks,
                'metrics': asdict(metrics),
                'thresholds': self.thresholds
            }
            
            # 记录历史
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            # 日志记录
            if all_ok:
                logger.info(f"Health check passed: score={health_score:.1f}%")
            else:
                failed_checks = [k for k, v in checks.items() if not v]
                logger.warning(f"Health check failed: score={health_score:.1f}%, failed={failed_checks}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def get_health_summary(self, hours: int = 1) -> Dict[str, Any]:
        """
        获取健康状态摘要
        
        Args:
            hours: 统计时间范围（小时）
            
        Returns:
            健康状态摘要
        """
        try:
            if not self.metrics_history:
                return {'message': 'No metrics available'}
            
            # 过滤指定时间范围内的指标
            cutoff_time = time.time() - hours * 3600
            recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {'message': f'No metrics in the last {hours} hour(s)'}
            
            # 计算统计数据
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            
            summary = {
                'time_range_hours': hours,
                'data_points': len(recent_metrics),
                'cpu': {
                    'avg': sum(cpu_values) / len(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values)
                },
                'memory': {
                    'avg': sum(memory_values) / len(memory_values),
                    'max': max(memory_values),
                    'min': min(memory_values)
                },
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'total_errors': self.error_count,
                'total_warnings': self.warning_count
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return {'error': str(e)}
    
    def record_data_event(self):
        """记录数据事件（用于计算数据流速率）"""
        self.data_count += 1
    
    def record_error(self):
        """记录错误事件"""
        self.error_count += 1
    
    def record_warning(self):
        """记录警告事件"""
        self.warning_count += 1
    
    def reset_counters(self):
        """重置计数器"""
        self.error_count = 0
        self.warning_count = 0
        self.data_count = 0
        logger.info("Health monitor counters reset")
    
    def export_metrics(self, filepath: str):
        """
        导出指标到文件
        
        Args:
            filepath: 导出文件路径
        """
        try:
            import json
            
            data = {
                'export_time': datetime.now().isoformat(),
                'metrics': [asdict(m) for m in self.metrics_history]
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

# 全局健康监控器实例
_health_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """获取全局健康监控器实例"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor

def set_health_monitor(monitor: HealthMonitor):
    """设置全局健康监控器实例"""
    global _health_monitor
    _health_monitor = monitor

# 示例用法
if __name__ == "__main__":
    from utils.logger_config import setup_logger
    logger = setup_logger("health_test")
    
    # 创建健康监控器
    monitor = HealthMonitor(check_interval=10)
    
    # 执行健康检查
    result = monitor.perform_health_check()
    
    print("\n=== Health Check Result ===")
    print(f"Status: {result['status']}")
    print(f"Health Score: {result['health_score']:.1f}%")
    print(f"\nChecks:")
    for check, status in result['checks'].items():
        print(f"  {check}: {'✓' if status else '✗'}")
    
    print(f"\nMetrics:")
    for key, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 获取摘要
    print("\n=== Health Summary ===")
    summary = monitor.get_health_summary(hours=1)
    for key, value in summary.items():
        print(f"{key}: {value}")
