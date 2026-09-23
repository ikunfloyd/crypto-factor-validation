#!/usr/bin/env python3
"""性能监控和优化工具"""

import time
import functools
from typing import Callable, Any, Dict
from collections import defaultdict
from loguru import logger

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
    
    def record_timing(self, func_name: str, duration: float):
        """记录函数执行时间"""
        self.timings[func_name].append(duration)
        self.call_counts[func_name] += 1
    
    def get_stats(self, func_name: str = None) -> Dict[str, Any]:
        """获取性能统计"""
        if func_name:
            if func_name not in self.timings:
                return {}
            
            timings = self.timings[func_name]
            return {
                'function': func_name,
                'calls': self.call_counts[func_name],
                'total_time': sum(timings),
                'avg_time': sum(timings) / len(timings),
                'min_time': min(timings),
                'max_time': max(timings)
            }
        else:
            # 返回所有函数的统计
            stats = {}
            for fname in self.timings.keys():
                stats[fname] = self.get_stats(fname)
            return stats
    
    def reset(self):
        """重置统计数据"""
        self.timings.clear()
        self.call_counts.clear()
    
    def print_report(self):
        """打印性能报告"""
        stats = self.get_stats()
        if not stats:
            logger.info("No performance data collected")
            return
        
        logger.info("\n" + "="*60)
        logger.info("Performance Report")
        logger.info("="*60)
        
        # 按平均时间排序
        sorted_stats = sorted(
            stats.items(),
            key=lambda x: x[1].get('avg_time', 0),
            reverse=True
        )
        
        for func_name, data in sorted_stats:
            logger.info(f"\n{func_name}:")
            logger.info(f"  Calls: {data['calls']}")
            logger.info(f"  Total: {data['total_time']:.4f}s")
            logger.info(f"  Avg: {data['avg_time']*1000:.2f}ms")
            logger.info(f"  Min: {data['min_time']*1000:.2f}ms")
            logger.info(f"  Max: {data['max_time']*1000:.2f}ms")
        
        logger.info("="*60)

# 全局性能监控器实例
_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    return _performance_monitor

def timeit(func: Callable = None, log_result: bool = False):
    """
    性能计时装饰器
    
    Usage:
        @timeit
        def my_function():
            pass
        
        @timeit(log_result=True)
        def my_function():
            pass
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                _performance_monitor.record_timing(fn.__name__, duration)
                
                if log_result:
                    logger.debug(f"{fn.__name__} took {duration*1000:.2f}ms")
        
        return wrapper
    
    # 支持 @timeit 和 @timeit() 两种用法
    if func is None:
        return decorator
    else:
        return decorator(func)

class Timer:
    """上下文管理器计时器"""
    
    def __init__(self, name: str = "Timer", log_result: bool = True):
        self.name = name
        self.log_result = log_result
        self.start_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.duration = time.time() - self.start_time
        
        if self.log_result:
            logger.debug(f"{self.name} took {self.duration*1000:.2f}ms")
        
        _performance_monitor.record_timing(self.name, self.duration)

# 示例用法
if __name__ == "__main__":
    from utils.logger_config import setup_logger
    logger = setup_logger("performance_test")
    
    # 使用装饰器
    @timeit(log_result=True)
    def slow_function():
        time.sleep(0.1)
        return "done"
    
    # 使用上下文管理器
    def another_function():
        with Timer("SomeOperation", log_result=True):
            time.sleep(0.05)
    
    # 测试
    for _ in range(5):
        slow_function()
        another_function()
    
    # 打印报告
    monitor = get_performance_monitor()
    monitor.print_report()
