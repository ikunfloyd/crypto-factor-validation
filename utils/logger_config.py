#!/usr/bin/env python3
"""统一日志配置模块"""

import sys
import os
from pathlib import Path
from loguru import logger
from datetime import datetime

class LoggerConfig:
    """统一的日志配置管理器"""
    
    _configured = False
    
    @classmethod
    def setup(cls, 
              module_name: str = "main",
              level: str = None,
              log_to_file: bool = True,
              log_dir: str = "./logs",
              rotation: str = "100 MB",
              retention: str = "30 days",
              colorize: bool = True,
              diagnose: bool = False):
        """
        设置统一的日志配置
        
        Args:
            module_name: 模块名称，用于日志文件名
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: 是否输出到文件
            log_dir: 日志目录
            rotation: 日志轮转大小
            retention: 日志保留时间
            colorize: 是否彩色输出
            diagnose: 是否启用诊断模式（包含额外的调试信息）
        """
        if cls._configured:
            logger.debug(f"Logger already configured for {module_name}")
            return logger
        
        # 从环境变量获取日志级别
        if level is None:
            level = os.getenv('LOG_LEVEL', 'INFO').upper()
        
        # 移除默认处理器
        logger.remove()
        
        # 日志格式
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        # 添加控制台处理器
        logger.add(
            sys.stdout,
            format=console_format,
            level=level,
            colorize=colorize,
            diagnose=diagnose,
            backtrace=True,
            enqueue=True  # 线程安全
        )
        
        # 添加文件处理器
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            # 主日志文件
            logger.add(
                log_path / f"{module_name}_{datetime.now():%Y%m%d}.log",
                format=file_format,
                level=level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                diagnose=diagnose,
                backtrace=True,
                enqueue=True
            )
            
            # 错误专用日志文件
            logger.add(
                log_path / f"{module_name}_error_{datetime.now():%Y%m%d}.log",
                format=file_format,
                level="ERROR",
                rotation=rotation,
                retention=retention,
                compression="zip",
                diagnose=diagnose,
                backtrace=True,
                enqueue=True
            )
        
        cls._configured = True
        logger.info(f"Logger configured for {module_name} at level {level}")
        
        return logger
    
    @classmethod
    def get_logger(cls, module_name: str = "main", level: str = None):
        """
        获取配置好的logger实例
        
        Args:
            module_name: 模块名称
            level: 日志级别
            
        Returns:
            配置好的logger实例
        """
        if not cls._configured:
            cls.setup(module_name=module_name, level=level)
        return logger
    
    @classmethod
    def reset(cls):
        """重置日志配置"""
        logger.remove()
        cls._configured = False
    
    @classmethod
    def set_level(cls, level: str):
        """动态修改日志级别"""
        logger.info(f"Changing log level to {level}")
        # 注意：这会影响所有已配置的处理器
        # 如果需要更精细的控制，需要重新配置
        cls.reset()
        cls.setup(level=level)
    
    @classmethod
    def add_custom_handler(cls, sink, **kwargs):
        """
        添加自定义日志处理器
        
        Args:
            sink: 日志输出目标
            **kwargs: 传递给logger.add的其他参数
        """
        logger.add(sink, **kwargs)

# 便捷函数
def setup_logger(module_name: str = "main", level: str = None, **kwargs):
    """便捷的日志设置函数"""
    return LoggerConfig.setup(module_name=module_name, level=level, **kwargs)

def get_logger(module_name: str = "main", level: str = None):
    """便捷的获取logger函数"""
    return LoggerConfig.get_logger(module_name=module_name, level=level)

# 预设配置
class LogPresets:
    """日志预设配置"""
    
    @staticmethod
    def production():
        """生产环境配置"""
        return {
            'level': 'INFO',
            'log_to_file': True,
            'colorize': False,
            'diagnose': False,
            'rotation': '500 MB',
            'retention': '90 days'
        }
    
    @staticmethod
    def development():
        """开发环境配置"""
        return {
            'level': 'DEBUG',
            'log_to_file': True,
            'colorize': True,
            'diagnose': True,
            'rotation': '100 MB',
            'retention': '7 days'
        }
    
    @staticmethod
    def testing():
        """测试环境配置"""
        return {
            'level': 'DEBUG',
            'log_to_file': False,
            'colorize': True,
            'diagnose': True
        }

# 示例用法
if __name__ == "__main__":
    # 开发环境
    logger = setup_logger("test_module", **LogPresets.development())
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    try:
        1 / 0
    except Exception as e:
        logger.exception("An error occurred")
