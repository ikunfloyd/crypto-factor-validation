#!/usr/bin/env python3
"""交易配置管理模块"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json
from loguru import logger

@dataclass
class RiskConfig:
    """风险管理配置"""
    max_position_pct: float = 0.8  # 最大持仓比例
    max_daily_loss_pct: float = 0.05  # 最大日损失比例
    max_drawdown_pct: float = 0.15  # 最大回撤比例
    position_size_limit: float = 1000000.0  # 单笔最大仓位（USDT）
    leverage: int = 1  # 杠杆倍数
    stop_loss_pct: float = 0.02  # 止损比例
    take_profit_pct: float = 0.05  # 止盈比例

@dataclass
class ModelConfig:
    """模型配置"""
    model_dir: str = "./data/models"
    fallback_mode: bool = True  # 模型加载失败时是否使用降级策略
    update_interval: int = 3600  # 模型更新间隔（秒）
    confidence_threshold: float = 0.6  # 信号置信度阈值
    ensemble_method: str = "weighted"  # 集成方法: weighted, voting, stacking

@dataclass
class DataConfig:
    """数据配置"""
    data_dir: str = "./data"
    checkpoint_dir: str = "./data/checkpoints"
    checkpoint_interval: int = 300  # 检查点保存间隔（秒）
    wal_dir: str = "./data/wal"
    replay_dir: str = "./data/replay"
    max_candle_buffer: int = 5000  # K线缓冲区大小
    max_tick_buffer: int = 3600  # Tick缓冲区大小

@dataclass
class ExecutionConfig:
    """执行配置"""
    calculation_interval: int = 60  # 计算间隔（秒）
    order_timeout: int = 30  # 订单超时时间（秒）
    max_retry_attempts: int = 3  # 最大重试次数
    slippage_tolerance_pct: float = 0.001  # 滑点容忍度
    enable_paper_trading: bool = False  # 是否启用模拟交易

@dataclass
class APIConfig:
    """API配置"""
    api_key: str = ""
    api_secret: str = ""
    api_password: str = ""
    base_url: str = "https://www.okx.com"
    ws_public_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    ws_private_url: str = "wss://ws.okx.com:8443/ws/v5/private"
    request_timeout: int = 30  # API请求超时（秒）
    rate_limit: int = 20  # 每秒请求限制

@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_monitoring: bool = True
    health_check_interval: int = 60  # 健康检查间隔（秒）
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None
    performance_log_interval: int = 300  # 性能日志间隔（秒）

@dataclass
class TradingConfig:
    """完整交易配置"""
    # 子配置
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 通用配置
    environment: str = "development"  # development, staging, production
    symbol: str = "DOGE-USDT"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """从环境变量加载配置"""
        config = cls()
        
        # API配置从环境变量加载
        config.api.api_key = os.getenv('OKX_API_KEY', '')
        config.api.api_secret = os.getenv('OKX_SECRET', '')
        config.api.api_password = os.getenv('OKX_PASSWORD', '')
        
        # 风险配置
        config.risk.max_position_pct = float(os.getenv('MAX_POSITION_PCT', '0.8'))
        config.risk.max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PCT', '0.05'))
        config.risk.leverage = int(os.getenv('LEVERAGE', '1'))
        
        # 执行配置
        config.execution.calculation_interval = int(os.getenv('CALCULATION_INTERVAL', '60'))
        config.execution.enable_paper_trading = os.getenv('PAPER_TRADING', 'false').lower() == 'true'
        
        # 数据配置
        config.data.checkpoint_interval = int(os.getenv('CHECKPOINT_INTERVAL', '300'))
        
        # 通用配置
        config.environment = os.getenv('ENVIRONMENT', 'development')
        config.symbol = os.getenv('SYMBOL', 'DOGE-USDT')
        config.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        logger.info(f"Configuration loaded from environment for {config.environment}")
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TradingConfig':
        """从JSON文件加载配置"""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            config = cls()
            
            # 更新各个子配置
            if 'risk' in config_dict:
                config.risk = RiskConfig(**config_dict['risk'])
            if 'model' in config_dict:
                config.model = ModelConfig(**config_dict['model'])
            if 'data' in config_dict:
                config.data = DataConfig(**config_dict['data'])
            if 'execution' in config_dict:
                config.execution = ExecutionConfig(**config_dict['execution'])
            if 'api' in config_dict:
                config.api = APIConfig(**config_dict['api'])
            if 'monitoring' in config_dict:
                config.monitoring = MonitoringConfig(**config_dict['monitoring'])
            
            # 更新通用配置
            config.environment = config_dict.get('environment', 'development')
            config.symbol = config_dict.get('symbol', 'DOGE-USDT')
            config.log_level = config_dict.get('log_level', 'INFO')
            
            logger.info(f"Configuration loaded from file: {config_path}")
            
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using default configuration")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config from file: {e}")
            return cls()
    
    def to_file(self, config_path: str):
        """保存配置到JSON文件"""
        try:
            config_dict = {
                'risk': asdict(self.risk),
                'model': asdict(self.model),
                'data': asdict(self.data),
                'execution': asdict(self.execution),
                'api': asdict(self.api),
                'monitoring': asdict(self.monitoring),
                'environment': self.environment,
                'symbol': self.symbol,
                'log_level': self.log_level
            }
            
            # 创建目录
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to file: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to file: {e}")
    
    def validate(self) -> bool:
        """验证配置合法性"""
        errors = []
        
        # 验证API配置
        if not self.api.api_key or not self.api.api_secret:
            errors.append("API credentials are required")
        
        # 验证风险配置
        if not 0 < self.risk.max_position_pct <= 1:
            errors.append("max_position_pct must be between 0 and 1")
        
        if not 0 < self.risk.max_daily_loss_pct <= 1:
            errors.append("max_daily_loss_pct must be between 0 and 1")
        
        if self.risk.leverage < 1:
            errors.append("leverage must be >= 1")
        
        # 验证执行配置
        if self.execution.calculation_interval < 1:
            errors.append("calculation_interval must be >= 1")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'risk': asdict(self.risk),
            'model': asdict(self.model),
            'data': asdict(self.data),
            'execution': asdict(self.execution),
            'api': asdict(self.api),
            'monitoring': asdict(self.monitoring),
            'environment': self.environment,
            'symbol': self.symbol,
            'log_level': self.log_level
        }
    
    def get_summary(self) -> str:
        """获取配置摘要"""
        summary = f"""
Trading Configuration Summary:
=============================
Environment: {self.environment}
Symbol: {self.symbol}
Log Level: {self.log_level}

Risk Management:
  - Max Position: {self.risk.max_position_pct:.1%}
  - Max Daily Loss: {self.risk.max_daily_loss_pct:.1%}
  - Leverage: {self.risk.leverage}x

Model:
  - Model Dir: {self.model.model_dir}
  - Fallback Mode: {self.model.fallback_mode}
  - Confidence Threshold: {self.model.confidence_threshold:.1%}

Execution:
  - Calculation Interval: {self.execution.calculation_interval}s
  - Paper Trading: {self.execution.enable_paper_trading}
  - Max Retry: {self.execution.max_retry_attempts}

Data:
  - Checkpoint Interval: {self.data.checkpoint_interval}s
  - Max Candle Buffer: {self.data.max_candle_buffer}
  
Monitoring:
  - Enabled: {self.monitoring.enable_monitoring}
  - Health Check Interval: {self.monitoring.health_check_interval}s
"""
        return summary

# 全局配置实例
_config_instance: Optional[TradingConfig] = None

def get_config() -> TradingConfig:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = TradingConfig.from_env()
    return _config_instance

def set_config(config: TradingConfig):
    """设置全局配置实例"""
    global _config_instance
    _config_instance = config

def reload_config(config_path: Optional[str] = None):
    """重新加载配置"""
    global _config_instance
    if config_path:
        _config_instance = TradingConfig.from_file(config_path)
    else:
        _config_instance = TradingConfig.from_env()
    return _config_instance

# 示例用法
if __name__ == "__main__":
    from utils.logger_config import setup_logger
    logger = setup_logger("config_test")
    
    # 从环境变量加载
    config = TradingConfig.from_env()
    print(config.get_summary())
    
    # 验证配置
    config.validate()
    
    # 保存配置示例
    config.to_file("./config/trading_config.json")
    
    # 从文件加载
    config2 = TradingConfig.from_file("./config/trading_config.json")
    print("\nLoaded from file:")
    print(config2.get_summary())
