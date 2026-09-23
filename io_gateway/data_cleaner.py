#!/usr/bin/env python3
"""数据清洗和预处理模块"""

import numpy as np
import polars as pl
from typing import Optional, Dict, Any
from loguru import logger

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, 
                 remove_outliers: bool = True,
                 fill_missing: bool = True,
                 validate_integrity: bool = True):
        """
        初始化数据清洗器
        
        Args:
            remove_outliers: 是否移除异常值
            fill_missing: 是否填充缺失值
            validate_integrity: 是否验证数据完整性
        """
        self.remove_outliers = remove_outliers
        self.fill_missing = fill_missing
        self.validate_integrity = validate_integrity
        
        # 统计信息
        self.stats = {
            'total_records': 0,
            'cleaned_records': 0,
            'outliers_removed': 0,
            'missing_filled': 0,
            'invalid_records': 0
        }
        
        logger.info("DataCleaner initialized")
    
    def clean_candle_data(self, data: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        清洗K线数据
        
        Args:
            data: 原始K线数据
            
        Returns:
            清洗后的数据
        """
        try:
            if data is None or len(data) == 0:
                return None
            
            self.stats['total_records'] += len(data)
            cleaned_data = data.clone()
            
            # 1. 验证数据完整性
            if self.validate_integrity:
                cleaned_data = self._validate_candle_integrity(cleaned_data)
            
            # 2. 移除异常值
            if self.remove_outliers:
                cleaned_data = self._remove_price_outliers(cleaned_data)
            
            # 3. 填充缺失值
            if self.fill_missing:
                cleaned_data = self._fill_missing_values(cleaned_data)
            
            # 4. 移除重复数据
            cleaned_data = self._remove_duplicates(cleaned_data)
            
            self.stats['cleaned_records'] += len(cleaned_data)
            
            logger.debug(f"Cleaned {len(data)} -> {len(cleaned_data)} records")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error cleaning candle data: {e}")
            return data
    
    def _validate_candle_integrity(self, data: pl.DataFrame) -> pl.DataFrame:
        """验证K线数据完整性"""
        try:
            # 检查必需列
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                logger.warning(f"Missing required columns: {required_cols}")
                return data
            
            # 过滤无效记录
            initial_len = len(data)
            
            # 价格必须为正
            data = data.filter(
                (pl.col('open') > 0) &
                (pl.col('high') > 0) &
                (pl.col('low') > 0) &
                (pl.col('close') > 0)
            )
            
            # High >= Low
            data = data.filter(pl.col('high') >= pl.col('low'))
            
            # High >= max(Open, Close) 且 Low <= min(Open, Close)
            data = data.filter(
                (pl.col('high') >= pl.col('open')) &
                (pl.col('high') >= pl.col('close')) &
                (pl.col('low') <= pl.col('open')) &
                (pl.col('low') <= pl.col('close'))
            )
            
            # 成交量非负
            data = data.filter(pl.col('volume') >= 0)
            
            removed = initial_len - len(data)
            if removed > 0:
                self.stats['invalid_records'] += removed
                logger.debug(f"Removed {removed} invalid records")
            
            return data
            
        except Exception as e:
            logger.error(f"Error validating candle integrity: {e}")
            return data
    
    def _remove_price_outliers(self, data: pl.DataFrame) -> pl.DataFrame:
        """移除价格异常值"""
        try:
            if 'close' not in data.columns or len(data) < 10:
                return data
            
            initial_len = len(data)
            
            # 使用滚动窗口检测异常值
            # 异常值定义：价格偏离移动平均线超过3倍标准差
            window_size = min(20, len(data))
            
            # 计算移动平均和标准差
            data = data.with_columns([
                pl.col('close').rolling_mean(window_size).alias('ma'),
                pl.col('close').rolling_std(window_size).alias('std')
            ])
            
            # 过滤异常值
            data = data.filter(
                (pl.col('close') - pl.col('ma')).abs() <= 3 * pl.col('std')
            )
            
            # 移除临时列
            data = data.drop(['ma', 'std'])
            
            removed = initial_len - len(data)
            if removed > 0:
                self.stats['outliers_removed'] += removed
                logger.debug(f"Removed {removed} outliers")
            
            return data
            
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return data
    
    def _fill_missing_values(self, data: pl.DataFrame) -> pl.DataFrame:
        """填充缺失值"""
        try:
            # 检查缺失值
            missing_count = data.null_count()
            
            if missing_count.sum().sum() == 0:
                return data
            
            # 对价格列使用前向填充
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns:
                    data = data.with_columns(
                        pl.col(col).fill_null(strategy='forward').alias(col)
                    )
            
            # 对成交量使用0填充
            if 'volume' in data.columns:
                data = data.with_columns(
                    pl.col('volume').fill_null(0).alias('volume')
                )
            
            self.stats['missing_filled'] += missing_count.sum().sum()
            
            return data
            
        except Exception as e:
            logger.error(f"Error filling missing values: {e}")
            return data
    
    def _remove_duplicates(self, data: pl.DataFrame) -> pl.DataFrame:
        """移除重复数据"""
        try:
            if 'timestamp' not in data.columns:
                return data
            
            initial_len = len(data)
            
            # 按时间戳去重，保留最后一条
            data = data.unique(subset=['timestamp'], keep='last')
            
            removed = initial_len - len(data)
            if removed > 0:
                logger.debug(f"Removed {removed} duplicates")
            
            return data
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            return data
    
    def get_stats(self) -> Dict[str, Any]:
        """获取清洗统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_records': 0,
            'cleaned_records': 0,
            'outliers_removed': 0,
            'missing_filled': 0,
            'invalid_records': 0
        }
        logger.debug("DataCleaner stats reset")

# 示例用法
if __name__ == "__main__":
    from utils.logger_config import setup_logger
    logger = setup_logger("data_cleaner_test")
    
    # 创建测试数据
    test_data = pl.DataFrame({
        'timestamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        'close': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
    })
    
    # 创建清洗器
    cleaner = DataCleaner()
    
    # 清洗数据
    cleaned = cleaner.clean_candle_data(test_data)
    
    print(f"\nOriginal: {len(test_data)} records")
    print(f"Cleaned: {len(cleaned)} records")
    print(f"\nStats: {cleaner.get_stats()}")
