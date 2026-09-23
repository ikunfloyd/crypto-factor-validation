import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import numba
from numba import jit

class DataProcessor:
    """数据预处理和特征工程处理器"""
    
    def __init__(self):
        self.column_mapping = {
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'volume_ccy': 'volume_ccy'
        }
    
    def process_candle_data(self, candle_buffer_data: np.ndarray) -> pl.DataFrame:
        """处理K线数据"""
        try:
            if len(candle_buffer_data) == 0:
                return pl.DataFrame()
            
            # 转换为Polars DataFrame
            df = pl.DataFrame({
                'timestamp': candle_buffer_data['timestamp'],
                'open': candle_buffer_data['open'],
                'high': candle_buffer_data['high'],
                'low': candle_buffer_data['low'],
                'close': candle_buffer_data['close'],
                'volume': candle_buffer_data['volume'],
                'volume_ccy': candle_buffer_data['volume_ccy']
            })
            
            # 添加时间特征
            df = self._add_time_features(df)
            
            # 添加价格特征
            df = self._add_price_features(df)
            
            # 添加波动率特征
            df = self._add_volatility_features(df)
            
            # 添加诊断日志：数据处理状态
            logger.info(f"Candle processing - input: {len(candle_buffer_data)} records, output: {len(df)} records with {len(df.columns)} columns")
            logger.debug(f"Processed {len(df)} candle records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing candle data: {e}")
            return pl.DataFrame()
    
    def process_tick_data(self, tick_buffer_data: np.ndarray) -> pl.DataFrame:
        """处理tick数据"""
        try:
            if len(tick_buffer_data) == 0:
                return pl.DataFrame()
            
            df = pl.DataFrame({
                'timestamp': tick_buffer_data['timestamp'],
                'ask1_price': tick_buffer_data['ask1_price'],
                'ask1_quantity': tick_buffer_data['ask1_quantity'],
                'bid1_price': tick_buffer_data['bid1_price'],
                'bid1_quantity': tick_buffer_data['bid1_quantity'],
                'ask5_price': tick_buffer_data['ask5_price'],
                'ask5_quantity': tick_buffer_data['ask5_quantity'],
                'bid5_price': tick_buffer_data['bid5_price'],
                'bid5_quantity': tick_buffer_data['bid5_quantity']
            })
            
            # 计算价差和深度
            df = df.with_columns([
                (pl.col('ask1_price') - pl.col('bid1_price')).alias('spread'),
                ((pl.col('ask1_price') - pl.col('bid1_price')) / 
                 ((pl.col('ask1_price') + pl.col('bid1_price')) / 2)).alias('spread_pct'),
                (pl.col('ask1_quantity') + pl.col('bid1_quantity')).alias('depth_top'),
                (pl.col('ask5_quantity') + pl.col('bid5_quantity')).alias('depth_5')
            ])
            
            # 重采样到1分钟（如果需要）
            df_resampled = self._resample_ticks(df)
            
            logger.debug(f"Processed {len(df)} tick records, resampled to {len(df_resampled)}")
            return df_resampled
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
            return pl.DataFrame()
    
    def _add_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加时间特征"""
        try:
            # 转换为datetime（时间戳单位为秒，转换为毫秒）
            df = df.with_columns([
                (pl.col("timestamp").cast(pl.Int64) * 1000).cast(pl.Datetime(time_unit="ms")).alias("datetime")
            ])
            
            # 提取基本时间特征
            df = df.with_columns([
                pl.col("datetime").dt.hour().alias("hour"),
                pl.col("datetime").dt.minute().alias("minute"),
                pl.col("datetime").dt.second().alias("second"),
                pl.col("datetime").dt.weekday().alias("weekday")
            ])
            
            # 添加市场时段特征（基于已创建的hour列）
            df = df.with_columns([
                ((pl.col("hour") >= 9) & (pl.col("hour") < 17)).alias("market_hours"),
                (((pl.col("hour") >= 1) & (pl.col("hour") < 9)).cast(pl.Int8)).alias("asia_session"),
                (((pl.col("hour") >= 9) & (pl.col("hour") < 17)).cast(pl.Int8)).alias("europe_session"),
                (((pl.col("hour") >= 17) | (pl.col("hour") < 1)).cast(pl.Int8)).alias("us_session")
            ])
            
            return df
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return df
    
    def _add_price_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加价格特征"""
        try:
            df = df.with_columns([
                # 价格变化
                pl.col("close").pct_change().alias("returns"),
                # 高低价差
                ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_pct"),
                # 实体大小
                ((pl.col("close") - pl.col("open")).abs() / pl.col("close")).alias("body_pct"),
                # 上下影线
                ((pl.col("high") - pl.max_horizontal(pl.col("close"), pl.col("open"))) / pl.col("close")).alias("upper_shadow"),
                ((pl.min_horizontal(pl.col("close"), pl.col("open")) - pl.col("low")) / pl.col("close")).alias("lower_shadow"),
            ])
            
            # 对数收益率
            df = df.with_columns([
                pl.col("close").log().diff().alias("log_returns")
            ])
            
            return df
        except Exception as e:
            logger.error(f"Error adding price features: {e}")
            return df
    
    def _add_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加波动率特征"""
        try:
            # 计算滚动波动率
            window_sizes = [5, 10, 20, 60]  # 5分钟到1小时
            
            for window in window_sizes:
                if len(df) >= window:
                    df = df.with_columns([
                        pl.col("returns").rolling_std(window).alias(f"volatility_{window}m")
                    ])
            
            # ATR（平均真实波幅）
            if len(df) > 0:
                df = df.with_columns([
                    (pl.col("high") - pl.col("low")).alias("tr")
                ])
                
                # 滚动ATR
                if len(df) >= 14:
                    df = df.with_columns([
                        pl.col("tr").rolling_mean(14).alias("atr_14")
                    ])
                    df = df.with_columns([
                        (pl.col("atr_14") / pl.col("close")).alias("atr_pct_14")
                    ])
            
            return df
        except Exception as e:
            logger.error(f"Error adding volatility features: {e}")
            return df
    
    @staticmethod
    @jit(nopython=True)
    def _resample_ticks_numba(timestamps, values, target_interval=60):
        """使用Numba加速的tick重采样"""
        if len(timestamps) == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        
        # 找出时间范围
        start_time = timestamps[0]
        end_time = timestamps[-1]
        
        # 创建目标时间点
        num_buckets = int((end_time - start_time) / target_interval) + 1
        bucket_times = np.arange(num_buckets) * target_interval + start_time
        
        # 初始化结果数组
        bucket_values = np.full(num_buckets, np.nan)
        bucket_counts = np.zeros(num_buckets, dtype=np.int32)
        
        # 将tick分配到桶中
        current_bucket = 0
        for i in range(len(timestamps)):
            ts = timestamps[i]
            val = values[i]
            
            # 找到对应的桶
            while current_bucket < num_buckets - 1 and ts >= bucket_times[current_bucket + 1]:
                current_bucket += 1
            
            if not np.isnan(val):
                if np.isnan(bucket_values[current_bucket]):
                    bucket_values[current_bucket] = val
                else:
                    bucket_values[current_bucket] = (bucket_values[current_bucket] * bucket_counts[current_bucket] + val) / (bucket_counts[current_bucket] + 1)
                bucket_counts[current_bucket] += 1
        
        return bucket_times, bucket_values
    
    def _resample_ticks(self, df: pl.DataFrame, interval_seconds: int = 60) -> pl.DataFrame:
        """将tick数据重采样到指定间隔"""
        try:
            if len(df) == 0:
                return pl.DataFrame()
            
            # 提取数据
            timestamps = df['timestamp'].to_numpy()
            
            # 对每个需要重采样的列进行处理
            resampled_columns = {}
            
            for col in ['ask1_price', 'bid1_price', 'spread', 'spread_pct', 'depth_top']:
                if col in df.columns:
                    values = df[col].to_numpy()
                    bucket_times, bucket_values = self._resample_ticks_numba(
                        timestamps, values, interval_seconds
                    )
                    
                    # 过滤掉没有数据的桶
                    mask = ~np.isnan(bucket_values)
                    if np.any(mask):
                        if col not in resampled_columns:
                            resampled_columns['timestamp'] = bucket_times[mask]
                        resampled_columns[col] = bucket_values[mask]
            
            # 创建重采样后的DataFrame
            if resampled_columns:
                resampled_df = pl.DataFrame(resampled_columns)
                return resampled_df
            else:
                return pl.DataFrame()
                
        except Exception as e:
            logger.error(f"Error resampling ticks: {e}")
            return df
    
    def align_dataframes(self, dataframes: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        """对齐不同频率的DataFrame"""
        try:
            if not dataframes:
                return {}
            
            # 找到时间戳最多的DataFrame作为基准
            main_key = None
            max_length = 0
            
            for key, df in dataframes.items():
                if len(df) > max_length:
                    max_length = len(df)
                    main_key = key
            
            if main_key is None:
                return dataframes
            
            main_df = dataframes[main_key]
            main_timestamps = set(main_df['timestamp'].to_list())
            
            # 对齐其他DataFrame
            aligned_data = {main_key: main_df}
            
            for key, df in dataframes.items():
                if key == main_key:
                    continue
                
                if len(df) == 0:
                    aligned_data[key] = df
                    continue
                
                # 使用最近邻插值对齐时间戳
                aligned_df = self._align_to_reference(df, main_timestamps)
                aligned_data[key] = aligned_df
            
            return aligned_data
            
        except Exception as e:
            logger.error(f"Error aligning dataframes: {e}")
            return dataframes
    
    def _align_to_reference(self, df: pl.DataFrame, reference_timestamps: set) -> pl.DataFrame:
        """将DataFrame对齐到参考时间戳"""
        try:
            if len(df) == 0:
                return df
            
            # 获取DataFrame的时间戳
            df_timestamps = df['timestamp'].to_list()
            df_values = {ts: idx for idx, ts in enumerate(df_timestamps)}
            
            # 为每个参考时间戳找到最近的原始时间戳
            aligned_rows = []
            
            for ref_ts in sorted(reference_timestamps):
                # 找到最近的时间戳
                nearest_idx = min(
                    range(len(df_timestamps)),
                    key=lambda i: abs(df_timestamps[i] - ref_ts)
                )
                
                # 如果时间差太大，跳过
                if abs(df_timestamps[nearest_idx] - ref_ts) > 300:  # 5分钟阈值
                    continue
                
                # 获取该行的数据
                row = df.row(nearest_idx)
                row_dict = dict(zip(df.columns, row))
                row_dict['timestamp'] = ref_ts  # 使用参考时间戳
                
                aligned_rows.append(row_dict)
            
            if aligned_rows:
                return pl.DataFrame(aligned_rows)
            else:
                return pl.DataFrame()
                
        except Exception as e:
            logger.error(f"Error aligning to reference: {e}")
            return df