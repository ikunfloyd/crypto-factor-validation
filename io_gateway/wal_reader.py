import os
import struct
import msgpack
import zlib
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, Tuple, List
import pandas as pd
from loguru import logger
import numpy as np

class WALReader:
    """WAL日志读取器，支持回放和灾难恢复"""
    
    def __init__(self, base_path: str = "./data/wal"):
        self.base_path = Path(base_path)
        
    def read_wal_file(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """读取单个WAL文件"""
        try:
            with open(file_path, 'rb') as f:
                while True:
                    # 读取记录头
                    header = f.read(1)  # 类型
                    if not header:
                        break
                        
                    # 读取时间戳
                    timestamp_bytes = f.read(8)
                    if not timestamp_bytes or len(timestamp_bytes) < 8:
                        break
                    timestamp = struct.unpack('d', timestamp_bytes)[0]
                    
                    # 读取数据长度
                    length_bytes = f.read(4)
                    if not length_bytes or len(length_bytes) < 4:
                        break
                    data_length = struct.unpack('I', length_bytes)[0]
                    
                    # 读取数据
                    data_bytes = f.read(data_length)
                    if not data_bytes or len(data_bytes) < data_length:
                        break
                    
                    # 读取CRC32校验
                    crc_bytes = f.read(4)
                    if not crc_bytes or len(crc_bytes) < 4:
                        break
                    
                    # 验证CRC32
                    crc_input = header + timestamp_bytes + length_bytes + data_bytes
                    expected_crc = struct.unpack('I', crc_bytes)[0]
                    actual_crc = zlib.crc32(crc_input)
                    
                    if expected_crc != actual_crc:
                        logger.warning(f"CRC32 mismatch in {file_path}, skipping record")
                        continue
                    
                    # 解析数据
                    try:
                        record_type = header.decode('ascii')
                        data = msgpack.unpackb(data_bytes, raw=False)
                        
                        yield {
                            'type': record_type,
                            'timestamp': timestamp,
                            'data': data
                        }
                    except Exception as e:
                        logger.error(f"Failed to parse record: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading WAL file {file_path}: {e}")
            
    def get_wal_files(self, start_date: date, end_date: date = None) -> List[Path]:
        """获取指定日期范围内的WAL文件"""
        if end_date is None:
            end_date = date.today()
            
        files = []
        current_date = start_date
        
        while current_date <= end_date:
            date_dir = self.base_path / current_date.strftime("%Y/%m/%d")
            filename = date_dir / f"{current_date.strftime('%Y%m%d')}.bin"
            
            if filename.exists():
                files.append(filename)
            
            current_date += timedelta(days=1)
            
        return files
    
    def replay(self, start_time: datetime, end_time: datetime = None, 
               callback=None) -> List[Dict[str, Any]]:
        """
        回放指定时间范围内的WAL日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间，默认为当前时间
            callback: 回调函数，处理每条回放记录
        """
        if end_time is None:
            end_time = datetime.now()
            
        start_date = start_time.date()
        end_date = end_time.date()
        
        wal_files = self.get_wal_files(start_date, end_date)
        all_records = []
        
        logger.info(f"Replaying WAL from {start_time} to {end_time}")
        logger.info(f"Found {len(wal_files)} WAL files")
        
        for file_path in wal_files:
            logger.debug(f"Processing WAL file: {file_path}")
            
            for record in self.read_wal_file(file_path):
                record_time = datetime.fromtimestamp(record['timestamp'])
                
                # 检查时间范围
                if record_time < start_time:
                    continue
                if record_time > end_time:
                    break
                
                # 处理记录
                if callback:
                    try:
                        callback(record)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                all_records.append(record)
                
        logger.info(f"Replay completed. Processed {len(all_records)} records")
        return all_records
    
    def get_latest_checkpoint(self) -> Optional[datetime]:
        """获取最新的检查点时间"""
        checkpoint_file = Path("./data/checkpoints/latest_checkpoint.txt")
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    return datetime.fromtimestamp(float(timestamp_str))
            except Exception as e:
                logger.error(f"Error reading checkpoint: {e}")
        return None
    
    def replay_from_checkpoint(self, callback=None) -> List[Dict[str, Any]]:
        """从最新检查点开始回放"""
        checkpoint_time = self.get_latest_checkpoint()
        
        if checkpoint_time:
            logger.info(f"Found checkpoint at {checkpoint_time}")
            return self.replay(checkpoint_time, callback=callback)
        else:
            logger.info("No checkpoint found, replaying from 1 hour ago")
            start_time = datetime.now() - timedelta(hours=1)
            return self.replay(start_time, callback=callback)
    
    def read_range(self, start_time: datetime, end_time: datetime = None,
                   callback=None) -> List[Dict[str, Any]]:
        """
        别名方法，兼容旧API
        向后兼容：read_range() -> replay()
        
        Args:
            start_time: 开始时间
            end_time: 结束时间，默认为当前时间
            callback: 回调函数，处理每条记录
        
        Returns:
            记录列表
        """
        return self.replay(start_time, end_time, callback)