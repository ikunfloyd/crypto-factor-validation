import os
import struct
import msgpack
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import zlib

class WALWriter:
    """Write-Ahead Log 日志写入器"""
    
    def __init__(self, base_path: str = "./data/wal"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.current_file = None
        self.current_date = None
        self._open_today_file()
        
    def _open_today_file(self):
        """打开今天的WAL文件"""
        today = date.today()
        
        if self.current_date != today:
            if self.current_file:
                self.current_file.close()
                
            # 创建按日期分区的目录
            date_dir = self.base_path / today.strftime("%Y/%m/%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # 打开新的WAL文件
            filename = date_dir / f"{today.strftime('%Y%m%d')}.bin"
            self.current_file = open(filename, 'ab')
            self.current_date = today
            
            logger.info(f"Opened WAL file: {filename}")
            
    def _write_record(self, record_type: bytes, data: bytes):
        """写入记录到WAL文件"""
        self._open_today_file()
        
        # 记录格式: [类型(1字节) | 时间戳(8字节) | 数据长度(4字节) | 数据 | CRC32(4字节)]
        timestamp = struct.pack('d', datetime.now().timestamp())
        data_length = struct.pack('I', len(data))
        
        # 计算CRC32校验
        crc_input = record_type + timestamp + data_length + data
        crc = struct.pack('I', zlib.crc32(crc_input))
        
        # 写入完整记录
        try:
            self.current_file.write(crc_input + crc)
            self.current_file.flush()  # 确保立即写入磁盘
            logger.debug(f"Written WAL record type: {record_type}, size: {len(crc_input) + 4}")
        except Exception as e:
            logger.error(f"Failed to write to WAL file: {e}")
        
    def write_tick(self, tick_data: Dict[str, Any]):
        """写入tick数据"""
        try:
            packed_data = msgpack.packb({
                'type': 'tick',
                'data': tick_data
            }, use_bin_type=True)
            
            self._write_record(b'T', packed_data)
        except Exception as e:
            logger.error(f"Error writing tick to WAL: {e}")
            
    def write_candle(self, candle_data: Dict[str, Any]):
        """写入K线数据"""
        try:
            packed_data = msgpack.packb({
                'type': 'candle',
                'data': candle_data
            }, use_bin_type=True)
            
            self._write_record(b'C', packed_data)
        except Exception as e:
            logger.error(f"Error writing candle to WAL: {e}")
            
    def write_trade(self, trade_data: Dict[str, Any]):
        """写入交易数据"""
        try:
            packed_data = msgpack.packb({
                'type': 'trade',
                'data': trade_data
            }, use_bin_type=True)
            
            self._write_record(b'T', packed_data)
        except Exception as e:
            logger.error(f"Error writing trade to WAL: {e}")
            
    def write_funding_rate(self, fr_data: Dict[str, Any]):
        """写入资金费率"""
        try:
            packed_data = msgpack.packb({
                'type': 'funding_rate',
                'data': fr_data
            }, use_bin_type=True)
            
            self._write_record(b'F', packed_data)
        except Exception as e:
            logger.error(f"Error writing funding rate to WAL: {e}")
            
    def write_open_interest(self, oi_data: Dict[str, Any]):
        """写入持仓量数据"""
        try:
            packed_data = msgpack.packb({
                'type': 'open_interest',
                'data': oi_data
            }, use_bin_type=True)
            
            self._write_record(b'O', packed_data)
        except Exception as e:
            logger.error(f"Error writing open interest to WAL: {e}")
            
    def close(self):
        """关闭WAL文件"""
        if self.current_file:
            self.current_file.close()
            logger.info("WAL writer closed")