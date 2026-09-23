import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from loguru import logger
import numpy as np

from shared_memory.ring_buffer import ArrowRingBuffer, RING_BUFFER_CONFIGS
from io_gateway.wal_reader import WALReader
from io_gateway.wal_writer import WALWriter

class ReplayClient:
    """WAL日志回放客户端"""
    
    def __init__(self, speed_factor: float = 1.0):
        self.speed_factor = speed_factor  # 回放速度因子
        self.running = False
        
        # 初始化共享内存缓冲区
        self.tick_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['fast_tick'])
        self.candle_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['candle'])
        
        # WAL读取器和写入器
        self.wal_reader = WALReader()
        self.wal_writer = WALWriter("./data/replay")  # 回放时写入单独的目录
        
        # 回放状态
        self.replay_start_time = None
        self.processed_records = 0
        
    async def replay_from_checkpoint(self):
        """从最新检查点开始回放"""
        logger.info("Starting replay from latest checkpoint...")
        
        def process_record(record: Dict[str, Any]):
            """处理回放记录"""
            try:
                record_type = record['type']
                record_data = record['data']['data']
                
                if record_type == 'T':  # Tick数据
                    self._process_replay_tick(record_data)
                elif record_type == 'C':  # K线数据
                    self._process_replay_candle(record_data)
                    
                self.processed_records += 1
                
                if self.processed_records % 100 == 0:
                    logger.info(f"Processed {self.processed_records} records")
                    
            except Exception as e:
                logger.error(f"Error processing replay record: {e}")
        
        # 开始回放
        records = self.wal_reader.replay_from_checkpoint(callback=process_record)
        
        logger.info(f"Replay completed. Total records: {self.processed_records}")
        
    def _process_replay_tick(self, tick_data: Dict[str, Any]):
        """处理回放的tick数据"""
        try:
            # 准备tick数据
            tick = np.array([(
                tick_data['timestamp'],
                float(tick_data['asks'][0][0]),  # ask1_price
                float(tick_data['asks'][0][1]),  # ask1_quantity
                float(tick_data['bids'][0][0]),  # bid1_price
                float(tick_data['bids'][0][1]),  # bid1_quantity
                float(tick_data['asks'][4][0]) if len(tick_data['asks']) > 4 else float(tick_data['asks'][0][0]),  # ask5_price
                float(tick_data['asks'][4][1]) if len(tick_data['asks']) > 4 else float(tick_data['asks'][0][1]),  # ask5_quantity
                float(tick_data['bids'][4][0]) if len(tick_data['bids']) > 4 else float(tick_data['bids'][0][0]),  # bid5_price
                float(tick_data['bids'][4][1]) if len(tick_data['bids']) > 4 else float(tick_data['bids'][0][1])   # bid5_quantity
            )], dtype=self.tick_buffer.config.dtype)
            
            # 写入共享内存
            self.tick_buffer.write(tick)
            
        except Exception as e:
            logger.error(f"Error processing replay tick: {e}")
            
    def _process_replay_candle(self, candle_data: Dict[str, Any]):
        """处理回放的K线数据"""
        try:
            # 准备K线数据
            candle_np = np.array([(
                candle_data['timestamp'],
                float(candle_data['open']),
                float(candle_data['high']),
                float(candle_data['low']),
                float(candle_data['close']),
                float(candle_data['volume']),
                float(candle_data['volume_ccy'])
            )], dtype=self.candle_buffer.config.dtype)
            
            # 写入共享内存
            self.candle_buffer.write(candle_np)
            
        except Exception as e:
            logger.error(f"Error processing replay candle: {e}")
    
    def measure_replay_speed(self, duration_seconds: int = 10):
        """测试回放速度"""
        logger.info(f"Measuring replay speed for {duration_seconds} seconds...")
        
        start_time = time.time()
        records_processed = 0
        
        def count_record(record: Dict[str, Any]):
            nonlocal records_processed
            records_processed += 1
        
        # 回放最近的数据
        end_time = datetime.now()
        start_time_replay = end_time - timedelta(seconds=duration_seconds * 2)  # 确保有足够数据
        
        self.wal_reader.replay(start_time_replay, end_time, callback=count_record)
        
        elapsed = time.time() - start_time
        speed = records_processed / elapsed if elapsed > 0 else 0
        
        logger.info(f"Replay speed: {speed:.1f} records/sec")
        logger.info(f"M3 performance: {elapsed:.2f} seconds for {records_processed} records")
        
        # 检查是否满足性能要求（1小时数据 < 5秒）
        if speed > 0:
            estimated_time_for_1h = 3600 / speed
            logger.info(f"Estimated time for 1 hour data: {estimated_time_for_1h:.2f} seconds")
            
            if estimated_time_for_1h < 5:
                logger.success("✅ M3 performance requirement met!")
            else:
                logger.warning(f"⚠️  M3 performance requirement not met (needs < 5s, got {estimated_time_for_1h:.2f}s)")
        
        return speed
    
    async def stop(self):
        """停止回放客户端"""
        self.running = False
        self.tick_buffer.cleanup()
        self.candle_buffer.cleanup()
        self.wal_writer.close()
        logger.info("Replay client stopped")

async def test_replay():
    """测试回放功能"""
    client = ReplayClient()
    
    try:
        # 测试回放速度
        client.measure_replay_speed(5)
        
        # 实际回放数据
        await client.replay_from_checkpoint()
        
    finally:
        await client.stop()

if __name__ == "__main__":
    asyncio.run(test_replay())