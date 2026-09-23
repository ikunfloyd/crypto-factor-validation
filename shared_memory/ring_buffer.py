import atexit
import signal
import sys
import numpy as np
import pyarrow as pa
from multiprocessing import shared_memory
from typing import Dict, Tuple, Optional
import struct
from dataclasses import dataclass
from loguru import logger

# 全局注册需要清理的共享内存对象
_shared_memory_objects = []
_cleanup_registered = False

def _cleanup_shared_memory():
    """清理所有注册的共享内存对象"""
    global _shared_memory_objects
    if not _shared_memory_objects:
        return
    
    logger.info(f"Cleaning up {len(_shared_memory_objects)} shared memory objects")
    for shm_name in _shared_memory_objects:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()  # 彻底删除
            logger.debug(f"Cleaned up shared memory: {shm_name}")
        except FileNotFoundError:
            logger.debug(f"Shared memory already removed: {shm_name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup shared memory {shm_name}: {e}")
    
    _shared_memory_objects.clear()

def _signal_handler(signum, frame):
    """处理终止信号，确保清理共享内存"""
    signal_name = signal.Signals(signum).name
    logger.warning(f"Received signal {signal_name}, cleaning up shared memory...")
    _cleanup_shared_memory()
    sys.exit(0)

def _register_cleanup_handlers():
    """注册清理处理器（退出和信号）"""
    global _cleanup_registered
    if _cleanup_registered:
        return
    
    # 注册退出时的清理函数
    atexit.register(_cleanup_shared_memory)
    
    # 只在主线程中注册信号处理器
    try:
        # 注册信号处理器
        signal.signal(signal.SIGTERM, _signal_handler)  # 终止信号
        signal.signal(signal.SIGINT, _signal_handler)   # Ctrl+C
        
        # 在Unix系统上注册更多信号
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _signal_handler)  # 挂起信号
        
        logger.debug("Registered shared memory cleanup handlers with signal handlers")
    except ValueError as e:
        # 在非主线程中无法注册信号处理器（如Streamlit）
        logger.debug(f"Could not register signal handlers (not in main thread): {e}")
        logger.debug("Only atexit cleanup handler registered")
    
    _cleanup_registered = True

# 自动注册清理处理器
_register_cleanup_handlers()

@dataclass
class RingBufferConfig:
    """环形缓冲区配置"""
    name: str
    dtype: np.dtype
    capacity: int
    item_size: int

class ArrowRingBuffer:
    """基于PyArrow的高性能环形缓冲区"""
    
    def __init__(self, config: RingBufferConfig, cleanup_on_exit: bool = True):
        self.config = config
        self.shm = None
        self.buffer = None
        self.write_idx = 0
        self.read_idx = 0
        self.cleanup_on_exit = cleanup_on_exit
        self._init_shared_memory()
        
    def _init_shared_memory(self):
        """初始化共享内存"""
        total_size = self.config.capacity * self.config.item_size + 16
        self._is_creator = False  # 标记是否是创建者
        
        try:
            # 尝试连接现有的共享内存
            self.shm = shared_memory.SharedMemory(name=self.config.name)
            logger.info(f"Connected to existing shared memory: {self.config.name}")
        except FileNotFoundError:
            # 创建新的共享内存
            self.shm = shared_memory.SharedMemory(
                name=self.config.name,
                create=True,
                size=total_size
            )
            self._is_creator = True
            logger.info(f"Created new shared memory: {self.config.name}")
            
            # 注册到清理列表
            if self.cleanup_on_exit:
                _shared_memory_objects.append(self.config.name)
        
        # 只有创建者才初始化索引区域，避免重置已有数据的索引
        if self._is_creator:
            self._init_indices()
        
    def _init_indices(self):
        """初始化读写索引"""
        # 前16字节存储索引
        indices = struct.pack('QQ', 0, 0)
        self.shm.buf[:16] = indices
        
    @property
    def write_index(self) -> int:
        """获取当前写索引"""
        return struct.unpack('Q', self.shm.buf[0:8])[0]
    
    @write_index.setter
    def write_index(self, value: int):
        """设置写索引"""
        self.shm.buf[0:8] = struct.pack('Q', value)
        
    @property
    def read_index(self) -> int:
        """获取当前读索引"""
        return struct.unpack('Q', self.shm.buf[8:16])[0]
    
    @read_index.setter
    def read_index(self, value: int):
        """设置读索引"""
        self.shm.buf[8:16] = struct.pack('Q', value)
    
    def write(self, data: np.ndarray):
        """写入数据到环形缓冲区。

        语义为「最新覆盖旧数据」(drop-oldest)：写入不检查读指针，缓冲区写满一圈
        后会覆盖尚未读取的最旧数据。这是刻意为之——消费者只关心最新行情，并以
        read_latest(n)/peek_latest(n) 高频拉取最近 n 条；本缓冲不区分「满」与「空」，
        因此不适合当作可靠队列使用。
        """
        if data.dtype != self.config.dtype:
            raise ValueError(f"Data dtype mismatch: {data.dtype} != {self.config.dtype}")
        
        n_items = len(data)
        start_idx = self.write_index
        end_idx = (start_idx + n_items) % self.config.capacity
        
        # 计算在共享内存中的位置
        data_start = 16 + start_idx * self.config.item_size
        data_end = 16 + end_idx * self.config.item_size
        
        if end_idx > start_idx:
            # 连续写入
            self.shm.buf[data_start:data_end] = data.tobytes()
        else:
            # 环绕写入
            first_chunk_size = (self.config.capacity - start_idx) * self.config.item_size
            self.shm.buf[data_start:data_start+first_chunk_size] = data[:self.config.capacity-start_idx].tobytes()
            self.shm.buf[16:data_end] = data[self.config.capacity-start_idx:].tobytes()
        
        # 更新写索引
        self.write_index = end_idx
        
    def read_latest(self, n: int = 1) -> Optional[np.ndarray]:
        """读取最新的n条数据"""
        read_idx = self.read_index
        write_idx = self.write_index
        
        if read_idx == write_idx:
            return None
        
        # 计算可读取的数据量
        if write_idx > read_idx:
            available = write_idx - read_idx
        else:
            available = self.config.capacity - read_idx + write_idx
        
        n = min(n, available)
        
        # 读取数据
        result = np.zeros(n, dtype=self.config.dtype)
        
        for i in range(n):
            idx = (read_idx + i) % self.config.capacity
            data_start = 16 + idx * self.config.item_size
            data_end = data_start + self.config.item_size
            result[i] = np.frombuffer(self.shm.buf[data_start:data_end], dtype=self.config.dtype)[0]
        
        # 更新读索引
        self.read_index = (read_idx + n) % self.config.capacity
        
        return result
    
    def peek_latest(self, n: int = 1) -> Optional[np.ndarray]:
        """
        读取最新的n条数据但不更新读索引（只读模式）
        适用于多个消费者读取同一数据的场景，如 Dashboard 可视化
        """
        read_idx = self.read_index
        write_idx = self.write_index
        
        # 计算可用的数据量
        if write_idx >= read_idx:
            available = write_idx - read_idx
        else:
            # 环形缓冲区绕了一圈
            available = self.config.capacity - read_idx + write_idx
        
        # 没有可用数据
        if available == 0:
            return None
        
        # 实际读取的数据量
        actual_n = min(n, available)
        
        # 从最新位置往前读取n条数据
        # 最新数据的位置是 write_idx - 1
        start_idx = (write_idx - actual_n) % self.config.capacity
        
        # 读取数据
        result = np.zeros(actual_n, dtype=self.config.dtype)
        
        for i in range(actual_n):
            idx = (start_idx + i) % self.config.capacity
            data_start = 16 + idx * self.config.item_size
            data_end = data_start + self.config.item_size
            result[i] = np.frombuffer(self.shm.buf[data_start:data_end], dtype=self.config.dtype)[0]
        
        # 不更新读索引！
        return result
    
    def cleanup(self):
        """手动清理共享内存"""
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
                logger.info(f"Unlinked shared memory: {self.config.name}")
            except:
                pass

    def close(self):
        """关闭共享内存连接（不取消链接）"""
        if self.shm:
            self.shm.close()
                
    def __del__(self):
        """析构函数 - 确保正确清理共享内存"""
        if self.shm:
            try:
                self.shm.close()
                # 只有创建者在cleanup_on_exit=True时才unlink
                if hasattr(self, '_is_creator') and self._is_creator and self.cleanup_on_exit:
                    try:
                        self.shm.unlink()
                        logger.debug(f"Unlinked shared memory in __del__: {self.config.name}")
                    except FileNotFoundError:
                        # 已经被清理
                        pass
                    except Exception as e:
                        logger.warning(f"Error unlinking shared memory in __del__: {e}")
            except Exception as e:
                logger.warning(f"Error in __del__ for {self.config.name}: {e}")

# 预定义的缓冲区配置
RING_BUFFER_CONFIGS = {
    'fast_tick': RingBufferConfig(
        name='ring_buffer_fast',
        dtype=np.dtype([
            ('timestamp', 'f8'),
            ('ask1_price', 'f8'),
            ('ask1_quantity', 'f8'),
            ('bid1_price', 'f8'),
            ('bid1_quantity', 'f8'),
            ('ask5_price', 'f8'),
            ('ask5_quantity', 'f8'),
            ('bid5_price', 'f8'),
            ('bid5_quantity', 'f8')
        ]),
        capacity=3600,
        item_size=72  # 9 * 8 bytes
    ),
    
    'candle': RingBufferConfig(
        name='ring_buffer_candle',
        dtype=np.dtype([
            ('timestamp', 'f8'),
            ('open', 'f8'),
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('volume', 'f8'),
            ('volume_ccy', 'f8')
        ]),
        capacity=5000,
        item_size=56  # 7 * 8 bytes
    )
}