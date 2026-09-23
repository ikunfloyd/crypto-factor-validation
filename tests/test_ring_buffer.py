#!/usr/bin/env python3
"""共享内存环形缓冲区测试"""

import pytest
import numpy as np
from shared_memory.ring_buffer import ArrowRingBuffer, RingBufferConfig

class TestRingBuffer:
    """环形缓冲区测试类"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.config = RingBufferConfig(
            name='test_ring_buffer',
            dtype=np.dtype([('timestamp', 'f8'), ('value', 'f8')]),
            capacity=10,
            item_size=16
        )
    
    def teardown_method(self):
        """每个测试方法后的清理"""
        try:
            buffer = ArrowRingBuffer(self.config, cleanup_on_exit=True)
            buffer.cleanup()
        except:
            pass
    
    def test_create_buffer(self):
        """测试创建缓冲区"""
        buffer = ArrowRingBuffer(self.config, cleanup_on_exit=False)
        assert buffer is not None
        assert buffer.config.name == 'test_ring_buffer'
        buffer.close()
    
    def test_write_read(self):
        """测试写入和读取数据"""
        buffer = ArrowRingBuffer(self.config, cleanup_on_exit=False)
        
        # 写入数据
        test_data = np.array([(1.0, 100.0), (2.0, 200.0)], dtype=self.config.dtype)
        buffer.write(test_data)
        
        # 读取数据
        read_data = buffer.read_latest(2)
        assert read_data is not None
        assert len(read_data) == 2
        assert read_data[0]['timestamp'] == 1.0
        assert read_data[0]['value'] == 100.0
        
        buffer.close()
    
    def test_ring_wrap(self):
        """测试环形缓冲区绕圈"""
        buffer = ArrowRingBuffer(self.config, cleanup_on_exit=False)
        
        # 写入超过容量的数据
        for i in range(15):
            data = np.array([(float(i), float(i * 10))], dtype=self.config.dtype)
            buffer.write(data)
        
        # 读取最新数据
        read_data = buffer.read_latest(5)
        assert read_data is not None
        assert len(read_data) <= 5
        
        buffer.close()
    
    def test_peek_latest(self):
        """测试只读模式读取"""
        buffer = ArrowRingBuffer(self.config, cleanup_on_exit=False)
        
        # 写入数据
        test_data = np.array([(1.0, 100.0)], dtype=self.config.dtype)
        buffer.write(test_data)
        
        # Peek不改变读索引
        read_idx_before = buffer.read_index
        peeked_data = buffer.peek_latest(1)
        read_idx_after = buffer.read_index
        
        assert read_idx_before == read_idx_after
        assert peeked_data is not None
        
        buffer.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
