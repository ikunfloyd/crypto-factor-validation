#!/usr/bin/env python3
"""pytest配置文件"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """项目根目录路径fixture"""
    return project_root

@pytest.fixture
def sample_candle_data():
    """示例K线数据fixture"""
    import numpy as np
    import polars as pl
    
    n = 100
    data = {
        'timestamp': np.arange(n, dtype=float),
        'open': np.random.randn(n) * 10 + 100,
        'high': np.random.randn(n) * 10 + 105,
        'low': np.random.randn(n) * 10 + 95,
        'close': np.random.randn(n) * 10 + 100,
        'volume': np.random.rand(n) * 1000000,
        'volume_ccy': np.random.rand(n) * 10000,
    }
    
    return pl.DataFrame(data)

@pytest.fixture
def sample_tick_data():
    """示例tick数据fixture"""
    import numpy as np
    
    dtype = np.dtype([
        ('timestamp', 'f8'),
        ('ask1_price', 'f8'),
        ('ask1_quantity', 'f8'),
        ('bid1_price', 'f8'),
        ('bid1_quantity', 'f8'),
        ('ask5_price', 'f8'),
        ('ask5_quantity', 'f8'),
        ('bid5_price', 'f8'),
        ('bid5_quantity', 'f8')
    ])
    
    n = 50
    data = np.array([
        (
            float(i),
            100.0 + i * 0.1,
            1000.0,
            99.9 + i * 0.1,
            1000.0,
            100.5 + i * 0.1,
            500.0,
            99.4 + i * 0.1,
            500.0
        )
        for i in range(n)
    ], dtype=dtype)
    
    return data

def pytest_configure(config):
    """pytest配置钩子"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
