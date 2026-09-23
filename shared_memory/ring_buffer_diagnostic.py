"""
Ring Buffer 诊断脚本
用于诊断共享内存数据访问问题
"""

import sys
from pathlib import Path
import time

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared_memory.ring_buffer import ArrowRingBuffer, RING_BUFFER_CONFIGS
from utils.logger_config import get_logger

logger = get_logger(__name__)

def diagnose_ring_buffer(buffer_name='candle', duration_seconds=30):
    """
    诊断环形缓冲区的访问模式
    
    Args:
        buffer_name: 缓冲区名称 ('candle' 或 'fast_tick')
        duration_seconds: 监控时长（秒）
    """
    logger.info(f"开始诊断 {buffer_name} 缓冲区，监控时长：{duration_seconds}秒")
    logger.info("=" * 60)
    
    try:
        # 连接到共享内存
        config = RING_BUFFER_CONFIGS[buffer_name]
        buffer = ArrowRingBuffer(config, cleanup_on_exit=False)
        
        # 记录初始状态
        initial_write_idx = buffer.write_index
        initial_read_idx = buffer.read_index
        
        logger.info(f"[初始状态]")
        logger.info(f"  写入索引 (write_idx): {initial_write_idx}")
        logger.info(f"  读取索引 (read_idx):  {initial_read_idx}")
        logger.info(f"  可用数据量: {_calc_available(initial_write_idx, initial_read_idx, config.capacity)}")
        logger.info("")
        
        # 周期性监控索引变化
        start_time = time.time()
        last_write_idx = initial_write_idx
        last_read_idx = initial_read_idx
        check_count = 0
        
        while time.time() - start_time < duration_seconds:
            time.sleep(2)  # 每2秒检查一次
            check_count += 1
            
            current_write_idx = buffer.write_index
            current_read_idx = buffer.read_index
            
            # 检测变化
            write_changed = current_write_idx != last_write_idx
            read_changed = current_read_idx != last_read_idx
            
            if write_changed or read_changed:
                logger.info(f"[检查 #{check_count}] 时间: {time.time() - start_time:.1f}s")
                
                if write_changed:
                    write_delta = _calc_delta(last_write_idx, current_write_idx, config.capacity)
                    logger.warning(f"  ✓ 写入索引变化: {last_write_idx} → {current_write_idx} (增加 {write_delta} 条)")
                else:
                    logger.info(f"  ✗ 写入索引未变: {current_write_idx}")
                
                if read_changed:
                    read_delta = _calc_delta(last_read_idx, current_read_idx, config.capacity)
                    logger.warning(f"  ⚠️  读取索引变化: {last_read_idx} → {current_read_idx} (增加 {read_delta} 条)")
                    logger.warning(f"  ⚠️  有进程正在使用 read_latest() 消费数据！")
                else:
                    logger.info(f"  ✓ 读取索引未变: {current_read_idx}")
                
                available = _calc_available(current_write_idx, current_read_idx, config.capacity)
                logger.info(f"  当前可用数据量: {available}")
                logger.info("")
                
                last_write_idx = current_write_idx
                last_read_idx = current_read_idx
        
        # 最终报告
        final_write_idx = buffer.write_index
        final_read_idx = buffer.read_index
        
        logger.info("=" * 60)
        logger.info(f"[最终状态]")
        logger.info(f"  写入索引: {initial_write_idx} → {final_write_idx}")
        logger.info(f"  读取索引: {initial_read_idx} → {final_read_idx}")
        logger.info(f"  最终可用数据量: {_calc_available(final_write_idx, final_read_idx, config.capacity)}")
        logger.info("")
        
        # 诊断结论
        logger.info("[诊断结论]")
        
        write_delta = _calc_delta(initial_write_idx, final_write_idx, config.capacity)
        read_delta = _calc_delta(initial_read_idx, final_read_idx, config.capacity)
        
        if write_delta == 0:
            logger.error("✗ 问题1: 没有新数据写入！io_gateway 可能未运行或已停止")
        else:
            logger.info(f"✓ 数据写入正常：监控期间写入了 {write_delta} 条数据")
        
        if read_delta > 0:
            logger.error(f"✗ 问题2: 读取索引增加了 {read_delta} 条！")
            logger.error("  有进程正在使用 read_latest() 消费数据")
            logger.error("  这会导致 Dashboard 的 peek_latest() 读不到数据")
            logger.error("  嫌疑进程: quant_core, execution, orchestrator")
        else:
            logger.info("✓ 读取索引未变化，没有进程消费数据")
        
        if final_write_idx == final_read_idx:
            logger.error("✗ 问题3: write_idx == read_idx，缓冲区为空状态")
            logger.error("  这就是为什么 Dashboard 显示 'No candles available'")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"诊断过程出错: {e}")
        import traceback
        logger.error(traceback.format_exc())


def _calc_available(write_idx, read_idx, capacity):
    """计算可用数据量"""
    if write_idx >= read_idx:
        return write_idx - read_idx
    else:
        return capacity - read_idx + write_idx


def _calc_delta(old_idx, new_idx, capacity):
    """计算索引增量"""
    if new_idx >= old_idx:
        return new_idx - old_idx
    else:
        return capacity - old_idx + new_idx


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='诊断环形缓冲区访问问题')
    parser.add_argument('--buffer', choices=['candle', 'fast_tick'], default='candle',
                        help='要诊断的缓冲区')
    parser.add_argument('--duration', type=int, default=30,
                        help='监控时长（秒）')
    
    args = parser.parse_args()
    
    diagnose_ring_buffer(args.buffer, args.duration)
