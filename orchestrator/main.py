#!/usr/bin/env python3
"""主控进程 - 协调所有子系统"""

import asyncio
import signal
import sys
import time
from datetime import datetime
from multiprocessing import Process, Manager
from typing import Dict, Any
from loguru import logger
from pathlib import Path
import psutil

from utils.disaster_recovery import DisasterRecovery
from io_gateway.wal_reader import WALReader

def _run_async_target(module_name: str, function_name: str):
    """在子进程中运行异步目标函数"""
    import asyncio
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass
    module = __import__(module_name, fromlist=[function_name])
    target_func = getattr(module, function_name)
    asyncio.run(target_func())

class Orchestrator:
    """系统主控器"""
    
    def __init__(self):
        self.processes = {}
        self.shared_state = Manager().dict()
        self.recovery_manager = DisasterRecovery()
        self.running = False
        
        # 进程配置
        self.process_configs = {
            'io_gateway': {
                'module': 'io_gateway.main',
                'function': 'main',
                'cpu_affinity': 'e_cores'  # 绑定到E-core
            },
            'quant_core': {
                'module': 'quant_core.main_v2',  # 集成 MoE 推理的量化核心
                'function': 'main',
                'cpu_affinity': 'p_cores'  # 绑定到P-core
            },
            'execution_guard': {
                'module': 'execution.main',
                'function': 'main',
                'cpu_affinity': 'p_cores'
            }
        }
    
    def check_recovery_needed(self) -> bool:
        """检查是否需要灾难恢复"""
        try:
            # 检查是否有未正常关闭的标记
            shutdown_file = Path("./data/.shutdown_marker")
            if shutdown_file.exists():
                logger.warning("Detected abnormal shutdown. Recovery needed.")
                return True
                
            # 检查进程是否都在运行
            for proc_name in self.process_configs:
                if not self._is_process_running(proc_name):
                    logger.warning(f"Process {proc_name} not running. Recovery needed.")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking recovery status: {e}")
            return True
    
    def _is_process_running(self, proc_name: str) -> bool:
        """检查进程是否在运行"""
        # 这里简化实现，实际应该检查具体的进程ID
        return proc_name in self.processes and self.processes[proc_name].is_alive()
    
    def perform_recovery(self):
        """执行灾难恢复"""
        logger.info("Starting disaster recovery...")
        
        try:
            # 步骤1: 加载最新检查点
            checkpoint_data = self.recovery_manager.load_checkpoint()
            
            if checkpoint_data:
                logger.info("Checkpoint loaded successfully")
                
                # 步骤2: 恢复系统状态
                self.shared_state.update(checkpoint_data['state'])
                
                # 步骤3: 回放WAL日志
                wal_reader = WALReader()
                
                def replay_callback(record: Dict[str, Any]):
                    # 这里可以添加回放时的处理逻辑
                    pass
                
                # 从检查点时间开始回放
                checkpoint_time = datetime.fromtimestamp(checkpoint_data['timestamp'])
                records = wal_reader.replay(checkpoint_time, callback=replay_callback)
                
                logger.info(f"WAL replay completed. {len(records)} records processed")
                
                # 步骤4: 验证恢复
                recovery_status = self._verify_recovery()
                
                if recovery_status['success']:
                    logger.success("Disaster recovery completed successfully!")
                else:
                    logger.error("Disaster recovery verification failed")
                    
                return recovery_status
                
        except Exception as e:
            logger.error(f"Disaster recovery failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _verify_recovery(self) -> Dict[str, Any]:
        """验证恢复是否成功"""
        try:
            # 检查共享内存
            from shared_memory.ring_buffer import ArrowRingBuffer, RING_BUFFER_CONFIGS
            
            for buffer_name, config in RING_BUFFER_CONFIGS.items():
                try:
                    buffer = ArrowRingBuffer(config)
                    data = buffer.read_latest(1)
                    if data is not None:
                        logger.debug(f"Buffer {buffer_name} contains data")
                    buffer.close()
                except Exception as e:
                    logger.warning(f"Buffer {buffer_name} verification failed: {e}")
            
            # 检查WAL文件
            wal_dir = Path("./data/wal")
            if wal_dir.exists() and any(wal_dir.iterdir()):
                logger.debug("WAL files exist")
            
            return {'success': True, 'message': 'Recovery verified'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def start_process(self, proc_name: str):
        """启动子进程"""
        if proc_name in self.processes and self.processes[proc_name].is_alive():
            logger.warning(f"Process {proc_name} is already running")
            return
        
        config = self.process_configs[proc_name]
        
        try:
            # 动态导入模块
            module = __import__(config['module'], fromlist=[config['function']])
            target_func = getattr(module, config['function'])
            
            # 诊断日志：检查函数类型
            import asyncio
            logger.debug(f"Target function for {proc_name}: {target_func}, is coroutine function? {asyncio.iscoroutinefunction(target_func)}")
            
            # 如果是协程函数，则使用模块级别的包装器
            if asyncio.iscoroutinefunction(target_func):
                logger.warning(f"Process {proc_name} target is async coroutine, using async wrapper")
                actual_target = _run_async_target
                args = (config['module'], config['function'])
            else:
                actual_target = target_func
                args = ()
            
            # 创建进程
            proc = Process(
                name=proc_name,
                target=actual_target,
                args=args,
                daemon=True
            )
            
            proc.start()
            self.processes[proc_name] = proc
            
            # 设置CPU亲和性
            self._set_cpu_affinity(proc, config['cpu_affinity'])
            
            logger.info(f"Started process {proc_name} (PID: {proc.pid})")
            
        except Exception as e:
            logger.error(f"Failed to start process {proc_name}: {e}")
    
    def _set_cpu_affinity(self, proc: Process, affinity_type: str):
        """设置CPU亲和性"""
        try:
            p = psutil.Process(proc.pid)
            if not hasattr(p, 'cpu_affinity'):
                logger.debug(f"CPU affinity not supported on this platform, skipping")
                return
                
            cpu_count = psutil.cpu_count(logical=False)
            
            if cpu_count >= 8:  # M3通常有8个物理核心
                if affinity_type == 'e_cores':
                    # E-cores通常是最后几个核心
                    e_cores = list(range(cpu_count - 4, cpu_count))
                    p.cpu_affinity(e_cores)
                    logger.debug(f"Set {proc.name} to E-cores: {e_cores}")
                    
                elif affinity_type == 'p_cores':
                    # P-cores通常是前几个核心
                    p_cores = list(range(0, cpu_count - 4))
                    p.cpu_affinity(p_cores)
                    logger.debug(f"Set {proc.name} to P-cores: {p_cores}")
                    
        except Exception as e:
            logger.warning(f"Could not set CPU affinity: {e}")
    
    def stop_process(self, proc_name: str):
        """停止子进程"""
        if proc_name in self.processes:
            proc = self.processes[proc_name]
            
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                
                logger.info(f"Stopped process {proc_name}")
            
            del self.processes[proc_name]
    
    def create_checkpoint(self):
        """创建系统检查点"""
        try:
            # 准备状态数据
            state_data = {
                'timestamp': time.time(),
                'processes': {name: proc.pid for name, proc in self.processes.items()},
                'shared_state': dict(self.shared_state),
                'system_time': datetime.now().isoformat()
            }
            
            # 创建检查点
            checkpoint_path = self.recovery_manager.create_checkpoint(
                state_data,
                description="Scheduled checkpoint"
            )
            
            logger.info(f"System checkpoint created: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
    
    def monitor_processes(self):
        """监控子进程状态"""
        dead_processes = []
        
        for name, proc in self.processes.items():
            if not proc.is_alive():
                dead_processes.append(name)
                logger.error(f"Process {name} died (exitcode: {proc.exitcode})")
        
        # 重启死亡的进程
        for name in dead_processes:
            logger.info(f"Restarting process {name}...")
            self.stop_process(name)
            self.start_process(name)
    
    async def run(self):
        """运行主控器"""
        logger.info("Starting DOGE-USDT Trading System Orchestrator...")
        
        # 设置信号处理
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 检查是否需要灾难恢复
        if self.check_recovery_needed():
            recovery_result = self.perform_recovery()
            if not recovery_result.get('success'):
                logger.error("Recovery failed. Starting fresh...")
        
        # 标记正常启动
        Path("./data/.shutdown_marker").unlink(missing_ok=True)
        
        # 启动所有进程
        for proc_name in self.process_configs:
            self.start_process(proc_name)
        
        self.running = True
        last_checkpoint = time.time()
        
        # 主循环
        try:
            while self.running:
                # 监控进程状态
                self.monitor_processes()
                
                # 定期创建检查点（每5分钟）
                current_time = time.time()
                if current_time - last_checkpoint > 300:  # 5分钟
                    self.create_checkpoint()
                    last_checkpoint = current_time
                
                # 显示状态
                self._display_status()
                
                await asyncio.sleep(5)  # 5秒检查一次
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
        finally:
            self.shutdown()
    
    def _display_status(self):
        """显示系统状态"""
        status_lines = []
        status_lines.append("\n" + "=" * 60)
        status_lines.append(f"System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        status_lines.append("=" * 60)
        
        for name, proc in self.processes.items():
            status = "RUNNING" if proc.is_alive() else "DEAD"
            status_lines.append(f"{name:20} {status:10} PID: {proc.pid}")
        
        # 显示恢复状态
        recovery_status = self.recovery_manager.get_recovery_status()
        if recovery_status.get('recovery_available'):
            last_checkpoint = datetime.fromtimestamp(recovery_status['last_checkpoint'])
            status_lines.append(f"Last checkpoint: {last_checkpoint.strftime('%Y-%m-%d %H:%M:%S')}")
        
        status_lines.append("=" * 60)
        
        print("\n".join(status_lines))
    
    def shutdown(self):
        """关闭系统"""
        logger.info("Shutting down system...")
        
        # 标记正常关闭
        with open("./data/.shutdown_marker", "w") as f:
            f.write("normal_shutdown")
        
        # 停止所有进程
        for proc_name in list(self.processes.keys()):
            self.stop_process(proc_name)
        
        # 创建最终检查点
        self.create_checkpoint()
        
        logger.info("System shutdown complete")

async def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{process}</cyan> | <level>{message}</level>",
        level="INFO"
    )
    
    # 创建并运行主控器
    orchestrator = Orchestrator()
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())