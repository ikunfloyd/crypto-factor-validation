#!/usr/bin/env python3
"""进程间通信管理器"""

import multiprocessing as mp
from multiprocessing import Queue, Event, Lock
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from loguru import logger

# 类型检查时导入正确的类型
if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass, Lock as LockClass
    from multiprocessing.queues import Queue as QueueClass

class MessageType(Enum):
    """消息类型"""
    DATA = "data"
    SIGNAL = "signal"
    COMMAND = "command"
    STATUS = "status"
    ERROR = "error"

@dataclass
class IPCMessage:
    """IPC消息"""
    msg_type: MessageType
    source: str
    destination: str
    payload: Any
    timestamp: float

class IPCManager:
    """进程间通信管理器"""
    
    def __init__(self):
        """初始化IPC管理器"""
        # 消息队列：进程名 -> Queue
        self.queues: Dict[str, 'QueueClass'] = {}
        
        # 事件：用于进程同步
        self.events: Dict[str, 'EventClass'] = {}
        
        # 锁：用于资源同步
        self.locks: Dict[str, 'LockClass'] = {}
        
        # 注册的进程
        self.registered_processes: Dict[str, mp.Process] = {}
        
        # 消息处理器
        self.message_handlers: Dict[str, Callable] = {}
        
        logger.info("IPCManager initialized")
    
    def register_process(self, process_name: str, queue_size: int = 1000) -> 'QueueClass':
        """
        注册进程并创建对应的消息队列
        
        Args:
            process_name: 进程名称
            queue_size: 队列大小
            
        Returns:
            消息队列
        """
        try:
            if process_name in self.queues:
                logger.warning(f"Process {process_name} already registered")
                return self.queues[process_name]
            
            # 创建消息队列
            queue = Queue(maxsize=queue_size)
            self.queues[process_name] = queue
            
            # 创建事件
            self.events[process_name] = Event()
            
            # 创建锁
            self.locks[process_name] = Lock()
            
            logger.info(f"Registered process: {process_name}")
            
            return queue
            
        except Exception as e:
            logger.error(f"Error registering process {process_name}: {e}")
            return None
    
    def unregister_process(self, process_name: str):
        """注销进程"""
        try:
            if process_name in self.queues:
                del self.queues[process_name]
            if process_name in self.events:
                del self.events[process_name]
            if process_name in self.locks:
                del self.locks[process_name]
            if process_name in self.registered_processes:
                del self.registered_processes[process_name]
            
            logger.info(f"Unregistered process: {process_name}")
            
        except Exception as e:
            logger.error(f"Error unregistering process {process_name}: {e}")
    
    def send_message(self, 
                    source: str,
                    destination: str,
                    msg_type: MessageType,
                    payload: Any,
                    block: bool = True,
                    timeout: Optional[float] = None) -> bool:
        """
        发送消息
        
        Args:
            source: 源进程名
            destination: 目标进程名
            msg_type: 消息类型
            payload: 消息负载
            block: 是否阻塞
            timeout: 超时时间（秒）
            
        Returns:
            是否发送成功
        """
        try:
            if destination not in self.queues:
                logger.error(f"Destination process not registered: {destination}")
                return False
            
            import time
            message = IPCMessage(
                msg_type=msg_type,
                source=source,
                destination=destination,
                payload=payload,
                timestamp=time.time()
            )
            
            queue = self.queues[destination]
            
            if block:
                queue.put(message, timeout=timeout)
            else:
                queue.put_nowait(message)
            
            logger.debug(f"Message sent: {source} -> {destination} ({msg_type.value})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def receive_message(self,
                       process_name: str,
                       block: bool = True,
                       timeout: Optional[float] = None) -> Optional[IPCMessage]:
        """
        接收消息
        
        Args:
            process_name: 进程名
            block: 是否阻塞
            timeout: 超时时间（秒）
            
        Returns:
            消息对象
        """
        try:
            if process_name not in self.queues:
                logger.error(f"Process not registered: {process_name}")
                return None
            
            queue = self.queues[process_name]
            
            if block:
                message = queue.get(timeout=timeout)
            else:
                message = queue.get_nowait()
            
            logger.debug(f"Message received by {process_name} from {message.source}")
            
            return message
            
        except mp.queues.Empty:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    def set_event(self, event_name: str):
        """设置事件"""
        if event_name in self.events:
            self.events[event_name].set()
            logger.debug(f"Event set: {event_name}")
    
    def clear_event(self, event_name: str):
        """清除事件"""
        if event_name in self.events:
            self.events[event_name].clear()
            logger.debug(f"Event cleared: {event_name}")
    
    def wait_event(self, event_name: str, timeout: Optional[float] = None) -> bool:
        """等待事件"""
        if event_name in self.events:
            return self.events[event_name].wait(timeout=timeout)
        return False
    
    def acquire_lock(self, lock_name: str, block: bool = True, timeout: Optional[float] = None) -> bool:
        """获取锁"""
        if lock_name in self.locks:
            return self.locks[lock_name].acquire(block=block, timeout=timeout)
        return False
    
    def release_lock(self, lock_name: str):
        """释放锁"""
        if lock_name in self.locks:
            self.locks[lock_name].release()
            logger.debug(f"Lock released: {lock_name}")
    
    def broadcast_message(self,
                         source: str,
                         msg_type: MessageType,
                         payload: Any) -> int:
        """
        广播消息给所有进程
        
        Returns:
            成功发送的消息数
        """
        success_count = 0
        
        for process_name in self.queues.keys():
            if process_name != source:
                if self.send_message(source, process_name, msg_type, payload, block=False):
                    success_count += 1
        
        logger.debug(f"Broadcast from {source}: {success_count} messages sent")
        
        return success_count
    
    def get_queue_size(self, process_name: str) -> int:
        """获取队列大小"""
        if process_name in self.queues:
            return self.queues[process_name].qsize()
        return 0
    
    def cleanup(self):
        """清理所有资源"""
        logger.info("Cleaning up IPC resources...")
        
        # 清空所有队列
        for process_name, queue in self.queues.items():
            try:
                while not queue.empty():
                    queue.get_nowait()
            except:
                pass
        
        # 清除所有事件
        for event in self.events.values():
            event.clear()
        
        logger.info("IPC cleanup complete")

# 全局IPC管理器实例
_ipc_manager: Optional[IPCManager] = None

def get_ipc_manager() -> IPCManager:
    """获取全局IPC管理器"""
    global _ipc_manager
    if _ipc_manager is None:
        _ipc_manager = IPCManager()
    return _ipc_manager

def set_ipc_manager(manager: IPCManager):
    """设置全局IPC管理器"""
    global _ipc_manager
    _ipc_manager = manager

# 示例用法
if __name__ == "__main__":
    from utils.logger_config import setup_logger
    logger = setup_logger("ipc_test")
    
    # 创建IPC管理器
    ipc = IPCManager()
    
    # 注册进程
    ipc.register_process("process_A")
    ipc.register_process("process_B")
    
    # 发送消息
    success = ipc.send_message(
        source="process_A",
        destination="process_B",
        msg_type=MessageType.DATA,
        payload={"data": [1, 2, 3]},
        block=False
    )
    
    print(f"Message sent: {success}")
    
    # 接收消息
    message = ipc.receive_message("process_B", block=False)
    
    if message:
        print(f"Received message from {message.source}")
        print(f"Type: {message.msg_type.value}")
        print(f"Payload: {message.payload}")
    
    # 清理
    ipc.cleanup()
