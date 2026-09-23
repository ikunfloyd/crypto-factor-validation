import pickle
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import shutil

class DisasterRecovery:
    """灾难恢复管理器"""
    
    def __init__(self, checkpoint_dir: str = "./data/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查点文件名模板
        self.checkpoint_prefix = "checkpoint_v5"
        
    def create_checkpoint(self, state_data: Dict[str, Any], 
                          description: str = "") -> str:
        """
        创建系统检查点
        
        Args:
            state_data: 系统状态数据
            description: 检查点描述
        """
        try:
            # 生成时间戳
            timestamp = time.time()
            timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            
            # 检查点文件名
            checkpoint_name = f"{self.checkpoint_prefix}_{timestamp_str}.pkl"
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            # 准备检查点数据
            checkpoint_data = {
                'timestamp': timestamp,
                'description': description,
                'state': state_data,
                'version': '5.0'
            }
            
            # 保存检查点
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 更新最新检查点标记
            latest_marker = self.checkpoint_dir / "latest_checkpoint.txt"
            with open(latest_marker, 'w') as f:
                f.write(str(timestamp))
            
            logger.info(f"Checkpoint created: {checkpoint_name}")
            
            # 清理旧的检查点（保留最近5个）
            self._cleanup_old_checkpoints(keep_count=5)
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径，如果为None则加载最新的
        """
        try:
            if checkpoint_path is None:
                # 查找最新的检查点
                checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*.pkl"))
                if not checkpoint_files:
                    raise FileNotFoundError("No checkpoint files found")
                
                # 按时间排序，获取最新的
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                checkpoint_path = checkpoint_files[0]
            
            checkpoint_path = Path(checkpoint_path)
            
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # 验证检查点版本
            if checkpoint_data.get('version') != '5.0':
                logger.warning(f"Checkpoint version mismatch: {checkpoint_data.get('version')}")
            
            logger.info(f"Checkpoint loaded (timestamp: {checkpoint_data['timestamp']})")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _cleanup_old_checkpoints(self, keep_count: int = 5):
        """清理旧的检查点，只保留最近的几个"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*.pkl"))
            
            if len(checkpoint_files) > keep_count:
                # 按时间排序
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
                
                # 删除旧的检查点
                files_to_delete = checkpoint_files[:-keep_count]
                for file_path in files_to_delete:
                    file_path.unlink()
                    logger.debug(f"Deleted old checkpoint: {file_path.name}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {e}")
    
    def backup_wal_data(self, backup_dir: str = "./data/backups"):
        """备份WAL数据"""
        try:
            wal_dir = Path("./data/wal")
            backup_path = Path(backup_dir) / f"wal_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if wal_dir.exists():
                # 创建备份目录
                backup_path.mkdir(parents=True, exist_ok=True)
                
                # 复制WAL文件
                for item in wal_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, backup_path / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, backup_path / item.name)
                
                logger.info(f"WAL data backed up to: {backup_path}")
                return str(backup_path)
                
        except Exception as e:
            logger.error(f"Failed to backup WAL data: {e}")
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """获取恢复状态"""
        try:
            status = {
                'last_checkpoint': None,
                'checkpoint_count': 0,
                'recovery_available': False
            }
            
            # 检查检查点文件
            checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*.pkl"))
            status['checkpoint_count'] = len(checkpoint_files)
            
            if checkpoint_files:
                # 获取最新的检查点
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_checkpoint = checkpoint_files[0]
                
                with open(latest_checkpoint, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    status['last_checkpoint'] = checkpoint_data['timestamp']
                    status['recovery_available'] = True
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting recovery status: {e}")
            return {'error': str(e)}