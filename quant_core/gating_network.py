#!/usr/bin/env python3
"""动态门控网络 - 根据市场状态调整专家权重"""

import copy
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger
import json
from pathlib import Path

class GatingNetwork:
    """动态门控网络（在线学习专家权重）"""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        
        # 专家表现记忆
        self.performance_memory = {
            'trending': {'hunter': [], 'sniper': [], 'defender': []},
            'ranging': {'hunter': [], 'sniper': [], 'defender': []},
            'high_vol': {'hunter': [], 'sniper': [], 'defender': []},
            'crash': {'hunter': [], 'sniper': [], 'defender': []}
        }
        
        # 基础权重（架构设计中定义的）
        self.base_weights = {
            'trending': {'hunter': 0.7, 'sniper': 0.3, 'defender': 0.0},
            'ranging': {'hunter': 0.2, 'sniper': 0.8, 'defender': 0.0},
            'high_vol': {'hunter': 0.0, 'sniper': 0.0, 'defender': 1.0},
            'crash': {'hunter': 0.0, 'sniper': 0.0, 'defender': 1.0}
        }
        
        # 当前权重（会根据学习调整）。必须深拷贝：base_weights 作为衰减锚点，
        # 浅拷贝会让内层 regime 字典与 base 共享对象，在线更新时把锚点一并改坏。
        self.current_weights = copy.deepcopy(self.base_weights)
        
        # 学习参数
        self.learning_rate = 0.1
        self.decay_factor = 0.99  # 权重衰减到基础权重
        
        logger.info("GatingNetwork initialized with online learning capability")
    
    def update_performance(self, regime: str, expert: str, performance: float):
        """更新专家表现记录"""
        try:
            if regime in self.performance_memory and expert in self.performance_memory[regime]:
                memory = self.performance_memory[regime][expert]
                memory.append(performance)
                
                # 限制记忆大小
                if len(memory) > self.memory_size:
                    memory.pop(0)
                
                logger.debug(f"Updated {regime}.{expert} performance: {performance:.3f}")
            else:
                logger.warning(f"Invalid regime or expert: {regime}.{expert}")
                
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def calculate_expert_performance(self, regime: str, expert: str) -> float:
        """计算专家在特定状态下的平均表现"""
        try:
            if regime in self.performance_memory and expert in self.performance_memory[regime]:
                performances = self.performance_memory[regime][expert]
                if performances:
                    # 使用指数加权平均，更重视最近的表现
                    weights = np.exp(np.linspace(0, -1, len(performances)))
                    weights /= weights.sum()
                    weighted_avg = np.average(performances, weights=weights)
                    return float(weighted_avg)
            
            # 如果没有历史数据，返回中性表现
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating expert performance: {e}")
            return 0.5
    
    def update_weights(self, regime: str):
        """根据专家表现更新权重"""
        try:
            if regime not in self.current_weights:
                logger.warning(f"Unknown regime: {regime}")
                return
            
            # 计算各个专家的表现
            performances = {}
            for expert in self.current_weights[regime]:
                performances[expert] = self.calculate_expert_performance(regime, expert)
            
            # 如果所有专家表现都未知，使用基础权重
            if all(p == 0.5 for p in performances.values()):
                logger.debug(f"No performance data for {regime}, using base weights")
                return
            
            # 归一化表现作为权重调整基础
            total_perf = sum(performances.values())
            if total_perf > 0:
                # 计算softmax权重
                exp_perf = {e: np.exp(p) for e, p in performances.items()}
                sum_exp = sum(exp_perf.values())
                softmax_weights = {e: exp_perf[e]/sum_exp for e in performances}
                
                # 混合基础权重和学习到的权重
                for expert in self.current_weights[regime]:
                    base_weight = self.base_weights[regime][expert]
                    learned_weight = softmax_weights[expert]
                    
                    # 线性插值：学习率控制从基础权重到学习权重的转换
                    new_weight = (1 - self.learning_rate) * base_weight + \
                                self.learning_rate * learned_weight
                    
                    self.current_weights[regime][expert] = new_weight
                
                # 归一化确保权重和为1
                total_weight = sum(self.current_weights[regime].values())
                if total_weight > 0:
                    for expert in self.current_weights[regime]:
                        self.current_weights[regime][expert] /= total_weight
                
                logger.info(f"Updated weights for {regime}: {self.current_weights[regime]}")
                
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
    
    def get_weights(self, regime: str) -> Dict[str, float]:
        """获取指定状态下的专家权重"""
        try:
            if regime in self.current_weights:
                return self.current_weights[regime].copy()
            else:
                logger.warning(f"Unknown regime: {regime}, using trending weights")
                return self.current_weights['trending'].copy()
        except Exception as e:
            logger.error(f"Error getting weights: {e}")
            return {'hunter': 0.5, 'sniper': 0.5, 'defender': 0.0}
    
    def decay_to_base(self):
        """权重衰减回基础权重（防止过拟合）"""
        try:
            for regime in self.current_weights:
                for expert in self.current_weights[regime]:
                    current = self.current_weights[regime][expert]
                    base = self.base_weights[regime][expert]
                    
                    # 指数衰减回基础权重
                    self.current_weights[regime][expert] = \
                        current * self.decay_factor + base * (1 - self.decay_factor)
            
            logger.debug(f"Weights decayed (factor: {self.decay_factor})")
            
        except Exception as e:
            logger.error(f"Error decaying weights: {e}")
    
    def evaluate_signal_performance(self, regime: str, expert: str, 
                                   signal: str, actual_return: float) -> float:
        """
        评估信号表现
        
        Args:
            signal: 'LONG', 'SHORT', 'NEUTRAL'
            actual_return: 实际收益率
        """
        try:
            if signal == 'NEUTRAL':
                # 中性信号的表现：如果市场波动大，持有现金是好的
                volatility_score = 1.0 if abs(actual_return) < 0.001 else 0.0
                return volatility_score
            
            elif signal == 'LONG':
                # 做多信号：实际收益为正则表现好
                return max(0.0, min(1.0, actual_return * 10 + 0.5))
            
            elif signal == 'SHORT':
                # 做空信号：实际收益为负则表现好
                return max(0.0, min(1.0, -actual_return * 10 + 0.5))
            
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error evaluating signal performance: {e}")
            return 0.5
    
    def save_state(self, filepath: str = "./data/models/gating_network_state.json"):
        """保存门控网络状态"""
        try:
            state = {
                'current_weights': self.current_weights,
                'performance_memory': self.performance_memory,
                'saved_at': datetime.now().isoformat()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Gating network state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving gating network state: {e}")
    
    def load_state(self, filepath: str = "./data/models/gating_network_state.json"):
        """加载门控网络状态"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.current_weights = state.get('current_weights', self.base_weights)
                self.performance_memory = state.get('performance_memory', self.performance_memory)
                
                logger.info(f"Gating network state loaded from {filepath}")
                return True
            else:
                logger.info(f"No saved state found at {filepath}, using default")
                return False
                
        except Exception as e:
            logger.error(f"Error loading gating network state: {e}")
            return False

def test_gating_network():
    """测试门控网络"""
    logger.info("Testing Gating Network...")
    
    gating = GatingNetwork()
    
    # 模拟一些性能数据
    test_cases = [
        ('trending', 'hunter', 0.8),
        ('trending', 'sniper', 0.6),
        ('trending', 'defender', 0.3),
        ('ranging', 'hunter', 0.4),
        ('ranging', 'sniper', 0.9),
        ('ranging', 'defender', 0.5),
    ]
    
    for regime, expert, performance in test_cases:
        for _ in range(10):  # 添加多个数据点
            gating.update_performance(regime, expert, performance + np.random.normal(0, 0.1))
    
    # 更新权重
    for regime in ['trending', 'ranging']:
        gating.update_weights(regime)
        weights = gating.get_weights(regime)
        logger.info(f"{regime} weights: {weights}")
    
    # 测试信号表现评估
    test_signals = [
        ('LONG', 0.02, 0.7),  # 做多，收益2%，期望表现0.7
        ('SHORT', -0.01, 0.6),  # 做空，收益-1%，期望表现0.6
        ('NEUTRAL', 0.0, 1.0),  # 中性，无波动，期望表现1.0
    ]
    
    for signal, return_, expected in test_signals:
        perf = gating.evaluate_signal_performance('trending', 'hunter', signal, return_)
        logger.info(f"Signal {signal} with return {return_:.2%}: performance={perf:.2f}")
    
    return True

if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}")
    
    test_gating_network()