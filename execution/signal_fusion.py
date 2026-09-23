import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger

class SignalFusion:
    """信号融合模块 - 结合多个信号源和市场状态生成最终信号"""
    
    def __init__(self):
        # 信号源权重
        self.source_weights = {
            'quant_core': 0.6,    # 量化核心模型
            'regime_based': 0.3,  # 基于市场状态的规则
            'risk_adjustment': 0.1,  # 风险调整
        }
        
        # 市场状态特定规则
        self.regime_rules = {
            'trending_up': {
                'bias': 'LONG',  # 趋势向上时有做多倾向
                'position_multiplier': 1.2,  # 仓位乘数
                'confidence_boost': 0.1,  # 置信度提升
            },
            'trending_down': {
                'bias': 'SHORT',
                'position_multiplier': 1.2,
                'confidence_boost': 0.1,
            },
            'ranging': {
                'bias': 'NEUTRAL',
                'position_multiplier': 0.8,  # 震荡市减仓
                'confidence_boost': -0.1,  # 置信度降低
            },
            'high_vol': {
                'bias': 'NEUTRAL',
                'position_multiplier': 0.5,  # 高波动市大幅减仓
                'confidence_boost': -0.2,
            },
            'high_risk': {
                'bias': 'NEUTRAL',
                'position_multiplier': 0.3,  # 高风险市极低仓位
                'confidence_boost': -0.3,
            }
        }
        
        # 信号历史
        self.signal_history = []
        self.max_history_size = 100
        
        logger.info("SignalFusion initialized")
    
    def fuse(self, quant_signal: Dict[str, Any], 
             regime: str, 
             current_position: float,
             risk_level: str = 'normal') -> Dict[str, Any]:
        """
        融合信号
        
        Args:
            quant_signal: 量化核心信号
            regime: 市场状态
            current_position: 当前仓位 (-1到1)
            risk_level: 风险等级 (normal, high, extreme)
        """
        try:
            # 1. 获取量化信号
            quant_score = self._extract_quant_score(quant_signal)
            
            # 2. 获取基于市场状态的信号
            regime_score = self._get_regime_based_score(regime)
            
            # 3. 获取风险调整信号
            risk_score = self._get_risk_adjustment(current_position, risk_level)
            
            # 4. 加权融合
            final_score = (
                self.source_weights['quant_core'] * quant_score +
                self.source_weights['regime_based'] * regime_score +
                self.source_weights['risk_adjustment'] * risk_score
            )
            
            # 5. 应用市场状态特定规则
            final_signal = self._apply_regime_rules(
                final_score, regime, current_position, quant_signal
            )
            
            # 6. 限制信号强度
            final_signal = self._limit_signal_strength(final_signal, risk_level)
            
            # 记录历史
            self._record_signal(final_signal, quant_signal, regime)
            
            logger.debug(f"Signal fused: {final_signal['direction']} "
                        f"(score: {final_score:.3f}, confidence: {final_signal['confidence']:.1%})")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error fusing signal: {e}")
            return self._get_neutral_signal()
    
    def _extract_quant_score(self, quant_signal: Dict[str, Any]) -> float:
        """从量化信号中提取分数"""
        try:
            if not quant_signal:
                return 0.0
            
            # 量化信号通常有score字段
            if 'score' in quant_signal:
                return quant_signal['score']
            
            # 或者从direction和confidence推断
            elif 'direction' in quant_signal:
                direction = quant_signal['direction']
                confidence = quant_signal.get('confidence', 0.5)
                
                if direction == 'LONG':
                    return confidence
                elif direction == 'SHORT':
                    return -confidence
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error extracting quant score: {e}")
            return 0.0
    
    def _get_regime_based_score(self, regime: str) -> float:
        """基于市场状态生成信号分数"""
        try:
            # 根据市场状态给出偏向性信号
            regime_scores = {
                'trending_up': 0.3,      # 趋势向上时轻微做多偏向
                'trending_down': -0.3,   # 趋势向下时轻微做空偏向
                'ranging': 0.0,          # 震荡市无偏向
                'high_vol': -0.1,        # 高波动市轻微避险偏向
                'high_risk': -0.2,       # 高风险市较强避险偏向
                'liquid': 0.05,          # 高流动性市轻微积极
                'illiquid': -0.1,        # 低流动性市避险
            }
            
            return regime_scores.get(regime, 0.0)
            
        except Exception as e:
            logger.error(f"Error getting regime based score: {e}")
            return 0.0
    
    def _get_risk_adjustment(self, current_position: float, risk_level: str) -> float:
        """基于当前仓位和风险等级的风险调整"""
        try:
            # 仓位过重时，反向调整
            position_adjustment = -current_position * 0.2  # 仓位越重，反向信号越强
            
            # 风险等级调整
            risk_adjustments = {
                'normal': 0.0,
                'high': -0.1,
                'extreme': -0.3,
            }
            
            risk_adjustment = risk_adjustments.get(risk_level, 0.0)
            
            return position_adjustment + risk_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating risk adjustment: {e}")
            return 0.0
    
    def _apply_regime_rules(self, score: float, regime: str, 
                           current_position: float, 
                           original_signal: Dict[str, Any]) -> Dict[str, Any]:
        """应用市场状态特定规则"""
        try:
            # 获取该状态的规则
            rules = self.regime_rules.get(regime, {})
            
            # 确定方向
            if score > 0.1:
                direction = 'LONG'
                confidence = min(1.0, score)
            elif score < -0.1:
                direction = 'SHORT'
                confidence = min(1.0, -score)
            else:
                direction = 'NEUTRAL'
                confidence = 0.0
            
            # 应用状态偏向
            bias = rules.get('bias', 'NEUTRAL')
            if bias != 'NEUTRAL':
                if direction == bias:
                    confidence_boost = rules.get('confidence_boost', 0)
                    confidence = min(1.0, confidence + confidence_boost)
                elif direction != 'NEUTRAL' and direction != bias:
                    # 与状态偏向相反，降低置信度
                    confidence_reduction = rules.get('confidence_boost', 0)
                    confidence = max(0.0, confidence - abs(confidence_reduction))
            
            # 计算仓位大小
            position_multiplier = rules.get('position_multiplier', 1.0)
            base_position = confidence * 0.5  # 基础仓位为置信度的一半
            
            # 考虑当前仓位
            if (direction == 'LONG' and current_position > 0.5) or \
               (direction == 'SHORT' and current_position < -0.5):
                # 同向仓位过重，减仓
                position_size = base_position * 0.5
            else:
                position_size = base_position * position_multiplier
            
            # 构建最终信号
            final_signal = {
                'direction': direction,
                'score': score,
                'confidence': confidence,
                'position_size': min(1.0, max(0.0, position_size)),  # 限制在0-1
                'regime': regime,
                'original_signal': original_signal,
                'fused_at': datetime.now().isoformat(),
                'position_multiplier': position_multiplier,
                'current_position': current_position
            }
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error applying regime rules: {e}")
            return self._get_neutral_signal()
    
    def _limit_signal_strength(self, signal: Dict[str, Any], risk_level: str) -> Dict[str, Any]:
        """根据风险等级限制信号强度"""
        try:
            if risk_level == 'high':
                # 高风险时减半仓位
                signal['position_size'] *= 0.5
                signal['confidence'] *= 0.8
            elif risk_level == 'extreme':
                # 极端风险时只允许极低仓位
                signal['position_size'] *= 0.2
                signal['confidence'] *= 0.5
                
                # 强制转向中性如果信号太强
                if abs(signal['score']) > 0.5:
                    signal['direction'] = 'NEUTRAL'
                    signal['position_size'] = 0
            
            return signal
            
        except Exception as e:
            logger.error(f"Error limiting signal strength: {e}")
            return signal
    
    def _record_signal(self, fused_signal: Dict[str, Any], 
                      original_signal: Dict[str, Any], 
                      regime: str):
        """记录信号历史"""
        try:
            record = {
                'timestamp': datetime.now(),
                'fused_signal': fused_signal.copy(),
                'original_signal': original_signal.copy() if original_signal else {},
                'regime': regime,
                'final_direction': fused_signal['direction'],
                'final_confidence': fused_signal['confidence']
            }
            
            self.signal_history.append(record)
            
            # 限制历史大小
            if len(self.signal_history) > self.max_history_size:
                self.signal_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error recording signal: {e}")
    
    def _get_neutral_signal(self) -> Dict[str, Any]:
        """获取中性信号"""
        return {
            'direction': 'NEUTRAL',
            'score': 0.0,
            'confidence': 0.0,
            'position_size': 0.0,
            'regime': 'unknown',
            'fused_at': datetime.now().isoformat()
        }
    
    def get_signal_history(self, n: int = 10) -> list:
        """获取最近的信号历史"""
        return self.signal_history[-n:] if self.signal_history else []
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        if not self.signal_history:
            return {}
        
        # 计算各种信号方向的比例
        directions = [s['final_direction'] for s in self.signal_history]
        total = len(directions)
        
        long_count = directions.count('LONG')
        short_count = directions.count('SHORT')
        neutral_count = directions.count('NEUTRAL')
        
        return {
            'total_signals': total,
            'long_percentage': long_count / total if total > 0 else 0,
            'short_percentage': short_count / total if total > 0 else 0,
            'neutral_percentage': neutral_count / total if total > 0 else 0,
            'avg_confidence': np.mean([s['final_confidence'] for s in self.signal_history]) if self.signal_history else 0
        }