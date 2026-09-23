import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger

class RiskManager:
    """风险管理系统 - 监控和控制交易风险"""
    
    def __init__(self):
        # 风险阈值
        self.risk_thresholds = {
            'max_position_pct': 0.8,  # 最大仓位百分比
            'max_daily_loss_pct': 0.05,  # 单日最大亏损百分比
            'max_consecutive_losses': 3,  # 最大连续亏损次数
            'max_drawdown_pct': 0.1,  # 最大回撤百分比
            'min_confidence': 0.3,  # 最小信号置信度
            'max_slippage_pct': 0.005,  # 最大允许滑点百分比
            'max_spread_pct': 0.001,  # 最大允许价差百分比
        }
        
        # 风险状态
        self.risk_level = 'normal'  # normal, high, extreme
        self.last_rejection_reason = None
        
        # 风险历史
        self.risk_history = []
        self.trade_history = []
        self.max_history_size = 1000
        
        # 风险监控变量
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.peak_equity = 100000.0  # 初始权益（示例）
        self.current_equity = 100000.0
        
        logger.info("RiskManager initialized")
    
    def check(self, signal: Dict[str, Any], 
              current_position: float,
              pnl_history: List[Dict[str, Any]]) -> bool:
        """
        检查信号风险
        
        Args:
            signal: 交易信号
            current_position: 当前仓位
            pnl_history: 盈亏历史
            
        Returns:
            bool: 是否通过风险检查
        """
        try:
            # 重置拒绝原因
            self.last_rejection_reason = None
            
            # 1. 检查基本信号有效性
            if not self._check_signal_validity(signal):
                return False
            
            # 2. 检查仓位风险
            if not self._check_position_risk(signal, current_position):
                return False
            
            # 3. 检查市场条件风险
            if not self._check_market_risk(signal):
                return False
            
            # 4. 检查执行风险
            if not self._check_execution_risk(signal):
                return False
            
            # 5. 检查账户风险
            if not self._check_account_risk(pnl_history):
                return False
            
            # 6. 检查系统性风险
            if not self._check_systemic_risk():
                return False
            
            # 所有检查通过
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            self.last_rejection_reason = f"Risk check error: {e}"
            return False
    
    def _check_signal_validity(self, signal: Dict[str, Any]) -> bool:
        """检查信号有效性"""
        try:
            # 检查信号方向
            direction = signal.get('direction', 'NEUTRAL')
            if direction == 'NEUTRAL':
                self.last_rejection_reason = "Neutral signal"
                return True  # 中性信号允许通过
            
            # 检查置信度
            confidence = signal.get('confidence', 0.0)
            if confidence < self.risk_thresholds['min_confidence']:
                self.last_rejection_reason = f"Low confidence: {confidence:.1%} < {self.risk_thresholds['min_confidence']:.1%}"
                return False
            
            # 检查信号完整性
            required_fields = ['direction', 'confidence', 'position_size']
            for field in required_fields:
                if field not in signal:
                    self.last_rejection_reason = f"Missing field: {field}"
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal validity: {e}")
            return False
    
    def _check_position_risk(self, signal: Dict[str, Any], current_position: float) -> bool:
        """检查仓位风险"""
        try:
            direction = signal.get('direction', 'NEUTRAL')
            position_size = signal.get('position_size', 0.0)
            
            if direction == 'NEUTRAL':
                return True
            
            # 计算新仓位
            if direction == 'LONG':
                new_position = current_position + position_size
            elif direction == 'SHORT':
                new_position = current_position - position_size
            else:
                new_position = current_position
            
            # 检查是否超过最大仓位
            max_position = self.risk_thresholds['max_position_pct']
            if abs(new_position) > max_position:
                self.last_rejection_reason = f"Position too large: {new_position:.1%} > {max_position:.1%}"
                return False
            
            # 检查仓位反转风险（从多转空或空转多）
            if (current_position > 0.1 and direction == 'SHORT') or \
               (current_position < -0.1 and direction == 'LONG'):
                # 仓位反转需要更高的置信度
                confidence = signal.get('confidence', 0.0)
                if confidence < 0.6:  # 反转需要60%以上置信度
                    self.last_rejection_reason = f"Position reversal requires higher confidence: {confidence:.1%} < 60%"
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position risk: {e}")
            return False
    
    def _check_market_risk(self, signal: Dict[str, Any]) -> bool:
        """检查市场条件风险"""
        try:
            regime = signal.get('regime', 'unknown')
            
            # 高风险市场状态
            high_risk_regimes = ['high_vol', 'high_risk', 'illiquid', 'crash']
            
            if regime in high_risk_regimes:
                # 高风险市场需要减仓
                position_size = signal.get('position_size', 0.0)
                if position_size > 0.2:  # 高风险市场最大仓位20%
                    self.last_rejection_reason = f"High risk regime {regime}: position size {position_size:.1%} > 20%"
                    return False
            
            # 检查资金费率（如果信号中有）
            funding_rate = signal.get('funding_rate', 0)
            if abs(funding_rate) > 0.001:  # 资金费率绝对值 > 0.1%
                # 高资金费率时，反向交易风险高
                direction = signal.get('direction', 'NEUTRAL')
                if (funding_rate > 0 and direction == 'LONG') or \
                   (funding_rate < 0 and direction == 'SHORT'):
                    self.last_rejection_reason = f"High funding cost: {funding_rate:.4%} against position"
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market risk: {e}")
            return False
    
    def _check_execution_risk(self, signal: Dict[str, Any]) -> bool:
        """检查执行风险"""
        try:
            # 检查预估滑点
            estimated_slippage = signal.get('estimated_slippage_pct', 0)
            if estimated_slippage > self.risk_thresholds['max_slippage_pct']:
                self.last_rejection_reason = f"High estimated slippage: {estimated_slippage:.4%} > {self.risk_thresholds['max_slippage_pct']:.4%}"
                return False
            
            # 检查预估价差
            estimated_spread = signal.get('estimated_spread_pct', 0)
            if estimated_spread > self.risk_thresholds['max_spread_pct']:
                self.last_rejection_reason = f"High estimated spread: {estimated_spread:.4%} > {self.risk_thresholds['max_spread_pct']:.4%}"
                return False
            
            # 检查订单大小
            order_size_usdt = signal.get('size_usdt', 0)
            if order_size_usdt > 50000:  # 单笔订单最大$50k
                self.last_rejection_reason = f"Order too large: ${order_size_usdt:.0f} > $50,000"
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking execution risk: {e}")
            return False
    
    def _check_account_risk(self, pnl_history: List[Dict[str, Any]]) -> bool:
        """检查账户风险"""
        try:
            # 计算当日盈亏（简化：使用最近24小时）
            now = datetime.now()
            recent_pnl = 0.0
            recent_losses = 0
            
            for pnl_record in pnl_history[-100:]:  # 检查最近100笔
                record_time = datetime.fromtimestamp(pnl_record.get('timestamp', 0))
                if now - record_time < timedelta(hours=24):
                    pnl = pnl_record.get('pnl', 0)
                    recent_pnl += pnl
                    
                    if pnl < 0:
                        recent_losses += 1
                    else:
                        recent_losses = 0  # 重置连续亏损计数
            
            # 检查当日亏损
            daily_loss_pct = abs(min(recent_pnl, 0)) / self.current_equity if self.current_equity > 0 else 0
            if daily_loss_pct > self.risk_thresholds['max_daily_loss_pct']:
                self.last_rejection_reason = f"Daily loss exceeded: {daily_loss_pct:.2%} > {self.risk_thresholds['max_daily_loss_pct']:.2%}"
                return False
            
            # 检查连续亏损
            if recent_losses >= self.risk_thresholds['max_consecutive_losses']:
                self.last_rejection_reason = f"Consecutive losses: {recent_losses} >= {self.risk_thresholds['max_consecutive_losses']}"
                return False
            
            # 更新风险等级
            self._update_risk_level(daily_loss_pct, recent_losses)
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking account risk: {e}")
            return False
    
    def _check_systemic_risk(self) -> bool:
        """检查系统性风险"""
        try:
            # 这里可以添加对整体市场风险的检查
            # 例如：BTC暴跌、交易所问题、网络延迟等
            
            # 模拟：随机系统性风险事件
            if np.random.random() < 0.001:  # 0.1%概率发生系统性风险
                self.last_rejection_reason = "Systemic risk event detected"
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking systemic risk: {e}")
            return False
    
    def _update_risk_level(self, daily_loss_pct: float, consecutive_losses: int):
        """更新风险等级"""
        try:
            old_level = self.risk_level
            
            # 判断风险等级
            if daily_loss_pct > self.risk_thresholds['max_daily_loss_pct'] * 0.8 or \
               consecutive_losses >= self.risk_thresholds['max_consecutive_losses'] - 1:
                self.risk_level = 'high'
            elif daily_loss_pct > self.risk_thresholds['max_daily_loss_pct'] * 1.2:
                self.risk_level = 'extreme'
            else:
                self.risk_level = 'normal'
            
            # 如果风险等级变化，记录日志
            if old_level != self.risk_level:
                logger.warning(f"Risk level changed: {old_level} -> {self.risk_level}")
                
                # 记录风险事件
                risk_event = {
                    'timestamp': datetime.now(),
                    'old_level': old_level,
                    'new_level': self.risk_level,
                    'daily_loss_pct': daily_loss_pct,
                    'consecutive_losses': consecutive_losses
                }
                self.risk_history.append(risk_event)
                
                if len(self.risk_history) > self.max_history_size:
                    self.risk_history.pop(0)
                    
        except Exception as e:
            logger.error(f"Error updating risk level: {e}")
    
    def record_trade(self, trade_result: Dict[str, Any]):
        """记录交易结果"""
        try:
            self.trade_history.append(trade_result.copy())
            
            # 更新账户权益
            pnl = trade_result.get('estimated_pnl', 0)
            self.current_equity += pnl
            
            # 更新最大回撤
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # 限制历史大小
            if len(self.trade_history) > self.max_history_size:
                self.trade_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def get_risk_level(self) -> str:
        """获取当前风险等级"""
        return self.risk_level
    
    def get_last_rejection_reason(self) -> Optional[str]:
        """获取上次拒绝原因"""
        return self.last_rejection_reason
    
    def get_risk_stats(self) -> Dict[str, Any]:
        """获取风险统计"""
        return {
            'risk_level': self.risk_level,
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'max_drawdown': self.max_drawdown,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'total_trades': len(self.trade_history),
            'recent_rejection_reason': self.last_rejection_reason
        }
    
    def get_risk_history(self, n: int = 10) -> list:
        """获取最近的风险历史"""
        return self.risk_history[-n:] if self.risk_history else []