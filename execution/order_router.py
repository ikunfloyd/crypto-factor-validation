import time 
import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger

class OrderRouter:
    """订单路由器 - 负责执行订单，支持TWAP等智能算法"""
    
    def __init__(self, api_client=None):
        self.api_client = api_client  # 实际的交易所API客户端
        self.order_history = []
        self.max_history_size = 1000
        
        # TWAP参数
        self.twap_params = {
            'default_slice_duration': 10,  # 默认每片持续时间（秒）
            'max_slices': 10,  # 最大切片数
            'min_slice_value': 100,  # 最小切片价值（USDT）
        }
        
        # 执行统计
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume_usdt': 0.0,
            'avg_slippage_pct': 0.0,
            'avg_execution_time': 0.0
        }
        
        logger.info("OrderRouter initialized")
    
    def execute_market(self, order_details: Dict[str, Any], 
                      tick_data: np.ndarray) -> Dict[str, Any]:
        """
        执行市价订单
        
        Args:
            order_details: 订单详情
            tick_data: 当前tick数据
        """
        try:
            logger.info(f"Executing market order: {order_details['direction']} "
                       f"${order_details['size_usdt']:.0f}")
            
            # 模拟执行（实际应调用交易所API）
            execution_result = self._simulate_market_execution(order_details, tick_data)
            
            # 更新统计
            self._update_stats(execution_result)
            
            # 记录历史
            self._record_order(execution_result)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing market order: {e}")
            return self._create_error_result(order_details, str(e))
    
    def execute_twap(self, order_details: Dict[str, Any], 
                    tick_data: np.ndarray) -> Dict[str, Any]:
        """
        执行TWAP订单
        
        Args:
            order_details: 订单详情
            tick_data: 当前tick数据
        """
        try:
            size_usdt = order_details['size_usdt']
            
            # 判断是否需要TWAP
            if size_usdt <= 10000:  # 小于$10k使用市价单
                logger.info(f"Order size ${size_usdt:.0f} too small for TWAP, using market order")
                return self.execute_market(order_details, tick_data)
            
            logger.info(f"Executing TWAP order: {order_details['direction']} "
                       f"${size_usdt:.0f}")
            
            # 计算TWAP切片
            slices = self._calculate_twap_slices(order_details)
            
            # 执行切片
            all_results = []
            for i, slice_details in enumerate(slices):
                logger.info(f"Executing TWAP slice {i+1}/{len(slices)}: "
                           f"${slice_details['size_usdt']:.0f}")
                
                # 等待到下一个执行时间
                if i > 0:
                    wait_time = slice_details['execute_after'] - time.time()
                    if wait_time > 0:
                        time.sleep(wait_time)
                
                # 执行切片
                slice_result = self.execute_market(slice_details, tick_data)
                all_results.append(slice_result)
                
                # 如果切片失败，停止TWAP
                if not slice_result.get('success', False):
                    logger.error(f"TWAP slice {i+1} failed, stopping TWAP")
                    break
            
            # 汇总结果
            twap_result = self._aggregate_twap_results(order_details, all_results)
            
            return twap_result
            
        except Exception as e:
            logger.error(f"Error executing TWAP order: {e}")
            return self._create_error_result(order_details, str(e))
    
    def _calculate_twap_slices(self, order_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """计算TWAP切片"""
        try:
            total_size = order_details['size_usdt']
            direction = order_details['direction']
            symbol = order_details.get('symbol', 'DOGE-USDT')
            
            # 计算切片数量
            max_slices = self.twap_params['max_slices']
            min_slice_value = self.twap_params['min_slice_value']
            
            # 确保每个切片不小于最小价值
            n_slices = min(max_slices, int(total_size / min_slice_value))
            n_slices = max(1, n_slices)  # 至少1个切片
            
            # 计算切片大小（等分）
            slice_size = total_size / n_slices
            
            # 计算切片时间间隔
            total_duration = self.twap_params['default_slice_duration'] * n_slices
            slice_interval = total_duration / n_slices
            
            # 创建切片
            slices = []
            current_time = time.time()
            
            for i in range(n_slices):
                slice_details = order_details.copy()
                slice_details.update({
                    'size_usdt': slice_size,
                    'size_coin': slice_size / order_details.get('current_price', 1),
                    'is_twap_slice': True,
                    'slice_index': i,
                    'total_slices': n_slices,
                    'execute_after': current_time + (i * slice_interval),
                    'slice_id': f"{order_details.get('order_id', 'unknown')}_slice_{i}"
                })
                slices.append(slice_details)
            
            logger.debug(f"TWAP calculated: {n_slices} slices of ${slice_size:.0f} "
                        f"every {slice_interval:.1f}s")
            
            return slices
            
        except Exception as e:
            logger.error(f"Error calculating TWAP slices: {e}")
            # 出错时返回单个切片（即整个订单）
            return [order_details]
    
    def _simulate_market_execution(self, order_details: Dict[str, Any], 
                                 tick_data: np.ndarray) -> Dict[str, Any]:
        """模拟市价订单执行"""
        try:
            direction = order_details['direction']
            size_usdt = order_details['size_usdt']
            current_price = order_details.get('current_price', 0.08)
            
            if len(tick_data) == 0:
                logger.warning("No tick data for execution simulation")
                return self._create_error_result(order_details, "No tick data")
            
            # 获取最新tick
            latest_tick = tick_data[-1]
            
            # 计算执行价格（考虑买卖价差）
            if direction == 'LONG':
                # 买单价（吃单）
                execution_price = latest_tick['ask1_price']
                spread_cost = execution_price - latest_tick['bid1_price']
            elif direction == 'SHORT':
                # 卖单价（吃单）
                execution_price = latest_tick['bid1_price']
                spread_cost = latest_tick['ask1_price'] - execution_price
            else:
                execution_price = current_price
                spread_cost = 0
            
            # 计算滑点（模拟市场冲击）
            size_coin = size_usdt / execution_price if execution_price > 0 else 0
            slippage = self._calculate_slippage(size_coin, direction, tick_data)
            
            # 最终执行价格
            final_price = execution_price * (1 + slippage)
            
            # 计算执行成本
            spread_pct = spread_cost / execution_price if execution_price > 0 else 0
            slippage_pct = slippage
            total_cost_pct = spread_pct + slippage_pct
            
            # 模拟执行时间
            execution_time = np.random.uniform(0.1, 0.5)  # 100-500ms
            
            # 构建结果
            result = {
                'order_id': f"order_{int(time.time() * 1000)}",
                'direction': direction,
                'symbol': order_details.get('symbol', 'DOGE-USDT'),
                'size_usdt': size_usdt,
                'size_coin': size_coin,
                'requested_price': current_price,
                'execution_price': final_price,
                'avg_price': final_price,  # 市价单只有一个价格
                'filled_pct': 1.0,
                'status': 'executed',
                'success': True,
                'spread_pct': spread_pct,
                'slippage_pct': slippage_pct,
                'total_cost_pct': total_cost_pct,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'estimated_pnl': 0.0,  # 需要后续计算
                'is_twap_slice': order_details.get('is_twap_slice', False),
                'slice_index': order_details.get('slice_index', 0),
                'total_slices': order_details.get('total_slices', 1)
            }
            
            logger.debug(f"Order executed: {final_price:.6f} "
                        f"(slippage: {slippage_pct:.4%}, spread: {spread_pct:.4%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating market execution: {e}")
            return self._create_error_result(order_details, str(e))
    
    def _calculate_slippage(self, size_coin: float, direction: str, 
                           tick_data: np.ndarray) -> float:
        """计算滑点"""
        try:
            if len(tick_data) == 0:
                return 0.0
            
            latest_tick = tick_data[-1]
            
            # 简单滑点模型：基于订单大小和盘口深度
            if direction == 'LONG':
                # 买单滑点：消耗卖单深度
                available_volume = latest_tick['ask1_quantity']
                
                # 如果订单小于第一档深度，滑点很小
                if size_coin <= available_volume:
                    slippage = 0.0001  # 0.01%
                else:
                    # 需要吃多档，估计滑点
                    depth_needed = size_coin
                    total_depth = 0
                    price_levels = []
                    
                    # 简单估计（实际应看多档深度）
                    for i in range(min(5, len(tick_data))):
                        tick = tick_data[-(i+1)]
                        total_depth += tick['ask5_quantity']
                        price_levels.append(tick['ask5_price'])
                    
                    if total_depth > 0:
                        # 简单线性估计
                        slippage = min(0.01, 0.0005 * (depth_needed / total_depth))
                    else:
                        slippage = 0.001  # 默认1‰
            
            elif direction == 'SHORT':
                # 卖单滑点：消耗买单深度
                available_volume = latest_tick['bid1_quantity']
                
                if size_coin <= available_volume:
                    slippage = -0.0001  # 负滑点（可能更好价格）
                else:
                    slippage = -0.0005  # 默认负滑点
            
            else:
                slippage = 0.0
            
            # 添加随机波动
            slippage += np.random.normal(0, 0.0001)
            
            return slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return 0.001  # 默认1‰滑点
    
    def _aggregate_twap_results(self, original_order: Dict[str, Any],
                               slice_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """汇总TWAP切片结果"""
        try:
            if not slice_results:
                return self._create_error_result(original_order, "No slice results")
            
            # 过滤成功的切片
            successful_slices = [r for r in slice_results if r.get('success', False)]
            
            if not successful_slices:
                return self._create_error_result(original_order, "All slices failed")
            
            # 计算总成交量
            total_size_usdt = sum(s['size_usdt'] for s in successful_slices)
            total_size_coin = sum(s['size_coin'] for s in successful_slices)
            
            # 计算加权平均价格
            if total_size_coin > 0:
                avg_price = total_size_usdt / total_size_coin
            else:
                avg_price = original_order.get('current_price', 0)
            
            # 计算平均滑点和成本
            avg_slippage = np.mean([s.get('slippage_pct', 0) for s in successful_slices])
            avg_spread = np.mean([s.get('spread_pct', 0) for s in successful_slices])
            
            # 构建TWAP汇总结果
            result = {
                'order_id': f"twap_{int(time.time() * 1000)}",
                'direction': original_order['direction'],
                'symbol': original_order.get('symbol', 'DOGE-USDT'),
                'size_usdt': total_size_usdt,
                'size_coin': total_size_coin,
                'requested_price': original_order.get('current_price', 0),
                'execution_price': avg_price,
                'avg_price': avg_price,
                'filled_pct': total_size_usdt / original_order['size_usdt'] if original_order['size_usdt'] > 0 else 0,
                'status': 'executed',
                'success': True,
                'spread_pct': avg_spread,
                'slippage_pct': avg_slippage,
                'total_cost_pct': avg_spread + avg_slippage,
                'execution_time': sum(s.get('execution_time', 0) for s in successful_slices),
                'timestamp': datetime.now().isoformat(),
                'estimated_pnl': 0.0,
                'is_twap': True,
                'total_slices': len(slice_results),
                'successful_slices': len(successful_slices),
                'slice_results': successful_slices
            }
            
            logger.info(f"TWAP completed: {result['filled_pct']:.1%} filled, "
                       f"avg price: {avg_price:.6f}, "
                       f"total cost: {result['total_cost_pct']:.4%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error aggregating TWAP results: {e}")
            return self._create_error_result(original_order, str(e))
    
    def _update_stats(self, execution_result: Dict[str, Any]):
        """更新执行统计"""
        try:
            self.execution_stats['total_orders'] += 1
            
            if execution_result.get('success', False):
                self.execution_stats['successful_orders'] += 1
                self.execution_stats['total_volume_usdt'] += execution_result.get('size_usdt', 0)
                
                # 更新平均滑点（指数移动平均）
                current_slippage = execution_result.get('slippage_pct', 0)
                current_avg = self.execution_stats['avg_slippage_pct']
                n_success = self.execution_stats['successful_orders']
                
                # 指数移动平均
                alpha = 0.1
                self.execution_stats['avg_slippage_pct'] = (
                    current_slippage * alpha + current_avg * (1 - alpha)
                )
                
                # 更新平均执行时间
                exec_time = execution_result.get('execution_time', 0)
                current_avg_time = self.execution_stats['avg_execution_time']
                self.execution_stats['avg_execution_time'] = (
                    exec_time * alpha + current_avg_time * (1 - alpha)
                )
            else:
                self.execution_stats['failed_orders'] += 1
                
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def _record_order(self, execution_result: Dict[str, Any]):
        """记录订单历史"""
        try:
            self.order_history.append(execution_result.copy())
            
            # 限制历史大小
            if len(self.order_history) > self.max_history_size:
                self.order_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error recording order: {e}")
    
    def _create_error_result(self, order_details: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'order_id': f"error_{int(time.time() * 1000)}",
            'direction': order_details.get('direction', 'NEUTRAL'),
            'symbol': order_details.get('symbol', 'DOGE-USDT'),
            'size_usdt': order_details.get('size_usdt', 0),
            'status': 'failed',
            'success': False,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_order_history(self, n: int = 10) -> list:
        """获取最近的订单历史"""
        return self.order_history[-n:] if self.order_history else []
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return self.execution_stats.copy()