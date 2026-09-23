import asyncio
import json
import time
from typing import Dict, Any, Optional
import aiohttp
from websockets import connect
from loguru import logger
import msgpack
from datetime import datetime
import numpy as np

from shared_memory.ring_buffer import ArrowRingBuffer, RING_BUFFER_CONFIGS
from io_gateway.wal_writer import WALWriter

class OKXWebSocketClient:
    """OKX V5 WebSocket客户端 - 支持双端点连接"""
    
    def __init__(self):
        # Public端点用于orderbook, trades, funding-rate, open-interest
        self.ws_public_url = "wss://ws.okx.com:8443/ws/v5/public"
        # Business端点用于K线数据
        self.ws_business_url = "wss://ws.okx.com:8443/ws/v5/business"
        
        self.ws_public = None
        self.ws_business = None
        self.running = False
        
        # 初始化共享内存缓冲区
        self.tick_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['fast_tick'])
        self.candle_buffer = ArrowRingBuffer(RING_BUFFER_CONFIGS['candle'])
        
        # WAL日志写入器
        self.wal_writer = WALWriter()
        
        # 连接状态监控
        self.last_heartbeat_public = time.time()
        self.last_heartbeat_business = time.time()
        self.reconnect_attempts = 0
        
    async def connect(self):
        """连接两个WebSocket服务器"""
        try:
            # 连接Public端点
            logger.info(f"Connecting to public endpoint: {self.ws_public_url}")
            self.ws_public = await connect(
                self.ws_public_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            logger.success("Connected to OKX Public WebSocket")
            
            # 连接Business端点
            logger.info(f"Connecting to business endpoint: {self.ws_business_url}")
            self.ws_business = await connect(
                self.ws_business_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            logger.success("Connected to OKX Business WebSocket")
            
            # 订阅频道
            await self._subscribe_channels()
            
            self.running = True
            self.reconnect_attempts = 0
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self.reconnect()
            
    async def _subscribe_channels(self):
        """订阅交易对频道"""
        # Public频道订阅
        public_subscriptions = [
            {"channel": "books", "instId": "DOGE-USDT-SWAP"},
            {"channel": "trades", "instId": "DOGE-USDT-SWAP"},
            {"channel": "funding-rate", "instId": "DOGE-USDT-SWAP"},
            {"channel": "open-interest", "instId": "DOGE-USDT-SWAP"}
        ]
        
        for sub in public_subscriptions:
            await self.ws_public.send(json.dumps({
                "op": "subscribe",
                "args": [sub]
            }))
            logger.info(f"[Public] Subscribed to {sub['channel']}")
            await asyncio.sleep(0.1)
        
        # Business频道订阅（K线）
        business_subscriptions = [
            {"channel": "candle1m", "instId": "DOGE-USDT-SWAP"}
        ]
        
        for sub in business_subscriptions:
            await self.ws_business.send(json.dumps({
                "op": "subscribe",
                "args": [sub]
            }))
            logger.info(f"[Business] Subscribed to {sub['channel']}")
            await asyncio.sleep(0.1)
            
    async def start(self):
        """启动WebSocket客户端"""
        await self.connect()
        
        # 启动心跳检测
        asyncio.create_task(self._heartbeat_monitor())
        
        # 并行接收两个端点的消息
        await asyncio.gather(
            self._receive_public_messages(),
            self._receive_business_messages()
        )
        
    async def _receive_public_messages(self):
        """接收Public端点消息"""
        while self.running and self.ws_public:
            try:
                message = await asyncio.wait_for(self.ws_public.recv(), timeout=30)
                await self._process_message(message, source='public')
                self.last_heartbeat_public = time.time()
                
            except asyncio.TimeoutError:
                logger.warning("Public WebSocket receive timeout")
                break
            except Exception as e:
                logger.error(f"Error receiving public message: {e}")
                break
    
    async def _receive_business_messages(self):
        """接收Business端点消息"""
        while self.running and self.ws_business:
            try:
                message = await asyncio.wait_for(self.ws_business.recv(), timeout=30)
                await self._process_message(message, source='business')
                self.last_heartbeat_business = time.time()
                
            except asyncio.TimeoutError:
                logger.warning("Business WebSocket receive timeout")
                break
            except Exception as e:
                logger.error(f"Error receiving business message: {e}")
                break
                
    async def _process_message(self, message: str, source: str = 'public'):
        """处理接收到的消息"""
        try:
            data = json.loads(message)
            
            # 处理心跳
            if data.get('event') == 'pong':
                return
            
            # 处理订阅确认
            if data.get('event') == 'subscribe':
                logger.debug(f"[{source}] Subscription confirmed: {data.get('arg')}")
                return
            
            # 处理错误
            if data.get('event') == 'error':
                logger.error(f"[{source}] WebSocket error: {data}")
                return
                
            # 处理数据
            if 'arg' in data and 'data' in data:
                channel = data['arg']['channel']
                logger.debug(f"[{source}] Received message for channel: {channel}, data length: {len(data['data'])}")
                
                if channel == 'books':
                    await self._process_orderbook(data['data'])
                elif channel.startswith('candle'):
                    await self._process_candle(data['data'])
                elif channel == 'trades':
                    await self._process_trades(data['data'])
                elif channel == 'funding-rate':
                    await self._process_funding_rate(data['data'])
                elif channel == 'open-interest':
                    await self._process_open_interest(data['data'])
                    
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
    
    def _validate_orderbook_data(self, book: dict) -> bool:
        """验证订单簿数据的完整性和有效性"""
        try:
            # 检查必需字段
            required_fields = ['asks', 'bids', 'ts']
            if not all(field in book for field in required_fields):
                logger.warning(f"Orderbook missing required fields: {book.keys()}")
                return False
            
            # 检查asks和bids不为空
            if not book['asks'] or not book['bids']:
                logger.warning("Orderbook has empty asks or bids")
                return False
            
            # 验证价格和数量格式
            for ask in book['asks'][:5]:  # 只验证前5档
                if len(ask) < 2:
                    logger.warning(f"Invalid ask format: {ask}")
                    return False
                try:
                    price = float(ask[0])
                    quantity = float(ask[1])
                    if price <= 0 or quantity <= 0:
                        logger.warning(f"Invalid ask values: price={price}, qty={quantity}")
                        return False
                except (ValueError, TypeError) as e:
                    logger.warning(f"Cannot convert ask to float: {e}")
                    return False
            
            for bid in book['bids'][:5]:
                if len(bid) < 2:
                    logger.warning(f"Invalid bid format: {bid}")
                    return False
                try:
                    price = float(bid[0])
                    quantity = float(bid[1])
                    if price <= 0 or quantity <= 0:
                        logger.warning(f"Invalid bid values: price={price}, qty={quantity}")
                        return False
                except (ValueError, TypeError) as e:
                    logger.warning(f"Cannot convert bid to float: {e}")
                    return False
            
            # 验证买卖价差合理性（卖价应该高于买价）
            best_ask = float(book['asks'][0][0])
            best_bid = float(book['bids'][0][0])
            if best_ask <= best_bid:
                logger.warning(f"Invalid spread: ask={best_ask} <= bid={best_bid}")
                return False
            
            # 验证时间戳
            try:
                ts = float(book['ts'])
                if ts <= 0:
                    logger.warning(f"Invalid timestamp: {ts}")
                    return False
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert timestamp: {book['ts']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating orderbook data: {e}")
            return False
    
    def _validate_candle_data(self, candle: list) -> bool:
        """验证K线数据的完整性和有效性"""
        try:
            # K线数据应该至少包含7个字段：[ts, o, h, l, c, vol, volCcy]
            if len(candle) < 7:
                logger.warning(f"Candle data too short: {len(candle)} fields")
                return False
            
            # 验证各字段可以转换为数字
            try:
                ts = float(candle[0])
                o = float(candle[1])
                h = float(candle[2])
                l = float(candle[3])
                c = float(candle[4])
                vol = float(candle[5])
                vol_ccy = float(candle[6])
            except (ValueError, TypeError) as e:
                logger.warning(f"Cannot convert candle data to float: {e}")
                return False
            
            # 验证时间戳
            if ts <= 0:
                logger.warning(f"Invalid candle timestamp: {ts}")
                return False
            
            # 验证价格关系：high >= max(open, close) 且 low <= min(open, close)
            if h < max(o, c) or l > min(o, c):
                logger.warning(f"Invalid candle price relationship: O={o}, H={h}, L={l}, C={c}")
                return False
            
            # 验证价格和成交量为正
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                logger.warning(f"Non-positive candle prices: O={o}, H={h}, L={l}, C={c}")
                return False
            
            if vol < 0 or vol_ccy < 0:
                logger.warning(f"Negative candle volume: vol={vol}, volCcy={vol_ccy}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating candle data: {e}")
            return False
            
    async def _process_orderbook(self, book_data: list):
        """处理订单簿数据"""
        try:
            if not book_data:
                return
                
            book = book_data[0]
            
            # 添加诊断日志：查看原始订单簿数据
            logger.debug(f"Raw orderbook data: asks_count={len(book.get('asks', []))}, bids_count={len(book.get('bids', []))}")
            
            # 验证数据
            if not self._validate_orderbook_data(book):
                # 添加诊断日志：详细的验证失败信息
                asks_sample = book.get('asks', [])[:2] if book.get('asks') else []
                bids_sample = book.get('bids', [])[:2] if book.get('bids') else []
                logger.error(f"Invalid orderbook detail - asks: {asks_sample if asks_sample else 'empty'}, bids: {bids_sample if bids_sample else 'empty'}, ts: {book.get('ts', 'missing')}")
                return
            
            timestamp = float(book['ts']) / 1000  # 转换为秒
            
            # 解析买一卖一
            asks = book.get('asks', [])
            bids = book.get('bids', [])
            
            if asks and bids:
                # 获取买一卖一
                ask1 = asks[0]
                bid1 = bids[0]
                
                # 获取第五档深度
                ask5 = asks[4] if len(asks) > 4 else ask1
                bid5 = bids[4] if len(bids) > 4 else bid1
                
                # 准备tick数据 - 使用np.array
                tick = np.array([(
                    timestamp,
                    float(ask1[0]),  # ask1_price
                    float(ask1[1]),  # ask1_quantity
                    float(bid1[0]),  # bid1_price
                    float(bid1[1]),  # bid1_quantity
                    float(ask5[0]),  # ask5_price
                    float(ask5[1]),  # ask5_quantity
                    float(bid5[0]),  # bid5_price
                    float(bid5[1])   # bid5_quantity
                )], dtype=self.tick_buffer.config.dtype)
                
                # Write-ahead: 先持久化到 WAL（落盘），再发布到共享内存。
                # 这样消费者永远不会看到尚未持久化的数据，崩溃也不丢已可见事件。
                self.wal_writer.write_tick({
                    'timestamp': timestamp,
                    'asks': asks[:5],
                    'bids': bids[:5]
                })

                # 发布到共享内存（易失，供计算层零拷贝读取）
                self.tick_buffer.write(tick)
                
        except Exception as e:
            logger.error(f"Error processing orderbook: {e}")
            
    async def _process_candle(self, candle_data: list):
        """处理K线数据"""
        try:
            if not candle_data:
                logger.debug("Received empty candle data, skipping")
                return
            
            # 记录收到的K线数据
            logger.info(f"Processing candle data: {len(candle_data)} candles received")
            
            for candle in candle_data:
                # 验证K线数据
                if not self._validate_candle_data(candle):
                    logger.error(f"Invalid candle data received, skipping: {candle}")
                    continue
                
                timestamp = float(candle[0]) / 1000
                
                # 记录详细的K线数据
                logger.debug(f"Candle: TS={timestamp}, O={candle[1]}, H={candle[2]}, L={candle[3]}, C={candle[4]}, V={candle[5]}")
                
                # 准备K线数据
                candle_np = np.array([(
                    timestamp,
                    float(candle[1]),  # open
                    float(candle[2]),  # high
                    float(candle[3]),  # low
                    float(candle[4]),  # close
                    float(candle[5]),  # volume
                    float(candle[6])   # volume_ccy
                )], dtype=self.candle_buffer.config.dtype)
                
                # Write-ahead: 先持久化到 WAL（落盘），再发布到共享内存。
                self.wal_writer.write_candle({
                    'timestamp': timestamp,
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'volume_ccy': candle[6]
                })
                logger.debug(f"Written candle to WAL: timestamp={timestamp}")

                # 发布到共享内存（易失）
                self.candle_buffer.write(candle_np)
                logger.debug(f"Written candle to SHM, write_index={self.candle_buffer.write_index}")
            
        except Exception as e:
            logger.error(f"Error processing candle: {e}")
            import traceback
            traceback.print_exc()
            
    async def _process_trades(self, trades_data: list):
        """处理交易数据"""
        try:
            for trade in trades_data:
                self.wal_writer.write_trade({
                    'timestamp': float(trade['ts']) / 1000,
                    'side': trade['side'],
                    'price': float(trade['px']),
                    'size': float(trade['sz'])
                })
        except Exception as e:
            logger.error(f"Error processing trades: {e}")
            
    async def _process_funding_rate(self, fr_data: list):
        """处理资金费率数据"""
        try:
            if fr_data:
                rate = fr_data[0]
                self.wal_writer.write_funding_rate({
                    'timestamp': time.time(),
                    'rate': float(rate['fundingRate']),
                    'next_time': float(rate['nextFundingTime'])
                })
        except Exception as e:
            logger.error(f"Error processing funding rate: {e}")
            
    async def _process_open_interest(self, oi_data: list):
        """处理持仓量数据"""
        try:
            if oi_data:
                oi = oi_data[0]
                self.wal_writer.write_open_interest({
                    'timestamp': time.time(),
                    'oi': float(oi['oi']),
                    'oi_ccy': float(oi['oiCcy'])
                })
        except Exception as e:
            logger.error(f"Error processing open interest: {e}")
            
    async def _heartbeat_monitor(self):
        """心跳检测"""
        while self.running:
            await asyncio.sleep(5)
            
            current_time = time.time()
            if current_time - self.last_heartbeat_public > 30:
                logger.warning("Public heartbeat timeout, reconnecting...")
                await self.reconnect()
            if current_time - self.last_heartbeat_business > 30:
                logger.warning("Business heartbeat timeout, reconnecting...")
                await self.reconnect()
                
    async def reconnect(self):
        """重新连接"""
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > 5:
            logger.error("Max reconnection attempts reached")
            self.running = False
            return
            
        logger.info(f"Reconnecting... (attempt {self.reconnect_attempts})")
        
        # 关闭现有连接
        if self.ws_public:
            try:
                await self.ws_public.close()
            except:
                pass
        if self.ws_business:
            try:
                await self.ws_business.close()
            except:
                pass
            
        await asyncio.sleep(min(2 ** self.reconnect_attempts, 30))  # 指数退避
        
        await self.connect()
        
    async def stop(self):
        """停止客户端"""
        self.running = False
        
        if self.ws_public:
            try:
                await self.ws_public.close()
            except:
                pass
        if self.ws_business:
            try:
                await self.ws_business.close()
            except:
                pass
            
        self.tick_buffer.cleanup()
        self.candle_buffer.cleanup()
        self.wal_writer.close()
        
        logger.info("WebSocket client stopped")

async def main():
    """测试函数"""
    client = OKXWebSocketClient()
    try:
        await client.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await client.stop()

if __name__ == "__main__":
    asyncio.run(main())
