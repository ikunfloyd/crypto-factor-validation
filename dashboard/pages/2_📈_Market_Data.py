"""
市场数据页面
显示实时市场行情、K线图、订单簿等
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.theme import apply_custom_css
from dashboard.utils.data_loader import get_data_loader
from dashboard.utils.formatters import format_price, format_percentage, format_number
from dashboard.components.charts import (
    render_candlestick_chart, render_pie_chart, render_volume_chart
)

# 页面配置
st.set_page_config(
    page_title="市场数据 - DOGE Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# 应用自定义CSS
apply_custom_css()

# 初始化数据加载器
data_loader = get_data_loader()

# 页面标题
st.title("📈 市场数据 - DOGE-USDT")
st.markdown("---")

# 加载真实市场数据
market_data = data_loader.load_market_data()

# 关键市场指标
col1, col2, col3, col4 = st.columns(4)

with col1:
    price = market_data.get('price', 0)
    change = market_data.get('change_24h', 0)
    st.metric(
        "最新价",
        format_price(price) if price > 0 else "N/A",
        format_percentage(change, show_sign=True) if change != 0 else "0.00%",
        delta_color="normal"
    )

with col2:
    volume_24h = market_data.get('volume_24h', 0)
    st.metric(
        "24h成交量",
        format_number(volume_24h) if volume_24h > 0 else "N/A",
        ""
    )

with col3:
    funding_rate = market_data.get('funding_rate', 0)
    st.metric(
        "资金费率",
        f"{funding_rate:.4f}" if funding_rate != 0 else "N/A",
        ""
    )

with col4:
    # 计算振幅
    high_24h = market_data.get('high_24h', 0)
    low_24h = market_data.get('low_24h', 0)
    if high_24h > 0 and low_24h > 0:
        amplitude = ((high_24h - low_24h) / low_24h) * 100
        st.metric(
            "24h振幅",
            format_percentage(amplitude, show_sign=False),
            ""
        )
    else:
        st.metric("24h振幅", "N/A", "")

st.markdown("---")

# K线图和成交量
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📊 K线图")
    
    try:
        # 从共享内存加载真实K线数据
        shm_data = data_loader.load_shared_memory_data()
        candles = shm_data.get('candles')
        
        # 添加调试信息
        st.caption(f"🔍 调试信息: 共享内存状态={shm_data.get('status', 'unknown')}")
        
        if candles is not None and len(candles) > 0:
            st.caption(f"🔍 调试信息: 加载了 {len(candles)} 根K线")
            
            # 转换为DataFrame - 正确处理时区
            # OKX的timestamp是UTC时间（秒），需要转换为本地时区显示
            df_candles = pd.DataFrame({
                'timestamp': pd.to_datetime([c['timestamp'] for c in candles], unit='s', utc=True).tz_convert('America/Los_Angeles'),
                'open': [float(c['open']) for c in candles],
                'high': [float(c['high']) for c in candles],
                'low': [float(c['low']) for c in candles],
                'close': [float(c['close']) for c in candles],
                'volume': [float(c['volume']) for c in candles]
            })
            
            # 显示时间范围（本地时间）
            time_min = df_candles['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S %Z')
            time_max = df_candles['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S %Z')
            st.caption(f"🔍 K线时间范围: {time_min} 至 {time_max}")
            
            # 计算布林带
            df_candles['bollinger_ma'] = df_candles['close'].rolling(window=20).mean()
            df_candles['bollinger_std'] = df_candles['close'].rolling(window=20).std()
            df_candles['bollinger_upper'] = df_candles['bollinger_ma'] + (df_candles['bollinger_std'] * 2)
            df_candles['bollinger_lower'] = df_candles['bollinger_ma'] - (df_candles['bollinger_std'] * 2)
            
            render_candlestick_chart(df_candles, title="DOGE-USDT 1分钟K线（真实数据）")
        else:
            st.warning("⚠️ 暂无K线数据，请确保系统正在运行")
            st.caption(f"🔍 调试信息: candles={candles}, type={type(candles)}")
        
    except Exception as e:
        st.error(f"❌ 无法加载K线图: {e}")
        import traceback
        st.code(traceback.format_exc())

with col2:
    st.subheader("📋 关键数据")
    
    # 显示更多市场数据
    st.markdown("**价格统计**")
    if high_24h > 0 and low_24h > 0:
        st.text(f"最高: {format_price(high_24h)}")
        st.text(f"最低: {format_price(low_24h)}")
        st.text(f"均价: {format_price((high_24h + low_24h) / 2)}")
    else:
        st.text("暂无24h价格统计")
    
    st.markdown("---")
    
    st.markdown("**交易统计**")
    if volume_24h > 0 and price > 0:
        st.text(f"成交量: {format_number(volume_24h)}")
        st.text(f"成交额: ${format_number(volume_24h * price)}")
    else:
        st.text("暂无交易统计")
    
    # 显示最新tick时间
    shm_data = data_loader.load_shared_memory_data()
    ticks = shm_data.get('ticks')
    if ticks is not None and len(ticks) > 0:
        latest_time = pd.to_datetime(ticks[-1]['timestamp'], unit='s')
        st.markdown("---")
        st.markdown("**数据更新**")
        st.text(f"最新: {latest_time.strftime('%H:%M:%S')}")

st.markdown("---")

# 订单簿和资金流
col1, col2 = st.columns(2)

with col1:
    st.subheader("📖 订单簿（买卖五档）")
    
    try:
        # 从共享内存获取真实订单簿数据
        ticks = shm_data.get('ticks')
        
        if ticks is not None and len(ticks) > 0:
            latest_tick = ticks[-1]
            current_price = float(latest_tick['bid1_price'])
            
            # Ask（卖单）- 从tick数据提取
            st.markdown("**卖单 (Ask)**")
            ask_data = [{
                '价格': format_price(float(latest_tick['ask1_price'])),
                '数量': format_number(float(latest_tick['ask1_quantity'])),
                '累计': format_number(float(latest_tick['ask1_quantity']))
            }]
            
            # 如果有ask5数据
            if float(latest_tick['ask5_price']) > 0:
                ask_data.append({
                    '价格': format_price(float(latest_tick['ask5_price'])),
                    '数量': format_number(float(latest_tick['ask5_quantity'])),
                    '累计': format_number(float(latest_tick['ask1_quantity']) + float(latest_tick['ask5_quantity']))
                })
            
            df_asks = pd.DataFrame(ask_data)
            st.dataframe(df_asks, use_container_width=True, hide_index=True, height=150)
            
            # 当前价格
            st.markdown(f"<div style='text-align:center; font-size:1.5em; font-weight:bold; color:#2ca02c; padding:10px; background:#e8f5e9; border-radius:5px;'>{format_price(current_price)}</div>", unsafe_allow_html=True)
            
            # Bid（买单）
            st.markdown("**买单 (Bid)**")
            bid_data = [{
                '价格': format_price(float(latest_tick['bid1_price'])),
                '数量': format_number(float(latest_tick['bid1_quantity'])),
                '累计': format_number(float(latest_tick['bid1_quantity']))
            }]
            
            # 如果有bid5数据
            if float(latest_tick['bid5_price']) > 0:
                bid_data.append({
                    '价格': format_price(float(latest_tick['bid5_price'])),
                    '数量': format_number(float(latest_tick['bid5_quantity'])),
                    '累计': format_number(float(latest_tick['bid1_quantity']) + float(latest_tick['bid5_quantity']))
                })
            
            df_bids = pd.DataFrame(bid_data)
            st.dataframe(df_bids, use_container_width=True, hide_index=True, height=150)
        else:
            st.warning("暂无订单簿数据")
        
    except Exception as e:
        st.error(f"无法加载订单簿: {e}")

with col2:
    st.subheader("💰 资金流向分析")
    
    try:
        # 从K线数据分析资金流向
        candles = shm_data.get('candles')
        
        if candles is not None and len(candles) > 100:
            # 分析最近100根K线的涨跌分布
            recent_candles = candles[-100:]
            up_count = sum(1 for c in recent_candles if c['close'] > c['open'])
            down_count = len(recent_candles) - up_count
            
            buy_ratio = up_count / len(recent_candles)
            sell_ratio = down_count / len(recent_candles)
            
            flow_data = {
                '主动买入': buy_ratio * 100,
                '主动卖出': sell_ratio * 100
            }
            
            render_pie_chart(flow_data, title="主动买卖比例（基于K线）")
            
            # 显示详细数据
            st.markdown("---")
            st.markdown("**资金流向详情**")
            
            if volume_24h > 0:
                buy_volume = volume_24h * buy_ratio
                sell_volume = volume_24h * sell_ratio
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("主动买入", format_number(buy_volume), f"{buy_ratio*100:.1f}%")
                with col_b:
                    st.metric("主动卖出", format_number(sell_volume), f"{sell_ratio*100:.1f}%")
                
                # 净流入
                net_flow = (buy_ratio - sell_ratio) * volume_24h
                net_flow_value = net_flow * price if price > 0 else 0
                
                st.metric(
                    "净流入",
                    f"${format_number(abs(net_flow_value))}" if net_flow_value != 0 else "$0",
                    "流入" if net_flow > 0 else "流出" if net_flow < 0 else "",
                    delta_color="normal" if net_flow > 0 else "inverse"
                )
            else:
                st.info("暂无24h成交量数据")
        else:
            st.warning("数据不足，无法分析资金流向")
        
    except Exception as e:
        st.error(f"无法生成资金流向分析: {e}")

st.markdown("---")

# 最近成交
st.subheader("📊 最近成交")

try:
    # 从执行状态加载真实成交记录
    execution_state = data_loader.load_execution_state()
    trades = execution_state.get('trades', [])
    
    if trades and len(trades) > 0:
        # 只显示最近20条
        recent_trades = trades[-20:]
        
        trades_data = []
        for trade in recent_trades:
            trade_time = trade.get('timestamp', datetime.now())
            if isinstance(trade_time, (int, float)):
                trade_time = pd.to_datetime(trade_time, unit='s')
            elif isinstance(trade_time, str):
                trade_time = pd.to_datetime(trade_time)
            
            trades_data.append({
                '时间': trade_time.strftime('%H:%M:%S'),
                '方向': '买入' if trade.get('side', '').upper() == 'BUY' else '卖出',
                '价格': format_price(trade.get('price', 0)),
                '数量': format_number(trade.get('quantity', 0)),
                '金额': format_price(trade.get('price', 0) * trade.get('quantity', 0))
            })
        
        df_trades = pd.DataFrame(trades_data)
        st.dataframe(df_trades, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("暂无成交记录")
    
except Exception as e:
    st.error(f"无法加载成交记录: {e}")

# 页脚
st.markdown("---")
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据延迟: ~100ms")
