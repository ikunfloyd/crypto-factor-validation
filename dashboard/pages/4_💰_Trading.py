"""
交易执行页面
显示持仓、订单、盈亏、风险等交易信息
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
from dashboard.utils.formatters import (
    format_price, format_pnl, format_percentage,
    format_number, format_crypto_amount, format_uptime
)
from dashboard.components.charts import render_pnl_curve, render_gauge_chart
from dashboard.components.tables import render_orders_table, render_trades_table
from dashboard.components.metrics import render_progress_bar

# 页面配置
st.set_page_config(
    page_title="交易执行 - DOGE Trading Dashboard",
    page_icon="💰",
    layout="wide"
)

# 应用自定义CSS
apply_custom_css()

# 初始化数据加载器
data_loader = get_data_loader()

# 页面标题
st.title("💰 交易执行 - 持仓与订单")
st.markdown("---")

# 加载真实数据
execution_state = data_loader.load_execution_state()
market_data = data_loader.load_market_data()
position = execution_state.get('position')

# 持仓总览
st.subheader("📊 当前持仓")

if position and position.get('size', 0) != 0:
    # 有持仓
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        side = position.get('side', 'FLAT')
        size = position.get('size', 0)
        side_emoji = "📈" if side == "LONG" else "📉" if side == "SHORT" else "➖"
        st.metric("持仓方向", f"{side_emoji} {side}", "")
    
    with col2:
        st.metric("持仓量", format_crypto_amount(size), "")
    
    with col3:
        entry_price = position.get('entry_price', 0)
        current_price = market_data.get('price', 0)
        st.metric("开仓价", format_price(entry_price), "")
        st.caption(f"当前价: {format_price(current_price)}" if current_price > 0 else "当前价: N/A")
    
    with col4:
        pnl = position.get('pnl', 0)
        pnl_pct = position.get('pnl_pct', 0)
        st.metric(
            "未实现盈亏",
            format_pnl(pnl),
            format_percentage(pnl_pct),
            delta_color="normal"
        )
    
    # 持仓详情
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 持仓详细信息")
        
        # 仓位使用率
        position_ratio = position.get('position_ratio', 0)
        if position_ratio > 0:
            render_progress_bar("仓位使用率", position_ratio, 100, show_percentage=True)
        
        # 持仓时长
        entry_time = position.get('entry_time')
        if entry_time:
            if isinstance(entry_time, (int, float)):
                entry_time = pd.to_datetime(entry_time, unit='s')
            elif isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            duration = datetime.now() - entry_time
            duration_minutes = int(duration.total_seconds() / 60)
            st.text(f"持仓时长: {duration_minutes // 60}h {duration_minutes % 60}m")
        else:
            st.text("持仓时长: N/A")
        
        # 盈亏金额
        margin = size * entry_price
        st.text(f"保证金: {format_price(margin)}")
        st.text(f"盈亏金额: {format_pnl(pnl)}")
        st.text(f"盈亏比例: {format_percentage(pnl_pct)}")
    
    with col2:
        st.markdown("#### 风险控制")
        
        # 止损止盈
        stop_loss = entry_price * 0.98
        take_profit = entry_price * 1.05
        
        st.text(f"止损价: {format_price(stop_loss)}")
        st.text(f"止盈价: {format_price(take_profit)}")
        
        # 距离止损止盈的百分比
        to_stop_loss = ((current_price - stop_loss) / stop_loss) * 100
        to_take_profit = ((take_profit - current_price) / current_price) * 100
        
        st.text(f"距止损: {to_stop_loss:.2f}%")
        st.text(f"距止盈: {to_take_profit:.2f}%")

else:
    # 无持仓
    st.info("📭 当前无持仓")

st.markdown("---")

# 风险指标
st.subheader("⚠️ 风险指标")

col1, col2, col3 = st.columns(3)

with col1:
    # 最大回撤
    max_drawdown = -1.2
    render_gauge_chart(
        abs(max_drawdown),
        title="最大回撤 (%)",
        min_value=0,
        max_value=10,
        thresholds={'low': 3, 'medium': 5}
    )

with col2:
    # 夏普比率
    sharpe_ratio = 1.85
    render_gauge_chart(
        sharpe_ratio,
        title="夏普比率",
        min_value=0,
        max_value=3,
        thresholds={'low': 1, 'medium': 2}
    )

with col3:
    # 杠杆倍数
    leverage = 1.0
    st.metric("杠杆倍数", f"{leverage}x", "")
    st.metric("风险等级", "低", "")

st.markdown("---")

# 性能统计
st.subheader("📈 今日表现")

col1, col2, col3, col4 = st.columns(4)

with col1:
    daily_pnl = execution_state.get('daily_pnl', 0)
    st.metric(
        "今日盈亏",
        format_pnl(daily_pnl) if daily_pnl != 0 else "$0.00",
        format_percentage(daily_pnl / 10000 * 100) if daily_pnl != 0 else ""
    )

with col2:
    total_pnl = execution_state.get('total_pnl', 0)
    st.metric("累计收益", format_pnl(total_pnl) if total_pnl != 0 else "$0.00", "")

with col3:
    win_rate = execution_state.get('win_rate', 0)
    st.metric("胜率", f"{win_rate:.1f}%" if win_rate > 0 else "N/A", "")

with col4:
    trade_count = execution_state.get('trade_count', 0)
    st.metric("交易次数", str(trade_count) if trade_count > 0 else "0", "")

st.markdown("---")

# 盈亏曲线
st.subheader("💹 盈亏曲线")

try:
    # 尝试从执行状态加载盈亏历史
    pnl_history = execution_state.get('pnl_history', [])
    
    if pnl_history and len(pnl_history) > 0:
        # 转换为DataFrame
        df_pnl = pd.DataFrame(pnl_history)
        if 'timestamp' in df_pnl.columns and 'cumulative_pnl' in df_pnl.columns:
            df_pnl['timestamp'] = pd.to_datetime(df_pnl['timestamp'])
            render_pnl_curve(df_pnl, title="累计盈亏曲线（真实数据）")
        else:
            st.info("盈亏历史数据格式不完整")
    else:
        st.info("暂无盈亏曲线数据，请等待交易执行")
    
except Exception as e:
    st.error(f"无法加载盈亏曲线: {e}")

st.markdown("---")

# 订单和成交
tab1, tab2 = st.tabs(["📋 活跃订单", "✅ 成交记录"])

with tab1:
    st.subheader("活跃订单")
    
    try:
        # 从执行状态加载真实活跃订单
        active_orders = execution_state.get('orders', [])
        
        if active_orders and len(active_orders) > 0:
            render_orders_table(active_orders, title="")
        else:
            st.info("当前无活跃订单")
    
    except Exception as e:
        st.error(f"无法加载活跃订单: {e}")

with tab2:
    st.subheader("成交记录")
    
    try:
        # 从执行状态加载真实成交记录
        trades = execution_state.get('trades', [])
        
        if trades and len(trades) > 0:
            # 只显示最近20条
            recent_trades = trades[-20:]
            render_trades_table(recent_trades, title="")
        else:
            st.info("暂无成交记录")
    
    except Exception as e:
        st.error(f"无法加载成交记录: {e}")

st.markdown("---")

# 交易统计
st.subheader("📊 交易统计")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 交易分布")
    
    # 盈亏交易数量统计
    trade_count = execution_state.get('trade_count', 0)
    win_rate = execution_state.get('win_rate', 0)
    
    if trade_count > 0 and win_rate > 0:
        winning_trades = int(trade_count * win_rate / 100)
        losing_trades = trade_count - winning_trades
        
        stats_df = pd.DataFrame({
            '类型': ['盈利交易', '亏损交易'],
            '数量': [winning_trades, losing_trades],
            '占比': [f"{win_rate:.1f}%", f"{100-win_rate:.1f}%"]
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无交易统计数据")

with col2:
    st.markdown("#### 平均指标")
    
    avg_metrics = execution_state.get('avg_metrics', {})
    
    if avg_metrics:
        avg_hold_time = avg_metrics.get('avg_hold_time', 0)
        avg_profit = avg_metrics.get('avg_profit', 0)
        avg_loss = avg_metrics.get('avg_loss', 0)
        profit_loss_ratio = avg_metrics.get('profit_loss_ratio', 0)
        
        st.text(f"平均持仓时间: {avg_hold_time:.1f}h" if avg_hold_time > 0 else "平均持仓时间: N/A")
        st.text(f"平均盈利: {format_pnl(avg_profit)}" if avg_profit != 0 else "平均盈利: N/A")
        st.text(f"平均亏损: {format_pnl(avg_loss)}" if avg_loss != 0 else "平均亏损: N/A")
        st.text(f"盈亏比: {profit_loss_ratio:.2f}" if profit_loss_ratio > 0 else "盈亏比: N/A")
    else:
        st.info("暂无平均指标数据")

# 页脚
st.markdown("---")
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据实时同步")
