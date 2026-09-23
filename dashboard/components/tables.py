"""
表格组件
提供各种数据表格展示
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.formatters import (
    format_timestamp, format_price, format_pnl,
    format_percentage, format_number
)


def render_dataframe(df: pd.DataFrame, title: Optional[str] = None,
                    max_rows: int = 20, use_container_width: bool = True):
    """
    渲染DataFrame表格
    
    Args:
        df: DataFrame数据
        title: 标题
        max_rows: 最大显示行数
        use_container_width: 是否使用容器宽度
    """
    if title:
        st.subheader(title)
    
    if df is None or df.empty:
        st.info("暂无数据")
        return
    
    # 限制行数
    display_df = df.head(max_rows) if len(df) > max_rows else df
    
    st.dataframe(display_df, use_container_width=use_container_width, height=min(400, 35 * len(display_df) + 38))


def render_orders_table(orders: List[Dict], title: str = "订单列表"):
    """
    渲染订单表格
    
    Args:
        orders: 订单列表
        title: 标题
    """
    st.subheader(title)
    
    if not orders:
        st.info("暂无订单")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(orders)
    
    # 格式化列
    if 'timestamp' in df.columns:
        df['时间'] = df['timestamp'].apply(lambda x: format_timestamp(x, fmt="%H:%M:%S"))
    
    if 'price' in df.columns:
        df['价格'] = df['price'].apply(format_price)
    
    if 'quantity' in df.columns:
        df['数量'] = df['quantity'].apply(format_number)
    
    # 选择要显示的列
    display_cols = []
    col_mapping = {
        'order_id': '订单ID',
        'side': '方向',
        'type': '类型',
        '价格': '价格',
        '数量': '数量',
        'status': '状态',
        '时间': '时间'
    }
    
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns or new_col in df.columns:
            display_cols.append(new_col if new_col in df.columns else old_col)
    
    if display_cols:
        st.dataframe(df[display_cols], use_container_width=True, height=300)
    else:
        st.dataframe(df, use_container_width=True, height=300)


def render_trades_table(trades: List[Dict], title: str = "成交记录"):
    """
    渲染成交记录表格
    
    Args:
        trades: 成交记录列表
        title: 标题
    """
    st.subheader(title)
    
    if not trades:
        st.info("暂无成交记录")
        return
    
    df = pd.DataFrame(trades)
    
    # 格式化
    if 'timestamp' in df.columns:
        df['时间'] = df['timestamp'].apply(lambda x: format_timestamp(x, fmt="%H:%M:%S"))
    
    if 'price' in df.columns:
        df['价格'] = df['price'].apply(format_price)
    
    if 'pnl' in df.columns:
        df['盈亏'] = df['pnl'].apply(format_pnl)
    
    st.dataframe(df, use_container_width=True, height=300)


def render_position_table(position: Dict, title: str = "当前持仓"):
    """
    渲染持仓表格
    
    Args:
        position: 持仓信息字典
        title: 标题
    """
    st.subheader(title)
    
    if not position:
        st.info("当前无持仓")
        return
    
    # 创建两列显示
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**基本信息**")
        for key in ['side', 'size', 'entry_price', 'current_price']:
            if key in position:
                label = {
                    'side': '方向',
                    'size': '持仓量',
                    'entry_price': '开仓价',
                    'current_price': '当前价'
                }.get(key, key)
                
                value = position[key]
                if 'price' in key:
                    value = format_price(value)
                elif key == 'size':
                    value = format_number(value)
                
                st.text(f"{label}: {value}")
    
    with col2:
        st.markdown("**盈亏信息**")
        for key in ['unrealized_pnl', 'pnl_pct', 'duration']:
            if key in position:
                label = {
                    'unrealized_pnl': '未实现盈亏',
                    'pnl_pct': '盈亏比例',
                    'duration': '持仓时长'
                }.get(key, key)
                
                value = position[key]
                if 'pnl' in key and key != 'pnl_pct':
                    value = format_pnl(value)
                elif key == 'pnl_pct':
                    value = format_percentage(value)
                
                st.text(f"{label}: {value}")


def render_factors_table(factors: Dict[str, float], title: str = "因子值"):
    """
    渲染因子表格
    
    Args:
        factors: 因子字典
        title: 标题
    """
    st.subheader(title)
    
    if not factors:
        st.info("暂无因子数据")
        return
    
    # 创建DataFrame
    df = pd.DataFrame([
        {'因子名称': k, '当前值': f"{v:.4f}"} 
        for k, v in factors.items()
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(400, 35 * len(df) + 38))


def render_performance_table(metrics: Dict[str, any], title: str = "性能统计"):
    """
    渲染性能统计表格
    
    Args:
        metrics: 性能指标字典
        title: 标题
    """
    st.subheader(title)
    
    if not metrics:
        st.info("暂无性能数据")
        return
    
    # 分组显示
    col1, col2 = st.columns(2)
    
    items = list(metrics.items())
    mid = len(items) // 2
    
    with col1:
        for key, value in items[:mid]:
            # 根据key类型格式化value
            if 'pnl' in key.lower():
                formatted_value = format_pnl(value)
            elif 'rate' in key.lower() or 'ratio' in key.lower():
                formatted_value = format_percentage(value) if isinstance(value, (int, float)) else value
            else:
                formatted_value = format_number(value) if isinstance(value, (int, float)) else value
            
            st.text(f"{key}: {formatted_value}")
    
    with col2:
        for key, value in items[mid:]:
            if 'pnl' in key.lower():
                formatted_value = format_pnl(value)
            elif 'rate' in key.lower() or 'ratio' in key.lower():
                formatted_value = format_percentage(value) if isinstance(value, (int, float)) else value
            else:
                formatted_value = format_number(value) if isinstance(value, (int, float)) else value
            
            st.text(f"{key}: {formatted_value}")


def render_simple_table(data: List[Dict], columns: Optional[List[str]] = None,
                       title: Optional[str] = None):
    """
    渲染简单表格
    
    Args:
        data: 数据列表
        columns: 要显示的列
        title: 标题
    """
    if title:
        st.subheader(title)
    
    if not data:
        st.info("暂无数据")
        return
    
    df = pd.DataFrame(data)
    
    if columns:
        df = df[columns] if all(c in df.columns for c in columns) else df
    
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_styled_table(df: pd.DataFrame, title: Optional[str] = None,
                       highlight_cols: Optional[List[str]] = None):
    """
    渲染带样式的表格
    
    Args:
        df: DataFrame数据
        title: 标题
        highlight_cols: 需要高亮的列
    """
    if title:
        st.subheader(title)
    
    if df is None or df.empty:
        st.info("暂无数据")
        return
    
    # 应用样式
    if highlight_cols:
        def highlight_values(row):
            styles = [''] * len(row)
            for idx, col in enumerate(df.columns):
                if col in highlight_cols:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        if val > 0:
                            styles[idx] = 'background-color: #d4edda'
                        elif val < 0:
                            styles[idx] = 'background-color: #f8d7da'
            return styles
        
        styled_df = df.style.apply(highlight_values, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def render_comparison_table(data1: Dict, data2: Dict, 
                           labels: tuple = ("当前", "之前"),
                           title: str = "对比表"):
    """
    渲染对比表格
    
    Args:
        data1: 第一组数据
        data2: 第二组数据
        labels: 标签元组
        title: 标题
    """
    st.subheader(title)
    
    if not data1 and not data2:
        st.info("暂无数据")
        return
    
    # 合并键
    all_keys = set(list(data1.keys()) + list(data2.keys()))
    
    comparison_data = []
    for key in all_keys:
        val1 = data1.get(key, '-')
        val2 = data2.get(key, '-')
        
        # 计算变化
        change = '-'
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val2 != 0:
                change = f"{((val1 - val2) / val2 * 100):+.2f}%"
        
        comparison_data.append({
            '指标': key,
            labels[0]: val1,
            labels[1]: val2,
            '变化': change
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
