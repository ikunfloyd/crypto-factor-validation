"""
图表组件
提供各种Plotly图表
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.theme import CHART_COLORS, PLOTLY_LAYOUT


def render_candlestick_chart(df: pd.DataFrame, title: str = "DOGE-USDT K线图"):
    """
    渲染K线图
    
    Args:
        df: 包含OHLC数据的DataFrame (timestamp, open, high, low, close)
        title: 图表标题
    """
    if df is None or df.empty:
        st.warning("暂无K线数据")
        return
    
    fig = go.Figure()
    
    # 添加K线
    fig.add_trace(go.Candlestick(
        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='DOGE-USDT',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # 如果有布林带数据，添加布林带
    if 'bollinger_upper' in df.columns and 'bollinger_lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
            y=df['bollinger_upper'],
            name='上轨',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
            y=df['bollinger_lower'],
            name='下轨',
            line=dict(color='rgba(0,255,0,0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='时间 (太平洋时区)',
        yaxis_title='价格 (USDT)',
        height=500,
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d\n%H:%M',  # 显示日期和时间
        ),
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_line_chart(df: pd.DataFrame, x_col: str, y_cols: Union[str, List[str]],
                     title: str = "时序图", x_label: str = "时间", y_label: str = "值"):
    """
    渲染折线图
    
    Args:
        df: 数据DataFrame
        x_col: X轴列名
        y_cols: Y轴列名（单个或多个）
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
    """
    if df is None or df.empty:
        st.warning("暂无数据")
        return
    
    fig = go.Figure()
    
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    
    for idx, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                name=col,
                mode='lines',
                line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        hovermode='x unified',
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_bar_chart(data: Dict[str, float], title: str = "柱状图",
                    x_label: str = "类别", y_label: str = "值",
                    orientation: str = 'v'):
    """
    渲染柱状图
    
    Args:
        data: 数据字典 {label: value}
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        orientation: 方向 ('v'垂直 或 'h'水平)
    """
    if not data:
        st.warning("暂无数据")
        return
    
    df = pd.DataFrame(list(data.items()), columns=['label', 'value'])
    
    if orientation == 'h':
        fig = px.bar(df, x='value', y='label', orientation='h',
                    title=title, color='value',
                    color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title=y_label, yaxis_title=x_label)
    else:
        fig = px.bar(df, x='label', y='value',
                    title=title, color='value',
                    color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    
    fig.update_layout(height=400, **PLOTLY_LAYOUT)
    
    st.plotly_chart(fig, use_container_width=True)


def render_pie_chart(data: Dict[str, float], title: str = "饼图"):
    """
    渲染饼图
    
    Args:
        data: 数据字典 {label: value}
        title: 图表标题
    """
    if not data:
        st.warning("暂无数据")
        return
    
    labels = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=CHART_COLORS[:len(labels)])
    )])
    
    fig.update_layout(
        title=title,
        height=400,
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(data: Dict[str, float], title: str = "雷达图"):
    """
    渲染雷达图（用于模型预测等）
    
    Args:
        data: 数据字典 {label: value}
        title: 图表标题
    """
    if not data:
        st.warning("暂无数据")
        return
    
    categories = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='预测值',
        line=dict(color=CHART_COLORS[0]),
        fillcolor=f'rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title,
        height=400,
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_gauge_chart(value: float, title: str = "仪表盘",
                      min_value: float = 0, max_value: float = 100,
                      thresholds: Optional[Dict] = None):
    """
    渲染仪表盘图表
    
    Args:
        value: 当前值
        title: 标题
        min_value: 最小值
        max_value: 最大值
        thresholds: 阈值 {'low': 33, 'medium': 66}
    """
    if thresholds is None:
        thresholds = {'low': 33, 'medium': 66}
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        delta={'reference': (min_value + max_value) / 2},
        gauge={
            'axis': {'range': [min_value, max_value]},
            'bar': {'color': CHART_COLORS[0]},
            'steps': [
                {'range': [min_value, thresholds.get('low', 33)], 'color': "lightgray"},
                {'range': [thresholds.get('low', 33), thresholds.get('medium', 66)], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds.get('medium', 66)
            }
        }
    ))
    
    fig.update_layout(height=300, **PLOTLY_LAYOUT)
    
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(data: pd.DataFrame, title: str = "热力图",
                  x_label: str = "X", y_label: str = "Y"):
    """
    渲染热力图
    
    Args:
        data: 数据DataFrame（2D矩阵）
        title: 标题
        x_label: X轴标签
        y_label: Y轴标签
    """
    if data is None or data.empty:
        st.warning("暂无数据")
        return
    
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdYlGn',
        text=data.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_area_chart(df: pd.DataFrame, x_col: str, y_cols: Union[str, List[str]],
                     title: str = "面积图"):
    """
    渲染面积图
    
    Args:
        df: 数据DataFrame
        x_col: X轴列名
        y_cols: Y轴列名（单个或多个）
        title: 图表标题
    """
    if df is None or df.empty:
        st.warning("暂无数据")
        return
    
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    
    fig = go.Figure()
    
    for idx, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                name=col,
                mode='lines',
                fill='tonexty' if idx > 0 else 'tozeroy',
                line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)], width=2)
            ))
    
    fig.update_layout(
        title=title,
        height=400,
        hovermode='x unified',
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                       title: str = "散点图", color_col: Optional[str] = None):
    """
    渲染散点图
    
    Args:
        df: 数据DataFrame
        x_col: X轴列名
        y_col: Y轴列名
        title: 标题
        color_col: 颜色分类列名
    """
    if df is None or df.empty:
        st.warning("暂无数据")
        return
    
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                    title=title, color_continuous_scale='Viridis')
    
    fig.update_layout(height=400, **PLOTLY_LAYOUT)
    
    st.plotly_chart(fig, use_container_width=True)


def render_volume_chart(df: pd.DataFrame, title: str = "成交量"):
    """
    渲染成交量柱状图
    
    Args:
        df: 包含volume数据的DataFrame
        title: 标题
    """
    if df is None or df.empty or 'volume' not in df.columns:
        st.warning("暂无成交量数据")
        return
    
    # 根据涨跌着色
    colors = ['red' if close < open_price else 'green' 
              for close, open_price in zip(df['close'], df['open'])]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
        y=df['volume'],
        name='成交量',
        marker_color=colors
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='时间',
        yaxis_title='成交量',
        height=200,
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_factor_bars(factors: Dict[str, float], title: str = "实时因子值"):
    """
    渲染因子柱状图
    
    Args:
        factors: 因子字典 {name: value}
        title: 标题
    """
    if not factors:
        st.warning("暂无因子数据")
        return
    
    df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Value'])
    
    # 根据值的正负着色
    colors = [CHART_COLORS[0] if v > 0 else CHART_COLORS[1] for v in df['Value']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Value'],
        y=df['Factor'],
        orientation='h',
        marker=dict(
            color=df['Value'],
            colorscale='RdYlGn',
            cmin=-1,
            cmax=1
        ),
        text=df['Value'].apply(lambda x: f"{x:.4f}"),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='因子值',
        yaxis_title='因子名称',
        height=max(300, len(factors) * 40),
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_pnl_curve(df: pd.DataFrame, title: str = "盈亏曲线"):
    """
    渲染盈亏曲线
    
    Args:
        df: 包含cumulative_pnl的DataFrame
        title: 标题
    """
    if df is None or df.empty:
        st.warning("暂无盈亏数据")
        return
    
    fig = go.Figure()
    
    # 累计盈亏线
    fig.add_trace(go.Scatter(
        x=df['timestamp'] if 'timestamp' in df.columns else df.index,
        y=df['cumulative_pnl'] if 'cumulative_pnl' in df.columns else df['pnl'],
        name='累计盈亏',
        mode='lines',
        fill='tozeroy',
        line=dict(color=CHART_COLORS[0], width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='时间',
        yaxis_title='盈亏 (USDT)',
        height=400,
        hovermode='x unified',
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_matrix(df: pd.DataFrame, title: str = "因子相关性矩阵"):
    """
    渲染相关性矩阵热力图
    
    Args:
        df: 数据DataFrame
        title: 标题
    """
    if df is None or df.empty:
        st.warning("暂无数据")
        return
    
    # 计算相关性
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmid=0,
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        **PLOTLY_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)
