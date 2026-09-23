"""
指标卡片组件
提供各种指标展示卡片
"""

import streamlit as st
from typing import Union, Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.formatters import (
    format_number, format_percentage, format_price,
    format_status_emoji, color_by_value
)


def render_metric_card(title: str, value: Union[str, int, float], 
                       delta: Optional[Union[str, int, float]] = None,
                       delta_color: str = "normal"):
    """
    渲染指标卡片
    
    Args:
        title: 指标名称
        value: 指标值
        delta: 变化量
        delta_color: 变化颜色 ("normal", "inverse", "off")
    """
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)


def render_process_card(name: str, status: str, pid: Optional[int] = None,
                       cpu: float = 0.0, memory: float = 0.0):
    """
    渲染进程状态卡片
    
    Args:
        name: 进程名称
        status: 进程状态
        pid: 进程ID
        cpu: CPU使用率
        memory: 内存使用量(MB)
    """
    status_emoji = format_status_emoji(status)
    status_class = status.lower() if status else 'unknown'
    
    # 根据状态设置颜色 - 使用高对比度配色
    status_colors = {
        'running': '#10b981',
        'stopped': '#ef4444',
        'unknown': '#6b7280'
    }
    status_color = status_colors.get(status_class, '#6b7280')
    
    card_html = f"""
    <div class="process-card {status_class}">
        <h4 style="color: {status_color}; font-size: 1.3em; margin: 0 0 15px 0; font-weight: 700;">
            {status_emoji} {name}
        </h4>
        <div class="info-row" style="margin: 8px 0;">
            <span style="font-weight: 700; color: #1f2937;">状态:</span>
            <span style="font-weight: 700; color: {status_color}; font-size: 1.1em;">
                {status.upper()}
            </span>
        </div>
        <div class="info-row" style="margin: 8px 0;">
            <span style="font-weight: 700; color: #1f2937;">PID:</span>
            <span style="font-weight: 600; color: #374151; font-family: 'Monaco', 'Menlo', monospace;">
                {pid if pid else 'N/A'}
            </span>
        </div>
        <div class="info-row" style="margin: 8px 0;">
            <span style="font-weight: 700; color: #1f2937;">CPU:</span>
            <span style="font-weight: 600; color: #374151; font-family: 'Monaco', 'Menlo', monospace;">
                {cpu:.1f}%
            </span>
        </div>
        <div class="info-row" style="margin: 8px 0;">
            <span style="font-weight: 700; color: #1f2937;">内存:</span>
            <span style="font-weight: 600; color: #374151; font-family: 'Monaco', 'Menlo', monospace;">
                {memory:.0f} MB
            </span>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_kpi_card(title: str, value: Union[str, int, float],
                   subtitle: Optional[str] = None,
                   icon: str = "📊"):
    """
    渲染KPI卡片
    
    Args:
        title: 标题
        value: 值
        subtitle: 副标题
        icon: 图标
    """
    subtitle_html = f"<p style='margin:0; font-size:0.9em; opacity:0.8;'>{subtitle}</p>" if subtitle else ""
    
    card_html = f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <div class="value">{value}</div>
        {subtitle_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_progress_bar(label: str, value: float, max_value: float = 100.0,
                       show_percentage: bool = True):
    """
    渲染进度条
    
    Args:
        label: 标签
        value: 当前值
        max_value: 最大值
        show_percentage: 是否显示百分比
    """
    percentage = (value / max_value * 100) if max_value > 0 else 0
    percentage = min(100, max(0, percentage))  # 限制在0-100之间
    
    display_text = f"{percentage:.0f}%" if show_percentage else f"{value:.1f}/{max_value:.1f}"
    
    progress_html = f"""
    <div style="margin: 10px 0;">
        <div style="font-size: 0.9em; margin-bottom: 5px; font-weight: 600;">{label}</div>
        <div class="progress-bar">
            <div class="fill" style="width: {percentage}%;">
                {display_text}
            </div>
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)


def render_status_badge(status: str, text: Optional[str] = None):
    """
    渲染状态徽章
    
    Args:
        status: 状态类型 (success, danger, warning, info)
        text: 显示文本
    """
    if text is None:
        text = status
    
    badge_html = f"""
    <span class="badge {status}">{text}</span>
    """
    
    st.markdown(badge_html, unsafe_allow_html=True)


def render_stat_box(title: str, value: Union[str, int, float],
                   change: Optional[float] = None,
                   trend: str = "neutral"):
    """
    渲染统计框
    
    Args:
        title: 标题
        value: 值
        change: 变化量
        trend: 趋势 ("up", "down", "neutral")
    """
    trend_icon = {"up": "↑", "down": "↓", "neutral": "→"}.get(trend, "→")
    trend_color = {"up": "green", "down": "red", "neutral": "gray"}.get(trend, "gray")
    
    change_html = ""
    if change is not None:
        change_html = f"<span style='color:{trend_color}; font-size:0.9em;'>{trend_icon} {change:+.2f}%</span>"
    
    stat_html = f"""
    <div style="background:white; padding:18px; border-radius:10px; border:2px solid #e5e7eb; text-align:center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
        <div style="font-size:0.9em; color:#6b7280; margin-bottom:8px; font-weight:600;">{title}</div>
        <div style="font-size:2em; font-weight:bold; margin:8px 0; color:#1f2937;">{value}</div>
        {change_html}
    </div>
    """
    
    st.markdown(stat_html, unsafe_allow_html=True)


def render_info_row(label: str, value: Union[str, int, float],
                   highlight: bool = False):
    """
    渲染信息行
    
    Args:
        label: 标签
        value: 值
        highlight: 是否高亮
    """
    bg_color = "#dbeafe" if highlight else "transparent"
    
    row_html = f"""
    <div style="background:{bg_color}; padding:10px; margin:5px 0; border-radius:6px; display:flex; justify-content:space-between; border: 1px solid {'#93c5fd' if highlight else 'transparent'};">
        <span style="font-weight:700; color:#1f2937;">{label}:</span>
        <span style="font-weight:600; color:#374151;">{value}</span>
    </div>
    """
    
    st.markdown(row_html, unsafe_allow_html=True)


def render_grid_metrics(metrics: dict, columns: int = 4):
    """
    渲染网格指标
    
    Args:
        metrics: 指标字典 {name: value}
        columns: 列数
    """
    cols = st.columns(columns)
    
    for idx, (name, value) in enumerate(metrics.items()):
        with cols[idx % columns]:
            st.metric(label=name, value=value)


def render_color_coded_value(value: float, thresholds: dict = None):
    """
    根据阈值渲染彩色编码的值
    
    Args:
        value: 值
        thresholds: 阈值字典 {'good': (min, max), 'warning': (min, max), 'danger': (min, max)}
    """
    if thresholds is None:
        thresholds = {
            'good': (80, 100),
            'warning': (60, 80),
            'danger': (0, 60)
        }
    
    color = 'gray'
    for status, (min_val, max_val) in thresholds.items():
        if min_val <= value <= max_val:
            if status == 'good':
                color = 'green'
            elif status == 'warning':
                color = 'orange'
            elif status == 'danger':
                color = 'red'
            break
    
    value_html = f"""
    <div style="font-size:1.8em; font-weight:bold; color:{color};">
        {value:.1f}
    </div>
    """
    
    st.markdown(value_html, unsafe_allow_html=True)


def render_comparison_metrics(title: str, current: float, previous: float,
                             format_func=None):
    """
    渲染对比指标
    
    Args:
        title: 标题
        current: 当前值
        previous: 之前值
        format_func: 格式化函数
    """
    if format_func is None:
        format_func = lambda x: f"{x:.2f}"
    
    change = ((current - previous) / previous * 100) if previous != 0 else 0
    change_color = color_by_value(change)
    
    comp_html = f"""
    <div style="padding:18px; background:white; border-radius:10px; border:2px solid #e5e7eb; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
        <div style="font-size:0.95em; color:#6b7280; margin-bottom:12px; font-weight:600;">{title}</div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:1.3em; font-weight:bold; color:#1f2937;">{format_func(current)}</div>
                <div style="font-size:0.85em; color:#6b7280; font-weight:600;">当前</div>
            </div>
            <div style="font-size:2em; color:#d1d5db;">→</div>
            <div>
                <div style="font-size:1.1em; color:#374151; font-weight:600;">{format_func(previous)}</div>
                <div style="font-size:0.85em; color:#6b7280; font-weight:600;">之前</div>
            </div>
            <div style="color:{change_color}; font-weight:bold; font-size:1.1em;">
                {change:+.1f}%
            </div>
        </div>
    </div>
    """
    
    st.markdown(comp_html, unsafe_allow_html=True)
