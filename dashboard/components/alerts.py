"""
告警组件
提供各种告警和通知展示
"""

import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.formatters import format_timestamp, format_level_color


def render_alert(message: str, level: str = "info", icon: Optional[str] = None):
    """
    渲染单个告警
    
    Args:
        message: 告警消息
        level: 告警级别 (info, success, warning, error)
        icon: 自定义图标
    """
    level_map = {
        'info': (st.info, 'ℹ️'),
        'success': (st.success, '✅'),
        'warning': (st.warning, '⚠️'),
        'error': (st.error, '🔴'),
        'debug': (st.info, '🐛')
    }
    
    func, default_icon = level_map.get(level.lower(), (st.info, 'ℹ️'))
    display_icon = icon if icon else default_icon
    
    func(f"{display_icon} {message}")


def render_alerts_list(alerts: List[Dict], title: str = "🔔 最近告警",
                       max_alerts: int = 10):
    """
    渲染告警列表
    
    Args:
        alerts: 告警列表 [{'level': 'ERROR', 'message': '...', 'timestamp': '...'}]
        title: 标题
        max_alerts: 最大显示数量
    """
    st.subheader(title)
    
    if not alerts:
        st.info("暂无告警")
        return
    
    # 限制数量
    display_alerts = alerts[:max_alerts] if len(alerts) > max_alerts else alerts
    
    for alert in display_alerts:
        level = alert.get('level', 'INFO').upper()
        message = alert.get('message', '')
        timestamp = alert.get('timestamp', '')
        
        # 格式化时间戳
        if timestamp:
            try:
                ts_str = format_timestamp(timestamp, fmt="%H:%M:%S")
            except:
                ts_str = str(timestamp)
        else:
            ts_str = 'N/A'
        
        # 根据级别选择样式
        level_class = level.lower() if level.lower() in ['error', 'warning', 'info'] else 'info'
        
        alert_html = f"""
        <div class="alert-item {level_class}">
            <span class="timestamp">[{ts_str}]</span>
            <span><b>[{level}]</b> {message}</span>
        </div>
        """
        
        st.markdown(alert_html, unsafe_allow_html=True)
    
    if len(alerts) > max_alerts:
        st.caption(f"显示 {max_alerts}/{len(alerts)} 条告警")


def render_log_stream(logs: List[Dict], title: str = "📋 实时日志",
                     max_logs: int = 50, 
                     level_filter: Optional[str] = None):
    """
    渲染日志流
    
    Args:
        logs: 日志列表
        title: 标题
        max_logs: 最大显示数量
        level_filter: 日志级别过滤
    """
    st.subheader(title)
    
    if not logs:
        st.info("暂无日志")
        return
    
    # 过滤日志
    if level_filter and level_filter != "ALL":
        logs = [log for log in logs if log.get('level', '').upper() == level_filter.upper()]
    
    # 限制数量
    display_logs = logs[-max_logs:] if len(logs) > max_logs else logs
    
    # 使用容器以便滚动
    with st.container():
        for log in reversed(display_logs):  # 最新的在上面
            level = log.get('level', 'INFO')
            message = log.get('message', '')
            timestamp = log.get('timestamp', '')
            
            # 获取颜色
            color = format_level_color(level)
            
            # 格式化时间
            if timestamp:
                try:
                    ts_str = format_timestamp(timestamp, fmt="%H:%M:%S")
                except:
                    ts_str = str(timestamp)
            else:
                ts_str = datetime.now().strftime("%H:%M:%S")
            
            log_html = f"""
            <div style="
                font-family: monospace;
                font-size: 0.85em;
                padding: 4px 8px;
                margin: 2px 0;
                border-left: 3px solid {color};
                background: rgba(0,0,0,0.02);
                overflow-x: auto;
            ">
                <span style="color: #666;">[{ts_str}]</span>
                <span style="color: {color}; font-weight: bold;">[{level}]</span>
                <span>{message}</span>
            </div>
            """
            
            st.markdown(log_html, unsafe_allow_html=True)
    
    if len(logs) > max_logs:
        st.caption(f"显示最新 {max_logs} 条日志")


def render_notification(title: str, message: str, 
                        notification_type: str = "info",
                        auto_close: bool = False):
    """
    渲染通知卡片
    
    Args:
        title: 标题
        message: 消息内容
        notification_type: 类型 (info, success, warning, error)
        auto_close: 是否自动关闭
    """
    type_styles = {
        'info': ('ℹ️', '#3b82f6', '#dbeafe'),
        'success': ('✅', '#10b981', '#d1fae5'),
        'warning': ('⚠️', '#f59e0b', '#fef3c7'),
        'error': ('🔴', '#ef4444', '#fee2e2')
    }
    
    icon, border_color, bg_color = type_styles.get(notification_type, type_styles['info'])
    
    notification_html = f"""
    <div style="
        border: 3px solid {border_color};
        border-radius: 10px;
        padding: 18px;
        margin: 12px 0;
        background: {bg_color};
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.8em;">{icon}</span>
            <div style="flex: 1;">
                <h4 style="margin: 0 0 8px 0; font-weight: 700; color: #1f2937;">{title}</h4>
                <p style="margin: 0; color: #374151; font-weight: 500;">{message}</p>
            </div>
        </div>
    </div>
    """
    
    st.markdown(notification_html, unsafe_allow_html=True)


def render_status_timeline(events: List[Dict], title: str = "状态时间线"):
    """
    渲染状态时间线
    
    Args:
        events: 事件列表 [{'timestamp': '...', 'event': '...', 'status': '...'}]
        title: 标题
    """
    st.subheader(title)
    
    if not events:
        st.info("暂无事件")
        return
    
    for event in events:
        timestamp = event.get('timestamp', '')
        event_name = event.get('event', '')
        status = event.get('status', 'info')
        
        # 格式化时间
        if timestamp:
            try:
                ts_str = format_timestamp(timestamp)
            except:
                ts_str = str(timestamp)
        else:
            ts_str = 'N/A'
        
        # 状态图标
        status_icons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'running': '▶️',
            'stopped': '⏸️'
        }
        icon = status_icons.get(status.lower(), 'ℹ️')
        
        timeline_html = f"""
        <div style="
            border-left: 3px solid #ddd;
            padding-left: 20px;
            margin: 10px 0;
            position: relative;
        ">
            <div style="
                position: absolute;
                left: -8px;
                top: 5px;
                background: white;
                font-size: 1.2em;
            ">{icon}</div>
            <div style="color: #666; font-size: 0.85em;">{ts_str}</div>
            <div style="font-weight: 600; margin-top: 3px;">{event_name}</div>
        </div>
        """
        
        st.markdown(timeline_html, unsafe_allow_html=True)


def render_error_summary(errors: List[Dict], title: str = "错误汇总"):
    """
    渲染错误汇总
    
    Args:
        errors: 错误列表
        title: 标题
    """
    st.subheader(title)
    
    if not errors:
        st.success("✅ 没有错误")
        return
    
    # 按类型分组
    error_types = {}
    for error in errors:
        error_type = error.get('type', 'Unknown')
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)
    
    # 显示每种错误类型
    for error_type, error_list in error_types.items():
        with st.expander(f"🔴 {error_type} ({len(error_list)})", expanded=False):
            for error in error_list[:10]:  # 每种类型最多显示10个
                timestamp = error.get('timestamp', '')
                message = error.get('message', '')
                
                if timestamp:
                    try:
                        ts_str = format_timestamp(timestamp, fmt="%H:%M:%S")
                    except:
                        ts_str = str(timestamp)
                else:
                    ts_str = 'N/A'
                
                st.text(f"[{ts_str}] {message}")
            
            if len(error_list) > 10:
                st.caption(f"还有 {len(error_list) - 10} 个类似错误...")


def render_system_status(status: str, message: str = ""):
    """
    渲染系统状态横幅
    
    Args:
        status: 状态 (healthy, warning, error, unknown)
        message: 状态消息
    """
    status_config = {
        'healthy': ('🟢', '系统运行正常', 'success', '#d1fae5'),
        'warning': ('🟡', '系统存在警告', 'warning', '#fef3c7'),
        'error': ('🔴', '系统存在错误', 'error', '#fee2e2'),
        'degraded': ('🟡', '系统性能下降', 'warning', '#fef3c7'),
        'unknown': ('⚪', '系统状态未知', 'info', '#dbeafe')
    }
    
    icon, default_msg, alert_type, bg_color = status_config.get(status.lower(), status_config['unknown'])
    display_message = message if message else default_msg
    
    banner_html = f"""
    <div style="
        background: {bg_color};
        border-radius: 12px;
        padding: 20px 24px;
        margin: 18px 0;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        border: 2px solid {'#10b981' if status == 'healthy' else '#f59e0b' if status in ['warning', 'degraded'] else '#ef4444' if status == 'error' else '#6b7280'};
    ">
        <span style="font-size: 2.5em;">{icon}</span>
        <div style="flex: 1;">
            <div style="font-size: 1.3em; font-weight: 700; color: #1f2937;">
                系统状态: {status.upper()}
            </div>
            <div style="margin-top: 8px; font-weight: 600; color: #374151;">
                {display_message}
            </div>
        </div>
    </div>
    """
    
    st.markdown(banner_html, unsafe_allow_html=True)


def render_metric_alert(metric_name: str, current_value: float,
                       threshold: float, condition: str = "above"):
    """
    渲染指标告警
    
    Args:
        metric_name: 指标名称
        current_value: 当前值
        threshold: 阈值
        condition: 条件 ("above" 或 "below")
    """
    is_alert = (condition == "above" and current_value > threshold) or \
               (condition == "below" and current_value < threshold)
    
    if is_alert:
        alert_html = f"""
        <div style="
            background: #fee2e2;
            border: 3px solid #ef4444;
            border-radius: 10px;
            padding: 14px 16px;
            margin: 10px 0;
            box-shadow: 0 3px 10px rgba(239,68,68,0.2);
        ">
            <span style="font-size: 1.3em;">⚠️</span>
            <b style="color: #7f1d1d;">{metric_name}</b>: 当前值 <b style="color: #991b1b;">{current_value:.2f}</b>
            {'超过' if condition == 'above' else '低于'}阈值
            <b style="color: #991b1b;">{threshold:.2f}</b>
        </div>
        """
        
        st.markdown(alert_html, unsafe_allow_html=True)
