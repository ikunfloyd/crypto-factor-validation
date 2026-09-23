"""
系统总览页面
显示系统整体状态和关键指标
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
    format_number, format_uptime, format_percentage,
    format_bytes, format_rate, format_latency
)
from dashboard.components.metrics import render_process_card, render_metric_card
from dashboard.components.charts import render_line_chart, render_area_chart
from dashboard.components.alerts import render_alerts_list, render_system_status

# 页面配置
st.set_page_config(
    page_title="系统总览 - DOGE Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# 应用自定义CSS
apply_custom_css()

# 初始化数据加载器
data_loader = get_data_loader()

# 页面标题
st.title("📊 系统总览")
st.markdown("---")

# 加载健康状态
try:
    health_metrics = data_loader.load_health_metrics()
    health_score = health_metrics.get('health_score', 0)
    uptime = health_metrics.get('uptime', 0)
    
    # 确定系统状态
    if health_score >= 80:
        system_status = 'healthy'
    elif health_score >= 60:
        system_status = 'warning'
    elif health_score > 0:
        system_status = 'error'
    else:
        system_status = 'unknown'
    
    # 显示系统状态
    status_message = f"健康评分: {health_score}/100 | 运行时长: {format_uptime(uptime)}"
    render_system_status(system_status, status_message)
    
except Exception as e:
    st.error(f"无法加载健康状态: {e}")
    health_score = 0
    uptime = 0

# 关键指标
st.subheader("⚡ 关键指标")

col1, col2, col3, col4, col5 = st.columns(5)

try:
    # 加载真实数据
    shm_data = data_loader.load_shared_memory_data()
    execution_state = data_loader.load_execution_state()
    
    with col1:
        # 数据流速（从共享内存tick数据计算）
        ticks = shm_data.get('ticks')
        if ticks is not None and len(ticks) > 1:
            time_diff = ticks[-1]['timestamp'] - ticks[0]['timestamp']
            data_rate = len(ticks) / time_diff if time_diff > 0 else 0
            render_metric_card("数据流速", format_rate(int(data_rate), per_second=True), "")
        else:
            render_metric_card("数据流速", "N/A", "")
    
    with col2:
        # 延迟（从tick时间戳计算）
        if ticks is not None and len(ticks) > 0:
            import time
            current_time = time.time()
            latest_tick_time = ticks[-1]['timestamp']
            latency = (current_time - latest_tick_time) * 1000  # 转为毫秒
            render_metric_card("平均延迟", format_latency(int(latency)), "", delta_color="inverse")
        else:
            render_metric_card("平均延迟", "N/A", "")
    
    with col3:
        # 健康评分
        render_metric_card("健康评分", f"{health_score}/100",
                          f"{health_score - 93:+d}" if health_score > 0 else "0")
    
    with col4:
        # 今日盈亏（真实数据）
        daily_pnl = execution_state.get('daily_pnl', 0)
        render_metric_card("今日盈亏", f"${daily_pnl:.2f}" if daily_pnl != 0 else "$0.00",
                          f"+{daily_pnl:.2f}" if daily_pnl > 0 else "")
    
    with col5:
        # 活跃订单（真实数据）
        active_orders = len(execution_state.get('orders', []))
        render_metric_card("活跃订单", str(active_orders), "")

except Exception as e:
    st.error(f"加载关键指标失败: {e}")

st.markdown("---")

# 进程状态
st.subheader("🖥️ 进程状态")

try:
    processes = data_loader.load_process_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        io_gateway = processes.get('io_gateway', {})
        render_process_card(
            "IO Gateway",
            status=io_gateway.get('status', 'unknown'),
            pid=io_gateway.get('pid'),
            cpu=io_gateway.get('cpu', 0),
            memory=io_gateway.get('memory', 0)
        )
    
    with col2:
        quant_core = processes.get('quant_core', {})
        render_process_card(
            "Quant Core",
            status=quant_core.get('status', 'unknown'),
            pid=quant_core.get('pid'),
            cpu=quant_core.get('cpu', 0),
            memory=quant_core.get('memory', 0)
        )
    
    with col3:
        execution = processes.get('execution', {})
        render_process_card(
            "Execution",
            status=execution.get('status', 'unknown'),
            pid=execution.get('pid'),
            cpu=execution.get('cpu', 0),
            memory=execution.get('memory', 0)
        )

except Exception as e:
    st.error(f"加载进程状态失败: {e}")

st.markdown("---")

# 实时监控图表
st.subheader("📈 实时监控")

tab1, tab2, tab3 = st.tabs(["数据流", "系统资源", "性能指标"])

with tab1:
    st.markdown("#### 数据流速（最近1小时）")
    try:
        # 尝试从因子历史获取真实数据流统计
        factors_history = data_loader.load_factor_history(hours=1)
        
        if not factors_history.empty and 'timestamp' in factors_history.columns:
            # 计算每分钟的数据点数作为数据流速
            factors_history['timestamp'] = pd.to_datetime(factors_history['timestamp'])
            factors_history['minute'] = factors_history['timestamp'].dt.floor('min')
            data_rate_df = factors_history.groupby('minute').size().reset_index(name='data_rate')
            data_rate_df.rename(columns={'minute': 'timestamp'}, inplace=True)
            
            render_line_chart(data_rate_df, 'timestamp', 'data_rate',
                             title="Tick数据流速", x_label="时间", y_label="条/分钟")
        else:
            st.info("暂无足够的历史数据显示数据流速趋势")
            
    except Exception as e:
        st.error(f"无法生成数据流图表: {e}")

with tab2:
    st.markdown("#### CPU使用率（最近1小时）")
    try:
        # 显示当前CPU使用率
        processes = data_loader.load_process_status()
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            io_cpu = processes.get('io_gateway', {}).get('cpu', 0)
            st.metric("IO Gateway", f"{io_cpu:.1f}%")
        with col_b:
            quant_cpu = processes.get('quant_core', {}).get('cpu', 0)
            st.metric("Quant Core", f"{quant_cpu:.1f}%")
        with col_c:
            exec_cpu = processes.get('execution', {}).get('cpu', 0)
            st.metric("Execution", f"{exec_cpu:.1f}%")
        
        st.info("CPU历史趋势需要启用性能监控记录功能")
        
    except Exception as e:
        st.error(f"无法获取CPU数据: {e}")

with tab3:
    st.markdown("#### 延迟统计（最近1小时）")
    try:
        # 显示当前延迟
        shm_data = data_loader.load_shared_memory_data()
        ticks = shm_data.get('ticks')
        
        if ticks is not None and len(ticks) > 0:
            import time
            current_time = time.time()
            latest_tick_time = ticks[-1]['timestamp']
            current_latency = (current_time - latest_tick_time) * 1000
            
            st.metric("当前延迟", f"{current_latency:.0f} ms")
            st.info("延迟历史趋势需要启用性能监控记录功能")
        else:
            st.warning("无法获取实时延迟数据")
            
    except Exception as e:
        st.error(f"无法获取延迟数据: {e}")

st.markdown("---")

# 系统资源统计
col1, col2 = st.columns(2)

with col1:
    st.subheader("💾 存储资源")
    try:
        wal_stats = data_loader.load_wal_stats()
        checkpoint_info = data_loader.load_checkpoint_info()
        
        st.metric("WAL日志", 
                 f"{wal_stats.get('total_size', 0):.2f} GB",
                 f"{wal_stats.get('file_count', 0)} 个文件")
        
        st.metric("检查点", 
                 f"{checkpoint_info.get('total_size', 0):.2f} MB",
                 f"{checkpoint_info.get('checkpoint_count', 0)} 个文件")
        
        latest_checkpoint = checkpoint_info.get('latest_checkpoint', 'N/A')
        st.info(f"最新检查点: {latest_checkpoint}")
        
    except Exception as e:
        st.error(f"加载存储资源失败: {e}")

with col2:
    st.subheader("🔔 最近告警")
    try:
        # 加载最近日志作为告警
        logs = data_loader.load_recent_logs(level="ALL", max_lines=20)
        
        # 过滤出警告和错误
        alerts = []
        for log in logs:
            level = log.get('level', 'INFO')
            if level in ['WARNING', 'ERROR']:
                alerts.append({
                    'level': level,
                    'message': log.get('message', ''),
                    'timestamp': log.get('timestamp', '')
                })
        
        if alerts:
            render_alerts_list(alerts, title="", max_alerts=5)
        else:
            st.success("✅ 无告警")
            
    except Exception as e:
        st.error(f"加载告警失败: {e}")

st.markdown("---")

# 页脚信息
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 自动刷新间隔: 5秒")
