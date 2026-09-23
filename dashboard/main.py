"""
DOGE-USDT 高频量化交易系统 - 实时监控仪表板
主入口文件
"""

import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.theme import PAGE_CONFIG, apply_custom_css
from dashboard.utils.data_loader import get_data_loader
from dashboard.utils.formatters import format_uptime, format_number
from dashboard.components.alerts import render_system_status

# 页面配置
st.set_page_config(**PAGE_CONFIG)

# 应用自定义CSS
apply_custom_css()

# 初始化数据加载器
data_loader = get_data_loader()

# 主标题
st.markdown("""
<div class="dashboard-title">
    <h1>🚀 DOGE-USDT 高频量化交易系统</h1>
    <h3>实时监控仪表板</h3>
</div>
""", unsafe_allow_html=True)

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
    
    # 显示系统状态横幅
    status_message = f"健康评分: {health_score}/100 | 运行时长: {format_uptime(uptime)}"
    render_system_status(system_status, status_message)
    
except Exception as e:
    st.warning(f"无法加载健康状态: {e}")

# 侧边栏
with st.sidebar:
    st.markdown("### 📊 监控模块")
    st.markdown("---")
    
    # 快速状态视图
    st.markdown("#### 快速状态")
    
    try:
        # 加载进程状态
        processes = data_loader.load_process_status()
        
        # 统计运行中的进程
        running_count = sum(1 for p in processes.values() if p.get('status') == 'running')
        total_count = len(processes)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("进程", f"{running_count}/{total_count}")
        with col2:
            st.metric("状态", "🟢" if running_count == total_count else "🟡")
        
        # WAL统计
        wal_stats = data_loader.load_wal_stats()
        if wal_stats.get('status') == 'active':
            st.metric("WAL大小", f"{wal_stats.get('total_size', 0):.2f} GB")
        
    except Exception as e:
        st.error(f"加载状态失败: {e}")
    
    st.markdown("---")
    
    # 导航提示
    st.markdown("""
    #### 📖 使用指南
    
    使用左侧导航栏切换到不同的监控页面：
    
    - **📊 系统总览**: 整体状态和关键指标
    - **📈 市场数据**: K线、深度、成交量
    - **🧠 量化引擎**: 因子、模型、信号
    - **💰 交易执行**: 持仓、订单、盈亏
    - **📋 日志查看**: 实时日志和错误追踪
    - **⚙️ 系统设置**: 配置和控制
    """)
    
    st.markdown("---")
    st.caption("Dashboard v1.0.0")

# 主内容区域
st.markdown("## 欢迎使用交易监控仪表板")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 系统监控
    
    实时监控系统的各个组件状态：
    - IO Gateway
    - Quant Core  
    - Execution Engine
    
    查看CPU、内存使用情况和错误日志。
    """)

with col2:
    st.markdown("""
    ### 📈 市场分析
    
    实时市场数据可视化：
    - K线图表
    - 订单簿深度
    - 成交量分析
    - 资金流向
    """)

with col3:
    st.markdown("""
    ### 💰 交易管理
    
    监控交易执行情况：
    - 当前持仓
    - 订单状态
    - 盈亏统计
    - 风险指标
    """)

st.markdown("---")

# 快速统计
st.markdown("## 📈 今日快览")

try:
    # 加载真实数据
    market_data = data_loader.load_market_data()
    execution_state = data_loader.load_execution_state()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price = market_data.get('price', 0)
        change = market_data.get('change_24h', 0)
        st.metric(
            "DOGE-USDT",
            f"${price:.5f}" if price > 0 else "N/A",
            f"{change:+.2f}%" if change != 0 else "0.00%"
        )
    
    with col2:
        position = execution_state.get('position')
        if position and position.get('size', 0) != 0:
            pnl = position.get('pnl', 0)
            pnl_pct = position.get('pnl_pct', 0)
            st.metric(
                "未实现盈亏",
                f"${pnl:.2f}",
                f"{pnl_pct:+.2f}%"
            )
        else:
            st.metric("未实现盈亏", "N/A", "")
    
    with col3:
        daily_pnl = execution_state.get('daily_pnl', 0)
        st.metric(
            "今日盈亏",
            f"${daily_pnl:.2f}" if daily_pnl != 0 else "$0.00",
            f"{daily_pnl:+.2f}" if daily_pnl != 0 else ""
        )
    
    with col4:
        win_rate = execution_state.get('win_rate', 0)
        st.metric(
            "胜率",
            f"{win_rate:.1f}%" if win_rate > 0 else "N/A",
            ""
        )

except Exception as e:
    st.error(f"加载数据失败: {e}")

st.markdown("---")

# 最近活动
st.markdown("## 📋 最近活动")

try:
    # 加载最近日志
    recent_logs = data_loader.load_recent_logs(level="ALL", max_lines=5)
    
    if recent_logs:
        for log in recent_logs:
            level = log.get('level', 'INFO')
            message = log.get('message', '')
            timestamp = log.get('timestamp', '')
            
            # 根据级别显示不同样式
            if level == 'ERROR':
                st.error(f"[{timestamp}] {message}")
            elif level == 'WARNING':
                st.warning(f"[{timestamp}] {message}")
            else:
                st.info(f"[{timestamp}] {message}")
    else:
        st.info("暂无最近活动记录")

except Exception as e:
    st.warning(f"无法加载最近活动: {e}")

st.markdown("---")

# 页脚
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🚀 DOGE-USDT 高频量化交易系统 | ⚡ 实时监控 | 📊 数据驱动</p>
    <p style="font-size: 0.85em;">请使用左侧导航栏访问各个监控模块</p>
</div>
""", unsafe_allow_html=True)
