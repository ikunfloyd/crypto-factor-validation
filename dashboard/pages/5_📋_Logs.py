"""
日志查看页面
显示实时日志流、错误追踪、性能日志等
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.utils.theme import apply_custom_css
from dashboard.utils.data_loader import get_data_loader
from dashboard.components.alerts import render_log_stream, render_error_summary
from dashboard.components.charts import render_pie_chart

# 页面配置
st.set_page_config(
    page_title="日志查看 - DOGE Trading Dashboard",
    page_icon="📋",
    layout="wide"
)

# 应用自定义CSS
apply_custom_css()

# 初始化数据加载器
data_loader = get_data_loader()

# 页面标题
st.title("📋 系统日志查看")
st.markdown("---")

# 日志过滤选项
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    log_level = st.selectbox(
        "日志级别",
        ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
        index=0
    )

with col2:
    log_module = st.selectbox(
        "模块",
        ["ALL", "io_gateway", "quant_core", "execution", "orchestrator"],
        index=0
    )

with col3:
    max_lines = st.slider("显示行数", 10, 200, 50, 10)

st.markdown("---")

# 日志统计
col1, col2, col3, col4 = st.columns(4)

try:
    # 加载所有日志进行统计
    all_logs = data_loader.load_recent_logs(level="ALL", max_lines=500)
    
    # 统计各级别日志数量
    log_counts = {
        'DEBUG': 0,
        'INFO': 0,
        'WARNING': 0,
        'ERROR': 0
    }
    
    for log in all_logs:
        level = log.get('level', 'INFO')
        if level in log_counts:
            log_counts[level] += 1
    
    with col1:
        st.metric("总日志", len(all_logs), "")
    
    with col2:
        st.metric("INFO", log_counts['INFO'], "")
    
    with col3:
        st.metric("WARNING", log_counts['WARNING'], "")
    
    with col4:
        st.metric("ERROR", log_counts['ERROR'], "")

except Exception as e:
    st.error(f"无法加载日志统计: {e}")

st.markdown("---")

# 日志分布图表
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 日志级别分布")
    try:
        if all_logs:
            # 过滤掉0值
            non_zero_counts = {k: v for k, v in log_counts.items() if v > 0}
            if non_zero_counts:
                render_pie_chart(non_zero_counts, title="")
            else:
                st.info("暂无日志数据")
        else:
            st.info("暂无日志数据")
    except Exception as e:
        st.error(f"无法生成图表: {e}")

with col2:
    st.subheader("🔍 快速搜索")
    search_query = st.text_input("搜索日志内容", "")
    
    if search_query:
        st.info(f"正在搜索: {search_query}")
        # 这里可以添加搜索逻辑
    else:
        st.caption("输入关键词搜索日志")

st.markdown("---")

# 实时日志流
st.subheader("📜 实时日志流")

try:
    # 加载日志
    logs = data_loader.load_recent_logs(level=log_level if log_level != "ALL" else "ALL", 
                                        max_lines=max_lines)
    
    # 如果有模块过滤
    if log_module != "ALL":
        # 这里可以添加模块过滤逻辑
        pass
    
    # 如果有搜索
    if search_query:
        logs = [log for log in logs if search_query.lower() in log.get('message', '').lower()]
    
    if logs:
        render_log_stream(logs, title="", max_logs=max_lines, level_filter=None)
    else:
        st.info("未找到符合条件的日志")

except Exception as e:
    st.error(f"无法加载日志: {e}")

st.markdown("---")

# 错误汇总
st.subheader("🔴 错误汇总")

try:
    # 从日志中提取错误
    error_logs = [log for log in all_logs if log.get('level') == 'ERROR']
    
    if error_logs:
        # 转换为错误格式
        errors = []
        for log in error_logs:
            # 简单分类错误类型
            message = log.get('message', '')
            error_type = 'Unknown'
            
            if 'connection' in message.lower() or 'timeout' in message.lower():
                error_type = 'Connection Error'
            elif 'permission' in message.lower() or 'access' in message.lower():
                error_type = 'Permission Error'
            elif 'file' in message.lower() or 'path' in message.lower():
                error_type = 'File Error'
            elif 'memory' in message.lower():
                error_type = 'Memory Error'
            elif 'data' in message.lower() or 'parse' in message.lower():
                error_type = 'Data Error'
            else:
                error_type = 'Runtime Error'
            
            errors.append({
                'type': error_type,
                'message': message,
                'timestamp': log.get('timestamp', '')
            })
        
        render_error_summary(errors, title="")
    else:
        st.success("✅ 没有错误日志")

except Exception as e:
    st.error(f"无法加载错误汇总: {e}")

st.markdown("---")

# 性能日志
st.subheader("⚡ 性能日志")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 模块性能统计")
    
    # 模拟性能数据
    perf_data = {
        'io_gateway': {'avg_latency': 45, 'throughput': 1250},
        'quant_core': {'avg_latency': 120, 'throughput': 800},
        'execution': {'avg_latency': 35, 'throughput': 500}
    }
    
    for module, metrics in perf_data.items():
        with st.expander(f"📦 {module}", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("平均延迟", f"{metrics['avg_latency']} ms")
            with col_b:
                st.metric("吞吐量", f"{metrics['throughput']} 条/秒")

with col2:
    st.markdown("#### 最近慢查询")
    
    # 模拟慢查询数据
    slow_queries = [
        {'operation': 'Factor计算', 'duration': 230, 'timestamp': '12:35:20'},
        {'operation': '数据库查询', 'duration': 180, 'timestamp': '12:30:15'},
        {'operation': '模型推理', 'duration': 150, 'timestamp': '12:25:10'}
    ]
    
    for query in slow_queries:
        st.text(f"[{query['timestamp']}] {query['operation']}: {query['duration']}ms")

st.markdown("---")

# 日志导出
st.subheader("💾 日志导出")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("导出当前视图"):
        st.info("日志导出功能将在后续版本中实现")

with col2:
    if st.button("导出错误日志"):
        st.info("错误日志导出功能将在后续版本中实现")

with col3:
    if st.button("清理旧日志"):
        st.warning("日志清理功能需要管理员权限")

# 页脚
st.markdown("---")
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 自动刷新: 开启")
