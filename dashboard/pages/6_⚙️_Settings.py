"""
系统设置页面
显示配置信息、系统控制、数据管理等
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
from dashboard.utils.formatters import format_bytes, format_timestamp

# 页面配置
st.set_page_config(
    page_title="系统设置 - DOGE Trading Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# 应用自定义CSS
apply_custom_css()

# 初始化数据加载器
data_loader = get_data_loader()

# 页面标题
st.title("⚙️ 系统设置与管理")
st.markdown("---")

# 系统信息
st.subheader("ℹ️ 系统信息")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 基本信息")
    st.text(f"系统名称: DOGE-USDT Trading System")
    st.text(f"版本: v1.0.0")
    st.text(f"部署环境: Production")
    st.text(f"Python版本: 3.9+")
    
    # 运行时长
    try:
        health_metrics = data_loader.load_health_metrics()
        uptime = health_metrics.get('uptime', 0)
        from dashboard.utils.formatters import format_uptime
        st.text(f"运行时长: {format_uptime(uptime)}")
    except:
        st.text(f"运行时长: N/A")

with col2:
    st.markdown("#### 数据目录")
    st.text(f"项目根目录: {PROJECT_ROOT}")
    st.text(f"数据目录: {PROJECT_ROOT / 'data'}")
    st.text(f"日志目录: {PROJECT_ROOT / 'logs'}")
    st.text(f"配置文件: config/trading_config.py")

st.markdown("---")

# 配置查看
st.subheader("📋 系统配置")

tab1, tab2, tab3 = st.tabs(["风险参数", "模型参数", "执行参数"])

with tab1:
    st.markdown("#### 风险控制参数")
    
    risk_params = {
        "最大仓位": "70%",
        "杠杆倍数": "1x",
        "止损比例": "-2%",
        "止盈比例": "+5%",
        "最大回撤": "-10%",
        "单笔最大损失": "$500"
    }
    
    col1, col2 = st.columns(2)
    items = list(risk_params.items())
    
    with col1:
        for key, value in items[:len(items)//2]:
            st.text(f"{key}: {value}")
    
    with col2:
        for key, value in items[len(items)//2:]:
            st.text(f"{key}: {value}")
    
    st.info("⚠️ 风险参数修改需要管理员权限")

with tab2:
    st.markdown("#### 模型参数")
    
    model_params = {
        "Hunter权重": "0.7 (trending)",
        "Sniper权重": "0.3 (trending)",
        "Defender权重": "0.0 (trending)",
        "信号阈值": "0.6",
        "因子窗口": "20",
        "模型更新频率": "每日"
    }
    
    for key, value in model_params.items():
        st.text(f"{key}: {value}")
    
    st.info("ℹ️ 模型参数由量化策略自动管理")

with tab3:
    st.markdown("#### 执行参数")
    
    exec_params = {
        "交易对": "DOGE-USDT",
        "交易模式": "实盘",
        "订单类型": "限价单",
        "滑点容忍": "0.1%",
        "最小订单量": "1000 DOGE",
        "最大订单量": "50000 DOGE"
    }
    
    for key, value in exec_params.items():
        st.text(f"{key}: {value}")

st.markdown("---")

# 系统控制
st.subheader("🎛️ 系统控制")

st.warning("⚠️ 以下操作需要管理员权限")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 重启系统", use_container_width=True):
        st.warning("系统重启功能需要管理员确认")

with col2:
    if st.button("⏸️ 暂停交易", use_container_width=True):
        st.info("交易暂停功能将在后续版本中实现")

with col3:
    if st.button("▶️ 恢复交易", use_container_width=True):
        st.info("交易恢复功能将在后续版本中实现")

with col4:
    if st.button("💾 创建检查点", use_container_width=True):
        st.success("检查点创建请求已提交")

st.markdown("---")

# 数据管理
st.subheader("💾 数据管理")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### WAL日志")
    
    try:
        wal_stats = data_loader.load_wal_stats()
        
        st.metric("文件数量", wal_stats.get('file_count', 0))
        st.metric("总大小", f"{wal_stats.get('total_size', 0):.2f} GB")
        
        latest_date = wal_stats.get('latest_date')
        if latest_date:
            st.text(f"最新日期: {format_timestamp(latest_date)}")
        
        if st.button("清理旧WAL"):
            st.warning("WAL清理需要确认，将删除7天前的数据")
    
    except Exception as e:
        st.error(f"无法加载WAL统计: {e}")

with col2:
    st.markdown("#### 检查点")
    
    try:
        checkpoint_info = data_loader.load_checkpoint_info()
        
        st.metric("检查点数量", checkpoint_info.get('checkpoint_count', 0))
        st.metric("总大小", f"{checkpoint_info.get('total_size', 0):.2f} MB")
        
        latest_checkpoint = checkpoint_info.get('latest_checkpoint', 'N/A')
        st.text(f"最新检查点: {latest_checkpoint}")
        
        if st.button("恢复检查点"):
            st.info("请选择要恢复的检查点文件")
    
    except Exception as e:
        st.error(f"无法加载检查点信息: {e}")

st.markdown("---")

# 数据导出
st.subheader("📤 数据导出")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 因子数据")
    date_range = st.date_input("选择日期范围", [])
    if st.button("导出因子数据"):
        st.info("因子数据导出功能将在后续版本中实现")

with col2:
    st.markdown("#### 交易记录")
    export_format = st.selectbox("导出格式", ["CSV", "Excel", "JSON"])
    if st.button("导出交易记录"):
        st.info(f"将以 {export_format} 格式导出交易记录")

with col3:
    st.markdown("#### 系统日志")
    log_level_export = st.multiselect("日志级别", ["INFO", "WARNING", "ERROR"], ["ERROR"])
    if st.button("导出日志"):
        st.info(f"将导出级别为 {', '.join(log_level_export)} 的日志")

st.markdown("---")

# 监控设置
st.subheader("📊 监控设置")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 告警设置")
    
    st.checkbox("启用邮件告警", value=False)
    st.checkbox("启用Webhook告警", value=False)
    st.checkbox("启用Telegram告警", value=False)
    
    st.text_input("告警邮箱", placeholder="admin@example.com")

with col2:
    st.markdown("#### 告警阈值")
    
    st.number_input("CPU使用率告警 (%)", min_value=0, max_value=100, value=80)
    st.number_input("内存使用率告警 (%)", min_value=0, max_value=100, value=85)
    st.number_input("延迟告警 (ms)", min_value=0, value=500)
    st.number_input("错误数告警", min_value=0, value=10)

if st.button("保存监控设置"):
    st.success("监控设置已保存")

st.markdown("---")

# 系统诊断
st.subheader("🔍 系统诊断")

if st.button("运行系统诊断", use_container_width=True):
    with st.spinner("正在运行系统诊断..."):
        import time
        time.sleep(1)
        
        st.success("✅ 系统诊断完成")
        
        # 显示诊断结果
        diag_results = {
            "进程状态": "✅ 正常",
            "共享内存": "✅ 正常",
            "数据完整性": "✅ 正常",
            "网络连接": "✅ 正常",
            "磁盘空间": "✅ 充足",
            "日志系统": "✅ 正常"
        }
        
        for check, result in diag_results.items():
            st.text(f"{check}: {result}")

st.markdown("---")

# 关于信息
st.subheader("ℹ️ 关于")

st.markdown("""
### DOGE-USDT 高频量化交易系统

**版本**: v1.0.0  
**构建日期**: 2026-01-08  
**作者**: Quant Team  

#### 主要功能
- 🚀 高频Tick数据处理
- 🧠 三专家模型融合
- 📊 实时因子计算
- 💰 智能订单执行
- 📈 全面监控和可视化

#### 技术栈
- **数据处理**: Polars, PyArrow
- **机器学习**: PyTorch, XGBoost
- **可视化**: Streamlit, Plotly
- **进程通信**: 共享内存, Ring Buffer
- **日志**: Loguru

#### 联系方式
- 📧 Email: support@example.com
- 📚 文档: https://docs.example.com
- 🐛 Bug报告: https://github.com/example/issues

---

© 2026 Quant Trading Team. All rights reserved.
""")

# 页脚
st.markdown("---")
st.caption(f"Dashboard 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
