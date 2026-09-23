"""
量化引擎页面
显示因子分析、模型预测、信号生成等
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
from dashboard.utils.formatters import format_regime, format_signal
from dashboard.components.charts import render_radar_chart, render_factor_bars, render_line_chart
from dashboard.components.tables import render_factors_table

# 页面配置
st.set_page_config(
    page_title="量化引擎 - DOGE Trading Dashboard",
    page_icon="🧠",
    layout="wide"
)

# 应用自定义CSS
apply_custom_css()

# 初始化数据加载器
data_loader = get_data_loader()

# 页面标题
st.title("🧠 量化引擎 - 因子与模型分析")
st.markdown("---")

# 加载真实数据
shm_data = data_loader.load_shared_memory_data()
factors = shm_data.get('factors', {})
model_predictions = data_loader.load_model_predictions()

# 当前状态
col1, col2, col3, col4 = st.columns(4)

with col1:
    regime = model_predictions.get('regime', 'unknown')
    st.metric("市场状态", format_regime(regime), "")

with col2:
    signal = model_predictions.get('signal', 'FLAT')
    st.metric("当前信号", format_signal(signal), "")

with col3:
    # 融合得分
    fusion_score = model_predictions.get('fusion_score', 0.0)
    st.metric("融合得分", f"{fusion_score:.2f}" if fusion_score > 0 else "N/A", "")

with col4:
    confidence = model_predictions.get('confidence', 0.0)
    st.metric("信号置信度", f"{confidence*100:.0f}%" if confidence > 0 else "N/A", "")

st.markdown("---")

# 因子分析
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 实时因子值")
    
    try:
        # 尝试加载实际因子数据
        shm_data = data_loader.load_shared_memory_data()
        actual_factors = shm_data.get('factors')
        
        if actual_factors:
            render_factor_bars(actual_factors, title="")
        else:
            # 使用模拟数据
            render_factor_bars(factors, title="")
        
    except Exception as e:
        st.error(f"无法加载因子数据: {e}")
        render_factor_bars(factors, title="")

with col2:
    st.subheader("📋 因子列表")
    
    try:
        # 显示因子表格
        render_factors_table(factors, title="")
        
    except Exception as e:
        st.error(f"无法显示因子表格: {e}")

st.markdown("---")

# 模型预测
st.subheader("🎯 三专家模型预测")

col1, col2 = st.columns([1, 2])

with col1:
    # 雷达图
    try:
        models = {
            'Hunter': model_predictions.get('Hunter', 0.0),
            'Sniper': model_predictions.get('Sniper', 0.0),
            'Defender': model_predictions.get('Defender', 0.0)
        }
        if any(v > 0 for v in models.values()):
            render_radar_chart(models, title="模型预测雷达图")
        else:
            st.info("暂无模型预测数据")
    except Exception as e:
        st.error(f"无法生成雷达图: {e}")

with col2:
    # 模型详情
    st.markdown("### 模型预测详情")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**🎯 Hunter (趋势捕捉)**")
        hunter_score = model_predictions.get('Hunter', 0.0)
        if hunter_score > 0:
            st.progress(hunter_score)
            st.metric("预测值", f"{hunter_score:.2f}", "")
        else:
            st.metric("预测值", "N/A", "")
        st.caption("专注于捕捉中长期趋势")
    
    with col_b:
        st.markdown("**🎯 Sniper (精准入场)**")
        sniper_score = model_predictions.get('Sniper', 0.0)
        if sniper_score > 0:
            st.progress(sniper_score)
            st.metric("预测值", f"{sniper_score:.2f}", "")
        else:
            st.metric("预测值", "N/A", "")
        st.caption("专注于寻找最佳入场点")
    
    with col_c:
        st.markdown("**🛡️ Defender (风险防御)**")
        defender_score = model_predictions.get('Defender', 0.0)
        if defender_score > 0:
            st.progress(defender_score)
            st.metric("预测值", f"{defender_score:.2f}", "")
        else:
            st.metric("预测值", "N/A", "")
        st.caption("专注于风险控制和止损")

st.markdown("---")

# 门控网络权重
st.subheader("⚖️ 门控网络权重分配")

col1, col2 = st.columns([2, 1])

with col1:
    # 根据regime显示权重分配
    regime_weights = {
        'trending_up': {'Hunter': 0.7, 'Sniper': 0.3, 'Defender': 0.0},
        'trending_down': {'Hunter': 0.7, 'Sniper': 0.2, 'Defender': 0.1},
        'ranging': {'Hunter': 0.2, 'Sniper': 0.6, 'Defender': 0.2},
        'volatile': {'Hunter': 0.1, 'Sniper': 0.3, 'Defender': 0.6}
    }
    
    regime = model_predictions.get('regime', 'unknown')
    current_weights = regime_weights.get(regime, {'Hunter': 0.33, 'Sniper': 0.33, 'Defender': 0.33})
    
    # 显示权重条
    st.markdown(f"**当前市场状态: {format_regime(regime)}**")
    st.markdown("---")
    
    for model_name, weight in current_weights.items():
        col_model, col_bar = st.columns([1, 3])
        with col_model:
            st.text(f"{model_name}:")
        with col_bar:
            st.progress(weight)
            st.caption(f"{weight*100:.0f}%")

with col2:
    st.markdown("**权重说明**")
    st.info("""
    门控网络根据市场状态动态调整三个专家模型的权重：
    
    - **上涨趋势**: Hunter主导
    - **下跌趋势**: Hunter + Defender
    - **震荡市**: Sniper主导
    - **高波动**: Defender主导
    """)

st.markdown("---")

# 因子历史趋势
st.subheader("📈 因子历史趋势")

try:
    # 尝试加载因子历史
    factors_history = data_loader.load_factor_history(hours=1)
    
    if not factors_history.empty:
        # 选择几个关键因子显示
        key_factors = ['rsi_14', 'macd_hist', 'rolling_vol_ratio']
        available_factors = [f for f in key_factors if f in factors_history.columns]
        
        if available_factors and 'timestamp' in factors_history.columns:
            render_line_chart(
                factors_history,
                'timestamp',
                available_factors,
                title="关键因子趋势（最近1小时 - 真实数据）",
                x_label="时间",
                y_label="因子值"
            )
        else:
            st.info("因子历史数据不完整")
    else:
        st.warning("暂无因子历史数据，请确保量化引擎正在运行")
        
except Exception as e:
    st.error(f"无法加载因子历史: {e}")

st.markdown("---")

# 信号历史
st.subheader("📊 信号生成历史")

try:
    # 从因子历史提取信号记录
    factors_history = data_loader.load_factor_history(hours=2)
    
    if not factors_history.empty and 'signal' in factors_history.columns:
        # 过滤出信号变化的记录
        signal_changes = factors_history[factors_history['signal'].shift() != factors_history['signal']].tail(20)
        
        if not signal_changes.empty:
            signal_history = []
            for _, row in signal_changes.iterrows():
                timestamp = pd.to_datetime(row['timestamp'])
                signal_type = str(row.get('signal', 'FLAT'))
                confidence = float(row.get('confidence', 0)) if 'confidence' in row else 0
                
                signal_history.append({
                    '时间': timestamp.strftime('%H:%M:%S'),
                    '信号': format_signal(signal_type),
                    '置信度': f"{confidence*100:.0f}%" if confidence > 0 else "N/A",
                    '状态': '已记录'
                })
            
            df_signals = pd.DataFrame(signal_history)
            st.dataframe(df_signals, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("暂无信号变化记录")
    else:
        st.warning("暂无信号历史数据")
    
except Exception as e:
    st.error(f"无法加载信号历史: {e}")

# 页脚
st.markdown("---")
st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 因子计算频率: 1秒")
