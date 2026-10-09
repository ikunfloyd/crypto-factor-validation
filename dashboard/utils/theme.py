"""
主题配置模块
定义Dashboard的样式和主题
"""

# 颜色配置 - 使用高对比度配色方案
COLORS = {
    'primary': '#0066cc',      # 深蓝色
    'success': '#10b981',      # 绿色
    'warning': '#f59e0b',      # 橙色
    'danger': '#ef4444',       # 红色
    'info': '#3b82f6',         # 蓝色
    'dark': '#1f2937',         # 深灰色
    'light': '#f3f4f6',        # 浅灰色
    'green': '#10b981',
    'red': '#ef4444',
    'blue': '#3b82f6',
    'gray': '#6b7280'
}

# 状态颜色映射
STATUS_COLORS = {
    'running': COLORS['success'],
    'stopped': COLORS['danger'],
    'error': COLORS['danger'],
    'warning': COLORS['warning'],
    'healthy': COLORS['success'],
    'degraded': COLORS['warning'],
    'unknown': COLORS['gray']
}

# 自定义CSS样式 - 优化对比度和可读性
CUSTOM_CSS = """
<style>
    /* 主题色 */
    :root {
        --primary-color: #0066cc;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
    }
    
    /* 卡片样式 - 使用高对比度配色 */
    .metric-card {
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(0,102,204,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,102,204,0.3);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 1.2em;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-card .value {
        font-size: 2.2em;
        font-weight: bold;
        margin: 12px 0;
        color: #ffffff;
    }
    
    .metric-card .delta {
        font-size: 0.95em;
        opacity: 0.95;
        color: #e0f2fe;
    }
    
    /* 进程状态卡片 - 增强对比度 */
    .process-card {
        border: 3px solid #e5e7eb;
        border-radius: 12px;
        padding: 22px;
        background: white;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: box-shadow 0.2s;
    }
    
    .process-card:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .process-card.running {
        border-color: #10b981;
        background: linear-gradient(135deg, #d1fae5 0%, #ffffff 100%);
    }
    
    .process-card.stopped {
        border-color: #ef4444;
        background: linear-gradient(135deg, #fee2e2 0%, #ffffff 100%);
    }
    
    .process-card.unknown {
        border-color: #9ca3af;
        background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%);
    }
    
    .process-card h4 {
        margin: 0 0 12px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.2em;
    }
    
    .process-card .status-indicator {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .process-card .status-indicator.running {
        background-color: #10b981;
        box-shadow: 0 0 12px rgba(16,185,129,0.5);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .process-card .status-indicator.stopped {
        background-color: #ef4444;
        box-shadow: 0 0 8px rgba(239,68,68,0.3);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .process-card .info-row {
        display: flex;
        justify-content: space-between;
        margin: 8px 0;
        font-size: 0.95em;
        padding: 6px 0;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .process-card .info-row:last-child {
        border-bottom: none;
    }
    
    /* 告警样式 - 增强对比度 */
    .alert-item {
        padding: 14px 16px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    .alert-item.error {
        background: #fee2e2;
        border-color: #ef4444;
        color: #7f1d1d;
    }
    
    .alert-item.warning {
        background: #fef3c7;
        border-color: #f59e0b;
        color: #78350f;
    }
    
    .alert-item.info {
        background: #dbeafe;
        border-color: #3b82f6;
        color: #1e3a8a;
    }
    
    .alert-item .timestamp {
        font-size: 0.85em;
        color: inherit;
        opacity: 0.8;
        font-family: 'Monaco', 'Menlo', monospace;
        font-weight: 600;
    }
    
    /* 表格样式 - 现代化设计 */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .custom-table th {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        padding: 14px 16px;
        text-align: left;
        font-weight: 700;
        font-size: 0.95em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .custom-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #e5e7eb;
        color: #374151;
        font-size: 0.95em;
    }
    
    .custom-table tr:hover {
        background-color: #f9fafb;
    }
    
    .custom-table tr:last-child td {
        border-bottom: none;
    }
    
    /* 进度条 - 现代化样式 */
    .progress-bar {
        height: 24px;
        background: #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
    }
    
    .progress-bar .fill {
        height: 100%;
        background: linear-gradient(90deg, #0066cc 0%, #3b82f6 100%);
        transition: width 0.4s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.85em;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    
    /* 徽章 - 高对比度 */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.85em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .badge.success {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #10b981;
    }
    
    .badge.danger {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #ef4444;
    }
    
    .badge.warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
    }
    
    .badge.info {
        background: #dbeafe;
        color: #1e40af;
        border: 1px solid #3b82f6;
    }
    
    /* 标签页美化 - 现代化设计 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        background-color: #f9fafb;
        color: #6b7280;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f3f4f6;
        color: #374151;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,102,204,0.2);
    }
    
    /* 指标容器 */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    /* 图表容器 */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #0066cc 0%, #003d7a 100%);
    }
    
    /* 标题样式 */
    .dashboard-title {
        text-align: center;
        padding: 24px;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 6px 16px rgba(0,102,204,0.25);
        font-weight: 700;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .metrics-container {
            grid-template-columns: 1fr;
        }
        
        .metric-card .value {
            font-size: 1.5em;
        }
    }
</style>
"""

# Plotly图表主题
PLOTLY_THEME = "plotly_dark"

PLOTLY_LAYOUT = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0.05)',
    'font': {'color': '#2c3e50'},
    'xaxis': {
        'gridcolor': 'rgba(0,0,0,0.1)',
        'zeroline': False
    },
    'yaxis': {
        'gridcolor': 'rgba(0,0,0,0.1)',
        'zeroline': False
    }
}

# 图表颜色方案 - 高对比度配色
CHART_COLORS = [
    '#0066cc',  # 深蓝
    '#10b981',  # 绿
    '#f59e0b',  # 橙
    '#ef4444',  # 红
    '#8b5cf6',  # 紫
    '#06b6d4',  # 青
    '#ec4899',  # 粉
    '#14b8a6'   # 蓝绿
]

# 页面配置
PAGE_CONFIG = {
    'page_title': 'DOGE-USDT Trading Dashboard',
    'page_icon': '🚀',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'menu_items': {
        'Get Help': 'https://github.com/ikunfloyd/crypto-factor-validation',
        'Report a bug': 'https://github.com/ikunfloyd/crypto-factor-validation/issues',
        'About': '# DOGE-USDT 高频量化交易系统\n\n实时监控仪表板 v1.0.0'
    }
}

# 自动刷新配置（毫秒）
AUTO_REFRESH_INTERVAL = 5000  # 5秒

# 缓存配置（秒）
CACHE_TTL = {
    'shared_memory': 5,
    'health_metrics': 10,
    'process_status': 30,
    'factor_history': 60,
    'logs': 10
}


def get_status_color(status: str) -> str:
    """获取状态对应的颜色"""
    return STATUS_COLORS.get(status.lower() if isinstance(status, str) else '', COLORS['gray'])


def apply_custom_css():
    """应用自定义CSS（用于Streamlit）"""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
