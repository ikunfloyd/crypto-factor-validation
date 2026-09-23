# DOGE-USDT 高频量化交易系统 - 实时监控仪表板

基于 Streamlit 构建的实时监控和可视化仪表板。

## 📋 功能特性

### 多页面架构
- **🏠 主页**: 系统概览和快速导航
- **📊 系统总览**: 整体状态、进程监控、关键指标
- **📈 市场数据**: K线图、订单簿、成交记录
- **🧠 量化引擎**: 因子分析、模型预测、信号生成
- **💰 交易执行**: 持仓管理、订单追踪、盈亏统计
- **📋 日志查看**: 实时日志流、错误追踪
- **⚙️ 系统设置**: 配置管理、数据导出、系统控制

### 核心功能
✅ **实时数据**: 5秒自动刷新，实时监控系统状态  
✅ **交互式图表**: 基于 Plotly 的高性能可视化  
✅ **智能缓存**: 优化数据加载性能  
✅ **响应式设计**: 适配不同屏幕尺寸  
✅ **模块化架构**: 易于扩展和维护  

## 🚀 快速开始

### 1. 安装依赖

确保已激活 conda 环境：

```bash
conda activate doge_trading
```

依赖已包含在 `environment.yml` 中：
- `streamlit==1.33.0`
- `plotly==5.18.0`
- `streamlit-autorefresh==1.0.1`

如需单独安装：

```bash
pip install streamlit plotly streamlit-autorefresh
```

### 2. 启动 Dashboard

```bash
# 从项目根目录启动
streamlit run dashboard/main.py

# 指定端口
streamlit run dashboard/main.py --server.port=8501

# 后台运行
nohup streamlit run dashboard/main.py > dashboard.log 2>&1 &
```

### 3. 访问界面

打开浏览器访问: **http://localhost:8501**

## 📁 项目结构

```
dashboard/
├── main.py                          # 主入口文件
├── pages/                           # 多页面目录
│   ├── 1_📊_System_Overview.py     # 系统总览
│   ├── 2_📈_Market_Data.py         # 市场数据
│   ├── 3_🧠_Quant_Engine.py        # 量化引擎
│   ├── 4_💰_Trading.py             # 交易执行
│   ├── 5_📋_Logs.py                # 日志查看
│   └── 6_⚙️_Settings.py            # 系统设置
├── components/                      # 可复用组件
│   ├── __init__.py
│   ├── metrics.py                  # 指标卡片组件
│   ├── charts.py                   # 图表组件
│   ├── tables.py                   # 表格组件
│   └── alerts.py                   # 告警组件
└── utils/                          # 工具模块
    ├── __init__.py
    ├── data_loader.py              # 数据加载器
    ├── formatters.py               # 格式化工具
    └── theme.py                    # 主题配置
```

## 🎨 页面介绍

### 📊 系统总览
- 整体健康状态
- 三大进程监控（IO Gateway, Quant Core, Execution）
- 实时性能图表（数据流速、CPU、延迟）
- 最近告警

### 📈 市场数据
- DOGE-USDT K线图（含布林带）
- 买卖五档订单簿
- 资金流向分析
- 最近成交记录

### 🧠 量化引擎
- 9个实时因子展示
- 三专家模型预测雷达图
- 门控网络权重分配
- 因子历史趋势
- 信号生成历史

### 💰 交易执行
- 当前持仓详情
- 风险指标（最大回撤、夏普比率）
- 盈亏曲线
- 活跃订单和成交记录
- 交易统计

### 📋 日志查看
- 实时日志流
- 日志级别过滤
- 错误汇总和分类
- 性能日志统计

### ⚙️ 系统设置
- 系统信息查看
- 配置参数展示
- WAL和检查点管理
- 数据导出功能
- 监控告警设置

## 🔧 配置说明

### 自动刷新
默认5秒刷新一次，可在 [`dashboard/utils/theme.py`](utils/theme.py:70) 修改：

```python
AUTO_REFRESH_INTERVAL = 5000  # 毫秒
```

### 缓存策略
不同数据源使用不同的缓存时间，配置在 [`dashboard/utils/theme.py`](utils/theme.py:73-79)：

```python
CACHE_TTL = {
    'shared_memory': 5,      # 共享内存数据缓存5秒
    'health_metrics': 10,    # 健康指标缓存10秒
    'process_status': 30,    # 进程状态缓存30秒
    'factor_history': 60,    # 因子历史缓存60秒
    'logs': 10               # 日志缓存10秒
}
```

### 主题颜色
在 [`dashboard/utils/theme.py`](utils/theme.py:10-19) 自定义颜色方案：

```python
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'warning': '#ff7f0e',
    'danger': '#d62728',
    # ...
}
```

## 📊 数据源

Dashboard 从以下数据源加载数据：

1. **共享内存**: 实时 Tick、K线、因子数据
2. **数据文件**: 
   - `data/factors_latest.csv` - 最新因子
   - `data/factors_history.csv` - 因子历史
   - `data/execution_state.pkl` - 交易状态
3. **WAL日志**: `data/wal/` 目录
4. **检查点**: `data/checkpoints/` 目录
5. **系统日志**: 项目根目录的 `.log` 文件

## 🐛 故障排查

### Dashboard 无法启动

```bash
# 检查依赖
pip list | grep streamlit

# 重新安装
pip install --upgrade streamlit plotly
```

### 数据无法加载

检查数据文件是否存在：
```bash
ls -l data/factors_latest.csv
ls -l data/factors_history.csv
```

### 进程状态显示 unknown

确保交易系统的各个进程正在运行：
```bash
ps aux | grep python
```

## 🔐 安全建议

1. **生产环境**: 添加密码保护（参考 [`dashboard/utils/theme.py`](utils/theme.py) 中的认证示例）
2. **网络访问**: 限制只允许本地访问
3. **数据权限**: 确保 dashboard 只有只读权限

## ⚠️ 重要：导入路径说明

Dashboard有自己的`utils`目录，与项目根目录的`utils`不同。为避免导入冲突，请使用正确的导入路径：

### ✅ 正确的导入方式

**从外部调用Dashboard模块**（如在测试或其他模块中）：
```python
# 正确 - 使用完整路径
from dashboard.utils.data_loader import DataLoader
from dashboard.utils.formatters import format_number
from dashboard.utils.theme import apply_custom_css

# 正确 - 使用项目根utils
from utils.logger_config import setup_logger
from utils.health_monitor import HealthMonitor
```

**在Dashboard内部**（如在pages或components中）：
```python
# 正确 - 相对导入Dashboard自己的utils
from dashboard.utils.data_loader import DataLoader
from dashboard.utils.formatters import format_number

# 或使用相对导入（如果在dashboard目录下）
from .utils.data_loader import DataLoader
from .utils.formatters import format_number
```

### ❌ 错误的导入方式

```python
# 错误 - 会导致"No module named 'utils.data_loader'"错误
from utils.data_loader import DataLoader

# 错误 - 混淆了两个不同的utils
from utils.formatters import format_number  # 这会查找项目根的utils，而不是dashboard/utils
```

### 目录结构说明

```
项目根目录/
├── utils/                    # 项目全局工具（logger, health_monitor, 等）
│   ├── logger_config.py
│   ├── health_monitor.py
│   └── ...
└── dashboard/
    ├── utils/               # Dashboard专用工具（data_loader, formatters, 等）
    │   ├── data_loader.py
    │   ├── formatters.py
    │   └── theme.py
    └── ...
```

### 最佳实践

1. **始终使用完整路径**: 使用`dashboard.utils.xxx`而不是`utils.xxx`
2. **在Dashboard内**: 可以使用相对导入`.utils.xxx`
3. **避免星号导入**: 不要使用`from dashboard.utils import *`
4. **IDE配置**: 确保IDE的Python路径包含项目根目录

## 📝 开发指南

### 添加新页面

1. 在 `dashboard/pages/` 创建新文件
2. 文件名格式: `序号_emoji_名称.py`
3. 使用标准模板：

```python
import streamlit as st
from dashboard.utils.theme import apply_custom_css
from dashboard.utils.data_loader import get_data_loader

st.set_page_config(page_title="页面标题", page_icon="📊", layout="wide")
apply_custom_css()

st.title("页面标题")
# 页面内容...
```

### 添加新组件

在 `dashboard/components/` 创建新组件模块，遵循现有组件的设计模式。

### 添加新数据源

在 [`dashboard/utils/data_loader.py`](utils/data_loader.py) 的 `DataLoader` 类中添加新方法。

## 🚀 性能优化

1. **使用缓存**: 对慢速数据源使用 `@st.cache_data`
2. **增量更新**: 使用 `st.empty()` 进行局部更新
3. **懒加载**: 使用 `st.expander` 延迟加载大数据集
4. **限制数据量**: 只展示必要的数据行数

## 📚 参考资源

- [Streamlit 官方文档](https://docs.streamlit.io/)
- [Plotly Python 文档](https://plotly.com/python/)
- [项目设计文档](../plans/dashboard_design.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

与主项目保持一致

---

**Dashboard Version**: v1.0.0  
**最后更新**: 2026-01-08
