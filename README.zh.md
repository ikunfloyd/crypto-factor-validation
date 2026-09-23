# DOGE-USDT 量化交易系统

> 面向 OKX DOGE-USDT 永续合约的事件驱动、多进程高频交易引擎，
> 一个在 Apple Silicon 上进行的系统工程实践。

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey.svg)](https://www.apple.com/mac/)

[English](README.md) · **中文**

---

这是一个个人项目，探索如何用 Python 搭建低延迟交易栈：I/O、计算、执行分别跑在
独立进程里，进程之间用**共享内存零拷贝**通信（而非消息队列），配合预写日志实现
崩溃恢复，并用一个小型混合专家（MoE）模型做决策。真正有意思的是这套工程管道，
而不是策略本身——阅读前请先看 [项目真实状态](#项目真实状态)。

## 亮点

- **零拷贝 IPC** —— 基于 `multiprocessing.shared_memory` 手写的环形缓冲区，
  生产者与消费者读同一段字节、无序列化
  ([`shared_memory/ring_buffer.py`](shared_memory/ring_buffer.py))。
- **崩溃恢复** —— 每条 tick 先写预写日志（WAL）再进共享内存，配合定期检查点，
  `kill -9` 也不丢数据
  ([`io_gateway/wal_writer.py`](io_gateway/wal_writer.py)、[`utils/disaster_recovery.py`](utils/disaster_recovery.py))。
- **异步行情** —— OKX 双 WebSocket 端点（public + business），带心跳监控与
  指数退避重连 ([`io_gateway/ws_client.py`](io_gateway/ws_client.py))。
- **进程编排** —— 主控进程把每个阶段作为独立进程拉起，绑定性能核/能效核、
  健康检查、崩溃自动重启 ([`orchestrator/main.py`](orchestrator/main.py))。
- **向量化计算** —— Polars 数据框、Numba-JIT 因子核、三专家 LightGBM MoE + 
  市场状态门控网络 ([`quant_core/`](quant_core/))。

## 30 秒离线试跑（无需密钥）

demo 会跑通完整计算链路——合成 K 线 → 特征工程 → 因子 → MoE → 交易信号——
**无需交易所账户、API 密钥或 Apple Silicon 硬件**。首次运行会自动训练缺失的模型。

```bash
conda env create -f environment.yml      # 或：pip install -r requirements.txt
conda activate doge_trading
python scripts/demo.py
```

```
[2/4] data processing + factor engine ...
      factors computed: 5 -> rsi_14, macd_hist, bollinger_pctb, rolling_vol_ratio, basis_z_score
[4/4] inference:
  regime          : high_vol
  fused score     : +0.267   ->  NEUTRAL   confidence 0.0%

同一批特征在不同市场状态下的门控路由：
  trending  -> SHORT   conf=95.8%  pos=47.9%
  ranging   -> SHORT   conf=87.3%  pos=30.6%
  high_vol  -> NEUTRAL conf= 0.0%  pos= 0.0%
```

## 架构

```
        OKX WebSocket (public + business)
                    │
          ┌─────────▼─────────┐        ┌────────────────────┐
          │    IO Gateway     │──────▶ │  WAL (msgpack+zlib) │  持久化
          │   异步采集         │        └────────────────────┘
          └─────────┬─────────┘
                    │ 写入（零拷贝）
          ┌─────────▼─────────┐
          │   Shared Memory   │  环形缓冲：tick(3600) · candle(5000)
          └─────────┬─────────┘
                    │ read_latest / peek_latest
          ┌─────────▼─────────┐
          │    Quant Core     │  数据处理 → 因子引擎 → MoE 推理
          └─────────┬─────────┘
                    │ 信号
          ┌─────────▼─────────┐
          │    Execution      │  信号融合 → 风险管理 → 订单路由
          └───────────────────┘

   Orchestrator 统管所有进程：CPU 亲和性绑定、健康检查、自动重启、
   检查点 + WAL 回放的灾难恢复。
```

完整设计文档（含设计理由与数据流图）见 [`ARCHITECTURE.zh.md`](ARCHITECTURE.zh.md)（英文精简版见 [`ARCHITECTURE.md`](ARCHITECTURE.md)）。

## 性能基准

在 Apple Silicon 笔记本上用 `python scripts/benchmark.py` 实测(单次运行、Numba 已预热)。
数字与机器相关——请自行复现，而非轻信下表:

| 热点路径 | 结果 |
|----------|------|
| Ring buffer 写入(共享内存) | ~1.2M items/s · 0.81 µs/次 |
| `peek_latest(1)` 读延迟 | 中位 1.4 µs |
| 数据处理(numpy → Polars + 特征) | ~1.5M candles/s |
| 因子计算(K 线因子, Numba) | ~68K candles/s · 2000 根 30 ms |
| MoE 推理(取特征 + 预测) | 中位 218 µs · ~4.6K 次/s |

```bash
python scripts/benchmark.py               # 或 --candles 5000 --iters 100000
```

## 运行完整系统

实盘链路需要 OKX API 凭证，且面向 Apple Silicon（主控进程会绑定 P/E 核）。
每个阶段也可单独运行。

```bash
cp .env.example .env && source .env        # 填入 OKX_API_KEY / OKX_SECRET / OKX_PASSWORD

python -m orchestrator.main                # 受管：拉起所有阶段
# —— 或单独运行各阶段 ——
python -m io_gateway.main                  # OKX 采集 → 共享内存 + WAL
python -m quant_core.main_v2               # 因子 + MoE 推理
python -m execution.main                   # 融合、风控、路由
streamlit run dashboard/main.py            # 监控界面
```

重新训练专家模型：

```bash
python -m quant_core.training.train_models  # 输出到 data/models/{hunter,sniper,defender}/
```

## 测试

```bash
pip install -r requirements-dev.txt
pytest                                    # 配置见 pytest.ini
```

## 项目真实状态

这是一个学习项目，不是生产级交易系统。具体来说：

- **模型基于合成数据训练**（[`quant_core/training/train_models.py`](quant_core/training/train_models.py)）。
  demo 展示的是「因子 → 标准化 → MoE → 信号」这套**工程管道**，不是可盈利的策略，
  请勿把其中数字当作 alpha。
- **单交易对**。目前一切都围绕 `DOGE-USDT-SWAP`；多交易对需要为每个标的
  分配独立的共享内存段。
- **部分因子需要实时数据**。资金费率、持仓量、订单流相关因子依赖离线 demo 拿不到
  的数据源，因此在 demo 中被跳过，只计算 K 线派生因子。
- **未经实盘充分验证**。请把实盘链路视为参考实现，而非可直接对接账户的成品。

## 免责声明

仅供学习与研究。加密货币交易存在重大亏损风险，本项目不构成任何投资建议，
作者不对使用本软件产生的任何后果负责。

## 许可证

[MIT](LICENSE)。
