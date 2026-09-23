# 交易对配置
SYMBOL = "DOGE-USDT-SWAP"
INITIAL_CAPITAL = 10000.0  # 初始资金（USDT）

# 风险控制
MAX_POSITION_PCT = 0.8     # 最大仓位百分比
MAX_DAILY_LOSS_PCT = 0.05  # 单日最大亏损

# 性能优化
M3_P_CORE_AFFINITY = [0, 1, 2, 3]  # P-core绑定
M3_E_CORE_AFFINITY = [4, 5, 6, 7]  # E-core绑定