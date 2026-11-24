# 十个因子能否覆盖 DOGE-USDT 永续合约的吃单成本？一项预注册检验

ikunfloyd · [github.com/ikunfloyd/crypto-factor-validation](https://github.com/ikunfloyd/crypto-factor-validation) · 2026-10-12 预注册

**问题。** 十个交易因子、两种 K 线周期（15 分钟、1 小时）和三种持仓期（1、5、15 根 K 线），组成 60 个检验单元。在开发时没有用过的数据上，是否有任何一个单元在 OKX DOGE-USDT 永续合约上扣除 10 bp 的吃单（taker）往返手续费后，每笔交易的平均收益为正？

**设计。** 预注册文件固定了检验单元、统计量、判定规则和功效分析，冻结补充文件记录了每个单元的交易方向；两者都在下载检验集之前公开。预注册文件的 §3 列出了此前每一次接触检验期数据的情况，包括规划阶段下载过的几个存档文件。
- 每个单元在因子处于上、下五分位时交易，在下一根 K 线收盘时入场，持有 h 根 K 线，支付 10 bp，交易方向取该单元在一个样本内 DOGE 年份上的方向。每根有信号的 K 线算一笔交易。
- 统计量是每笔交易的平均净收益，标准误用 Hansen–Hodrick 方法。判定规则：在主检验集（DOGE，2023-09-23 至 2024-09-23）上，对 60 个单元做单侧 Holm 检验，族错误率 5%（第一道门槛 t = 3.144），通过即为「确认」。另外两个检验集（第二个 DOGE 年份；BTC、ETH、SOL 在主检验年的数据）只能给已确认的单元加上 `replicated`（另一年也显著）或「在主流币上一致」的标签。

**结果。** 没有任何单元得到确认：最大的 t 是 1.71，而门槛是 3.144。
- 这个单元是一个成交量比值因子（`rolling_vol_ratio`，1 小时 K 线，h = 5：每笔 +17.6 bp，标准误 10.3），经 Holm 调整后的 p 为 1。它在第二个 DOGE 年份上是每笔 +14.3 bp（t 1.46），作者早先的研究部分覆盖过那一年；在合并的主流币上是每笔 −6.1 bp。
- 60 个单元中有 40 个的每笔平均净收益 95% 置信上界低于零（若要求上界对 60 个单元同时成立，则为 34 个）。在对全部 60 个单元同时成立的 95% 置信度下，15 分钟、持有 1 根的每个单元平均每笔亏损超过 7.5 bp，而它们的毛收益均值都在 −0.65 到 +0.72 bp 之间：亏掉的就是手续费。即使不计成本、不论哪个方向，也没有单元的 t 超过 2.68，所以即便持续的信号少付几次手续费，结论也不会变。
- 在第二个 DOGE 年份上（最大 t 2.23），以及把主流币合并检验时（最大 t 0.08），Holm 校正后都没有任何单元显著。

**「没有发现」未能排除的情形。** 在 1 小时 K 线持有 15 根时，检验只对每笔 52 到 123 bp（视单元而定）的净优势有 80% 的统计功效。结论只针对一个合约、一家交易所和一套费率；主流币与主检验集是同一年，因此它们说明的是覆盖面，而不是独立的确认。

**样本内之后的两个月（仅作描述）。** DOGE 在 2026-09-23 至 2026-11-23 的两个月不作判定：没有任何单元的 95% 区间在零以上，60 个中有 35 个整个区间在零以下。主检验年 t 最大的那个单元在这里是每笔 +5.9 bp（区间 −13.6 到 +25.3）。

**执行方式（探索性）。** 在样本内随机抽取的 14 天 OKX 订单簿存档（每边 400 档）上，在最优价挂一张合约 60 秒、未成交再吃单，从决策时的中间价算起每边成本 2.9–4.0 bp（含手续费和逆向选择），直接吃单则是 5.5 bp。这能把一次往返的成本从约 11 bp 降到约 6–8 bp，但改变不了结论：即使零成本，也没有任何单元达到门槛。

**方法上的收获。**
- **功效分析要留余量。** 预注册只按样本量把样本内标准误换算到主检验年，结果低估了实际的标准误：60 个单元中有 58 个的实际最小可检测效应高于冻结值，两者之比的中位数为 1.44。
- **报告置信上界。** 有了上界，结论就能从「不显著」变成「不赚钱」。
- **能预测不等于能赚钱。** 有 7 个单元的信息系数经 Holm 校正后仍显著，但每一个的毛收益均值与零相差都不到 2 bp，只有成本的五分之一。
- **把研究流程写进代码，并留下独立的记录。** 冻结补充文件公开之前，脚本拒绝下载检验集；代码有改动、输入不符或结果已存在时，脚本拒绝运行。任何偏离都只能是有意为之，而每一次偏离都按日期记入了日志。

**链接。** [报告](https://github.com/ikunfloyd/crypto-factor-validation/blob/main/docs/factor-research-v2.md)（英文） · [预注册文件（发布时版本）](https://github.com/ikunfloyd/crypto-factor-validation/blob/prereg-publication/docs/preregistration.md) · [偏离日志（当前版本）](https://github.com/ikunfloyd/crypto-factor-validation/blob/main/docs/preregistration.md#12-deviations-log) · [GitHub Release](https://github.com/ikunfloyd/crypto-factor-validation/releases/tag/prereg-publication) · [冻结补充文件](https://github.com/ikunfloyd/crypto-factor-validation/blob/main/docs/preregistration-freeze.md) · [结果与预注册的对照](https://github.com/ikunfloyd/crypto-factor-validation/blob/main/docs/preregistration-results.md) · [结果文件](https://github.com/ikunfloyd/crypto-factor-validation/tree/main/results) · Software Heritage [`swh:1:snp:7baf60bb2da0a53f8120a3ff56074ac410fb1b28`](https://archive.softwareheritage.org/swh:1:snp:7baf60bb2da0a53f8120a3ff56074ac410fb1b28)
