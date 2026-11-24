# Do ten factors beat taker costs on the DOGE-USDT perpetual? A pre-registered test

ikunfloyd · [github.com/ikunfloyd/crypto-factor-validation](https://github.com/ikunfloyd/crypto-factor-validation) · pre-registered 2026-10-12

**Question.** Ten trading factors, two bar sizes (15 minutes, 1 hour) and three holding horizons
(1, 5 and 15 bars) make 60 cells. Does any cell earn a positive mean return per trade on the OKX
DOGE-USDT perpetual after a 10 bp taker round trip, on data it was not developed on?

**Design.** A pre-registration fixed the cells, the statistic, the decision rule and the power
analysis, and a freeze amendment recorded each cell's direction; both were public before the test
sets were downloaded. The pre-registration's §3 lists every earlier contact with test-period data, including
archive files downloaded while planning.
- Each cell trades the factor's top and bottom quintiles, enters at the next bar's close, holds
  for h bars and pays 10 bp, in the direction the cell took on an in-sample DOGE year. Each signal
  bar counts as one trade.
- A cell is confirmed if its mean net return per trade, with a Hansen–Hodrick standard error,
  passes a one-sided Holm test over the 60 cells (family-wise error 5%, first threshold
  t = 3.144) on DOGE from 2023-09-23 to 2024-09-23. Two further sets, a second DOGE year and BTC,
  ETH and SOL over the primary year, could only label a confirmed cell as replicated or consistent
  on majors.

**Result.** No cell was confirmed: the largest t was 1.71, against the 3.144 needed.
- That cell is a volume ratio (`rolling_vol_ratio`, 1-hour bars, h = 5: +17.6 bp per trade,
  SE 10.3), with a Holm-adjusted p of 1. It was +14.3 bp (t 1.46) on the second DOGE year, which
  the author's earlier research partly covered, and −6.1 bp on the pooled majors.
- For 40 of the 60 cells the 95% upper bound on the mean net return per trade is below zero (34 if
  the bound must hold for all 60 at once). With 95% simultaneous confidence, every 15-minute
  one-bar cell loses more than 7.5 bp per trade on average, while its gross mean lies between
  −0.65 and +0.72 bp: the loss is the cost. Even before costs, and in either direction, no cell's
  t exceeds 2.68, so charging a persistent signal's fee less often would not change the answer.
- Holm rejected nothing on the second DOGE year (largest t 2.23) or on the pooled majors (largest
  t 0.08).

**What the null leaves open.** At 1-hour bars held for 15 bars the test had 80% power only against
net edges of 52 to 123 bp per trade, depending on the cell. It covers one contract, exchange and
fee schedule, and the majors share the primary year, so they show breadth, not independent
confirmation.

**After the in-sample year (descriptive).** On DOGE from 2026-09-23 to 2026-11-23, two months that
decide nothing, no cell's 95% interval lies above zero and 35 of the 60 lie entirely below it. The
cell with the primary year's largest t is +5.9 bp there (interval −13.6 to +25.3).

**Execution (exploratory).** On 14 random in-sample days of OKX's order-book archive (400 levels
a side), posting one contract at the best price for 60 seconds before crossing cost 2.9–4.0 bp per
side from the mid, fees and adverse selection included, against 5.5 bp for crossing at once. That would cut a round trip from about 11
to about 6–8 bp, and it cannot change the answer: no cell reaches its threshold even at zero cost.

**Lessons on method.**
- **Leave a margin in power planning.** Scaling in-sample standard errors to the primary year's
  length, as the plan did, understated them: the realised minimum detectable effects exceeded the
  frozen ones in 58 of 60 cells, by a median factor of 1.44.
- **Report upper bounds.** They turn "not significant" into "does not pay".
- **Prediction is not profit.** Seven cells had a Holm-significant information coefficient, yet each
  had a gross mean within 2 bp of zero, a fifth of the cost.
- **Put the protocol in code and in independent records.** Scripts refused to download test sets
  before the freeze amendment was public, or to run on changed code, changed inputs or existing results. Any
  deviation had to be deliberate, and each went into a dated log.

**Links.**
[Report](https://github.com/ikunfloyd/crypto-factor-validation/blob/main/docs/factor-research-v2.md)
 ·
[Pre-registration, as published](https://github.com/ikunfloyd/crypto-factor-validation/blob/prereg-publication/docs/preregistration.md) ·
[Deviations log, current](https://github.com/ikunfloyd/crypto-factor-validation/blob/main/docs/preregistration.md#12-deviations-log) ·
[GitHub Release](https://github.com/ikunfloyd/crypto-factor-validation/releases/tag/prereg-publication) ·
[Freeze amendment](https://github.com/ikunfloyd/crypto-factor-validation/blob/main/docs/preregistration-freeze.md) ·
[Results compared with the plan](https://github.com/ikunfloyd/crypto-factor-validation/blob/main/docs/preregistration-results.md) ·
[Results files](https://github.com/ikunfloyd/crypto-factor-validation/tree/main/results) ·
Software Heritage [`swh:1:snp:7baf60bb2da0a53f8120a3ff56074ac410fb1b28`](https://archive.softwareheritage.org/swh:1:snp:7baf60bb2da0a53f8120a3ff56074ac410fb1b28)
