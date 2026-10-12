# Pre-registration: do any of ten factors beat taker costs on the DOGE-USDT perpetual out of sample?

**Summary.** The first study in this repository ("v1", [`docs/factor-research.md`](factor-research.md))
found no factor whose return on the DOGE-USDT perpetual, net of a 10 bp taker round trip, was
distinguishable from noise over one in-sample year. This document fixes, before the test data is
downloaded, how ten factors will be tested on years and instruments that study did not use.

- **Test.** There are 60 cells: 10 factors × {15m, 1H} bars × {1, 5, 15}-bar horizons. Each cell
  trades the factor's expanding-window top and bottom quintiles in the direction the cell took in
  sample, enters one bar after the signal, and pays 10 bp per trade (§6).
- **Decision.** A cell is *confirmed* if its mean net return per trade is significantly positive
  on DOGE from 2023-09-23 to 2024-09-23 (set `doge_2023`). The test is one-sided, uses a
  Hansen–Hodrick standard error, and applies Holm's correction across all 60 cells at a family-wise
  error rate of 5%. The first threshold is t > 3.14 (§7).
- **Other sets.** DOGE 2024–25 (`doge_2024`) and BTC/ETH/SOL over 2023–24 (`majors_2023`) can only
  add the labels *replicated* or *consistent on majors* to a confirmed cell. DOGE after the
  in-sample year (`doge_fwd`) is descriptive (§4.1, §7).
- **A null is informative.** Each cell's minimum detectable effect is fixed before the download,
  and every cell's 95% upper confidence bound is reported (§9).
- **Prior exposure.** §3 lists every earlier look at test data. The author's earlier research used
  no DOGE data from the primary year, though it analysed BTC 15-minute data over 267 of its 366
  days. The planning-stage checks touched DOGE in that year only through one close and the
  timestamps of a funding file. Its early-2024 rally is general knowledge.

| | |
|---|---|
| **Written** | 2026-10-08. Apart from the contacts listed in §3 (the author's earlier strategy research and the planning-stage checks), no test-set data had been downloaded or analysed. |
| **Integrity** | This file is made public before the test-set download (§10, step 4), with a GitHub Release cut from the commit that contains it. That commit is also archived independently with Software Heritage, and the identifier is recorded in the freeze amendment, so the publication date can be checked without relying on GitHub. The files fetched in the planning stage (§3) are not reused; every test series is downloaded again in step 4. |
| **Changes** | After publication nothing in this file is edited in place. Every change, before or after the freeze, is a dated entry in the deviations log (§12) giving its reason and whether test-set data had been seen. The freeze amendment is a separate file, `docs/preregistration-freeze.md` (§10, step 3), and changes nothing here. |
| **Machine-readable spec** | Appendix A. The implementation's tests (§10, step 2) check the code's constants against it. |

## 1. Question

Do any of ten candle-, basis- and funding-derived factors predict the DOGE-USDT perpetual's
forward return strongly enough that trading the extreme quintiles earns a **positive return per
trade after a 10 bp taker round trip**? The test uses data the first study did not use; §3 lists
every earlier look at it.

The primary return is the price return. Funding paid or received while a position is open is
reported alongside it for every primary cell (R8). A trade spans at most two 8-hour settlements.
In the in-sample funding data (2026-06-22 → 2026-09-23), no settlement exceeded 1 bp in absolute
value. Rates can be several times larger in strong rallies, though, and `funding_cost_pressure`
selects high- and low-funding periods by construction, so for that factor R8 can differ
materially from the primary result.

The answer may well be no. A negative result is reported in full, together with the smallest
effect the test could have detected (§9), so that "no" carries information.

## 2. What is already known, and what v1 got wrong

v1 used one year of DOGE-USDT-SWAP 1-minute candles from 2025-09-23 to 2026-09-23 (the
**in-sample year**, "IS"). It ran 36 tests: 6 factors × bars {15m, 1H} × horizons {1, 5, 15}.
None reached |t| > 2 on v1's statistic, which item 2 below shows was mis-sized. None survived a
Bonferroni correction.

Re-reading the code turned up six problems. None of them rescues v1. Once cost is charged per
trade (item 1), none of its 36 spreads clears cost, whatever standard error is used: the largest
is 18.83 bp against a 20 bp break-even. The other fixes can move a t-statistic either way. Each is
adopted because it is correct, not for its effect. Under the in-sample direction rule (§6.5), cells
with large negative v1 spreads are reversed and may show a positive in-sample return. That is
selection on the in-sample year, which is why only the test sets decide.

1. **Cost was undercharged by half.** The net spread subtracted one 10 bp round trip from a
   top-minus-bottom spread that takes two sets of trades: long the top bucket and short the bottom
   one. Per trade, break-even is a 20 bp spread. Under per-trade accounting in v1's orientation
   (long top, short bottom), none of the 36 cleared cost, not 6. → §6.6.
2. **The overlap correction was not a variance estimator.** Dividing t by √h is roughly right only
   when bucket membership persists far longer than h bars. It over-corrects for signals that change
   bucket quickly. → a heteroskedasticity- and autocorrelation-consistent (HAC) estimator, §6.7.
3. **Quantile edges used future data.** Edges were computed over the full sample. `macd_hist` is
   in price units, so its buckets partly tracked the price level. → expanding-window edges (§6.4)
   and `macd_hist / close` (§5.1).
4. **`oi_change_rate` looks ahead.** OKX labels each 5-minute open-interest value with the start of
   its period, but the value is the end-of-period snapshot. The OI change correlates 0.60
   (Spearman) with the price return over the 5 minutes *after* its timestamp, and about 0.2 with the
   5 minutes before it. v1 left the factor out of its 36 tests only because two days of OI fell
   below the minimum sample size. In a separate run on 1-minute bars, outside those tests, it gave
   the only t above a Bonferroni line: 5.39 at h = 1, on 2,858 bars. That t is look-ahead.
   → excluded, §5.3.
5. **Nine factors were registered; six were tested.** `basis_z_score` needs a spot leg that was
   never fetched. `oi_change_rate` had two days of data. `downside_beta` never receives BTC data
   from the research script. → §5.
6. **The saved results file was overwritten** by a later partial run, so v1's 36-row table has no
   persisted artefact. → write-once outputs, §10.

## 3. Prior exposure to the test data

Everything below was fetched or computed before this document was written.

**The author's earlier strategy research.** This was a separate project. It ran walk-forward
machine-learning classifiers and rule backtests.

- **DOGE inputs:** RSI(14); Bollinger %B and band width (20, 2); the MACD histogram (12/26/9);
  price z-scores over 20, 55 and 144 bars; ATR; ADX; a 20-bar relative volume; strength relative
  to BTC; and a Triple Screen system (MACD 12/26/9 on a higher timeframe plus a 14-bar
  stochastic).
- **BTC inputs:** feature sets on the BTC-USDT spot pair that included, at 15 minutes, RSI(14),
  Bollinger band width (20, 2), the MACD histogram, Chaikin Money Flow (20) and a volume ratio. A
  1-hour BTC dataset also has an RSI column.
- **Data spans:**
  - DOGE, BTC, ETH, SOL, BNB and XRP at 5 minutes, from 2024-12-17 to 2026-01-03.
  - BTC at 15 minutes, from 2024-01-01 to 2026-01-03, and a BTC 1-hour feature file from
    2024-01-03 to 2026-01-02.
  - 1-minute candles for the four perpetuals from 2025-06-23.
  - Derivatives series (funding, open interest, positioning) for BTC, ETH and SOL from late
    2025.
- **Unanalysed downloads:** none of the project's retained scripts analyses the 1-minute candles
  or the derivatives series.
- **Statistic:** no retained script computes the quintile-spread statistic registered here, but
  the project's results were seen.

What that project touched, by set (sets are defined in §4.1):

| Set | Overlap with that project |
|---|---|
| `doge_2023` (primary) | No DOGE data. BTC 15-minute and 1-hour data cover 2024-01-01 → 2024-09-23 (267 of 366 days). |
| `majors_2023` | Its BTC leg: those same 267 days. ETH and SOL: none. |
| `doge_2024` | DOGE data on about 281 of 365 days (77%). BTC 15-minute data on all 365 days; BTC, ETH, SOL, BNB and XRP 5-minute data on the same 281 days. |

Of the factors here, these were DOGE inputs there: `rsi_14`, `bollinger_pctb` (same parameters),
`macd_hist_norm` (as the unscaled histogram), `triple_screen` and, loosely, `rolling_vol_ratio`.
`net_taker_flow` is Chaikin Money Flow over 60 bars; that project used it over 20 bars on BTC.
On BTC, `rsi_14` was an input with the same parameter at 15 minutes over 267 days of
`majors_2023`, on the spot pair rather than the swap.
No factor parameter is taken from that work. Each is a textbook default, a v1 value or a
structural choice whose reason is stated in §5.

**Planning-stage checks, 2026-10-06 to 2026-10-08 UTC.** Automated agents ran these checks on the author's
behalf. The author saw only written summaries, not the raw output.

- **Availability probes.** These requested single candles, short pages of up to 400 candles, and
  archive file listings, to learn how far back each source reaches. Only timestamps and file names
  were printed, plus one DOGE close at 2024-09-01, inside `doge_2023`. A listing on 2026-10-08
  showed monthly funding files for DOGE-, BTC-, ETH- and SOL-USDT-SWAP for every month from
  2023-06 to 2024-08. Their contents were not downloaded.
- **Archive check.** Two monthly DOGE funding files were downloaded from OKX's archive.
  - June 2026 (in-sample) was compared with the in-sample REST file on their 24 shared settlements,
    and matched exactly.
  - September 2024 (2024-08-31 16:00 → 2024-09-30 08:00 UTC) was checked only for its row count
    and 8-hour spacing; its rates were not analysed. 70 of its 90 settlements fall in the last 23
    days of `doge_2023`, which are also `doge_2024`'s warm-up; the rest fall in `doge_2024`'s first
    week.
- **Methodology check on the majors.** Hourly closes of BTC-, ETH- and SOL-USDT-SWAP for
  2025-09-23 → 2026-09-23 were downloaded. That is the in-sample year, not a test set.
  - For `rsi_14` and `bollinger_pctb` at 1H, h ∈ {5, 15}, the check computed cross-instrument
    return and factor correlations, spread t-statistics, and a circular-shift simulation of how
    correlated the four instruments' t-statistics are under the null.
  - The author has not seen the per-instrument t-statistics, and no choice here cites them.
  - Two aggregate observations did inform the design. First, DOGE co-moves strongly with the
    majors, so the majors are secondary evidence and are pooled rather than counted as three
    replications (§7). Second, the in-sample spreads concentrate in a few market-wide volatile
    days, which motivates R4.
- **Open-interest timestamp check.** DOGE open interest was polled live for about three minutes,
  inside `doge_fwd`'s window, to learn how OKX timestamps it. Together with an in-sample lead-lag
  test, this is the evidence for §2, item 4. The three most recent DOGE trades, with their prices,
  were fetched at the same time. Neither feeds any factor here.
- **Maker-fill sketch.** Two OKX archive files for DOGE-USDT-SWAP were downloaded:
  - the trade file labelled 2026-10-04, dated in UTC+8, covering 2026-10-03 16:00 → 2026-10-04
    16:00 UTC;
  - the 400-level order-book file labelled 2026-10-04, dated in UTC, covering 2026-10-04 00:00 →
    2026-10-05 00:00 UTC.

  The first 256 KB of that day's 5000-level file were also fetched. All of this lies inside
  `doge_fwd`. A sketch of the probe in §11 ran on the 16 hours the files share. It produced fill
  rates, mid-price markouts and implementation shortfall under four fill models, and printed
  summary statistics of the trades, including their price range. These were seen. They are why
  §11 brackets fills between two models and uses a 60-second order life. No factor was computed.
  `doge_fwd`'s report flags 2026-10-03 16:00 → 2026-10-05 00:00 UTC.
- **General knowledge.** DOGE's rallies in early 2024 (inside `doge_2023`) and late 2024 (inside
  `doge_2024`) are widely known. The late-2024 rally was one of the reasons given for dividing
  `macd_hist` by close. The other reason, its buckets drifting with the price level, was seen in
  sample. No other choice here refers to either rally.

## 4. Data

### 4.1 Sets

Windows are half-open, `[start, end)`, in UTC. Boundaries fall at 20:00 UTC because the in-sample
download covers 2025-09-23 20:02 → 2026-09-23 20:00: the DOGE test years end and begin there.

A bar is evaluated only if two conditions hold:

- its start lies in the evaluation window;
- its forward-return window lies inside the window, i.e. start(t+1+h) + B ≤ end (§6.3).

The warm-up supplies indicator history and quantile burn-in, and is never evaluated.

| Set | Role | Instrument(s) | Evaluation window | Warm-up starts |
|---|---|---|---|---|
| `IS` | direction rule and power only (§6.5, §9) | DOGE-USDT-SWAP | 2025-12-23 00:00 → 2026-09-23 20:00 | 2025-09-24 00:00 (inside IS) |
| **`doge_2023`** | **primary, confirmatory** | DOGE-USDT-SWAP | 2023-09-23 20:00 → 2024-09-23 20:00 | 2023-06-25 20:00 |
| `doge_2024` | secondary: replication | DOGE-USDT-SWAP | 2024-09-23 20:00 → 2025-09-23 20:00 | 2024-06-25 20:00 |
| `majors_2023` | secondary: other instruments, same period as the primary | BTC-, ETH-, SOL-USDT-SWAP | 2023-09-23 20:00 → 2024-09-23 20:00 | 2023-06-25 20:00 |
| `doge_fwd` | descriptive only | DOGE-USDT-SWAP | 2026-09-23 20:00 → 2026-11-23 20:00 | 2026-06-25 20:00 (from the in-sample files and `data/history/is_supp/`, not re-downloaded) |

**Why these sets.** `doge_2023` is primary because it is the most recent year on the sets'
calendar (years starting 23 September, 20:00 UTC) in which the author's earlier research used no
DOGE data. Its exposure is not zero: that research analysed BTC over 267 of its days, DOGE
co-moves with BTC, and §3 lists the remaining DOGE contacts.

`doge_2024`, the year directly before IS, would be the natural choice, but 77% of it was seen in
earlier work, so it can only replicate.

`majors_2023` shares the primary window. Its statistics therefore move with the primary's, so its
label describes breadth rather than independent confirmation (§7). It is, however, disjoint in time
from IS, where the directions are fixed (§6.5); a majors set inside IS would be biased by that
choice.
`doge_fwd` is analysed once, after its window closes. Its end date is fixed here so that choosing
when to stop cannot become a degree of freedom.

### 4.2 Sources and integrity

- **Candles.** 1-minute candles for the perpetuals and their spot legs (DOGE-USDT, BTC-USDT,
  ETH-USDT, SOL-USDT) come from OKX's public REST endpoint `history-candles`, confirmed bars only
  (`confirm = "1"`).
  - `volume` is the `vol` field: contracts for swaps, base currency for spot.
  - OKX's monthly archive (`market-data-history`, module 2) is used only for a day REST cannot
    return. Its files are cut at UTC+8 day boundaries and are re-cut to UTC.
  - Before an archive day is used, the three REST-available days nearest to it within the same
    set (the earlier day first on ties) are compared with the archive field by field. If any value differs by more than 10⁻⁹ relative, the archive is not
    used, and its bars count as missing.
- **Funding rates.** These come from the monthly archive (module 3), concatenated and de-duplicated
  on settlement time, with REST `funding-rate-history` for months not yet archived.
  - The rate is OKX's `fundingRate` (the archive's `funding_rate`), stamped at `fundingTime`.
  - Archive files are listed for every month the sets need (§3). If a set's funding is
    nonetheless unavailable, its `funding_cost_pressure` cells are untestable there (§6.8).
  - Any settlement spacing other than 8 h is reported. On a bar whose start is 8 h or more after
    the latest settlement at or before it, `funding_cost_pressure` is undefined. The limit is a
    fixed 8 h, the regular interval, so no bar's mask depends on later settlements. The research pipeline applies this mask to the output
    of `calculate_funding_cost_pressure`, which is not changed.
- **No interpolation.** A resampled bar is kept only if all its 1-minute bars exist. Forward returns
  that span a missing bar are undefined (§6.3).
- **Manifest.** A manifest lists, for every series:
  - its source, day by day;
  - row count;
  - first and last timestamp;
  - gap list;
  - SHA-256.

  It is committed and pushed before any analysis of that series (§10).
- **The in-sample files are never modified.** Their SHA-256 values are in Appendix A. The
  supplementary in-sample series (full-year funding and the DOGE-USDT spot leg for the IS year)
  are stored separately and get their own manifest (§10, step 2).

## 5. Factors

Ten factors, computed on the bars of the analysis frequency. Windows count **bars**, not clock
time. This matches v1 and avoids re-choosing any window. The 60- and 1440-bar windows were sized
for 1-minute bars, i.e. one hour and one day. At 15m they span 15 hours and 15 days; at 1H, 60
hours and 60 days.

A dropped bucket (§6.1) is removed, not filled. Every factor treats the remaining bars as
consecutive, as `factor_engine.py` does. Gaps affect only forward returns (§6.3). C, Hi and Lo
denote a bar's close, high and low.

For the seven existing factors, "code" means the functions in `quant_core/factor_engine.py` as
they stand in the commit that contains this document. A test checks that they, and the helpers
they call (`_calculate_rsi_numba`, `_calculate_ema_numba`, `_asof_backward`), are unchanged at the
freeze tag. Any change is a deviation (§12).

Where an unchanged function's output at a bar depends on later bars, the research pipeline keeps a
value only if the function, run on the data up to that bar, would produce it. Two cases exist, both
inside the warm-up:

- `calculate_macd_hist` defines index slow + signal − 2 only when at least slow + signal bars
  exist;
- `calculate_basis_z_score` returns nothing unless the whole series has ≥ 1440 finite basis
  values.

### 5.1 Existing factors

| Factor | Definition (code) | Parameters |
|---|---|---|
| `rsi_14` | Wilder RSI of close (`calculate_rsi`) | 14 |
| `macd_hist_norm` | output of `calculate_macd_hist` divided by close. **Changed from v1** so the factor is scale-free. It is computed in the research pipeline; `calculate_macd_hist` itself is not changed, because the model's 17-feature list uses `macd_hist`. | 12 / 26 / 9 |
| `bollinger_pctb` | (close − lower) / (upper − lower), bands = SMA ± 2 population σ; 0.5 if σ = 0 (`calculate_bollinger_pctb`) | 20, 2.0 |
| `rolling_vol_ratio` | mean volume, last 60 bars ÷ mean volume, last 1440 bars; 1.0 if the long mean is 0 (`calculate_rolling_vol_ratio`). A volume ratio, despite its name. | 60, 1440 |
| `net_taker_flow` | Σ CLV·volume ÷ Σ volume over 60 bars, CLV = ((C−Lo) − (Hi−C)) / (Hi−Lo), 0 if Hi = Lo. This is Chaikin Money Flow (`calculate_net_taker_flow`, without trade-side data). | 60 |
| `basis_z_score` | z-score (population σ, non-finite values ignored) of (perp close − spot close) / spot close over 1440 bars (`calculate_basis_z_score`, joined as-of backward on bar start). The research pipeline supplies the spot leg resampled to the same buckets on the perp bar grid, with a missing close where a spot bucket was dropped, so the basis is undefined there rather than carried forward. | 1440 |
| `funding_cost_pressure` | on the settlement timeline, z-score (population σ) of the funding rate over the last 90 settlements including the current one, or all available if fewer, minimum 10; joined as-of backward on bar start (`calculate_funding_cost_pressure`) | 90, 10 |

RSI 14, MACD 12/26/9 and Bollinger 20/2 are textbook defaults. The other windows are v1's values.

### 5.2 New factors

x_t = ln C_t and r_t = x_t − x_{t−1}.
"Positive" means the factor expects the price to rise.

**`ou_reversion`: mean-reversion forecast from a rolling AR(1).** Over the last W = 240 pairs
(x_{s−1}, x_s), s = t−W+1 … t, fit by OLS x_s = a + φ·x_{s−1} + e_s. Let
σ_e = √(Σ e_s² / (W − 2)). Then

  ou_t = (a + (φ − 1)·x_t) / σ_e

This is the fitted next-bar change in residual-σ units. It is undefined if σ_e = 0 or
Σ (x_{s−1} − x̄)² = 0, where x̄ is the mean of x_{s−1} over the window. There is no gate on φ.

W = 240 rather than a shorter window, because near a unit root the OLS estimate of φ is biased
down by about (1 + 3φ)/W ≈ 4/W (Kendall, 1954). With W = 60 the forecast would mostly reflect that
bias.

**`hurst_regime`: momentum signed by the variance-ratio regime.** Over the last W = 1440 returns r,
with q = 5, use the bias-corrected estimators of Lo and MacKinlay (1988):

- r̄ = mean r;
- σ²_a = Σ (r − r̄)² / (W − 1);
- σ²_c = Σ_k (r_k + … + r_{k−q+1} − q·r̄)² / M, over the W − q + 1 overlapping q-sums, with
  M = q(W − q + 1)(1 − q/W).

Then VR = σ²_c / σ²_a, Ĥ = ½ + ln VR / (2 ln q), σ_a = √σ²_a, and
m_t = (x_t − x_{t−q}) / (σ_a √q), and

  hurst_t = (Ĥ − ½) · m_t

The factor follows recent momentum when the window trends (Ĥ > ½) and fades it when the window
mean-reverts (Ĥ < ½). It is undefined if VR ≤ 0 or σ_a = 0. This interaction counts as one
hypothesis. W = 1440 matches the long windows of §5.1; q = 5 is the middle tested horizon.

**`triple_screen`: higher-timeframe trend, analysis-timeframe pullback.**

- **HTF bars.** HTF bars are 4 × the analysis bar B (15m → 1H, 1H → 4H), built from 1-minute bars
  by the §6.1 rule.
- **Trend.** tr = sign(HTF MACD histogram, 12/26/9, from `calculate_macd_hist` on HTF closes).
  An analysis bar starting at s, which closes at s + B, uses the latest existing HTF bar starting
  at or before ⌊(s + B) / 4B⌋·4B − 4B. That is the latest HTF bar already closed. For example, the
  15m bar starting 10:30 uses the 1H bar starting 09:00, and the 1H bar starting 10:00 (closing
  11:00) uses the 4H bar starting 04:00, because the 4H bar starting 08:00 closes at 12:00.
- **Pullback.** %K_t = 100 (C_t − min Lo_{t−13..t}) / (max Hi_{t−13..t} − min Lo_{t−13..t}),
  undefined if the range is zero.

Then

  triple_t = tr · max(0, tr·(50 − %K_t)) / 50 ∈ [−1, 1]

The factor is positive in an uptrend pullback, negative in a downtrend rally, and 0 otherwise. It
is undefined while the HTF histogram is undefined.

MACD 12/26/9 and the 14-bar %K are the usual defaults. The rest are choices made here, not
Elder's (1993):

- **Ratio.** 4× keeps both higher timeframes on standard bar sizes. Elder's ratio of about five
  would give 75m and 5h. The implementation ported here ran 5m → 1H (12×).
- **Trend screen.** It uses the sign of the HTF histogram, as that implementation did; Elder uses
  its slope.
- **Look-ahead removed.** That implementation also mapped each HTF bar onto the base bars inside
  it, a look-ahead the rule above removes.
- **Signal form.** Its ternary signal (+1 when %K turns up from below 20 in an uptrend, −1 when it
  turns down from above 80 in a downtrend) is 0 on almost every bar, so the continuous form above,
  centred on %K = 50, is used instead.

### 5.3 Excluded, in writing

- `oi_change_rate`: timestamp look-ahead (§2, item 4). Also, OKX keeps 5-minute OI for a few days,
  15-minute OI for about 15 days and 1-hour OI for about 60 days, so no test set can support it at
  15m or 1H.
- `downside_beta`: it pairs the BTC leg with the candles by array index, returns a constant 0 when
  samples are scarce, and computes a correlation rather than a beta.

## 6. The test statistic

A **cell** is one factor × bar × horizon: 10 × 2 × 3 = 60 per set.

**6.1 Bars.** B ∈ {15m, 1H}. Bucket k covers [k·B, (k+1)·B) in UTC milliseconds. OHLCV are
first / max / min / last / sum, and the timestamp is the bucket start. A bucket missing any
1-minute bar is dropped.

**6.2 Factors** are computed on each set's bar sequence, warm-up included (§5).

**6.3 Forward return.** For a signal at bar t, the position is entered at the close of t+1 (a
one-bar delay) and exited at the close of t+1+h:

  fwd_t = C_{t+1+h} / C_{t+1} − 1

It is defined only if bars t … t+1+h are consecutive, i.e. every step is exactly B.

**6.4 Buckets: expanding window, strict edges.** At each evaluation bar t, the 20th and 80th
percentiles P20 and P80 (linear interpolation) are computed from the factor's finite values at all
earlier bars of the set, warm-up included. At least 500 earlier values are required, otherwise the
bar is unassigned. Then:

- s_t = +1 if f_t > P80;
- s_t = −1 if f_t < P20;
- s_t = 0 otherwise, including values equal to an edge and non-finite f_t.

For a factor without repeated values this is the usual quintile rule. For a factor with a point
mass, such as `triple_screen` at 0 (about 60% of bars in synthetic simulations), the mass stays out
of both traded buckets even when an edge falls on it. The realised share of each traded bucket is
reported for every cell.

**6.5 Direction is fixed in sample, by rule.** For each cell, d is the sign of the mean of
s_t · fwd_t over the active bars of the IS evaluation window, computed with the frozen code. d = +1
if that mean is exactly 0. If the IS cell has fewer than 300 active bars or lacks an input, d = +1
and the cell is flagged.

The 60 directions are written to `results/insample_v2.csv` in the commit tagged `prereg-freeze`,
and are copied into the freeze amendment before the test-set download (§10). Each test then asks:
does the in-sample direction earn money out of sample?

`results/insample_v2.csv` has one row per (bar, horizon, factor), with these columns:

- direction;
- flagged (true/false) and flag_reason (`few_active`, `missing_input` or blank);
- n_active;
- mean_s_fwd_bps (the mean that sets d);
- se_bps (§6.7, on the IS evaluation window) and se_method;
- mde_bps (§9).

se_bps and mde_bps are blank when the IS cell has no active bars, and the freeze amendment then
records that cell's MDE as undefined.

**6.6 Per-trade net return.** Each active bar is one trade:

- g_t = d · s_t · fwd_t;
- n_t = g_t − c, with c = 10 bp per round trip (taker, 5 bp a side).

The estimate is μ̂ = the mean of n_t over the active set A = {t : s_t ≠ 0, fwd_t defined}. When
h > 1, consecutive trades overlap: μ̂ is the average trade, not the return of a capital-constrained
strategy. R5 reports the capital-constrained version.

**6.7 Standard error: Hansen–Hodrick.** Let z_t = 1{t ∈ A}·(n_t − μ̂), indexed by bucket start on
the regular B-grid of the evaluation window, with z_t = 0 for dropped buckets and inactive bars.

  Ω = Σ_{k = −ℓ..ℓ} Σ_t z_t z_{t+k}, with lag ℓ = h − 1 (uniform weights)

Then SE = √Ω / |A|, t_HH = μ̂ / SE, and the one-sided p = 1 − Φ(t_HH). If Ω ≤ 0, Bartlett weights
with ℓ = 2h are used instead, and the fallback is flagged. If SE = 0, t_HH is undefined and the
cell's p is 1.

The lag ℓ = h − 1 covers exactly the overlap between forward-return windows. In synthetic
simulations, with iid, persistent and heavy-tailed cases, one-sided 5% rejection rates were
4.3–5.8%. It does not cover dependence from a persistent signal interacting with a slowly varying
mean return, which makes the test liberal in strongly trending periods. In synthetic runs with
the persistence of the slowest in-sample signal (AR coefficient 0.98–0.99) and a drift of ±0.03σ
per bar switching monthly, rejection at the first Holm threshold was about 2–2.5 times nominal,
and about 5 times at ±0.05σ. `doge_2023` contains a widely known rally (§3). R1, which keeps both
the signal's persistence and the realised drift path, checks this, and §7 fixes in advance how a
confirmation that does not replicate is described.

**6.8 Untestable cells.** A cell is untestable if an input series is unavailable for the set, or if
|A| < 300. It is recorded with p = 1 and **stays in the family**.

## 7. Hypotheses, families and the decision rule

- **Primary family F_primary.** The 60 cells on `doge_2023`, each with H0: E[n_t] ≤ 0.
  - It is tested by Holm step-down at a family-wise error rate of 0.05, one-sided. The first step
    rejects when p ≤ 0.05/60 ≈ 0.000833, i.e. t_HH ≥ 3.144 (rounded). Decisions use p.
  - A cell rejected in F_primary is **confirmed**.
- **Secondary family F_doge2024.** The same 60 cells on `doge_2024`, Holm at 0.05. A confirmed
  cell also rejected here is labelled **replicated**. For the five factors that were DOGE inputs in
  the earlier research (§3), which covered 77% of `doge_2024`, `replicated` is not independent of
  that research.
- **Secondary family F_majors.** The same 60 cells on `majors_2023`, pooled across BTC, ETH and SOL,
  each in its DOGE in-sample direction.
  - Pooled null: ⅓ Σ_i E[n_{i,t}] ≤ 0, with estimate μ̂_pool = ⅓ Σ_i μ̂_i.
  - SE_pool = √Ω_w, where Ω_w is the §6.7 sum (same ℓ and fallback) computed on
    w_t = ⅓ Σ_i z_{i,t} / |A_i|. Here t is the bucket start on the common B-grid, with z_{i,t} = 0
    where instrument i has no active bar.
  - Because μ̂_pool − E[μ̂_pool] = Σ_t w_t, and w_t already contains 1/|A_i|, Ω_w is not divided by
    any |A|. This keeps the same-time correlation between instruments.
  - The pooled cell is untestable if any instrument's cell is.
  - Holm at 0.05. A confirmed cell also rejected here is labelled **consistent on majors**.
  - `majors_2023` shares the primary window and DOGE co-moves with the majors, so F_majors's
    t-statistics are positively correlated with F_primary's even when no factor works. This label
    is therefore not independent confirmation: it describes breadth across instruments in the same
    year. Only `replicated` comes from a different year.
- **Only F_primary confirms.** The secondary families add labels only.
- **Confirmed but not replicated.** For a confirmed cell, the report gives the realised mean run
  length of s_t, R3 and R4 next to the decision. If the cell is not also `replicated`, the report
  calls it significant on `doge_2023` only, under a test known to be liberal for persistent signals
  in trending periods (§6.7), and not evidence of a profitable factor.
- **`doge_fwd`** reports μ̂ and a two-sided 95% interval per cell, and makes no decision.
- **Descriptive q-values.** Benjamini–Hochberg (BH) and Benjamini–Yekutieli (BY) q-values are
  reported for the three families as descriptive figures only.
- **Headline.** If no cell is confirmed, the report says that no factor × bar × horizon cell showed
  a significantly positive per-trade return at taker cost on `doge_2023` (Holm, family-wise error
  0.05). It does not say the returns were zero or negative. It lists the cells whose upper
  confidence bound is below zero (§9), and gives every other cell's μ̂, upper bound and minimum
  detectable effect.

## 8. Robustness: reported, never decisive

All rows are computed on `doge_2023` only. The other sets get no robustness rows; their columns
are listed in §10.

| | |
|---|---|
| R1 | **Circular-shift placebo.** G is the evaluation window's grid of N_G bar positions. The offsets are δ_j = D_30 + round(j·(N_G − 2·D_30)/(K − 1)), for j = 0 … K−1, with K = 2000 and D_30 = 30 days in bars. Duplicates are dropped, leaving K′. For each δ_j, s is rotated within G (s′_t = s_{(t−δ_j) mod N_G}), and A, n_t and t_HH are recomputed with the same d and c. Per-cell p = (1 + #{t_j ≥ t_HH}) / (1 + K′), using that cell's own t_HH. Family-wise p = (1 + #{max over testable cells of t_j ≥ t_HH}) / (1 + K′), with the same offset index j for every cell; an index is kept only if its offset is new on both bar grids. A rotated cell is evaluated even if it has fewer than 300 active bars. Both test "no association between signal and returns", not E[n] ≤ 0, and neither enters Holm. |
| R2 | **v1's statistic.** `_spread_t_stat`, as in the commit containing this document: a gross top-minus-bottom two-sample t on quintile edges computed over the whole evaluation window, ÷ √h, not multiplied by d. Kept for continuity. |
| R3 | **μ̂ by quarter.** Four blocks, split at 20:00 UTC on 2023-12-23, 2024-03-23 and 2024-06-23. Trades are assigned by the start of bar t. |
| R4 | **Leave out the k most volatile UTC days**, k ∈ {0, 5, 10}. Days are ranked by the sum of squared simple 1-minute DOGE returns within the window. A 1-minute return belongs to the UTC day of its later minute. A trade is dropped if its holding interval [close of t+1, close of t+1+h] shares a stretch of positive length with one of those days. |
| R5 | **Turnover-aware P&L.** The position after the close of bar u is p_u = (1/h) Σ_{j=1..h} d·s_{u−j}, with s = 0 where unassigned, so each signal is held from the close of t+1 to the close of t+1+h. It earns p_u·r_{u+1} and pays c/2 per unit of \|p_u − p_{u−1}\|. Only evaluable bars' signals are traded, and a return across a dropped bucket contributes nothing. Reported as net return per day: the sum divided by the window's length in days. |
| R6 | **Cost sensitivity**, c ∈ {0, 4, 7, 11} bp: 0 is gross; 4 is maker on both sides at §11's fees; 7 is maker in, taker out; 11 is taker plus half of a one-tick spread a side (a 0.00001 tick at a price near 0.1 is about 1 bp). |
| R7 | **Winsorised returns.** fwd_t winsorised at the 1st and 99th percentiles of all defined fwd_t in the window, for that bar and h. |
| R8 | **Funding included.** n_t − d·s_t·Σ_u FR_u, summing the settled rates FR_u whose settlement time τ_u satisfies entry ≤ τ_u < exit (entry and exit are the closing times of bars t+1 and t+1+h). A long position pays a positive rate. R8 is undefined for a trade whose holding interval contains an expected settlement time with no recorded rate; it is averaged over the trades for which it is defined, and that count, with the §6.7 SE, goes to the robustness file. |
| R9 | **Predictability rather than profitability.** The Spearman IC between f_t and fwd_t over the evaluable bars where both are defined: IC = mean of u_t·v_t, where u and v are the ranks of f and fwd (average ranks for ties) standardised to mean 0 and variance 1. Its SE is the §6.7 Hansen–Hodrick SE of that mean (ℓ = h − 1, products indexed on the grid, zero where undefined). The test is two-sided, and Holm at 0.05 is applied within this separate family of 60. This separates "predicts" from "pays". A mean of daily ICs was registered first and replaced before publication: ranking within a day biases it for persistent factors even on a random walk (§12). |
| R10 | **Monotonicity.** Mean gross fwd_t (not multiplied by d) in each of the five expanding-window quintiles, with edges e1 < … < e4 at the 20th to 80th percentiles: Q1 = f < e1, Q2 = [e1, e2), Q3 = [e2, e3], Q4 = (e3, e4], Q5 = f > e4, so Q1 and Q5 are §6.4's traded buckets. Where e1 = e2, Q2 folds into Q3; where e3 = e4, Q4 folds into Q3. |

## 9. Power: what a null result will mean

Before the test-set download (§10, step 4), each cell's in-sample standard error SE_IS from the
frozen code is scaled to the primary set:

  SE_primary ≈ SE_IS · √(N_IS / N_primary)

Here N_set = (end − start)/B − (h + 1) is the number of evaluable bars, assuming no gaps; for
example N_IS = 6,596 − h − 1 and N_primary = 8,784 − h − 1 at 1H. This keeps each cell's own
in-sample active share.

The minimum detectable effect at 80% power is MDE = (3.144 + 0.842) · SE_primary, in bp per trade
net of the 10 bp cost (gross equivalent: MDE + 10 bp), assuming every other cell is null. The MDE
table goes into the freeze amendment. After the run, each cell's MDE from its realised standard
error is reported too.

The results report, for every `doge_2023` cell:

- **ub95 = μ̂ + 1.645·SE.** This is a one-sided 95% upper confidence bound on its true mean net
  return E[n_t]. Claims about one cell use ub95. A negative ub95 is evidence that the cell does not
  pay at taker cost, not merely an absence of evidence.
- **ub95_family = μ̂ + 3.144·SE.** This holds simultaneously over the 60 cells. Claims about the
  whole family ("no cell earns more than X bp per trade") use ub95_family.

## 10. Protocol

1. **Publication.** This document is committed, made public, and a GitHub Release is cut from that
   commit. The commit is archived with Software Heritage at the same time.
2. **Implementation.**
   - The statistics (§6–§8) and the three new factors (§5.2) are written with tests.
     `tests/test_prereg_spec.py` parses Appendix A and checks the code's constants against it, and
     checks that the §5.1 functions, and the helpers they call, are unchanged.
   - Kernels that test for undefined values are compiled without Numba's `fastmath`, under which
     `np.isnan(x)` and `x != x` evaluate to False. Each new factor has a test that every "undefined
     if" condition in §5.2 yields NaN.
   - The supplementary in-sample series (DOGE-USDT spot and DOGE-USDT-SWAP funding, 2025-09-24 →
     2026-09-23) are downloaded to `data/history/is_supp/`. Their manifest,
     `results/manifest_is_supp.json`, is committed and pushed.
   - The IS rerun, from a clean checkout, writes `results/insample_v2.csv`. That file is created
     exclusively and never overwritten. It is added in the next commit with no other change, and
     that commit is tagged `prereg-freeze`.
3. **Freeze amendment.** A commit adds `docs/preregistration-freeze.md`, recording:
   - the commit that `prereg-freeze` points to;
   - the 60 directions and flags (§6.5) and the MDE table (§9), as they appear in
     `results/insample_v2.csv`;
   - the Software Heritage identifiers of the publication commit and of `prereg-freeze`;
   - the maker-probe days (§11).

   It is pushed before step 4.
4. **Download.** Test-set data goes to `data/history/oos/<set>/<instrument>/`. Each set's manifest,
   `results/oos/manifest_<set>.json`, is committed and pushed before any analysis.
5. **One run.** `scripts/run_oos.py` runs once per set, from a clean checkout of `main` whose code
   is that of `prereg-freeze`.
   - It stops unless `prereg-freeze` is an ancestor of HEAD and resolves to the commit recorded in
     the freeze amendment.
   - It stops unless every path listed by `git diff --name-only prereg-freeze HEAD` is under
     `docs/` or `results/`.
   - It reads each cell's direction from `git show prereg-freeze:results/insample_v2.csv`, and
     stops if any direction differs from the freeze amendment.
   - It stops if any input file's SHA-256 differs from the set's committed manifest.
   - Outputs are created exclusively, never overwritten, under `results/oos/`. They are committed to
     `main` and pushed unedited.
6. **Comparison.** The results are compared with this document in a separate commit. Any rerun
   goes to `results/oos/rerun-N/` with its reason, and is reported alongside the first run, which
   stays primary.

Steps 4–6 for `doge_fwd` happen after 2026-11-23 20:00 UTC.

**Results files.** `results/oos/<set>.csv` has one row per (instrument, bar, horizon, factor); those
four key columns come first. For `majors_2023`, each cell also has a row with
instrument = `POOLED`, which carries the F_majors statistics. The other columns, in order:

- direction, direction_flagged, n_eval (bars whose forward return is defined), n_active (|A|),
  share_top, share_bottom
- gross_bps, net_bps, se_bps, se_method (hh | bartlett_fallback), t_hh, p_one_sided, p_holm
- q_bh, q_by, ub95_net_bps, ub95_family_net_bps, ci95_lo_bps, ci95_hi_bps (two-sided, μ̂ ∓ 1.960·SE)
- mde_bps (from the freeze amendment), mde_realised_bps
- p_placebo, p_placebo_family (R1), t_sqrt_h (R2), net_funding_bps (R8), ic, ic_t, ic_p,
  ic_p_holm (R9)
- status: `confirmed`, `not_rejected` or `untestable` on the primary; `rejected`, `not_rejected` or
  `untestable` on the secondary families; `descriptive` elsewhere. Plus `replicated` and
  `consistent_on_majors` (true/false; false for cells that are not confirmed) on primary rows.
- code_commit, manifest_sha256

Columns are filled as follows, and are blank otherwise:

- **Every row:** direction, direction_flagged, gross_bps, net_bps, se_bps, se_method, status,
  code_commit, manifest_sha256.
- **Every row except `POOLED`:** n_eval, n_active, share_top, share_bottom.
- **`doge_2023` rows, `doge_2024` rows and `majors_2023` `POOLED` rows:** t_hh, p_one_sided,
  p_holm, q_bh, q_by.
- **Per-instrument `majors_2023` rows:** no t or p; status `descriptive`.
- **`doge_fwd` rows:** ci95_lo_bps, ci95_hi_bps.
- **`doge_2023` rows only:** ub95_net_bps, ub95_family_net_bps, mde_bps, mde_realised_bps, the R1,
  R2, R8 and R9 columns, replicated and consistent_on_majors.

The multi-valued rows R3–R7 and R10 go to `results/oos/robustness_doge_2023.csv`, one row per
(bar, horizon, factor, row, parameter), with columns value_bps, se_bps, t_hh and n_active. What
those columns hold depends on the row:

- **R3, R4, R6 and R7:** se_bps and t_hh are the §6.7 statistics.
- **R5:** value_bps is net bp per day, and se_bps and t_hh are blank.
- **RL:** the realised mean run length of s_t in bars that §7 requires for a confirmed cell, over
  the evaluable bars (a sign change, s = 0 or a dropped bucket ends a run). value_bps holds the
  length in bars, n_active the number of runs, and se_bps and t_hh are blank. It is written for
  every cell so that §7 needs no second pass over the data.
- **R10:** value_bps is the quintile's mean gross fwd_t, and se_bps and t_hh are the §6.7
  statistics at ℓ = h − 1 on that quintile's bars.

## 11. Maker-execution probe (exploratory, descriptive)

**Question.** Does resting at the touch beat crossing the spread, once adverse selection is
counted?

- **Days.** The candidates are the UTC days D from 2025-09-24 to 2026-09-22 (the IS year) for which
  OKX's archive lists four files:
  - the order-book files (module 4, UTC dates) for D and D+1;
  - the trade files (module 1, UTC+8 dates) for D and D+1.

  The candidate list, sorted ascending, is saved in `results/manifest_maker_probe.json`, committed
  before the draw. 14 days are drawn with
  `numpy.random.default_rng(20261008).choice(candidates, size=14, replace=False)` under numpy
  1.24.3, as pinned in `environment.yml`. If fewer than 14 qualify, all are used. The rebuilt book
  is checked against the archive's 15-minute snapshots. The sketch's files (§3) lie outside the IS
  year and cannot be drawn.
- **Design.** At decision times every 5 minutes, post one contract at the best bid (a buy) and,
  separately, at the best ask (a sell), each with a 60-second life. Two fill models bound the true
  fill:
  - **M1, lower bound.** A fill only when a trade prints strictly through the order's price, or the
    opposite best crosses it.
  - **M2, upper bound.** The queue ahead starts at the displayed size at the order's price. It
    shrinks with trades at that price and with size decreases at the level, capped at the displayed
    size. The order fills once the queue is exhausted and a trade prints at its price, or whenever
    M1 would fill it.
- **Metrics.**
  - fill rate;
  - mid-price markout at 1, 5 and 15 minutes after a fill;
  - implementation shortfall of "post, then cross after 60 s if unfilled" against crossing
    immediately, from the decision-time mid and including fees. Fees are maker 2 bp and taker 5 bp,
    the regular tier from published secondary sources, unverified;
  - standard errors across days.
- **Use.** The output informs the discussion of cost only. It never replaces c = 10 bp.
- **Limits.** Time-boxed to two evenings. If the data cannot be had, the report states what is
  missing and stops.

## 12. Deviations log

| Date | Section | Deviation | Reason | Test-set data seen? |
|---|---|---|---|---|
| 2026-10-09 | §8 R9 | The mean of daily ICs was replaced by a full-window Spearman IC with a Hansen–Hodrick SE. | Synthetic tests during implementation showed the daily version is biased for persistent price-derived factors on a pure random walk (mean daily IC about −0.07 at 15m and −0.19 at 1H, \|t\| about 14 and 20), because ranking within a day shares the day's mean between the forward returns and the recent returns the factor is built from. | No. Made before publication; no in-sample v2 or test-set statistic had been computed. |
| 2026-10-09 | §5, §6.7, §8 R1, R4, R5, R8, R10, §10 | Implementation details made explicit: warm-up values that depend on later bars; p = 1 when SE = 0; R1 per-cell t and offset indexing; R4 day assignment and overlap; R5 trading and per-day rule; R8 count location; R10 quintile rule; results key columns and label values; funding spacing taken from the series. | Ambiguities found while implementing; each was resolved by its most literal reading. | No. Made before publication; no in-sample v2 or test-set statistic had been computed. |
| 2026-10-11 | §4.2, §10 | The funding staleness limit is a fixed 8 h instead of the series' median spacing. A robustness row RL (mean run length of s_t) is added. | Code review found that a median over the whole series lets later settlements decide whether an early bar is masked, and that §7's run length was computed nowhere, so it would have needed a second pass over test data. | No. Made before publication; no in-sample v2 or test-set statistic had been computed. |
| 2026-10-12 | §11 | The probe's archive files are downloaded only after the last test set has run (`doge_fwd`, after 2026-11-23 20:00 UTC). | Two files the population needs reach past the in-sample year: the trade file dated 2025-09-24 (UTC+8) starts at 2025-09-23 16:00 UTC, inside `doge_2024`, and the order book dated 2026-09-23 runs to 2026-09-24 00:00 UTC, inside `doge_fwd`; either is read if its day is drawn. | No. Made before publication; only the archive's file listing had been read. |
| 2026-10-12 | §10 step 1 | The publication commit was archived with Software Heritage at 08:55 UTC, not at the time of the Release (07:37). | The save request made at 07:40 failed. A later visit, at 08:55, archived the publication commit and the freeze commit (07:51) together, in snapshot `swh:1:snp:7baf60bb2da0a53f8120a3ff56074ac410fb1b28`. That was before the freeze amendment (09:01), which `scripts/fetch_v2.py` requires on the public `main` before it downloads test data, so it was before every test-set download. Until 08:55 the publication time rests on GitHub's record alone. | Yes. Logged after the `doge_2024`, `majors_2023` and `doge_2023` runs; the archive itself predates every test-set download. |

## References

- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B* 57(1), 289–300.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics* 29(4), 1165–1188.
- Elder, A. (1993). *Trading for a Living*. Wiley.
- Hansen, L. P. & Hodrick, R. J. (1980). Forward exchange rates as optimal predictors of future spot rates. *Journal of Political Economy* 88(5), 829–853.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 65–70.
- Kendall, M. G. (1954). Note on bias in the estimation of autocorrelation. *Biometrika* 41(3–4), 403–404.
- Lo, A. W. & MacKinlay, A. C. (1988). Stock market prices do not follow random walks. *Review of Financial Studies* 1(1), 41–66.
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.

## Appendix A: machine-readable specification

```yaml
version: 1
written: "2026-10-08"
timezone: UTC
windows_half_open: true

sets:
  IS:          {inst: [DOGE-USDT-SWAP], warmup_start_ms: 1758672000000, start_ms: 1766448000000, end_ms: 1790193600000, role: direction_and_power}
  doge_2023:   {inst: [DOGE-USDT-SWAP], warmup_start_ms: 1687723200000, start_ms: 1695499200000, end_ms: 1727121600000, role: primary}
  doge_2024:   {inst: [DOGE-USDT-SWAP], warmup_start_ms: 1719345600000, start_ms: 1727121600000, end_ms: 1758657600000, role: secondary_replication}
  majors_2023: {inst: [BTC-USDT-SWAP, ETH-USDT-SWAP, SOL-USDT-SWAP], warmup_start_ms: 1687723200000, start_ms: 1695499200000, end_ms: 1727121600000, role: secondary_pooled}
  doge_fwd:    {inst: [DOGE-USDT-SWAP], warmup_start_ms: 1782417600000, start_ms: 1790193600000, end_ms: 1795464000000, role: descriptive}

spot_legs: {DOGE-USDT-SWAP: DOGE-USDT, BTC-USDT-SWAP: BTC-USDT, ETH-USDT-SWAP: ETH-USDT, SOL-USDT-SWAP: SOL-USDT}

data:
  candles: {source: rest_history_candles, confirm_only: true, volume_field: vol,
            archive_fallback: {module: 2, file_tz: UTC+8, recut_to: UTC, crosscheck_rest_days: {n: 3, choice: nearest_in_set_earlier_first}, max_rel_diff: 1.0e-9, on_mismatch: treat_as_missing}}
  funding: {source: [archive_module_3, rest_funding_rate_history], field: fundingRate, timestamp: fundingTime, dedupe_on: fundingTime, expected_spacing_hours: 8}

insample_dir: data/history/DOGE-USDT-SWAP
insample_sha256:
  candles_1m.parquet:    275ba73aef1640f7d4fd09b9e57ce31462db877050b34f9a1112e5bebb42895e
  funding_rates.parquet: 4e9631824cc6ee3542a455233793f69c0e28b3972e199b84ba4f97a491241079
  open_interest.parquet: 86d8d25220d7057a6978a9e6277a7457004e01c44f971aecc0494e1799ab00fe

bars: [15m, 1H]
horizons: [1, 5, 15]          # in bars of the analysis frequency
bar_aggregation: {open: first, high: max, low: min, close: last, volume: sum, timestamp: bucket_start, drop_incomplete: true}
forward_return: {entry: close_t_plus_1, exit: close_t_plus_1_plus_h, require_consecutive_bars: true, window_inside_eval: true}
gaps: {factor_windows: treat_remaining_bars_as_consecutive, interpolation: none}

factors:
  rsi_14:                {period: 14}
  macd_hist_norm:        {fast: 12, slow: 26, signal: 9, divide_by_close: true, source_function_unchanged: calculate_macd_hist}
  bollinger_pctb:        {period: 20, n_std: 2.0, std_ddof: 0, zero_sigma_value: 0.5}
  rolling_vol_ratio:     {short: 60, long: 1440, zero_long_mean_value: 1.0}
  net_taker_flow:        {lookback: 60, side_data: none}
  basis_z_score:         {lookback: 1440, std_ddof: 0, nan_policy: omit_in_window, join: asof_backward_bar_start, spot_grid: perp_grid_nan_where_spot_dropped}
  funding_cost_pressure: {lookback_settlements: 90, min_settlements: 10, window: trailing_partial, std_ddof: 0, join: asof_backward_bar_start, max_staleness_hours: 8, staleness_test: ge}
  ou_reversion:          {window: 240, gate: none, sigma_df: W_minus_2}
  hurst_regime:          {window: 1440, q: 5, estimator: lo_mackinlay_bias_corrected}
  triple_screen:         {htf_multiple: 4, macd: [12, 26, 9], stoch_k: 14, trend: sign_of_histogram, form: continuous_signed, k_center: 50, htf_bar: latest_closed}
excluded_factors: [oi_change_rate, downside_beta]

buckets:
  quantiles: 5
  edges_pct: [20, 80]
  top_if: gt
  bottom_if: lt
  edge_ties: middle
  edges: expanding_prior_bars_including_warmup
  percentile_method: linear
  min_prior_values: 500
direction_rule: {statistic: sign_of_IS_mean_s_times_fwd, tie: 1, min_active_IS: 300, fallback: 1}

cost_round_trip_bps: 10.0
statistic: per_trade_net_mean
se: {method: hansen_hodrick, lag: h_minus_1, index: bucket_start_grid, fallback: {method: bartlett, lag: 2h}}
test: {sidedness: one_sided_greater, alpha: 0.05, correction: holm}
families:
  F_primary:  {set: doge_2023, cells: 60, label: confirmed}
  F_doge2024: {set: doge_2024, cells: 60, label: replicated, requires: confirmed}
  F_majors:   {set: majors_2023, cells: 60, label: consistent_on_majors, requires: confirmed, pooling: equal_weight_mean,
               se: {series: w_t, index: bucket_start_grid, divide_by_A: false}, untestable_if_any_instrument_untestable: true}
first_holm_threshold_t: 3.144
untestable: {min_active_trades: 300, p_value: 1.0, counts_in_family: true}
descriptive_sets: [doge_fwd]
descriptive_stats: [q_bh, q_by]
intervals: {ub95: {sides: 1, z: 1.645}, ub95_family: {sides: 1, z: 3.144}, doge_fwd_ci: {sides: 2, z: 1.960}}

robustness:
  sets: [doge_2023]
  placebo: {method: circular_shift_of_signal, k: 2000, min_shift_days: 30, offsets: "D + round(j*(N-2D)/(K-1)), deduplicated", family_max_t: true, tests: association_not_profit}
  v1_statistic: {function: _spread_t_stat, edges: full_window}
  quarter_bounds_ms: [1703361600000, 1711224000000, 1719172800000]
  quarter_assign: bar_t_start
  leave_out_volatile_days: {k: [0, 5, 10], metric: sum_sq_simple_1m_return_DOGE, drop_if: holding_interval_overlaps}
  turnover_aware: {hold_bars: h, staggered: true, fee_per_unit_change_bps: 5.0}
  cost_sensitivity_bps: [0.0, 4.0, 7.0, 11.0]
  winsorize_pct: [1, 99]
  funding_included: {window: "entry <= settlement < exit", long_pays_positive: true, missing_settlement: undefined}
  ic: {method: spearman_full_window, ties: average, statistic: mean_of_standardised_rank_products, se: hansen_hodrick, lag: h_minus_1, family_correction: holm, sidedness: two_sided}
  monotonicity: {buckets: 5, edges: expanding, merge_coincident_edges: true}

power: {power: 0.8, z_alpha: 3.144, z_power: 0.842, scaling: sqrt_N_IS_over_N_primary, report_realised_mde: true}

maker_probe:
  population: {from: "2025-09-24", to: "2026-09-22", requires: [orderbook_D, orderbook_D1, trades_D, trades_D1], order: ascending}
  candidates_file: results/manifest_maker_probe.json
  seed: 20261008
  n_days: 14
  numpy_version: "1.24.3"
  decision_every_minutes: 5
  order_life_seconds: 60
  size_contracts: 1
  fill_models: [M1_trade_through, M2_queue_position_or_M1]
  markout_minutes: [1, 5, 15]
  fees_bps: {maker: 2.0, taker: 5.0, verified: false}

freeze:
  tag: prereg-freeze
  amendment_file: docs/preregistration-freeze.md
  directions_file: results/insample_v2.csv
  directions_columns: [bar, horizon, factor, direction, flagged, flag_reason, n_active, mean_s_fwd_bps, se_bps, se_method, mde_bps]
  independent_archive: software_heritage
```
