# Results compared with the pre-registration

This file is step 6 of the protocol in `docs/preregistration.md` (§10). It compares the out-of-sample
results with what that document said would be tested and how the results would be read. It covers
the three sets that decide or label cells: `doge_2023` (primary), `doge_2024` and `majors_2023`.
`doge_fwd`, which decides nothing, was downloaded and run on 2026-11-24 and has its own section,
and so does the maker probe of §11, run the same day.

**Terms.**
- **Cell.** A cell is one factor × bar (15m or 1H) × horizon h (1, 5 or 15 bars).
- **Trade.** Each evaluable bar on which the factor is above its expanding-window 80th percentile,
  or below its 20th, is one trade. It is entered at the close of the next bar and held for h bars
  (§6.3–§6.6).
- **Direction d.** d was fixed in sample (§6.5). d = +1 is long above the 80th percentile and short
  below the 20th; d = −1 is the reverse.
- **μ̂.** Trades overlap when h > 1, so μ̂, a cell's mean net return per trade, is the average
  trade, not a strategy's return. R5 below is the turnover-aware P&L.
- **Units.** Returns are in basis points (bp) per trade, net of the 10 bp taker round trip unless
  marked gross.
- **Statistics.** SE is the Hansen–Hodrick standard error (§6.7) and t = μ̂/SE. Holm's correction
  runs over each family of 60 cells, one-sided, at a family-wise error of 0.05.

**Sources.**
- From the Headline through "The largest primary cell", every number is a value from, or a count
  over, the committed results files:
  - `results/oos/doge_2023.csv`
  - `results/oos/robustness_doge_2023.csv`
  - `results/oos/doge_2024.csv`
  - `results/oos/majors_2023.csv`
  - and, for in-sample values, `results/insample_v2.csv`;
  - that section's last paragraph also quotes `results/oos/doge_fwd.csv`.
- The `doge_fwd` section's numbers come from `results/oos/doge_fwd.csv` and its manifest, compared
  with `doge_2023.csv` and `doge_2024.csv`; its window and warm-up from Appendix A, its record from
  git, and the earlier contact from §3.
- The maker probe's numbers come from `results/maker_probe/` and, for the candidate days,
  `results/manifest_maker_probe.json`; its record from git.
- A range of values is given to the precision of the values it spans, each end rounded to the
  nearest: "2.9–4.0" spans 2.935 to 3.980.
- The Record comes from git, the GitHub Release and Software Heritage.
- The Data section counts the downloaded series under `data/history/`. These are not committed;
  their SHA-256 hashes are in the manifests (`results/oos/manifest_*.json`).

## Record

Each set was downloaded, its manifest committed and pushed, and then run once, in this order.

| Step | Commit | Time (UTC, 2026-10-12) |
|---|---|---|
| Publication: Release `prereg-publication` | `0992805` | 07:37 (Release) |
| In-sample supplement manifest | `c0296aa` | 07:46 |
| Maker-probe candidate days | `d7b9b8d` | 07:47 |
| Freeze: `results/insample_v2.csv`, tag `prereg-freeze` | `4a4e4b0` | 07:51 |
| Software Heritage visit archiving both tagged commits | `swh:1:snp:7baf60bb2da0a53f8120a3ff56074ac410fb1b28` | 08:55 |
| Freeze amendment, `docs/preregistration-freeze.md` | `aaddee7` | 09:01 |
| `doge_2024`: manifest, then results | `4565e37`, `c7c71b6` | 09:21, 09:22 |
| `majors_2023`: manifest, then results | `4062b7a`, `6b9c548` | 10:23, 10:23 |
| `doge_2023`: manifest, then results | `00dd6ef`, `dae635b` | 10:44, 10:45 |

- **Independent timestamps.** Commit times are committer dates. The Release time and the Software
  Heritage visit come from GitHub and Software Heritage, not from the author's clock.
- **Software Heritage timing.** §10 step 1 asks for the archive at publication. The save request at
  07:40 failed, and the visit at 08:55 archived the publication and freeze commits together. That
  visit was before the freeze amendment, which `scripts/fetch_v2.py` requires on the public `main`
  before it downloads test data, so it was before every download. The independent record therefore
  shows the pre-registration existing by 08:55. Its 07:37 publication time rests on GitHub alone.
  The departure is logged in §12.
- **Why the primary set ran last.** `doge_2023`'s `replicated` and `consistent_on_majors` labels
  are read from the two secondary results. Before reading them, its run recomputed both from their
  committed inputs and checked that the committed files matched (to 10⁻⁹). That check kept no file
  and is not a rerun in the sense of §10 step 6.
- **Code.** Every run used the code of `prereg-freeze`. Each results file's `code_commit` is the
  checkout's HEAD when that set ran, which is the set's manifest commit above, not `4a4e4b0`.
  `scripts/run_oos.py` refuses to run unless every path in `git diff --name-only prereg-freeze HEAD`
  is under `docs/` or `results/` (§10 step 5). For all three runs that diff listed only
  `docs/preregistration-freeze.md` and files under `results/`.
- **No reruns.** No rerun was made.

## Headline (§7)

**No factor × bar × horizon cell showed a significantly positive per-trade return at taker cost
on `doge_2023` (Holm, family-wise error 0.05).** This does not say that the returns were zero or
negative.

- **Coverage.** All 60 cells were testable. None had a flagged direction, and every SE is
  Hansen–Hodrick (no Bartlett fallback).
- **The largest t.** It is 1.71, for 1H, h = 5, `rolling_vol_ratio` (μ̂ = +17.6, SE 10.3), where
  Holm's first step needs 3.144. Its one-sided p of 0.044 is the only p below 0.05. Every
  Holm-adjusted p is 1, and so is every BH and BY q-value.
- **Positive cells.** Three cells have a positive μ̂: the one above, and 1H, h = 15 `ou_reversion`
  and `hurst_regime` (+1.1 each, t 0.05 and 0.08).
- **Gross means.** 39 of the 60 cells have a positive gross mean. Before costs, though, only 3 have
  a gross t above 1.645, all of them `rolling_vol_ratio`, against 6 below −1.645 (R6, c = 0). Each
  factor's signal is scored on the same bars at all three horizons, so these are not 60 independent
  draws.
- **Labels.** No cell is confirmed. `replicated` and `consistent_on_majors` are therefore false
  throughout, and §7's rule for confirmed cells that are not replicated has nothing to apply to.

## Cells that do not pay at taker cost (§9)

40 of the 60 cells have ub95 < 0. ub95 = μ̂ + 1.645·SE is a one-sided 95% upper bound on the
cell's mean net return, so a negative ub95 means that cell does not pay at taker cost. They are:

- **15m, h = 1 (all 10):** `rsi_14`, `macd_hist_norm`, `bollinger_pctb`, `rolling_vol_ratio`,
  `net_taker_flow`, `basis_z_score`, `funding_cost_pressure`, `ou_reversion`, `hurst_regime`,
  `triple_screen`
- **15m, h = 5 (all 10):** the same ten
- **15m, h = 15 (7):** `rsi_14`, `macd_hist_norm`, `bollinger_pctb`, `rolling_vol_ratio`,
  `basis_z_score`, `hurst_regime`, `triple_screen`
- **1H, h = 1 (all 10):** the same ten as at 15m
- **1H, h = 5 (1):** `net_taker_flow`
- **1H, h = 15 (2):** `rolling_vol_ratio`, `net_taker_flow`

**The family bound.** ub95_family = μ̂ + 3.144·SE holds for all 60 cells at once.
- It is below zero for 34 cells.
- Six cells have ub95 < 0 but a family bound that is not below zero:
  - 15m: h = 5 `rolling_vol_ratio` and h = 15 `basis_z_score`;
  - 1H: h = 1 `rolling_vol_ratio`, h = 5 `net_taker_flow`, and h = 15 `rolling_vol_ratio` and
    `net_taker_flow`.
- At 15m, h = 1 the largest family bound is −7.6. So with 95% simultaneous confidence, every 15m
  one-bar cell has a true mean net return below −7.5 bp per trade. Their gross means lie between
  −0.7 and +0.7 bp, so almost all of that loss is the 10 bp cost.
- Over the whole family the largest family bound is +71.7 (1H, h = 15, `ou_reversion`). The data
  cannot rule out large effects at the longest horizon.

**Every other cell.** The table gives μ̂, SE, t, both bounds, and two minimum detectable effects
(MDE): the frozen MDE, from the freeze amendment, and the realised MDE, from the realised SE. All
values are in bp.

| Bar | h | Factor | d | μ̂ | SE | t | ub95 | ub95_family | MDE frozen | MDE realised |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15m | 15 | `net_taker_flow` | +1 | −6.7 | 5.6 | −1.19 | +2.5 | +10.9 | 17.5 | 22.3 |
| 15m | 15 | `funding_cost_pressure` | −1 | −4.3 | 6.1 | −0.70 | +5.7 | +14.8 | 15.1 | 24.3 |
| 15m | 15 | `ou_reversion` | −1 | −8.4 | 5.5 | −1.52 | +0.7 | +9.0 | 15.5 | 22.1 |
| 1H | 5 | `rsi_14` | +1 | −8.0 | 7.2 | −1.11 | +3.9 | +14.7 | 26.3 | 28.9 |
| 1H | 5 | `macd_hist_norm` | +1 | −2.4 | 7.2 | −0.33 | +9.5 | +20.3 | 31.7 | 28.8 |
| 1H | 5 | `bollinger_pctb` | +1 | −6.7 | 6.3 | −1.07 | +3.6 | +13.0 | 24.4 | 24.9 |
| 1H | 5 | `rolling_vol_ratio` | +1 | +17.6 | 10.3 | +1.71 | +34.5 | +50.0 | 26.7 | 41.0 |
| 1H | 5 | `basis_z_score` | −1 | −7.3 | 6.5 | −1.13 | +3.3 | +13.0 | 16.0 | 25.8 |
| 1H | 5 | `funding_cost_pressure` | −1 | −5.2 | 7.9 | −0.65 | +7.9 | +19.8 | 20.0 | 31.7 |
| 1H | 5 | `ou_reversion` | +1 | −5.0 | 8.1 | −0.61 | +8.4 | +20.6 | 22.9 | 32.4 |
| 1H | 5 | `hurst_regime` | −1 | −10.7 | 7.0 | −1.53 | +0.8 | +11.3 | 16.7 | 27.8 |
| 1H | 5 | `triple_screen` | −1 | −6.7 | 6.1 | −1.09 | +3.4 | +12.5 | 19.3 | 24.4 |
| 1H | 15 | `rsi_14` | +1 | −8.5 | 18.2 | −0.47 | +21.4 | +48.6 | 67.7 | 72.4 |
| 1H | 15 | `macd_hist_norm` | +1 | −9.1 | 14.5 | −0.63 | +14.7 | +36.4 | 62.0 | 57.6 |
| 1H | 15 | `bollinger_pctb` | +1 | −10.8 | 14.7 | −0.73 | +13.4 | +35.5 | 50.7 | 58.7 |
| 1H | 15 | `basis_z_score` | −1 | −14.8 | 18.0 | −0.82 | +14.8 | +41.8 | 35.1 | 71.7 |
| 1H | 15 | `funding_cost_pressure` | −1 | −12.7 | 23.0 | −0.55 | +25.2 | +59.7 | 48.6 | 91.8 |
| 1H | 15 | `ou_reversion` | +1 | +1.1 | 22.4 | +0.05 | +38.0 | +71.7 | 69.1 | 89.5 |
| 1H | 15 | `hurst_regime` | −1 | +1.1 | 13.1 | +0.08 | +22.6 | +42.3 | 34.3 | 52.2 |
| 1H | 15 | `triple_screen` | −1 | −4.9 | 16.0 | −0.31 | +21.4 | +45.3 | 46.4 | 63.7 |

The 40 cells with ub95 < 0 are listed with all their columns in `results/oos/doge_2023.csv`.

## Power (§9)

The freeze amendment's MDEs were set by scaling each cell's in-sample SE to the primary year. In 58
of the 60 cells the realised SE was larger, so the test was less sensitive than planned. The two
exceptions are `macd_hist_norm` at 1H, h = 5 and h = 15. Across the 60 cells, the ratio of realised
to frozen MDE has a median of 1.44 and runs from 0.91 to 2.37.

| Bar | h | Median MDE frozen | Median MDE realised | Median ratio |
|---|---:|---:|---:|---:|
| 15m | 1 | 1.3 | 2.0 | 1.52 |
| 15m | 5 | 5.4 | 7.8 | 1.46 |
| 15m | 15 | 14.8 | 17.4 | 1.32 |
| 1H | 1 | 5.1 | 7.3 | 1.50 |
| 1H | 5 | 23.6 | 28.8 | 1.34 |
| 1H | 15 | 56.3 | 72.1 | 1.33 |

An MDE is the net edge per trade that the first Holm step detects with 80% power, with every other
cell null.
- **15m, h = 1.** The test had an 80% chance of detecting a net edge of about 2 bp per trade
  (12 bp gross) in any one cell; the realised MDEs run from 1.7 to 2.5.
- **1H, h = 15.** The realised MDEs run from 52 to 123 bp, with a median of 72. A cell at the
  median had about a 35% chance of detecting a 50 bp net edge.

At long horizons a null result says little.

## Secondary families (§7)

Only F_primary confirms. The secondary families can only label a confirmed cell, so with none
confirmed they decide nothing here.

- **F_doge2024.**
  - All 60 cells are testable, and Holm rejects none.
  - The largest t is 2.23, for 1H, h = 15, `rsi_14` (μ̂ = +58.4, SE 26.2, one-sided p 0.013,
    Holm-adjusted p 0.77).
  - One other cell has p < 0.05: 1H, h = 15, `bollinger_pctb` (μ̂ = +38.2, t 1.92).
  - Both factors, with the same parameters, were DOGE inputs in the author's earlier research,
    which covered about 77% of `doge_2024` (§3). For them that set is not independent of that
    research (§7).
  - 8 of the 60 cells have a positive μ̂ and 26 a positive gross mean.
- **F_majors.** Here each cell is pooled over BTC, ETH and SOL.
  - All 60 pooled cells are testable, and Holm rejects none.
  - The largest t is 0.08, for 1H, h = 15, `hurst_regime` (μ̂ = +0.5).
  - 2 of the 60 pooled cells have a positive μ̂.
  - The per-instrument rows are descriptive. A positive μ̂ appears in 1 of 60 cells for BTC, 1 for
    ETH and 5 for SOL.
  - `majors_2023` shares the primary window, and DOGE moves with the majors. F_majors's
    t-statistics are therefore correlated with F_primary's, and its label would describe breadth
    across instruments in the same year, not independent confirmation.
- **Descriptive q-values.**
  - On `doge_2024`, the smallest BH q is 0.77 (1H, h = 15, `rsi_14`) and every BY q is 1.
  - On `majors_2023`, every pooled BH and BY q is 1.

## Robustness (§8: reported, never decisive)

All rows are on `doge_2023`.

- **R1, circular-shift placebo.**
  - Eight cells have a per-cell placebo p ≤ 0.05.
  - The family-wise placebo p is at least 0.099 for every cell. The smallest is for 1H, h = 5,
    `rolling_vol_ratio`, and four of the eight are `rolling_vol_ratio` cells.
  - Five of the eight are 15m, h = 1 cells, with t between −17 and −21, that lose 9.3 to 10.4 bp
    per trade. For them t is close to −10 bp divided by the SE. A t less negative than a shifted
    copy's therefore mostly reflects a larger SE, not a better-aligned signal. Their gross means lie
    within 0.8 bp of zero, and `rsi_14`'s is negative in the traded direction.
  - R1 tests association, not profit.
- **R2, v1's statistic.** This is the gross top-minus-bottom two-sample t over full-window quintile
  edges, divided by √h and not multiplied by d. No cell reaches 3.144 in absolute value; the
  largest is 2.00.
- **R3, R4, R7.**
  - The largest t in any quarter is 1.81 (R3).
  - After dropping the 5 or 10 most volatile days it is 1.38 (R4).
  - With winsorised returns it is 1.51 (R7).
- **R5, turnover-aware P&L.**
  - R5 is positive for 13 of the 60 cells. The median is −13.4 bp per day.
  - The four largest values are one signal, `rolling_vol_ratio`: +45.3 and +44.2 bp per day at 1H
    (h = 1 and 5), and +22.9 and +17.2 at 15m (h = 5 and 1).
  - That signal is persistent. Its mean run is 55 bars at 1H and 44 at 15m (RL). A position that
    trades only when it changes therefore pays far less in costs than the per-trade statistic,
    which charges 10 bp on every active bar. R5 is accordingly close to the gross P&L.
  - Before costs, those two 1H cells' per-trade gross t's are 2.55 and 2.68 (R6, c = 0), below
    3.144.
  - At h = 15 the in-sample direction is −1, so the same signal is traded the other way. Its R5
    there is −43.6 (1H) and −35.2 (15m) bp per day.
  - R5 has no standard error. With 55-bar runs, the year holds only 54 runs at 1H.
- **R6, cost sensitivity.**
  - No cell reaches t ≥ 3.144 at any cost, gross included.
  - Gross (c = 0), 39 cells are positive and t runs from −2.49 to +2.68: in either direction, no
    cell's |t| exceeds 2.68.
  - At c = 4 bp (maker on both sides at §11's fees, which §11 calls unverified), 10 cells are
    positive and the largest t is 2.29.
  - That row assumes every entry and exit rests passively and fills at the bar's close, with no
    missed fills and no adverse selection. The §11 probe, which measures both, is reported below.
- **R8, funding included.**
  - R8 is positive for 2 cells: 1H, h = 5 `rolling_vol_ratio` (+22.4, t 1.72) and 1H, h = 15
    `hurst_regime` (+24.0, t 0.98).
  - R8 is not μ̂ with funding added. It is averaged only over trades whose funding record is
    complete. The late funding stamps described under *Data* leave out every trade that spans a
    settlement up to and including 2024-01-17 08:00.
  - R8 and μ̂ therefore differ partly because they average different trades. For example,
    `hurst_regime`'s +24.0 comes from 1,237 of its 2,612 trades, while its μ̂ over all of them is
    +1.1.

  | Bar | h | Share of trades with R8 defined, median (range) |
  |---|---:|---|
  | 15m | 1 | 99% (98–99%) |
  | 15m | 5 | 94% (92–95%) |
  | 15m | 15 | 84% (78–86%) |
  | 1H | 1 | 96% (93–96%) |
  | 1H | 5 | 80% (67–82%) |
  | 1H | 15 | 68% (47–70%) |

- **R9, information coefficient.** The IC is the Spearman rank correlation of factor and forward
  return over the whole window, not multiplied by d. Seven cells have an IC that is significant in
  its own Holm family (two-sided, 0.05):

  | Bar | h | Factor | d | IC | IC t | Gross (bp) |
  |---|---:|---|---:|---:|---:|---:|
  | 15m | 1 | `rsi_14` | +1 | −0.039 | −6.97 | −0.4 |
  | 15m | 1 | `macd_hist_norm` | −1 | −0.025 | −4.24 | +0.3 |
  | 15m | 1 | `bollinger_pctb` | +1 | −0.041 | −7.46 | −0.7 |
  | 15m | 1 | `hurst_regime` | +1 | +0.020 | +3.66 | −0.4 |
  | 15m | 5 | `rsi_14` | +1 | −0.045 | −4.25 | −0.6 |
  | 15m | 5 | `bollinger_pctb` | +1 | −0.045 | −4.46 | −2.0 |
  | 1H | 1 | `rsi_14` | +1 | −0.037 | −3.33 | +0.1 |

  - For five of them (`rsi_14` and `bollinger_pctb`), the IC's sign is opposite to the in-sample
    direction: in 2023–24, high readings were followed by lower returns.
  - The factors predicted, but not by enough to pay. Every gross mean in the table is within 2 bp
    of zero, so even the opposite direction would not have covered the 10 bp cost.
- **R10, monotonicity.** No cell's five quintile means rise monotonically in the traded direction.
- **RL, run length.** §7 needs it only for confirmed cells. It is written for all 60 and runs from
  1.6 to 55.0 bars.

## The largest primary cell, descriptively

1H, h = 5, `rolling_vol_ratio` is the only cell with p < 0.05 on `doge_2023`, so readers will ask
about it. It is not confirmed: its Holm-adjusted p is 1, and its family-wise placebo p of 0.099 is
the smallest of any cell (R1). None of what follows is a test.

- **On the other sets.**
  - On `doge_2024`, μ̂ = +14.3 (t 1.46). `rolling_vol_ratio` was, loosely, a DOGE input in the
    author's earlier research, which covered about 77% of `doge_2024`. That value is therefore not
    independent of that research (§3, §7).
  - On `majors_2023`, the pooled μ̂ is −6.1 (t −1.05): −8.5 for BTC, −6.5 for ETH and −3.3 for
    SOL.
- **Within `doge_2023`.**
  - By quarter (R3), μ̂ is +3.3, +22.7, +26.6 and +37.0, with t 0.32, 0.98, 1.33 and 1.04, on
    1,083, 1,057, 575 and 256 trades.
  - Without the 5 or 10 most volatile days (R4), μ̂ is +11.9 (t 1.32) and +11.8 (t 1.38), against
    +17.6 with them. Dropping the 119 of 2,971 trades that touch the 5 days lowers μ̂ by a third.
  - With winsorised returns (R7), μ̂ is +12.7 (t 1.51).
- **The same factor at h = 15.**
  - The in-sample rule (§6.5) set d = +1 for `rolling_vol_ratio` at h = 1 and 5, and d = −1 at
    h = 15, on both bars. All six in-sample means that set these directions had t between −0.03 and +0.31.
  - Traded with d = −1, 1H, h = 15 has the lowest μ̂ in the family, −87.1 (t −2.82), and 15m,
    h = 15 has −22.8 (t −3.22). On `doge_2024` they are −71.7 and −27.0.
  - The factor's IC is positive in all six of its cells, though none is significant in R9's Holm
    family. In sign, then, high readings were followed by higher returns at every horizon.
  - Changing a direction after seeing test data is outside §6.5, so this is description only.
- **Caveat.** §7 calls the Hansen–Hodrick test liberal for persistent signals in trending periods.
  At a mean run of 55 bars, this is the most persistent signal in the study (RL).

`doge_fwd` reports this cell's μ̂ and 95% interval along with every other cell's, and decides
nothing (§7): +5.9 bp, from −13.6 to +25.3 (next section).

## `doge_fwd`, descriptively (§4.1, §7)

`doge_fwd` is DOGE from 2026-09-23 20:00 to 2026-11-23 20:00 UTC, the two months after the in-sample
year. §7 makes it descriptive: each cell reports μ̂ and a two-sided 95% interval, μ̂ ± 1.960·SE, and
nothing is decided.

| Step | Commit | Time (UTC, 2026-11-24) |
|---|---|---|
| `doge_fwd`: manifest, then results | `7a8e5c6`, `70c2596` | 08:53, 08:54 |

- **Window.** `scripts/fetch_v2.py` refuses to download a protected window that has not closed by
  OKX's clock, and `scripts/run_oos.py` checks again that it has (guard 6).
- **Code.** The run used the code of `prereg-freeze`. At its HEAD, `git diff --name-only
  prereg-freeze HEAD` listed `docs/preregistration.md` (a §12 row), `docs/preregistration-freeze.md`,
  this file, and files under `results/`.
- **Warm-up.** The warm-up, from 2026-06-25 20:00, is read from the in-sample files. The run
  matched their hashes against Appendix A and the in-sample supplement's manifest.
- **Data.** Both candle series, the perpetual and its spot leg, have 87,840 rows, one per minute of
  the window, with no gaps. Funding has 183 settlements, one in each 8-hour slot, all stamped on
  the grid.
- **Earlier contact.** §3 lists two looks inside this window during planning: a three-minute poll
  of open interest, and the maker-fill sketch, whose archive files cover 2026-10-03 16:00 →
  2026-10-05 00:00 UTC. No factor was computed on either. That stretch is flagged here, as §3
  says; the cells include it unchanged.

**Results.**
- No cell's interval lies above zero, and 35 of the 60 lie entirely below it.
- Six cells have μ̂ > 0, all on 1H bars at h = 5 or 15. Each interval is at least 36 bp wide. The
  largest μ̂ is +17.1 (1H, h = 15, `basis_z_score`), from −20.3 to +54.4.
- The ten 15m, h = 1 cells again lose about the cost. Their gross means lie between −1.7 and
  +0.4 bp and their net means between −11.7 and −9.6, and every interval ends below −8.0.
- The largest primary cell, 1H, h = 5, `rolling_vol_ratio`, has μ̂ = +5.9, against +17.6 on
  `doge_2023` and +14.3 on `doge_2024`.
- In 55 of the 60 cells, μ̂ has the same sign as on `doge_2023`, mostly because both are negative.
  The rank correlation of the two sets' μ̂ across cells is 0.21.

**Reading.** Two months give wide intervals at the longer horizons, so `doge_fwd` can neither
confirm nor rule out the edges the primary year left open there. Where the data are dense, at
short horizons, it agrees with the primary result: the cells lose the cost.

## Data

**Completeness.** Every candle series is complete: it has no gaps and one row per minute of its
window. Every test-set funding series has a settlement in each 8-hour slot: 1,368 for `doge_2023`
and for each majors instrument, and 1,365 for `doge_2024`.

**Funding timestamps off the 8-hour grid.** OKX's funding archive stamps some settlements late, by
1 to 24 s after the grid time (DOGE 3–23 s, SOL 3–24 s, ETH 1–3 s). This happens only up to and
including 2024-01-17 08:00 UTC; from 2024-01-17 16:00 every settlement falls exactly on the grid.

- **Counts.**
  - For DOGE and SOL every settlement up to the cut-over is late: 617 of the 1,368 in each file,
    warm-up included, and 347 of the 1,098 in the evaluation window.
  - For ETH, 199 of those 617 are late (142 of the 347 in the window), and the rest are on the
    grid.
  - BTC, `doge_2024` and the in-sample series have none.
- **Spacings (§4.2).** §4.2 requires spacings other than 8 h to be reported. The late stamps make
  345 of DOGE's spacings differ from 8 h, 359 of SOL's and 270 of ETH's. They differ by at most
  20 s, and ETH's by at most 3 s. Each run logged them, and the manifests list them as
  `spacing_other_than_expected`.
- **Manifest `gaps`.** The manifests' funding `gaps` (175 for DOGE, 134 for ETH, 188 for SOL) are
  the subset of those spacings that exceed 8 h. No 8-hour slot is missing a settlement.

The frozen code takes each settlement at the time it was stamped; it does not move a late stamp to
its grid time. Two things follow:

- **R8.** R8 counts an expected settlement time as recorded only if a settlement is stamped exactly
  on it, though late-stamped rates are still charged. An expected settlement time with no on-grid
  record therefore makes R8 undefined for every trade spanning it. This is why R8 covers fewer
  trades above, and why every trade it leaves out falls in about the first four months of the
  primary year.
- **`funding_cost_pressure`.**
  - Through the as-of join on bar start and §4.2's fixed 8-hour staleness limit, the factor takes
    the latest settlement at or before each bar's start. On the bar that starts at a late
    settlement's grid time, that is the previous settlement.
  - For DOGE and SOL, whose previous settlements were also late, the bar takes the previous value:
    347 bars of the evaluation window on each bar grid. The same happens in the warm-up, which feeds
    the expanding quintile edges.
  - For ETH, where the previous settlement is often on the grid, the staleness limit makes the bar
    undefined instead (82 of its 142 affected evaluation bars; the other 60 take the previous
    value).
  - Every `funding_cost_pressure` cell on `doge_2023` has t ≤ −0.55, against the 3.144 that
    confirmation needs.

**Not run.** A reading that moves each late stamp to its grid time was not run. It would need code
other than `prereg-freeze`'s, so it would be a deviation logged in §12. It would be run into
`results/oos/rerun-N/` and reported alongside the committed runs, which stay primary (§10 steps
5–6). None has been made.

## Maker probe (§11, exploratory)

§11 asks whether resting at the touch beats crossing the spread once adverse selection is counted.
Its output informs the discussion of cost only; it never replaces c = 10 bp. The numbers below are
from `results/maker_probe/summary.csv`, `daily.csv` and `checks.csv`, and the archive files are
listed with their SHA-256 in `results/maker_probe/manifest_archives.json`.

| Step | Commit | Time (UTC, 2026-11-24) |
|---|---|---|
| Engine, tests and download script; archive parsers left as stubs | `461ae82` | 09:02 |
| Archive files downloaded (56 files, 5.2 GB), then recorded | `aee7856` | 09:03–09:16, 09:22 |
| Parsers and run script, with three §12 rows on the files' format | `8f5ad83` | 09:22 |
| Trades marked `source` 1 left out, with a §12 row | `b526d26` | 09:26 |
| Results, both runs | `99e20f0` | 09:35 |
| Snapshot cadence corrected in §12 and in the parser's notes | `34be6d4` | 09:37 |
| Readers' modes: `fetch_maker_probe.py` checks the listed files, `run_maker_probe.py --verify` | `14b245d` | 09:43 |

- **Order.** The engine's rules (fill models, timing, skips, the snapshot check) were committed and
  pushed before any archive file was downloaded, and the download began only after `doge_fwd` had
  run, as §12 (2026-10-12) requires. The parsers depend on the files' format, so they came after
  the download. Every §12 row on the format was committed before the run. After it, `34be6d4`
  corrected what one row said about the snapshot cadence (every minute on five of the 14 days, not
  on all), which the run had checked either way, and changed no result; `14b245d` added the
  readers' modes, which compute nothing new.
- **One execution.** A first attempt (§12's "first run") stopped on its second day, at the first
  trade file with a `source` column, and wrote nothing. The execution recorded above is the only
  one that produced output, and it computed both runs together: `parser_gaps` treats only missing data as gaps, and
  `touch_gaps` also drops the interval since a side's previous snapshot wherever that snapshot's
  best level differs from the rebuilt book (§12, 2026-11-24).
- **As planned.** The 14 days are the freeze amendment's, drawn from 364 candidates. On each day,
  288 decisions 5 minutes apart post one contract at the best bid and, separately, at the best
  ask, each for 60 seconds. M1, the lower bound, fills only on a trade strictly through the order's
  price or an opposite best that reaches it. M2, the upper bound, also fills when a trade at the
  price reaches the order after the queue ahead, starting at the displayed size, has been used up
  by trades at the price and by size decreases at the level. Fees are maker 2 bp and taker 5 bp,
  unverified.

**Data.** Each day was checked before it was simulated (`checks.csv`).
- **Complete.** No book paused for 5 s or more (the longest pause was 1.6 s), no trade id jumped,
  and no timestamp ran backwards, so neither run has a gap from missing data. Each day's 00:00
  decision comes before the day's first snapshot, stamped a few ms later, and is skipped: 14 of
  the 4,032 decisions per side.
- **Depth.** The rebuilt book never holds more than 400 levels a side, so the files delete levels
  that leave the top 400.
- **Clocks and units.** On every day, 98–99% of trades are followed within 100 ms by a decrease of
  the level they hit. The median lag is 5–6 ms and 95–98% come within 10 ms, which fits book
  updates batched every 10 ms. For trades alone at their price in their millisecond, the median
  ratio of the decrease to the trade's size is 1.0 on every day: both files count 0.01 contract
  as one unit.
- **Snapshots.** 2,531 of the 16,210 snapshot sides (15.6%) differ from the rebuilt book at the
  touch, 145 of them in price and the rest in size only. `touch_gaps` therefore drops 62 of the
  336 hours, and 1,065 of the 4,018 decisions per side.
- **Trades left out.** 1,790 trades marked `source` 1, all in the input of the day 2026-09-11.

**Results.** Means over the 14 days, each day weighted equally, with the standard error across days
in brackets. Markouts are gross, from the fill price to the mid k minutes later, positive when the
mid moved in the order's favour. Shortfalls are measured from the decision-time mid, fees
included. Crossing at once costs 5.48 bp (SE 0.04) on either side in both runs: the 5 bp taker fee
plus half the spread. "Difference" is posting, then crossing after 60 s if unfilled, minus crossing
at once; negative means resting was cheaper.

| Run | Side | Model | Fill rate | Markout 1 min | 5 min | 15 min | Post, then cross | Difference |
|---|---|---|---|---|---|---|---|---|
| `parser_gaps` | buy | M1 | 0.78 (0.02) | −1.69 (0.21) | −1.06 (0.45) | −0.71 (1.12) | 3.72 (0.10) | −1.75 (0.08) |
| `parser_gaps` | buy | M2 | 0.87 (0.01) | −1.03 (0.21) | −0.17 (0.37) | +0.16 (1.05) | 2.94 (0.07) | −2.54 (0.07) |
| `parser_gaps` | sell | M1 | 0.76 (0.03) | −1.01 (0.21) | −1.62 (0.38) | −2.26 (1.06) | 3.98 (0.14) | −1.50 (0.13) |
| `parser_gaps` | sell | M2 | 0.86 (0.01) | −0.21 (0.19) | −0.98 (0.34) | −1.65 (1.05) | 3.11 (0.12) | −2.37 (0.12) |
| `touch_gaps` | buy | M1 | 0.77 (0.02) | −1.47 (0.24) | −0.74 (0.41) | +0.20 (0.81) | 3.74 (0.14) | −1.74 (0.13) |
| `touch_gaps` | buy | M2 | 0.87 (0.01) | −0.85 (0.21) | −0.09 (0.34) | +0.74 (0.76) | 2.94 (0.10) | −2.54 (0.10) |
| `touch_gaps` | sell | M1 | 0.75 (0.03) | −1.20 (0.24) | −1.53 (0.40) | −2.56 (0.68) | 3.97 (0.15) | −1.51 (0.14) |
| `touch_gaps` | sell | M2 | 0.86 (0.01) | −0.42 (0.24) | −0.84 (0.32) | −1.80 (0.64) | 3.12 (0.13) | −2.35 (0.13) |

**Reading.**
- **Resting fills.** Within a minute, 75–78% of orders fill under M1 and 86–87% under M2.
- **Fills are adversely selected.** A minute after a fill, the mid is on average 0.2–1.7 bp on the
  wrong side of the fill price. At 15 minutes, with standard errors of 0.6–1.1 bp, the buy side
  cannot be told from zero, while the sell side stays 1.7–2.6 bp against the order.
- **Resting is still cheaper.** Posting first and crossing after 60 s if unfilled cost 2.9–4.0 bp
  per side, against 5.48 bp for crossing at once: 1.5–2.5 bp less, with standard errors of
  0.07–0.14 bp. The two runs agree on the difference to within 0.02 bp.
- **What it means for cost.** Per round trip, posting first comes to about 6.0 (M2) to 7.7 (M1) bp
  from the mid, against 11.0 bp for two crossings. The study's c = 10 bp counts fees only, so the
  two are not like for like, and §11 rules out putting one in place of the other.
- **What it means for the result.** R6 above shows that no cell reaches its threshold at any cost,
  gross included. Cheaper execution cannot change the primary result.
- **Limits.** Fourteen days; one contract, which takes no queue position from anyone; fees not
  verified; the true fill lies between M1 and M2. The decisions are unconditional, every 5 minutes,
  while a factor trades when its signal fires, and signals may meet more or less adverse selection
  than an average minute. The 10 ms batching of the book hides what happens inside a batch.

## Not done

Nothing else is planned under the pre-registration. Any further analysis would be a deviation:
logged in §12, run apart from the committed results and reported beside them.
