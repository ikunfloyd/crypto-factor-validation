# Factor validity study — DOGE-USDT-SWAP

**Question.** Do the nine factors in this repo predict returns well enough to
pay for the fees it costs to trade them?

**Answer.** No. Across 36 factor × horizon combinations on a year of real
data, six produce a positive long-short spread after costs, none reaches
`|t| > 2`, and none survives correcting for the number of tests run. The
strategy in this repository has no demonstrated edge, and the reason is
structural rather than a shortfall of model capacity.

This document records how that was measured, because the method matters more
than the verdict.

---

## 1. Data

| | |
|---|---|
| Instrument | `DOGE-USDT-SWAP` (OKX perpetual) |
| Period | 2025-09-23 → 2026-09-23 (one year) |
| Bars | 525,599 × 1m, **no gaps** |
| Also pulled | funding rates (3 months retained), open interest (2 days retained) |
| Source | `scripts/fetch_history.py` — public REST, no API key |

Two ingest rules exist to stop look-ahead bias at the door:

**Unconfirmed bars are dropped.** OKX marks the in-progress bar with
`confirm="0"` and keeps mutating it until its minute closes. Its high, low and
close are not knowable at the time they appear. Training on one leaks the
future.

**Gaps are reported, never interpolated.** A backtest that treats a
three-hour hole as three consecutive bars books a return that never happened.
This year's series happened to be continuous; the check stays because the next
one may not be.

## 2. Method

**Entry is delayed one bar.** A factor computed from bar `t`'s close cannot be
filled at that same close — that price is gone by the time the factor exists.
Returns are measured from `close[t+1]` to `close[t+1+h]`. Skipping this single
step is the most common way a crypto backtest invents an edge.

**Costs are charged at the taker rate.** 5 bp per side, 10 bp round trip. A
factor has to beat that spread before anything else about it matters.

**IC here is time-series, not cross-sectional.** With one instrument there is
no cross-section to rank. This measures whether a factor's own level predicts
its own future return — the weaker of the two claims, and it should not be
quoted as though it were the usual cross-sectional IC.

**Overlapping windows are penalised.** For `h > 1` the forward-return windows
of consecutive bars overlap, so observations are not independent and the naive
`t` is inflated by roughly `sqrt(h)`. Every reported `t` is divided by that —
a cheap stand-in for Newey-West that errs toward calling results
insignificant.

**Thin series are excluded, not silently compared.** OKX retains only days of
open-interest history and months of funding history. Any factor × horizon with
fewer than 1,000 usable observations is skipped rather than reported next to a
factor measured on 35,000 bars.

Reproduce with:

```bash
python scripts/fetch_history.py --days 365 --with-spot
python scripts/factor_research.py --bars 15m 1H
```

The v1 fetcher is now locked for any window that touches the pre-registered test windows (`docs/preregistration.md` §4.1, §10), and it never writes into the in-sample directory or over an existing file, so this exact command is refused; post-freeze downloads use `scripts/fetch_v2.py`.

## 3. The cost problem comes first

Before any factor is tested, the return distribution already constrains what
is possible:

```
1-minute return distribution (basis points)
  std dev           12.57 bp
  |return| median    4.23 bp
  |return| p90      14.15 bp
  ────────────────────────────
  round-trip cost   10.00 bp
  bars whose full move exceeds cost:  18.5%
```

For 81.5% of minutes, capturing the **entire** bar — perfect direction, bought
at the low, sold at the high — still does not pay the fees. That is an upper
bound no model can exceed. A realistic strategy capturing 20–30% of the bar
range is off by a factor of three or more.

This alone rules out 1-minute taker trading on this instrument. Everything
below therefore tests 15m and 1H bars, where the signal grows roughly as
`sqrt(t)` while the fee stays flat.

## 4. Results

Best combinations at 15m, sorted by spread net of cost:

| factor | h | n | IC | IR | spread | net | t | mono |
|---|---|---|---|---|---|---|---|---|
| `rsi_14` | 15 | 35,011 | −0.0069 | −0.36 | 18.83 | **+8.83** | 1.70 | 0.56 |
| `macd_hist` | 15 | 34,992 | +0.0322 | 0.23 | 17.80 | **+7.80** | 1.56 | 0.92 |
| `funding_cost_pressure` | 15 | 8,641 | +0.0417 | 1.45 | 15.21 | **+5.21** | 0.98 | 0.74 |
| `bollinger_pctb` | 15 | 35,006 | +0.0082 | −0.10 | 13.54 | **+3.54** | 1.33 | 0.95 |
| `net_taker_flow` | 15 | 34,966 | −0.0178 | −0.33 | 7.01 | −2.99 | 0.68 | 0.40 |

```
36 factor x horizon combinations tested
  6 clear the 10 bp round-trip cost
  0 of those also reach |t| > 2          (highest: 1.70)
  0 survive Bonferroni for 36 tests      (|t| > 3.20)
```

Two things make the positive-looking rows worth nothing:

**The ICs are at noise level.** The largest is 0.042, and it belongs to the
factor with the thinnest coverage. For context, a cross-sectional IC around
0.03 is considered weak-but-real; these are time-series ICs, which are easier
to produce by chance, and most sit below 0.02.

**IC and spread disagree.** `rsi_14` has the best net spread (+8.83 bp) and an
IC of −0.007 — essentially zero, and the wrong sign. A spread that appears
without a matching IC is not coming from broad predictive power; it is coming
from a handful of extreme observations in the tail buckets. That is the
signature of an artefact, not an edge.

Searching 36 combinations and reporting the best few is precisely how
backtests manufacture edges. With `|t|` topping out at 1.70 against a
Bonferroni threshold of 3.20, nothing here is distinguishable from noise.

## 5. The factors overlap more than they appear to

Rank correlations, computed pairwise (each cell uses the rows where that pair
is defined):

```
                        rsi_14  macd_hi  bolling  rolling  net_tak  funding
rsi_14                    1.00     0.53     0.90    -0.01     0.39     0.03
macd_hist                 0.53     1.00     0.72     0.00     0.02    -0.05
bollinger_pctb            0.90     0.72     1.00    -0.00     0.25     0.01
rolling_vol_ratio        -0.01     0.00    -0.00     1.00     0.01     0.09
net_taker_flow            0.39     0.02     0.25     0.01     1.00    -0.02
funding_cost_pressure     0.03    -0.05     0.01     0.09    -0.02     1.00
```

`rsi_14` and `bollinger_pctb` correlate at **0.90** over 35,022 bars — two
spellings of "where is price inside its recent range". `macd_hist` and
`bollinger_pctb` reach 0.72. The nine-factor set contains roughly four
independent ideas: range position, volatility, order flow, and funding.

Feeding near-duplicates to a tree model splits the credit for one idea across
several features, inflating their apparent importance and disguising how
little the model is actually working with.

*Methodological note.* An earlier version required every factor to be present
on a row before computing any correlation. Open interest is retained for two
days, so that rule shrank the sample to 190 rows and reported
`rsi_14 ↔ net_taker_flow = +0.87`. The pairwise figure over 35,022 rows is
**+0.39**. The listwise version was not merely imprecise — it inverted the
conclusion about whether that factor was independent.

## 6. What was wrong with the factors

Four of the nine were broken before this study could run. Each is now fixed
with a regression test:

| factor | defect | consequence |
|---|---|---|
| `basis_z_score` | `spot_price = perp_price`, so basis ≡ 0 | constant 0; a dead feature occupying a slot |
| `funding_cost_pressure` | returned a series shorter than the candle frame | dropped by the length check — **never reached a model** |
| `oi_change_rate` | 5m OI paired with 1m candles **by array index** | random look-ahead: sometimes future OI, sometimes past |
| `net_taker_flow` | `sign(Δclose)`, i.e. the return's own sign | collinear with the label; a leak waiting for a label change |

The common fix is `FactorEngine._asof_backward`: every series coarser than the
candle timeline (funding at 8h, OI at 5m, the spot leg) is joined by
timestamp, taking the most recent observation *at or before* each bar. Index
pairing is now forbidden, and `tests/test_factor_engine.py::TestAsofAlignment`
asserts that no output can come from a future observation.

A factor that cannot be computed returns `None` — the engine skips it — rather
than a constant standing in for missing data.

## 7. The models were not learning from data

`quant_core/training/train_models.py` generates its labels like this:

```python
future_returns_hunter = np.random.normal(0, 0.03, n_samples)
labels['hunter'] = (future_returns_hunter > 0.02).astype(int)
```

The labels are random draws, independent of the features. `_add_patterns()`
then injects three hand-written rules (low RSI + high volume → higher win
probability, and two more) to make the data learnable.

So the three LightGBM experts can only recover those three rules; everything
else is noise. The Mixture-of-Experts, as trained, is three `if` statements
re-encoded as gradient-boosted trees.

One thing the training code does get right: the split is `TimeSeriesSplit`,
not a random `train_test_split`. On time-series data a random split trains on
the future and tests on the past.

## 8. Limitations

- **One instrument, one year.** Nothing here generalises to other symbols or
  regimes without re-running it.
- **In-sample throughout.** No walk-forward validation was performed — there
  was nothing significant enough to be worth validating.
- **Taker fees only.** The maker path, where fees are much lower, is untested
  and is the most promising remaining direction.
- **Short histories for funding and OI.** OKX retains months and days
  respectively, so those factors rest on far less evidence than the rest.
- **No spot leg in this run.** `basis_z_score` was skipped; re-run with
  `--with-spot` to include it.

## 9. Where this goes next

In order of expected payoff:

1. **Maker orders.** The fee difference is larger than every effect measured
   here. The cost of that is adverse selection — a resting order fills
   preferentially when the market is moving against it — which is itself worth
   measuring.
2. **Longer horizons.** The trend from 1m to 15m is in the right direction;
   hours rather than minutes is the untested region.
3. **Factors with economic content.** The current set is textbook technical
   indicators, whose edge on a liquid perpetual is long since arbitraged. A
   mean-reversion half-life estimated from an Ornstein-Uhlenbeck fit, or a
   Hurst exponent driving regime classification, would at least be a testable
   hypothesis rather than a repackaged moving average.
4. **Real labels.** Retrain on forward returns from this dataset instead of
   synthetic ones. This only becomes worth doing after a factor survives
   step 1–3.

Model capacity is not the bottleneck. A gross edge smaller than the fee cannot
be fixed by a larger model.
