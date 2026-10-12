# Freeze amendment to the pre-registration

This file records the freeze of the out-of-sample study (docs/preregistration.md §10, step 3).
It changes nothing in the pre-registration. It is committed and pushed before any test-set
data is downloaded (§10, step 4).

- **Freeze commit.** The tag `prereg-freeze` points to the commit below, which adds
  `results/insample_v2.csv` and nothing else.
- **Publication commit.** `09928054908946b398752ebf88cdbafdbefb096b` (the GitHub Release).
- **Software Heritage.** Publication commit: `swh:1:rev:09928054908946b398752ebf88cdbafdbefb096b`.
  `prereg-freeze`: `swh:1:rev:4a4e4b07635361dfa132390af98ddf0af58016ba`.
- **Maker-probe days (§11).** Drawn from `results/manifest_maker_probe.json` as §11 specifies (14 of
  364 candidate days): 2025-10-19, 2025-11-02, 2025-11-07, 2025-11-12, 2025-12-07, 2025-12-18,
  2026-01-24, 2026-02-12, 2026-06-11, 2026-06-25, 2026-07-27, 2026-08-04, 2026-09-02, 2026-09-11.

The table copies the 60 directions, flags and minimum detectable effects (§6.5, §9) from
`results/insample_v2.csv` at the tag, as text. An MDE is `undefined` where the in-sample cell had no
active bars.

```maker-probe-days
2025-10-19
2025-11-02
2025-11-07
2025-11-12
2025-12-07
2025-12-18
2026-01-24
2026-02-12
2026-06-11
2026-06-25
2026-07-27
2026-08-04
2026-09-02
2026-09-11
```

```freeze-commit
4a4e4b07635361dfa132390af98ddf0af58016ba
```

```csv freeze-directions
bar,horizon,factor,direction,flagged,flag_reason,mde_bps
15m,1,rsi_14,1,false,,1.3909291746396342
15m,1,macd_hist_norm,-1,false,,1.6652538446557232
15m,1,bollinger_pctb,1,false,,1.3546085952631133
15m,1,rolling_vol_ratio,1,false,,1.4152499408416455
15m,1,net_taker_flow,1,false,,1.3240720700958861
15m,1,basis_z_score,-1,false,,1.216576642928787
15m,1,funding_cost_pressure,-1,false,,1.1584427673434754
15m,1,ou_reversion,-1,false,,1.2314162695630775
15m,1,hurst_regime,1,false,,1.3979378138355083
15m,1,triple_screen,-1,false,,1.122975703929582
15m,5,rsi_14,1,false,,5.896393281644412
15m,5,macd_hist_norm,-1,false,,6.787712270895472
15m,5,bollinger_pctb,1,false,,5.408335519024207
15m,5,rolling_vol_ratio,1,false,,6.526536416907964
15m,5,net_taker_flow,1,false,,5.982588700559239
15m,5,basis_z_score,-1,false,,4.355276074809487
15m,5,funding_cost_pressure,-1,false,,5.321645112864357
15m,5,ou_reversion,-1,false,,5.447498472508094
15m,5,hurst_regime,1,false,,4.92355123513936
15m,5,triple_screen,1,false,,4.686952068361249
15m,15,rsi_14,1,false,,16.43551436188675
15m,15,macd_hist_norm,1,false,,14.462899407980105
15m,15,bollinger_pctb,1,false,,12.874296611166736
15m,15,rolling_vol_ratio,-1,false,,19.233341352938858
15m,15,net_taker_flow,1,false,,17.52888551736519
15m,15,basis_z_score,-1,false,,11.428022420812916
15m,15,funding_cost_pressure,-1,false,,15.059496414977172
15m,15,ou_reversion,-1,false,,15.457050356622092
15m,15,hurst_regime,-1,false,,10.13255896105815
15m,15,triple_screen,-1,false,,11.856338828818485
1H,1,rsi_14,1,false,,5.543228818233713
1H,1,macd_hist_norm,1,false,,6.676079162272466
1H,1,bollinger_pctb,1,false,,5.423456348364361
1H,1,rolling_vol_ratio,1,false,,5.642137930837067
1H,1,net_taker_flow,-1,false,,5.784523013748992
1H,1,basis_z_score,1,false,,4.6396759824814335
1H,1,funding_cost_pressure,-1,false,,4.388987197637321
1H,1,ou_reversion,1,false,,4.73598145884007
1H,1,hurst_regime,1,false,,4.039297728134281
1H,1,triple_screen,-1,false,,4.307853122483085
1H,5,rsi_14,1,false,,26.321874731385638
1H,5,macd_hist_norm,1,false,,31.701754197747434
1H,5,bollinger_pctb,1,false,,24.360964271093174
1H,5,rolling_vol_ratio,1,false,,26.746939011727402
1H,5,net_taker_flow,-1,false,,27.45114715256928
1H,5,basis_z_score,-1,false,,15.980069077515253
1H,5,funding_cost_pressure,-1,false,,20.001901696861
1H,5,ou_reversion,1,false,,22.877173300455276
1H,5,hurst_regime,-1,false,,16.67338912056509
1H,5,triple_screen,-1,false,,19.299793148911867
1H,15,rsi_14,1,false,,67.6928457919975
1H,15,macd_hist_norm,1,false,,61.97509864328789
1H,15,bollinger_pctb,1,false,,50.68606963853065
1H,15,rolling_vol_ratio,-1,false,,75.16982530878367
1H,15,net_taker_flow,-1,false,,73.29725993120067
1H,15,basis_z_score,-1,false,,35.07778352096869
1H,15,funding_cost_pressure,-1,false,,48.58657290118521
1H,15,ou_reversion,1,false,,69.13290626407603
1H,15,hurst_regime,-1,false,,34.343254580274674
1H,15,triple_screen,-1,false,,46.412285433311624
```
