Phased workflow to reach 55%

Phase 0 — Data & validation foundation (no accuracy gain, but everything depends on it)

New data_layer.py: jugaad-data bhavcopy → cached parquet panel (OHLC + volume + delivery qty). yfinance kept behind the same interface for cross-check.
Reconstruct point-in-time NIFTY-100 membership (historical index constituents), so backtests only hold names that were actually in the index that week. Removes survivorship bias.
Lock a holdout window (e.g. last 12 months) — untouched until Phase 3 final scoring.
Exit gate: IC analysis re-run on survivorship-free data; honest baseline number recorded.

Phase 1 — Signal expansion (where the accuracy actually comes from)
4. Add orthogonal factors the current model lacks:

Delivery-% / delivery-volume trend (only available from bhavcopy — real informational edge, retail can't easily replicate).
Low-volatility and idiosyncratic-vol factors.
Fundamentals (earnings yield, ROE, earnings-revision) via one fundamentals source — the biggest orthogonal lift.
Drop mom_12_1 (measured IC≈0) or demote it; keep reversal + liquidity which carry the real signal.
Rank each factor by calibrated t_nonovlp IC on train data only.
Exit gate: ≥2 factors with genuine, sign-stable IC out-of-sample.

Phase 2 — Model & weighting
7. Replace the asserted 50/50 blend with a walk-forward-fitted combiner (ship pooled_model_v1.py's XGBoost/logistic with the embargo you already have, or a ridge on ranked factors).
8. Add FII/DII the correct way — as a market-regime exposure overlay (long/flat timing), not a cross-sectional factor (it's market-wide, so it can't rank stocks — that's why it did nothing before).
Exit gate: walk-forward top-quintile hit-rate on train/validation ≥ 55% with sensible error bars.

Phase 3 — Honest out-of-sample confirmation
9. Run the frozen model once on the Phase-0 holdout. Report hit-rate, AUC, Newey-West/non-overlap t-stat, net-of-cost quintile spread.
10. Only if holdout ≥ 55% do we ship. Otherwise iterate Phase 1 (more/better signal), not Phase 2 (that's overfitting).
Exit gate: holdout directional hit-rate ≥ 55%, statistically distinguishable from 50%.

The honest lever is Phase 1 (new orthogonal signal, especially delivery-% and fundamentals) — not model tuning. Tuning a price-only model that shows 0/18 significant factors will not reach 55%.

I'd start with Phase 0, step 1: build data_layer.py on jugaad-data bhavcopy with a parquet cache, behind the same interface as the current fetch_panel, and verify it against yfinance on a few tickers. Want me to build that now and run the survivorship-free IC baseline?