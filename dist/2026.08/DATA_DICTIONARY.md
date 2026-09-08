# Data Dictionary -- Gauravi NSE Factor Dataset `2026.08`

Built 2026-08-29 00:18 UTC | 100 symbols | 2021-08-30 to 2026-08-27

| Column | Type | Meaning |
|---|---|---|
| `date` | datetime | Trading date (NSE calendar). |
| `sym` | string | NSE symbol with .NS suffix. |
| `close/high/low` | float | **Split/dividend ADJUSTED** price (yfinance). Use these for returns. |
| `volume` | float | Traded quantity. |
| `deliv_pct` | float 0-100 | **Delivery percentage** (NSE via jugaad-data). Share of traded volume actually delivered. Not available from yfinance. |
| `ret_20 / rev` | float | 20-day return; `rev` is its negation (short-term reversal). |
| `mom_12_1` | float | 12-month momentum skipping the last month. |
| `liq` | float | 20-day mean of volume x close (traded value). |
| `deliv_chg20` | float | deliv_pct minus its own 20-day mean. |
| `deliv_x_ret` | float | deliv_pct x ret_20 (delivery-conditioned reversal). |
| `eps/roe/earn_growth/profit_margin` | float | Annual fundamentals, **point-in-time**: only visible after `avail_date`. |
| `earn_yield` | float | eps / close (value factor, varies daily). |
| `avail_date` | datetime | fy_end + 90 days. The date the figure is treated as public. Never join on fy_end. |
| `rank_*` | float 0-1 | Cross-sectional percentile rank of the factor within that date. |
| `score_mr` | float 0-1 | Reference composite: mean of rank_mom_12_1 and rank_rev. Best-measured model in our own testing (~53% hit-rate at H=30d, SE ~2pp). Not a recommendation. |
| `score_fund3` | float 0-1 | Legacy composite: mean of rank_rev, (1 - rank_deliv_x_ret), rank_earn_yield. Kept for reproducibility only -- it scored WORSE than score_mr after a delivery-data date bug was fixed. |
| `fwd_excess` | float | 30-day forward return minus benchmark. **Target variable -- for research only, not tradable.** |

## Caveats you must read

1. **Prices are adjusted; delivery data is not price-derived.** `close/high/low`
   come from yfinance with `auto_adjust=True`. NSE/jugaad raw prices are
   **unadjusted** for splits and dividends -- driving returns off raw NSE prices
   produces nonsense (we measured an 11% hit-rate from exactly this bug).
2. **Fundamentals are point-in-time with a 90-day reporting lag.** Join on
   `avail_date`, never on `fy_end`, or you introduce lookahead.
3. **Fundamentals are ANNUAL.** Free quarterly history for NSE names goes back
   only ~5 quarters, too short for multi-year work.
4. **The universe is current NIFTY-100 membership** -- it is NOT point-in-time,
   so backtests carry survivorship bias. Point-in-time membership is on the
   roadmap.
5. **`fwd_excess` is a forward-looking research target.** It contains future
   information by construction. Never use it as a feature.
6. **No factor here is statistically significant.** Best measured directional
   hit-rate is ~53% (SE ~2pp) against a ~51% baseline, and delivery-based
   factors did **not** improve on a plain momentum+reversal model once a
   date-parsing bug in the delivery source was fixed.
   This is data for research, **not investment advice and not a signal service.**
