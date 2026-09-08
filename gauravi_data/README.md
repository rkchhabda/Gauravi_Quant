# Gauravi NSE Factor Dataset

Point-in-time NIFTY-100 factor data with **delivery percentage** — a field absent
from yfinance and every other mainstream free source — plus adjusted prices and
point-in-time fundamentals, packaged as parquet with a thin loader.

100 symbols · 5 years daily · 94% delivery coverage · versioned + checksummed.

## Install

```bash
pip install pandas pyarrow
```

Then unzip the release and point the loader at it:

```bash
export GAURAVI_DATA_DIR=/path/to/gauravi-data
```

(The loader also finds `./data` or the newest `./dist/<version>/` automatically.)

## Quickstart

```python
import gauravi_data as gd

gd.manifest()["version"]          # '2026.08'
gd.universe()[:3]                 # ['ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS']

panel = gd.load_panel()           # date, sym, close, high, low, volume, deliv_pct
fund = gd.load_fundamentals()     # point-in-time annual fundamentals
factors = gd.load_factors()       # factor values + rank_* percentiles + fwd_excess

# filter at load time
tcs = gd.load_panel(symbols="TCS.NS", start="2025-01-01")

# long -> wide
closes = gd.wide(panel, "close")  # DataFrame [date x symbol]

# mean cross-sectional IC of a factor against 30-day forward excess return
gd.factor_ic(factors, "rank_rev")
```

Full column-by-column reference: **`DATA_DICTIONARY.md`**. Coverage per column:
**`coverage.csv`**. Build provenance and SHA-256 checksums: **`MANIFEST.json`**.

## What makes this dataset worth paying for

| Problem | How this dataset solves it |
|---|---|
| NSE raw prices are **unadjusted** — splits silently destroy your returns | `close/high/low` are yfinance-adjusted; delivery % is taken from NSE separately |
| Delivery % is **not in yfinance at all**, and NSE endpoints rate-limit on bulk history | Pre-fetched, date-aligned delivery panel at 94% coverage |
| Fundamentals leak lookahead when joined naively | `avail_date = fy_end + 90 days`, validated `avail_date > fy_end` on every build |
| Silent data bugs | Five validation gates run on every release; the build fails rather than ships |

## Read this before you backtest

1. **Use `close`, not NSE raw prices, for returns.** Driving returns off unadjusted
   prices produced an 11% hit-rate in our own testing — that is what the bug looks like.
2. **Join fundamentals on `avail_date`, never on `fy_end`.**
3. **Fundamentals are annual.** Free quarterly history for NSE names is too short.
4. **The universe is *current* NIFTY-100 membership** — not point-in-time, so
   backtests carry survivorship bias. Point-in-time membership is on the roadmap.
5. **`fwd_excess` is a forward-looking research target.** Never use it as a feature.
6. **No factor here is statistically significant.** Best measured directional
   hit-rate is ~53% (SE ~2pp) versus a ~51% baseline, from `score_mr`
   (momentum + reversal) at a 30-day horizon. Delivery-based factors did **not**
   beat it once a date-parsing bug in the delivery source was fixed.

This is **data for research. Not investment advice, not a signal service, not a
recommendation to buy or sell anything.**

## Licence

Per-seat. No redistribution or resale of the data, in whole or in part. See
`LICENSE.txt`.
