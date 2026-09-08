"""Phase 1: point-in-time fundamentals factor layer.

Source: yfinance ANNUAL statements (5 fiscal years available for NSE names --
quarterly only goes back ~5 quarters, too short for a 5y backtest).

Point-in-time discipline: a fiscal-year figure is treated as UNKNOWN until
`REPORT_LAG_DAYS` after the fiscal year end (Indian annual results land ~60d
out; 90d is the conservative choice). Merged onto the daily panel with
merge_asof on that availability date, so no lookahead.

Factors produced (daily, cross-sectional):
    earn_yield   = diluted EPS / price      (value; varies daily via price)
    roe          = net income / stockholders equity
    earn_growth  = YoY net income growth
    profit_margin= net income / total revenue

Usage:
    from fundamentals import fundamentals_panel
    fund = fundamentals_panel(symbols)   # tidy df: sym, avail_date, factors
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = os.path.join("data", "fundamentals.parquet")
REPORT_LAG_DAYS = 90


def _row(df, *names):
    """First matching row from a yfinance statement, as a Series by period."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            return pd.to_numeric(df.loc[n], errors="coerce")
    return None


def _fetch_one(sym):
    import yfinance as yf
    try:
        t = yf.Ticker(sym)
        inc, bs = t.income_stmt, t.balance_sheet
        ni = _row(inc, "Net Income Common Stockholders",
                  "Net Income From Continuing Operation Net Minority Interest")
        eps = _row(inc, "Diluted EPS", "Basic EPS")
        rev = _row(inc, "Total Revenue", "Operating Revenue")
        eq = _row(bs, "Stockholders Equity", "Common Stock Equity",
                  "Total Equity Gross Minority Interest")
        if ni is None or eq is None:
            return None
        periods = sorted(set(ni.index) & set(eq.index))
        if not periods:
            return None
        out = []
        ni_s = ni.sort_index()
        for p in periods:
            prev = ni_s.index[ni_s.index < p]
            growth = ((ni[p] / ni_s[prev[-1]] - 1)
                      if len(prev) and ni_s[prev[-1]] not in (0, np.nan) else np.nan)
            e = eq[p]
            out.append({
                "sym": sym,
                "fy_end": pd.Timestamp(p),
                "avail_date": pd.Timestamp(p) + pd.Timedelta(days=REPORT_LAG_DAYS),
                "eps": eps[p] if eps is not None and p in eps.index else np.nan,
                "roe": ni[p] / e if e and e > 0 else np.nan,
                "earn_growth": growth,
                "profit_margin": (ni[p] / rev[p]
                                  if rev is not None and p in rev.index
                                  and rev[p] else np.nan),
            })
        return pd.DataFrame(out)
    except Exception as e:
        print(f"[fund] {sym}: {type(e).__name__}")
        return None


def fundamentals_panel(symbols, refresh=False):
    """Tidy point-in-time fundamentals. Cached to data/fundamentals.parquet."""
    if not refresh and os.path.exists(CACHE):
        df = pd.read_parquet(CACHE)
        if set(symbols) & set(df["sym"]):
            print(f"[fund] cache hit: {df['sym'].nunique()} symbols")
            return df

    import concurrent.futures as cf
    frames = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_fetch_one, symbols):
            if r is not None and not r.empty:
                frames.append(r)
    if not frames:
        raise RuntimeError("[fund] no fundamentals fetched")
    df = pd.concat(frames, ignore_index=True).sort_values(["sym", "avail_date"])
    os.makedirs("data", exist_ok=True)
    df.to_parquet(CACHE)
    print(f"[fund] fetched {df['sym'].nunique()} symbols, {len(df)} rows")
    return df


def attach_fundamentals(daily, fund):
    """As-of merge fundamentals onto a tidy daily frame (cols: date, sym).

    Uses backward merge_asof on avail_date, so each row only sees figures
    already published. Adds earn_yield = eps / close (needs a `close` column).
    """
    out = []
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
    fund = fund.copy()
    fund["avail_date"] = pd.to_datetime(fund["avail_date"]).astype("datetime64[ns]")
    for s, g in daily.groupby("sym", sort=False):
        f = fund[fund["sym"] == s].sort_values("avail_date")
        if f.empty:
            continue
        g = g.sort_values("date")
        m = pd.merge_asof(g, f.drop(columns=["sym", "fy_end"]),
                          left_on="date", right_on="avail_date",
                          direction="backward")
        out.append(m)
    res = pd.concat(out, ignore_index=True)
    if "close" in res.columns:
        res["earn_yield"] = res["eps"] / res["close"]
    return res


if __name__ == "__main__":
    from pooled_model_v1 import load_symbols
    syms = [s for s in load_symbols() if not s.startswith("^")]
    f = fundamentals_panel(syms)
    print(f.head(6).to_string(index=False))
    print("\ncoverage by factor (non-null rows):")
    print(f[["eps", "roe", "earn_growth", "profit_margin"]].notna().sum())
