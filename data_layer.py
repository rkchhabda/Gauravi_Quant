"""
Data layer for the NIFTY-100 factor model (Phase 0).

Primary source: jugaad-data NSE stock history (point-in-time OHLC, VOLUME and
-- crucially -- DELIVERY QTY / DELIVERY %, which yfinance does not provide).
Results are cached to a local parquet store so NSE endpoints are hit once.

Exposes fetch_panel(...) with the SAME return signature as
pooled_model_v1.fetch_panel so it is a drop-in alternative, plus a delivery
panel used for the new delivery-based factors.

    close_panel : DataFrame [date x symbol] of CLOSE
    highs/lows/vols : {symbol: Series}
    deliv : {symbol: Series} of DELIVERY % (0-100), benchmark excluded

yfinance stays available (via pooled_model_v1.fetch_panel) purely as a
cross-check; jugaad is the source of record.

Usage:
    from data_layer import fetch_panel_jugaad
    close, highs, lows, vols, deliv = fetch_panel_jugaad(load_symbols(), years=5)
"""
import os
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join("data", "cache_jugaad")
BENCH = "^NSEI"


def _cache_path(sym):
    safe = sym.replace("^", "IDX_")
    return os.path.join(CACHE_DIR, f"{safe}.parquet")


def _to_nse(sym):
    """RELIANCE.NS -> RELIANCE (jugaad uses bare NSE symbols)."""
    return sym[:-3] if sym.endswith(".NS") else sym


def _load_cached(sym):
    p = _cache_path(sym)
    if os.path.exists(p):
        try:
            return pd.read_parquet(p)
        except Exception:
            return None
    return None


def _fetch_one(sym, start, end):
    """Return a tidy DataFrame indexed by date with columns
    close/high/low/volume/deliv_pct for one symbol, cached to parquet."""
    cached = _load_cached(sym)
    if cached is not None and not cached.empty:
        c_first, c_last = cached.index.min().date(), cached.index.max().date()
        # The cache must be BOTH recent enough and start early enough. Checking
        # only recency silently pins a symbol to whatever (possibly short)
        # window it was first fetched with.
        if c_last >= end - timedelta(days=5) and c_first <= start + timedelta(days=10):
            return cached

    try:
        if sym.startswith("^"):
            from jugaad_data.nse import index_df
            name = {"^NSEI": "NIFTY 50"}.get(sym, "NIFTY 50")
            raw = index_df(symbol=name, from_date=start, to_date=end)
            raw = raw.rename(columns={"HistoricalDate": "DATE", "DATE": "DATE",
                                      "CLOSE": "CLOSE", "Close": "CLOSE",
                                      "OPEN": "OPEN", "HIGH": "HIGH", "LOW": "LOW"})
            df = pd.DataFrame({
                "close": pd.to_numeric(raw["CLOSE"], errors="coerce"),
                "high": pd.to_numeric(raw.get("HIGH", raw["CLOSE"]), errors="coerce"),
                "low": pd.to_numeric(raw.get("LOW", raw["CLOSE"]), errors="coerce"),
                "volume": np.nan, "deliv_pct": np.nan,
            })
            df.index = pd.to_datetime(raw["DATE"])
        else:
            from jugaad_data.nse import stock_df
            raw = stock_df(symbol=_to_nse(sym), from_date=start, to_date=end,
                           series="EQ")
            df = pd.DataFrame({
                "close": pd.to_numeric(raw["CLOSE"], errors="coerce"),
                "high": pd.to_numeric(raw["HIGH"], errors="coerce"),
                "low": pd.to_numeric(raw["LOW"], errors="coerce"),
                "volume": pd.to_numeric(raw["VOLUME"], errors="coerce"),
                "deliv_pct": pd.to_numeric(raw["DELIVERY %"], errors="coerce"),
            })
            df.index = pd.to_datetime(raw["DATE"])
    except Exception as e:
        print(f"[data_layer] {sym}: fetch failed ({type(e).__name__}: {e})")
        return None

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df = _fix_weekend_dates(df)
    if df.empty:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(_cache_path(sym))
    return df


def _fix_weekend_dates(df):
    """Correct a jugaad-data date-parsing bug.

    NSE trades Mon-Fri, but jugaad returns each week's FRIDAY row labelled as
    the following SUNDAY (and never returns a Friday). Left uncorrected this
    silently drops ~20% of trading days -- one per week -- when the panel is
    aligned to a real trading calendar.

    Saturday/Sunday-labelled rows are mapped back to the preceding Friday.
    NSE has no weekend session, so any weekend label is by definition wrong.
    """
    if df.empty:
        return df
    dow = df.index.dayofweek                     # Mon=0 ... Sat=5, Sun=6
    if not (dow >= 5).any():
        return df
    shift = np.where(dow == 5, 1, np.where(dow == 6, 2, 0))
    df = df.copy()
    df.index = df.index - pd.to_timedelta(shift, unit="D")
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_panel_jugaad(symbols, years=5, benchmark=BENCH, min_bars=300):
    """Drop-in for pooled_model_v1.fetch_panel, plus a delivery-% panel.

    Returns (close_panel, highs, lows, vols, deliv). Symbols with fewer than
    `min_bars` usable closes are dropped, matching the yfinance path.
    """
    end = date.today()
    start = end - timedelta(days=int(365.25 * years) + 10)
    print(f"[jugaad] fetching {len(symbols) + 1} symbols "
          f"{start} -> {end} (cache: {CACHE_DIR}) ...")

    closes, highs, lows, vols, deliv = {}, {}, {}, {}, {}
    for sym in list(symbols) + [benchmark]:
        df = _fetch_one(sym, start, end)
        if df is None or len(df["close"].dropna()) < min_bars:
            continue
        c = df["close"].dropna()
        closes[sym] = c
        if sym != benchmark:
            highs[sym] = df["high"].reindex(c.index)
            lows[sym] = df["low"].reindex(c.index)
            vols[sym] = df["volume"].reindex(c.index)
            deliv[sym] = df["deliv_pct"].reindex(c.index)

    close_panel = pd.DataFrame(closes).sort_index()
    if close_panel.empty:
        raise RuntimeError("[jugaad] no usable data fetched")
    print(f"[jugaad] usable data for {len(closes)} symbols "
          f"({close_panel.index[0].date()} to {close_panel.index[-1].date()})")
    return close_panel, highs, lows, vols, deliv


def fetch_hybrid(symbols, period="5y", benchmark=BENCH):
    """Adjusted prices from yfinance (split/dividend-safe returns) + delivery %
    from jugaad. jugaad CLOSE is UNADJUSTED, so it must NOT drive returns; we
    take only DELIVERY % from it and align to the yfinance calendar.

    Returns (close_panel, highs, lows, vols, deliv) like fetch_panel_jugaad.
    """
    from pooled_model_v1 import fetch_panel as yf_panel
    close, highs, lows, vols = yf_panel(symbols, period=period,
                                        benchmark=benchmark)
    close.index = pd.to_datetime(close.index).normalize()
    years = {"1y": 1, "2y": 2, "5y": 5, "10y": 10}.get(period, 5)
    # Delivery history is cheap and cached, so always pull the full 5y and use a
    # low bar count -- otherwise short --period runs drop most symbols here.
    _, _, _, _, jd = fetch_panel_jugaad(symbols, years=max(years, 5),
                                        benchmark=benchmark, min_bars=100)
    deliv = {}
    for s in close.columns:
        if s == benchmark or s not in jd:
            continue
        ser = jd[s].copy()
        ser.index = pd.to_datetime(ser.index).normalize()
        deliv[s] = ser.reindex(close.index)
    print(f"[hybrid] adjusted prices (yfinance) + delivery% (jugaad) "
          f"for {len(deliv)} symbols")
    return close, highs, lows, vols, deliv


if __name__ == "__main__":
    # Smoke test + yfinance cross-check on a couple of names.
    syms = load_symbols()[:8]
    close, highs, lows, vols, deliv = fetch_panel_jugaad(syms, years=1,
                                                          min_bars=200)
    print("\nshape:", close.shape, "| delivery symbols:", len(deliv))
    s = [x for x in close.columns if x != BENCH][0]
    print(f"{s}: last close {close[s].dropna().iloc[-1]:.2f}, "
          f"last deliv% {deliv[s].dropna().iloc[-1]:.1f}")
