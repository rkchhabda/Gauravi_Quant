import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta


MACRO_SYMBOLS = {
    "usd_inr": "INR=X",
    "india_vix": "^INDIAVIX",
    "nifty50": "^NSEI"
}


def fetch_fii_dii_flow(period: str = "30d") -> Dict[str, float]:
    """
    Fetch Foreign Institutional Investor (FII) and Domestic Institutional Investor (DII) flow data.
    
    Note: This is a proxy using NIFTY 50 and Bank NIFTY as indicators.
    For production, integrate with NSE API or Bloomberg for actual FII/DII data.
    
    Returns:
        Dictionary with:
        - fii_flow_proxy: Proxy for FII activity (NIFTY 50 momentum)
        - dii_flow_proxy: Proxy for DII activity (Bank NIFTY vs NIFTY ratio)
        - fii_dii_sentiment: Combined sentiment score (-1 to 1)
    """
    try:
        # Fetch NIFTY 50 and Bank NIFTY as proxies
        nifty = yf.Ticker("^NSEI")
        bank_nifty = yf.Ticker("^NSEBANK")
        
        nifty_df = nifty.history(period=period, interval="1d")
        bank_df = bank_nifty.history(period=period, interval="1d")
        
        if nifty_df.empty or bank_df.empty:
            return {"fii_flow_proxy": 0.0, "dii_flow_proxy": 0.0, "fii_dii_sentiment": 0.0}
        
        # FII proxy: NIFTY 50 10-day momentum (FIIs typically drive large-caps)
        nifty_close = nifty_df["Close"].dropna()
        if len(nifty_close) >= 10:
            fii_proxy = float((nifty_close.iloc[-1] / nifty_close.iloc[-10] - 1.0) * 100)
        else:
            fii_proxy = 0.0
        
        # DII proxy: Bank NIFTY / NIFTY 50 ratio momentum (DIIs are heavy in banking)
        bank_close = bank_df["Close"].dropna()
        if len(bank_close) >= 10 and len(nifty_close) >= 10:
            ratio = bank_close.values[-10:] / nifty_close.values[-10:]
            dii_proxy = float((ratio[-1] / ratio[0] - 1.0) * 100)
        else:
            dii_proxy = 0.0
        
        # Combined sentiment: positive = risk-on (FII buying), negative = risk-off
        sentiment = np.clip((fii_proxy * 0.6 + dii_proxy * 0.4) / 2.0, -1.0, 1.0)
        
        return {
            "fii_flow_proxy": round(fii_proxy, 4),
            "dii_flow_proxy": round(dii_proxy, 4),
            "fii_dii_sentiment": round(float(sentiment), 4)
        }
        
    except Exception as e:
        print(f"[WARN] Failed to fetch FII/DII flow data: {e}")
        return {"fii_flow_proxy": 0.0, "dii_flow_proxy": 0.0, "fii_dii_sentiment": 0.0}


def fetch_macro_data(symbols: List[str] = None, period: str = "3y") -> Dict[str, float]:
    """
    Fetch latest macro indicator values from Yahoo Finance.
    
    Args:
        symbols: List of macro symbols to fetch. Defaults to USD/INR, India VIX, NIFTY 50.
        period: Lookback period for data fetch (default: 3y for 3 years of history)
    
    Returns:
        Dictionary with latest values and derived features:
        - usd_inr: Latest USD/INR exchange rate
        - india_vix: Latest India VIX value
        - nifty50_ret14d: 14-day return of NIFTY 50
        - nifty50_ret5d: 5-day return of NIFTY 50
        - nifty50_vol20: 20-day volatility of NIFTY 50
        - fii_flow_proxy: Proxy for FII activity
        - dii_flow_proxy: Proxy for DII activity
        - fii_dii_sentiment: Combined FII/DII sentiment
    """
    if symbols is None:
        symbols = list(MACRO_SYMBOLS.values())
    
    result = {}
    
    for name, symbol in MACRO_SYMBOLS.items():
        if symbol not in symbols:
            continue
            
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d", auto_adjust=False)
            
            if df.empty:
                result[name] = 0.0
                continue
                
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            close = df["Close"].dropna()
            
            if len(close) == 0:
                result[name] = 0.0
                continue
            
            latest_close = float(close.iloc[-1])
            
            if name == "usd_inr":
                result["usd_inr"] = latest_close
            elif name == "india_vix":
                result["india_vix"] = latest_close
            elif name == "nifty50":
                result["nifty50_close"] = latest_close
                # 14-day return
                if len(close) >= 14:
                    ret14 = (close.iloc[-1] / close.iloc[-14] - 1.0) * 100
                    result["nifty50_ret14d"] = float(ret14)
                else:
                    result["nifty50_ret14d"] = 0.0
                # 5-day return
                if len(close) >= 5:
                    ret5 = (close.iloc[-1] / close.iloc[-5] - 1.0) * 100
                    result["nifty50_ret5d"] = float(ret5)
                else:
                    result["nifty50_ret5d"] = 0.0
                # 20-day volatility
                if len(close) >= 20:
                    vol20 = close.pct_change().rolling(20, min_periods=5).std().iloc[-1] * 100
                    result["nifty50_vol20"] = float(vol20) if not np.isnan(vol20) else 0.0
                else:
                    result["nifty50_vol20"] = 0.0
                    
        except Exception as e:
            print(f"[WARN] Failed to fetch macro data for {symbol}: {e}")
            if name == "usd_inr":
                result["usd_inr"] = 83.0
            elif name == "india_vix":
                result["india_vix"] = 15.0
            elif name == "nifty50":
                result["nifty50_close"] = 25000.0
                result["nifty50_ret14d"] = 0.0
                result["nifty50_ret5d"] = 0.0
                result["nifty50_vol20"] = 0.0
    
    # Add FII/DII flow data
    fii_dii = fetch_fii_dii_flow()
    result.update(fii_dii)
    
    return result


def fetch_macro_history(symbol: str, period: str = "3y") -> pd.DataFrame:
    """
    Fetch full historical data for a macro symbol.
    
    Returns:
        DataFrame with timestamps, close, and derived features
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", auto_adjust=False)
        
        if df.empty:
            return pd.DataFrame()
            
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
            
        dt_col = "Date" if "Date" in df.columns else "Datetime"
        df["timestamps"] = pd.to_datetime(df[dt_col]).dt.tz_localize(None)
        df = df.rename(columns={"Close": "close", "Volume": "volume"})
        df = df[["timestamps", "close", "volume"]].dropna()
        df = df.sort_values("timestamps").reset_index(drop=True)
        
        # Add derived features
        df["ret_1d"] = df["close"].pct_change(1) * 100
        df["ret_5d"] = df["close"].pct_change(5) * 100
        df["ret_14d"] = df["close"].pct_change(14) * 100
        df["vol_20d"] = df["ret_1d"].rolling(20, min_periods=5).std() * 100
        
        return df
    except Exception as e:
        print(f"[WARN] Failed to fetch macro history for {symbol}: {e}")
        return pd.DataFrame()


def align_macro_to_intraday(df_intraday: pd.DataFrame, macro_data: Dict[str, float]) -> pd.DataFrame:
    """
    Broadcast macro features to all rows of an intraday DataFrame.
    """
    out = df_intraday.copy()
    for key, val in macro_data.items():
        out[key] = val
    return out


if __name__ == "__main__":
    # Quick test
    data = fetch_macro_data()
    print("Macro data fetched:")
    for k, v in data.items():
        print(f"  {k}: {v}")