"""
Fast Walk-Forward Validation - Skip Kronos for speed
Uses only technical features for quick validation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime

from daily_kronos_pipeline import DailyFeatureEngine, fetch_macro_data, MARKET_CONTEXT_SYMBOLS

SYMBOLS = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"]
HORIZON_DAYS = 10
TRAIN_WINDOW = 60
TEST_WINDOW = 10


def fetch_daily(symbol, period="6mo"):
    ticker = yf.Ticker(symbol)
    raw = ticker.history(period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.reset_index()
    dt = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["timestamps"] = pd.to_datetime(raw[dt]).dt.tz_localize(None)
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume"})
    return raw[["timestamps", "open", "high", "low", "close", "volume"]].dropna().sort_values("timestamps").reset_index(drop=True)


def fetch_context(period="6mo"):
    ctx = {}
    for sym in MARKET_CONTEXT_SYMBOLS:
        try:
            ctx[sym] = fetch_daily(sym, period=period)
        except:
            ctx[sym] = pd.DataFrame()
    return ctx


def align_context(df, ctx):
    out = df.copy()
    for sym, cdf in ctx.items():
        if cdf.empty:
            continue
        pre = sym.replace("^", "").replace("=", "").replace("-", "_")
        merged = out.merge(cdf[["timestamps", "close"]].rename(
            columns={"close": f"{pre}_close"}), on="timestamps", how="left")
        merged[f"{pre}_ret_1"] = merged[f"{pre}_close"].pct_change(1)
        merged[f"{pre}_ret_5"] = merged[f"{pre}_close"].pct_change(5)
        merged[f"{pre}_vol_20"] = merged[f"{pre}_close"].pct_change().rolling(20, min_periods=5).std()
        out = merged.drop(columns=[c for c in merged.columns if c.startswith(f"{pre}_close")], errors="ignore")
    return out


def walk_forward_fast(symbol, macro):
    """Fast walk-forward without Kronos (technical features only)."""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD (FAST): {symbol}")
    print(f"{'='*60}")
    
    df = fetch_daily(symbol, period="6mo")
    ctx = fetch_context(period="6mo")
    df = align_context(df, ctx)
    
    if len(df) < TRAIN_WINDOW + TEST_WINDOW + HORIZON_DAYS:
        print(f"  Insufficient data: {len(df)} rows")
        return None
    
    # Generate features for entire dataset (no Kronos)
    df_feat = DailyFeatureEngine.generate(df, macro=macro)
    
    # Remove Kronos-dependent features
    kronos_cols = [c for c in df_feat.columns if 'kronos' in c.lower()]
    for col in kronos_cols:
        df_feat[col] = 0.0
    
    # Create target
    df_feat['target'] = (df_feat['close'].shift(-HORIZON_DAYS) / df_feat['close'] - 1 > 0.02).astype(int)
    df_feat = df_feat.dropna(subset=['target', 'ret_1']).reset_index(drop=True)
    
    folds = []
    start_idx = TRAIN_WINDOW
    
    while start_idx + TEST_WINDOW + HORIZON_DAYS <= len(df_feat):
        fold_num = len(folds) + 1
        
        train = df_feat.iloc[start_idx - TRAIN_WINDOW:start_idx]
        test = df_feat.iloc[start_idx:min(start_idx + TEST_WINDOW, len(df_feat) - HORIZON_DAYS)]
        
        if len(test) == 0:
            break
        
        # Use available features (excluding Kronos)
        feature_cols = [c for c in DailyFeatureEngine.FEATURES if c in df_feat.columns and 'kronos' not in c.lower()]
        
        X_train = train[feature_cols].fillna(0)
        y_train = train['target']
        X_test = test[feature_cols].fillna(0)
        y_test = test['target']
        
        # Simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        hits = (preds == y_test).sum()
        total = len(y_test)
        accuracy = hits / total * 100
        
        fold_summary = {
            "fold": fold_num,
            "test_start": test.iloc[0]["timestamps"].strftime("%Y-%m-%d"),
            "test_end": test.iloc[-1]["timestamps"].strftime("%Y-%m-%d"),
            "total": total,
            "hits": int(hits),
            "accuracy": accuracy
        }
        folds.append(fold_summary)
        
        print(f"  Fold {fold_num}: {fold_summary['test_start']} to {fold_summary['test_end']} | Acc={accuracy:.1f}% ({hits}/{total})")
        
        start_idx += TEST_WINDOW
    
    if folds:
        total_preds = sum(f["total"] for f in folds)
        total_hits = sum(f["hits"] for f in folds)
        avg_acc = total_hits / total_preds * 100 if total_preds > 0 else 0
        
        return {
            "symbol": symbol,
            "total_folds": len(folds),
            "total_predictions": total_preds,
            "total_hits": total_hits,
            "overall_accuracy": avg_acc,
            "folds": folds
        }
    
    return None


def main():
    print(f"\n{'#'*60}")
    print("FAST WALK-FORWARD VALIDATION (6-month window)")
    print(f"{'#'*60}")
    
    macro = fetch_macro_data()
    
    all_results = []
    for sym in SYMBOLS:
        try:
            res = walk_forward_fast(sym, macro)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"  ERROR on {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*60}")
    print("WALK-FORWARD SUMMARY (FAST)")
    print(f"{'#'*60}")
    print(f"{'Symbol':<15} {'Folds':>6} {'Total':>6} {'Hits':>6} {'Acc':>7}")
    print("-"*45)
    for r in all_results:
        print(f"{r['symbol']:<15} {r['total_folds']:>6} {r['total_predictions']:>6} {r['total_hits']:>6} {r['overall_accuracy']:>6.1f}%")
    
    if all_results:
        total_preds = sum(r["total_predictions"] for r in all_results)
        total_hits = sum(r["total_hits"] for r in all_results)
        avg_acc = total_hits / total_preds * 100 if total_preds > 0 else 0
        print("-"*45)
        print(f"{'AVERAGE':<15} {'':>6} {total_preds:>6} {total_hits:>6} {avg_acc:>6.1f}%")
    
    with open("walkforward_fast_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to walkforward_fast_results.json")


if __name__ == "__main__":
    main()
