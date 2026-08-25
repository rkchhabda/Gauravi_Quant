"""
Train ML Models for Top NIFTY 50 Stocks
Trains and saves models for stocks with ≥60% accuracy
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime

from daily_kronos_pipeline import (
    DailyFeatureEngine, fetch_macro_data, MARKET_CONTEXT_SYMBOLS
)

# Stocks to train (from validation results)
STOCKS_TO_TRAIN = {
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "BHARTIARTL.NS": "Bharti Airtel",
    "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
}

HORIZON_DAYS = 10


def fetch_daily(symbol, period="3y"):
    """Fetch daily OHLCV data."""
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


def fetch_context(period="3y"):
    """Fetch market context (NIFTY50, VIX, USD/INR)."""
    ctx = {}
    for sym in MARKET_CONTEXT_SYMBOLS:
        try:
            ctx[sym] = fetch_daily(sym, period=period)
        except:
            ctx[sym] = pd.DataFrame()
    return ctx


def align_context(df, ctx):
    """Align market context features."""
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


def train_model_for_stock(symbol, name, macro):
    """Train and save ML model for a stock."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {symbol} ({name})")
    print(f"{'='*60}")
    
    # Fetch 3 years of data
    print("  Fetching 3y data...")
    df = fetch_daily(symbol, period="3y")
    ctx = fetch_context(period="3y")
    df = align_context(df, ctx)
    
    if len(df) < 100:
        print(f"  Insufficient data: {len(df)} rows")
        return None
    
    # Generate features
    print("  Generating features...")
    df_feat = DailyFeatureEngine.generate(df, macro=macro)
    
    # Remove Kronos-dependent features
    kronos_cols = [c for c in df_feat.columns if 'kronos' in c.lower()]
    for col in kronos_cols:
        df_feat[col] = 0.0
    
    # Create target
    df_feat['target'] = (df_feat['close'].shift(-HORIZON_DAYS) / df_feat['close'] - 1 > 0.02).astype(int)
    df_feat = df_feat.dropna(subset=['target', 'ret_1']).reset_index(drop=True)
    
    if len(df_feat) < 100:
        print(f"  Insufficient feature data: {len(df_feat)} rows")
        return None
    
    # Prepare features
    feature_cols = [c for c in DailyFeatureEngine.FEATURES if c in df_feat.columns and 'kronos' not in c.lower()]
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['target']
    
    print(f"  Training on {len(X)} samples, {len(feature_cols)} features...")
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Time series split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, proba) if len(np.unique(y_val)) == 2 else 0.5
        scores.append({'accuracy': acc, 'auc': auc})
    
    avg_acc = np.mean([s['accuracy'] for s in scores])
    avg_auc = np.mean([s['auc'] for s in scores])
    
    print(f"  Cross-validation: Acc={avg_acc:.1%}, AUC={avg_auc:.4f}")
    
    # Train final model on all data
    final_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/ml_{symbol}.pkl"
    joblib.dump(final_model, model_path)
    print(f"  Saved: {model_path}")
    
    # Feature importance
    importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(10)
    print(f"\n  Top 10 Features:")
    for feat, imp in top_features.items():
        print(f"    {feat}: {imp:.4f}")
    
    return {
        "symbol": symbol,
        "name": name,
        "samples": len(X),
        "features": len(feature_cols),
        "accuracy": round(avg_acc * 100, 1),
        "auc": round(avg_auc, 4),
        "model_path": model_path
    }


def main():
    print(f"\n{'#'*60}")
    print("TRAINING ML MODELS FOR TOP NIFTY 50 STOCKS")
    print(f"{'#'*60}")
    
    # Fetch macro data
    print("Fetching macro data...")
    macro = fetch_macro_data()
    
    # Train models
    results = []
    for symbol, name in STOCKS_TO_TRAIN.items():
        try:
            result = train_model_for_stock(symbol, name, macro)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR training {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'#'*60}")
    print("TRAINING SUMMARY")
    print(f"{'#'*60}")
    
    print(f"\n{'Symbol':<15} {'Name':<25} {'Samples':>8} {'Features':>9} {'Accuracy':>9} {'AUC':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r['symbol']:<15} {r['name']:<25} {r['samples']:>8} {r['features']:>9} {r['accuracy']:>8.1f}% {r['auc']:>6.4f}")
    
    # Save results
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to training_results.json")
    print(f"\nModels saved to models/ directory")


if __name__ == "__main__":
    main()
