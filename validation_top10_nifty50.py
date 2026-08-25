"""
Expanded Validation - Top 10 NIFTY 50 High Market Cap Stocks
Tests walk-forward accuracy on major Indian large-cap stocks
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
    DailyKronosPipeline, MarketMoodEngine, DailyFeatureEngine,
    fetch_macro_data, KRONOS_MODEL_NAME, TOKENIZER_NAME,
    KRONOS_DAILY_LOOKBACK, KRONOS_DAILY_PRED_LEN, MARKET_CONTEXT_SYMBOLS
)

import torch
import warnings
warnings.filterwarnings('ignore')

# Top 10 NIFTY 50 by Market Cap (2024-2026)
TOP_10_NIFTY50 = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
}

HORIZON_DAYS = 10


def fetch_daily(symbol, period="6mo"):
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


def fetch_context(period="6mo"):
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


def validate_symbol(symbol, name, macro, mood_engine):
    """Validate a single symbol with comprehensive metrics."""
    print(f"\n{'='*70}")
    print(f"VALIDATING: {symbol} ({name})")
    print(f"{'='*70}")
    
    # Fetch data
    df = fetch_daily(symbol, period="6mo")
    ctx = fetch_context(period="6mo")
    df = align_context(df, ctx)
    
    if len(df) < 50:
        print(f"  Insufficient data: {len(df)} rows")
        return None
    
    # Generate features
    df_feat = DailyFeatureEngine.generate(df, macro=macro)
    
    # Remove Kronos-dependent features for speed
    kronos_cols = [c for c in df_feat.columns if 'kronos' in c.lower()]
    for col in kronos_cols:
        df_feat[col] = 0.0
    
    # Create target
    df_feat['target'] = (df_feat['close'].shift(-HORIZON_DAYS) / df_feat['close'] - 1 > 0.02).astype(int)
    df_feat = df_feat.dropna(subset=['target', 'ret_1']).reset_index(drop=True)
    
    if len(df_feat) < 60:
        print(f"  Insufficient feature data: {len(df_feat)} rows")
        return None
    
    # Walk-forward validation
    TRAIN_WINDOW = 60
    TEST_WINDOW = 10
    
    folds = []
    start_idx = TRAIN_WINDOW
    
    while start_idx + TEST_WINDOW + HORIZON_DAYS <= len(df_feat):
        fold_num = len(folds) + 1
        
        train = df_feat.iloc[start_idx - TRAIN_WINDOW:start_idx]
        test = df_feat.iloc[start_idx:min(start_idx + TEST_WINDOW, len(df_feat) - HORIZON_DAYS)]
        
        if len(test) == 0:
            break
        
        # Use available features
        feature_cols = [c for c in DailyFeatureEngine.FEATURES if c in df_feat.columns and 'kronos' not in c.lower()]
        
        X_train = train[feature_cols].fillna(0)
        y_train = train['target']
        X_test = test[feature_cols].fillna(0)
        y_test = test['target']
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        hits = (preds == y_test).sum()
        total = len(y_test)
        accuracy = hits / total * 100
        
        # Calculate additional metrics
        precision = hits / total * 100 if total > 0 else 0
        avg_proba = proba.mean()
        
        fold_summary = {
            "fold": fold_num,
            "test_start": test.iloc[0]["timestamps"].strftime("%Y-%m-%d"),
            "test_end": test.iloc[-1]["timestamps"].strftime("%Y-%m-%d"),
            "total": total,
            "hits": int(hits),
            "accuracy": accuracy,
            "avg_confidence": round(avg_proba * 100, 1)
        }
        folds.append(fold_summary)
        
        print(f"  Fold {fold_num}: {fold_summary['test_start']} to {fold_summary['test_end']} | Acc={accuracy:.1f}% ({hits}/{total}) | Conf={fold_summary['avg_confidence']:.1f}%")
        
        start_idx += TEST_WINDOW
    
    if not folds:
        return None
    
    # Calculate overall metrics
    total_preds = sum(f["total"] for f in folds)
    total_hits = sum(f["hits"] for f in folds)
    avg_acc = total_hits / total_preds * 100 if total_preds > 0 else 0
    avg_conf = np.mean([f["avg_confidence"] for f in folds])
    
    # Calculate recent trend (last 3 folds)
    recent_folds = folds[-3:] if len(folds) >= 3 else folds
    recent_acc = sum(f["hits"] for f in recent_folds) / sum(f["total"] for f in recent_folds) * 100
    
    # Get current mood
    mood = mood_engine.compute(symbol)
    
    # Get current price info
    latest_price = float(df["close"].iloc[-1])
    ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
    ret_20d = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100 if len(df) >= 21 else 0
    
    result = {
        "symbol": symbol,
        "name": name,
        "latest_price": round(latest_price, 2),
        "ret_5d": round(ret_5d, 2),
        "ret_20d": round(ret_20d, 2),
        "total_folds": len(folds),
        "total_predictions": total_preds,
        "total_hits": total_hits,
        "overall_accuracy": round(avg_acc, 1),
        "recent_accuracy": round(recent_acc, 1),
        "avg_confidence": round(avg_conf, 1),
        "mood": mood["label"],
        "mood_score": mood["score"],
        "folds": folds
    }
    
    return result


def main():
    print(f"\n{'#'*70}")
    print("EXPANDED VALIDATION - TOP 10 NIFTY 50 HIGH MARKET CAP STOCKS")
    print(f"{'#'*70}")
    print(f"Horizon: {HORIZON_DAYS} days")
    print(f"Stocks: {len(TOP_10_NIFTY50)}")
    print(f"{'#'*70}")
    
    # Initialize
    macro = fetch_macro_data()
    mood_engine = MarketMoodEngine()
    
    # Validate all symbols
    all_results = []
    for symbol, name in TOP_10_NIFTY50.items():
        try:
            result = validate_symbol(symbol, name, macro, mood_engine)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR on {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x["overall_accuracy"], reverse=True)
    
    # Print summary
    print(f"\n{'#'*70}")
    print("COMPREHENSIVE RESULTS - TOP 10 NIFTY 50")
    print(f"{'#'*70}")
    
    # Table header
    print(f"\n{'Rank':<5} {'Symbol':<15} {'Name':<25} {'Price':>10} {'5D%':>7} {'20D%':>7} {'Acc%':>7} {'Recent%':>8} {'Mood':<10}")
    print("-" * 95)
    
    for i, r in enumerate(all_results, 1):
        print(f"{i:<5} {r['symbol']:<15} {r['name']:<25} {r['latest_price']:>10.2f} {r['ret_5d']:>+6.1f}% {r['ret_20d']:>+6.1f}% {r['overall_accuracy']:>6.1f}% {r['recent_accuracy']:>+7.1f}% {r['mood']:<10}")
    
    # Statistics
    print(f"\n{'#'*70}")
    print("STATISTICS")
    print(f"{'#'*70}")
    
    accuracies = [r["overall_accuracy"] for r in all_results]
    recent_accuracies = [r["recent_accuracy"] for r in all_results]
    
    print(f"Overall Accuracy:")
    print(f"  Mean:   {np.mean(accuracies):.1f}%")
    print(f"  Median: {np.median(accuracies):.1f}%")
    print(f"  Std:    {np.std(accuracies):.1f}%")
    print(f"  Min:    {min(accuracies):.1f}% ({all_results[accuracies.index(min(accuracies))]['symbol']})")
    print(f"  Max:    {max(accuracies):.1f}% ({all_results[accuracies.index(max(accuracies))]['symbol']})")
    
    print(f"\nRecent Accuracy (Last 3 Folds):")
    print(f"  Mean:   {np.mean(recent_accuracies):.1f}%")
    print(f"  Median: {np.median(recent_accuracies):.1f}%")
    
    # Mood distribution
    mood_counts = {}
    for r in all_results:
        mood = r["mood"]
        mood_counts[mood] = mood_counts.get(mood, 0) + 1
    
    print(f"\nMood Distribution:")
    for mood, count in sorted(mood_counts.items()):
        print(f"  {mood}: {count} stocks")
    
    # Performance categories
    print(f"\nPerformance Categories:")
    excellent = [r for r in all_results if r["overall_accuracy"] >= 70]
    good = [r for r in all_results if 50 <= r["overall_accuracy"] < 70]
    average = [r for r in all_results if 40 <= r["overall_accuracy"] < 50]
    poor = [r for r in all_results if r["overall_accuracy"] < 40]
    
    print(f"  Excellent (≥70%): {len(excellent)} stocks")
    for r in excellent:
        print(f"    - {r['symbol']}: {r['overall_accuracy']:.1f}%")
    
    print(f"  Good (50-69%): {len(good)} stocks")
    for r in good:
        print(f"    - {r['symbol']}: {r['overall_accuracy']:.1f}%")
    
    print(f"  Average (40-49%): {len(average)} stocks")
    for r in average:
        print(f"    - {r['symbol']}: {r['overall_accuracy']:.1f}%")
    
    print(f"  Poor (<40%): {len(poor)} stocks")
    for r in poor:
        print(f"    - {r['symbol']}: {r['overall_accuracy']:.1f}%")
    
    # Top performers
    print(f"\nTop 3 Performers:")
    for i, r in enumerate(all_results[:3], 1):
        print(f"  {i}. {r['symbol']} ({r['name']}): {r['overall_accuracy']:.1f}% accuracy")
    
    # Worst performers
    print(f"\nBottom 3 Performers:")
    for i, r in enumerate(all_results[-3:], 1):
        print(f"  {i}. {r['symbol']} ({r['name']}): {r['overall_accuracy']:.1f}% accuracy")
    
    # Save results
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "horizon": HORIZON_DAYS,
        "total_stocks": len(all_results),
        "statistics": {
            "mean_accuracy": round(np.mean(accuracies), 1),
            "median_accuracy": round(np.median(accuracies), 1),
            "std_accuracy": round(np.std(accuracies), 1),
            "min_accuracy": round(min(accuracies), 1),
            "max_accuracy": round(max(accuracies), 1),
            "mean_recent_accuracy": round(np.mean(recent_accuracies), 1),
        },
        "results": all_results
    }
    
    with open("top10_nifty50_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to top10_nifty50_results.json")
    
    # Recommendations
    print(f"\n{'#'*70}")
    print("RECOMMENDATIONS")
    print(f"{'#'*70}")
    
    print("\nFor Production Trading:")
    print("1. Focus on stocks with ≥60% accuracy AND ≥55% recent accuracy")
    print("2. Avoid stocks with <45% accuracy")
    print("3. Consider market mood alignment for entry timing")
    print("4. Use position sizing based on confidence level")
    
    tradeable = [r for r in all_results if r["overall_accuracy"] >= 60 and r["recent_accuracy"] >= 55]
    print(f"\nRecommended for Trading ({len(tradeable)} stocks):")
    for r in tradeable:
        print(f"  - {r['symbol']}: Acc={r['overall_accuracy']:.1f}%, Recent={r['recent_accuracy']:.1f}%, Mood={r['mood']}")


if __name__ == "__main__":
    main()
