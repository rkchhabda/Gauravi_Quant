"""
Walk-Forward Validation - 6-month window for all symbols
Implements WORK_LOG.md priority: Run full walk-forward on 6-month window
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime

sys.path.append("./kronos_lib")
from model import Kronos, KronosTokenizer, KronosPredictor

from daily_kronos_pipeline import (
    DailyKronosPipeline, MarketMoodEngine, DailyFeatureEngine,
    fetch_macro_data, KRONOS_MODEL_NAME, TOKENIZER_NAME,
    KRONOS_DAILY_LOOKBACK, KRONOS_DAILY_PRED_LEN, MARKET_CONTEXT_SYMBOLS
)

import torch

SYMBOLS = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"]
HORIZON_DAYS = 10
TRAIN_WINDOW = 60  # 60 days training
TEST_WINDOW = 10   # 10 days testing (walk-forward step)


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


def get_kronos_forecast(df, predictor):
    lookback = min(len(df), KRONOS_DAILY_LOOKBACK)
    x_df = df.iloc[-lookback:][["open", "high", "low", "close"]].reset_index(drop=True)
    x_ts = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)
    last_date = x_ts.iloc[-1]
    y_ts = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=KRONOS_DAILY_PRED_LEN))
    pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                             pred_len=KRONOS_DAILY_PRED_LEN, T=0.7, top_p=0.9, sample_count=1, verbose=False)
    last_c = float(x_df["close"].iloc[-1])
    proj_c = float(pred["close"].iloc[-1])
    return {"kronos_proj_return": ((proj_c - last_c) / last_c) * 100,
            "kronos_proj_direction": 1.0 if proj_c >= last_c else 0.0,
            "kronos_proj_range_pct": ((pred["high"].max() - pred["low"].min()) / last_c) * 100,
            "kronos_proj_std_pct": float(pred["close"].std() / last_c * 100),
            "kronos_confidence": min(100, max(50, 50 + abs((proj_c - last_c) / last_c) * 1000))}


def walk_forward_validation(symbol, predictor, macro, mood_engine):
    """Run walk-forward validation on 6-month window."""
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD: {symbol}")
    print(f"Train: {TRAIN_WINDOW}d | Test: {TEST_WINDOW}d | Horizon: {HORIZON_DAYS}d")
    print(f"{'='*60}")
    
    df = fetch_daily(symbol, period="6mo")
    ctx = fetch_context(period="6mo")
    df = align_context(df, ctx)
    
    if len(df) < TRAIN_WINDOW + TEST_WINDOW + HORIZON_DAYS:
        print(f"  Insufficient data: {len(df)} rows")
        return None
    
    folds = []
    start_idx = TRAIN_WINDOW
    
    while start_idx + TEST_WINDOW + HORIZON_DAYS <= len(df):
        fold_num = len(folds) + 1
        
        # Training data
        train_df = df.iloc[start_idx - TRAIN_WINDOW:start_idx].copy()
        
        # Test data
        test_start = start_idx
        test_end = min(start_idx + TEST_WINDOW, len(df) - HORIZON_DAYS)
        
        fold_preds = []
        
        for test_idx in range(test_start, test_end):
            # Get Kronos forecast on data up to test point
            train_to_test = df.iloc[:test_idx + 1].copy()
            
            mood = mood_engine.compute(symbol)
            kronos_f = get_kronos_forecast(train_to_test, predictor)
            test_feat = DailyFeatureEngine.generate(train_to_test, kronos_feats=kronos_f, macro=macro, mood=mood)
            
            for col in DailyFeatureEngine.FEATURES:
                if col not in test_feat.columns:
                    test_feat[col] = 0.0
            X_test = test_feat[DailyFeatureEngine.FEATURES].iloc[-1:]
            
            kronos_dir = "UP" if kronos_f["kronos_proj_return"] >= 0 else "DOWN"
            
            # Get actual outcome
            actual_idx = min(test_idx + HORIZON_DAYS, len(df) - 1)
            actual_ret = (df.iloc[actual_idx]["close"] - df.iloc[test_idx]["close"]) / df.iloc[test_idx]["close"]
            actual_dir = "UP" if actual_ret >= 0 else "DOWN"
            is_hit = (kronos_dir == actual_dir)
            
            fold_preds.append({
                "test_idx": test_idx,
                "date": df.iloc[test_idx]["timestamps"].strftime("%Y-%m-%d"),
                "kronos_dir": kronos_dir,
                "kronos_ret": round(kronos_f["kronos_proj_return"], 2),
                "actual_dir": actual_dir,
                "actual_ret": round(actual_ret * 100, 2),
                "is_hit": is_hit
            })
        
        # Fold summary
        if fold_preds:
            hits = sum(1 for p in fold_preds if p["is_hit"])
            total = len(fold_preds)
            accuracy = hits / total * 100
            
            fold_summary = {
                "fold": fold_num,
                "train_start": df.iloc[start_idx - TRAIN_WINDOW]["timestamps"].strftime("%Y-%m-%d"),
                "train_end": df.iloc[start_idx - 1]["timestamps"].strftime("%Y-%m-%d"),
                "test_start": df.iloc[test_start]["timestamps"].strftime("%Y-%m-%d"),
                "test_end": df.iloc[min(test_end - 1, len(df) - 1)]["timestamps"].strftime("%Y-%m-%d"),
                "total": total,
                "hits": hits,
                "accuracy": accuracy,
                "predictions": fold_preds
            }
            folds.append(fold_summary)
            
            print(f"  Fold {fold_num}: {fold_summary['test_start']} to {fold_summary['test_end']} | Acc={accuracy:.1f}% ({hits}/{total})")
        
        start_idx += TEST_WINDOW
    
    # Overall summary
    if folds:
        all_preds = []
        for f in folds:
            all_preds.extend(f["predictions"])
        
        total = len(all_preds)
        hits = sum(1 for p in all_preds if p["is_hit"])
        accuracy = hits / total * 100
        
        kronos_correct = sum(1 for p in all_preds if p["kronos_dir"] == p["actual_dir"])
        kronos_acc = kronos_correct / total * 100
        
        summary = {
            "symbol": symbol,
            "total_folds": len(folds),
            "total_predictions": total,
            "total_hits": hits,
            "overall_accuracy": accuracy,
            "kronos_accuracy": kronos_acc,
            "folds": folds
        }
        
        print(f"\n  SUMMARY: {total} predictions, Accuracy={accuracy:.1f}% ({hits}/{total})")
        print(f"  Kronos Accuracy: {kronos_acc:.1f}%")
        
        return summary
    
    return None


def main():
    print(f"\n{'#'*60}")
    print("WALK-FORWARD VALIDATION (6-month window)")
    print(f"Symbols: {SYMBOLS}")
    print(f"{'#'*60}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer_kwargs = {"token": os.environ.get("HF_TOKEN")} if os.environ.get("HF_TOKEN") else {}
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME, **tokenizer_kwargs)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME, **tokenizer_kwargs)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    
    mood_engine = MarketMoodEngine()
    macro = fetch_macro_data()
    
    all_results = []
    for sym in SYMBOLS:
        try:
            res = walk_forward_validation(sym, predictor, macro, mood_engine)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"  ERROR on {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*60}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'#'*60}")
    print(f"{'Symbol':<15} {'Folds':>6} {'Total':>6} {'Hits':>6} {'Acc':>7} {'Kronos':>8}")
    print("-"*55)
    for r in all_results:
        print(f"{r['symbol']:<15} {r['total_folds']:>6} {r['total_predictions']:>6} {r['total_hits']:>6} {r['overall_accuracy']:>6.1f}% {r['kronos_accuracy']:>7.1f}%")
    
    if all_results:
        total_preds = sum(r["total_predictions"] for r in all_results)
        total_hits = sum(r["total_hits"] for r in all_results)
        avg_acc = total_hits / total_preds * 100 if total_preds > 0 else 0
        print("-"*55)
        print(f"{'AVERAGE':<15} {'':>6} {total_preds:>6} {total_hits:>6} {avg_acc:>6.1f}%")
    
    with open("walkforward_6mo_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to walkforward_6mo_results.json")


if __name__ == "__main__":
    main()
