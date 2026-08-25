"""
Fast Walk-Forward Validation - Single Symbol Quick Test
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

sys.path.append("./kronos_lib")
from model import Kronos, KronosTokenizer, KronosPredictor

from daily_kronos_pipeline import (
    DailyKronosPipeline, MarketMoodEngine, DailyFeatureEngine,
    fetch_macro_data, KRONOS_MODEL_NAME, TOKENIZER_NAME,
    KRONOS_DAILY_LOOKBACK, KRONOS_DAILY_PRED_LEN, MARKET_CONTEXT_SYMBOLS
)

import torch
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SYMBOLS = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"]
HORIZON_DAYS = 10
MIN_TRAIN_DAYS = 250


def fetch_daily_cached(symbol, period="2y"):
    ticker = yf.Ticker(symbol)
    raw = ticker.history(period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.reset_index()
    dt = "Datetime" if "Datetime" in raw.columns else "Date"
    raw["timestamps"] = pd.to_datetime(raw[dt]).dt.tz_localize(None)
    raw = raw.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    return raw[["timestamps","open","high","low","close","volume"]].dropna().sort_values("timestamps").reset_index(drop=True)


def fetch_context(period="2y"):
    ctx = {}
    for sym in MARKET_CONTEXT_SYMBOLS:
        try:
            ctx[sym] = fetch_daily_cached(sym, period=period)
        except:
            ctx[sym] = pd.DataFrame()
    return ctx


def align_context(df, ctx):
    out = df.copy()
    for sym, cdf in ctx.items():
        if cdf.empty: continue
        pre = sym.replace("^","").replace("=","").replace("-","_")
        merged = out.merge(cdf[["timestamps","close"]].rename(columns={"close":f"{pre}_close"}), on="timestamps", how="left")
        merged[f"{pre}_ret_1"] = merged[f"{pre}_close"].pct_change(1)
        merged[f"{pre}_ret_5"] = merged[f"{pre}_close"].pct_change(5)
        merged[f"{pre}_vol_20"] = merged[f"{pre}_close"].pct_change().rolling(20, min_periods=5).std()
        out = merged.drop(columns=[c for c in merged.columns if c.startswith(f"{pre}_close")], errors="ignore")
    return out


def get_kronos_forecast(df, predictor):
    lookback = min(len(df), KRONOS_DAILY_LOOKBACK)
    x_df = df.iloc[-lookback:][["open","high","low","close"]].reset_index(drop=True)
    x_ts = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)
    last_date = x_ts.iloc[-1]
    y_ts = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=KRONOS_DAILY_PRED_LEN))
    pred = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                              pred_len=KRONOS_DAILY_PRED_LEN, T=0.7, top_p=0.9, sample_count=1, verbose=False)
    last_c = float(x_df["close"].iloc[-1]); proj_c = float(pred["close"].iloc[-1])
    return {"kronos_proj_return": ((proj_c - last_c) / last_c) * 100,
            "kronos_proj_direction": 1.0 if proj_c >= last_c else 0.0,
            "kronos_proj_range_pct": ((pred["high"].max() - pred["low"].min()) / last_c) * 100,
            "kronos_proj_std_pct": float(pred["close"].std() / last_c * 100),
            "kronos_confidence": min(100, max(50, 50 + abs((proj_c - last_c) / last_c) * 1000))}


def run_walkforward(symbol):
    print(f"\n{'='*50}")
    print(f"FAST WALK-FORWARD: {symbol}")
    print(f"{'='*50}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    
    print("  Fetching data...")
    df = fetch_daily_cached(symbol, period="2y")
    ctx = fetch_context(period="2y")
    df = align_context(df, ctx)
    
    if len(df) < MIN_TRAIN_DAYS + HORIZON_DAYS * 5:
        print(f"  Insufficient data: {len(df)} days")
        return None
    
    mood_engine = MarketMoodEngine()
    macro = fetch_macro_data()
    
    end_idx = len(df) - HORIZON_DAYS
    start_train = MIN_TRAIN_DAYS
    
    predictions = []
    step = 30  # Test every 30 days for speed
    
    for test_idx in range(start_train + HORIZON_DAYS, end_idx, step):
        train_df = df.iloc[:test_idx].copy()
        if len(train_df) < MIN_TRAIN_DAYS:
            continue
        
        mood = mood_engine.compute(symbol)
        kronos_f = get_kronos_forecast(train_df, predictor)
        train_feat = DailyFeatureEngine.generate(train_df, kronos_feats=kronos_f, macro=macro, mood=mood)
        
        future_ret = train_feat['close'].shift(-1) / train_feat['close'] - 1
        train_feat['TARGET'] = (future_ret >= 0).astype(int)
        valid = train_feat.iloc[:-1].dropna()
        
        if len(valid) < 50:
            continue
        
        for col in DailyFeatureEngine.FEATURES:
            if col not in valid.columns:
                valid[col] = 0.0
        
        X = valid[DailyFeatureEngine.FEATURES]
        y = valid['TARGET']
        
        if len(np.unique(y)) < 2:
            continue
        
        # Train base models (fast config)
        cv = TimeSeriesSplit(n_splits=3)
        lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1)
        et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=6, max_features=0.7, random_state=42, n_jobs=-1)
        rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=8, max_features=0.65, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        
        oof = np.zeros((len(X), 4))
        for tr, te in cv.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            for i, m in enumerate([lgbm, et, rf, xgb]):
                m.fit(Xtr, ytr)
                oof[te, i] = m.predict_proba(Xte)[:, 1]
        
        meta = LogisticRegression(random_state=42, max_iter=1000, C=1.0).fit(oof, y)
        
        # Test point
        test_idx = min(test_idx + HORIZON_DAYS, len(df) - 1)
        test_df = df.iloc[:test_idx + HORIZON_DAYS].copy()
        test_mood = mood_engine.compute(symbol)
        test_kronos = get_kronos_forecast(test_df, predictor)
        test_feat = DailyFeatureEngine.generate(test_df, kronos_feats=test_kronos, macro=macro, mood=test_mood)
        
        for col in DailyFeatureEngine.FEATURES:
            if col not in test_feat.columns:
                test_feat[col] = 0.0
        X_test = test_feat[DailyFeatureEngine.FEATURES].iloc[-1:]
        
        base_probs = np.column_stack([
            lgbm.predict_proba(X_test)[:, 1],
            et.predict_proba(X_test)[:, 1],
            rf.predict_proba(X_test)[:, 1],
            xgb.predict_proba(X_test)[:, 1],
        ])
        ml_prob = float(meta.predict_proba(base_probs)[0, 1])
        
        kronos_dir = "UP" if test_kronos["kronos_proj_return"] >= 0 else "DOWN"
        ml_dir = "UP" if ml_prob >= 0.5 else "DOWN"
        
        models_agree = (kronos_dir == ml_dir)
        mood_score = test_mood["score"]
        mood_opposed = (mood_score < -0.4 and kronos_dir == "UP") or (mood_score > 0.4 and kronos_dir == "DOWN")
        
        if models_agree and not mood_opposed and ml_prob >= 0.55:
            direction = kronos_dir
            tradeable = True
        else:
            direction = kronos_dir
            tradeable = False
        
        actual_idx = min(test_idx + HORIZON_DAYS, len(df) - 1)
        actual_ret = (df.iloc[actual_idx]["close"] - df.iloc[test_idx]["close"]) / df.iloc[test_idx]["close"]
        actual_dir = "UP" if actual_ret >= 0 else "DOWN"
        is_hit = (direction == actual_dir)
        
        pred = {
            "date": df.iloc[test_idx]["timestamps"].strftime("%Y-%m-%d"),
            "kronos_dir": kronos_dir, "kronos_ret": round(test_kronos["kronos_proj_return"], 2),
            "ml_prob": round(ml_prob, 3), "ml_dir": ml_dir,
            "mood": test_mood["label"], "mood_score": round(mood_score, 3),
            "direction": direction, "tradeable": tradeable,
            "actual_dir": actual_dir, "actual_ret": round(actual_ret * 100, 2),
            "is_hit": is_hit
        }
        predictions.append(pred)
        
        hit_str = "HIT" if is_hit else "MISS"
        trade_str = "TRADE" if tradeable else "SKIP"
        print(f"  {pred['date']} | K:{kronos_dir}({pred['kronos_ret']:+.1f}%) ML:{ml_dir}({ml_prob:.1%}) Mood:{test_mood['label']} -> {direction} {trade_str} | Actual:{actual_dir}({pred['actual_ret']:+.1f}%) {hit_str}")
    
    if not predictions:
        return None
    
    all_df = pd.DataFrame(predictions)
    total = len(all_df)
    hits = all_df["is_hit"].sum()
    accuracy = hits / total * 100
    
    tradeable_df = all_df[all_df["tradeable"]]
    trade_total = len(tradeable_df)
    trade_hits = tradeable_df["is_hit"].sum() if trade_total > 0 else 0
    trade_acc = trade_hits / trade_total * 100 if trade_total > 0 else 0
    
    kronos_acc = (all_df["kronos_dir"] == all_df["actual_dir"]).mean() * 100
    ml_acc = (all_df["ml_dir"] == all_df["actual_dir"]).mean() * 100
    
    print(f"\n  RESULTS: Total={total}, Acc={accuracy:.1f}% ({int(hits)}/{total})")
    print(f"  Kronos={kronos_acc:.1f}%, ML={ml_acc:.1f}%, Tradeable={trade_total}/{total} ({trade_acc:.1f}%)")
    
    return {
        "symbol": symbol,
        "total": total,
        "accuracy": accuracy,
        "hits": int(hits),
        "kronos_acc": kronos_acc,
        "ml_acc": ml_acc,
        "tradeable": trade_total,
        "trade_acc": trade_acc,
        "predictions": predictions
    }


if __name__ == "__main__":
    print(f"\n{'#'*50}")
    print(f"FAST WALK-FORWARD VALIDATION")
    print(f"Horizon: {HORIZON_DAYS} days | Symbols: {SYMBOLS}")
    print(f"{'#'*50}")
    
    all_results = []
    for sym in SYMBOLS:
        try:
            res = run_walkforward(sym)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"  ERROR on {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*50}")
    print("SUMMARY")
    print(f"{'#'*50}")
    print(f"{'Symbol':<15} {'Total':>6} {'Acc':>7} {'Kronos':>8} {'ML':>7} {'Trades':>8} {'Trade%':>7}")
    print("-"*60)
    for r in all_results:
        print(f"{r['symbol']:<15} {r['total']:>6} {r['accuracy']:>6.1f}% {r['kronos_acc']:>7.1f}% {r['ml_acc']:>6.1f}% {r['tradeable']:>8} {r['trade_acc']:>6.1f}%")
    
    if all_results:
        total_preds = sum(r["total"] for r in all_results)
        total_hits = sum(r["hits"] for r in all_results)
        avg_acc = np.mean([r["accuracy"] for r in all_results])
        print("-"*60)
        print(f"{'AVERAGE':<15} {total_preds:>6} {avg_acc:>6.1f}%")
    
    with open("walkforward_fast_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to walkforward_fast_results.json")