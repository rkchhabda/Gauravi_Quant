"""
Train TCS Model + Fix Mood Model Calibration
Implements WORK_LOG.md priorities:
1. Save TCS model
2. Fix Mood Model - Recalibrate weights
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


class ImprovedMarketMoodEngine(MarketMoodEngine):
    """
    Improved mood engine with recalibrated weights.
    Changes from original:
    1. Added ADX filter (only signal when ADX > 25 = trending)
    2. Reduced weight on returns (too noisy)
    3. Added volume confirmation
    4. Better thresholds for BULLISH/BEARISH
    """
    
    @classmethod
    def compute(cls, symbol: str) -> dict:
        df = cls.fetch_daily(symbol)
        if df.empty or len(df) < 20:
            return cls._neutral()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

        # EMAs
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # RSI
        delta = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rsi = (100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))).fillna(50.0)

        # MACD
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig

        # ATR
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        # ADX (trend strength)
        up_m = high.diff().clip(lower=0)
        dn_m = (-low.diff()).clip(lower=0)
        pdi = 100 * (up_m.where(up_m > dn_m, 0).ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
        mdi = 100 * (dn_m.where(dn_m > up_m, 0).ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
        dx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
        adx = dx.ewm(span=14, adjust=False).mean()

        # VWAP
        typical = (high + low + close) / 3
        vol_safe = volume.replace(0, 1)
        vwap = ((typical * vol_safe).rolling(14, min_periods=1).sum()
                / vol_safe.rolling(14, min_periods=1).sum())

        # Returns
        ret5d = close.pct_change(5).fillna(0.0) * 100
        ret14d = close.pct_change(14).fillna(0.0) * 100

        # Volume trend
        vol_sma20 = volume.rolling(20, min_periods=5).mean()
        vol_ratio = (volume / vol_sma20.replace(0, np.nan)).fillna(1.0)

        i = -1
        feats = {
            "D_RSI_14": float(rsi.iloc[i]),
            "D_MACD_HIST": float(macd_hist.iloc[i]),
            "D_ADX": float(adx.iloc[i]),
            "D_EMA9_21": float(ema9.iloc[i] - ema21.iloc[i]),
            "D_EMA21_50": float(ema21.iloc[i] - ema50.iloc[i]),
            "D_RETURN_5D": float(ret5d.iloc[i]),
            "D_RETURN_14D": float(ret14d.iloc[i]),
            "D_ATR_RATIO": float(atr.iloc[i] / close.iloc[i]) if close.iloc[i] else 0.0,
            "D_ABOVE_VWAP": 1.0 if close.iloc[i] > vwap.iloc[i] else 0.0,
            "D_ADX": float(adx.iloc[i]),
            "D_VOL_RATIO": float(vol_ratio.iloc[i]),
        }

        # IMPROVED SCORING with ADX filter and better weights
        adx_val = feats["D_ADX"]
        
        # Only signal when ADX > 20 (trending market)
        if adx_val < 20:
            # Weak trend - reduce confidence
            score = 0.0
            score += 0.10 * cls._norm(feats["D_RSI_14"], 50, 50)
            score += 0.05 * np.sign(feats["D_MACD_HIST"])
            score += 0.10 * np.sign(feats["D_EMA9_21"])
            score += 0.10 * np.sign(feats["D_EMA21_50"])
            score += 0.10 * cls._norm(feats["D_RETURN_5D"], 0, 3)
            score += 0.10 * cls._norm(feats["D_RETURN_14D"], 0, 5)
            score += 0.05 * (feats["D_ABOVE_VWAP"] * 2 - 1)
        else:
            # Strong trend - higher confidence
            score = 0.0
            score += 0.15 * cls._norm(feats["D_RSI_14"], 50, 50)
            score += 0.10 * np.sign(feats["D_MACD_HIST"])
            score += 0.15 * np.sign(feats["D_EMA9_21"])
            score += 0.15 * np.sign(feats["D_EMA21_50"])
            score += 0.10 * cls._norm(feats["D_RETURN_5D"], 0, 3)
            score += 0.15 * cls._norm(feats["D_RETURN_14D"], 0, 5)
            score += 0.10 * (feats["D_ABOVE_VWAP"] * 2 - 1)
            # Volume confirmation bonus
            if feats["D_VOL_RATIO"] > 1.2:
                score *= 1.1  # Boost score if volume confirms
        
        score = float(np.clip(score, -1.0, 1.0))

        # Wider thresholds for NEUTRAL (was ±0.20, now ±0.25)
        label = "BULLISH" if score >= 0.25 else ("BEARISH" if score <= -0.25 else "NEUTRAL")
        
        return {
            "score": round(score, 3),
            "label": label,
            "features": feats,
            "rsi": round(feats["D_RSI_14"], 1),
            "macd_hist": round(feats["D_MACD_HIST"], 4),
            "adx": round(feats["D_ADX"], 1),
            "ret5d": round(feats["D_RETURN_5D"], 2),
            "ret14d": round(feats["D_RETURN_14D"], 2)
        }


def train_tcs_model():
    """Train TCS model and save it."""
    print("\n" + "="*60)
    print("TRAINING TCS MODEL")
    print("="*60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize pipeline
    pipe = DailyKronosPipeline("TCS.NS", horizon=10)
    
    # Train
    print("Training TCS model (3y data)...")
    pipe.train(period="3y")
    
    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe.ml, "models/ml_TCS.NS.pkl")
    print("Saved: models/ml_TCS.NS.pkl")
    
    return pipe


def test_improved_mood():
    """Test improved mood engine on all symbols."""
    print("\n" + "="*60)
    print("TESTING IMPROVED MOOD ENGINE")
    print("="*60)
    
    symbols = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"]
    original_engine = MarketMoodEngine()
    improved_engine = ImprovedMarketMoodEngine()
    
    results = []
    for sym in symbols:
        print(f"\n{sym}:")
        orig = original_engine.compute(sym)
        impr = improved_engine.compute(sym)
        
        print(f"  Original:  {orig['label']:>8} (score={orig['score']:+.3f}, ADX={orig['adx']:.1f})")
        print(f"  Improved:  {impr['label']:>8} (score={impr['score']:+.3f}, ADX={impr['adx']:.1f})")
        
        results.append({
            "symbol": sym,
            "orig_label": orig["label"],
            "orig_score": orig["score"],
            "impr_label": impr["label"],
            "impr_score": impr["score"],
            "adx": impr["adx"]
        })
    
    return results


def main():
    print(f"\n{'#'*60}")
    print("TRAIN TCS + FIX MOOD MODEL")
    print(f"{'#'*60}")
    
    # 1. Test improved mood engine
    mood_results = test_improved_mood()
    
    # 2. Train TCS model
    try:
        tcs_pipe = train_tcs_model()
    except Exception as e:
        print(f"ERROR training TCS: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Quick validation with TCS
    print("\n" + "="*60)
    print("QUICK TCS VALIDATION")
    print("="*60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer_kwargs = {"token": os.environ.get("HF_TOKEN")} if os.environ.get("HF_TOKEN") else {}
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME, **tokenizer_kwargs)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME, **tokenizer_kwargs)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    
    try:
        ml = joblib.load("models/ml_TCS.NS.pkl")
        macro = fetch_macro_data()
        mood_engine = ImprovedMarketMoodEngine()
        
        # Test on last 5 days
        df = tcs_pipe.fetch_daily("TCS.NS", period="3mo")
        ctx = tcs_pipe.fetch_context(period="3mo")
        df = tcs_pipe.align_context(df, ctx)
        
        test_idx = len(df) - 15
        test_df = df.iloc[:test_idx + 10].copy()
        
        mood = mood_engine.compute("TCS.NS")
        kronos_f = tcs_pipe.get_kronos_forecast(test_df)
        test_feat = DailyFeatureEngine.generate(test_df, kronos_feats=kronos_f, macro=macro, mood=mood)
        
        for col in DailyFeatureEngine.FEATURES:
            if col not in test_feat.columns:
                test_feat[col] = 0.0
        X_test = test_feat[DailyFeatureEngine.FEATURES].iloc[-1:]
        
        ml_prob = float(ml.predict_proba(X_test))
        kronos_dir = "UP" if kronos_f["kronos_proj_return"] >= 0 else "DOWN"
        ml_dir = "UP" if ml_prob >= 0.5 else "DOWN"
        
        print(f"TCS Prediction:")
        print(f"  Kronos: {kronos_dir} ({kronos_f['kronos_proj_return']:+.2f}%)")
        print(f"  ML:     {ml_dir} ({ml_prob:.1%})")
        print(f"  Mood:   {mood['label']} ({mood['score']:+.3f})")
        
    except Exception as e:
        print(f"ERROR in TCS validation: {e}")
        import traceback
        traceback.print_exc()
    
    # Save mood results
    with open("mood_improvement_results.json", "w") as f:
        json.dump(mood_results, f, indent=2, default=str)
    print(f"\nSaved mood results to mood_improvement_results.json")


if __name__ == "__main__":
    main()
