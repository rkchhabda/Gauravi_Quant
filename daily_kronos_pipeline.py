"""
Daily Kronos + ML Confirmation Pipeline
========================================
Architecture:
- Layer 1: Kronos Foundation Model (Daily, 10-day horizon) → Primary directional signal
- Layer 2: Daily ML Stacking Ensemble → Confidence weighting & confirmation
- Layer 3: Market Mood (Daily 14-candle) → Regime filter
- Layer 4: 15m Execution (VWAP/Trend/Reversal) → Entry/exit timing only
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import joblib
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

from macro_utils import fetch_macro_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.append("./kronos_lib")
from model import Kronos, KronosTokenizer, KronosPredictor

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME    = "NeoQuasar/Kronos-Tokenizer-base"
DAILY_LOOKBACK    = 14
KRONOS_DAILY_LOOKBACK = 90
KRONOS_DAILY_PRED_LEN = 10

def get_hf_token():
    """Get Hugging Face token from environment at runtime."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

MARKET_CONTEXT_SYMBOLS = ["^NSEI", "^INDIAVIX", "INR=X"]


class MarketMoodEngine:
    FEATURE_NAMES = [
        "D_RSI_14", "D_MACD_HIST", "D_ADX", "D_EMA9_21", "D_EMA21_50",
        "D_RETURN_5D", "D_RETURN_14D", "D_ATR_RATIO", "D_ABOVE_VWAP", "D_VOL_RATIO"
    ]

    @staticmethod
    def fetch_daily(symbol: str, lookback: int = DAILY_LOOKBACK) -> pd.DataFrame:
        extra = 60
        raw = yf.download(symbol, period=f"{lookback + extra}d", interval="1d", progress=False)
        if raw.empty:
            return pd.DataFrame()
        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw = raw.rename(columns={"Open":"open","High":"high","Low":"low",
                                  "Close":"close","Volume":"volume"})
        raw["Date"] = pd.to_datetime(raw["Date"])
        return raw.sort_values("Date").reset_index(drop=True)

    @classmethod
    def compute(cls, symbol: str) -> dict:
        df = cls.fetch_daily(symbol)
        if df.empty or len(df) < 20:
            return cls._neutral()

        close = df["close"]; high = df["high"]; low = df["low"]
        vol = df["volume"].replace(0, 1)

        ema9  = close.ewm(span=9,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rsi      = (100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))).fillna(50.0)

        macd      = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_sig  = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig

        tr  = pd.concat([high - low,
                         (high - close.shift(1)).abs(),
                         (low  - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        up_m = high.diff().clip(lower=0)
        dn_m = (-low.diff()).clip(lower=0)
        pdi  = 100 * (up_m.where(up_m > dn_m, 0).ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
        mdi  = 100 * (dn_m.where(dn_m > up_m, 0).ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
        dx   = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
        adx  = dx.ewm(span=14, adjust=False).mean()

        typical  = (high + low + close) / 3
        vol_safe = df["volume"].replace(0, 1)
        vwap     = ((typical * vol_safe).rolling(14, min_periods=1).sum()
                    / vol_safe.rolling(14, min_periods=1).sum())

        ret5d  = close.pct_change(5).fillna(0.0)  * 100
        ret14d = close.pct_change(14).fillna(0.0) * 100
        vol_sma20 = vol.rolling(20, min_periods=5).mean()
        vol_ratio = (vol / vol_sma20.replace(0, np.nan)).fillna(1.0)

        i = -1
        adx_val = float(adx.iloc[i])
        feats = {
            "D_RSI_14"    : float(rsi.iloc[i]),
            "D_MACD_HIST" : float(macd_hist.iloc[i]),
            "D_ADX"       : adx_val,
            "D_EMA9_21"   : float(ema9.iloc[i]  - ema21.iloc[i]),
            "D_EMA21_50"  : float(ema21.iloc[i] - ema50.iloc[i]),
            "D_RETURN_5D" : float(ret5d.iloc[i]),
            "D_RETURN_14D": float(ret14d.iloc[i]),
            "D_ATR_RATIO" : float(atr.iloc[i] / close.iloc[i]) if close.iloc[i] else 0.0,
            "D_ABOVE_VWAP": 1.0 if close.iloc[i] > vwap.iloc[i] else 0.0,
            "D_VOL_RATIO" : float(vol_ratio.iloc[i]),
        }

        w = 0.10 if adx_val < 20 else 0.15
        score = 0.0
        score += w * cls._norm(feats["D_RSI_14"],     50,  50)
        score += (w * 0.67) * np.sign(feats["D_MACD_HIST"])
        score += w * np.sign(feats["D_EMA9_21"])
        score += w * np.sign(feats["D_EMA21_50"])
        score += w * cls._norm(feats["D_RETURN_5D"],   0,   3)
        score += w * cls._norm(feats["D_RETURN_14D"],  0,   5)
        score += (w * 0.67) * (feats["D_ABOVE_VWAP"] * 2 - 1)
        if adx_val >= 20 and feats["D_VOL_RATIO"] > 1.2:
            score *= 1.1
        score  = float(np.clip(score, -1.0, 1.0))

        label = "BULLISH" if score >= 0.25 else ("BEARISH" if score <= -0.25 else "NEUTRAL")
        return {"score": round(score,3), "label": label, "features": feats,
                "rsi": round(feats["D_RSI_14"],1), "macd_hist": round(feats["D_MACD_HIST"],4),
                "adx": round(adx_val,1), "ret5d": round(feats["D_RETURN_5D"],2),
                "ret14d": round(feats["D_RETURN_14D"],2)}

    @staticmethod
    def _norm(val, center, scale):
        return float(np.clip((val - center) / scale, -1.0, 1.0)) if scale else 0.0

    @staticmethod
    def _neutral():
        return {"score": 0.0, "label": "NEUTRAL",
                "features": {k: 0.0 for k in MarketMoodEngine.FEATURE_NAMES},
                "rsi": 50.0, "macd_hist": 0.0, "adx": 25.0, "ret5d": 0.0, "ret14d": 0.0}


class DailyFeatureEngine:
    FEATURES = [
        "ret_1", "ret_5", "ret_10", "ret_20",
        "momentum_5", "momentum_10", "momentum_20",
        "ema_gap_5_10", "ema_gap_10_20", "ema_gap_20_50",
        "close_vs_ema10", "close_vs_ema20", "close_vs_ema50",
        "rsi_14", "rsi_slope_5", "macd_hist", "macd_slope_5",
        "atr_pct", "atr_slope_5", "range_pct", "range_10", "range_ratio_10",
        "volatility_5", "volatility_10", "volatility_20", "vol_regime",
        "volume_change_1", "volume_ratio_20", "volume_trend_10",
        "dist_bb_upper", "dist_bb_lower", "bb_width", "bb_squeeze",
        "day_of_week", "month",
        "gap_up", "opening_range", "close_location",
        "stop_loss_pct", "risk_reward_ratio",
        "usd_inr", "india_vix", "nifty50_ret14d", "nifty50_ret5d", "nifty50_vol20",
        "kronos_proj_return", "kronos_proj_direction", "kronos_proj_range_pct",
        "kronos_proj_std_pct", "kronos_confidence",
        "macro_vix_regime", "macro_nifty_trend", "macro_usdinr_mom",
        "mood_score", "mood_label_enc", "mood_momentum_align", "mood_vol_align",
        "kronos_mood_align", "kronos_macro_align",
    ]

    @classmethod
    def generate(cls, df: pd.DataFrame, kronos_feats: dict = None, macro: dict = None, mood: dict = None) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]; high = out["high"]; low = out["low"]
        volume = out["volume"].replace(0, np.nan)

        out["ret_1"] = close.pct_change(1)
        out["ret_5"] = close.pct_change(5)
        out["ret_10"] = close.pct_change(10)
        out["ret_20"] = close.pct_change(20)

        out["momentum_5"]  = close / close.shift(5)  - 1.0
        out["momentum_10"] = close / close.shift(10) - 1.0
        out["momentum_20"] = close / close.shift(20) - 1.0

        ema5 = close.ewm(span=5, adjust=False).mean()
        ema10 = close.ewm(span=10, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        out["ema_gap_5_10"]  = ema5 / ema10 - 1.0
        out["ema_gap_10_20"] = ema10 / ema20 - 1.0
        out["ema_gap_20_50"] = ema20 / ema50 - 1.0
        out["close_vs_ema10"] = close / ema10 - 1.0
        out["close_vs_ema20"] = close / ema20 - 1.0
        out["close_vs_ema50"] = close / ema50 - 1.0

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(50.0)
        out["rsi_14"] = rsi
        out["rsi_slope_5"] = rsi.diff(5)

        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        out["macd_hist"] = macd - macd_sig
        out["macd_slope_5"] = out["macd_hist"].diff(5)

        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low  - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=3).mean()
        out["atr_pct"] = (atr / close).fillna(0.0)
        out["atr_slope_5"] = out["atr_pct"].diff(5)
        out["range_pct"] = (high - low) / close
        out["range_10"] = out["range_pct"].rolling(10, min_periods=3).mean()
        out["range_ratio_10"] = out["range_pct"] / out["range_10"].replace(0, np.nan)

        out["volatility_5"]  = out["ret_1"].rolling(5,  min_periods=3).std()
        out["volatility_10"] = out["ret_1"].rolling(10, min_periods=5).std()
        out["volatility_20"] = out["ret_1"].rolling(20, min_periods=8).std()
        vol_median = out["volatility_20"].rolling(60, min_periods=20).median()
        out["vol_regime"] = (out["volatility_20"] / vol_median.replace(0, np.nan)).fillna(1.0)

        out["volume_change_1"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        out["volume_ratio_20"] = volume / volume.rolling(20, min_periods=5).mean()
        out["volume_trend_10"] = volume.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True)

        roll_std = close.rolling(20, min_periods=1).std().fillna(1.0)
        bb_mid = close.rolling(20, min_periods=1).mean()
        bb_up = bb_mid + 2 * roll_std
        bb_lo = bb_mid - 2 * roll_std
        out["dist_bb_upper"] = (bb_up - close) / close
        out["dist_bb_lower"] = (close - bb_lo) / close
        out["bb_width"] = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)
        out["bb_squeeze"] = (out["bb_width"] / out["bb_width"].rolling(20).mean()).fillna(1.0)

        if "timestamps" in out.columns:
            ts = pd.to_datetime(out["timestamps"])
            out["day_of_week"] = ts.dt.dayofweek.astype(float)
            out["month"] = ts.dt.month.astype(float)
        else:
            out["day_of_week"] = 0.0; out["month"] = 0.0

        out["gap_up"] = (out["open"] / close.shift(1) - 1.0).fillna(0.0)
        out["opening_range"] = (high - low) / out["open"].replace(0, np.nan)
        out["close_location"] = (close - low) / (high - low).replace(0, np.nan)

        out["stop_loss_pct"] = out["atr_pct"] * 2.0
        out["risk_reward_ratio"] = 0.02 / (out["atr_pct"] * 2.0).replace(0, np.nan)

        macro = macro or {}
        out["usd_inr"] = macro.get("usd_inr", 83.0)
        out["india_vix"] = macro.get("india_vix", 15.0)
        out["nifty50_ret14d"] = macro.get("nifty50_ret14d", 0.0)
        out["nifty50_ret5d"] = macro.get("nifty50_ret5d", 0.0)
        out["nifty50_vol20"] = macro.get("nifty50_vol20", 0.0)

        if kronos_feats:
            out["kronos_proj_return"] = kronos_feats.get("kronos_proj_return", 0.0)
            out["kronos_proj_direction"] = kronos_feats.get("kronos_proj_direction", 0.5)
            out["kronos_proj_range_pct"] = kronos_feats.get("kronos_proj_range_pct", 0.0)
            out["kronos_proj_std_pct"] = kronos_feats.get("kronos_proj_std_pct", 0.0)
            out["kronos_confidence"] = kronos_feats.get("kronos_confidence", 50.0)
        else:
            out["kronos_proj_return"] = 0.0
            out["kronos_proj_direction"] = 0.5
            out["kronos_proj_range_pct"] = 0.0
            out["kronos_proj_std_pct"] = 0.0
            out["kronos_confidence"] = 50.0

        # Macro interaction features (varying per row)
        vix_val = macro.get("india_vix", 15.0)
        out["macro_vix_regime"] = np.clip((vix_val - 12) / 15, -1, 1) * np.sign(out["ret_20"])
        nifty_ret14 = macro.get("nifty50_ret14d", 0.0)
        out["macro_nifty_trend"] = np.clip(nifty_ret14 / 5, -1, 1) * np.sign(out["ema_gap_20_50"])
        usdinr = macro.get("usd_inr", 83.0)
        out["macro_usdinr_mom"] = np.clip((usdinr - 83) / 2, -1, 1) * np.sign(out["ret_5"])

        # Mood interaction features
        if mood:
            mood_score = mood.get("score", 0.0)
            mood_label = mood.get("label", "NEUTRAL")
            out["mood_score"] = mood_score
            out["mood_label_enc"] = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(mood_label, 0)
            out["mood_momentum_align"] = mood_score * np.sign(out["ret_14"]) if "ret_14" in out.columns else mood_score * np.sign(out["ret_10"])
            mf = mood.get("features", {})
            out["mood_vol_align"] = mf.get("D_ATR_RATIO", 0) * out["atr_pct"]
            # Kronos alignment
            out["kronos_mood_align"] = np.sign(kronos_feats.get("kronos_proj_return", 0)) * mood_score if kronos_feats else 0
            out["kronos_macro_align"] = np.sign(kronos_feats.get("kronos_proj_return", 0)) * np.sign(nifty_ret14) if kronos_feats else 0
        else:
            out["mood_score"] = 0.0
            out["mood_label_enc"] = 0
            out["mood_momentum_align"] = 0.0
            out["mood_vol_align"] = 0.0
            out["kronos_mood_align"] = 0.0
            out["kronos_macro_align"] = 0.0

        return out.dropna(subset=["ret_1", "ret_5"]).reset_index(drop=True)


class DailyStackingEnsemble:
    def __init__(self, target_horizon: int = 10):
        self.lgbm = None; self.et = None; self.rf = None; self.xgb = None; self.meta = None
        self.best_params = {}; self.trained = False
        self.feature_names = DailyFeatureEngine.FEATURES
        self.target_horizon = target_horizon

    def train(self, df: pd.DataFrame, n_trials: int = 30) -> float:
        close = df['close']
        # Multi-horizon target: majority direction over horizon
        future_rets = pd.DataFrame()
        for h in [5, 10, 20]:
            future_rets[f'ret_{h}'] = (close.shift(-h) / close - 1)
        # Target: majority of horizons positive
        future_rets['majority_up'] = (future_rets > 0).sum(axis=1) >= 2
        df['TARGET'] = future_rets['majority_up'].astype(int)
        
        valid = df.iloc[:-20].dropna()  # Leave buffer for horizon
        if len(valid) < 50: return 0.5

        for col in self.feature_names:
            if col not in valid.columns: valid[col] = 0.0

        X = valid[self.feature_names]; y = valid['TARGET']
        cv = TimeSeriesSplit(n_splits=3)
        pos_w = (y==0).sum() / max(1, (y==1).sum())

        def obj(trial):
            p = {"n_estimators": trial.suggest_int("n_estimators", 100, 400),
                 "num_leaves": trial.suggest_int("num_leaves", 15, 80),
                 "max_depth": trial.suggest_int("max_depth", 3, 8),
                 "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1),
                 "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 "min_child_samples": trial.suggest_int("min_child_samples", 15, 40),
                 "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5),
                 "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
                 "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 3.0),
                 "min_split_gain": trial.suggest_float("min_split_gain", 0, 0.1),
                 "min_child_weight": trial.suggest_float("min_child_weight", 1, 10)}
            scores = []
            for tr, te in cv.split(X):
                Xtr, Xte = X.iloc[tr], X.iloc[te]
                ytr, yte = y.iloc[tr], y.iloc[te]
                if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2: continue
                m = LGBMClassifier(objective="binary", verbosity=-1, random_state=42, n_jobs=-1, class_weight='balanced', **p)
                m.fit(Xtr, ytr)
                scores.append(roc_auc_score(yte, m.predict_proba(Xte)[:,1]))
            return np.mean(scores) if scores else 0.5

        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        self.best_params = study.best_trial.params

        self.lgbm = LGBMClassifier(objective="binary", verbosity=-1, random_state=42, n_jobs=-1, class_weight='balanced', **self.best_params)
        self.lgbm.fit(X, y)
        
        # Simpler base models for stability
        self.et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=10, max_features=0.7, random_state=42, n_jobs=-1, class_weight='balanced').fit(X, y)
        self.rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=12, max_features=0.65, random_state=42, n_jobs=-1, class_weight='balanced').fit(X, y)
        self.xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, scale_pos_weight=pos_w).fit(X, y)

        oof = np.zeros((len(X), 4))
        for tr, te in cv.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]; ytr, yte = y.iloc[tr], y.iloc[te]
            for i, m in enumerate([self.lgbm, self.et, self.rf, self.xgb]):
                oof[te, i] = m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        
        self.meta = LogisticRegression(random_state=42, max_iter=1000, C=0.5, class_weight='balanced').fit(oof, y)
        meta_auc = roc_auc_score(y, self.meta.predict_proba(oof)[:,1])
        print(f"[DAILY ENSEMBLE] Meta-learner ROC AUC: {meta_auc:.4f}")

        # Isotonic regression calibration
        self.calibrated_meta = CalibratedClassifierCV(self.meta, method='isotonic', cv=3)
        self.calibrated_meta.fit(oof, y)
        cal_auc = roc_auc_score(y, self.calibrated_meta.predict_proba(oof)[:,1])
        print(f"[DAILY ENSEMBLE] Calibrated Meta ROC AUC: {cal_auc:.4f}")

        # Feature importance analysis
        self._compute_feature_importance(X, y)

        self.trained = True
        return study.best_value

    def _compute_feature_importance(self, X, y):
        """Compute feature importance across all base models"""
        importances = np.zeros(len(self.feature_names))

        # LGBM feature importance
        lgbm_imp = self.lgbm.feature_importances_
        importances += lgbm_imp / lgbm_imp.sum()

        # ExtraTrees feature importance
        et_imp = self.et.feature_importances_
        importances += et_imp / et_imp.sum()

        # RandomForest feature importance
        rf_imp = self.rf.feature_importances_
        importances += rf_imp / rf_imp.sum()

        # XGBoost feature importance
        xgb_imp = self.xgb.feature_importances_
        importances += xgb_imp / xgb_imp.sum()

        # Average across models
        importances /= 4.0

        # Store feature importance
        self.feature_importance = dict(zip(self.feature_names, importances))

        # Identify top and bottom features
        sorted_imp = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_imp[:10]
        bottom_10 = sorted_imp[-10:]

        print(f"\n[FEATURE IMPORTANCE] Top 10 features:")
        for feat, imp in top_10:
            print(f"  {feat:<25} {imp:.4f}")

        print(f"\n[FEATURE IMPORTANCE] Bottom 10 features (candidates for pruning):")
        for feat, imp in bottom_10:
            print(f"  {feat:<25} {imp:.4f}")

        # Return features to prune (importance < 0.01)
        self.features_to_prune = [feat for feat, imp in self.feature_importance.items() if imp < 0.01]
        print(f"\n[FEATURE PRUNING] {len(self.features_to_prune)} features with importance < 0.01")
        if self.features_to_prune:
            print(f"  Features to prune: {self.features_to_prune}")

    def predict_proba(self, df: pd.DataFrame) -> float:
        for col in self.feature_names:
            if col not in df.columns: df[col] = 0.0
        X = df[self.feature_names].iloc[-1:]
        base = np.column_stack([m.predict_proba(X)[:,1] for m in [self.lgbm, self.et, self.rf, self.xgb]])
        prob = self.calibrated_meta.predict_proba(base)[:,1]
        return float(prob[0])


class IntradayExecutionEngine:
    def __init__(self):
        self.streak_flip = 3
        self.miss_dir = None; self.miss_count = 0

    def analyze(self, df: pd.DataFrame) -> dict:
        if len(df) < 6: return {"vwap": "NEU", "streak": "NEU", "count": 0, "flip": False}
        close, high, low = df['close'], df['high'], df['low']
        vol = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
        typical = (high + low + close) / 3
        vwap = (typical * vol.replace(0,1)).cumsum() / vol.replace(0,1).cumsum()
        vwap_dir = "UP" if float(vwap.iloc[-1]) > float(vwap.iloc[-6]) else "DOWN"
        rets = close.diff().iloc[-5:]
        dirs = ["UP" if r >= 0 else "DOWN" for r in rets.dropna()]
        streak_dir = dirs[-1] if dirs else "NEU"
        streak_count = sum(1 for d in reversed(dirs) if d == streak_dir)
        return {"vwap": vwap_dir, "streak": streak_dir, "count": streak_count, "flip": streak_count >= self.streak_flip}

    def update_miss(self, pred, actual, tradeable):
        if not tradeable: return
        if pred != actual:
            if self.miss_dir == actual: self.miss_count += 1
            else: self.miss_dir = actual; self.miss_count = 1
        else: self.miss_dir = None; self.miss_count = 0

    @property
    def flip_active(self): return self.miss_count >= self.streak_flip
    @property
    def flipped_dir(self): return self.miss_dir


class DailyKronosPipeline:
    def __init__(self, symbol: str, horizon: int = 10):
        self.symbol = symbol; self.horizon = horizon
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[LOAD] Kronos on {self.device}...")
        tokenizer_kwargs = {"token": get_hf_token()} if get_hf_token() else {}
        self.tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME, **tokenizer_kwargs)
        self.model = Kronos.from_pretrained(KRONOS_MODEL_NAME, **tokenizer_kwargs)
        self.predictor = KronosPredictor(self.model, self.tokenizer, device=self.device, max_context=512)

        self.ml = DailyStackingEnsemble()
        self.mood_engine = MarketMoodEngine()
        self.execution = IntradayExecutionEngine()
        self.macro = fetch_macro_data()

        self.mood = None; self.trained = False
        self._daily_cache = None
        self._context_cache = None

    def fetch_daily(self, symbol: str, period: str = "3y") -> pd.DataFrame:
        if self._daily_cache is not None and symbol == self.symbol:
            return self._daily_cache.copy()
        
        ticker = yf.Ticker(symbol)
        for attempt in range(3):
            try:
                raw = ticker.history(period=period, interval="1d", auto_adjust=False)
                if not raw.empty:
                    break
            except Exception as e:
                print(f"[WARN] Attempt {attempt+1} failed for {symbol}: {e}")
        if raw.empty: raise RuntimeError(f"No daily data for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = [c[0] for c in raw.columns]
        raw = raw.reset_index()
        dt = "Datetime" if "Datetime" in raw.columns else "Date"
        raw["timestamps"] = pd.to_datetime(raw[dt]).dt.tz_localize(None)
        raw = raw.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df = raw[["timestamps","open","high","low","close","volume"]].dropna().sort_values("timestamps").reset_index(drop=True)
        if symbol == self.symbol:
            self._daily_cache = df.copy()
        return df

    def fetch_context(self, period: str = "3y") -> dict:
        if self._context_cache is not None:
            return self._context_cache
        
        ctx = {}
        for sym in MARKET_CONTEXT_SYMBOLS:
            try: 
                ctx[sym] = self.fetch_daily(sym, period=period)
            except Exception as e:
                print(f"[WARN] Context fetch failed for {sym}: {e}")
                ctx[sym] = pd.DataFrame()
        self._context_cache = ctx
        return ctx

    def align_context(self, df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
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

    def get_kronos_forecast(self, df: pd.DataFrame) -> dict:
        lookback = min(len(df), KRONOS_DAILY_LOOKBACK)
        x_df = df.iloc[-lookback:][["open","high","low","close"]].reset_index(drop=True)
        x_ts = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)
        last_date = x_ts.iloc[-1]
        y_ts = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=KRONOS_DAILY_PRED_LEN))
        pred = self.predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                      pred_len=KRONOS_DAILY_PRED_LEN, T=0.7, top_p=0.9, sample_count=1, verbose=False)
        last_c = float(x_df["close"].iloc[-1]); proj_c = float(pred["close"].iloc[-1])
        return {"kronos_proj_return": ((proj_c - last_c) / last_c) * 100,
                "kronos_proj_direction": 1.0 if proj_c >= last_c else 0.0,
                "kronos_proj_range_pct": ((pred["high"].max() - pred["low"].min()) / last_c) * 100,
                "kronos_proj_std_pct": float(pred["close"].std() / last_c * 100),
                "kronos_confidence": min(100, max(50, 50 + abs((proj_c - last_c) / last_c) * 1000))}

    def train(self, period: str = "3y"):
        print(f"[DATA] Fetching daily data for {self.symbol}...")
        df = self.fetch_daily(self.symbol, period=period)
        ctx = self.fetch_context(period=period)
        df = self.align_context(df, ctx)

        print(f"[MOOD] Computing market mood...")
        self.mood = self.mood_engine.compute(self.symbol)

        print(f"[KRONOS] Generating Kronos features on full history...")
        kronos_f = self.get_kronos_forecast(df)

        print(f"[FEATURES] Generating daily features...")
        df_feat = DailyFeatureEngine.generate(df, kronos_feats=kronos_f, macro=self.macro, mood=self.mood)

        print(f"[TRAIN] Training stacking ensemble ({len(df_feat)} samples)...")
        self.ml.train(df_feat, n_trials=50)
        self.trained = True
    def predict(self) -> dict:
                if not self.trained: raise RuntimeError("Call train() first")

                df = self.fetch_daily(self.symbol, period="3y")
                ctx = self.fetch_context(period="3y")
                df = self.align_context(df, ctx)

                self.mood = self.mood_engine.compute(self.symbol)
                kronos_f = self.get_kronos_forecast(df)

                df_feat = DailyFeatureEngine.generate(df, kronos_feats=kronos_f, macro=self.macro, mood=self.mood)
                ml_prob = float(self.ml.predict_proba(df_feat))

                kronos_dir = "UP" if kronos_f["kronos_proj_return"] >= 0 else "DOWN"
                ml_dir = "UP" if ml_prob >= 0.5 else "DOWN"

                mood_score = self.mood["score"]
                kronos_conf = kronos_f["kronos_confidence"]
                ml_conf = max(ml_prob, 1 - ml_prob)

                models_agree = (kronos_dir == ml_dir)

                ml_conf_pct = max(ml_prob, 1 - ml_prob) * 100

                # IMPROVED Decision logic (lower thresholds)
                MOOD_OPPOSE_THRESHOLD = 0.4
                
                kronos_mood_opposed = (self.mood["score"] < -MOOD_OPPOSE_THRESHOLD and kronos_dir == "UP") or \
                                     (self.mood["score"] > MOOD_OPPOSE_THRESHOLD and kronos_dir == "DOWN")
                ml_mood_opposed = (self.mood["score"] < -MOOD_OPPOSE_THRESHOLD and ml_dir == "UP") or \
                                 (self.mood["score"] > MOOD_OPPOSE_THRESHOLD and ml_dir == "DOWN")
                
                ml_conf = max(ml_prob, 1 - ml_prob)
                kronos_conf_pct = kronos_f["kronos_confidence"]
                
                if kronos_dir == ml_dir:
                    # Both agree - lower threshold to 0.50
                    if not kronos_mood_opposed and ml_conf >= 0.50:
                        direction = kronos_dir
                        confidence = round(ml_conf * 100, 1)
                        tradeable = True
                        gate = "KRONOS+ML CONFIRMED"
                    elif ml_conf >= 0.48:
                        direction = kronos_dir
                        confidence = round(ml_conf * 100, 1)
                        tradeable = True
                        gate = "WEAK CONFIRM"
                    else:
                        direction = kronos_dir
                        confidence = round(kronos_conf_pct, 1)
                        tradeable = False
                        gate = "NO TRADE"
                else:
                    # Disagree - check which has higher confidence
                    # IMPROVED: Lower override threshold to 0.55 and handle ML DOWN case
                    if ml_conf >= 0.55 and not ml_mood_opposed and kronos_conf_pct < ml_conf * 100:
                        # ML strongly confident and mood supports ML
                        direction = ml_dir
                        confidence = round(ml_conf * 100, 1)
                        tradeable = True
                        gate = "ML OVERRIDE"
                    elif kronos_conf_pct >= 60 and kronos_conf_pct > ml_conf * 100 and not kronos_mood_opposed:
                        direction = kronos_dir
                        confidence = round(kronos_conf_pct, 1)
                        tradeable = True
                        gate = "KRONOS STRONG"
                    else:
                        # Default to higher confidence
                        if ml_conf * 100 > kronos_conf_pct:
                            direction = ml_dir
                            confidence = round(ml_conf * 100, 1)
                        else:
                            direction = kronos_dir
                            confidence = round(kronos_conf_pct, 1)
                        tradeable = False
                        gate = "NO TRADE"

                return {"symbol": self.symbol, "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "horizon_days": self.horizon,
                    "kronos_dir": kronos_dir, "kronos_ret": round(kronos_f["kronos_proj_return"], 2),
                    "ml_prob": round(ml_prob, 3), "ml_dir": ml_dir,
                    "mood": self.mood["label"], "mood_score": round(self.mood["score"], 3),
                    "direction": direction, "confidence": confidence,
                    "tradeable": tradeable, "gate": gate,
                    "kronos_conf": round(kronos_f["kronos_confidence"], 1)}


def main():
    parser = argparse.ArgumentParser(description="Daily Kronos + ML Confirmation Pipeline")
    parser.add_argument("--symbol", default="TCS.NS", help="NSE ticker")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon in days")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--predict", action="store_true", help="Generate prediction")
    args = parser.parse_args()

    pipe = DailyKronosPipeline(args.symbol, horizon=args.horizon)

    if args.train:
        pipe.train(period="3y")
        os.makedirs("models", exist_ok=True)
        torch.save(pipe.model.state_dict(), f"models/kronos_{args.symbol}.pt")
        joblib.dump(pipe.ml, f"models/ml_{args.symbol}.pkl")
        print("[SAVE] Models saved to models/")

    if args.predict:
        if not pipe.trained:
            try:
                pipe.ml = joblib.load(f"models/ml_{args.symbol}.pkl")
                pipe.trained = True
            except:
                print("[WARN] No trained model found, training now...")
                pipe.train(period="3y")
        pred = pipe.predict()
        print("\n" + "="*60)
        print(f"DAILY PREDICTION — {pred['symbol']} ({pred['horizon_days']}D)")
        print("="*60)
        print(f"Date           : {pred['date']}")
        print(f"Kronos Dir     : {pred['kronos_dir']} ({pred['kronos_ret']:+.2f}%)")
        print(f"ML Prob        : {pred['ml_prob']:.1%} ({pred['ml_dir']})")
        print(f"Mood           : {pred['mood']} (Score: {pred['mood_score']:+.3f})")
        print(f"Direction      : {pred['direction']}")
        print(f"Confidence     : {pred['confidence']:.1f}%")
        print(f"Tradeable      : {'YES' if pred['tradeable'] else 'NO'} ({pred['gate']})")
        print(f"Kronos Conf    : {pred['kronos_conf']:.1f}%")
        print("="*60)

if __name__ == "__main__":
    main()