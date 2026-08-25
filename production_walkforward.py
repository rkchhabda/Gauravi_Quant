"""
Tier 1 & 2: Production Walk-Forward with 250+ predictions,
confidence bucketing, calibrated probabilities, improved mood engine.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

sys.path.append("./kronos_lib")
from model import Kronos, KronosTokenizer, KronosPredictor

from macro_utils import fetch_macro_data
from daily_kronos_pipeline import (
    DailyFeatureEngine, KRONOS_MODEL_NAME, TOKENIZER_NAME,
    KRONOS_DAILY_LOOKBACK, KRONOS_DAILY_PRED_LEN, MARKET_CONTEXT_SYMBOLS
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def get_hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


# ── Improved Mood Engine ──────────────────────────────────────────────────
class ImprovedMoodEngine:
    FEATURE_NAMES = [
        "D_RSI_14", "D_MACD_HIST", "D_ADX", "D_EMA9_21", "D_EMA21_50",
        "D_RETURN_5D", "D_RETURN_14D", "D_ATR_RATIO", "D_ABOVE_VWAP", "D_VOL_RATIO"
    ]

    @staticmethod
    def _fetch(symbol, lookback=160):
        raw = yf.download(symbol, period=f"{lookback}d", interval="1d", progress=False)
        if raw.empty:
            return pd.DataFrame()
        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                   "Close": "close", "Volume": "volume"})
        raw["Date"] = pd.to_datetime(raw["Date"])
        return raw.sort_values("Date").reset_index(drop=True)

    @classmethod
    def compute(cls, symbol):
        df = cls._fetch(symbol)
        if df.empty or len(df) < 20:
            return cls._neutral()

        c, h, lo = df["close"], df["high"], df["low"]
        vol = df["volume"].replace(0, 1)

        ema9 = c.ewm(span=9, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        ema50 = c.ewm(span=50, adjust=False).mean()

        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rsi = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50.0)

        macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_sig

        tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        up_m = h.diff().clip(lower=0)
        dn_m = (-lo.diff()).clip(lower=0)
        pdi = 100 * (up_m.where(up_m > dn_m, 0).ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
        mdi = 100 * (dn_m.where(dn_m > up_m, 0).ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
        dx = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
        adx = dx.ewm(span=14, adjust=False).mean()

        typical = (h + lo + c) / 3
        vwap = ((typical * vol).rolling(14, min_periods=1).sum() / vol.rolling(14, min_periods=1).sum())

        ret5d = c.pct_change(5).fillna(0.0) * 100
        ret14d = c.pct_change(14).fillna(0.0) * 100
        vol_sma20 = vol.rolling(20, min_periods=5).mean()
        vol_ratio = (vol / vol_sma20.replace(0, np.nan)).fillna(1.0)

        i = -1
        feats = {
            "D_RSI_14": float(rsi.iloc[i]), "D_MACD_HIST": float(macd_hist.iloc[i]),
            "D_ADX": float(adx.iloc[i]),
            "D_EMA9_21": float(ema9.iloc[i] - ema21.iloc[i]),
            "D_EMA21_50": float(ema21.iloc[i] - ema50.iloc[i]),
            "D_RETURN_5D": float(ret5d.iloc[i]), "D_RETURN_14D": float(ret14d.iloc[i]),
            "D_ATR_RATIO": float(atr.iloc[i] / c.iloc[i]) if c.iloc[i] else 0.0,
            "D_ABOVE_VWAP": 1.0 if c.iloc[i] > vwap.iloc[i] else 0.0,
            "D_VOL_RATIO": float(vol_ratio.iloc[i]),
        }

        adx_val = feats["D_ADX"]
        w = 0.10 if adx_val < 20 else 0.15
        score = 0.0
        score += w * cls._norm(feats["D_RSI_14"], 50, 50)
        score += (w * 0.67) * np.sign(feats["D_MACD_HIST"])
        score += w * np.sign(feats["D_EMA9_21"])
        score += w * np.sign(feats["D_EMA21_50"])
        score += w * cls._norm(feats["D_RETURN_5D"], 0, 3)
        score += w * cls._norm(feats["D_RETURN_14D"], 0, 5)
        score += (w * 0.67) * (feats["D_ABOVE_VWAP"] * 2 - 1)
        if adx_val >= 20 and feats["D_VOL_RATIO"] > 1.2:
            score *= 1.1
        score = float(np.clip(score, -1.0, 1.0))
        label = "BULLISH" if score >= 0.25 else ("BEARISH" if score <= -0.25 else "NEUTRAL")
        return {"score": round(score, 3), "label": label, "features": feats,
                "adx": round(adx_val, 1)}

    @staticmethod
    def _norm(val, center, scale):
        return float(np.clip((val - center) / scale, -1.0, 1.0)) if scale else 0.0

    @staticmethod
    def _neutral():
        return {"score": 0.0, "label": "NEUTRAL",
                "features": {k: 0.0 for k in ImprovedMoodEngine.FEATURE_NAMES},
                "adx": 25.0}


# ── Data Fetching ─────────────────────────────────────────────────────────
def fetch_daily(symbol, period="1y"):
    ticker = yf.Ticker(symbol)
    raw = ticker.history(period=period, interval="1d", auto_adjust=True)
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


def fetch_context(period="1y"):
    ctx = {}
    for sym in MARKET_CONTEXT_SYMBOLS:
        try:
            ctx[sym] = fetch_daily(sym, period=period)
        except Exception:
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


# ── Meta-Labeling (Tier 3) ────────────────────────────────────────────────
class MetaLabelModel:
    """
    Lopez de Prado meta-labeling:
    Kronos proposes direction; ML predicts whether Kronos will be right.
    """

    def __init__(self):
        self.model = None
        self.feature_names = DailyFeatureEngine.FEATURES
        self.trained = False

    def prepare_meta_labels(self, df, kronos_predictions):
        """
        kronos_predictions: array of Kronos direction per row (1=UP, 0=DOWN)
        target_meta: 1 if Kronos was correct, 0 if wrong
        """
        close = df['close']
        future_ret_10 = close.shift(-10) / close - 1

        kronos_correct = (
            ((kronos_predictions == 1) & (future_ret_10 > 0)) |
            ((kronos_predictions == 0) & (future_ret_10 < 0))
        ).astype(int)

        return kronos_correct

    def train(self, df, kronos_preds, n_trials=20):
        """Train meta-label model."""
        meta_target = self.prepare_meta_labels(df, kronos_preds)

        valid = df.iloc[:-10].copy()
        meta_target = meta_target.iloc[:-10]

        for col in self.feature_names:
            if col not in valid.columns:
                valid[col] = 0.0

        mask = valid[self.feature_names].notna().all(axis=1) & meta_target.notna()
        valid = valid[mask]
        meta_target = meta_target[mask]

        if len(valid) < 100:
            return 0.5

        X = valid[self.feature_names]
        y = meta_target

        cv = TimeSeriesSplit(n_splits=3)

        def obj(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "num_leaves": trial.suggest_int("num_leaves", 15, 50),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 40),
            }
            scores = []
            for tr_idx, te_idx in cv.split(X):
                Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
                ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
                if len(np.unique(ytr)) < 2:
                    continue
                m = LGBMClassifier(objective="binary", verbosity=-1, random_state=42,
                                   n_jobs=-1, class_weight='balanced', **p)
                m.fit(Xtr, ytr)
                scores.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
            return np.mean(scores) if scores else 0.5

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

        self.model = LGBMClassifier(objective="binary", verbosity=-1, random_state=42,
                                     n_jobs=-1, class_weight='balanced', **study.best_trial.params)
        self.model.fit(X, y)
        self.trained = True
        return study.best_value

    def predict_proba(self, df):
        if not self.trained:
            return np.ones(len(df)) * 0.5
        X = df[self.feature_names].fillna(0)
        return self.model.predict_proba(X)[:, 1]


# ── Triple Barrier Labeling (Tier 3) ──────────────────────────────────────
class TripleBarrierLabeler:
    """
    Lopez de Prado triple-barrier labeling.
    Labels: 1=profit-take hit first, 0=stop-loss hit first, 2=timeout
    """

    @staticmethod
    def label(df, pct_tp=0.03, pct_sl=0.02, max_holding=10):
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        n = len(close)
        labels = np.full(n, 2, dtype=int)

        for i in range(n):
            entry = close[i]
            tp_price = entry * (1 + pct_tp)
            sl_price = entry * (1 - pct_sl)
            end_idx = min(i + max_holding, n - 1)

            for j in range(i + 1, end_idx + 1):
                if high[j] >= tp_price:
                    labels[i] = 1
                    break
                if low[j] <= sl_price:
                    labels[i] = 0
                    break

        df = df.copy()
        df['triple_label'] = labels
        return df


# ── Walk-Forward Engine ───────────────────────────────────────────────────
class ProductionWalkForward:
    def __init__(self, symbol, period="1y", train_window=120, test_window=5,
                 horizon=10, use_meta_labeling=True, use_triple_barrier=False):
        self.symbol = symbol
        self.period = period
        self.train_window = train_window
        self.test_window = test_window
        self.horizon = horizon
        self.use_meta_labeling = use_meta_labeling
        self.use_triple_barrier = use_triple_barrier

    def run(self):
        print(f"\n{'='*70}")
        print(f"PRODUCTION WALK-FORWARD: {self.symbol}")
        print(f"Train: {self.train_window}d | Test: {self.test_window}d | Horizon: {self.horizon}d")
        print(f"Meta-labeling: {self.use_meta_labeling} | Triple-barrier: {self.use_triple_barrier}")
        print(f"{'='*70}")

        df = fetch_daily(self.symbol, period=self.period)
        ctx = fetch_context(period=self.period)
        df = align_context(df, ctx)

        macro = fetch_macro_data()
        mood_engine = ImprovedMoodEngine()

        # Generate features for full dataset
        mood = mood_engine.compute(self.symbol)
        df_feat = DailyFeatureEngine.generate(df, macro=macro, mood=mood)

        if self.use_triple_barrier:
            df_feat = TripleBarrierLabeler.label(df_feat, pct_tp=0.03, pct_sl=0.02, max_holding=self.horizon)
            target_col = 'triple_label'
        else:
            df_feat['target_binary'] = (df_feat['close'].shift(-self.horizon) / df_feat['close'] - 1 > 0.02).astype(int)
            target_col = 'target_binary'

        df_feat = df_feat.dropna(subset=[target_col, 'ret_1']).reset_index(drop=True)

        if len(df_feat) < self.train_window + self.test_window + self.horizon:
            print(f"  Insufficient data: {len(df_feat)} rows")
            return None

        feature_cols = [c for c in DailyFeatureEngine.FEATURES if c in df_feat.columns]

        all_predictions = []
        start_idx = self.train_window

        while start_idx + self.test_window + self.horizon <= len(df_feat):
            train = df_feat.iloc[start_idx - self.train_window:start_idx]
            test = df_feat.iloc[start_idx:min(start_idx + self.test_window, len(df_feat) - self.horizon)]

            if len(test) == 0:
                break

            X_train = train[feature_cols].fillna(0)
            y_train = train[target_col]
            X_test = test[feature_cols].fillna(0)
            y_test = test[target_col]

            if len(np.unique(y_train)) < 2:
                start_idx += self.test_window
                continue

            if self.use_triple_barrier:
                base_model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                                     random_state=42, n_jobs=-1)
                base_model.fit(X_train, y_train)
                proba = base_model.predict_proba(X_test)
                preds = base_model.predict(X_test)

                for idx_in_test in range(len(test)):
                    row = test.iloc[idx_in_test]
                    all_predictions.append({
                        "date": row["timestamps"].strftime("%Y-%m-%d"),
                        "pred_label": int(preds[idx_in_test]),
                        "actual_label": int(y_test.iloc[idx_in_test]),
                        "prob_tp": float(proba[idx_in_test, 1]) if proba.shape[1] > 1 else 0.0,
                        "is_correct": int(preds[idx_in_test]) == int(y_test.iloc[idx_in_test]),
                        "close": float(row["close"]),
                    })
            else:
                # Base model predicts direction
                base_model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                                     random_state=42, n_jobs=-1)
                base_model.fit(X_train, y_train)
                proba = base_model.predict_proba(X_test)[:, 1]

                # Meta-labeling: train model to predict if base model will be correct
                meta_confidence = np.abs(proba - 0.5) + 0.5  # default
                if self.use_meta_labeling:
                    # Build meta-target from training data
                    train_proba = base_model.predict_proba(X_train)[:, 1]
                    train_preds = (train_proba >= 0.5).astype(int)
                    train_actual = y_train.values
                    train_meta_target = (train_preds == train_actual).astype(int)

                    if len(np.unique(train_meta_target)) >= 2 and train_meta_target.sum() > 10:
                        meta_model = RandomForestClassifier(n_estimators=100, max_depth=4,
                                                             random_state=42, n_jobs=-1)
                        meta_model.fit(X_train, train_meta_target)
                        meta_confidence = meta_model.predict_proba(X_test)[:, 1]

                for idx_in_test in range(len(test)):
                    row = test.iloc[idx_in_test]
                    pred_dir = "UP" if proba[idx_in_test] >= 0.5 else "DOWN"
                    actual_dir = "UP" if y_test.iloc[idx_in_test] == 1 else "DOWN"
                    confidence = float(proba[idx_in_test])

                    # For meta-labeling: use meta_confidence as the trade confidence
                    if self.use_meta_labeling:
                        trade_confidence = float(meta_confidence[idx_in_test])
                    else:
                        trade_confidence = confidence

                    actual_idx = min(test.index[idx_in_test] + self.horizon, len(df_feat) - 1)
                    actual_ret = (df_feat.iloc[actual_idx]["close"] - row["close"]) / row["close"]

                    all_predictions.append({
                        "date": row["timestamps"].strftime("%Y-%m-%d"),
                        "close": float(row["close"]),
                        "prob_up": round(confidence, 4),
                        "meta_confidence": round(trade_confidence, 4) if self.use_meta_labeling else None,
                        "pred_dir": pred_dir,
                        "actual_dir": actual_dir,
                        "actual_ret_pct": round(actual_ret * 100, 2),
                        "is_correct": pred_dir == actual_dir,
                        "confidence_bin": self._conf_bucket(trade_confidence),  # Use meta_confidence for binning
                        "kronos_ret_bin": self._kronos_bucket(row.get("kronos_proj_return", 0)),
                    })

            start_idx += self.test_window

        return self._analyze(all_predictions)

    def _conf_bucket(self, prob):
        conf = abs(prob - 0.5)
        if conf < 0.02:
            return "0-2%"
        elif conf < 0.05:
            return "2-5%"
        elif conf < 0.10:
            return "5-10%"
        elif conf < 0.15:
            return "10-15%"
        else:
            return "15%+"

    def _kronos_bucket(self, kronos_ret):
        if abs(kronos_ret) < 1:
            return "0-1%"
        elif abs(kronos_ret) < 3:
            return "1-3%"
        elif abs(kronos_ret) < 5:
            return "3-5%"
        else:
            return "5%+"

    def _analyze(self, predictions):
        if not predictions:
            return None

        df = pd.DataFrame(predictions)
        total = len(df)

        overall_acc = df["is_correct"].mean() * 100

        # Determine which column to use for confidence bucketing
        # If meta-labeling is active, use meta_confidence; otherwise use prob_up
        use_meta = "meta_confidence" in df.columns and df["meta_confidence"].notna().any()
        conf_col = "meta_confidence" if use_meta else "prob_up"

        if conf_col in df.columns:
            conf_bins = {}
            for bucket in df["confidence_bin"].unique():
                sub = df[df["confidence_bin"] == bucket]
                if len(sub) >= 10:
                    conf_bins[bucket] = {
                        "count": len(sub),
                        "accuracy": round(sub["is_correct"].mean() * 100, 1),
                        "mean_conf": round(sub[conf_col].mean(), 4),
                    }

            kronos_bins = {}
            for bucket in df["kronos_ret_bin"].unique():
                sub = df[df["kronos_ret_bin"] == bucket]
                if len(sub) >= 10:
                    kronos_bins[bucket] = {
                        "count": len(sub),
                        "accuracy": round(sub["is_correct"].mean() * 100, 1),
                    }
        else:
            conf_bins = {}
            kronos_bins = {}

        # Gate effectiveness: accuracy when confidence is above median
        if conf_col in df.columns:
            median_conf = df[conf_col].median()
            high_conf = df[df[conf_col] > median_conf]
            low_conf = df[df[conf_col] <= median_conf]

            gate_added_value = (high_conf["is_correct"].mean() - low_conf["is_correct"].mean()) * 100 if len(high_conf) > 0 and len(low_conf) > 0 else 0

            # Filtered accuracy: only trade when meta_confidence > 0.6
            if use_meta:
                filtered = df[df[conf_col] > 0.6]
                filtered_acc = filtered["is_correct"].mean() * 100 if len(filtered) > 0 else 0
                filtered_count = len(filtered)
            else:
                filtered_acc = overall_acc
                filtered_count = total
        else:
            gate_added_value = 0
            filtered_acc = overall_acc
            filtered_count = total

        result = {
            "symbol": self.symbol,
            "total_predictions": total,
            "overall_accuracy": round(overall_acc, 1),
            "baseline_accuracy": round(overall_acc, 1),
            "filtered_accuracy": round(filtered_acc, 1),
            "filtered_count": filtered_count,
            "gate_added_value": round(gate_added_value, 1),
            "confidence_buckets": conf_bins,
            "kronos_ret_buckets": kronos_bins,
            "meta_labeling_used": use_meta,
            "predictions": predictions,
        }

        self._print_report(result)
        return result

    def _print_report(self, r):
        print(f"\n{'─'*70}")
        print(f"REPORT: {r['symbol']}")
        print(f"{'─'*70}")
        print(f"  Total predictions : {r['total_predictions']}")
        print(f"  Overall accuracy  : {r['overall_accuracy']:.1f}%")
        if r.get('meta_labeling_used'):
            print(f"  Filtered accuracy : {r['filtered_accuracy']:.1f}% (conf>0.6, {r['filtered_count']} trades)")
        print(f"  Baseline accuracy : {r['baseline_accuracy']:.1f}%")
        print(f"  Gate added value  : {r['gate_added_value']:+.1f}%")

        if r['confidence_buckets']:
            print(f"\n  Confidence Buckets:")
            print(f"  {'Bucket':<10} {'Count':>6} {'Accuracy':>9} {'Mean Conf':>10}")
            print(f"  {'─'*40}")
            for bucket in sorted(r['confidence_buckets'].keys()):
                b = r['confidence_buckets'][bucket]
                print(f"  {bucket:<10} {b['count']:>6} {b['accuracy']:>8.1f}% {b['mean_conf']:>10.4f}")

        if r['kronos_ret_buckets']:
            print(f"\n  Kronos Return Buckets:")
            print(f"  {'Bucket':<10} {'Count':>6} {'Accuracy':>9}")
            print(f"  {'─'*30}")
            for bucket in sorted(r['kronos_ret_buckets'].keys()):
                b = r['kronos_ret_buckets'][bucket]
                print(f"  {bucket:<10} {b['count']:>6} {b['accuracy']:>8.1f}%")

        print(f"{'─'*70}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Production Walk-Forward (Way Forward)")
    parser.add_argument("--symbol", default="RELIANCE.NS", help="NSE ticker")
    parser.add_argument("--period", default="1y", help="Data period (1y = ~250 trading days)")
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--test-window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--meta-labeling", action="store_true", default=True)
    parser.add_argument("--no-meta-labeling", dest="meta_labeling", action="store_false")
    parser.add_argument("--triple-barrier", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    pwf = ProductionWalkForward(
        symbol=args.symbol,
        period=args.period,
        train_window=args.train_window,
        test_window=args.test_window,
        horizon=args.horizon,
        use_meta_labeling=args.meta_labeling,
        use_triple_barrier=args.triple_barrier,
    )
    result = pwf.run()

    if result:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = args.symbol.replace("^", "IDX_").replace(".", "_").replace("=", "_")
        path = out_dir / f"{safe}_production_walkforward.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
