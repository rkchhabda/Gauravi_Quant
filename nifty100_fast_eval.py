"""
nifty100_fast_eval.py  –  Kronos v2 · Nifty-100 Walk-Forward Evaluator
=======================================================================
Improvements over v1
--------------------
• 50+ features  (ATR regime, order-flow, candle patterns, lagged macro,
  Fibonacci pivots, CMF, Stochastic, Williams %R, OBV momentum, etc.)
• LightGBM + ExtraTrees + RandomForest  ensemble  (top-3 averaged probs)
• Optuna tuning of LightGBM hyper-params per symbol  (30 trials)
• Dynamic per-symbol confidence threshold chosen on validation split
• Train window  default = 500 daily bars  (~2 years)
• Tracks  symbols_above_60 / symbols_above_65 / symbols_above_70
• Saves daily metrics JSON  (outputs/metrics_<date>.json)
• Suppresses all sklearn / lightgbm verbosity
"""
import sys
import json
import time
import os
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── market-context tickers ─────────────────────────────────────────────────────
MARKET_CONTEXT_SYMBOLS = ["^NSEI", "^INDIAVIX", "INR=X", "GC=F"]   # + Gold


# ==============================================================================
# DATA FETCHING
# ==============================================================================
def fetch_daily(symbol: str, period: str = "3y") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    raw = pd.DataFrame()
    try:
        raw = ticker.history(period=period, interval="1d", auto_adjust=True)
    except Exception:
        pass
    if raw.empty:
        try:
            end   = pd.Timestamp.now()
            start = end - pd.DateOffset(years=3)
            raw   = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=True
            )
        except Exception:
            raw = pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    df = raw.reset_index()
    df["timestamps"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.rename(columns={"Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    return df[["timestamps", "open", "high", "low", "close", "volume"]].dropna().sort_values("timestamps").reset_index(drop=True)


def fetch_market_context(period: str = "3y"):
    ctx = {}
    for sym in MARKET_CONTEXT_SYMBOLS:
        try:
            ctx[sym] = fetch_daily(sym, period=period)
        except Exception:
            ctx[sym] = pd.DataFrame()
    return ctx


def align_context(df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    out = df.copy()
    for sym, ctx_df in ctx.items():
        if ctx_df.empty:
            continue
        prefix = sym.replace("^", "").replace("=", "").replace("-", "_").replace("/", "_")
        merged = out.merge(
            ctx_df[["timestamps", "close"]].rename(columns={"close": f"{prefix}_close"}),
            on="timestamps", how="left"
        )
        merged[f"{prefix}_ret_1"] = merged[f"{prefix}_close"].pct_change(1)
        merged[f"{prefix}_ret_5"] = merged[f"{prefix}_close"].pct_change(5)
        merged[f"{prefix}_ret_10"] = merged[f"{prefix}_close"].pct_change(10)
        out = merged.drop(columns=[c for c in merged.columns if c.endswith("_close") and c != "close"], errors="ignore")
    return out


# ==============================================================================
# FEATURE ENGINEERING  (50+ features)
# ==============================================================================
def add_features(df: pd.DataFrame, horizon: int = 5, move_threshold: float = 0.02) -> pd.DataFrame:
    out  = df.copy()
    close  = out["close"]
    high   = out["high"]
    low    = out["low"]
    volume = out["volume"].replace(0, np.nan)

    # ── Returns ───────────────────────────────────────────────────────
    for n in [1, 3, 5, 10, 20]:
        out[f"ret_{n}"] = close.pct_change(n)

    out[f"future_ret_{horizon}d"] = close.shift(-horizon) / close - 1.0
    out["target"] = (out[f"future_ret_{horizon}d"] > move_threshold).astype(int)

    # ── Target label lag (momentum signal) ────────────────────────────
    out["target_lag_1"] = out["target"].shift(1)
    out["target_lag_2"] = out["target"].shift(2)

    # ── Momentum ──────────────────────────────────────────────────────
    for n in [5, 10, 20, 60]:
        out[f"momentum_{n}"] = close / close.shift(n) - 1.0

    # ── EMAs ──────────────────────────────────────────────────────────
    ema5   = close.ewm(span=5,  adjust=False).mean()
    ema10  = close.ewm(span=10, adjust=False).mean()
    ema20  = close.ewm(span=20, adjust=False).mean()
    ema50  = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    ema200 = close.ewm(span=200, min_periods=1, adjust=False).mean()

    out["ema_gap_5_10"]   = ema5  / ema10  - 1.0
    out["ema_gap_10_20"]  = ema10 / ema20  - 1.0
    out["ema_gap_20_50"]  = ema20 / ema50  - 1.0
    out["ema_gap_50_200"] = ema50 / ema200 - 1.0
    out["close_vs_ema10"] = close / ema10  - 1.0
    out["close_vs_ema20"] = close / ema20  - 1.0
    out["close_vs_ema50"] = close / ema50  - 1.0
    out["bullish_ema_alignment"] = ((ema20 > ema50) & (ema50 > ema200)).astype(float)

    # ── RSI(14) ───────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50.0)
    out["rsi_slope_3"] = out["rsi_14"].diff(3)

    # ── Stochastic %K / %D ────────────────────────────────────────────
    low14  = low.rolling(14,  min_periods=3).min()
    high14 = high.rolling(14, min_periods=3).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    out["stoch_k"] = stoch_k.fillna(50.0)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean().fillna(50.0)

    # ── Williams %R ───────────────────────────────────────────────────
    out["williams_r"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)
    out["williams_r"] = out["williams_r"].fillna(-50.0)

    # ── MACD ──────────────────────────────────────────────────────────
    macd           = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal    = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd - macd_signal
    out["macd_slope_3"] = out["macd_hist"].diff(3)

    # ── ATR & ADX ─────────────────────────────────────────────────────
    tr  = pd.concat([high - low,
                     (high - close.shift(1)).abs(),
                     (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=3).mean()
    out["atr_pct"]          = (atr / close).fillna(0.0)
    out["range_pct"]        = (high - low) / close
    out["range_10"]         = out["range_pct"].rolling(10, min_periods=3).mean()
    out["range_ratio_10"]   = out["range_pct"] / out["range_10"].replace(0, np.nan)

    # ATR regime: compare current ATR to 30d rolling mean
    atr_mean_30 = atr.rolling(30, min_periods=5).mean()
    out["atr_regime"]       = (atr / atr_mean_30.replace(0, np.nan)).fillna(1.0)  # >1 = high vol

    up_m = high.diff().clip(lower=0)
    dn_m = (-low.diff()).clip(lower=0)
    pdi  = 100 * (up_m.where(up_m > dn_m, 0).ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
    mdi  = 100 * (dn_m.where(dn_m > up_m, 0).ewm(span=14, adjust=False).mean() / atr.replace(0, np.nan))
    dx   = (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).fillna(0)
    out["adx_14"]           = dx.ewm(span=14, adjust=False).mean().fillna(25.0)
    out["pdi_mdi_diff"]     = (pdi - mdi).fillna(0.0)

    # ── Bollinger Bands ───────────────────────────────────────────────
    roll_std    = close.rolling(20, min_periods=1).std().fillna(1.0)
    bb_middle   = close.rolling(20, min_periods=1).mean()
    bb_upper    = bb_middle + 2 * roll_std
    bb_lower    = bb_middle - 2 * roll_std
    out["dist_bb_upper"]    = (bb_upper - close) / close
    out["dist_bb_lower"]    = (close - bb_lower) / close
    out["bb_width"]         = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)   # squeeze indicator

    # ── Volatility ────────────────────────────────────────────────────
    ret1 = close.pct_change()
    out["volatility_5"]  = ret1.rolling(5,  min_periods=3).std()
    out["volatility_10"] = ret1.rolling(10, min_periods=5).std()
    out["volatility_20"] = ret1.rolling(20, min_periods=8).std()
    vol_mean_30 = out["volatility_20"].rolling(30, min_periods=5).mean()
    out["high_volatility"] = (out["volatility_20"] > vol_mean_30).astype(float)

    # ── Volume / Order-flow ───────────────────────────────────────────
    out["volume_change_1"]  = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    out["volume_ratio_20"]  = volume / volume.rolling(20, min_periods=5).mean()
    # OBV momentum
    obv  = (np.sign(close.diff()) * volume.fillna(0)).cumsum()
    out["obv_slope_5"]      = obv.diff(5) / volume.rolling(5, min_periods=1).mean().replace(0, np.nan)
    # Chaikin Money Flow (20-period)
    mfm  = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv  = mfm * volume.fillna(0)
    out["cmf_20"]           = mfv.rolling(20, min_periods=5).sum() / volume.rolling(20, min_periods=5).sum().replace(0, np.nan)

    # ── Candle patterns ───────────────────────────────────────────────
    body      = (close - out["open"]).abs()
    candle_range = (high - low).replace(0, np.nan)
    out["body_pct"]         = body / candle_range
    out["upper_wick"]       = (high - pd.concat([close, out["open"]], axis=1).max(axis=1)) / candle_range
    out["lower_wick"]       = (pd.concat([close, out["open"]], axis=1).min(axis=1) - low) / candle_range
    out["close_location"]   = (close - low) / candle_range   # 0=bearish, 1=bullish
    out["doji"]             = (body / candle_range < 0.1).astype(float)
    out["hammer"]           = ((out["lower_wick"] > 0.6) & (out["upper_wick"] < 0.2)).astype(float)
    out["shooting_star"]    = ((out["upper_wick"] > 0.6) & (out["lower_wick"] < 0.2)).astype(float)

    # ── Gap analysis ──────────────────────────────────────────────────
    out["gap_up"]           = (out["open"] / close.shift(1) - 1.0).fillna(0.0)
    out["opening_range"]    = (high - low) / out["open"].replace(0, np.nan)

    # ── Fibonacci pivot support/resistance ────────────────────────────
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    prev_close = close.shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3
    out["dist_fib_r1"] = (pivot + 0.382 * (prev_high - prev_low) - close) / close
    out["dist_fib_s1"] = (close - (pivot - 0.382 * (prev_high - prev_low))) / close

    # ── Seasonality ───────────────────────────────────────────────────
    out["day_of_week"]   = out["timestamps"].dt.dayofweek
    out["month"]         = out["timestamps"].dt.month
    out["quarter"]       = out["timestamps"].dt.quarter
    out["week_of_year"]  = out["timestamps"].dt.isocalendar().week.astype(int)

    # ── Risk / R:R ────────────────────────────────────────────────────
    out["stop_loss_pct"]      = out["atr_pct"] * 2.0
    out["risk_reward_ratio"]  = move_threshold / (out["atr_pct"] * 2.0).replace(0, np.nan)

    return out.dropna(subset=[f"future_ret_{horizon}d", "target"]).reset_index(drop=True)


# ── Feature column list ────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Returns
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    # Target lags
    "target_lag_1", "target_lag_2",
    # Momentum
    "momentum_5", "momentum_10", "momentum_20", "momentum_60",
    # EMA ratios / alignment
    "ema_gap_5_10", "ema_gap_10_20", "ema_gap_20_50", "ema_gap_50_200",
    "close_vs_ema10", "close_vs_ema20", "close_vs_ema50", "bullish_ema_alignment",
    # Oscillators
    "rsi_14", "rsi_slope_3", "stoch_k", "stoch_d", "williams_r",
    # MACD
    "macd_hist", "macd_slope_3",
    # ATR / ADX / volatility
    "atr_pct", "range_pct", "range_ratio_10", "atr_regime",
    "adx_14", "pdi_mdi_diff",
    "volatility_5", "volatility_10", "volatility_20", "high_volatility",
    # Bollinger
    "dist_bb_upper", "dist_bb_lower", "bb_width",
    # Volume / order-flow
    "volume_change_1", "volume_ratio_20", "obv_slope_5", "cmf_20",
    # Candle patterns
    "body_pct", "upper_wick", "lower_wick", "close_location",
    "doji", "hammer", "shooting_star",
    # Gap / range
    "gap_up", "opening_range",
    # Fibonacci
    "dist_fib_r1", "dist_fib_s1",
    # Seasonality
    "day_of_week", "month", "quarter", "week_of_year",
    # Risk
    "stop_loss_pct", "risk_reward_ratio",
    # Market context (filled 0 if unavailable)
    "NSEI_ret_1", "NSEI_ret_5", "NSEI_ret_10",
    "INDIAVIX_ret_1", "INDIAVIX_ret_5",
    "INR_X_ret_1", "INR_X_ret_5",
    "GCF_ret_1", "GCF_ret_5",
]


# ==============================================================================
# MODELS
# ==============================================================================
def make_lgbm_model(params: dict, pos_weight: float):
    """Build an LGBMClassifier from Optuna params."""
    return LGBMClassifier(
        objective="binary",
        class_weight={0: 1.0, 1: pos_weight},
        verbosity=-1,
        n_jobs=-1,
        random_state=42,
        **params
    )


def make_base_models(pos_weight: float):
    common = dict(random_state=42, n_jobs=-1)
    return [
        ExtraTreesClassifier(n_estimators=300, min_samples_leaf=6, max_features=0.7,
                             class_weight={0: 1.0, 1: pos_weight}, **common),
        RandomForestClassifier(n_estimators=200, min_samples_leaf=8, max_features=0.65,
                               class_weight={0: 1.0, 1: pos_weight}, **common),
    ]


def tune_lgbm(X_tr, y_tr, pos_weight: float, n_trials: int = 30) -> dict:
    """Optuna-tune LightGBM on the training split. Returns best_params dict."""
    cv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int  ("n_estimators",    100, 800),
            "num_leaves":        trial.suggest_int  ("num_leaves",       20, 127),
            "learning_rate":     trial.suggest_float("learning_rate",  0.01, 0.15),
            "max_depth":         trial.suggest_int  ("max_depth",         3,  12),
            "subsample":         trial.suggest_float("subsample",       0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",0.5, 1.0),
            "min_child_samples": trial.suggest_int  ("min_child_samples",10,  60),
            "reg_alpha":         trial.suggest_float("reg_alpha",         0, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda",        0, 2.0),
        }
        scores = []
        for tr_idx, te_idx in cv.split(X_tr):
            Xtr, Xte = X_tr.iloc[tr_idx], X_tr.iloc[te_idx]
            ytr, yte = y_tr.iloc[tr_idx], y_tr.iloc[te_idx]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            m = make_lgbm_model(params, pos_weight)
            m.fit(Xtr, ytr)
            prob = m.predict_proba(Xte)[:, 1]
            scores.append(roc_auc_score(yte, prob))
        return np.mean(scores) if scores else 0.50

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_trial.params


def predict_proba_1(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def ensemble_proba(models, X: pd.DataFrame) -> np.ndarray:
    return np.mean([predict_proba_1(m, X) for m in models], axis=0)


# ==============================================================================
# THRESHOLD SELECTION
# ==============================================================================
def choose_threshold(prob_up, y, future_ret, cost_bps, min_coverage, min_trades, target_precision):
    best_threshold = 0.50
    best_score     = -np.inf
    best_precision = 0.0
    best_trades    = 0
    for thr in np.arange(0.50, 0.96, 0.01):
        pred     = (prob_up >= thr).astype(int)
        coverage = float(pred.mean())
        trades   = int(pred.sum())
        if coverage < min_coverage or trades < min_trades:
            continue
        precision = precision_score(y, pred, zero_division=0)
        side      = np.where(prob_up >= thr, 1, np.where(prob_up <= 1.0 - thr, -1, 0))
        gross     = side * future_ret
        cost      = np.where(side != 0, cost_bps / 10000.0, 0.0)
        net       = gross - cost
        total_ret = float(np.prod(1.0 + net) - 1.0)
        score     = (10.0 + coverage + total_ret) if precision >= target_precision \
                    else (precision + 0.05 * coverage + total_ret)
        if score > best_score:
            best_score     = score
            best_threshold = float(round(thr, 2))
            best_precision = precision
            best_trades    = trades
    return best_threshold, best_precision, best_trades


# ==============================================================================
# PER-SYMBOL EVALUATOR
# ==============================================================================
def evaluate_symbol(
    symbol,
    horizon=5, move_threshold=0.02, period="3y",
    train_window=500, test_window=20,
    cost_bps=3.0, min_coverage=0.05, min_trades=3,
    target_precision=0.55, optuna_trials=30
):
    t0 = time.time()
    try:
        df = fetch_daily(symbol, period=period)
        if df.empty:
            return {"symbol": symbol, "error": "no data"}

        ctx = fetch_market_context(period=period)
        df  = align_context(df, ctx)
        df  = add_features(df, horizon=horizon, move_threshold=move_threshold)

        if len(df) < train_window + test_window:
            return {"symbol": symbol, "error": f"insufficient rows: {len(df)}"}

        # ensure all feature cols exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        available = [c for c in FEATURE_COLS if c in df.columns]

        fold_rows = []
        start = train_window
        fold  = 1

        while start + test_window <= len(df):
            train = df.iloc[start - train_window : start].copy()
            test  = df.iloc[start : start + test_window].copy()

            split          = max(int(len(train) * 0.80), len(train) - 100)
            train_core     = train.iloc[:split]
            threshold_val  = train.iloc[split:]
            if len(threshold_val) < 30:
                threshold_val = train.iloc[-30:]
                train_core    = train.iloc[:-30]

            pos        = max(1, int(train_core["target"].sum()))
            neg        = max(1, len(train_core) - pos)
            pos_weight = neg / pos

            # ── Tune LightGBM on train_core ───────────────────────────
            Xtr = train_core[available]
            ytr = train_core["target"]
            best_lgbm_params = tune_lgbm(Xtr, ytr, pos_weight, n_trials=optuna_trials)
            lgbm_model       = make_lgbm_model(best_lgbm_params, pos_weight)
            lgbm_model.fit(Xtr, ytr)

            # ── Base models ───────────────────────────────────────────
            base_models = make_base_models(pos_weight)
            for m in base_models:
                m.fit(Xtr, ytr)

            # Ensemble of 3 (LGBM + ET + RF)
            all_models = [lgbm_model] + base_models

            # ── Threshold calibration on validation slice ──────────────
            val_prob  = ensemble_proba(all_models, threshold_val[available])
            threshold, _, _ = choose_threshold(
                val_prob,
                threshold_val["target"].to_numpy(),
                threshold_val[f"future_ret_{horizon}d"].to_numpy(),
                cost_bps, min_coverage, min_trades, target_precision
            )

            # ── Full retrain on entire train window ───────────────────
            final_lgbm = make_lgbm_model(best_lgbm_params, pos_weight)
            final_lgbm.fit(train[available], train["target"])
            final_base = make_base_models(pos_weight)
            for m in final_base:
                m.fit(train[available], train["target"])
            final_all = [final_lgbm] + final_base

            # ── Predict test ──────────────────────────────────────────
            test_prob  = ensemble_proba(final_all, test[available])
            y_true     = test["target"].to_numpy()
            future_ret = test[f"future_ret_{horizon}d"].to_numpy()
            pred       = (test_prob >= threshold).astype(int)
            side       = np.where(test_prob >= threshold, 1, np.where(test_prob <= 1.0 - threshold, -1, 0))
            gross      = side * future_ret
            cost       = np.where(side != 0, cost_bps / 10000.0, 0.0)
            net        = gross - cost
            equity     = np.cumprod(1.0 + net)
            peak       = np.maximum.accumulate(equity) if len(equity) else np.array([1.0])
            drawdown   = equity / peak - 1.0 if len(equity) else np.array([0.0])

            fold_rows.append({
                "fold":            fold,
                "raw_accuracy":    float(accuracy_score(y_true, pred)),
                "precision":       float(precision_score(y_true, pred, zero_division=0)),
                "coverage":        float((side != 0).mean()),
                "trades":          int((side != 0).sum()),
                "trade_win_rate":  float((net[side != 0] > 0).mean()) if (side != 0).any() else 0.0,
                "total_return_pct":float((equity[-1] - 1.0) * 100) if len(equity) else 0.0,
                "max_drawdown_pct":float(drawdown.min() * 100) if len(drawdown) else 0.0,
            })
            fold  += 1
            start += test_window

        folds  = pd.DataFrame(fold_rows)
        result = {
            "symbol":            symbol,
            "horizon":           horizon,
            "move_threshold":    move_threshold,
            "folds":             int(len(folds)),
            "raw_accuracy":      float(folds["raw_accuracy"].mean()),
            "precision":         float(folds["precision"].mean()),
            "coverage":          float(folds["coverage"].mean()),
            "trades":            int(folds["trades"].sum()),
            "trade_win_rate":    float(folds["trade_win_rate"].mean()),
            "total_return_pct":  float(folds["total_return_pct"].mean()),
            "max_drawdown_pct":  float(folds["max_drawdown_pct"].mean()),
            "runtime_sec":       round(time.time() - t0, 1),
        }
        print(
            f"  {symbol}: acc={result['raw_accuracy']*100:.1f}%  "
            f"prec={result['precision']*100:.1f}%  "
            f"win={result['trade_win_rate']*100:.1f}%",
            flush=True
        )
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"symbol": symbol, "error": str(e), "runtime_sec": round(time.time() - t0, 1)}


# ==============================================================================
# SYMBOL LOADER
# ==============================================================================
def load_symbols(path: str = "nifty100_stocks.txt") -> list:
    symbols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sym = line.split(",")[0].strip() if "," in line else line.strip()
            if sym and sym.endswith(".NS") and sym not in symbols:
                symbols.append(sym)
    return symbols


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kronos v2 – Nifty-100 Walk-Forward Evaluator")
    parser.add_argument("--symbols-file",     default="nifty100_stocks.txt")
    parser.add_argument("--period",           default="3y")
    parser.add_argument("--horizon",          type=int,   default=5)
    parser.add_argument("--move-threshold",   type=float, default=0.02)
    parser.add_argument("--train-window",     type=int,   default=500)
    parser.add_argument("--test-window",      type=int,   default=20)
    parser.add_argument("--cost-bps",         type=float, default=3.0)
    parser.add_argument("--min-coverage",     type=float, default=0.05)
    parser.add_argument("--min-trades",       type=int,   default=3)
    parser.add_argument("--target-precision", type=float, default=0.55)
    parser.add_argument("--output-dir",       default="outputs")
    parser.add_argument("--sample",           type=int,   default=20)
    parser.add_argument("--optuna-trials",    type=int,   default=30)
    args = parser.parse_args()

    symbols  = load_symbols(args.symbols_file)[: args.sample]
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "nifty100_fast_details.csv"
    summary_path = out_dir / "nifty100_fast_summary.csv"
    date_str     = datetime.date.today().isoformat()
    metrics_path = out_dir / f"metrics_{date_str}.json"

    results = []
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Evaluating {sym} ...", flush=True)
        res = evaluate_symbol(
            sym,
            horizon=args.horizon,
            move_threshold=args.move_threshold,
            period=args.period,
            train_window=args.train_window,
            test_window=args.test_window,
            cost_bps=args.cost_bps,
            min_coverage=args.min_coverage,
            min_trades=args.min_trades,
            target_precision=args.target_precision,
            optuna_trials=args.optuna_trials,
        )
        results.append(res)

        pd.DataFrame(results).to_csv(details_path, index=False)

        valid = [r for r in results if "error" not in r]
        if valid:
            summary = {
                "symbols_tested":       len(valid),
                "avg_raw_accuracy":     float(np.mean([r["raw_accuracy"]    for r in valid])),
                "avg_precision":        float(np.mean([r["precision"]        for r in valid])),
                "avg_trade_win_rate":   float(np.mean([r["trade_win_rate"]   for r in valid])),
                "symbols_above_55":     int(sum(1 for r in valid if r["raw_accuracy"] >= 0.55)),
                "symbols_above_60":     int(sum(1 for r in valid if r["raw_accuracy"] >= 0.60)),
                "symbols_above_65":     int(sum(1 for r in valid if r["raw_accuracy"] >= 0.65)),
                "symbols_above_70":     int(sum(1 for r in valid if r["raw_accuracy"] >= 0.70)),
            }
            pd.DataFrame([summary]).to_csv(summary_path, index=False)
            print(
                f"  -> interim avg={summary['avg_raw_accuracy']*100:.1f}%  "
                f">=60%: {summary['symbols_above_60']}/{len(valid)}  "
                f">=70%: {summary['symbols_above_70']}/{len(valid)}",
                flush=True
            )

    print("\n===== FINAL NIFTY 100 FAST SUMMARY =====", flush=True)
    valid = [r for r in results if "error" not in r]
    if valid:
        summary = {
            "symbols_tested":     len(valid),
            "avg_raw_accuracy":   float(np.mean([r["raw_accuracy"]    for r in valid])),
            "avg_precision":      float(np.mean([r["precision"]        for r in valid])),
            "avg_trade_win_rate": float(np.mean([r["trade_win_rate"]   for r in valid])),
            "symbols_above_55":   int(sum(1 for r in valid if r["raw_accuracy"] >= 0.55)),
            "symbols_above_60":   int(sum(1 for r in valid if r["raw_accuracy"] >= 0.60)),
            "symbols_above_65":   int(sum(1 for r in valid if r["raw_accuracy"] >= 0.65)),
            "symbols_above_70":   int(sum(1 for r in valid if r["raw_accuracy"] >= 0.70)),
        }
        print(pd.DataFrame([summary]).to_string(index=False), flush=True)
        # Save daily metrics JSON
        with open(metrics_path, "w") as f:
            json.dump({"date": date_str, "summary": summary, "details": valid}, f, indent=2)
        print(f"Saved: {metrics_path}", flush=True)

    print(f"Saved: {details_path}", flush=True)
    print(f"Saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
