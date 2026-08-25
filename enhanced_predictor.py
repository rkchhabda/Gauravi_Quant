import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from lightgbm import LGBMClassifier

sys.path.append("./kronos_lib")
from model import Kronos, KronosTokenizer, KronosPredictor

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MARKET_CONTEXT_SYMBOLS = ["^NSEI", "^INDIAVIX", "INR=X"]

def get_hf_token():
    """Get Hugging Face token from environment at runtime."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

KRONOS_DAILY_LOOKBACK = 90
KRONOS_DAILY_PRED_LEN = 14


def fetch_daily(symbol: str, period: str = "3y") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    raw = pd.DataFrame()
    
    # Parse period to get years
    years = 3
    if period.endswith('y'):
        try:
            years = int(period[:-1])
        except Exception:
            years = 3
    
    # Use explicit date range with auto_adjust=False (more reliable for Indian stocks)
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=years)
    try:
        raw = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", auto_adjust=False)
    except Exception as e:
        print(f"[WARN] Explicit date fetch failed for {symbol}: {e}")
        raw = pd.DataFrame()
    
    # Fallback to period with auto_adjust=False
    if raw.empty:
        try:
            raw = ticker.history(period=period, interval="1d", auto_adjust=False)
        except Exception:
            pass
    
    if raw.empty:
        raise RuntimeError(f"No daily data returned for {symbol}.")
    
    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    
    df = raw.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df["timestamps"] = pd.to_datetime(df[dt_col]).dt.tz_localize(None)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
    df = df.sort_values("timestamps").reset_index(drop=True)
    return df


def fetch_market_context(period: str = "3y") -> dict[str, pd.DataFrame]:
    ctx = {}
    for sym in MARKET_CONTEXT_SYMBOLS:
        try:
            ctx[sym] = fetch_daily(sym, period=period)
        except Exception:
            ctx[sym] = pd.DataFrame()
    return ctx


def align_context(
    df: pd.DataFrame, ctx: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    out = df.copy()
    for sym, ctx_df in ctx.items():
        if ctx_df.empty:
            continue
        col_prefix = sym.replace("^", "").replace("=", "").replace("-", "_")
        merged = out.merge(
            ctx_df[["timestamps", "close"]].rename(
                columns={"close": f"{col_prefix}_close"}
            ),
            on="timestamps",
            how="left",
        )
        merged[f"{col_prefix}_ret_1"] = merged[f"{col_prefix}_close"].pct_change(1)
        merged[f"{col_prefix}_ret_5"] = merged[f"{col_prefix}_close"].pct_change(5)
        merged[f"{col_prefix}_volatility_20"] = (
            merged[f"{col_prefix}_close"].pct_change().rolling(20, min_periods=5).std()
        )
        out = merged.drop(
            columns=[c for c in merged.columns if c.startswith(f"{col_prefix}_close")],
            errors="ignore",
        )
    return out


def compute_kronos_features(df: pd.DataFrame) -> dict:
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[Kronos] Using device: {device}")
    except Exception as e:
        print(f"[Kronos] Device detection failed: {e}")
        device = "cpu"

    try:
        print("[Kronos] Loading tokenizer and model...")
        tokenizer_kwargs = {"token": get_hf_token()} if get_hf_token() else {}
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME, **tokenizer_kwargs)
        model = Kronos.from_pretrained(KRONOS_MODEL_NAME, **tokenizer_kwargs)
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

        lookback = min(len(df), KRONOS_DAILY_LOOKBACK)
        x_df = df.iloc[-lookback:][["open", "high", "low", "close"]].reset_index(drop=True)
        x_timestamp = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)
        last_date = x_timestamp.iloc[-1]
        print(f"[Kronos] Last date in data: {last_date}, lookback: {lookback}")
        
        # Use business day range for accurate trading day predictions
        y_timestamp = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=KRONOS_DAILY_PRED_LEN))

        print("[Kronos] Running prediction...")
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=KRONOS_DAILY_PRED_LEN,
            T=0.75,
            top_p=0.9,
            sample_count=3,
            verbose=False,
        )

        last_close = float(x_df["close"].iloc[-1])
        proj_close = float(pred_df["close"].iloc[-1])
        proj_return = ((proj_close - last_close) / last_close) * 100
        proj_direction = 1.0 if proj_return >= 0 else 0.0
        proj_high = float(pred_df["high"].max())
        proj_low = float(pred_df["low"].min())
        proj_range = ((proj_high - proj_low) / last_close) * 100
        proj_std = float(pred_df["close"].std() / last_close * 100)

        result = {
            "kronos_proj_return": proj_return,
            "kronos_proj_direction": proj_direction,
            "kronos_proj_range_pct": proj_range,
            "kronos_proj_std_pct": proj_std,
            "kronos_confidence": min(100.0, max(50.0, 50.0 + abs(proj_return) * 10.0)),
        }
        print(f"[Kronos] Success: {result}")
        return result
    except Exception as e:
        print(f"[Kronos] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "kronos_proj_return": 0.0,
            "kronos_proj_direction": 0.5,
            "kronos_proj_range_pct": 0.0,
            "kronos_proj_std_pct": 0.0,
            "kronos_confidence": 50.0,
        }


def add_features(df: pd.DataFrame, horizon: int = 5, move_threshold: float = 0.02) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"].replace(0, np.nan)

    # Returns
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)
    
    # Future return and target - HIGHER QUALITY TARGET
    out[f"future_ret_{horizon}d"] = close.shift(-horizon) / close - 1.0
    out["target"] = (out[f"future_ret_{horizon}d"] > move_threshold).astype(int)

    # Momentum
    out["momentum_5"] = close / close.shift(5) - 1.0
    out["momentum_10"] = close / close.shift(10) - 1.0
    out["momentum_20"] = close / close.shift(20) - 1.0

    # EMAs
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    out["ema_gap_5_10"] = ema5 / ema10 - 1.0
    out["ema_gap_10_20"] = ema10 / ema20 - 1.0
    out["ema_gap_20_50"] = ema20 / ema50 - 1.0
    out["close_vs_ema10"] = close / ema10 - 1.0
    out["close_vs_ema20"] = close / ema20 - 1.0
    out["close_vs_ema50"] = close / ema50 - 1.0

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50.0)

    # MACD
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(
        span=26, adjust=False
    ).mean()
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    # ATR and volatility
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=3).mean()
    out["atr_pct"] = (atr / close).fillna(0.0)
    out["range_pct"] = (high - low) / close
    out["range_10"] = out["range_pct"].rolling(10, min_periods=3).mean()
    out["range_ratio_10"] = out["range_pct"] / out["range_10"].replace(0, np.nan)

    out["volatility_5"] = out["ret_1"].rolling(5, min_periods=3).std()
    out["volatility_10"] = out["ret_1"].rolling(10, min_periods=5).std()
    out["volatility_20"] = out["ret_1"].rolling(20, min_periods=8).std()

    # Volume
    out["volume_change_1"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    out["volume_ratio_20"] = volume / volume.rolling(20, min_periods=5).mean()

    # Bollinger Bands
    roll_std = close.rolling(20, min_periods=1).std().fillna(1.0)
    bb_middle = close.rolling(20, min_periods=1).mean()
    bb_upper = bb_middle + 2 * roll_std
    bb_lower = bb_middle - 2 * roll_std
    out["dist_bb_upper"] = (bb_upper - close) / close
    out["dist_bb_lower"] = (close - bb_lower) / close

    # Time features
    out["day_of_week"] = out["timestamps"].dt.dayofweek
    out["month"] = out["timestamps"].dt.month

    # Market regime features
    out["gap_up"] = (out["open"] / close.shift(1) - 1.0).fillna(0.0)
    out["opening_range"] = (out["high"] - out["low"]) / out["open"]
    out["close_location"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)

    # Risk management features
    out["stop_loss_pct"] = out["atr_pct"] * 2.0  # 2x ATR stop
    out["risk_reward_ratio"] = move_threshold / (out["atr_pct"] * 2.0).replace(0, np.nan)  # R:R ratio

    return out.dropna(subset=[f"future_ret_{horizon}d", "target"]).reset_index(drop=True)


FEATURE_COLS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "ema_gap_5_10",
    "ema_gap_10_20",
    "ema_gap_20_50",
    "close_vs_ema10",
    "close_vs_ema20",
    "close_vs_ema50",
    "rsi_14",
    "macd_hist",
    "atr_pct",
    "range_pct",
    "range_ratio_10",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volume_change_1",
    "volume_ratio_20",
    "dist_bb_upper",
    "dist_bb_lower",
    "day_of_week",
    "month",
    "kronos_proj_return",
    "kronos_proj_direction",
    "kronos_proj_range_pct",
    "kronos_proj_std_pct",
    "kronos_confidence",
    "NSEI_ret_1",
    "NSEI_ret_5",
    "INDIAVIX_ret_1",
    "INR_X_ret_1",
    "gap_up",
    "opening_range",
    "close_location",
    "stop_loss_pct",
    "risk_reward_ratio",
]


def make_models(pos_weight: float):
    common = dict(
        random_state=42,
        n_jobs=-1,
    )
    return [
        LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight={0: 1.0, 1: pos_weight},
            verbosity=-1,
            **common,
        ),
        ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=8,
            max_features=0.7,
            class_weight={0: 1.0, 1: pos_weight},
            **common,
        ),
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            max_features=0.65,
            class_weight={0: 1.0, 1: pos_weight},
            **common,
        ),
    ]


def model_predict_proba(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def fit_models(models: list, x: pd.DataFrame, y: pd.Series) -> None:
    for model in models:
        model.fit(x, y)


def ensemble_predict_proba(models: list, x: pd.DataFrame) -> np.ndarray:
    probs = [model_predict_proba(model, x) for model in models]
    return np.mean(probs, axis=0)


def choose_threshold(
    prob_up: np.ndarray,
    y: np.ndarray,
    future_ret: np.ndarray,
    cost_bps: float,
    min_coverage: float,
    min_trades: int,
    target_precision: float,
) -> tuple[float, bool, float, int]:
    best_threshold = 0.5
    best_score = -np.inf
    best_hit_target = False
    best_precision = 0.0
    best_trades = 0
    for threshold in np.arange(0.50, 0.96, 0.01):
        pred = (prob_up >= threshold).astype(int)
        coverage = float(pred.mean())
        trades = int(pred.sum())
        if coverage < min_coverage or trades < min_trades:
            continue
        precision = precision_score(y, pred, zero_division=0)
        side = np.where(prob_up >= threshold, 1, np.where(prob_up <= 1.0 - threshold, -1, 0))
        gross = side * future_ret
        cost = np.where(side != 0, cost_bps / 10000.0, 0.0)
        net = gross - cost
        total_return = float(np.prod(1.0 + net) - 1.0)
        hit_target = precision >= target_precision
        if hit_target:
            score = 10.0 + coverage + total_return
        else:
            score = precision + 0.05 * coverage + total_return
        if score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 2))
            best_hit_target = hit_target
            best_precision = precision
            best_trades = trades
    return best_threshold, best_hit_target, best_precision, best_trades


def run_walk_forward(
    symbol: str,
    period: str,
    train_window: int,
    test_window: int,
    horizon: int,
    move_threshold: float,
    cost_bps: float,
    min_coverage: float,
    min_trades: int,
    target_precision: float,
    skip_kronos: bool,
) -> tuple[pd.DataFrame, dict]:
    df = fetch_daily(symbol, period=period)
    ctx = fetch_market_context(period=period)
    df = align_context(df, ctx)
    df = add_features(df, horizon=horizon, move_threshold=move_threshold)
    if len(df) < train_window + test_window:
        raise RuntimeError(
            f"Only {len(df)} feature rows available; need at least train_window + test_window."
        )

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = sorted(set(FEATURE_COLS) - set(available))
    if missing:
        print(f"[WARN] Missing features filled with 0: {missing}")
        for col in missing:
            df[col] = 0.0

    fold_rows = []
    prediction_rows = []
    start = train_window
    fold = 1

    while start + test_window <= len(df):
        train = df.iloc[start - train_window : start].copy()
        test = df.iloc[start : start + test_window].copy()

        if not skip_kronos:
            train_kronos = compute_kronos_features(train)
            for k, v in train_kronos.items():
                train[k] = v
                test[k] = v

        split = max(int(len(train) * 0.8), len(train) - 120)
        train_core = train.iloc[:split]
        threshold_val = train.iloc[split:]
        if len(threshold_val) < 30:
            threshold_val = train.iloc[-max(30, len(train) // 5) :]
            train_core = train.iloc[: -len(threshold_val)]

        pos = max(1, int(train_core["target"].sum()))
        neg = max(1, len(train_core) - pos)
        pos_weight = neg / pos

        threshold_models = make_models(pos_weight)
        fit_models(threshold_models, train_core[available], train_core["target"])
        val_prob = ensemble_predict_proba(threshold_models, threshold_val[available])
        threshold, val_hit_target, val_precision, val_trades = choose_threshold(
            val_prob,
            threshold_val["target"].to_numpy(),
            threshold_val[f"future_ret_{horizon}d"].to_numpy(),
            cost_bps=cost_bps,
            min_coverage=min_coverage,
            min_trades=min_trades,
            target_precision=target_precision,
        )

        final_models = make_models(pos_weight)
        fit_models(final_models, train[available], train["target"])
        test_prob = ensemble_predict_proba(final_models, test[available])
        y = test["target"].to_numpy()
        future_ret = test[f"future_ret_{horizon}d"].to_numpy()
        pred = (test_prob >= threshold).astype(int)
        side = np.where(test_prob >= threshold, 1, np.where(test_prob <= 1.0 - threshold, -1, 0))
        gross = side * future_ret
        cost = np.where(side != 0, cost_bps / 10000.0, 0.0)
        net = gross - cost
        equity = np.cumprod(1.0 + net)
        peak = np.maximum.accumulate(equity) if len(equity) else np.array([1.0])
        drawdown = equity / peak - 1.0 if len(equity) else np.array([0.0])

        row = {
            "fold": fold,
            "train_start": train["timestamps"].iloc[0],
            "train_end": train["timestamps"].iloc[-1],
            "test_start": test["timestamps"].iloc[0],
            "test_end": test["timestamps"].iloc[-1],
            "positive_rate": float(y.mean()),
            "majority_baseline": float(max(y.mean(), 1.0 - y.mean())),
            "validation_hit_target": val_hit_target,
            "validation_precision": val_precision,
            "validation_trades": val_trades,
            "raw_accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, test_prob)) if len(np.unique(y)) == 2 else float("nan"),
            "threshold": threshold,
            "coverage": float((side != 0).mean()),
            "trades": int((side != 0).sum()),
            "longs": int((side == 1).sum()),
            "shorts": int((side == -1).sum()),
            "trade_win_rate": float((net[side != 0] > 0).mean()) if (side != 0).any() else 0.0,
            "avg_trade_ret_pct": float(net[side != 0].mean() * 100) if (side != 0).any() else 0.0,
            "total_return_pct": float((equity[-1] - 1.0) * 100) if len(equity) else 0.0,
            "max_drawdown_pct": float(drawdown.min() * 100) if len(drawdown) else 0.0,
        }
        fold_rows.append(row)

        prediction_rows.append(
            pd.DataFrame(
                {
                    "timestamp": test["timestamps"].to_numpy(),
                    "close": test["close"].to_numpy(),
                    f"future_ret_{horizon}d": future_ret,
                    "target": y,
                    "prob_up": test_prob,
                    "threshold": threshold,
                    "side": side,
                    "net_ret": net,
                    "fold": fold,
                }
            )
        )

        fold += 1
        start += test_window

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    overall = {
        "raw_accuracy": float(accuracy_score(predictions["target"], (predictions["prob_up"] >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(predictions["target"], predictions["prob_up"])) if len(np.unique(predictions["target"])) == 2 else float("nan"),
        "precision": float(precision_score(predictions["target"], (predictions["prob_up"] >= 0.5).astype(int), zero_division=0)),
        "recall": float(recall_score(predictions["target"], (predictions["prob_up"] >= 0.5).astype(int), zero_division=0)),
        "f1": float(f1_score(predictions["target"], (predictions["prob_up"] >= 0.5).astype(int), zero_division=0)),
        "coverage": float((predictions["side"] != 0).mean()),
        "trades": int((predictions["side"] != 0).sum()),
        "longs": int((predictions["side"] == 1).sum()),
        "shorts": int((predictions["side"] == -1).sum()),
        "trade_win_rate": float((predictions.loc[predictions["side"] != 0, "net_ret"] > 0).mean()) if (predictions["side"] != 0).any() else 0.0,
        "avg_trade_ret_pct": float(predictions.loc[predictions["side"] != 0, "net_ret"].mean() * 100) if (predictions["side"] != 0).any() else 0.0,
        "total_return_pct": float((np.prod(1.0 + predictions["net_ret"]) - 1.0) * 100) if len(predictions) else 0.0,
        "max_drawdown_pct": float((np.cumprod(1.0 + predictions["net_ret"]) / np.maximum.accumulate(np.cumprod(1.0 + predictions["net_ret"])) - 1.0).min() * 100) if len(predictions) else 0.0,
        "symbol": symbol,
        "period": period,
        "train_window": train_window,
        "test_window": test_window,
        "horizon": horizon,
        "move_threshold": move_threshold,
        "target_precision": target_precision,
        "cost_bps": cost_bps,
        "folds": int(len(folds)),
        "folds_meeting_target": int((folds["precision"] >= target_precision).sum()),
        "target_reached": bool((precision_score(predictions["target"], (predictions["prob_up"] >= 0.5).astype(int), zero_division=0) if len(predictions) else 0.0) >= target_precision),
        "majority_baseline": float(max(predictions["target"].mean(), 1.0 - predictions["target"].mean())) if len(predictions) else 0.0,
    }
    return folds, overall


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced daily predictor: higher-quality targets, market regime, Kronos features."
    )
    parser.add_argument("--symbol", default="RELIANCE.NS", help="Stock ticker, e.g. RELIANCE.NS, TCS.NS")
    parser.add_argument("--period", default="3y", help="Data period, e.g. 3y")
    parser.add_argument("--train-window", type=int, default=400, help="Training window in daily bars")
    parser.add_argument("--test-window", type=int, default=40, help="Test window in daily bars")
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon in trading days (e.g. 5 = next 5 days)",
    )
    parser.add_argument(
        "--move-threshold",
        type=float,
        default=0.02,
        help="Minimum future return to label as positive (e.g. 0.02 = 2%%)",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.60,
        help="Requested precision for positive predictions (e.g. 0.60 means 60 percent)",
    )
    parser.add_argument("--cost-bps", type=float, default=3.0, help="Round-trip cost per trade in bps")
    parser.add_argument("--min-coverage", type=float, default=0.05, help="Minimum prediction coverage")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades per fold")
    parser.add_argument("--skip-kronos", action="store_true", help="Skip Kronos features for faster runs")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    folds, overall = run_walk_forward(
        symbol=args.symbol,
        period=args.period,
        train_window=args.train_window,
        test_window=args.test_window,
        horizon=args.horizon,
        move_threshold=args.move_threshold,
        cost_bps=args.cost_bps,
        min_coverage=args.min_coverage,
        min_trades=args.min_trades,
        target_precision=args.target_precision,
        skip_kronos=args.skip_kronos,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = args.symbol.replace("^", "IDX_").replace(".", "_").replace("=", "_")
    folds_path = out_dir / f"{safe_symbol}_enhanced_walk_forward_folds.csv"
    summary_path = out_dir / f"{safe_symbol}_enhanced_walk_forward_summary.json"
    folds.to_csv(folds_path, index=False)
    summary_path.write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")

    print("\nENHANCED DAILY PREDICTOR — WALK-FORWARD REPORT")
    print("=" * 72)
    print(f"Symbol              : {overall['symbol']}")
    print(f"Horizon / Folds     : {overall['horizon']}d / {overall['folds']}")
    print(f"Move threshold      : {overall['move_threshold']*100:.1f}%")
    print(f"Raw accuracy        : {overall['raw_accuracy'] * 100:.2f}%")
    print(f"Majority baseline   : {overall['majority_baseline'] * 100:.2f}%")
    print(f"ROC AUC             : {overall['roc_auc']:.4f}")
    print(f"Precision           : {overall['precision'] * 100:.2f}%")
    print(f"Recall              : {overall['recall'] * 100:.2f}%")
    print(f"F1                  : {overall['f1']:.4f}")
    print(f"Coverage            : {overall['coverage'] * 100:.2f}% ({overall['trades']} trades)")
    print(f"Trade win rate      : {overall['trade_win_rate'] * 100:.2f}%")
    print(f"Precision target    : {'YES' if overall['target_reached'] else 'NO'}")
    print(f"Folds >= target     : {overall['folds_meeting_target']}/{overall['folds']}")
    print(f"Total return        : {overall['total_return_pct']:.2f}% after {overall['cost_bps']} bps/trade")
    print(f"Max drawdown        : {overall['max_drawdown_pct']:.2f}%")
    print(f"Saved folds         : {folds_path}")
    print(f"Saved summary       : {summary_path}")


if __name__ == "__main__":
    main()
