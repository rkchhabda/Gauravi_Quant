import argparse
import json
import sys
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from lightgbm import LGBMClassifier

sys.path.append("./kronos_lib")
from model import Kronos, KronosTokenizer, KronosPredictor

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
MARKET_CONTEXT_SYMBOLS = ["^NSEI", "^INDIAVIX", "INR=X"]
KRONOS_DAILY_LOOKBACK = 90
KRONOS_DAILY_PRED_LEN = 14


def fetch_daily(symbol: str, period: str = "3y") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    raw = pd.DataFrame()
    try:
        raw = ticker.history(period=period, interval="1d", auto_adjust=True)
    except Exception:
        pass
    if raw.empty:
        try:
            end = pd.Timestamp.now()
            start = end - pd.DateOffset(years=2)
            raw = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True)
        except Exception:
            raw = pd.DataFrame()
    if raw.empty:
        raise RuntimeError(f"No daily data returned for {symbol}.")
    df = raw.reset_index()
    df["timestamps"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
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


def align_context(df: pd.DataFrame, ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    for sym, ctx_df in ctx.items():
        if ctx_df.empty:
            continue
        col_prefix = sym.replace("^", "").replace("=", "").replace("-", "_")
        merged = out.merge(
            ctx_df[["timestamps", "close"]].rename(columns={"close": f"{col_prefix}_close"}),
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
    except Exception:
        device = "cpu"
    try:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        lookback = min(len(df), KRONOS_DAILY_LOOKBACK)
        x_df = df.iloc[-lookback:][["open", "high", "low", "close"]].reset_index(drop=True)
        x_timestamp = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)
        last_date = x_timestamp.iloc[-1]
        y_timestamp = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=KRONOS_DAILY_PRED_LEN))
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=KRONOS_DAILY_PRED_LEN, T=0.75, top_p=0.9, sample_count=3, verbose=False,
        )
        last_close = float(x_df["close"].iloc[-1])
        proj_close = float(pred_df["close"].iloc[-1])
        proj_return = ((proj_close - last_close) / last_close) * 100
        proj_direction = 1.0 if proj_return >= 0 else 0.0
        proj_high = float(pred_df["high"].max())
        proj_low = float(pred_df["low"].min())
        proj_range = ((proj_high - proj_low) / last_close) * 100
        proj_std = float(pred_df["close"].std() / last_close * 100)
        return {
            "kronos_proj_return": proj_return,
            "kronos_proj_direction": proj_direction,
            "kronos_proj_range_pct": proj_range,
            "kronos_proj_std_pct": proj_std,
            "kronos_confidence": min(100.0, max(50.0, 50.0 + abs(proj_return) * 10.0)),
        }
    except Exception:
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

    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)
    out["ret_20"] = close.pct_change(20)
    out[f"future_ret_{horizon}d"] = close.shift(-horizon) / close - 1.0
    out["target"] = (out[f"future_ret_{horizon}d"] > move_threshold).astype(int)

    out["momentum_5"] = close / close.shift(5) - 1.0
    out["momentum_10"] = close / close.shift(10) - 1.0
    out["momentum_20"] = close / close.shift(20) - 1.0

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

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50.0)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=3).mean()
    out["atr_pct"] = (atr / close).fillna(0.0)
    out["range_pct"] = (high - low) / close
    out["range_10"] = out["range_pct"].rolling(10, min_periods=3).mean()
    out["range_ratio_10"] = out["range_pct"] / out["range_10"].replace(0, np.nan)

    out["volatility_5"] = out["ret_1"].rolling(5, min_periods=3).std()
    out["volatility_10"] = out["ret_1"].rolling(10, min_periods=5).std()
    out["volatility_20"] = out["ret_1"].rolling(20, min_periods=8).std()

    out["volume_change_1"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    out["volume_ratio_20"] = volume / volume.rolling(20, min_periods=5).mean()

    roll_std = close.rolling(20, min_periods=1).std().fillna(1.0)
    bb_middle = close.rolling(20, min_periods=1).mean()
    bb_upper = bb_middle + 2 * roll_std
    bb_lower = bb_middle - 2 * roll_std
    out["dist_bb_upper"] = (bb_upper - close) / close
    out["dist_bb_lower"] = (close - bb_lower) / close

    out["day_of_week"] = out["timestamps"].dt.dayofweek
    out["month"] = out["timestamps"].dt.month

    out["gap_up"] = (out["open"] / close.shift(1) - 1.0).fillna(0.0)
    out["opening_range"] = (out["high"] - out["low"]) / out["open"]
    out["close_location"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)

    out["stop_loss_pct"] = out["atr_pct"] * 2.0
    out["risk_reward_ratio"] = move_threshold / (out["atr_pct"] * 2.0).replace(0, np.nan)

    return out.dropna(subset=[f"future_ret_{horizon}d", "target"]).reset_index(drop=True)


FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "momentum_5", "momentum_10", "momentum_20",
    "ema_gap_5_10", "ema_gap_10_20", "ema_gap_20_50",
    "close_vs_ema10", "close_vs_ema20", "close_vs_ema50",
    "rsi_14", "macd_hist", "atr_pct", "range_pct", "range_ratio_10",
    "volatility_5", "volatility_10", "volatility_20",
    "volume_change_1", "volume_ratio_20",
    "dist_bb_upper", "dist_bb_lower",
    "day_of_week", "month",
    "kronos_proj_return", "kronos_proj_direction", "kronos_proj_range_pct", "kronos_proj_std_pct", "kronos_confidence",
    "NSEI_ret_1", "NSEI_ret_5", "INDIAVIX_ret_1", "INR_X_ret_1",
    "gap_up", "opening_range", "close_location",
    "stop_loss_pct", "risk_reward_ratio",
]


def make_models(pos_weight: float, n_estimators: int = 300):
    common = dict(random_state=42, n_jobs=-1)
    return [
        LGBMClassifier(
            objective="binary", n_estimators=n_estimators, learning_rate=0.03,
            num_leaves=31, min_child_samples=25, subsample=0.9, colsample_bytree=0.9,
            class_weight={0: 1.0, 1: pos_weight}, verbosity=-1, **common,
        ),
        ExtraTreesClassifier(
            n_estimators=n_estimators, min_samples_leaf=8, max_features=0.7,
            class_weight={0: 1.0, 1: pos_weight}, **common,
        ),
        RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=10, max_features=0.65,
            class_weight={0: 1.0, 1: pos_weight}, **common,
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


def choose_threshold(prob_up, y, future_ret, cost_bps, min_coverage, min_trades, target_precision):
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


def evaluate_symbol(symbol, period, train_window, test_window, horizon, move_threshold, cost_bps, min_coverage, min_trades, target_precision, skip_kronos):
    try:
        df = fetch_daily(symbol, period=period)
        ctx = fetch_market_context(period=period)
        df = align_context(df, ctx)
        df = add_features(df, horizon=horizon, move_threshold=move_threshold)
        if len(df) < train_window + test_window:
            return {"symbol": symbol, "error": f"insufficient data: {len(df)} rows"}

        available = [c for c in FEATURE_COLS if c in df.columns]
        missing = sorted(set(FEATURE_COLS) - set(available))
        if missing:
            for col in missing:
                df[col] = 0.0

        fold_rows = []
        start = train_window
        fold = 1
        while start + test_window <= len(df):
            train = df.iloc[start - train_window : start].copy()
            test = df.iloc[start : start + test_window].copy()

            if not skip_kronos:
                train_kronos = compute_kronos_features(train)
                test_kronos = compute_kronos_features(pd.concat([train, test], ignore_index=True))
                for k, v in train_kronos.items():
                    train[k] = v
                for k, v in test_kronos.items():
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

            threshold_models = make_models(pos_weight, n_estimators=200)
            fit_models(threshold_models, train_core[available], train_core["target"])
            val_prob = ensemble_predict_proba(threshold_models, threshold_val[available])
            threshold, _, _, _ = choose_threshold(
                val_prob, threshold_val["target"].to_numpy(), threshold_val[f"future_ret_{horizon}d"].to_numpy(),
                cost_bps=cost_bps, min_coverage=min_coverage, min_trades=min_trades, target_precision=target_precision,
            )

            final_models = make_models(pos_weight, n_estimators=200)
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

            fold_rows.append({
                "fold": fold,
                "raw_accuracy": float(accuracy_score(y, pred)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "coverage": float((side != 0).mean()),
                "trades": int((side != 0).sum()),
                "trade_win_rate": float((net[side != 0] > 0).mean()) if (side != 0).any() else 0.0,
                "total_return_pct": float((equity[-1] - 1.0) * 100) if len(equity) else 0.0,
                "max_drawdown_pct": float(drawdown.min() * 100) if len(drawdown) else 0.0,
            })
            fold += 1
            start += test_window

        folds = pd.DataFrame(fold_rows)
        return {
            "symbol": symbol,
            "folds": int(len(folds)),
            "raw_accuracy": float(folds["raw_accuracy"].mean()),
            "precision": float(folds["precision"].mean()),
            "coverage": float(folds["coverage"].mean()),
            "trades": int(folds["trades"].sum()),
            "trade_win_rate": float(folds["trade_win_rate"].mean()),
            "total_return_pct": float(folds["total_return_pct"].mean()),
            "max_drawdown_pct": float(folds["max_drawdown_pct"].mean()),
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def load_nifty100_symbols(path: str = "nifty100_stocks.txt") -> list[str]:
    symbols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                sym = line.split(",")[0].strip()
            else:
                sym = line.strip()
            if (sym.endswith((".NS", ".BO", "=X")) or sym.startswith("^")) and sym not in symbols:
                symbols.append(sym)
    return symbols


def main():
    parser = argparse.ArgumentParser(description="NIFTY 100 focused parameter sweep.")
    parser.add_argument("--symbols-file", default="nifty100_stocks.txt")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--train-window", type=int, default=400)
    parser.add_argument("--test-window", type=int, default=40)
    parser.add_argument("--cost-bps", type=float, default=3.0)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--target-precision", type=float, default=0.55)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sample", type=int, default=10, help="Number of symbols to sample from NIFTY 100")
    parser.add_argument("--fast", action="store_true", help="Use smaller train/test windows for speed")
    args = parser.parse_args()

    if args.fast:
        args.train_window = 250
        args.test_window = 20
        args.sample = min(args.sample, 10)

    symbols = load_nifty100_symbols(args.symbols_file)[: args.sample]
    param_grid = [
        {"horizon": 5, "move_threshold": 0.015},
        {"horizon": 5, "move_threshold": 0.02},
        {"horizon": 10, "move_threshold": 0.025},
        {"horizon": 10, "move_threshold": 0.03},
    ]

    results = []
    for params in param_grid:
        print(f"\n===== PARAM SWEEP: horizon={params['horizon']}, move={params['move_threshold']*100:.1f}% =====", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    evaluate_symbol,
                    symbol=sym,
                    period=args.period,
                    train_window=args.train_window,
                    test_window=args.test_window,
                    horizon=params["horizon"],
                    move_threshold=params["move_threshold"],
                    cost_bps=args.cost_bps,
                    min_coverage=args.min_coverage,
                    min_trades=args.min_trades,
                    target_precision=args.target_precision,
                    skip_kronos=True,
                ): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    res = future.result()
                    res["horizon"] = params["horizon"]
                    res["move_threshold"] = params["move_threshold"]
                    results.append(res)
                    if "error" in res:
                        print(f"  {sym}: ERROR - {res['error']}", flush=True)
                    else:
                        print(f"  {sym}: acc={res['raw_accuracy']*100:.1f}%, precision={res['precision']*100:.1f}%, win={res['trade_win_rate']*100:.1f}%", flush=True)
                except Exception as e:
                    print(f"  {sym}: EXCEPTION - {e}", flush=True)

    summary_rows = []
    for params in param_grid:
        subset = [r for r in results if r.get("horizon") == params["horizon"] and r.get("move_threshold") == params["move_threshold"] and "error" not in r]
        if not subset:
            continue
        summary_rows.append({
            "horizon": params["horizon"],
            "move_threshold": params["move_threshold"],
            "symbols_tested": len(subset),
            "avg_raw_accuracy": float(np.mean([r["raw_accuracy"] for r in subset])),
            "avg_precision": float(np.mean([r["precision"] for r in subset])),
            "avg_trade_win_rate": float(np.mean([r["trade_win_rate"] for r in subset])),
            "avg_total_return": float(np.mean([r["total_return_pct"] for r in subset])),
            "symbols_above_55": int(sum(1 for r in subset if r["raw_accuracy"] >= 0.55)),
            "symbols_above_60": int(sum(1 for r in subset if r["raw_accuracy"] >= 0.60)),
        })

    summary_df = pd.DataFrame(summary_rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "nifty100_param_sweep_summary.csv", index=False)
    pd.DataFrame(results).to_csv(out_dir / "nifty100_param_sweep_details.csv", index=False)

    print("\n===== NIFTY 100 PARAMETER SWEEP SUMMARY =====", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nSaved summary: {out_dir / 'nifty100_param_sweep_summary.csv'}", flush=True)
    print(f"Saved details: {out_dir / 'nifty100_param_sweep_details.csv'}", flush=True)


if __name__ == "__main__":
    main()
