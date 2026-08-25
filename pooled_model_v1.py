"""
Pooled Cross-Symbol Daily Model v1
Replaces per-symbol models with one pooled model across NIFTY 100 stocks.
- Strictly causal features (no look-ahead)
- Fixed 10-day forward-return labels
- Purged expanding walk-forward (10-day embargo between train and test)
- Baseline comparisons + top-K portfolio backtest with costs

Usage:
    python pooled_model_v1.py --period 5y --horizon 10 --top-k 5 --cost-bps 10
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

HORIZON = 10


def load_symbols(path="nifty100_stocks.txt"):
    symbols = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sym = line.split(",")[0].strip()
            if sym.endswith(".NS") or sym.startswith("^"):
                symbols.append(sym)
    return symbols


def fetch_panel(symbols, period="5y", benchmark="^NSEI"):
    print(f"Downloading {len(symbols) + 1} tickers ({period})...")
    data = yf.download(
        tickers=symbols + [benchmark],
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    closes, highs, lows, vols = {}, {}, {}, {}
    for sym in symbols + [benchmark]:
        try:
            df = data[sym]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            c = df["Close"].dropna()
            if len(c) < 300:
                continue
            closes[sym] = c
            if sym != benchmark:
                highs[sym] = df["High"].reindex(c.index)
                lows[sym] = df["Low"].reindex(c.index)
                vols[sym] = df["Volume"].reindex(c.index)
        except Exception:
            continue
    close_panel = pd.DataFrame(closes)
    print(f"Got usable data for {len(closes)} tickers "
          f"({close_panel.index[0].date()} to {close_panel.index[-1].date()})")
    return close_panel, highs, lows, vols


def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build_dataset(close_panel, highs, lows, vols, horizon=HORIZON):
    bench = "^NSEI" if "^NSEI" in close_panel.columns else close_panel.columns[0]
    bench_close = close_panel[bench].dropna()
    bench_ret_1 = bench_close.pct_change()
    bench_ret_20 = bench_close.pct_change(20)
    bench_vol = bench_close.pct_change().rolling(20, min_periods=10).std()

    frames = []
    for sym in close_panel.columns:
        if sym == bench:
            continue
        c = close_panel[sym].dropna()
        if len(c) < 400:
            continue
        ret1 = c.pct_change()
        feat = pd.DataFrame(index=c.index)
        feat["ret_1"] = ret1
        feat["ret_5"] = c.pct_change(5)
        feat["ret_20"] = c.pct_change(20)
        feat["vol_10"] = ret1.rolling(10).std()
        feat["vol_20"] = ret1.rolling(20).std()
        feat["rsi_14"] = rsi(c) / 100 - 0.5
        for w in (20, 50, 200):
            sma = c.rolling(w).mean()
            feat[f"dist_sma{w}"] = c / sma - 1
        atr = pd.concat([
            (highs[sym] - lows[sym]),
            (highs[sym] - c.shift()).abs(),
            (lows[sym] - c.shift()).abs(),
        ], axis=1).max(axis=1)
        feat["atr_ratio"] = (atr.ewm(span=14, adjust=False).mean() / c)
        vol_sma20 = vols[sym].rolling(20).mean()
        feat["volume_ratio"] = (vols[sym] / vol_sma20.replace(0, np.nan)).clip(upper=10)
        feat["bench_ret_1"] = bench_ret_1.reindex(c.index)
        feat["bench_ret_20"] = bench_ret_20.reindex(c.index)
        feat["bench_vol"] = bench_vol.reindex(c.index)
        feat["rel_ret_20"] = feat["ret_20"] - feat["bench_ret_20"]
        feat["mom_12_1"] = c.shift(21) / c.shift(252) - 1 if len(c) > 252 else np.nan
        feat["mom_6m"] = c.pct_change(126)
        feat["liq"] = (vols[sym] * c).rolling(20).mean()
        feat["fwd_ret"] = c.shift(-horizon) / c - 1
        feat["symbol"] = sym
        frames.append(feat)

    df = pd.concat(frames)
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df = df.dropna(subset=[c for c in df.columns if c not in ("symbol", "fwd_ret")])

    bench_fwd = bench_close.shift(-horizon) / bench_close - 1
    df["bench_fwd"] = df["date"].map(bench_fwd)
    df["fwd_excess"] = df["fwd_ret"] - df["bench_fwd"]

    for col in FEATURES:
        g = df.groupby("date")[col]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        df[col] = ((df[col] - mu) / sd).fillna(0)

    df["label"] = (df["fwd_excess"] > 0).astype(int)
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


FEATURES = [
    "ret_1", "ret_5", "ret_20", "vol_10", "vol_20", "rsi_14",
    "dist_sma20", "dist_sma50", "dist_sma200", "atr_ratio",
    "volume_ratio", "bench_ret_1", "bench_ret_20", "bench_vol", "rel_ret_20",
    "mom_12_1", "mom_6m", "liq",
]


def make_model(name):
    if name == "xgb":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_weight=20,
            eval_metric="logloss", n_jobs=-1, verbosity=0,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            random_state=42, n_jobs=-1,
        )
    return LogisticRegression(C=0.1, max_iter=1000)


def walk_forward(df, model_name="xgb", test_months=2, embargo_days=HORIZON,
                 cost_bps=10, top_k=5):
    dates = np.sort(df["date"].unique())
    date_idx = {d: i for i, d in enumerate(dates)}
    df["didx"] = df["date"].map(date_idx)

    month_key = pd.to_datetime(df["date"]).dt.to_period("M")
    unique_months = sorted(month_key.unique())
    folds = []
    start_pos = 12

    while start_pos < len(unique_months) - 1:
        test_months_list = unique_months[start_pos:start_pos + test_months]
        test_mask_raw = month_key.isin(test_months_list)
        test_start_date = pd.Period(test_months_list[0]).start_time
        train_mask = (pd.to_datetime(df["date"]) <
                      test_start_date - pd.Timedelta(days=embargo_days + 5))

        test_df = df[test_mask_raw & (df["fwd_ret"].notna())].copy()
        train_df = df[train_mask & (df["fwd_ret"].notna())].copy()
        if len(train_df) < 5000 or test_df.empty:
            start_pos += test_months
            continue

        model = make_model(model_name)
        model.fit(train_df[FEATURES], train_df["label"])
        proba = model.predict_proba(test_df[FEATURES])[:, 1]

        res = test_df[["date", "symbol", "fwd_ret", "fwd_excess"]].copy()
        res["proba"] = proba
        res["label"] = test_df["label"].values
        res["fold"] = len(folds) + 1
        folds.append(res)

        auc = roc_auc_score(test_df["label"], proba)
        acc = ((proba > 0.5).astype(int) == test_df["label"]).mean()
        base = test_df["label"].mean()
        print(f"Fold {len(folds):>2} {test_months_list[0]}-{test_months_list[-1]}: "
              f"AUC={auc:.3f}  Acc={acc * 100:.1f}%  BaseUp={base * 100:.1f}%  "
              f"(train={len(train_df)}, test={len(test_df)})")
        start_pos += test_months

    return pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()


def evaluate(oos, cost_bps=10, top_k=5):
    print(f"\n{'=' * 70}")
    print("POOLED OUT-OF-SAMPLE EVALUATION")
    print(f"{'=' * 70}")

    y, p = oos["label"], oos["proba"]
    overall_auc = roc_auc_score(y, p)
    acc = ((p > 0.5).astype(int) == y).mean()
    base_rate = y.mean()

    print(f"Total OOS predictions : {len(oos)}")
    print(f"AUC                   : {overall_auc:.4f}")
    print(f"Accuracy @0.5         : {acc * 100:.2f}%  (baseline 'always UP': {base_rate * 100:.2f}%)")

    by_fold = oos.groupby("fold").apply(
        lambda g: pd.Series({
            "auc": roc_auc_score(g["label"], g["proba"]),
            "n": len(g),
        }), include_groups=False)
    print(f"Fold AUC mean/std     : {by_fold['auc'].mean():.3f} / {by_fold['auc'].std():.3f}")
    pos_rate = (by_fold["auc"] > 0.5).mean()
    print(f"Folds with AUC>0.5    : {pos_rate * 100:.0f}%")

    # Top-K portfolio: non-overlapping 10-day holds, pick top_k by proba
    cost = cost_bps / 1e4
    all_dates = np.sort(oos["date"].unique())
    date_pos = {d: i for i, d in enumerate(all_dates)}
    trades = []
    i = 0
    while i < len(all_dates):
        d = all_dates[i]
        g = oos[oos["date"] == d]
        if len(g) < top_k:
            i += 1
            continue
        picks = g.nlargest(top_k, "proba")
        pnl = picks["fwd_excess"].mean() - cost * 2
        trades.append({"date": d, "pnl": pnl,
                       "hit": (picks["label"] == 1).mean()})
        i += HORIZON
    port = pd.DataFrame(trades).sort_values("date")
    port["equity"] = (1 + port["pnl"]).cumprod()

    n_periods = len(port)
    tot_ret = port["equity"].iloc[-1] - 1
    ann_ret = port["pnl"].mean() * (252 / HORIZON)
    ann_vol = port["pnl"].std() * np.sqrt(252 / HORIZON)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    dd = (port["equity"] / port["equity"].cummax() - 1).min()

    print(f"\n--- Top-{top_k} Long-Short-ish Portfolio (excess of NIFTY, "
          f"hold {HORIZON}d non-overlapping, cost {cost_bps}bps/side) ---")
    print(f"Rebalance periods     : {n_periods}")
    print(f"Total excess return   : {tot_ret * 100:.1f}%")
    print(f"Annualized excess ret : {ann_ret * 100:.1f}%")
    print(f"Sharpe (naive)        : {sharpe:.2f}")
    print(f"Max drawdown          : {dd * 100:.1f}%")
    print(f"Directional hit rate  : {port['hit'].mean() * 100:.1f}%")

    # Confidence buckets
    oos["bucket"] = pd.qcut(oos["proba"], 5, labels=False, duplicates="drop")
    bucket_stats = oos.groupby("bucket").agg(
        mean_excess=("fwd_excess", "mean"), hit=("label", "mean"),
        avg_proba=("proba", "mean"), n=("fwd_excess", "size"))
    print("\nProbability-quintile monotonicity check (excess returns):")
    print(bucket_stats.round(4).to_string())

    return {
        "n_predictions": int(len(oos)),
        "auc": round(float(overall_auc), 4),
        "accuracy": round(float(acc), 4),
        "baseline_up_rate": round(float(base_rate), 4),
        "fold_auc_mean": round(float(by_fold["auc"].mean()), 4),
        "fold_auc_std": round(float(by_fold["auc"].std()), 4),
        "portfolio_total_return": round(float(tot_ret), 4),
        "portfolio_annualized_return": round(float(ann_ret), 4),
        "portfolio_sharpe": round(float(sharpe), 3),
        "portfolio_max_drawdown": round(float(dd), 4),
        "monotonic": bool(bucket_stats["mean_excess"].is_monotonic_increasing),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--model", default="xgb", choices=["xgb", "rf", "logreg"])
    ap.add_argument("--test-months", type=int, default=2)
    ap.add_argument("--cost-bps", type=float, default=10)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    global HORIZON
    HORIZON = args.horizon

    symbols = load_symbols()
    close_panel, highs, lows, vols = fetch_panel(symbols, period=args.period)
    print("Building features...")
    df = build_dataset(close_panel, highs, lows, vols, horizon=HORIZON)
    print(f"Dataset: {len(df)} rows, {df['symbol'].nunique()} symbols, "
          f"{pd.to_datetime(df['date']).min().date()} to {pd.to_datetime(df['date']).max().date()}, "
          f"UP-rate={df['label'].mean() * 100:.1f}%")

    oos = walk_forward(df, model_name=args.model, test_months=args.test_months,
                       cost_bps=args.cost_bps, top_k=args.top_k)
    if oos.empty:
        print("No folds produced.")
        sys.exit(1)

    metrics = evaluate(oos, cost_bps=args.cost_bps, top_k=args.top_k)
    metrics["model"] = args.model
    metrics["generated"] = datetime.now().isoformat()

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/pooled_model_v1_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved outputs/pooled_model_v1_results.json")


if __name__ == "__main__":
    main()
