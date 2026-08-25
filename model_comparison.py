"""
Unified Model Comparison Benchmark
All models evaluated under identical conditions:
- Universe: NIFTY 100 (pooled cross-section)
- Horizon: 10-day forward excess return vs ^NSEI
- Features: causal, cross-sectionally z-scored per day
- Validation: expanding-window purged walk-forward (10d embargo), 2-month test folds
- Metrics: AUC, accuracy vs baseline, top-quintile spread, top-K portfolio

Usage:
    python model_comparison.py --period 5y --top-k 5 --cost-bps 10
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from pooled_model_v1 import (load_symbols, fetch_panel, build_dataset,
                             make_model, FEATURES)

HORIZON = 10


def factor_score_probas(test_df, kind):
    g = test_df.groupby("date")
    if kind == "liq":
        return g["liq"].rank(pct=True).values
    if kind == "rev":
        return (1 - g["ret_20"].rank(pct=True)).values
    if kind == "mom":
        return g["mom_12_1"].rank(pct=True).values
    if kind == "composite":
        return (g["liq"].rank(pct=True) +
                (1 - g["ret_20"].rank(pct=True))).values
    raise ValueError(kind)


def evaluate_oos(oos, cost_bps=10, top_k=5):
    y, p = oos["label"], oos["proba"]
    auc = roc_auc_score(y, p) if y.nunique() > 1 else np.nan
    acc = ((p > 0.5).astype(int) == y).mean()

    fold_aucs = []
    for _, g in oos.groupby("fold"):
        if g["label"].nunique() > 1:
            fold_aucs.append(roc_auc_score(g["label"], g["proba"]))

    o = oos.copy()
    o["bucket"] = o.groupby("fold")["proba"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
    q_top = o[o["bucket"] == o["bucket"].max()]["fwd_excess"].mean()
    q_bot = o[o["bucket"] == 0]["fwd_excess"].mean()

    cost = cost_bps / 1e4
    dates = np.sort(oos["date"].unique())
    pnls, hits = [], []
    i = 0
    while i < len(dates):
        d = dates[i]
        g = oos[oos["date"] == d]
        if len(g) >= top_k:
            picks = g.nlargest(top_k, "proba")
            pnls.append(picks["fwd_excess"].mean() - cost * 2)
            hits.append((picks["label"] == 1).mean())
        i += HORIZON
    pnl = pd.Series(pnls)
    ann_ret = pnl.mean() * (252 / HORIZON)
    ann_vol = pnl.std() * np.sqrt(252 / HORIZON) if len(pnl) > 2 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else 0.0

    return {
        "auc": round(float(auc), 4),
        "accuracy": round(float(acc), 4),
        "baseline_up_rate": round(float(y.mean()), 4),
        "fold_auc_mean": round(float(np.mean(fold_aucs)), 4),
        "fold_auc_std": round(float(np.std(fold_aucs)), 4),
        "fold_auc_positive_pct": round(float(np.mean(np.array(fold_aucs) > 0.5)) * 100, 0),
        "top_quintile_excess_per_10d": round(float(q_top), 4),
        "quintile_spread_per_10d": round(float(q_top - q_bot), 4),
        "portfolio_ann_excess_pct": round(float(ann_ret) * 100, 1),
        "portfolio_sharpe": round(float(sharpe), 2),
        "avg_hit_rate_pct": round(float(np.mean(hits)) * 100, 1),
        "n_rebalances": len(pnl),
        "n_predictions": int(len(oos)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    ap.add_argument("--test-months", type=int, default=2)
    ap.add_argument("--cost-bps", type=float, default=10)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    symbols = load_symbols()
    close_panel, highs, lows, vols = fetch_panel(symbols, period=args.period)
    df = build_dataset(close_panel, highs, lows, vols, horizon=HORIZON)
    df = df[df["symbol"] != "^NSEI"]
    print(f"Dataset: {len(df)} rows, {df['symbol'].nunique()} symbols, "
          f"UP-rate={df['label'].mean() * 100:.1f}%\n")

    dates_all = pd.to_datetime(df["date"])
    month_key = dates_all.dt.to_period("M")
    unique_months = sorted(month_key.unique())

    trained_models = {
        "LogisticRegression": make_model("logreg"),
        "RandomForest": make_model("rf"),
        "XGBoost": make_model("xgb"),
    }
    factor_models = ["liq", "rev", "mom", "composite"]

    oos_parts = {name: [] for name in list(trained_models) + factor_models}
    start_pos = 12

    while start_pos < len(unique_months) - 1:
        tm = unique_months[start_pos:start_pos + args.test_months]
        test_mask_raw = month_key.isin(tm)
        test_start_date = pd.Period(tm[0]).start_time
        train_mask = dates_all < test_start_date - pd.Timedelta(days=HORIZON + 5)

        test_df = df[test_mask_raw & df["fwd_ret"].notna()]
        train_df = df[train_mask & df["fwd_ret"].notna()]
        if len(train_df) < 5000 or test_df.empty:
            start_pos += args.test_months
            continue

        for name, mdl in trained_models.items():
            m = make_model({"LogisticRegression": "logreg", "RandomForest": "rf",
                            "XGBoost": "xgb"}[name])
            m.fit(train_df[FEATURES], train_df["label"])
            proba = m.predict_proba(test_df[FEATURES])[:, 1]
            r = test_df[["date", "symbol", "fwd_excess", "label"]].copy()
            r["proba"] = proba
            r["fold"] = start_pos
            oos_parts[name].append(r)

        for fname in factor_models:
            r = test_df[["date", "symbol", "fwd_excess", "label"]].copy()
            r["proba"] = factor_score_probas(test_df, fname)
            r["fold"] = start_pos
            oos_parts[fname].append(r)

        print(f"Fold {tm[0]}..{tm[-1]} done "
              f"(train={len(train_df)}, test={len(test_df)})")
        start_pos += args.test_months

    print(f"\n{'=' * 100}")
    print("MODEL COMPARISON - IDENTICAL CONDITIONS "
          f"(NIFTY100 pooled, {HORIZON}d horizon, leakage-safe walk-forward)")
    print("=" * 100)
    header = (f"{'Model':<20} {'AUC':>6} {'Acc%':>6} {'BaseUp':>7} {'FoldAUC':>8} "
              f"{'+Fold%':>7} {'TopQ/10d':>9} {'Spread':>7} {'AnnExc%':>8} {'Sharpe':>7} {'Hit%':>6}")
    print(header)
    print("-" * 100)

    results = {}
    for name, parts in oos_parts.items():
        oos = pd.concat(parts, ignore_index=True)
        res = evaluate_oos(oos, cost_bps=args.cost_bps, top_k=args.top_k)
        res["model"] = name
        results[name] = res
        print(f"{name:<20} {res['auc']:>6.4f} {res['accuracy'] * 100:>5.1f}% "
              f"{res['baseline_up_rate'] * 100:>6.1f}% {res['fold_auc_mean']:>8.4f} "
              f"{res['fold_auc_positive_pct']:>6.0f}% "
              f"{res['top_quintile_excess_per_10d'] * 100:>8.3f}% "
              f"{res['quintile_spread_per_10d'] * 100:>6.3f}% "
              f"{res['portfolio_ann_excess_pct']:>7.1f}% {res['portfolio_sharpe']:>7.2f} "
              f"{res['avg_hit_rate_pct']:>5.1f}%")

    os.makedirs("outputs", exist_ok=True)
    out = {"generated": datetime.now().isoformat(),
           "conditions": {"universe": "NIFTY100 pooled", "horizon_days": HORIZON,
                          "validation": "purged expanding walk-forward, 10d embargo",
                          "test_fold_months": args.test_months,
                          "cost_bps_side": args.cost_bps, "top_k": args.top_k},
           "results": results}
    with open("outputs/model_comparison_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved outputs/model_comparison_results.json")


if __name__ == "__main__":
    main()
