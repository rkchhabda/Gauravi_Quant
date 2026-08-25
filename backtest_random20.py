"""
Backtest: Composite factor strategy on 20 random NIFTY-100 stocks
Period: 2026-04-01 to latest. Non-overlapping 10-day holds, top-K picks.
Benchmark: equal-weight of the same 20 stocks + NIFTY 50.

Usage:
    python backtest_random20.py --seed 42 --top-k 5 --cost-bps 10
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from pooled_model_v1 import load_symbols, fetch_panel, build_dataset

HORIZON = 10
START = "2026-04-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=10)
    args = ap.parse_args()

    import random
    all_syms = [s for s in load_symbols() if s != "^NSEI"]
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(all_syms, 20))
    print("Randomly selected 20 stocks:")
    for i in range(0, 20, 5):
        print("  " + ", ".join(picked[i:i + 5]))

    close_panel, highs, lows, vols = fetch_panel(picked + ["^NSEI"], period="2y")
    df = build_dataset(close_panel, highs, lows, vols, horizon=HORIZON)
    df = df[df["symbol"] != "^NSEI"]

    oos = df[pd.to_datetime(df["date"]) >= START].dropna(
        subset=["fwd_excess", "liq", "ret_20", "label"]).copy()
    n_days = oos["date"].nunique()
    print(f"\nBacktest window: {pd.to_datetime(oos['date']).min().date()} to "
          f"{pd.to_datetime(oos['date']).max().date()} ({n_days} trading days)")

    # Cross-sectional ranks within the 20-stock universe
    oos["rank_liq"] = oos.groupby("date")["liq"].rank(pct=True)
    oos["rank_rev"] = 1 - oos.groupby("date")["ret_20"].rank(pct=True)
    oos["score"] = oos["rank_liq"] + oos["rank_rev"]

    cost = args.cost_bps / 1e4
    dates = np.sort(oos["date"].unique())
    trades = []
    i = 0
    while i < len(dates):
        d = dates[i]
        g = oos[oos["date"] == d]
        if len(g) < args.top_k:
            i += 1
            continue
        picks = g.nlargest(args.top_k, "score")
        strat_pnl = picks["fwd_ret"].mean() - cost * 2
        bench_pnl = g["fwd_ret"].mean() - cost * 2
        trades.append({
            "entry_date": pd.to_datetime(d).strftime("%Y-%m-%d"),
            "picks": ", ".join(picks["symbol"].str.replace(".NS", "")),
            "strat_ret": round(float(strat_pnl) * 100, 2),
            "bench_ret": round(float(bench_pnl) * 100, 2),
            "excess": round(float(strat_pnl - bench_pnl) * 100, 2),
            "hit_rate": round(float((picks["label"] == 1).mean()) * 100, 0),
        })
        i += HORIZON

    t = pd.DataFrame(trades)
    t["strat_eq"] = (1 + t["strat_ret"] / 100).cumprod()
    t["bench_eq"] = (1 + t["bench_ret"] / 100).cumprod()

    print(f"\n{'=' * 88}")
    print(f"{'Entry':<12} {'Picks':<38} {'Strat%':>8} {'Bench%':>8} {'Excess':>8} {'Hit%':>6}")
    print("-" * 88)
    for _, r in t.iterrows():
        print(f"{r['entry_date']:<12} {r['picks']:<38} {r['strat_ret']:>7.2f}% "
              f"{r['bench_ret']:>7.2f}% {r['excess']:>+7.2f}% {r['hit_rate']:>5.0f}%")
    print("=" * 88)

    sr, sb = t["strat_ret"], t["bench_ret"]
    tot_s = (t["strat_eq"].iloc[-1] - 1) * 100
    tot_b = (t["bench_eq"].iloc[-1] - 1) * 100
    ann_f = 252 / HORIZON
    ann_s = (sr.mean() / 100) * ann_f * 100
    ann_b = (sb.mean() / 100) * ann_f * 100
    vol_s = (sr.std() / 100) * np.sqrt(ann_f) * 100
    sharpe = (sr.mean() / 100) * ann_f / ((sr.std() / 100) * np.sqrt(ann_f)) if sr.std() > 0 else 0
    dd = ((t["strat_eq"] / t["strat_eq"].cummax()) - 1).min() * 100
    dd_b = ((t["bench_eq"] / t["bench_eq"].cummax()) - 1).min() * 100

    print(f"\nSUMMARY ({len(t)} rebalances x top-{args.top_k}, "
          f"{HORIZON}d holds, {args.cost_bps:.0f}bps/side costs)")
    print(f"  Strategy total return   : {tot_s:+.2f}%   (annualized ~{ann_s:+.1f}%)")
    print(f"  Equal-weight benchmark  : {tot_b:+.2f}%   (annualized ~{ann_b:+.1f}%)")
    print(f"  Excess vs benchmark     : {tot_s - tot_b:+.2f}%")
    print(f"  Strategy Sharpe         : {sharpe:.2f}   (vol ~{vol_s:.1f}% ann.)")
    print(f"  Max drawdown            : strategy {dd:.1f}% vs benchmark {dd_b:.1f}%")
    print(f"  Avg directional hit     : {t['hit_rate'].mean():.1f}%")
    wins = (t["excess"] > 0).sum()
    print(f"  Rebalances beating bench: {wins}/{len(t)}")

    os.makedirs("outputs", exist_ok=True)
    out = {
        "generated": datetime.now().isoformat(),
        "seed": args.seed,
        "universe": picked,
        "window": {"start": str(pd.to_datetime(oos['date']).min().date()),
                   "end": str(pd.to_datetime(oos['date']).max().date())},
        "top_k": args.top_k, "cost_bps": args.cost_bps,
        "rebalances": t.drop(columns=["strat_eq", "bench_eq"]).to_dict("records"),
        "summary": {
            "strategy_total_pct": round(tot_s, 2), "benchmark_total_pct": round(tot_b, 2),
            "strategy_annualized_pct": round(ann_s, 1), "benchmark_annualized_pct": round(ann_b, 1),
            "sharpe": round(sharpe, 2), "max_dd_strategy_pct": round(dd, 1),
            "max_dd_benchmark_pct": round(dd_b, 1), "avg_hit_rate_pct": round(float(t['hit_rate'].mean()), 1),
        },
    }
    path = f"outputs/backtest_random20_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
