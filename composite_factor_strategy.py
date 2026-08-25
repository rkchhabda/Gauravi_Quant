"""Composite factor strategy: liquidity + short-term reversal, honest OOS eval."""
import numpy as np
import pandas as pd

from pooled_model_v1 import load_symbols, fetch_panel, build_dataset

HORIZON = 10
COST = 10 / 1e4


def run(top_k=5):
    symbols = load_symbols()
    close_panel, highs, lows, vols = fetch_panel(symbols, period="5y")
    df = build_dataset(close_panel, highs, lows, vols, horizon=HORIZON)

    # Composite: high liquidity + recent losers (reversal). Pure cross-sectional ranks.
    df["rank_liq"] = df.groupby("date")["liq"].rank(pct=True)
    df["rank_rev"] = 1 - df.groupby("date")["ret_20"].rank(pct=True)
    df["score"] = df["rank_liq"] + df["rank_rev"]

    oos = df[pd.to_datetime(df["date"]) >= pd.to_datetime(df["date"]).max() - pd.Timedelta(days=730)].copy()
    oos = oos.dropna(subset=["fwd_excess", "liq", "ret_20", "label"])

    from scipy.stats import spearmanr
    ics = []
    for d, g in oos.groupby("date"):
        ic = spearmanr(g["score"], g["fwd_excess"]).statistic
        if not np.isnan(ic):
            ics.append(ic)
    ics = np.array(ics)
    print(f"Composite IC: {ics.mean():.4f}  IR={ics.mean() / (ics.std() / np.sqrt(len(ics))):.2f}  "
          f"pos_days={((ics > 0).mean()) * 100:.0f}%")

    oos["bucket"] = oos.groupby("date")["score"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
    qs = oos.groupby("bucket").agg(mean_excess=("fwd_excess", "mean"),
                                   hit=("label", "mean"), n=("fwd_ret", "size"))
    print("\nQuintile spread (Q1=low score, Q5=high score):")
    print(qs.round(4).to_string())

    dates = np.sort(oos["date"].unique())
    trades = []
    i = 0
    while i < len(dates):
        d = dates[i]
        g = oos[oos["date"] == d]
        if len(g) < top_k:
            i += 1
            continue
        picks = g.nlargest(top_k, "score")
        pnl = picks["fwd_excess"].mean() - COST * 2
        trades.append({"date": d, "pnl": pnl, "hit": (picks["label"] == 1).mean()})
        i += HORIZON
    port = pd.DataFrame(trades)
    port["equity"] = (1 + port["pnl"]).cumprod()
    ann_ret = port["pnl"].mean() * (252 / HORIZON)
    ann_vol = port["pnl"].std() * np.sqrt(252 / HORIZON)
    dd = (port["equity"] / port["equity"].cummax() - 1).min()
    print(f"\nTop-{top_k} composite portfolio ({len(port)} rebalances, non-overlapping {HORIZON}d holds):")
    print(f"  Total excess return : {(port['equity'].iloc[-1] - 1) * 100:.1f}%")
    print(f"  Annualized excess   : {ann_ret * 100:.1f}%")
    print(f"  Sharpe              : {ann_ret / ann_vol:.2f}")
    print(f"  Max drawdown        : {dd * 100:.1f}%")
    print(f"  Hit rate            : {port['hit'].mean() * 100:.1f}%")

    # Quintile long-short spread per period
    ls = []
    for d, g in oos.groupby("date"):
        q5 = g.nlargest(max(len(g) // 5, 1), "score")["fwd_excess"].mean()
        q1 = g.nsmallest(max(len(g) // 5, 1), "score")["fwd_excess"].mean()
        ls.append(q5 - q1)
    ls = np.array(ls)
    print(f"\nQ5-Q1 daily spread: mean={ls.mean() * 100:.3f}%  "
          f"t={ls.mean() / (ls.std() / np.sqrt(len(ls))):.2f}")


if __name__ == "__main__":
    run()
