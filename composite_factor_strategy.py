"""Composite factor strategy: liquidity + short-term reversal, honest OOS eval.

Significance is reported with Newey-West HAC t-stats (see stats_utils). The
naive t = mean/(std/sqrt(n)) that earlier versions printed is inflated ~sqrt(10)
because daily observations of a 10-day forward return overlap by 9 days; it is
still shown so the inflation is visible, but do not quote it.

Caveat that no statistic can fix: the liquidity+reversal PAIR was selected by
inspecting this same window. Treat the result as a hypothesis, not a validated
edge, until it holds on data chosen after the fact.
"""
import numpy as np
import pandas as pd

from pooled_model_v1 import load_symbols, fetch_panel, build_dataset
from stats_utils import mean_tstats, format_tstats

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
    ic_st = mean_tstats(ics, horizon=HORIZON)
    print(f"Composite IC: mean={ic_st['mean']:.4f}  pos_days={((ics > 0).mean()) * 100:.0f}%")
    print("  " + format_tstats("IC != 0", ic_st, horizon=HORIZON))

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
    ls_st = mean_tstats(ls, horizon=HORIZON)
    print(f"\nQ5-Q1 daily spread: mean={ls.mean() * 100:.3f}% per {HORIZON}d")
    print("  " + format_tstats("Q5-Q1 != 0", ls_st, horizon=HORIZON))
    print("\nNote: the factor pair was chosen on this same window, so even the "
          "HAC t-stat is optimistic.")


if __name__ == "__main__":
    run()
