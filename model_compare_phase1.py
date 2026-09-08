"""Compare shipped model vs new delivery-conditioned model on the SAME data.

Metric = directional hit-rate of the top-quintile long basket:
fraction of long picks whose 10-day forward return beats the universe
(equal-weight) benchmark. This is the number the 55% target is defined on.

Models (cross-sectional percentile-rank blends, rebalanced every 10 days):
  shipped : mom_12_1 + reversal(-ret_20)             (current product)
  newcore : reversal(-ret_20) + deliv_x_ret          (Phase-1 candidate)
"""
import numpy as np
import pandas as pd

from data_layer import fetch_hybrid, BENCH
from pooled_model_v1 import load_symbols

H = 10
close, highs, lows, vols, deliv = fetch_hybrid(load_symbols(), period="5y")
syms = [s for s in close.columns if s != BENCH]
bench = close[BENCH] if BENCH in close.columns else \
    (1 + close[syms].pct_change().mean(axis=1)).cumprod()

rows = []
for s in syms:
    c = close[s].dropna()
    if len(c) < 400:
        continue
    d = deliv.get(s)
    fwd = c.shift(-H) / c - 1
    fb = bench.reindex(c.index).shift(-H) / bench.reindex(c.index) - 1
    rows.append(pd.DataFrame({
        "date": c.index, "sym": s,
        "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "rev": -c.pct_change(20),
        "deliv_x_ret": (d.reindex(c.index) * c.pct_change(20)) if d is not None else np.nan,
        "fwd_excess": fwd - fb,
    }))
data = pd.concat(rows).reset_index(drop=True)

# rebalance dates: every H trading days
all_dates = np.sort(data["date"].unique())
rebal = set(all_dates[::H])
data = data[data["date"].isin(rebal)].dropna(subset=["fwd_excess"])


def rank(g, col):
    return g.groupby("date")[col].rank(pct=True)


def hit_rate(score_col):
    d = data.dropna(subset=[score_col]).copy()
    top = d[d[score_col] >= d.groupby("date")[score_col].transform(
        lambda x: x.quantile(0.8))]
    return (top["fwd_excess"] > 0).mean(), len(top)


data["shipped"] = (rank(data, "mom_12_1") + rank(data, "rev")) / 2
d2 = data.dropna(subset=["deliv_x_ret"])
# newcore: invert deliv_x_ret (it had negative IC) so high score = expected winner
data["newcore"] = (rank(data, "rev") + (1 - rank(data, "deliv_x_ret"))) / 2

print(f"\n=== Top-quintile long-basket directional hit-rate "
      f"({len(rebal)} rebalances, {H}-day holds) ===")
for name in ["shipped", "newcore"]:
    hr, n = hit_rate(name)
    if n == 0:
        print(f"  {name:8s}: no picks (delivery data missing)")
        continue
    se = np.sqrt(hr * (1 - hr) / max(n / H, 1))  # deflate n by overlap
    print(f"  {name:8s}: {hr*100:5.2f}%   (n_picks={n}, ~SE {se*100:.2f}pp)")
print("\nTarget for Phase 1 = 55%. SE uses independent-hold count (n/H).")
