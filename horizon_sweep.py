"""Improvement #1: does a longer holding horizon reach 55%?

Sweeps H = 10, 20, 30 days for the three models on identical cached data.
Metric: top-quintile long-basket hit-rate vs benchmark (non-overlapping holds).
"""
import numpy as np
import pandas as pd

from data_layer import fetch_hybrid, BENCH
from pooled_model_v1 import load_symbols
from fundamentals import fundamentals_panel, attach_fundamentals

syms_all = load_symbols()
close, highs, lows, vols, deliv = fetch_hybrid(syms_all, period="5y")
syms = [s for s in close.columns if s != BENCH]
bench = close[BENCH] if BENCH in close.columns else \
    (1 + close[syms].pct_change().mean(axis=1)).cumprod()
fund = fundamentals_panel([s for s in syms_all if not s.startswith("^")])

results = []
for H in (10, 20, 30):
    rows = []
    for s in syms:
        c = close[s].dropna()
        if len(c) < 400:
            continue
        d = deliv.get(s)
        fwd = c.shift(-H) / c - 1
        fb = bench.reindex(c.index).shift(-H) / bench.reindex(c.index) - 1
        rows.append(pd.DataFrame({
            "date": c.index, "sym": s, "close": c.values,
            "mom_12_1": c.shift(21) / c.shift(252) - 1,
            "rev": -c.pct_change(20),
            "deliv_x_ret": (d.reindex(c.index) * c.pct_change(20))
                            if d is not None else np.nan,
            "fwd_excess": fwd - fb,
        }))
    daily = pd.concat(rows, ignore_index=True)
    data = attach_fundamentals(daily, fund)
    data = data.dropna(subset=["fwd_excess"]).reset_index(drop=True)
    rb = set(np.sort(data["date"].unique())[::H])   # non-overlapping holds
    data = data[data["date"].isin(rb)].reset_index(drop=True)

    def rk(col):
        return data.groupby("date")[col].rank(pct=True)

    data["shipped"] = (rk("mom_12_1") + rk("rev")) / 2
    data["newcore"] = (rk("rev") + (1 - rk("deliv_x_ret"))) / 2
    # earn_yield: positive sign, economically defensible (safer than inverted ROE)
    data["fund3"] = (rk("rev") + (1 - rk("deliv_x_ret")) + rk("earn_yield")) / 3

    for m in ("shipped", "newcore", "fund3"):
        d = data.dropna(subset=[m])
        q = d.groupby("date")[m].transform(lambda x: x.quantile(0.8))
        top = d[d[m] >= q]
        if not len(top):
            continue
        hr = (top["fwd_excess"] > 0).mean()
        n_ind = len(top)                      # holds are non-overlapping
        se = np.sqrt(hr * (1 - hr) / max(n_ind, 1))
        results.append({"H": H, "model": m, "hit_rate_%": round(hr * 100, 2),
                        "SE_pp": round(se * 100, 2), "n_picks": n_ind,
                        "rebalances": len(rb)})

res = pd.DataFrame(results)
print("\n=== Horizon sweep: top-quintile hit-rate (non-overlapping holds) ===")
print(res.to_string(index=False))
best = res.loc[res["hit_rate_%"].idxmax()]
print(f"\nBest: {best['model']} at H={int(best['H'])}d = "
      f"{best['hit_rate_%']}% (SE {best['SE_pp']}pp)")
print("Phase 1 target = 55%. A result is only meaningful if "
      "hit_rate - 55 exceeds ~2*SE.")
