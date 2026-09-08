"""Phase 1 evaluation: do point-in-time fundamentals add cross-sectional IC,
and does a reversal + delivery + fundamentals composite beat the shipped model?

Data: adjusted prices (yfinance) + delivery % (jugaad) + annual fundamentals
as-of-merged with a 90-day reporting lag (no lookahead).

Reports calibrated non-overlap IC t-stats and top-quintile hit-rates.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_layer import fetch_hybrid, BENCH
from pooled_model_v1 import load_symbols
from fundamentals import fundamentals_panel, attach_fundamentals
from stats_utils import mean_tstats

H = 10
syms_all = load_symbols()
close, highs, lows, vols, deliv = fetch_hybrid(syms_all, period="5y")
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
        "date": c.index, "sym": s, "close": c.values,
        "mom_12_1": c.shift(21) / c.shift(252) - 1,
        "rev": -c.pct_change(20),
        "deliv_x_ret": (d.reindex(c.index) * c.pct_change(20)) if d is not None else np.nan,
        "fwd_excess": fwd - fb,
    }))
daily = pd.concat(rows, ignore_index=True)

fund = fundamentals_panel([s for s in syms_all if not s.startswith("^")])
data = attach_fundamentals(daily, fund)
data = data.dropna(subset=["fwd_excess"]).reset_index(drop=True)

# rebalance every H trading days
rb = np.sort(data["date"].unique())[::H]
data = data[data["date"].isin(set(rb))].reset_index(drop=True)
print(f"\nrows={len(data)}  rebalances={len(rb)}  symbols={data['sym'].nunique()}")

FUND = ["earn_yield", "roe", "earn_growth", "profit_margin"]
out = []
for col in FUND + ["rev", "deliv_x_ret"]:
    ics = []
    for _, g in data.groupby("date"):
        g = g.dropna(subset=[col, "fwd_excess"])
        if len(g) < 30:
            continue
        ic = spearmanr(g[col], g["fwd_excess"]).statistic
        if not np.isnan(ic):
            ics.append(ic)
    if len(ics) < 10:
        continue
    st = mean_tstats(np.array(ics), horizon=1)  # already non-overlapping
    out.append({"factor": col, "IC_mean": round(st["mean"], 4),
                "t": round(st["t_naive"], 2), "n_dates": len(ics),
                "sig": abs(st["t_naive"]) >= 1.96})
res = pd.DataFrame(out).sort_values("t", key=lambda s: s.abs(), ascending=False)
print("\n=== IC vs 10-day forward excess return (non-overlapping rebalances) ===")
print(res.to_string(index=False))


def rk(col):
    return data.groupby("date")[col].rank(pct=True)


def hit(col):
    d = data.dropna(subset=[col]).copy()
    q = d.groupby("date")[col].transform(lambda x: x.quantile(0.8))
    top = d[d[col] >= q]
    if not len(top):
        return 0.0, 0
    return (top["fwd_excess"] > 0).mean(), len(top)


data["shipped"] = (rk("mom_12_1") + rk("rev")) / 2
data["newcore"] = (rk("rev") + (1 - rk("deliv_x_ret"))) / 2
best_f = res[res["factor"].isin(FUND)].iloc[0]["factor"]
sign = 1 if res[res["factor"] == best_f].iloc[0]["IC_mean"] > 0 else -1
data["fund3"] = (rk("rev") + (1 - rk("deliv_x_ret"))
                 + (rk(best_f) if sign > 0 else 1 - rk(best_f))) / 3

print(f"\n=== Top-quintile hit-rate ({len(rb)} rebalances) === "
      f"[best fundamental: {best_f}, sign {sign:+d}]")
for m in ["shipped", "newcore", "fund3"]:
    hr, n = hit(m)
    se = np.sqrt(hr * (1 - hr) / max(n, 1))
    print(f"  {m:8s}: {hr*100:5.2f}%  (n={n}, SE {se*100:.2f}pp)")
print("\nPhase 1 target = 55%.")
