"""Phase 1 signal test: do delivery-based factors add cross-sectional IC?

Fetches the full NIFTY-100 via the cached jugaad data layer, builds the
existing best factors (reversal ret_20, liquidity) plus NEW delivery factors,
and reports calibrated non-overlap IC t-stats (stats_utils) vs 10-day forward
excess return. Judge on t_nonovlp.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_layer import fetch_panel_jugaad, BENCH
from pooled_model_v1 import load_symbols
from stats_utils import mean_tstats

H = 10
close, highs, lows, vols, deliv = fetch_panel_jugaad(load_symbols(), years=5)
syms = [s for s in close.columns if s != BENCH]
if BENCH in close.columns:
    bench = close[BENCH]
else:
    # equal-weight universe daily return -> synthetic benchmark price
    bench = (1 + close[syms].pct_change().mean(axis=1)).cumprod()
    print("[deliv_ic] ^NSEI unavailable; using equal-weight universe benchmark")

rows = []
for s in syms:
    c = close[s].dropna()
    if len(c) < 400:
        continue
    d = deliv.get(s)
    v = vols.get(s)
    fwd = c.shift(-H) / c - 1
    fwd_b = bench.reindex(c.index).shift(-H) / bench.reindex(c.index) - 1
    df = pd.DataFrame({
        "date": c.index,
        "ret_20": c.pct_change(20),
        "liq": (v * c).rolling(20).mean() if v is not None else np.nan,
        "deliv": d.reindex(c.index) if d is not None else np.nan,
        "deliv_chg20": (d.reindex(c.index) - d.reindex(c.index).rolling(20).mean())
                        if d is not None else np.nan,
        "deliv_x_ret": (d.reindex(c.index) * c.pct_change(20))
                        if d is not None else np.nan,
        "fwd_excess": fwd - fwd_b,
    })
    df["sym"] = s
    rows.append(df)

data = pd.concat(rows).dropna(subset=["fwd_excess"])
FACTORS = ["ret_20", "liq", "deliv", "deliv_chg20", "deliv_x_ret"]
# short-term reversal: invert ret_20 so "recent losers rank high"
data["ret_20"] = -data["ret_20"]

out = []
for col in FACTORS:
    ics = []
    for _, g in data.groupby("date"):
        g = g.dropna(subset=[col, "fwd_excess"])
        if len(g) < 30:
            continue
        ic = spearmanr(g[col], g["fwd_excess"]).statistic
        if not np.isnan(ic):
            ics.append(ic)
    ics = np.array(ics)
    st = mean_tstats(ics, horizon=H)
    out.append({"factor": col, "IC_mean": round(st["mean"], 4),
                "t_nonovlp": round(st["t_nonoverlap"], 2),
                "t_HAC": round(st["t_hac"], 2),
                "pos_days": round((ics > 0).mean(), 2),
                "sig": abs(st["t_nonoverlap"]) >= 1.96})

res = pd.DataFrame(out).sort_values("t_nonovlp", key=lambda s: s.abs(),
                                    ascending=False)
print("\n=== Delivery-factor IC (jugaad, 5y, NIFTY-100) ===")
print("(ret_20 shown inverted = reversal). Judge on t_nonovlp:")
print(res.to_string(index=False))
print(f"\n{int(res['sig'].sum())} of {len(res)} significant at 5% (calibrated).")
