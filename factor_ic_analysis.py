"""Factor IC analysis: which raw features actually predict cross-sectional excess returns?"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pooled_model_v1 import load_symbols, fetch_panel, build_dataset, FEATURES

symbols = load_symbols()
close_panel, highs, lows, vols = fetch_panel(symbols, period="5y")
df = build_dataset(close_panel, highs, lows, vols, horizon=10)

cutoff = pd.to_datetime(df["date"]).max() - pd.Timedelta(days=730)
df = df[pd.to_datetime(df["date"]) >= cutoff]
print(f"Analyzing {len(df)} rows from {pd.to_datetime(df['date']).min().date()} "
      f"to {pd.to_datetime(df['date']).max().date()}")

rows = []
for col in FEATURES:
    ics = []
    for d, g in df.groupby("date"):
        if len(g) < 30:
            continue
        ic = spearmanr(g[col], g["fwd_excess"]).statistic
        if not np.isnan(ic):
            ics.append(ic)
    ics = np.array(ics)
    t_stat = ics.mean() / (ics.std() / np.sqrt(len(ics))) if len(ics) > 1 else 0
    rows.append({"factor": col, "IC_mean": round(ics.mean(), 4),
                 "IC_ir": round(t_stat, 2), "pos_days": round((ics > 0).mean(), 2)})

res = pd.DataFrame(rows).sort_values("IC_ir", ascending=False)
print("\nSpearman IC vs 10-day forward excess return (last 2 years):")
print(res.to_string(index=False))
