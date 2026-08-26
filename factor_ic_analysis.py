"""Factor IC analysis: which raw features actually predict cross-sectional excess returns?

Three t-stats are printed per factor (see stats_utils). The naive
t = mean/(std/sqrt(n)) is inflated ~10x here, because each daily IC observation
describes a 10-day forward return and so overlaps the next nine; it is shown
only to make that inflation visible. **Judge factors on `t_nonovlp`** -- the
same test on every 10th observation, which is exactly calibrated on simulated
noise. `t_HAC` (Newey-West, lags = 2*horizon-1) has more power but still
over-rejects ~2x, so use it as a cross-check only.

With ~30 factors tested, ~1.5 will clear 5% by chance; a lone survivor is noise.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pooled_model_v1 import load_symbols, fetch_panel, build_dataset, FEATURES
from stats_utils import mean_tstats

HORIZON = 10

symbols = load_symbols()
close_panel, highs, lows, vols = fetch_panel(symbols, period="5y")
df = build_dataset(close_panel, highs, lows, vols, horizon=HORIZON)

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
    st = mean_tstats(ics, horizon=HORIZON)
    rows.append({"factor": col,
                 "IC_mean": round(st["mean"], 4),
                 "t_nonovlp": round(st["t_nonoverlap"], 2),
                 "t_HAC": round(st["t_hac"], 2),
                 "t_naive": round(st["t_naive"], 2),
                 "pos_days": round((ics > 0).mean(), 2),
                 "sig_5pct": abs(st["t_nonoverlap"]) >= 1.96})

res = pd.DataFrame(rows)
res = res.reindex(res["t_nonovlp"].abs().sort_values(ascending=False).index)
print(f"\nSpearman IC vs {HORIZON}-day forward excess return (last 2 years).")
print(f"n_effective ~= n_days / {HORIZON}. Judge on t_nonovlp (calibrated); "
      f"t_HAC over-rejects ~2x, t_naive ~10x:")
print(res.to_string(index=False))
n_sig = int(res["sig_5pct"].sum())
print(f"\n{n_sig} of {len(res)} factors significant at 5% on the calibrated test. "
      f"With {len(res)} factors tested, ~{len(res) * 0.05:.1f} would pass by chance "
      f"alone -- a lone survivor is not evidence.")
