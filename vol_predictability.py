"""Is VOLATILITY predictable here, even though direction is not?

Direction failed: top-quintile hit-rate = base rate, 0 factors significant.
But volatility clusters -- high-vol names tend to STAY high-vol -- so a vol
forecast can carry real, significant signal where a return forecast does not.

If this works it reframes the product from "which stocks go up" (dead) to
"how risky is each name next month" (a defensible risk-overlay / screening tool).

Metric parallels the direction test exactly so the comparison is fair:
  - feature  = trailing 20d realized vol (annualised), cross-sectional rank
  - target   = forward H-day realized vol
  - IC       = Spearman, non-overlapping rebalances only (honest t-stat)
  - economic = top vs bottom forecast-vol quintile: do their realized vols differ?
"""
import numpy as np
import pandas as pd

import gauravi_data as gd

H = 30
ANN = np.sqrt(252)

f = gd.load_factors()
panel = gd.load_panel()

# Build realized-vol features/target from the panel (daily log returns per name).
rows = []
for s, g in panel.sort_values("date").groupby("sym", sort=False):
    c = g["close"].astype(float)
    r = np.log(c / c.shift(1))
    vol20 = r.rolling(20).std() * ANN                       # trailing feature
    vol60 = r.rolling(60).std() * ANN                       # slower feature
    fwd_vol = r.shift(-1).rolling(H).std().shift(-(H - 1)) * ANN  # forward target
    rng = ((g["high"] - g["low"]) / c).rolling(20).mean()   # avg daily range
    rows.append(pd.DataFrame({
        "date": g["date"].values, "sym": s,
        "vol20": vol20.values, "vol60": vol60.values,
        "range20": rng.values, "fwd_vol": fwd_vol.values,
    }))
d = pd.concat(rows, ignore_index=True).dropna(subset=["fwd_vol"])

# bring in delivery % -- does low delivery (churn) predict higher forward vol?
if "deliv_pct" in panel.columns:
    d = d.merge(panel[["date", "sym", "deliv_pct"]], on=["date", "sym"], how="left")

dates = np.sort(d["date"].unique())
nonovlp = set(dates[::H])


def ic(feature, sign=1):
    sub = d[["date", feature, "fwd_vol"]].dropna()
    per = (sub.groupby("date")
              .apply(lambda g: (sign * g[feature]).corr(g["fwd_vol"], method="spearman")
                     if len(g) > 5 else np.nan)
              .dropna())
    indep = per[per.index.isin(nonovlp)]
    n = len(indep)
    t = indep.mean() / (indep.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
    return per.mean(), t, n


print(f"=== Volatility predictability (H={H}d, non-overlapping) ===")
print(f"{'feature':12s} {'IC':>8s} {'t_nonovlp':>10s} {'n':>5s}  {'significant':>11s}")
for feat, sign in [("vol20", 1), ("vol60", 1), ("range20", 1), ("deliv_pct", -1)]:
    if feat not in d.columns:
        continue
    m, t, n = ic(feat, sign)
    sig = "YES" if abs(t) >= 1.96 else "no"
    lbl = feat + ("(inv)" if sign < 0 else "")
    print(f"{lbl:12s} {m:+8.4f} {t:+10.2f} {n:5d}  {sig:>11s}")

# economic magnitude: split by forecast vol (=vol20), compare realized fwd_vol
dd = d[d["date"].isin(nonovlp)].dropna(subset=["vol20", "fwd_vol"])
q = dd.groupby("date")["vol20"].transform(lambda x: pd.qcut(x, 5, labels=False,
                                                            duplicates="drop"))
by_q = dd.assign(q=q).groupby("q")["fwd_vol"].mean()
print("\nForward realized vol by predicted-vol quintile (Q0=calm .. Q4=stormy):")
for qi, v in by_q.items():
    print(f"  Q{int(qi)}: {v*100:5.1f}% annualised")
if len(by_q) >= 5:
    spread = (by_q.iloc[-1] - by_q.iloc[0]) * 100
    ratio = by_q.iloc[-1] / by_q.iloc[0]
    print(f"  spread Q4-Q0: {spread:.1f}pp   ratio: {ratio:.2f}x")

# direction sanity check on the SAME rebalances, for contrast
dr = f.dropna(subset=["rev", "fwd_excess"])
dr = dr[dr["date"].isin(set(np.sort(dr['date'].unique())[::H]))]
per = (dr.groupby("date")
         .apply(lambda g: g["rev"].corr(g["fwd_excess"], method="spearman")
                if len(g) > 5 else np.nan).dropna())
t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))
print(f"\nContrast -- direction (rev vs fwd_excess): IC {per.mean():+.4f}, "
      f"t {t:+.2f}  <- the dead one")
