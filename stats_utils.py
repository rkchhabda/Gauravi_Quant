"""Significance tests for overlapping-horizon factor statistics.

Why this module exists
----------------------
Daily-sampled statistics about an H-day forward return are badly
autocorrelated: consecutive observations share H-1 of their H days. The naive
t-statistic

    t = mean / (std / sqrt(n))

therefore treats ~n/H independent observations as n, and overstates
significance by roughly sqrt(H). At H=10 that is a factor of ~3.2 -- enough to
turn noise (t~1.4) into an apparently decisive result (t~4.4).

Use `mean_tstats()`. Judge on `t_nonoverlap`: on simulated pure-noise
overlapping windows (H=10, n=500, 400 trials) it false-positives 7.0% of the
time against a nominal 5% -- slightly liberal, but the only one of the three
that is close. `t_hac` (Newey-West) came in at 9.5%, so treat it as a
cross-check rather than proof. `t_naive` rejected **57.8%** of the time on the
same pure noise; it is returned only so the inflation is visible rather than
hidden. These figures are re-measured by tests/test_stats_utils.py.
"""

import numpy as np


def newey_west_se(x, lags=None):
    """Newey-West (Bartlett-kernel) HAC standard error of the mean of `x`.

    se^2 = (g0 + 2 * sum_k w_k * g_k) / n,  w_k = 1 - k/(lags+1)

    where g_k is the lag-k autocovariance. `lags` should be at least H-1 for an
    H-day overlapping window; defaults to the Newey-West rule of thumb
    floor(4 * (n/100)^(2/9)).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = int(max(0, min(lags, n - 2)))

    d = x - x.mean()
    var = float(d @ d) / n  # gamma_0
    for k in range(1, lags + 1):
        gamma_k = float(d[k:] @ d[:-k]) / n
        var += 2.0 * (1.0 - k / (lags + 1.0)) * gamma_k
    if var <= 0:  # HAC variance can go negative in small samples
        return np.nan
    return np.sqrt(var / n)


def mean_tstats(x, horizon=10):
    """Naive, HAC and non-overlapping t-statistics for mean(x) != 0.

    `x` is a daily series of a statistic about an `horizon`-day forward return
    (e.g. a per-date IC, or a per-date long-short spread).

    Returns a dict with:
        mean            mean of x
        n               number of usable observations
        t_naive         mean / (std/sqrt(n)) -- inflated ~2.5-3x, never quote
        t_hac           Newey-West, lags = 2*horizon-1; higher power but still
                        over-rejects (~10% actual at 5% nominal)
        t_nonoverlap    same test on every horizon-th observation. Independent
                        subsample, near-calibrated (7.0% at 5% nominal);
                        this is the one to judge on.
        n_effective     n / horizon, the honest sample size
        inflation       t_naive / t_hac
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    out = {"mean": float(x.mean()) if n else np.nan, "n": int(n),
           "n_effective": round(n / horizon, 1) if n else 0.0,
           "t_naive": np.nan, "t_hac": np.nan, "t_nonoverlap": np.nan,
           "inflation": np.nan}
    if n < 3:
        return out

    sd = x.std(ddof=1)
    if sd > 0:
        out["t_naive"] = float(x.mean() / (sd / np.sqrt(n)))

    se_hac = newey_west_se(x, lags=max(1, 2 * horizon - 1))
    if se_hac and not np.isnan(se_hac) and se_hac > 0:
        out["t_hac"] = float(x.mean() / se_hac)
        if out["t_naive"] and not np.isnan(out["t_naive"]) and out["t_hac"] != 0:
            out["inflation"] = float(out["t_naive"] / out["t_hac"])

    sub = x[::horizon]
    if sub.size >= 3:
        sub_sd = sub.std(ddof=1)
        if sub_sd > 0:
            out["t_nonoverlap"] = float(sub.mean() / (sub_sd / np.sqrt(sub.size)))
    return out


def format_tstats(label, st, horizon=10):
    """One-line human-readable summary.

    The verdict comes from t_nonoverlap (nearest to calibrated). t_hac is shown as
    a higher-power cross-check; when the two disagree, the result is fragile.
    """
    t = st["t_nonoverlap"]
    if np.isnan(t):
        verdict = "inconclusive"
    elif abs(t) >= 2.58:
        verdict = "significant p<0.01"
    elif abs(t) >= 1.96:
        verdict = "significant p<0.05"
    elif abs(t) >= 1.65:
        verdict = "marginal p<0.10"
    else:
        verdict = "NOT significant"

    hac = st["t_hac"]
    disagree = (not np.isnan(hac) and not np.isnan(t)
                and (abs(hac) >= 1.96) != (abs(t) >= 1.96))
    flag = "  [HAC and non-overlap DISAGREE -> fragile]" if disagree else ""
    return (f"{label}: mean={st['mean']:.5f}  "
            f"t_nonoverlap={t:.2f} [{verdict}]  "
            f"t_HAC={hac:.2f}  "
            f"t_naive={st['t_naive']:.2f} (inflated {st['inflation']:.1f}x)  "
            f"n={st['n']} daily obs = ~{st['n_effective']} independent "
            f"({horizon}d overlap){flag}")
