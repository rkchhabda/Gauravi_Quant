"""Build a versioned, validated release of the Gauravi NSE factor dataset.

One command produces everything shipped to customers:

    python build_dataset_release.py                    # full NIFTY-100 release
    python build_dataset_release.py --sample           # free 10-symbol sample
    python build_dataset_release.py --outdir dist/2026.08

Outputs (under --outdir):
    panel_nifty100.parquet     adjusted OHLC + volume + delivery %
    fundamentals_pit.parquet   point-in-time fundamentals (90-day reporting lag)
    factors.parquet            factor values + cross-sectional percentile ranks
    universe.csv               symbols + as-of date
    DATA_DICTIONARY.md         every column, units, caveats
    coverage.csv               non-null % per column
    MANIFEST.json              version, row counts, checksums, validation results

Exits non-zero if any validation check fails, so it is safe to run in CI/cron.
"""
import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data_layer import fetch_hybrid, BENCH
from fundamentals import fundamentals_panel, attach_fundamentals
from pooled_model_v1 import load_symbols

SAMPLE_N = 10
HORIZON = 30          # production horizon (see PROGRESS.md horizon sweep)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def build_panel(symbols, period):
    """Long-form daily panel: date, sym, open/high/low/close, volume, deliv_pct."""
    close, highs, lows, vols, deliv = fetch_hybrid(symbols, period=period)
    syms = [s for s in close.columns if s != BENCH]
    frames = []
    for s in syms:
        c = close[s].dropna()
        if c.empty:
            continue
        frames.append(pd.DataFrame({
            "date": c.index, "sym": s, "close": c.values,
            "high": highs[s].reindex(c.index).values if s in highs else np.nan,
            "low": lows[s].reindex(c.index).values if s in lows else np.nan,
            "volume": vols[s].reindex(c.index).values if s in vols else np.nan,
            "deliv_pct": (deliv[s].reindex(c.index).values
                          if s in deliv else np.nan),
        }))
    panel = pd.concat(frames, ignore_index=True).sort_values(["sym", "date"])
    bench = close[BENCH] if BENCH in close.columns else None
    return panel.reset_index(drop=True), bench


def build_factors(panel, bench, fund, horizon=HORIZON):
    """Factor values + per-date percentile ranks + forward excess return."""
    frames = []
    for s, g in panel.groupby("sym", sort=False):
        g = g.sort_values("date").copy()
        c = g["close"]
        idx = pd.DatetimeIndex(g["date"])
        g["ret_20"] = c.pct_change(20).values
        g["rev"] = -g["ret_20"]
        g["mom_12_1"] = (c.shift(21) / c.shift(252) - 1).values
        g["mom_6m"] = c.pct_change(126).values
        g["liq"] = (g["volume"] * c).rolling(20).mean().values
        g["deliv_chg20"] = (
            g["deliv_pct"]
            - g["deliv_pct"].rolling(20, min_periods=5).mean()).values
        g["deliv_x_ret"] = (g["deliv_pct"] * g["ret_20"]).values
        fwd = np.array(c.shift(-horizon) / c - 1, dtype=float)
        if bench is not None:
            b = bench.reindex(idx)
            fwd = fwd - np.array(b.shift(-horizon) / b - 1, dtype=float)
        g["fwd_excess"] = fwd
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)

    df = attach_fundamentals(df, fund)
    df["earn_yield"] = df["eps"] / df["close"]

    RANKED = ["rev", "mom_12_1", "mom_6m", "liq", "deliv_pct",
              "deliv_chg20", "deliv_x_ret", "earn_yield", "roe",
              "earn_growth", "profit_margin"]
    for col in RANKED:
        if col in df.columns:
            df[f"rank_{col}"] = df.groupby("date")[col].rank(pct=True)

    # Reference composite: plain momentum + reversal, the best-measured model in
    # the horizon sweep (53.09% @ H=30, SE 1.96pp). Delivery/fundamental variants
    # scored WORSE once the jugaad date bug was fixed -- score_fund3 is kept only
    # for reproducing the earlier published comparison, not as a recommendation.
    df["score_mr"] = (df["rank_mom_12_1"] + df["rank_rev"]) / 2
    df["rank_score_mr"] = df.groupby("date")["score_mr"].rank(pct=True)
    df["score_fund3"] = (df["rank_rev"]
                         + (1 - df["rank_deliv_x_ret"])
                         + df["rank_earn_yield"]) / 3
    df["rank_score_fund3"] = df.groupby("date")["score_fund3"].rank(pct=True)
    return df.sort_values(["date", "sym"]).reset_index(drop=True), RANKED


# --------------------------------------------------------------------------
# validation -- each returns (name, passed, detail)
# --------------------------------------------------------------------------
def v_schema(panel, factors, fund):
    need_p = {"date", "sym", "close", "volume", "deliv_pct"}
    need_f = {"sym", "fy_end", "avail_date", "eps", "roe"}
    miss = (need_p - set(panel.columns)) | (need_f - set(fund.columns))
    if miss:
        return "schema", False, f"missing columns: {sorted(miss)}"
    if not pd.api.types.is_datetime64_any_dtype(panel["date"]):
        return "schema", False, "panel.date is not datetime"
    return "schema", True, f"{len(panel.columns)} panel / {len(factors.columns)} factor cols"


def v_no_lookahead(fund):
    """Fundamentals must only become available AFTER the fiscal year ends."""
    bad = fund[pd.to_datetime(fund["avail_date"])
               <= pd.to_datetime(fund["fy_end"])]
    if len(bad):
        return "no_lookahead", False, f"{len(bad)} rows with avail_date <= fy_end"
    return "no_lookahead", True, "all avail_date > fy_end"


def v_returns_sanity(factors):
    """Catches the unadjusted-price bug: a return can never be below -100%."""
    for col in ("ret_20", "fwd_excess"):
        s = factors[col].dropna()
        if s.empty:
            continue
        if (s < -1.0).any():
            n = int((s < -1.0).sum())
            return ("returns_sanity", False,
                    f"{col}: {n} values < -100% -- prices likely UNADJUSTED")
        if s.abs().max() > 10:
            return ("returns_sanity", False,
                    f"{col}: max |value| {s.abs().max():.1f} -- suspicious spike")
    return "returns_sanity", True, "no impossible or extreme returns"


def v_duplicates(panel):
    d = panel.duplicated(subset=["sym", "date"]).sum()
    return "duplicates", d == 0, f"{d} duplicate (sym, date) rows"


def v_coverage(factors, ranked, min_pct=0.30):
    cov = factors[ranked].notna().mean()
    weak = cov[cov < min_pct]
    detail = ", ".join(f"{k}={v:.0%}" for k, v in cov.items())
    if len(weak):
        return ("coverage", False,
                f"below {min_pct:.0%}: {', '.join(weak.index)} | {detail}")
    return "coverage", True, detail


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
DICT_ROWS = [
    ("date", "datetime", "Trading date (NSE calendar)."),
    ("sym", "string", "NSE symbol with .NS suffix."),
    ("close/high/low", "float", "**Split/dividend ADJUSTED** price (yfinance). "
     "Use these for returns."),
    ("volume", "float", "Traded quantity."),
    ("deliv_pct", "float 0-100", "**Delivery percentage** (NSE via jugaad-data). "
     "Share of traded volume actually delivered. Not available from yfinance."),
    ("ret_20 / rev", "float", "20-day return; `rev` is its negation "
     "(short-term reversal)."),
    ("mom_12_1", "float", "12-month momentum skipping the last month."),
    ("liq", "float", "20-day mean of volume x close (traded value)."),
    ("deliv_chg20", "float", "deliv_pct minus its own 20-day mean."),
    ("deliv_x_ret", "float", "deliv_pct x ret_20 (delivery-conditioned reversal)."),
    ("eps/roe/earn_growth/profit_margin", "float", "Annual fundamentals, "
     "**point-in-time**: only visible after `avail_date`."),
    ("earn_yield", "float", "eps / close (value factor, varies daily)."),
    ("avail_date", "datetime", "fy_end + 90 days. The date the figure is "
     "treated as public. Never join on fy_end."),
    ("rank_*", "float 0-1", "Cross-sectional percentile rank of the factor "
     "within that date."),
    ("score_mr", "float 0-1", "Reference composite: mean of rank_mom_12_1 and "
     "rank_rev. Best-measured model in our own testing (~53% hit-rate at "
     "H=30d, SE ~2pp). Not a recommendation."),
    ("score_fund3", "float 0-1", "Legacy composite: mean of rank_rev, "
     "(1 - rank_deliv_x_ret), rank_earn_yield. Kept for reproducibility only -- "
     "it scored WORSE than score_mr after a delivery-data date bug was fixed."),
    ("fwd_excess", "float", f"{HORIZON}-day forward return minus benchmark. "
     "**Target variable -- for research only, not tradable.**"),
]

CAVEATS = """\
## Caveats you must read

1. **Prices are adjusted; delivery data is not price-derived.** `close/high/low`
   come from yfinance with `auto_adjust=True`. NSE/jugaad raw prices are
   **unadjusted** for splits and dividends -- driving returns off raw NSE prices
   produces nonsense (we measured an 11% hit-rate from exactly this bug).
2. **Fundamentals are point-in-time with a 90-day reporting lag.** Join on
   `avail_date`, never on `fy_end`, or you introduce lookahead.
3. **Fundamentals are ANNUAL.** Free quarterly history for NSE names goes back
   only ~5 quarters, too short for multi-year work.
4. **The universe is current NIFTY-100 membership** -- it is NOT point-in-time,
   so backtests carry survivorship bias. Point-in-time membership is on the
   roadmap.
5. **`fwd_excess` is a forward-looking research target.** It contains future
   information by construction. Never use it as a feature.
6. **No factor here is statistically significant.** Best measured directional
   hit-rate is ~53% (SE ~2pp) against a ~51% baseline, and delivery-based
   factors did **not** improve on a plain momentum+reversal model once a
   date-parsing bug in the delivery source was fixed.
   This is data for research, **not investment advice and not a signal service.**
"""


def write_dictionary(path, meta):
    lines = [f"# Data Dictionary -- Gauravi NSE Factor Dataset `{meta['version']}`",
             "", f"Built {meta['built_utc']} | {meta['n_symbols']} symbols | "
             f"{meta['date_min']} to {meta['date_max']}", "",
             "| Column | Type | Meaning |", "|---|---|---|"]
    lines += [f"| `{c}` | {t} | {d} |" for c, t, d in DICT_ROWS]
    lines += ["", CAVEATS]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def sha256(path, blocks=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(blocks), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="output dir (default dist/<version>)")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--version", default=datetime.now().strftime("%Y.%m"))
    ap.add_argument("--sample", action="store_true",
                    help=f"free sample: first {SAMPLE_N} symbols, 2y")
    ap.add_argument("--refresh-fundamentals", action="store_true")
    args = ap.parse_args()

    version = args.version + ("-sample" if args.sample else "")
    outdir = args.outdir or os.path.join("dist", version)
    os.makedirs(outdir, exist_ok=True)

    symbols = load_symbols()
    if args.sample:
        eq = [s for s in symbols if not s.startswith("^")][:SAMPLE_N]
        symbols = eq + [BENCH]
        args.period = "2y"   # yfinance path drops series with <300 bars
    print(f"=== Building release {version} ({args.period}, "
          f"{len(symbols)} symbols) -> {outdir}")

    panel, bench = build_panel(symbols, args.period)
    fund = fundamentals_panel([s for s in symbols if not s.startswith("^")],
                              refresh=args.refresh_fundamentals)
    factors, ranked = build_factors(panel, bench, fund)

    print("\n=== Validation ===")
    checks = [v_schema(panel, factors, fund), v_no_lookahead(fund),
              v_returns_sanity(factors), v_duplicates(panel),
              v_coverage(factors, ranked)]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not all(ok for _, ok, _ in checks):
        print("\nRelease ABORTED -- fix the failures above, log in CHANGELOG.md.")
        return 1

    # ---- write artifacts
    meta = {
        "version": version,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": args.period,
        "horizon_days": HORIZON,
        "n_symbols": int(panel["sym"].nunique()),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "rows": {"panel": len(panel), "factors": len(factors),
                 "fundamentals": len(fund)},
        "validation": {n: {"passed": bool(ok), "detail": d}
                       for n, ok, d in checks},
    }

    files = {
        "panel_nifty100.parquet": panel,
        "fundamentals_pit.parquet": fund,
        "factors.parquet": factors,
    }
    for fn, df in files.items():
        df.to_parquet(os.path.join(outdir, fn), index=False)

    pd.DataFrame({"symbol": sorted(panel["sym"].unique()),
                  "as_of": meta["date_max"]}).to_csv(
        os.path.join(outdir, "universe.csv"), index=False)

    cov = factors[ranked].notna().mean().mul(100).round(1).rename("non_null_pct")
    cov.to_csv(os.path.join(outdir, "coverage.csv"))

    write_dictionary(os.path.join(outdir, "DATA_DICTIONARY.md"), meta)

    meta["checksums"] = {fn: sha256(os.path.join(outdir, fn)) for fn in files}
    meta["size_mb"] = {
        fn: round(os.path.getsize(os.path.join(outdir, fn)) / 1e6, 2)
        for fn in files}
    with open(os.path.join(outdir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=== Release {version} OK ===")
    print(f"  {meta['n_symbols']} symbols | {meta['date_min']} -> "
          f"{meta['date_max']} | horizon {HORIZON}d")
    for fn in files:
        print(f"  {fn:28s} {meta['size_mb'][fn]:6.2f} MB  "
              f"sha {meta['checksums'][fn]}")
    print(f"  + universe.csv, coverage.csv, DATA_DICTIONARY.md, MANIFEST.json")
    print(f"\nOutput: {os.path.abspath(outdir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
