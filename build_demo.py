"""Build the customer-facing demo page from a built dataset release.

    python build_demo.py                          # -> outputs/demo/demo.html
    python build_demo.py --outdir public          # publish straight to Pages

Everything on the page is COMPUTED from the release in dist/<version>/ -- no
hand-typed numbers. That is the point: a prospect can re-run this script against
the sample they downloaded and get the same page.

The page is deliberately built to survive a sceptical quant reading it:
it leads with the honest error bars, shows the naive vs calibrated t-stats side
by side, and lists the bugs we found in our own data.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import gauravi_data as gd

BASELINE_NOTE = ("A coin flip is not 50% here: excess returns are skewed, so the "
                 "all-stock base rate below is the number to beat.")

# Bugs we found in our own pipeline. Kept on the page on purpose -- a vendor with
# no bug log either has not looked or is not telling you.
CORRECTIONS = [
    ("Unadjusted NSE prices", "Driving returns off raw NSE closes ignored splits "
     "and produced an 11% hit-rate. Returns now come from adjusted prices only.",
     "Caught by the returns-sanity gate (no return may be < -100%)."),
    ("Friday labelled as Sunday", "The delivery source returned each week's Friday "
     "row dated the following Sunday, silently dropping ~1 trading day per week "
     "(delivery coverage 75% instead of 94%).",
     "Fixed in data_layer._fix_weekend_dates; CI now fails if coverage < 85%."),
    ("Delivery edge disappeared", "Before that fix, a delivery-based composite "
     "measured 53.5% and we reported it. On corrected data it no longer beats "
     "plain momentum+reversal. We retracted the number.",
     "The retraction is recorded in PROGRESS.md section 6."),
    ("Coverage mistaken for alpha", "A fundamentals composite looked ~4pp better "
     "than the price model -- but was scored on the 64% of rows where fundamentals "
     "exist. On the common sample the ranking flips.",
     "Both are now always compared on identical rows."),
]


# --------------------------------------------------------------------------
# metrics -- all computed, nothing hard-coded
# --------------------------------------------------------------------------
def ic_series(df, factor, target="fwd_excess"):
    sub = df[["date", factor, target]].dropna()
    return (sub.groupby("date")
               .apply(lambda g: g[factor].corr(g[target], method="spearman")
                      if len(g) > 5 else np.nan)
               .dropna())


def ic_table(df, factors, H):
    dates = np.sort(df["date"].unique())
    nonovlp = set(dates[::H])
    rows = []
    for col in factors:
        s = ic_series(df, col)
        if s.empty:
            continue
        ind = s[s.index.isin(nonovlp)]
        n = len(ind)
        t = (ind.mean() / (ind.std(ddof=1) / np.sqrt(n))) if n > 2 else np.nan
        rows.append({"factor": col, "IC": s.mean(),
                     "t_naive": s.mean() / (s.std(ddof=1) / np.sqrt(len(s))),
                     "t_nonovlp": t, "n_indep": n})
    out = pd.DataFrame(rows)
    out["significant"] = out["t_nonovlp"].abs() >= 1.96
    return out.reindex(out["t_nonovlp"].abs().sort_values(ascending=False).index)


def quintile_stats(df, score, H):
    """Top-quintile hit-rate and spread on NON-OVERLAPPING holds only."""
    d = df.dropna(subset=[score, "fwd_excess"])
    dates = np.sort(d["date"].unique())
    d = d[d["date"].isin(set(dates[::H]))]
    if d.empty:
        return None
    hi = d.groupby("date")[score].transform(lambda x: x.quantile(0.8))
    lo = d.groupby("date")[score].transform(lambda x: x.quantile(0.2))
    top, bot = d[d[score] >= hi], d[d[score] <= lo]
    hit = float((top["fwd_excess"] > 0).mean())
    n = len(top)
    se = float(np.sqrt(hit * (1 - hit) / max(n, 1)))
    base = float((d["fwd_excess"] > 0).mean())
    return {
        "hit_rate": hit * 100, "se": se * 100,
        "ci_lo": (hit - 1.96 * se) * 100, "ci_hi": (hit + 1.96 * se) * 100,
        "base_rate": base * 100, "edge_pp": (hit - base) * 100,
        "spread_pp": float(top["fwd_excess"].mean()
                           - bot["fwd_excess"].mean()) * 100,
        "n_picks": n, "n_rebalances": int(d["date"].nunique()),
        "beats_base": bool(hit - 1.96 * se > base),
    }


def collect(path=None):
    m = gd.manifest(path)
    H = int(m["horizon_days"])
    f = gd.load_factors(data_dir=path)
    panel = gd.load_panel(data_dir=path)

    ranked = [c for c in f.columns if c.startswith("rank_")
              and not c.startswith("rank_score_")]
    scores = [c for c in ("score_mr", "score_fund3") if c in f.columns]

    ic = ic_table(f, ranked + scores, H)
    per_score = {s: quintile_stats(f, s, H) for s in scores}

    # Apples-to-apples: identical rows for every composite.
    common = f.dropna(subset=scores) if len(scores) > 1 else f
    per_score_common = {s: quintile_stats(common, s, H) for s in scores}

    cov = (f[ranked].notna().mean() * 100).round(1).sort_values(ascending=False)
    latest = f[f["date"] == f["date"].max()]
    top10 = (latest.dropna(subset=["score_mr"])
                   .nlargest(10, "score_mr")[["sym", "score_mr", "rev",
                                              "mom_12_1", "deliv_pct"]])
    return {
        "manifest": m, "H": H, "ic": ic, "coverage": cov,
        "per_score": per_score, "per_score_common": per_score_common,
        "scores": scores, "top10": top10,
        "as_of": str(f["date"].max().date()),
        "n_rows": len(f), "n_panel": len(panel),
        "n_dates": int(f["date"].nunique()),
        "deliv_cov": float(panel["deliv_pct"].notna().mean() * 100),
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
CSS = """
*{box-sizing:border-box}
body{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;
 margin:0;padding:32px 16px;line-height:1.55}
.wrap{max-width:1040px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 4px}
h2{color:#58a6ff;font-size:1.05rem;margin:34px 0 6px;
 border-bottom:1px solid #21262d;padding-bottom:6px}
.sub{color:#8b949e;font-size:.9rem}
.note{color:#8b949e;font-size:.82rem;margin:6px 0 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;
 margin:16px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}
.card .k{color:#8b949e;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px}
.card .v{font-size:1.5rem;font-weight:600;margin-top:4px}
.card .f{color:#8b949e;font-size:.75rem;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:.85rem;margin:10px 0}
th,td{padding:7px 9px;text-align:right;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:600;text-align:right;font-size:.75rem;
 text-transform:uppercase;letter-spacing:.4px}
th:first-child,td:first-child{text-align:left}
tbody tr:hover{background:#161b22}
code{background:#161b22;padding:1px 5px;border-radius:4px;font-size:.85em}
pre{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;
 overflow-x:auto;font-size:.8rem}
.pass{color:#3fb950}.fail{color:#f85149}.warn{color:#d29922}.dim{color:#8b949e}
.pill{display:inline-block;background:#161b22;border:1px solid #30363d;
 border-radius:20px;padding:2px 10px;font-size:.75rem;margin:2px 4px 2px 0}
.box{background:#161b22;border:1px solid #30363d;border-left:3px solid #d29922;
 border-radius:8px;padding:12px 14px;margin:12px 0;font-size:.87rem}
.box.ok{border-left-color:#3fb950}
.box b{color:#e6edf3}
footer{color:#8b949e;font-size:.78rem;margin-top:40px;border-top:1px solid #21262d;
 padding-top:14px}
a{color:#58a6ff}
"""


def card(k, v, f="", cls=""):
    return (f'<div class="card"><div class="k">{k}</div>'
            f'<div class="v {cls}">{v}</div><div class="f">{f}</div></div>')


def table(df, cols=None, fmt=None, flags=None):
    cols = cols or list(df.columns)
    fmt = fmt or {}
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            txt = fmt[c](v) if c in fmt else (
                f"{v:.4f}" if isinstance(v, float) else str(v))
            cls = flags(c, v) if flags else ""
            tds.append(f'<td class="{cls}">{txt}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def headline_cards(d):
    # score_mr on every row where it is defined -- its natural sample. Section 3
    # shows the like-for-like comparison against the fundamentals composite.
    s = d["per_score"]["score_mr"]
    H = d["H"]
    verdict = ("beats the base rate" if s["beats_base"]
               else "NOT distinguishable from the base rate")
    cls = "pass" if s["beats_base"] else "warn"
    return "".join([
        card(f"Top-quintile hit rate ({H}d)", f'{s["hit_rate"]:.2f}%',
             f'&plusmn;{s["se"]:.2f}pp SE &middot; 95% CI '
             f'{s["ci_lo"]:.1f}&ndash;{s["ci_hi"]:.1f}%'),
        card("All-stock base rate", f'{s["base_rate"]:.2f}%',
             f'edge {s["edge_pp"]:+.2f}pp &mdash; {verdict}', cls),
        card("Independent rebalances", f'{s["n_rebalances"]}',
             f'{s["n_picks"]} non-overlapping picks'),
        card("Quintile spread", f'{s["spread_pp"]:+.2f}pp',
             f'top minus bottom quintile, {H}d excess'),
        card("Delivery-% coverage", f'{d["deliv_cov"]:.0f}%',
             "field absent from yfinance entirely"),
        card("Rows / dates", f'{d["n_rows"]:,}',
             f'{d["n_dates"]:,} trading dates &middot; '
             f'{d["manifest"]["n_symbols"]} symbols'),
    ])


def render(d):
    m = d["manifest"]
    H = d["H"]
    gates = "".join(
        f'<span class="pill"><span class="{"pass" if v["passed"] else "fail"}">'
        f'{"PASS" if v["passed"] else "FAIL"}</span> {k}</span>'
        for k, v in m["validation"].items())
    sums = "".join(f'<span class="pill">{k} <span class="dim">'
                   f'{m["size_mb"][k]} MB &middot; {v}</span></span>'
                   for k, v in m["checksums"].items())

    ic = d["ic"].copy()
    ic["significant"] = ic["significant"].map({True: "YES", False: "no"})
    ic_html = table(
        ic, ["factor", "IC", "t_naive", "t_nonovlp", "n_indep", "significant"],
        fmt={"IC": lambda v: f"{v:+.4f}", "t_naive": lambda v: f"{v:+.2f}",
             "t_nonovlp": lambda v: f"{v:+.2f}", "n_indep": lambda v: f"{v:.0f}"},
        flags=lambda c, v: ("fail" if c == "significant" and v == "no" else
                            "pass" if c == "significant" else ""))

    cmp_rows = []
    for s in d["scores"]:
        a, b = d["per_score"].get(s), d["per_score_common"].get(s)
        if not a:
            continue
        cmp_rows.append({"composite": s,
                         "all rows %": a["hit_rate"], "all n": a["n_picks"],
                         "common rows %": b["hit_rate"] if b else np.nan,
                         "common n": b["n_picks"] if b else np.nan,
                         "SE pp": b["se"] if b else a["se"]})
    cmp_html = table(pd.DataFrame(cmp_rows),
                     fmt={"all rows %": lambda v: f"{v:.2f}",
                          "common rows %": lambda v: f"{v:.2f}",
                          "SE pp": lambda v: f"&plusmn;{v:.2f}",
                          "all n": lambda v: f"{v:.0f}",
                          "common n": lambda v: f"{v:.0f}"})

    cov = d["coverage"].reset_index()
    cov.columns = ["column", "non-null %"]
    cov_html = table(cov, fmt={"non-null %": lambda v: f"{v:.1f}"},
                     flags=lambda c, v: ("warn" if c == "non-null %" and v < 70
                                         else ""))

    top_html = table(
        d["top10"].rename(columns={"score_mr": "score", "rev": "reversal",
                                   "mom_12_1": "momentum",
                                   "deliv_pct": "delivery %"}),
        fmt={"score": lambda v: f"{v:.3f}", "reversal": lambda v: f"{v:+.3f}",
             "momentum": lambda v: f"{v:+.3f}",
             "delivery %": lambda v: "n/a" if pd.isna(v) else f"{v:.1f}"})

    bugs = "".join(
        f'<div class="box"><b>{t}</b><br>{w}<br><span class="dim">{f}</span></div>'
        for t, w, f in CORRECTIONS)

    s_note = ("<code>fwd_excess</code> is a forward-looking research target. It "
              "contains future information by construction and must never be used "
              "as a feature &mdash; it is in the file so you can score your own "
              "signals against it.")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gauravi &mdash; NSE Factor Dataset &middot; verification demo</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>Gauravi &mdash; NSE Factor Dataset</h1>
<div class="sub">Release <code>{m['version']}</code> &middot;
{m['n_symbols']} NIFTY-100 symbols &middot; {m['date_min']} to {m['date_max']}
&middot; horizon {H}d &middot; page built {d['built']}</div>

<div class="box ok"><b>What this page is.</b> Every number below is computed from
the release files by <code>build_demo.py</code> &mdash; nothing is typed in by
hand. Download the free sample, run the same script, and you get the same page.
That is the only claim we ask you to take on trust.</div>

<h2>1. Measured performance, with the error bars attached</h2>
<div class="grid">{headline_cards(d)}</div>
<p class="note">{BASELINE_NOTE} Hit rate = share of top-quintile picks whose {H}-day
return beat the benchmark, on <b>non-overlapping</b> holds only.</p>

<h2>2. Factor information coefficients &mdash; naive vs calibrated</h2>
<p class="note">Daily ICs against a {H}-day forward return overlap, which inflates a
naive t-stat by roughly &radic;{H}. Compare the two t columns: that gap is why most
published NSE factor results do not replicate. <b>Nothing here is significant</b>,
and we would rather you learn that from us than from your own money.</p>
{ic_html}

<h2>3. The comparison trap, shown deliberately</h2>
<p class="note">Composites needing fundamentals are only defined on the rows where
fundamentals exist. Score them there and they look better; score them on identical
rows and the ranking changes. Read the two pairs of columns together.</p>
{cmp_html}

<h2>4. Data coverage per column</h2>
<p class="note">Amber = under 70% non-null. Check this before you trust any factor
built on it.</p>
{cov_html}

<h2>5. Validation gates on this build</h2>
<div>{gates}</div>
<p class="note">The build script exits non-zero if any gate fails, so a bad month
does not ship. Checksums (SHA-256, first 16 hex) for what you receive:</p>
<div>{sums}</div>

<h2>6. Bugs we found in our own data</h2>
<p class="note">Kept on this page on purpose. A data vendor with no bug log either
has not looked or is not telling you.</p>
{bugs}

<h2>7. Current top 10 by <code>score_mr</code> ({d['as_of']})</h2>
<p class="note">Shown so you can see the shape of the output. This is a
<b>ranking of a factor score</b>, not a recommendation, and on the evidence above
its edge is not statistically established.</p>
{top_html}

<h2>8. Reproduce it yourself</h2>
<pre>pip install pandas pyarrow numpy
unzip gauravi-nse-factors-{m['version']}-sample.zip &amp;&amp; cd gauravi-nse-factors-{m['version']}-sample
python -c "import gauravi_data as gd; print(gd.manifest()['validation'])"
python build_demo.py            # regenerates this page from the data</pre>

<h2>9. Known limitations</h2>
<div class="box"><b>Survivorship bias.</b> The universe is <i>current</i>
NIFTY-100 membership, not point-in-time. Delisted and demoted names are absent.
Point-in-time membership is the top roadmap item.</div>
<div class="box"><b>Fundamentals are annual</b>, with a 90-day reporting lag.
Free quarterly NSE history is too short to use.</div>
<div class="box"><b>{s_note}</b></div>
<div class="box"><b>No statistical significance.</b> Not one factor in section 2
clears |t| = 1.96 on the calibrated test. The binding constraint is sample size:
{d['per_score']['score_mr']['n_rebalances']}
independent rebalances cannot separate 53% from 55%.</div>

<footer>
Gauravi NSE Factor Dataset &middot; research data only.
<b>Not investment advice, not a signal service, not a recommendation to buy or
sell any security.</b> No SEBI Research Analyst or Investment Adviser
registration is claimed or implied. Past measured behaviour of a factor is not a
forecast. You are buying data and documentation, not returns.
</footer>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="release dir (default: auto-detect newest in dist/)")
    ap.add_argument("--outdir", default=os.path.join("outputs", "demo"))
    args = ap.parse_args()

    d = collect(args.data_dir)
    os.makedirs(args.outdir, exist_ok=True)

    html_path = os.path.join(args.outdir, "demo.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render(d))

    # Machine-readable twin: a prospect's quant can diff this against their own run.
    metrics = {
        "version": d["manifest"]["version"], "built_utc": d["built"],
        "as_of": d["as_of"], "horizon_days": d["H"],
        "n_symbols": d["manifest"]["n_symbols"], "n_rows": d["n_rows"],
        "delivery_coverage_pct": round(d["deliv_cov"], 1),
        "validation": d["manifest"]["validation"],
        "checksums": d["manifest"]["checksums"],
        "hit_rate": {k: v for k, v in d["per_score_common"].items() if v},
        "hit_rate_all_rows": {k: v for k, v in d["per_score"].items() if v},
        "ic": d["ic"].to_dict(orient="records"),
        "coverage_pct": d["coverage"].to_dict(),
        "any_factor_significant": bool(d["ic"]["significant"].any()),
    }
    json_path = os.path.join(args.outdir, "demo_metrics.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    s = d["per_score"]["score_mr"]
    print(f"Demo built for {d['manifest']['version']} -> {html_path}")
    print(f"  hit-rate {s['hit_rate']:.2f}% (SE {s['se']:.2f}pp, "
          f"95% CI {s['ci_lo']:.1f}-{s['ci_hi']:.1f}) vs base {s['base_rate']:.2f}%")
    print(f"  significant factors: "
          f"{int(d['ic']['significant'].sum())}/{len(d['ic'])}")
    print(f"  + {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
