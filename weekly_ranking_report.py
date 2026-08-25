"""
Weekly NIFTY-100 Ranking Report Generator (Product Artifact v1)
Blend: 12-1 month momentum + short-term reversal, cross-sectional ranks.
Outputs: Top-20 / Bottom-20 ranking report (.md + .html) + JSON snapshot for paper trading.

Usage:
    python weekly_ranking_report.py            # generate this week's report
    python weekly_ranking_report.py --no-html  # markdown only
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from pooled_model_v1 import load_symbols, fetch_panel, build_dataset


def fetch_fii_dii_flow():
    """Best-effort FII/DII net flow from NSE public API. Returns dict or None."""
    try:
        import requests
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
             "Accept": "application/json", "Accept-Language": "en-US,en;q=0.9",
             "Referer": "https://www.nseindia.com/reports/fii-dii"}
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=h, timeout=10)
        r = s.get("https://www.nseindia.com/api/fii-dii-money-flow",
                  headers=h, timeout=10)
        data = r.json()
        latest = data["category"][-1] if isinstance(data, dict) else None
        out = {}
        for row in (data.get("category") or []):
            date = row.get("date")
            for e in row.get("value", []):
                cat = e.get("category")
                val = e.get("netCr") if e.get("netCr") not in (None, "-") else 0
                key = f"fii_net_cr" if "FII" in str(cat) or "FPI" in str(cat) else "dii_net_cr"
                out.setdefault(date, {})[key] = float(val)
        if out:
            last_date = sorted(out)[-1]
            return {"date": last_date, **out[last_date]}
    except Exception as e:
        print(f"[FII/DII] unavailable ({type(e).__name__}) - proceeding without it")
    return None


def generate(top_n=20):
    symbols = load_symbols()
    close_panel, highs, lows, vols = fetch_panel(symbols, period="3y")
    df = build_dataset(close_panel, highs, lows, vols, horizon=10)
    df = df[df["symbol"] != "^NSEI"]
    df = df.dropna(subset=["mom_12_1", "ret_20", "liq"])

    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date].copy()
    print(f"\nRanking as of: {pd.to_datetime(latest_date).date()} "
          f"({len(snap)} stocks scored)")

    g = snap
    snap["score_raw"] = (g["mom_12_1"].rank(pct=True)
                         + (1 - g["ret_20"].rank(pct=True))) / 2.0
    snap = snap.sort_values("score_raw", ascending=False).reset_index(drop=True)
    snap["rank"] = np.arange(1, len(snap) + 1)

    fii = fetch_fii_dii_flow()

    def row(r):
        sym = r["symbol"]
        c = close_panel[sym].dropna()
        mom_raw = float(c.iloc[-1] / c.iloc[-253] - 1) if len(c) > 252 else np.nan
        ret20_raw = float(c.pct_change(20).iloc[-1])
        return {
            "rank": int(r["rank"]), "symbol": sym.replace(".NS", ""),
            "close": round(float(c.iloc[-1]), 1),
            "mom_12_1_pct": round(mom_raw * 100, 1) if not np.isnan(mom_raw) else None,
            "ret20_pct": round(ret20_raw * 100, 1),
            "score": round(float(r["score_raw"]), 3),
        }

    top = [row(snap.iloc[i]) for i in range(min(top_n, len(snap)))]
    bottom = [row(snap.iloc[len(snap) - 1 - i]) for i in range(min(top_n, len(snap)))]

    gen_dt = datetime.now()
    os.makedirs("outputs/rankings", exist_ok=True)
    dstr = pd.to_datetime(latest_date).strftime("%Y-%m-%d")

    snapshot = {
        "generated_at": gen_dt.isoformat(),
        "as_of": dstr,
        "model": "blend(mom_12_1 rank + reversal rank), v1",
        "universe_size": int(len(snap)),
        "fii_dii": fii,
        "top": top,
        "bottom": bottom,
    }
    json_path = f"outputs/rankings/snapshot_{dstr}.json"
    with open(json_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    md = [f"# NIFTY-100 Weekly Ranking — {dstr}",
          "",
          f"*Model: momentum(12-1) + reversal blend · Universe: {len(snap)} stocks · "
          f"Generated {gen_dt.strftime('%Y-%m-%d %H:%M')}*",
          ""]
    if fii:
        fii_v = fii.get("fii_net_cr", "n/a")
        dii_v = fii.get("dii_net_cr", "n/a")
        fii_sign = "+" if isinstance(fii_v, (int, float)) and fii_v >= 0 else ""
        dii_sign = "+" if isinstance(dii_v, (int, float)) and dii_v >= 0 else ""
        md.append(f"**Institutional flows ({fii['date']}):** "
                  f"FII {fii_sign}{fii_v} Cr · DII {dii_sign}{dii_v} Cr")
        md.append("")
    md += ["## Top 20 (Long candidates)", "",
           "| Rank | Stock | Price | 12-1M Mom% | 20d Ret% | Score |",
           "|---|---|---|---|---|---|"]
    for r in top:
        md.append(f"| {r['rank']} | {r['symbol']} | {r['close']} | {r['mom_12_1_pct']}% "
                  f"| {r['ret20_pct']}% | {r['score']} |")
    md += ["", "## Bottom 20 (Avoid / Short candidates)", "",
           "| Rank | Stock | Price | 12-1M Mom% | 20d Ret% | Score |",
           "|---|---|---|---|---|---|"]
    for r in bottom:
        md.append(f"| {r['rank']} | {r['symbol']} | {r['close']} | {r['mom_12_1_pct']}% "
                  f"| {r['ret20_pct']}% | {r['score']} |")
    md += ["",
           "---",
           "*Research purposes only. Historical validation shows ~50-51% directional edge "
           "(AUC ~0.51). Expect roughly half of picks to underperform. Not investment advice.*"]

    md_path = f"outputs/rankings/ranking_{dstr}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    html_path = None
    if not args_html():
        html_path = write_html(snapshot, md_path)

    print(f"\nTOP 5 : " + ", ".join(r["symbol"] for r in top[:5]))
    print(f"BOTTOM 5: " + ", ".join(r["symbol"] for r in bottom[:5]))
    print(f"\nSaved: {md_path}")
    print(f"Saved: {json_path}")
    if html_path:
        print(f"Saved: {html_path}")
    return snapshot


def args_html():
    return "--no-html" in __import__("sys").argv


def write_html(snap, md_path):
    rows = lambda lst, cls: "".join(
        f'<tr class="{cls}"><td>{r["rank"]}</td><td><b>{r["symbol"]}</b></td>'
        f'<td>{r["close"]}</td><td>{r["mom_12_1_pct"]}%</td>'
        f'<td>{r["ret20_pct"]}%</td><td>{r["score"]}</td></tr>' for r in lst)
    fii_line = ""
    if snap.get("fii_dii"):
        fd = snap["fii_dii"]
        fii_line = (f"<p>Institutional flows ({fd.get('date','n/a')}): "
                    f"FII <b class=\"{'pos' if fd.get('fii_net_cr',0)>=0 else 'neg'}\">"
                    f"{fd.get('fii_net_cr','n/a')} Cr</b> &middot; "
                    f"DII <b class=\"{'pos' if fd.get('dii_net_cr',0)>=0 else 'neg'}\">"
                    f"{fd.get('dii_net_cr','n/a')} Cr</b></p>")
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>NIFTY-100 Weekly Ranking {snap['as_of']}</title>
<style>
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:36px 20px;line-height:1.5}}
.wrap{{max-width:900px;margin:0 auto}}
h1{{font-size:1.6rem}}h2{{color:#58a6ff;font-size:1.15rem;margin:28px 0 10px}}
.sub{{color:#8b949e;margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th{{background:#21262d;text-align:left;padding:9px;border:1px solid #30363d}}
td{{padding:7px 9px;border:1px solid #30363d}}
.long td:nth-child(2){{color:#3fb950}}.avoid td:nth-child(2){{color:#f85149}}
.pos{{color:#3fb950}}.neg{{color:#f85149}}
.disc{{color:#8b949e;font-size:.78rem;margin-top:26px;border-top:1px solid #30363d;padding-top:12px}}
</style></head><body><div class="wrap">
<h1>NIFTY-100 Weekly Ranking</h1>
<div class="sub">As of {snap['as_of']} &middot; momentum(12-1)+reversal blend &middot; universe {snap['universe_size']} stocks</div>
{fii_line}
<h2>Top 20 — Long Candidates</h2>
<table><tr><th>Rank</th><th>Stock</th><th>Price</th><th>12-1M Mom%</th><th>20d Ret%</th><th>Score</th></tr>
{rows(snap['top'], 'long')}</table>
<h2>Bottom 20 — Avoid / Short Candidates</h2>
<table><tr><th>Rank</th><th>Stock</th><th>Price</th><th>12-1M Mom%</th><th>20d Ret%</th><th>Score</th></tr>
{rows(snap['bottom'], 'avoid')}</table>
<p class="disc">Research purposes only. Historical validation shows ~50-51% directional edge (AUC ~0.51).
Expect roughly half of picks to underperform. Not investment advice.</p>
</div></body></html>"""
    path = md_path.replace(".md", ".html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=20)
    known, _ = ap.parse_known_args()
    generate(top_n=known.top_n)
