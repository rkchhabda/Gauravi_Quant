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
    """FII/DII net flow. Sources: NSE API -> moneycontrol -> local CSV (data/fii_dii.csv)."""
    out = _fii_dii_nse()
    if out:
        return out
    out = _fii_dii_moneycontrol()
    if out:
        return out
    return _fii_dii_csv()


def _fii_dii_nse():
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
        out = {}
        for row in (data.get("category") or []):
            date = row.get("date")
            for e in row.get("value", []):
                cat = e.get("category")
                val = e.get("netCr") if e.get("netCr") not in (None, "-") else 0
                key = "fii_net_cr" if "FII" in str(cat) or "FPI" in str(cat) else "dii_net_cr"
                out.setdefault(date, {})[key] = float(val)
        if out:
            last_date = sorted(out)[-1]
            return {"source": "nse", "date": last_date, **out[last_date]}
    except Exception:
        pass
    return None


def _fii_dii_moneycontrol():
    try:
        import requests
        h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        tables = pd.read_html("https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.html",
                              storage_options=h) if hasattr(pd, "read_html") else []
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            flat = " ".join(cols)
            if "fii" in flat and ("net" in flat or "buy" in flat):
                date_col = t.columns[0]
                last = t.iloc[0]
                date = str(last[date_col])
                fii_col = next(c for c in t.columns if "fii" in str(c).lower() and "net" in str(c).lower())
                dii_col = next((c for c in t.columns if "dii" in str(c).lower() and "net" in str(c).lower()), None)

                def num(v):
                    s = str(v).replace(",", "").replace("₹", "").strip()
                    neg = s.startswith("(") and s.endswith(")")
                    s = s.strip("()")
                    try:
                        f = float(s)
                    except ValueError:
                        return 0.0
                    return -f if neg else f
                return {"source": "moneycontrol", "date": date,
                        "fii_net_cr": num(last[fii_col]),
                        "dii_net_cr": num(last[dii_col]) if dii_col else 0.0}
    except Exception:
        pass
    return None


def _fii_dii_csv():
    """Read manual weekly entry: data/fii_dii.csv with columns date,fii_net_cr,dii_net_cr."""
    try:
        path = os.path.join("data", "fii_dii.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        need = {"date", "fii_net_cr", "dii_net_cr"}
        if not need.issubset(df.columns):
            print(f"[FII/DII] {path} must have columns: date,fii_net_cr,dii_net_cr")
            return None
        df = df.dropna(subset=["fii_net_cr", "dii_net_cr"])
        if df.empty:
            return None
        last = df.sort_values("date").iloc[-1]
        return {"source": "csv", "date": str(last["date"]),
                "fii_net_cr": float(last["fii_net_cr"]),
                "dii_net_cr": float(last["dii_net_cr"])}
    except Exception as e:
        print(f"[FII/DII] CSV read failed: {e}")
        return None


def fetch_earnings_within(symbols, days=12):
    """Return {symbol: True} if earnings report within next `days` days. Best-effort."""
    import concurrent.futures as cf
    import yfinance as yf
    from datetime import timedelta
    result = {}

    def check(sym):
        try:
            cal = yf.Ticker(sym).calendar
            dates = None
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date")
            elif cal is not None and hasattr(cal, "loc"):
                idx = [str(x) for x in cal.index]
                if "Earnings Date" in idx:
                    dates = cal.loc["Earnings Date"].iloc[0]
            if dates is None:
                return sym, False
            if not isinstance(dates, (list, tuple, pd.DatetimeIndex)):
                dates = [dates]
            cutoff = pd.Timestamp.now().normalize() + pd.Timedelta(days=days)
            for d in dates:
                if pd.to_datetime(d) <= cutoff:
                    return sym, True
            return sym, False
        except Exception:
            return sym, False

    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        for sym, risky in ex.map(check, symbols):
            result[sym] = risky
    return result


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

    print("Checking upcoming earnings dates (best-effort)...")
    earnings_map = fetch_earnings_within(snap["symbol"].tolist())
    snap["earnings_soon"] = snap["symbol"].map(earnings_map).fillna(False)
    n_risky = int(snap["earnings_soon"].sum())
    print(f"  {n_risky} stock(s) report earnings within 12 days")

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
            "earnings_soon": bool(r["earnings_soon"]),
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
           "| Rank | Stock | Price | 12-1M Mom% | 20d Ret% | Score | Earnings |",
           "|---|---|---|---|---|---|---|"]
    for r in top:
        flag = "⚠️ this week" if r["earnings_soon"] else "clear"
        md.append(f"| {r['rank']} | {r['symbol']} | {r['close']} | {r['mom_12_1_pct']}% "
                  f"| {r['ret20_pct']}% | {r['score']} | {flag} |")
    md += ["", "## Bottom 20 (Avoid / Short candidates)", "",
           "| Rank | Stock | Price | 12-1M Mom% | 20d Ret% | Score | Earnings |",
           "|---|---|---|---|---|---|---|"]
    for r in bottom:
        flag = "⚠️ this week" if r["earnings_soon"] else "clear"
        md.append(f"| {r['rank']} | {r['symbol']} | {r['close']} | {r['mom_12_1_pct']}% "
                  f"| {r['ret20_pct']}% | {r['score']} | {flag} |")
    md += ["",
           "⚠️ *Earnings flag = company reports results within ~12 days; expect high volatility "
           "around results regardless of ranking.*",
           "",
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
        f'<td>{r["ret20_pct"]}%</td><td>{r["score"]}</td>'
        f'<td>{"⚠️ this week" if r["earnings_soon"] else "clear"}</td></tr>' for r in lst)
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
<table><tr><th>Rank</th><th>Stock</th><th>Price</th><th>12-1M Mom%</th><th>20d Ret%</th><th>Score</th><th>Earnings</th></tr>
{rows(snap['top'], 'long')}</table>
<h2>Bottom 20 — Avoid / Short Candidates</h2>
<table><tr><th>Rank</th><th>Stock</th><th>Price</th><th>12-1M Mom%</th><th>20d Ret%</th><th>Score</th><th>Earnings</th></tr>
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
