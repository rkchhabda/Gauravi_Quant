"""Generate the data artifact behind model_test.html.

Runs the REAL shipped factor models over the NIFTY-100 universe and computes,
per stock, an honest historical directional hit-rate from actual price data --
no fabricated accuracy. For each (model, horizon):

  signal      per-date cross-sectional percentile of the factor
  direction   LONG if signal >= 0.5 (median), else SHORT
  acc_excess  P(direction matches sign of the H-day return in EXCESS of NIFTY)
              -- this is the honest edge; its baseline is ~50%.
  acc_abs     P(direction matches plain up/down). Shown only alongside its
              up_rate baseline, because in a rising sample plain up/down is
              dominated by market drift and overstates skill (the base-rate
              illusion documented in accuracy_log.md).

Models: composite (shipped: mom_12_1 rank + reversal rank), momentum (12-1),
reversal (-ret_20), liquidity (20d dollar volume).

Output: outputs/model_test/model_test_data.js  (window.MODEL_TEST_DATA = {...})
        outputs/model_test/model_test_data.json (same payload)
        outputs/model_test/model_test.html      (copy of the template)

Usage:
    python build_model_test.py                 # horizons 5,10,20 ; period 3y
    python build_model_test.py --period 5y
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from pooled_model_v1 import load_symbols, fetch_panel, build_dataset

MODELS = ["composite", "momentum", "reversal", "liquidity"]
HORIZONS = [5, 10, 20]


def signal_pctile(df, model):
    """Per-date cross-sectional percentile (0..1) of the model's raw signal.

    build_dataset already z-scores mom_12_1 / ret_20 / liq per date, so ranking
    on them is monotone-equivalent to ranking on the raw factor.
    """
    g = df.groupby("date")
    if model == "momentum":
        return g["mom_12_1"].rank(pct=True)
    if model == "reversal":
        return (1 - g["ret_20"].rank(pct=True))
    if model == "liquidity":
        return g["liq"].rank(pct=True)
    if model == "composite":
        comp = g["mom_12_1"].rank(pct=True) + (1 - g["ret_20"].rank(pct=True))
        df = df.assign(_comp=comp)
        return df.groupby("date")["_comp"].rank(pct=True)
    raise ValueError(model)


def hit_columns(df):
    """LONG/SHORT correctness against excess and absolute forward return."""
    long = df["dir_long"]
    exc_up = df["fwd_excess"] > 0
    abs_up = df["fwd_ret"] > 0
    hit_exc = np.where(long, exc_up, ~exc_up)
    hit_abs = np.where(long, abs_up, ~abs_up)
    return pd.Series(hit_exc, index=df.index), pd.Series(hit_abs, index=df.index)


def build_payload(period="3y", horizons=HORIZONS):
    symbols = load_symbols()
    close_panel, highs, lows, vols = fetch_panel(symbols, period=period)

    names = {}
    with open("nifty100_stocks.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            sym = parts[0].strip()
            names[sym.replace(".NS", "")] = (parts[1].strip() if len(parts) > 1
                                             else sym.replace(".NS", ""))

    stocks = {}
    universe_acc = {m: {} for m in MODELS}
    as_of = None

    for h in horizons:
        df = build_dataset(close_panel, highs, lows, vols, horizon=h)
        df = df[df["symbol"] != "^NSEI"].copy()
        latest_date = df["date"].max()
        as_of = pd.to_datetime(latest_date).strftime("%Y-%m-%d")

        for model in MODELS:
            df["sig"] = signal_pctile(df, model)
            df["dir_long"] = df["sig"] >= 0.5
            scored = df.dropna(subset=["fwd_excess", "fwd_ret"]).copy()
            hit_exc, hit_abs = hit_columns(scored)
            scored["hit_exc"] = hit_exc
            scored["hit_abs"] = hit_abs

            # universe-level honest accuracy
            universe_acc[model][str(h)] = {
                "acc_excess_pct": round(float(scored["hit_exc"].mean()) * 100, 1),
                "acc_abs_pct": round(float(scored["hit_abs"].mean()) * 100, 1),
                "up_rate_pct": round(float((scored["fwd_ret"] > 0).mean()) * 100, 1),
                "n": int(len(scored)),
            }

            # per-stock history
            per = scored.groupby("symbol").agg(
                acc_excess=("hit_exc", "mean"),
                acc_abs=("hit_abs", "mean"),
                up_rate=("fwd_ret", lambda s: (s > 0).mean()),
                n=("hit_exc", "size"),
            )

            # latest-day prediction, and rank within universe on that day
            snap = df[df["date"] == latest_date].copy()
            snap = snap.sort_values("sig", ascending=False).reset_index(drop=True)
            snap["rank"] = np.arange(1, len(snap) + 1)
            uni_n = len(snap)

            for _, r in snap.iterrows():
                sym = r["symbol"].replace(".NS", "")
                st = stocks.setdefault(sym, {"name": names.get(sym, sym), "by": {}})
                bucket = st["by"].setdefault(model, {})
                pr = per.loc[r["symbol"]] if r["symbol"] in per.index else None
                bucket[str(h)] = {
                    "direction": "LONG" if bool(r["sig"] >= 0.5) else "SHORT",
                    "score_pctile": round(float(r["sig"]), 3),
                    "rank": int(r["rank"]),
                    "universe": uni_n,
                    "acc_excess_pct": round(float(pr["acc_excess"]) * 100, 1) if pr is not None else None,
                    "acc_abs_pct": round(float(pr["acc_abs"]) * 100, 1) if pr is not None else None,
                    "up_rate_pct": round(float(pr["up_rate"]) * 100, 1) if pr is not None else None,
                    "n": int(pr["n"]) if pr is not None else 0,
                }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": as_of,
        "period": period,
        "universe_size": len(stocks),
        "models": MODELS,
        "horizons": horizons,
        "universe_accuracy": universe_acc,
        "stocks": stocks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="3y")
    ap.add_argument("--horizons", default="5,10,20")
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    payload = build_payload(period=args.period, horizons=horizons)

    out_dir = os.path.join("outputs", "model_test")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model_test_data.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, "model_test_data.js"), "w", encoding="utf-8") as f:
        f.write("window.MODEL_TEST_DATA = ")
        json.dump(payload, f)
        f.write(";\n")

    tpl = "model_test_template.html"
    if os.path.exists(tpl):
        shutil.copy2(tpl, os.path.join(out_dir, "model_test.html"))

    u = payload["universe_accuracy"]["composite"].get("10", {})
    print(f"as_of={payload['as_of']}  stocks={payload['universe_size']}")
    print(f"composite/10d universe: excess {u.get('acc_excess_pct')}%  "
          f"abs {u.get('acc_abs_pct')}% (up-rate {u.get('up_rate_pct')}%)  n={u.get('n')}")
    print(f"Wrote {out_dir}/model_test_data.js + .json + model_test.html")


if __name__ == "__main__":
    main()
