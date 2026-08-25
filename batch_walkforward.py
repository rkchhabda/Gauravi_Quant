"""
Batch runner: Production walk-forward across all top 10 NIFTY 50 stocks.
Compares Tier 1+2 (calibrated) vs Tier 3 (meta-labeling + triple-barrier).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from production_walkforward import ProductionWalkForward

SYMBOLS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
}

# Tier config
TIER_CONFIGS = {
    "tier12_standard": {
        "use_meta_labeling": False,
        "use_triple_barrier": False,
        "desc": "Tier 1+2: Standard (confidence buckets + improved mood)"
    },
    "tier3_meta_label": {
        "use_meta_labeling": True,
        "use_triple_barrier": False,
        "desc": "Tier 3: Meta-labeling (Kronos direction + ML confidence)"
    },
    "tier3_triple_barrier": {
        "use_meta_labeling": False,
        "use_triple_barrier": True,
        "desc": "Tier 3: Triple-barrier (TP/SL/Timeout)"
    },
}


def run_batch(tier_key="tier12_standard", period="1y"):
    config = TIER_CONFIGS[tier_key]
    print(f"\n{'#'*70}")
    print(f"BATCH WALK-FORWARD: {config['desc']}")
    print(f"Period: {period} | Symbols: {len(SYMBOLS)}")
    print(f"{'#'*70}")

    all_results = []
    for symbol, name in SYMBOLS.items():
        try:
            pwf = ProductionWalkForward(
                symbol=symbol,
                period=period,
                train_window=120,
                test_window=5,
                horizon=10,
                use_meta_labeling=config["use_meta_labeling"],
                use_triple_barrier=config["use_triple_barrier"],
            )
            result = pwf.run()
            if result:
                result["name"] = name
                all_results.append(result)
        except Exception as e:
            print(f"  ERROR on {symbol}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No results!")
        return

    all_results.sort(key=lambda x: x["overall_accuracy"], reverse=True)

    print(f"\n{'#'*70}")
    print(f"SUMMARY: {config['desc']}")
    print(f"{'#'*70}")
    print(f"\n{'Rank':<5} {'Symbol':<15} {'Name':<25} {'Preds':>6} {'Acc%':>7} {'Gate+':>7}")
    print("-" * 70)
    for i, r in enumerate(all_results, 1):
        print(f"{i:<5} {r['symbol']:<15} {r['name']:<25} {r['total_predictions']:>6} {r['overall_accuracy']:>6.1f}% {r['gate_added_value']:>+6.1f}%")

    accs = [r["overall_accuracy"] for r in all_results]
    print("-" * 70)
    print(f"{'':5} {'AVERAGE':<15} {'':25} {'':>6} {np.mean(accs):>6.1f}%")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"batch_{tier_key}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {path}")

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch Production Walk-Forward")
    parser.add_argument("--tier", default="tier12_standard",
                        choices=["tier12_standard", "tier3_meta_label", "tier3_triple_barrier"],
                        help="Which tier to run")
    parser.add_argument("--period", default="1y", help="Data period")
    parser.add_argument("--all-tiers", action="store_true", help="Run all three tiers sequentially")
    args = parser.parse_args()

    if args.all_tiers:
        for tier in TIER_CONFIGS:
            run_batch(tier_key=tier, period=args.period)
    else:
        run_batch(tier_key=args.tier, period=args.period)


if __name__ == "__main__":
    main()
