#!/usr/bin/env python3
"""
Daily Production Scheduler for Kronos Trading System
Runs at 3:30 PM IST (market close) to generate predictions for next trading day.

Usage:
    python daily_scheduler.py                    # Run for all top stocks
    python daily_scheduler.py --symbol RELIANCE.NS  # Run for specific stock
    python daily_scheduler.py --alert email      # Send email alerts
    python daily_scheduler.py --alert telegram   # Send Telegram alerts
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import schedule
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from daily_kronos_pipeline import DailyKronosPipeline
from macro_utils import fetch_macro_data
from alerts.email_alerts import get_alert_system
from alerts.telegram_alerts import get_telegram_system
from paper_trading import PaperTradingSimulator


# Top 10 NIFTY 50 stocks (production ready)
PRODUCTION_STOCKS = {
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


def run_daily_predictions(symbol=None, alert_type=None, paper_trade=False):
    """Run daily predictions for specified stock(s)"""
    print(f"\n{'='*70}")
    print(f"DAILY PREDICTION RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Fetch macro data once
    print("[MACRO] Fetching macro data...")
    macro = fetch_macro_data()
    print(f"  USD/INR: {macro.get('usd_inr', 'N/A')}")
    print(f"  India VIX: {macro.get('india_vix', 'N/A')}")
    print(f"  FII Flow Proxy: {macro.get('fii_flow_proxy', 'N/A')}")
    print(f"  DII Flow Proxy: {macro.get('dii_flow_proxy', 'N/A')}")
    print()
    
    # Initialize paper trading simulator if requested
    simulator = None
    if paper_trade:
        simulator = PaperTradingSimulator()
        print(f"[PAPER] Paper trading enabled. Capital: ₹{simulator.capital:,.2f}")
        # Check existing positions
        simulator.check_positions()
        print()
    
    # Determine which stocks to run
    if symbol:
        stocks = {symbol: PRODUCTION_STOCKS.get(symbol, symbol)}
    else:
        stocks = PRODUCTION_STOCKS
    
    results = []
    
    for sym, name in stocks.items():
        print(f"\n{'─'*70}")
        print(f"Running prediction for {name} ({sym})...")
        print(f"{'─'*70}")
        
        try:
            pipeline = DailyKronosPipeline(sym)
            pipeline.train()
            prediction = pipeline.predict()
            
            result = {
                "symbol": sym,
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction
            }
            results.append(result)
            
            # Print summary
            print(f"\n  Direction: {prediction['direction']}")
            print(f"  Confidence: {prediction['confidence']}%")
            print(f"  Tradeable: {prediction['tradeable']}")
            print(f"  Gate: {prediction['gate']}")
            
            # Paper trading: open position if tradeable
            if simulator and prediction['tradeable']:
                # Get current price
                import yfinance as yf
                ticker = yf.Ticker(sym)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])
                    simulator.open_position(
                        symbol=sym,
                        name=name,
                        direction=prediction['direction'],
                        confidence=prediction['confidence'],
                        current_price=current_price
                    )
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "symbol": sym,
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            })
    
    # Save results
    output_dir = Path("outputs/daily_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"daily_predictions_{date_str}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_file}")
    
    # Print paper trading report
    if simulator:
        simulator.print_report()
    
    # Send alerts if requested
    if alert_type:
        send_alerts(results, alert_type)
    
    return results


def send_alerts(results, alert_type):
    """Send alerts via email or Telegram"""
    # Filter for tradeable predictions
    tradeable = [r for r in results if r.get("prediction", {}).get("tradeable", False)]
    
    if not tradeable:
        print("\n[ALERT] No tradeable predictions today.")
        return
    
    if alert_type == "email":
        email_system = get_alert_system()
        if email_system.is_configured():
            email_system.send_daily_predictions(results)
        else:
            print("\n[EMAIL] System not configured. Set environment variables:")
            print("  KRONOS_EMAIL_ADDRESS=your.email@gmail.com")
            print("  KRONOS_EMAIL_PASSWORD=your_app_password")
            print("  KRONOS_ALERT_RECIPIENT=recipient@email.com")
            
    elif alert_type == "telegram":
        telegram_system = get_telegram_system()
        if telegram_system.is_configured():
            telegram_system.send_daily_predictions(results)
        else:
            print("\n[TELEGRAM] System not configured. Set environment variables:")
            print("  KRONOS_TELEGRAM_BOT_TOKEN=your_bot_token")
            print("  KRONOS_TELEGRAM_CHAT_ID=your_chat_id")


def schedule_daily_run(time_str="15:30"):
    """Schedule daily run at specified time (IST)"""
    print(f"Scheduling daily predictions at {time_str} IST...")
    
    schedule.every().day.at(time_str).do(run_daily_predictions)
    
    print(f"Scheduler started. Press Ctrl+C to stop.")
    print(f"Next run: {schedule.next_run()}")
    
    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Daily Kronos Scheduler")
    parser.add_argument("--symbol", help="Specific stock symbol (e.g., RELIANCE.NS)")
    parser.add_argument("--alert", choices=["email", "telegram"], help="Send alerts via email or Telegram")
    parser.add_argument("--schedule", action="store_true", help="Run as scheduled job")
    parser.add_argument("--time", default="15:30", help="Schedule time in HH:MM format (default: 15:30)")
    parser.add_argument("--paper-trade", action="store_true", help="Enable paper trading")
    args = parser.parse_args()
    
    if args.schedule:
        schedule_daily_run(args.time)
    else:
        run_daily_predictions(args.symbol, args.alert, args.paper_trade)


if __name__ == "__main__":
    main()
