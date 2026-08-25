import sys
import os
import json
import requests
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from jinja2 import Environment, FileSystemLoader

# Configure terminal stdout for safe execution on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Fixed models as mandated by specification
OLLAMA_MODEL = "llama3.2:3b"
KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
OLLAMA_API = "http://localhost:11434/api/generate"

def get_hf_token():
    """Get Hugging Face token from environment at runtime."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Add Kronos library to system path
sys.path.append("./kronos_lib")
# pyrefly: ignore [missing-import]
from model import Kronos, KronosTokenizer, KronosPredictor

# --------------------------------------------------------------
# NEW: Load stocks from a simple text file called "stocks.txt"
# If the file does not exist, it falls back to default stocks.
# --------------------------------------------------------------
INDIAN_ASSETS = []
try:
    with open("stocks.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and lines starting with # (comments)
            if line and not line.startswith("#"):
                parts = line.split(",")
                if len(parts) == 2:
                    symbol = parts[0].strip()
                    name = parts[1].strip()
                    INDIAN_ASSETS.append({"symbol": symbol, "name": name})
                else:
                    # If no name provided, use the symbol as name
                    INDIAN_ASSETS.append({"symbol": line, "name": line})
    print(f"[INFO] Loaded {len(INDIAN_ASSETS)} stocks from stocks.txt")
except FileNotFoundError:
    # Fallback default stocks if file doesn't exist
    print("[INFO] stocks.txt not found. Using default fallback stocks.")
    print("[INFO] To customize, create a stocks.txt file with one stock per line (symbol, Name).")
    INDIAN_ASSETS = [
        {"symbol": "^NSEI", "name": "NIFTY 50 Index"},
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd"},
        {"symbol": "GOLDIAM.NS", "name": "Goldiam International Ltd"},
        {"symbol": "GOLDBEES.NS", "name": "Nippon India Gold ETF"},
        {"symbol": "INR=X", "name": "USD / INR Exchange Rate"}
    ]


def generate_local_explanation(symbol_name, current, support, resistance, ret_pct, trend_text):
    prompt = (
        f"You are a friendly Indian financial advisor explaining stock trends to a beginner from India in simple English with relatable everyday analogies (like shopping at bazaar, gold rates, or bank savings). "
        f"No technical jargon! Explain this 14-day AI quant forecast for {symbol_name}: Current price is Rs. {current}, projected resistance is Rs. {resistance}, support is Rs. {support}, expected return is {ret_pct}% ({trend_text}). "
        f"Keep your explanation clear, practical, motivating, and strictly 2 to 3 concise sentences."
    )
    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"think": False, "temperature": 0.3}
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        pass
    
    # Clean beginner fallback explanation if Ollama service is unavailable
    if trend_text == "Neutral":
        return (f"{symbol_name} is currently in a steady consolidation phase (projected {ret_pct:+.2f}%). "
                f"It is prudent for beginners to wait for a breakout above Rs. {resistance} or accumulate near Rs. {support}.")
    elif ret_pct >= 0:
        return (f"Think of {symbol_name} like buying high-demand festival gold—our Kronos AI expects positive momentum of +{ret_pct}%. "
                f"Beginners can watch Rs. {support} as a safe buying bottom and Rs. {resistance} as a target for booking profit.")
    else:
        return (f"Right now, {symbol_name} is showing cooling momentum ({ret_pct:.2f}%), similar to seasonal discounts in wholesale markets. "
                f"It is advisable for new investors to stay patient, using Rs. {support} as a protective safety net before entering.")


def fetch_historical_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y", interval="1d")
        if not df.empty and len(df) >= 60:
            df = df.reset_index()
            date_col = "Date" if "Date" in df.columns else "Datetime"
            df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            return df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
    except Exception as e:
        print(f"[WARN] Failed to fetch data for {symbol}: {e}")

    print(f"[WARN] Insufficient or unavailable real market data for {symbol}. Skipping asset.")
    return pd.DataFrame()


def main():
    print("[INFO] Initializing Kronos Indian Market Advisor Pipeline...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[LOAD] Loading ~400MB Foundation Model ({KRONOS_MODEL_NAME}) on device: {device}...")
    
    tokenizer_kwargs = {"token": get_hf_token()} if get_hf_token() else {}
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME, **tokenizer_kwargs)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME, **tokenizer_kwargs)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    lookback = 120
    pred_len = 14
    forecasts = []

    print("[EXEC] Analyzing Indian equities and currency benchmarks with Ollama commentary...")
    for asset in INDIAN_ASSETS:
        df = fetch_historical_data(asset["symbol"])
        if len(df) < lookback:
            if not df.empty:
                print(f"[WARN] Skipping {asset['symbol']}: only {len(df)} candles available (need {lookback}).")
            continue

        x_df = df.iloc[-lookback:][['open', 'high', 'low', 'close']].reset_index(drop=True)
        x_timestamp = df.iloc[-lookback:]['timestamps'].reset_index(drop=True)
        
        last_date = x_timestamp.iloc[-1]
        y_timestamp = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_len))

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.8,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )

        curr_close = round(float(x_df["close"].iloc[-1]), 2)
        proj_close = float(pred_df["close"].iloc[-1])
        support = round(float(pred_df["low"].min()), 2)
        resistance = round(float(pred_df["high"].max()), 2)
        ret_pct = round(((proj_close - curr_close) / curr_close) * 100, 2)
        
        trend = "Bullish" if ret_pct >= 0.5 else ("Bearish" if ret_pct <= -0.5 else "Neutral")
        
        raw_exp = generate_local_explanation(asset["name"], curr_close, support, resistance, ret_pct, trend)
        explanation = " ".join(raw_exp.split())

        forecast_item = {
            "symbol": asset["symbol"],
            "name": asset["name"],
            "current_price": f"{curr_close:,.2f}",
            "support": f"{support:,.2f}",
            "resistance": f"{resistance:,.2f}",
            "return_pct": ret_pct,
            "trend": trend,
            "explanation": explanation
        }
        forecasts.append(forecast_item)
        print(f"  [OK] Completed Forecast: {asset['name']} ({trend}) | Proj Return: {ret_pct}%")

    # Save outputs.md
    md_lines = [
        "# 🇮🇳 Kronos Indian Market Advisory Report\n",
        "**Powered by NeoQuasar/Kronos-base (~400MB) Foundation Model & Local Ollama Llama 3.2 3B**\n",
        "| Asset | Current Price (₹) | Support (₹) | Resistance (₹) | 14-Day Return | Outlook | Beginner Guidance |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]
    for item in forecasts:
        md_lines.append(f"| **{item['name']}** ({item['symbol']}) | ₹{item['current_price']} | ₹{item['support']} | ₹{item['resistance']} | **{item['return_pct']}%** | {item['trend']} | {item['explanation']} |")
    
    md_lines.extend([
        "\n---",
        "*Disclaimer: AI quantitative predictions are generated for educational and analytical purposes. Always consult a certified SEBI registered advisor before real-capital investments.*"
    ])

    with open("outputs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("[SAVE] Saved structured forecast report to outputs.md")

    # Render HTML visual deliverable using Jinja2
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report_template.html")
    html_out = template.render(forecasts=forecasts)
    
    with open("indian_market_report.html", "w", encoding="utf-8") as f:
        f.write(html_out)
    print("[SAVE] Generated responsive visualizer at indian_market_report.html")
    print("[DONE] Execution Successful! Indian market quant advisor pipeline ready.")

if __name__ == "__main__":
    main()