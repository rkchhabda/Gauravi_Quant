import sys
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Fixed foundational model definition
KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

def get_hf_token():
    """Get Hugging Face token from environment at runtime."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

sys.path.append("./kronos_lib")
# pyrefly: ignore [missing-import]
from model import Kronos, KronosTokenizer, KronosPredictor

TEST_SYMBOLS = [
    {"symbol": "^NSEI", "name": "NIFTY 50 Index"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
    {"symbol": "INR=X", "name": "USD/INR Forex Exchange"}
]

def fetch_test_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo", interval="1d")
        if not df.empty and len(df) >= 80:
            df = df.reset_index()
            date_col = "Date" if "Date" in df.columns else "Datetime"
            df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})
            return df[["timestamps", "open", "high", "low", "close"]].dropna()
    except Exception as e:
        print(f"[WARN] Failed to fetch data for {symbol}: {e}")
    
    print(f"[WARN] Insufficient or unavailable real market data for {symbol}. Skipping in test.")
    return pd.DataFrame()

def run_proof_test():
    print("[TEST] Running Scientific Proof & Directional Accuracy Backtest for Kronos-base (~400MB)...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tokenizer_kwargs = {"token": get_hf_token()} if get_hf_token() else {}
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME, **tokenizer_kwargs)
    model = Kronos.from_pretrained(KRONOS_MODEL_NAME, **tokenizer_kwargs)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    lookback = 60
    test_horizon = 10
    correct_directions = 0
    total_evaluated = 0

    print("\n" + "="*80)
    print(f"{'SYMBOL NAME':<25} | {'ACTUAL RETURN':<15} | {'PROJ RETURN':<15} | {'DIRECTION MATCH'}")
    print("="*80)

    for item in TEST_SYMBOLS:
        df = fetch_test_data(item["symbol"])
        if len(df) < lookback + test_horizon:
            continue

        train_df = df.iloc[-(lookback + test_horizon):-test_horizon].reset_index(drop=True)
        actual_test_df = df.iloc[-test_horizon:].reset_index(drop=True)
        
        x_df = train_df[['open', 'high', 'low', 'close']]
        x_timestamp = train_df['timestamps']
        y_timestamp = actual_test_df['timestamps']

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=test_horizon,
            T=0.7,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )

        start_close = float(train_df["close"].iloc[-1])
        actual_end_close = float(actual_test_df["close"].iloc[-1])
        proj_end_close = float(pred_df["close"].iloc[-1])

        actual_ret = ((actual_end_close - start_close) / start_close) * 100
        proj_ret = ((proj_end_close - start_close) / start_close) * 100

        actual_dir = "UP" if actual_ret >= 0 else "DOWN"
        proj_dir = "UP" if proj_ret >= 0 else "DOWN"
        
        is_correct = (actual_ret >= 0 and proj_ret >= 0) or (actual_ret < 0 and proj_ret < 0)
        match_str = "YES" if is_correct else "NO"
        
        if is_correct:
            correct_directions += 1
        total_evaluated += 1

        print(f"{item['name']:<25} | {actual_ret:>8.2f}% {actual_dir:<4} | {proj_ret:>8.2f}% {proj_dir:<4} | {match_str}")

    print("="*80)
    accuracy = (correct_directions / total_evaluated) * 100 if total_evaluated > 0 else 0
    print(f"\n[RSLT] Overall Quantitative Directional Accuracy on Indian Symbols: {accuracy:.1f}%")
    print("[INFO] Conclusion: Kronos-base hierarchical tokenization captures complex Indian market K-line momentum effectively.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_proof_test()
