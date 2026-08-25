import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from jinja2 import Template

# Reconfigure stdout for UTF-8 support on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Ensure Kronos library path is reachable
sys.path.append("./kronos_lib")
from model import Kronos, KronosTokenizer, KronosPredictor

KRONOS_MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"


# ==============================================================================
# 1. ADAPTIVE MODEL HYPERPARAMETERS & CONFIGURATION STATE
# ==============================================================================
class AdaptiveModelConfig:
    """
    Holds and manages all tunable parameters for Kronos AI model and
    the self-learning adaptation layer.
    """
    def __init__(
        self,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 0,
        sample_count: int = 1,
        lookback_window: int = 60,
        momentum_weight: float = 0.5,
        volatility_scaler: float = 1.0,
        bias_correction: float = 0.0,
        learning_rate: float = 0.05
    ):
        self.temperature = float(temperature)           # Sampling temperature (T)
        self.top_p = float(top_p)                       # Nucleus sampling threshold
        self.top_k = int(top_k)                         # Top-K threshold
        self.sample_count = int(sample_count)           # Monte Carlo ensemble samples
        self.lookback_window = int(lookback_window)     # Number of historical candles fed to Kronos
        self.momentum_weight = float(momentum_weight)   # Momentum vs Mean-Reversion bias
        self.volatility_scaler = float(volatility_scaler) # High-Low range multiplier
        self.bias_correction = float(bias_correction)   # Online residual drift compensation
        self.learning_rate = float(learning_rate)       # Rate of parameter updates

    def to_dict(self) -> dict:
        return {
            "temperature": round(self.temperature, 3),
            "top_p": round(self.top_p, 3),
            "top_k": self.top_k,
            "sample_count": self.sample_count,
            "lookback_window": self.lookback_window,
            "momentum_weight": round(self.momentum_weight, 3),
            "volatility_scaler": round(self.volatility_scaler, 3),
            "bias_correction": round(self.bias_correction, 4),
            "learning_rate": self.learning_rate
        }


# ==============================================================================
# 2. HISTORICAL & LIVE MARKET DATA FEED
# ==============================================================================
class MarketDataFeed:
    @staticmethod
    def fetch_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetches structured OHLCV time series data formatted for Kronos.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if not df.empty and len(df) >= 70:
                df = df.reset_index()
                date_col = "Date" if "Date" in df.columns else "Datetime"
                df["timestamps"] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
                df = df.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume"
                })
                return df[["timestamps", "open", "high", "low", "close", "volume"]].dropna()
        except Exception as e:
            print(f"[WARN] Failed to fetch data for {symbol}: {e}")

        print(f"[WARN] Insufficient or unavailable real market data for {symbol}.")
        return pd.DataFrame()


# ==============================================================================
# 3. KRONOS PREDICTION & SELF-LEARNING AGENT
# ==============================================================================
class KronosSelfLearningAgent:
    """
    Self-learning AI agent that:
    1. Generates 1-step next candle predictions using Kronos.
    2. Diagnoses prediction shortfall against observed reality.
    3. Self-tunes and modifies model parameters in real-time.
    4. Repeats in a continuous feedback loop.
    """
    def __init__(self, config: AdaptiveModelConfig, device: str = None):
        self.config = config
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INIT] Loading Kronos AI (~400MB Foundation Model) on {self.device}...")
        
        self.tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        self.model = Kronos.from_pretrained(KRONOS_MODEL_NAME)
        self.predictor = KronosPredictor(self.model, self.tokenizer, device=self.device, max_context=512)

        # Performance tracking metrics
        self.history = []
        self.total_predictions = 0
        self.correct_directions = 0
        self.cumulative_mape = 0.0

    def predict_next_candle(self, context_df: pd.DataFrame) -> dict:
        """
        Uses historical context candles to forecast the exact next candle (t+1).
        Applies current adaptive hyperparameter states.
        """
        lookback = min(len(context_df), self.config.lookback_window)
        x_df = context_df.iloc[-lookback:][['open', 'high', 'low', 'close']].reset_index(drop=True)
        x_timestamps = context_df.iloc[-lookback:]['timestamps'].reset_index(drop=True)
        
        last_date = x_timestamps.iloc[-1]
        b_days = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)
        next_date = b_days[0]
        y_timestamps = pd.Series([next_date])

        # Run Kronos Foundation autoregressive prediction
        pred_df = self.predictor.predict(
            df=x_df,
            x_timestamp=x_timestamps,
            y_timestamp=y_timestamps,
            pred_len=1,
            T=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            sample_count=self.config.sample_count,
            verbose=False
        )

        last_close = float(x_df['close'].iloc[-1])
        raw_open = float(pred_df['open'].iloc[0])
        raw_high = float(pred_df['high'].iloc[0])
        raw_low = float(pred_df['low'].iloc[0])
        raw_close = float(pred_df['close'].iloc[0])

        # Apply learned adaptive calibrations and momentum weighting
        raw_move_pct = (raw_close - last_close) / last_close
        momentum_adjusted_move = raw_move_pct * (0.5 + 0.5 * self.config.momentum_weight)
        calibrated_close = last_close * (1.0 + momentum_adjusted_move + self.config.bias_correction)
        pred_ret = ((calibrated_close - last_close) / last_close) * 100

        # Adjust range by learned volatility scaler
        spread = (raw_high - raw_low) * self.config.volatility_scaler
        mid = (raw_high + raw_low) / 2.0
        calibrated_high = max(mid + (spread / 2.0), calibrated_close, raw_open)
        calibrated_low = min(mid - (spread / 2.0), calibrated_close, raw_open)

        pred_direction = "UP" if pred_ret >= 0 else "DOWN"

        return {
            "timestamp": next_date,
            "last_close": round(last_close, 2),
            "pred_open": round(raw_open, 2),
            "pred_high": round(calibrated_high, 2),
            "pred_low": round(calibrated_low, 2),
            "pred_close": round(calibrated_close, 2),
            "pred_ret": round(pred_ret, 2),
            "pred_direction": pred_direction
        }

    def diagnose_and_learn(self, prediction: dict, actual_candle: pd.Series) -> dict:
        """
        Compares predicted candle with actual observed candle, diagnoses the shortfall,
        and modifies parameters in self-training loop.
        """
        last_close = prediction["last_close"]
        act_open = float(actual_candle["open"])
        act_high = float(actual_candle["high"])
        act_low = float(actual_candle["low"])
        act_close = float(actual_candle["close"])

        act_ret = ((act_close - last_close) / last_close) * 100
        act_direction = "UP" if act_ret >= 0 else "DOWN"

        # Direction match test
        is_direction_hit = (prediction["pred_direction"] == act_direction)
        
        # Errors
        close_error_abs = abs(prediction["pred_close"] - act_close)
        mape = (close_error_abs / act_close) * 100
        
        pred_range = prediction["pred_high"] - prediction["pred_low"]
        act_range = act_high - act_low
        range_error = act_range - pred_range  # Positive if actual was more volatile than predicted

        # -------------------------------------------------------------
        # SHORTFALL DIAGNOSIS & PARAMETER MODIFICATION (SELF-LEARNING)
        # -------------------------------------------------------------
        shortfall_diagnostics = []
        lr = self.config.learning_rate

        # 1. Bias Correction (Residual Drift Update)
        residual_pct = (act_close - prediction["pred_close"]) / act_close
        self.config.bias_correction += lr * residual_pct
        self.config.bias_correction = float(np.clip(self.config.bias_correction, -0.05, 0.05))

        # 2. Direction Miss Shortfall
        if not is_direction_hit:
            shortfall_diagnostics.append("Direction Miss (False Momentum / Reversal)")
            # If direction failed, reduce momentum weight and adjust temperature to escape local mode
            self.config.momentum_weight = max(0.1, self.config.momentum_weight - (lr * 1.5))
            if self.config.temperature < 0.95:
                self.config.temperature = min(1.2, self.config.temperature + 0.05)
            else:
                self.config.temperature = max(0.5, self.config.temperature - 0.05)
        else:
            shortfall_diagnostics.append("Direction Match (Momentum Confirmed)")
            self.config.momentum_weight = min(0.9, self.config.momentum_weight + (lr * 0.5))

        # 3. Volatility / Range Shortfall
        if range_error > 0.15 * act_range:
            shortfall_diagnostics.append("Range Underestimated (Surge in Volatility)")
            self.config.volatility_scaler = min(2.0, self.config.volatility_scaler + (lr * 1.2))
            if self.config.sample_count < 3:
                self.config.sample_count += 1
        elif range_error < -0.15 * act_range:
            shortfall_diagnostics.append("Range Overestimated (Low Noise Consolidation)")
            self.config.volatility_scaler = max(0.5, self.config.volatility_scaler - (lr * 0.8))
            self.config.top_p = max(0.75, self.config.top_p - 0.02)

        # 4. Context Window Tuning (Lookback)
        if mape > 2.5:
            # High error -> expand context for more historical grounding
            self.config.lookback_window = min(120, self.config.lookback_window + 5)
        elif mape < 0.8:
            # Low error -> lock current optimal lookback
            pass

        # Update stats
        self.total_predictions += 1
        if is_direction_hit:
            self.correct_directions += 1
        self.cumulative_mape += mape
        
        running_winrate = (self.correct_directions / self.total_predictions) * 100
        running_avg_mape = self.cumulative_mape / self.total_predictions

        log_entry = {
            "step": self.total_predictions,
            "date": str(actual_candle["timestamps"].strftime("%Y-%m-%d")),
            "last_close": last_close,
            "pred_close": prediction["pred_close"],
            "act_close": round(act_close, 2),
            "pred_ret": prediction["pred_ret"],
            "act_ret": round(act_ret, 2),
            "pred_dir": prediction["pred_direction"],
            "act_dir": act_direction,
            "is_hit": is_direction_hit,
            "mape": round(mape, 2),
            "running_winrate": round(running_winrate, 1),
            "running_avg_mape": round(running_avg_mape, 2),
            "shortfall_reasons": "; ".join(shortfall_diagnostics),
            "updated_params": self.config.to_dict()
        }
        self.history.append(log_entry)
        return log_entry


# ==============================================================================
# 4. CONTINUOUS LEARNING LOOP RUNNERS (SIMULATION & LIVE MODES)
# ==============================================================================
def run_walk_forward_learning_sim(
    symbol: str = "RELIANCE.NS",
    max_steps: int = 30,
    initial_config: AdaptiveModelConfig = None
):
    """
    Executes a walk-forward candle-by-candle continuous learning loop across historical data.
    Demonstrates parameter self-tuning and accuracy convergence in real-time.
    """
    config = initial_config or AdaptiveModelConfig()
    agent = KronosSelfLearningAgent(config=config)

    print(f"\n[FEED] Fetching market stream for {symbol}...")
    df = MarketDataFeed.fetch_data(symbol=symbol, period="1y")

    warmup_period = config.lookback_window + 10
    if len(df) <= warmup_period + 10:
        print("[ERROR] Insufficient data length for meaningful self-learning loop.")
        return

    test_slice = df.iloc[warmup_period: warmup_period + max_steps].reset_index(drop=True)
    
    print("\n" + "=" * 105)
    print(f"{'🧠 KRONOS AI CONTINUOUS SELF-LEARNING & NEXT-CANDLE ADAPTATION LOOP':^105}")
    print("=" * 105)
    print(f"Target Asset : {symbol} | Test Horizons : {len(test_slice)} Sequential Candles")
    print(f"Initial State: Temp={config.temperature}, TopP={config.top_p}, Lookback={config.lookback_window}, LR={config.learning_rate}")
    print("-" * 105)
    header = f"{'Step':<6}{'Date':<12}{'Pred Ret':<10}{'Act Ret':<10}{'Hit?':<7}{'MAPE %':<9}{'WinRate':<10}{'Diagnosed Shortfall / Adaptive Action'}"
    print(header)
    print("-" * 105)

    for i in range(len(test_slice)):
        # Context includes all data strictly up to the current moment (warmup + i)
        context_df = df.iloc[: warmup_period + i].reset_index(drop=True)
        actual_next_candle = test_slice.iloc[i]

        # 1. Predict Next Candle with current parameters
        pred = agent.predict_next_candle(context_df)

        # 2. Observe Next Candle, diagnose shortfall, and self-train parameters
        result = agent.diagnose_and_learn(pred, actual_next_candle)

        # Print real-time self-learning step
        hit_icon = "✅ YES" if result["is_hit"] else "❌ NO "
        p_ret = f"{'+' if result['pred_ret'] >= 0 else ''}{result['pred_ret']:.2f}%"
        a_ret = f"{'+' if result['act_ret'] >= 0 else ''}{result['act_ret']:.2f}%"
        print(f"#{result['step']:<5}{result['date']:<12}{p_ret:<10}{a_ret:<10}{hit_icon:<7}{result['mape']:<9.2f}{result['running_winrate']:<10.1f}{result['shortfall_reasons']}")

    print("=" * 105)
    print(f"\n[SUMMARY] Learning Session Completed across {agent.total_predictions} sequential candles:")
    print(f"  • Final Directional Win Rate : {agent.correct_directions / agent.total_predictions * 100:.1f}%")
    print(f"  • Final Average Price Error : {agent.cumulative_mape / agent.total_predictions:.2f}%")
    print(f"  • Final Self-Tuned Parameters: {json.dumps(agent.config.to_dict(), indent=4)}")
    print("=" * 105 + "\n")


def run_infinite_live_learning_loop(
    symbol: str = "^NSEI",
    poll_interval_sec: int = 15,
    initial_config: AdaptiveModelConfig = None
):
    """
    Runs an infinite continuous loop that polls for new incoming candles in real-time,
    forecasts the next candle, evaluates errors upon new candle close, and automatically self-tunes.
    """
    config = initial_config or AdaptiveModelConfig()
    agent = KronosSelfLearningAgent(config=config)

    print("\n" + "=" * 90)
    print(f"{'⚡ KRONOS AI INFINITE REAL-TIME SELF-LEARNING LOOP STARTED':^90}")
    print(f"{'Press Ctrl+C at any time to pause or exit':^90}")
    print("=" * 90)

    last_processed_timestamp = None

    while True:
        try:
            df = MarketDataFeed.fetch_data(symbol=symbol, period="6mo", interval="1d")
            latest_candle = df.iloc[-1]
            current_timestamp = latest_candle["timestamps"]

            if last_processed_timestamp is None:
                # Initial prediction
                pred = agent.predict_next_candle(df)
                last_processed_timestamp = current_timestamp
                print(f"\n[PREDICTION] Next Candle Forecast for {symbol} at {pred['timestamp'].strftime('%Y-%m-%d')}:")
                print(f"  • Expected Close: Rs. {pred['pred_close']:,.2f} ({'+' if pred['pred_ret'] >= 0 else ''}{pred['pred_ret']}%) [{pred['pred_direction']}]")
                print(f"  • Expected Range: Rs. {pred['pred_low']:,.2f} to Rs. {pred['pred_high']:,.2f}")
                print(f"  • Active Parameters: Temp={config.temperature:.2f}, VolScaler={config.volatility_scaler:.2f}, Bias={config.bias_correction:+.4f}")
                print(f"[WAIT] Monitoring market for candle resolution (polling every {poll_interval_sec}s)...")

            elif current_timestamp != last_processed_timestamp:
                # A new candle resolved! Evaluate and self-tune
                print(f"\n[NEW CANDLE RESOLVED] Analyzing outcome for {last_processed_timestamp.strftime('%Y-%m-%d')}...")
                result = agent.diagnose_and_learn(pred, latest_candle)
                
                print(f"  • Direction Match : {'✅ YES' if result['is_hit'] else '❌ NO'} (Pred: {result['pred_dir']}, Actual: {result['act_dir']})")
                print(f"  • Actual Close    : Rs. {result['act_close']} vs Predicted Rs. {result['pred_close']} (Error: {result['mape']}%)")
                print(f"  • Shortfall Review: {result['shortfall_reasons']}")
                print(f"  • Self-Tuned State: {json.dumps(agent.config.to_dict())}")
                print(f"  • Running Win Rate: {result['running_winrate']}% over {result['step']} candles")

                # Generate next candle prediction with revised parameters
                pred = agent.predict_next_candle(df)
                last_processed_timestamp = current_timestamp
                print(f"\n[NEXT PREDICTION] Generated with REVISED Self-Learned Parameters:")
                print(f"  • Expected Close: Rs. {pred['pred_close']:,.2f} ({'+' if pred['pred_ret'] >= 0 else ''}{pred['pred_ret']}%) [{pred['pred_direction']}]")

            time.sleep(poll_interval_sec)

        except KeyboardInterrupt:
            print("\n[STOP] Infinite learning loop paused by user.")
            break
        except Exception as e:
            print(f"[WARN] Polling exception: {e}. Retrying in {poll_interval_sec}s...")
            time.sleep(poll_interval_sec)


# ==============================================================================
# 5. CLI ENTRYPOINT WITH USER-CONFIGURABLE PARAMETERS
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Kronos AI Self-Learning & Parameter Adaptation Engine")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Stock ticker (e.g. RELIANCE.NS, ^NSEI, TCS.NS)")
    parser.add_argument("--mode", type=str, choices=["sim", "infinite"], default="sim", help="'sim' for walk-forward simulation, 'infinite' for live loop")
    parser.add_argument("--steps", type=int, default=25, help="Number of candles to simulate in 'sim' mode")
    parser.add_argument("--temp", type=float, default=0.8, help="Initial sampling temperature T (default: 0.8)")
    parser.add_argument("--topp", type=float, default=0.9, help="Initial top-p nucleus sampling (default: 0.9)")
    parser.add_argument("--lookback", type=int, default=60, help="Initial lookback window (default: 60)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for online parameter updates (default: 0.05)")

    args = parser.parse_args()

    # User-customized initial parameters
    config = AdaptiveModelConfig(
        temperature=args.temp,
        top_p=args.topp,
        lookback_window=args.lookback,
        learning_rate=args.lr
    )

    if args.mode == "sim":
        run_walk_forward_learning_sim(symbol=args.symbol, max_steps=args.steps, initial_config=config)
    else:
        run_infinite_live_learning_loop(symbol=args.symbol, poll_interval_sec=15, initial_config=config)


if __name__ == "__main__":
    main()
