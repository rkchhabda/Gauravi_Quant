<div align="center">
  <a href="https://youtu.be/AdtyZJ7rxsk">
    <img src="https://img.youtube.com/vi/AdtyZJ7rxsk/0.jpg" alt="Kronos AI: Tiny Model That Beats Expensive Trading Bots">
  </a>
  <h3>📺 <a href="https://youtu.be/AdtyZJ7rxsk">Watch the full tutorial on YouTube</a></h3>
</div>

# 🇮🇳 Kronos Indian Market Advisor

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![AI: Kronos Base](https://img.shields.io/badge/Foundation_Model-Kronos_Base_400MB-blue.svg)
![LLM: Llama 3.2 3B](https://img.shields.io/badge/Local_LLM-Ollama_Llama3.2:3B-purple.svg)
![Market: Indian Stock & Forex](https://img.shields.io/badge/Market-NSE_|_BSE_|_Forex-orange.svg)

An advanced, AI-driven quantitative prediction and interpretation system designed specifically for Indian financial investors and beginners. Powered by **Kronos-base** (the ~400MB open-source candlestick foundation model trained on global exchanges) and localized by **Llama 3.2 3B** via Ollama, this project translates complex K-line sequences into jargon-free, high-value financial insights for equities like NIFTY 50, Reliance Industries, HDFC Bank, Gold ETFs, and USD/INR exchange rates.

---

## 🚀 Key Highlights

* **Kronos Foundation Model**: Utilizes the 102.3M parameter (`~400MB`) base model with hierarchical K-line tokenization to predict future OHLCV candlestick paths.
* **Beginner-Friendly Interpretation**: Local Ollama AI analyzes quantitative forecasting metrics and translates them into intuitive analogies tailored to Indian market participants.
* **Empirical Proof Testing**: Includes an automated quantitative validation script to measure directional accuracy and support/resistance resilience against historical NSE data.
* **Responsive HTML Visualizers**: Generates aesthetic, interactive reporting templates featuring dark-mode glassmorphism and instant markdown export capabilities.

---

## 🛠️ Tech Stack

* **Quantitative Forecaster**: `NeoQuasar/Kronos-base` (Hugging Face Transformers & PyTorch)
* **Local Explainer AI**: `llama3.2:3b` (via Ollama, thinking disabled for rapid inference)
* **Market Data Provider**: `yfinance` (Live NSE Indian market indices, corporate stocks, and commodities)
* **Visual Templating**: `Jinja2` with custom responsive CSS styling
* **Data & Math Engine**: `pandas`, `numpy`, `matplotlib`, `torch`

---

## 📦 Installation & Setup

All commands below can be executed directly in your **Windows Antigravity PowerShell terminal** by copy-pasting.

### 1. Clone & Install Dependencies
First, clone the official Kronos foundation model repository into `kronos_lib` and install standard Python dependencies:
```powershell
git clone https://github.com/shiyu-coder/Kronos.git kronos_lib
pip install --upgrade transformers huggingface_hub safetensors pandas einops matplotlib tqdm yfinance jinja2 requests torch
```

### 2. Download Local Ollama Model
Ensure Ollama is running, and pull the required 3B instruction model:
```powershell
ollama run llama3.2:3b
```

### 3. Verify Foundation Model Cache
The first execution automatically caches the ~400MB `NeoQuasar/Kronos-base` model and `NeoQuasar/Kronos-Tokenizer-base` from Hugging Face Hub into your system storage.

---

## 🏃‍♂️ Run Instructions

### Execute Market Advisor
Run the primary pipeline to fetch latest Indian market K-lines, generate 14-day predictions, compute support/resistance levels, and produce beginner-friendly advisory reports:
```powershell
python market_advisor.py
```

Upon execution, the system creates:
* `outputs.md`: A structured markdown executive summary of forecasts.
* `indian_market_report.html`: A visually immersive HTML report with download triggers.

### Run Scientific Proof Test
Validate Kronos-base predictive directional reliability by backtesting recent historical periods against actual ground truth closing prices on Indian symbols:
```powershell
python proof_test.py
```

### Pooled Factor Model Research
Leakage-safe pooled cross-sectional model over NIFTY 100 stocks:
```powershell
python pooled_model_v1.py --period 5y --model xgb --horizon 10
python factor_ic_analysis.py
python composite_factor_strategy.py
```

### Random-20 Backtest
Backtests the liquidity+reversal composite on 20 randomly chosen NIFTY-100 stocks from a given date:
```powershell
python backtest_random20.py --seed 42 --top-k 5 --cost-bps 10
```

---

## ⚠️ Realistic Accuracy Expectations

This project is for **research and educational purposes only**. Honest out-of-sample validation on real NSE data (see `accuracy_log.md`) shows:

- **15-minute directional prediction**: no predictive signal (AUC ~0.50) - intraday modules removed
- **Per-symbol ML models**: AUC ~0.50 pooled across 97 stocks - not a reliable edge
- **Liquidity + short-term reversal composite**: the only validated edge (IC 0.046, t=4.38), modest: ~52% hit rate, mid-single-digit annualized excess vs NIFTY at quintile breadth

If you need trading signals, prefer **risk management, position sizing, and edge filtering** over chasing a single high-accuracy model.

---

## 📂 File Explanations

| File / Folder | Description |
| :--- | :--- |
| `market_advisor.py` | Main application script orchestrating data retrieval, Kronos K-line forecasting, LLM translation, and report output generation. |
| `stock_analyzer.py` | High-speed concurrent technical screener generating multi-factor health scores and `stock_screener_dashboard.html`. |
| `proof_test.py` | Scientific directional backtest verification tool evaluating foundation model hit ratios on Indian symbols. |
| `pooled_model_v1.py` | Leakage-safe pooled cross-sectional model over NIFTY 100 (walk-forward, purged, baseline-aware). |
| `composite_factor_strategy.py` | Validated liquidity + short-term reversal composite factor strategy. |
| `backtest_random20.py` | Random-20 NIFTY-100 backtest of the composite strategy from a start date. |
| `enhanced_predictor.py` | Multi-model ensemble (Kronos + LightGBM + ExtraTrees) walk-forward validator with market context features. |
| `kronos_self_learning_agent.py` | Continuous daily self-learning agent with adaptive hyperparameter calibration. |
| `backtest_engine.py` | Multi-asset portfolio backtester testing score monotonicity, IC, and Top-5 systematic rebalancing strategies. |
| `templates/report_template.html` | Jinja2 responsive HTML visualizer template featuring dynamic UI styling and standalone export features. |
| `outputs.md` | Automated generated advisory output detailing target levels and simplified Indian financial commentary. |
| `kronos_lib/` | Core library folder cloned from official Kronos repository providing autoregressive Transformer and hierarchical tokenization architectures. |

---

## 💡 5 Core Use Cases

1. **Retail NIFTY 50 Trend Navigation**: Helps novice Indian investors understand market momentum without mastering complex technical indicators like RSI or MACD.
2. **Blue-Chip Swing Level Identification**: Automates the detection of entry and profit-booking targets for high-liquidity Indian Equities (e.g., Reliance Industries, HDFC Bank).
3. **Gold ETF & Rupee Volatility Hedging**: Monitors commodities and USD/INR exchange paths to guide risk mitigation for household portfolios and importers.
4. **Jargon-Free Financial Literacy**: Converts complex quant output (standard deviation, nucleus sampling intervals, trend lines) into simple explanations using everyday analogies.
5. **Pre-Market quantitative Screening**: Enables disciplined traders to objectively benchmark their qualitative trading hypothesis against an unbiased foundation model.

---

## 🔮 5 Future Features

1. **Multi-Timeframe Confluence Engine**: Combining 5-minute intraday candlesticks with daily closing sequences to generate intraday scalping insights.
2. **Indian Economic Events Integration**: Injecting RBI monetary policy schedules and quarterly corporate earnings dates directly into Ollama context windows.
3. **Interactive Charting with Lightweight Charts**: Transitioning standard visual generation to full real-time interactive HTML5 trading canvases.
4. **Automated Telegram Alert Bot**: Periodic scheduled cron execution delivering customized morning advisory notes directly to mobile messaging channels.
5. **Custom NSE Finetuned LoRA Weights**: Specializing the core Kronos foundation architecture directly on 15-year micro-cap and small-cap Indian market sequences.

---

## 🏷️ Keywords & Tags

`kronos-ai` `time-series-forecasting` `candlestick-prediction` `nifty50-forecasting` `indian-stock-market` `financial-foundation-model` `ollama` `llama3.2` `algorithmic-trading` `quant-finance` `python-trading-bot`

---
*Built with simplicity, elegance, and AI quantitative accuracy for the Indian financial ecosystem.*
