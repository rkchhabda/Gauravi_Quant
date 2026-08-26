# 🇮🇳 Gauravi Indian Market Advisor

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![AI: Kronos Base](https://img.shields.io/badge/Foundation_Model-Kronos_Base_400MB-blue.svg)
![LLM: Llama 3.2 3B](https://img.shields.io/badge/Local_LLM-Ollama_Llama3.2:3B-purple.svg)
![Market: Indian Stock & Forex](https://img.shields.io/badge/Market-NSE_|_BSE_|_Forex-orange.svg)

An advanced, AI-driven quantitative prediction and interpretation system designed specifically for Indian financial investors and beginners. Powered by **Kronos-base** (the ~400MB open-source candlestick foundation model trained on global exchanges) and localized by **Llama 3.2 3B** via Ollama, this project translates complex K-line sequences into jargon-free, high-value financial insights for NIFTY 100 equities, and ships a **fully automated weekly stock-ranking product** published to GitHub Pages every Monday.

---

## 🚀 Key Highlights

* **Kronos Foundation Model**: Utilizes the 102.3M parameter (`~400MB`) base model with hierarchical K-line tokenization to predict future OHLCV candlestick paths.
* **Weekly Ranking Product (v1)**: Every Monday, `weekly_ranking_report.py` ranks NIFTY-100 stocks by a momentum + short-term-reversal blend, flags earnings-date risk, incorporates FII/DII flows, and snapshots Top-20 / Bottom-20 picks to `outputs/rankings/` as a permanent paper-trading track record.
* **CI/CD + GitHub Pages**: `.github/workflows/weekly_ranking.yml` runs the report automatically every Monday 09:00 IST, commits the snapshot, and publishes an archive site built by `build_rankings_index.py`.
* **Honest, Validated Research**: Pooled cross-sectional models over 97 NIFTY-100 stocks (~95k predictions) with purged walk-forward validation — including full model comparisons showing which factors have a real edge and which don't.
* **Beginner-Friendly Interpretation**: Local Ollama AI translates quantitative forecasting metrics into intuitive analogies tailored to Indian market participants.

---

## 🛠️ Tech Stack

* **Quantitative Forecaster**: `NeoQuasar/Kronos-base` (Hugging Face Transformers & PyTorch)
* **Local Explainer AI**: `llama3.2:3b` (via Ollama)
* **Factor Research**: `scikit-learn`, `xgboost`, `scipy` (pooled walk-forward models & IC analysis)
* **Market Data Provider**: `yfinance` (Live NSE Indian indices, corporate stocks, commodities) + manual FII/DII flow CSV (`data/fii_dii.csv`)
* **Automation**: GitHub Actions (scheduled weekly runs) + GitHub Pages (report publishing)
* **Data & Math Engine**: `pandas`, `numpy`

---

## 📦 Installation & Setup

### 1. Core Dependencies (rankings & research pipeline)

```powershell
git clone https://github.com/rkchhabda/Gauravi_Quant.git
cd Gauravi_Quant
pip install -r requirements.txt
```

### 2. Optional: Kronos Foundation Model

Only needed for `market_advisor.py` / `proof_test.py`. Clone the official Kronos repository into `kronos_lib`:

```powershell
git clone https://github.com/shiyu-coder/Kronos.git kronos_lib
pip install --upgrade transformers huggingface_hub safetensors einops matplotlib tqdm jinja2 requests torch
```

The first execution automatically caches the ~400MB `NeoQuasar/Kronos-base` model and `NeoQuasar/Kronos-Tokenizer-base` from Hugging Face Hub.

### 3. Optional: Local Ollama Model

```powershell
ollama run llama3.2:3b
```

---

## 🏃‍♂️ Run Instructions

### Weekly Ranking Report (main product)

Generates the weekly Top-20 / Bottom-20 NIFTY-100 ranking with earnings flags and FII/DII context:

```powershell
python weekly_ranking_report.py
```

Outputs `outputs/rankings/ranking_YYYY-MM-DD.md/.html/.json` snapshots. Runs automatically via GitHub Actions every Monday 09:00 IST, or on-demand in Colab via `colab_run.ipynb`.

> **Weekly rhythm:** run the report → update `data/fii_dii.csv` (`date,fii_net_cr,dii_net_cr` rows from NSDL) → verify last week's picks.

### Execute Market Advisor

Fetches latest Indian market K-lines, generates predictions, computes support/resistance levels, and produces beginner-friendly advisory reports:

```powershell
python market_advisor.py
```

Upon execution, the system creates:
* `outputs.md`: A structured markdown executive summary of forecasts.
* `indian_market_report.html`: A visually immersive HTML report with download triggers.

### Model Comparison Research

```powershell
python pooled_model_v1.py --period 5y --model xgb --horizon 10
python factor_ic_analysis.py
python composite_factor_strategy.py
python model_comparison.py
```

### Random-20 Backtest

Backtests the liquidity+reversal composite on 20 randomly chosen NIFTY-100 stocks from a given date:

```powershell
python backtest_random20.py --seed 42 --top-k 5 --cost-bps 10
```

---

## ⚠️ Realistic Accuracy Expectations

This project is for **research and educational purposes only**. Honest out-of-sample validation on real NSE data (see `accuracy_log.md`) shows:

| Model | Verdict |
|-------|---------|
| Momentum 12-1 | Best & most consistent factor (AUC ~0.510) |
| Reversal | Best tradable portfolio (+7.3%/yr excess, Sharpe ~0.54) |
| Liquidity + reversal composite | Strong 2024–26 but regime-dependent across 2023–26 |
| XGBoost pooled | Marginal (AUC ~0.5025) |
| Random Forest / LogReg | No edge (~random) |
| Per-symbol ML models | Claims retracted (leakage found) |
| 15-minute intraday | Removed — zero signal (AUC ~0.50) |

If you need trading signals, prefer **risk management, position sizing, and edge filtering** over chasing a single high-accuracy model.

---

## 📂 File Explanations

| File / Folder | Description |
| :--- | :--- |
| `weekly_ranking_report.py` | Main product: weekly Top-20/Bottom-20 NIFTY-100 rankings with earnings-date flags and FII/DII flows; writes dated snapshots. |
| `build_rankings_index.py` | Builds the GitHub Pages site (index + archive of all weekly reports). |
| `daily_kronos_pipeline.py` | Daily pipeline orchestrating Kronos forecasting, market mood, and reporting. |
| `market_advisor.py` | Data retrieval, Kronos K-line forecasting, LLM translation, and advisory report generation. |
| `stock_analyzer.py` | High-speed concurrent technical screener generating multi-factor health scores and `stock_screener_dashboard.html`. |
| `pooled_model_v1.py` | Leakage-safe pooled cross-sectional model over NIFTY 100 (walk-forward, purged, baseline-aware). |
| `factor_ic_analysis.py` | Information-coefficient analysis of individual factors. |
| `composite_factor_strategy.py` | Liquidity + short-term reversal composite factor strategy. |
| `backtest_random20.py` | Random-20 NIFTY-100 backtest of the composite strategy from a start date. |
| `model_comparison.py` | Head-to-head comparison of all models/factors under identical conditions. |
| `enhanced_predictor.py` | Multi-model ensemble (Kronos + LightGBM + ExtraTrees) walk-forward validator with market context features. |
| `paper_trading.py` | Paper-trading engine tracking simulated positions against live prices. |
| `macro_utils.py` | Macro data utilities: USD/INR, India VIX, NIFTY 50. |
| `data/fii_dii.csv` | Manually updated weekly FII/DII net-flow data (source: NSDL). |
| `colab_run.ipynb` | On-demand Colab runner with optional push-back to GitHub. |
| `.github/workflows/weekly_ranking.yml` | Scheduled CI workflow: runs the weekly report and publishes to GitHub Pages. |
| `outputs/rankings/` | Dated weekly ranking snapshots (md/html/json) forming the paper-trading track record. |
| `marketing/` | YouTube launch package: video slides deck, script, upload pack, thumbnail. |
| `templates/report_template.html` | Jinja2 responsive HTML visualizer template featuring dynamic UI styling and standalone export features. |
| `kronos_lib/` | Core library cloned from the official Kronos repository providing autoregressive Transformer and hierarchical tokenization architectures. |

---

## 💡 5 Core Use Cases

1. **Weekly Stock Rankings**: A disciplined, reproducible Monday ranking of NIFTY-100 long and short candidates — no discretionary guesswork.
2. **Blue-Chip Swing Level Identification**: Automates the detection of entry and profit-booking targets for high-liquidity Indian Equities (e.g., Reliance Industries, HDFC Bank).
3. **Factor Research Education**: Learn how momentum, reversal, liquidity, and ML factors are validated (or invalidated) on real Indian market data.
4. **Jargon-Free Financial Literacy**: Converts complex quant output (standard deviation, sampling intervals, IC, hit rate) into simple explanations using everyday analogies.
5. **Pre-Market Objective Screening**: Enables disciplined traders to benchmark qualitative hypotheses against unbiased quantitative signals.

---

## 🔮 Roadmap

1. **Live Track Record Validation**: Paper-trade the weekly rankings for 10+ weeks, then compute live IC/hit-rate vs NIFTY from snapshots.
2. **Subscription Tier Launch**: After live IC > 0.03, open a paid tier for the weekly ranking product.
3. **Indian Economic Events Integration**: Injecting RBI monetary policy schedules and quarterly corporate earnings dates directly into model context.
4. **Interactive Charting with Lightweight Charts**: Transitioning standard visual generation to full real-time interactive HTML5 trading canvases.
5. **Automated Telegram Alert Bot**: Periodic scheduled execution delivering customized morning advisory notes directly to mobile messaging channels.

---

## 🏷️ Keywords & Tags

`quant-finance` `time-series-forecasting` `candlestick-prediction` `nifty50-forecasting` `indian-stock-market` `financial-foundation-model` `kronos-ai` `ollama` `llama3.2` `momentum-factor` `algorithmic-trading` `python-trading-bot`

---
*Built with simplicity, elegance, and honest quantitative research for the Indian financial ecosystem.*
