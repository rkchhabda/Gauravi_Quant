import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from jinja2 import Template

# Reconfigure stdout for UTF-8 support on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==============================================================================
# 1. DATA INGESTION ENGINE (FAST CONCURRENT NSE/BSE & GLOBAL TICKERS)
# ==============================================================================
class StockDataFetcher:
    """
    Robust, multithreaded data fetcher for Indian (NSE/BSE) and global equities.
    Parses tickers from file (supporting CSV format, comments, and bare symbols).
    """
    def __init__(self, ticker_file: str = "stocks.txt"):
        self.ticker_file = ticker_file

    def load_tickers(self) -> list[dict]:
        """
        Loads tickers and company names from file.
        Returns a list of dicts: [{'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries Ltd'}, ...]
        """
        fallback_tickers = [
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd"},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services Ltd"},
            {"symbol": "INFY.NS", "name": "Infosys Ltd"},
            {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd"},
            {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd"},
            {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd"},
            {"symbol": "ITC.NS", "name": "ITC Ltd"},
            {"symbol": "LT.NS", "name": "Larsen & Toubro Ltd"}
        ]

        if not os.path.exists(self.ticker_file):
            return fallback_tickers

        assets = []
        with open(self.ticker_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) >= 2:
                    raw_sym, name = parts[0], parts[1]
                else:
                    raw_sym, name = parts[0], parts[0]

                # Append .NS extension for NSE if missing, unless index/currency/special
                if raw_sym.startswith(("^", "INR=")) or raw_sym.endswith((".NS", ".BO", "=X")):
                    symbol = raw_sym
                else:
                    symbol = f"{raw_sym}.NS"

                assets.append({"symbol": symbol, "name": name})

        return assets if assets else fallback_tickers

    def fetch_historical(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        """
        Fetches historical daily OHLCV dataframe for a single ticker safely.
        """
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            if df.empty:
                return pd.DataFrame()
            
            # Handle MultiIndex columns from recent yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns.levels[1]:
                    df = df.xs(ticker, axis=1, level=1)
                else:
                    df.columns = df.columns.get_level_values(0)
            
            df = df.dropna(subset=['Close'])
            return df
        except Exception:
            return pd.DataFrame()

    def fetch_all_concurrent(self, assets: list[dict], period: str = "6mo", max_workers: int = 10) -> dict[str, pd.DataFrame]:
        """
        Fetches historical data for multiple tickers concurrently in parallel threads.
        """
        data_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {
                executor.submit(self.fetch_historical, asset["symbol"], period): asset["symbol"]
                for asset in assets
            }
            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    df = future.result()
                    if not df.empty and len(df) >= 30:
                        data_map[sym] = df
                except Exception:
                    pass
        return data_map


# ==============================================================================
# 2. ADVANCED FEATURE ENGINEERING & TECHNICAL INDICATORS
# ==============================================================================
class TechnicalAnalyzer:
    """
    Computes professional-grade technical indicators and performance metrics:
    - Moving Averages (SMA 20/50, EMA 20/50)
    - Wilder's Smoothing RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence) & Signal Line
    - Bollinger Bands & Volatility
    - Multi-timeframe Returns (1-Week, 1-Month, 6-Month Period Returns)
    - Volume Breakout Ratio
    """
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df['Close']
        
        # Moving Averages
        df['SMA_20'] = close.rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = close.rolling(window=50, min_periods=1).mean()
        df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
        
        # Wilder's Smoothing RSI (14 periods)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        # Prevent division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50.0) # Neutral fallback if indeterminate
        
        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands (20 periods, 2 std dev)
        rolling_std_20 = close.rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['SMA_20'] + (2 * rolling_std_20)
        df['BB_Lower'] = df['SMA_20'] - (2 * rolling_std_20)
        
        # Performance Returns (%)
        df['Return_1D'] = close.pct_change() * 100
        df['Return_1W'] = close.pct_change(periods=5) * 100
        df['Return_1M'] = close.pct_change(periods=21) * 100
        first_valid_close = close.dropna().iloc[0] if not close.dropna().empty else close.iloc[-1]
        df['Return_Period'] = ((close - first_valid_close) / first_valid_close) * 100
        
        # Volume Expansion Ratio (Current Volume / 20-day Volume SMA)
        if 'Volume' in df.columns and (df['Volume'] > 0).any():
            vol_sma = df['Volume'].rolling(window=20, min_periods=1).mean()
            df['Vol_Ratio'] = (df['Volume'] / vol_sma.replace(0, np.nan)).fillna(1.0)
        else:
            df['Vol_Ratio'] = 1.0
            
        # Annualized Volatility
        df['Volatility'] = df['Return_1D'].rolling(window=20, min_periods=5).std() * np.sqrt(252)
        df['Volatility'] = df['Volatility'].fillna(0.0)
        
        return df


# ==============================================================================
# 3. PREDICTIVE & MULTI-FACTOR ADVISORY SCORING ENGINE
# ==============================================================================
class MarketAdvisor:
    """
    Multi-Factor quantitative scoring and performance ranking engine.
    Calculates composite technical health scores (0-100) and identifies top performers.
    """
    def __init__(self, ticker_file: str = "stocks.txt"):
        self.fetcher = StockDataFetcher(ticker_file=ticker_file)

    def evaluate_stock(self, asset: dict, df: pd.DataFrame) -> dict:
        """
        Analyzes a single stock dataframe and computes a comprehensive health score and signal.
        """
        symbol = asset["symbol"]
        name = asset["name"]
        
        if df.empty or len(df) < 30:
            return {"symbol": symbol, "name": name, "status": "Insufficient Data"}
        
        df = TechnicalAnalyzer.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest
        
        close = float(latest['Close'])
        sma_20 = float(latest['SMA_20'])
        sma_50 = float(latest['SMA_50'])
        ema_20 = float(latest['EMA_20'])
        rsi = float(latest['RSI'])
        macd = float(latest['MACD'])
        macd_signal = float(latest['MACD_Signal'])
        ret_1w = float(latest['Return_1W']) if not np.isnan(latest['Return_1W']) else 0.0
        ret_1m = float(latest['Return_1M']) if not np.isnan(latest['Return_1M']) else 0.0
        ret_period = float(latest['Return_Period']) if not np.isnan(latest['Return_Period']) else 0.0
        vol_ratio = float(latest['Vol_Ratio'])
        volatility = float(latest['Volatility'])
        
        # -------------------------------------------------------------
        # COMPOSITE QUANTITATIVE SCORING SYSTEM (Total: 0 to 100 pts)
        # -------------------------------------------------------------
        score = 0.0
        
        # 1. Trend Alignment (Max 35 points)
        if close > sma_20:
            score += 12.0
        if sma_20 > sma_50:
            score += 13.0  # Bullish moving average alignment
        if close > ema_20:
            score += 10.0  # Immediate short-term momentum
            
        # 2. Momentum & RSI (Max 25 points)
        if 50.0 <= rsi <= 68.0:
            score += 20.0  # Prime sweet spot for healthy bullish momentum
        elif 68.0 < rsi <= 75.0:
            score += 15.0  # Strong momentum, near upper band
        elif 40.0 <= rsi < 50.0:
            score += 10.0  # Consolidation / mild recovery
        elif rsi < 30.0:
            score += 12.0  # Oversold bounce opportunity
        else:
            score += 5.0   # Extreme overbought (>75) or deep weakness (<40)
            
        # MACD Crossover Bonus
        if macd > macd_signal:
            score += 5.0
            
        # 3. Performance & Price Velocity (Max 25 points)
        if ret_1m > 10.0:
            score += 15.0
        elif ret_1m > 5.0:
            score += 12.0
        elif ret_1m > 0.0:
            score += 8.0
        else:
            score += max(0.0, 8.0 + (ret_1m * 0.5))
            
        if ret_period > 15.0:
            score += 10.0
        elif ret_period > 5.0:
            score += 7.0
        elif ret_period > 0.0:
            score += 4.0
            
        # 4. Volume Confirmation (Max 15 points)
        if vol_ratio >= 1.5 and close > prev['Close']:
            score += 15.0  # High volume breakout confirmation
        elif vol_ratio >= 1.0:
            score += 10.0  # Normal to healthy volume
        else:
            score += 5.0   # Low volume
            
        score = round(min(max(score, 5.0), 100.0), 1)
        
        # -------------------------------------------------------------
        # SIGNAL GENERATION CLASSIFICATION
        # -------------------------------------------------------------
        if score >= 80:
            signal = "STRONG BUY (Top Performer)"
        elif score >= 65:
            signal = "BULLISH (Uptrend Momentum)"
        elif score >= 50:
            signal = "NEUTRAL / ACCUMULATE"
        elif score >= 35:
            signal = "CAUTION / CONSOLIDATION"
        else:
            signal = "BEARISH (Downtrend / Exit)"

        # Target levels
        support = round(float(df['Low'].tail(20).min()), 2)
        resistance = round(float(df['High'].tail(20).max()), 2)

        return {
            "symbol": symbol,
            "name": name,
            "close_price": round(close, 2),
            "support": support,
            "resistance": resistance,
            "rsi": round(rsi, 2),
            "sma_20": round(sma_20, 2),
            "sma_50": round(sma_50, 2),
            "return_1w": round(ret_1w, 2),
            "return_1m": round(ret_1m, 2),
            "return_period": round(ret_period, 2),
            "vol_ratio": round(vol_ratio, 2),
            "volatility": round(volatility, 2),
            "signal": signal,
            "score": score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def run_pipeline(self, period: str = "6mo") -> list[dict]:
        """
        Runs the full ingestion and analysis pipeline across all loaded tickers.
        """
        assets = self.fetcher.load_tickers()
        print(f"[INGEST] Loaded {len(assets)} target assets. Fetching market data concurrently...")
        
        data_map = self.fetcher.fetch_all_concurrent(assets, period=period)
        print(f"[ANALYZE] Successfully fetched {len(data_map)}/{len(assets)} stocks. Evaluating technicals...")
        
        results = []
        for asset in assets:
            sym = asset["symbol"]
            df = data_map.get(sym, pd.DataFrame())
            evaluated = self.evaluate_stock(asset, df)
            results.append(evaluated)
            
        return results

    @staticmethod
    def get_top_performers(results: list[dict], top_n: int = 5) -> list[dict]:
        """
        Identifies and ranks the Top N performing stocks based on composite score & returns.
        """
        valid_results = [r for r in results if "status" not in r]
        # Multi-factor ranking: Primary sort by Technical Score, secondary by 1-Month Return
        ranked = sorted(
            valid_results,
            key=lambda x: (x["score"], x["return_1m"], x["return_period"]),
            reverse=True
        )
        top_stocks = ranked[:top_n]
        for rank, stock in enumerate(top_stocks, start=1):
            stock["rank"] = rank
        return top_stocks


# ==============================================================================
# 4. REPORT GENERATION ENGINE (PREMIUM HTML + MARKDOWN)
# ==============================================================================
class ReportGenerator:
    """
    Renders an executive-grade dashboard report in HTML with Top 5 Performers highlight,
    market statistics, and interactive filtering.
    """
    @staticmethod
    def export_html(results: list[dict], top_performers: list[dict], output_file: str = "stock_screener_dashboard.html"):
        valid_results = [r for r in results if "status" not in r]
        
        # Calculate summary statistics
        total_analyzed = len(valid_results)
        bullish_count = sum(1 for r in valid_results if "BUY" in r["signal"] or "BULLISH" in r["signal"])
        bearish_count = sum(1 for r in valid_results if "BEARISH" in r["signal"] or "CAUTION" in r["signal"])
        neutral_count = total_analyzed - bullish_count - bearish_count
        avg_score = round(np.mean([r["score"] for r in valid_results]), 1) if valid_results else 0
        avg_1m_return = round(np.mean([r["return_1m"] for r in valid_results]), 2) if valid_results else 0
        top_gainer = max(valid_results, key=lambda x: x["return_1m"]) if valid_results else None

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indian Stock Market AI Advisor | Top 5 Performers</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0e17;
            --bg-surface: #111827;
            --bg-card: rgba(17, 24, 39, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-green: #10b981;
            --accent-green-glow: rgba(16, 185, 129, 0.25);
            --accent-red: #f43f5e;
            --accent-amber: #f59e0b;
            --accent-blue: #3b82f6;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            line-height: 1.5;
            padding: 2.5rem 1.5rem;
            min-height: 100vh;
        }

        .container {
            max-width: 1300px;
            margin: 0 auto;
        }

        /* HEADER */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 1.5rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: var(--text-primary);
        }

        .header h1 span {
            color: var(--accent-green);
        }

        .header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }

        .badge-live {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            padding: 0.4rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .dot-live {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
        }

        /* SUMMARY STATS BAR */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2.5rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            padding: 1.25rem;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .stat-label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }

        .stat-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-value.bullish { color: var(--accent-green); }
        .stat-value.bearish { color: var(--accent-red); }
        .stat-value.amber { color: var(--accent-amber); }

        /* TOP 5 PERFORMERS SECTION */
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .top-performers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.25rem;
            margin-bottom: 3rem;
        }

        .top-card {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.4rem;
            position: relative;
            transition: all 0.2s ease;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }

        .top-card:hover {
            transform: translateY(-3px);
            border-color: rgba(16, 185, 129, 0.4);
            box-shadow: 0 15px 30px -10px var(--accent-green-glow);
        }

        .rank-medal {
            position: absolute;
            top: 1rem;
            right: 1rem;
            font-size: 1.25rem;
            font-weight: 800;
            background: rgba(255, 255, 255, 0.06);
            width: 34px;
            height: 34px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .top-card-ticker {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .top-card-name {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .top-metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.35rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.85rem;
        }

        .top-metric-label {
            color: var(--text-secondary);
        }

        .top-metric-val {
            font-weight: 600;
        }

        .score-pill {
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
            border: 1px solid rgba(16, 185, 129, 0.3);
            margin-top: 0.8rem;
            text-align: center;
            width: 100%;
        }

        /* TABLE SECTION */
        .table-container {
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .table-toolbar {
            padding: 1.25rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 1rem;
        }

        .table-title {
            font-size: 1.15rem;
            font-weight: 700;
        }

        .search-input {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            outline: none;
            width: 250px;
        }

        .search-input:focus {
            border-color: var(--accent-blue);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        th {
            background: rgba(0, 0, 0, 0.3);
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
        }

        td {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 0.9rem;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .stock-cell {
            display: flex;
            flex-direction: column;
        }

        .stock-sym {
            font-weight: 700;
            color: var(--text-primary);
        }

        .stock-desc {
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .badge-sig {
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
        }

        .sig-strong-buy { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
        .sig-bullish { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #2563eb; }
        .sig-neutral { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
        .sig-bearish { background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid #e11d48; }

        .progress-bar-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            width: 100%;
        }

        .progress-bar-bg {
            flex-grow: 1;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 9999px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            border-radius: 9999px;
        }

        .positive { color: var(--accent-green); font-weight: 600; }
        .negative { color: var(--accent-red); font-weight: 600; }

        footer {
            margin-top: 3rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 1.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <header class="header">
            <div>
                <h1>Indian Market <span>AI Advisor</span></h1>
                <p>Advanced Quantitative Screening, Performance Scoring & Trend Analytics</p>
            </div>
            <div class="badge-live">
                <span class="dot-live"></span>
                Analysis Time: {{ timestamp }}
            </div>
        </header>

        <!-- STATS OVERVIEW -->
        <section class="stats-grid">
            <div class="stat-card">
                <span class="stat-label">Total Analyzed</span>
                <span class="stat-value">{{ total_analyzed }} Stocks</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Bullish Momentum</span>
                <span class="stat-value bullish">{{ bullish_count }}</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Bearish / Caution</span>
                <span class="stat-value bearish">{{ bearish_count }}</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Market Avg 1M Return</span>
                <span class="stat-value {% if avg_1m_return >= 0 %}bullish{% else %}bearish{% endif %}">
                    {% if avg_1m_return >= 0 %}+{% endif %}{{ avg_1m_return }}%
                </span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Avg Health Score</span>
                <span class="stat-value amber">{{ avg_score }}/100</span>
            </div>
        </section>

        <!-- TOP 5 PERFORMING STOCKS PODIUM -->
        <div class="section-header">
            <h2 class="section-title">🏆 Top 5 Performing Stocks</h2>
        </div>
        <div class="top-performers-grid">
            {% for stock in top_performers %}
            <div class="top-card">
                <span class="rank-medal">
                    {% if stock.rank == 1 %}🥇{% elif stock.rank == 2 %}🥈{% elif stock.rank == 3 %}🥉{% else %}#{{ stock.rank }}{% endif %}
                </span>
                <div class="top-card-ticker">{{ stock.symbol }}</div>
                <div class="top-card-name" title="{{ stock.name }}">{{ stock.name }}</div>

                <div class="top-metric-row">
                    <span class="top-metric-label">Close Price</span>
                    <span class="top-metric-val">₹{{ stock.close_price }}</span>
                </div>
                <div class="top-metric-row">
                    <span class="top-metric-label">1-Month Return</span>
                    <span class="top-metric-val {% if stock.return_1m >= 0 %}positive{% else %}negative{% endif %}">
                        {% if stock.return_1m >= 0 %}+{% endif %}{{ stock.return_1m }}%
                    </span>
                </div>
                <div class="top-metric-row">
                    <span class="top-metric-label">Period Return</span>
                    <span class="top-metric-val {% if stock.return_period >= 0 %}positive{% else %}negative{% endif %}">
                        {% if stock.return_period >= 0 %}+{% endif %}{{ stock.return_period }}%
                    </span>
                </div>
                <div class="top-metric-row">
                    <span class="top-metric-label">RSI (14)</span>
                    <span class="top-metric-val">{{ stock.rsi }}</span>
                </div>
                
                <div class="score-pill">
                    Score: {{ stock.score }}/100 • {{ stock.signal }}
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- FULL MARKET SCREENER TABLE -->
        <div class="table-container">
            <div class="table-toolbar">
                <span class="table-title">Full Market Screener & Indicator Analysis</span>
                <input type="text" id="searchInput" class="search-input" placeholder="Filter by ticker or name..." onkeyup="filterTable()">
            </div>
            <table id="screenerTable">
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Close (₹)</th>
                        <th>1W Return</th>
                        <th>1M Return</th>
                        <th>Period Return</th>
                        <th>RSI (14)</th>
                        <th>20 SMA</th>
                        <th>50 SMA</th>
                        <th>AI Signal</th>
                        <th>Quant Score</th>
                    </tr>
                </thead>
                <tbody>
                    {% for r in results %}
                    {% if not r.status %}
                    <tr>
                        <td>
                            <div class="stock-cell">
                                <span class="stock-sym">{{ r.symbol }}</span>
                                <span class="stock-desc">{{ r.name }}</span>
                            </div>
                        </td>
                        <td><strong>₹{{ r.close_price }}</strong></td>
                        <td class="{% if r.return_1w >= 0 %}positive{% else %}negative{% endif %}">
                            {% if r.return_1w >= 0 %}+{% endif %}{{ r.return_1w }}%
                        </td>
                        <td class="{% if r.return_1m >= 0 %}positive{% else %}negative{% endif %}">
                            {% if r.return_1m >= 0 %}+{% endif %}{{ r.return_1m }}%
                        </td>
                        <td class="{% if r.return_period >= 0 %}positive{% else %}negative{% endif %}">
                            {% if r.return_period >= 0 %}+{% endif %}{{ r.return_period }}%
                        </td>
                        <td>{{ r.rsi }}</td>
                        <td>₹{{ r.sma_20 }}</td>
                        <td>₹{{ r.sma_50 }}</td>
                        <td>
                            <span class="badge-sig {% if 'STRONG' in r.signal %}sig-strong-buy{% elif 'BULLISH' in r.signal %}sig-bullish{% elif 'BEARISH' in r.signal %}sig-bearish{% else %}sig-neutral{% endif %}">
                                {{ r.signal }}
                            </span>
                        </td>
                        <td>
                            <div class="progress-bar-container">
                                <span>{{ r.score }}</span>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill" style="width: {{ r.score }}%; background-color: {% if r.score >= 75 %}#10b981{% elif r.score >= 50 %}#3b82f6{% elif r.score >= 35 %}#f59e0b{% else %}#f43f5e{% endif %};"></div>
                                </div>
                            </div>
                        </td>
                    </tr>
                    {% endif %}
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <footer>
            Indian Stock Market AI Screener • Quantitative Multi-Factor Performance Ranking • Generated for educational and analytical purposes.
        </footer>
    </div>

    <script>
        function filterTable() {
            const input = document.getElementById("searchInput");
            const filter = input.value.toUpperCase();
            const table = document.getElementById("screenerTable");
            const tr = table.getElementsByTagName("tr");

            for (let i = 1; i < tr.length; i++) {
                const tdSym = tr[i].getElementsByTagName("td")[0];
                if (tdSym) {
                    const txtValue = tdSym.textContent || tdSym.innerText;
                    if (txtValue.toUpperCase().indexOf(filter) > -1) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }
            }
        }
    </script>
</body>
</html>
        """
        template = Template(html_template)
        rendered_html = template.render(
            results=valid_results,
            top_performers=top_performers,
            total_analyzed=total_analyzed,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            avg_score=avg_score,
            avg_1m_return=avg_1m_return,
            top_gainer=top_gainer,
            timestamp=datetime.now().strftime("%d %b %Y, %H:%M:%S IST")
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        print(f"\n[REPORT] Visual dashboard successfully saved to: {output_file}")

    @staticmethod
    def print_top_performers_console(top_performers: list[dict]):
        """
        Prints a clean formatted leaderboard table to the terminal stdout.
        """
        print("\n" + "=" * 90)
        print(f"{'TOP 5 PERFORMING STOCKS LEADERBOARD':^90}")
        print("=" * 90)
        header = f"{'Rank':<8}{'Symbol':<16}{'Price (INR)':<14}{'1M Return':<13}{'6M Return':<13}{'RSI(14)':<10}{'Score':<10}{'AI Signal'}"
        print(header)
        print("-" * 90)
        medals = {1: "[#1 🥇]", 2: "[#2 🥈]", 3: "[#3 🥉]", 4: "[#4   ]", 5: "[#5   ]"}
        for stock in top_performers:
            rank_str = medals.get(stock['rank'], f"[#{stock['rank']}   ]")
            sym = stock['symbol']
            price = f"Rs. {stock['close_price']:,.2f}"
            ret_1m = f"{'+' if stock['return_1m'] >= 0 else ''}{stock['return_1m']:.2f}%"
            ret_per = f"{'+' if stock['return_period'] >= 0 else ''}{stock['return_period']:.2f}%"
            rsi = f"{stock['rsi']:.1f}"
            score = f"{stock['score']:.1f}/100"
            sig = stock['signal']
            print(f"{rank_str:<8}{sym:<16}{price:<14}{ret_1m:<13}{ret_per:<13}{rsi:<10}{score:<10}{sig}")
        print("=" * 90 + "\n")


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    advisor = MarketAdvisor(ticker_file="stocks.txt")
    
    # 1. Run Quantitative Ingestion and Indicator Pipeline
    analysis_results = advisor.run_pipeline(period="6mo")
    
    # 2. Extract Top 5 Performing Stocks
    top_5_stocks = advisor.get_top_performers(analysis_results, top_n=5)
    
    # 3. Print Results in Terminal
    ReportGenerator.print_top_performers_console(top_5_stocks)
    
    # 4. Generate Interactive HTML Dashboard
    ReportGenerator.export_html(analysis_results, top_5_stocks, output_file="stock_screener_dashboard.html")
