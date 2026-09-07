#!/usr/bin/env python3
"""
FastAPI server for Gauravi Trading System dashboards.
Serves HTML dashboards and provides live data APIs.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from stock_analyzer import MarketAdvisor, ReportGenerator
from daily_scheduler import run_daily_predictions, PRODUCTION_STOCKS
from paper_trading import PaperTradingSimulator

app = FastAPI(title="Gauravi Trading System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DASHBOARDS_DIR = BASE_DIR / "dashboards"

# Cache for analysis results
_analysis_cache = {"data": None, "timestamp": None, "top_performers": None}
_daily_predictions_cache = {"data": None, "timestamp": None}
_paper_trading_cache = {"data": None, "timestamp": None}


def get_latest_ranking_snapshot():
    """Get the latest ranking snapshot from outputs/rankings/"""
    ranking_dir = OUTPUTS_DIR / "rankings"
    if not ranking_dir.exists():
        return None
    
    snapshots = list(ranking_dir.glob("snapshot_*.json"))
    if not snapshots:
        return None
    
    latest = max(snapshots, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)


def get_latest_daily_predictions():
    """Get the latest daily predictions from outputs/daily_predictions/"""
    pred_dir = OUTPUTS_DIR / "daily_predictions"
    if not pred_dir.exists():
        return None
    
    preds = list(pred_dir.glob("daily_predictions_*.json"))
    if not preds:
        return None
    
    latest = max(preds, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        return json.load(f)


@app.get("/")
async def root():
    return {"message": "Gauravi Trading System API", "docs": "/docs", "dashboards": ["/dashboard", "/live"]}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the stock screener dashboard"""
    dashboard_path = BASE_DIR / "stock_screener_dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return HTMLResponse("<h1>Dashboard not found. Run stock_analyzer.py first.</h1>")


@app.get("/live", response_class=HTMLResponse)
async def live_dashboard():
    """Serve the live trading dashboard"""
    dashboard_path = BASE_DIR / "live_dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return HTMLResponse("<h1>Live dashboard not found.</h1>")


@app.get("/api/rankings/latest")
async def api_rankings_latest():
    """Get latest ranking snapshot"""
    data = get_latest_ranking_snapshot()
    if data is None:
        raise HTTPException(status_code=404, detail="No ranking snapshot found")
    return data


@app.get("/api/predictions/daily")
async def api_daily_predictions():
    """Get latest daily predictions"""
    data = get_latest_daily_predictions()
    if data is None:
        raise HTTPException(status_code=404, detail="No daily predictions found")
    return data


@app.get("/api/predictions/daily/{symbol}")
async def api_daily_prediction_symbol(symbol: str):
    """Get daily prediction for specific symbol"""
    data = get_latest_daily_predictions()
    if data is None:
        raise HTTPException(status_code=404, detail="No daily predictions found")
    
    for pred in data:
        if pred.get("symbol") == symbol:
            return pred
    raise HTTPException(status_code=404, detail=f"No prediction for {symbol}")


@app.post("/api/analysis/run")
async def api_run_analysis():
    """Run full analysis pipeline and regenerate dashboard"""
    global _analysis_cache
    
    try:
        advisor = MarketAdvisor(ticker_file="stocks.txt")
        results = advisor.run_pipeline(period="6mo")
        top_performers = advisor.get_top_performers(results, top_n=5)
        
        ReportGenerator.export_html(results, top_performers, output_file="stock_screener_dashboard.html")
        
        _analysis_cache = {
            "data": results,
            "top_performers": top_performers,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "analyzed": len(results),
            "top_performers": len(top_performers),
            "timestamp": _analysis_cache["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/latest")
async def api_analysis_latest():
    """Get latest analysis results (from cache or run if empty)"""
    global _analysis_cache
    
    if _analysis_cache["data"] is None:
        await api_run_analysis()
    
    return {
        "results": _analysis_cache["data"],
        "top_performers": _analysis_cache["top_performers"],
        "timestamp": _analysis_cache["timestamp"]
    }


@app.get("/api/stocks/list")
async def api_stocks_list():
    """Get list of available stocks"""
    ticker_file = BASE_DIR / "stocks.txt"
    if not ticker_file.exists():
        return {"stocks": list(PRODUCTION_STOCKS.keys())}
    
    with open(ticker_file) as f:
        stocks = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    return {"stocks": stocks}


@app.get("/api/market/summary")
async def api_market_summary():
    """Get market summary statistics"""
    data = get_latest_ranking_snapshot()
    if data is None:
        return {"error": "No data available"}
    
    top = data.get("top", [])
    bottom = data.get("bottom", [])
    
    return {
        "as_of": data.get("as_of"),
        "universe_size": data.get("universe_size"),
        "top_5": top[:5],
        "bottom_5": bottom[:5],
        "fii_dii": data.get("fii_dii"),
        "generated_at": data.get("generated_at")
    }


@app.get("/api/paper-trading/status")
async def api_paper_trading_status():
    """Get paper trading simulator status"""
    global _paper_trading_cache
    
    if _paper_trading_cache["data"] is None or \
       (datetime.now() - datetime.fromisoformat(_paper_trading_cache["timestamp"])).seconds > 60:
        sim = PaperTradingSimulator()
        sim.check_positions()
        report = sim.get_performance_report()
        _paper_trading_cache = {
            "data": report,
            "timestamp": datetime.now().isoformat()
        }
    
    return _paper_trading_cache["data"]


@app.post("/api/paper-trading/check")
async def api_paper_trading_check():
    """Check paper trading positions for SL/TP hits"""
    sim = PaperTradingSimulator()
    sim.check_positions()
    report = sim.get_performance_report()
    return report


@app.post("/api/daily-predictions/run")
async def api_run_daily_predictions(paper_trade: bool = False):
    """Run daily predictions for all production stocks"""
    try:
        results = run_daily_predictions(paper_trade=paper_trade)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)