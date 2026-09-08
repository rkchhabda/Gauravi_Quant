#!/bin/bash
# Build script for Render - install pydantic first with no-deps to avoid Rust compilation

# Install pydantic 2.9.2 first (pre-built wheel, no Rust)
pip install --no-deps pydantic==2.9.2

# Install core dependencies
pip install pandas==2.2.2 numpy==1.26.4 yfinance==0.2.54 scikit-learn==1.5.0 xgboost==2.1.1 scipy==1.13.0 requests==2.32.3 lxml==5.2.2

# Install fastapi and uvicorn (will use already installed pydantic)
pip install fastapi==0.110.1 uvicorn==0.29.0

# Install test deps
pip install pytest==8.3.2