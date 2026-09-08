FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for better caching)
COPY requirements-render.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-render.txt

# Copy application code
COPY server.py .
COPY stock_analyzer.py .
COPY daily_scheduler.py .
COPY paper_trading.py .
COPY macro_utils.py .
COPY alerts/ ./alerts/
COPY stocks.txt .

EXPOSE 8000

CMD ["python", "server.py"]