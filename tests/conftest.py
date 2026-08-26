"""Shared fixtures. Everything here is offline -- no yfinance, no model downloads."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def synthetic_panel():
    """Deterministic close/high/low/volume panel for 6 symbols, 700 sessions.

    Shaped like `pooled_model_v1.fetch_panel` output so feature code can be
    exercised without touching the network.
    """
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2022-01-03", periods=700)
    syms = ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS", "EEE.NS", "FFF.NS"]

    closes, highs, lows, vols = {}, {}, {}, {}
    for i, s in enumerate(syms):
        drift = 0.0004 * (i - 2.5)
        steps = rng.normal(drift, 0.012, len(idx))
        c = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=idx)
        closes[s] = c
        highs[s] = c * (1 + rng.uniform(0.001, 0.01, len(idx)))
        lows[s] = c * (1 - rng.uniform(0.001, 0.01, len(idx)))
        vols[s] = pd.Series(rng.integers(1e5, 5e6, len(idx)).astype(float), index=idx)

    bench = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, len(idx)))), index=idx)
    closes["^NSEI"] = bench

    return pd.DataFrame(closes), highs, lows, vols
