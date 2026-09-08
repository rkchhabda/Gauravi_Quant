"""gauravi_data -- thin loader for the Gauravi NSE factor dataset.

    import gauravi_data as gd

    panel   = gd.load_panel()          # adjusted OHLC + volume + delivery %
    fund    = gd.load_fundamentals()   # point-in-time annual fundamentals
    factors = gd.load_factors()        # factor values + percentile ranks
    print(gd.manifest()["version"], gd.universe())

By default the loaders look for the dataset in, in order:
  1. the path passed as `data_dir=`
  2. the GAURAVI_DATA_DIR environment variable
  3. ./data, then the newest dist/<version>/ directory next to this package

Every loader returns a pandas DataFrame. Nothing here hits the network -- the
dataset is a static, versioned download.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd

__version__ = "1.0.0"

PANEL_FILE = "panel_nifty100.parquet"
FUND_FILE = "fundamentals_pit.parquet"
FACTOR_FILE = "factors.parquet"
UNIVERSE_FILE = "universe.csv"
MANIFEST_FILE = "MANIFEST.json"

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_ROOT)


class DatasetNotFound(FileNotFoundError):
    """Raised when the dataset directory or a required file is missing."""


# --------------------------------------------------------------------------
# locating the dataset
# --------------------------------------------------------------------------
def _candidate_dirs():
    env = os.environ.get("GAURAVI_DATA_DIR")
    if env:
        yield env
    for base in (os.getcwd(), _REPO_ROOT, _PKG_ROOT):
        # The release zip puts the parquet files at its root next to the loader
        # package, so `base` itself is a valid location -- not only base/data.
        yield base
        yield os.path.join(base, "data")
        dist = os.path.join(base, "dist")
        if os.path.isdir(dist):
            subs = [d for d in os.listdir(dist)
                    if os.path.isdir(os.path.join(dist, d))]
            # newest version first, but a free "-sample" release never wins over
            # a full one -- otherwise a leftover sample dir shadows the real data.
            subs.sort(key=lambda d: ("sample" in d.lower(),
                                     [-ord(ch) for ch in d]))
            for sub in subs:
                yield os.path.join(dist, sub)


def data_dir(path: Optional[str] = None) -> str:
    """Resolve the dataset directory. Raises DatasetNotFound if none has a panel."""
    if path:
        if not os.path.isfile(os.path.join(path, PANEL_FILE)):
            raise DatasetNotFound(f"{PANEL_FILE} not found in {path!r}")
        return path
    tried = []
    for cand in _candidate_dirs():
        tried.append(cand)
        if os.path.isfile(os.path.join(cand, PANEL_FILE)):
            return cand
    raise DatasetNotFound(
        "Could not locate the Gauravi dataset. Set GAURAVI_DATA_DIR or pass "
        f"data_dir=. Looked in: {tried}")


def _read(fname, path, parse_dates=()):
    d = data_dir(path)
    full = os.path.join(d, fname)
    if not os.path.isfile(full):
        raise DatasetNotFound(f"{fname} missing from {d!r}")
    df = (pd.read_parquet(full) if fname.endswith(".parquet")
          else pd.read_csv(full))
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_panel(data_dir=None, symbols=None, start=None, end=None):
    """Daily adjusted OHLC + volume + delivery %. Columns: date, sym, close,
    high, low, volume, deliv_pct."""
    df = _read(PANEL_FILE, data_dir, parse_dates=("date",))
    return _slice(df, symbols, start, end)


def load_factors(data_dir=None, symbols=None, start=None, end=None,
                 columns=None):
    """Factor values, `rank_*` percentile ranks, composites and `fwd_excess`.

    `fwd_excess` is the forward-looking research target -- never use it as a
    feature.
    """
    df = _read(FACTOR_FILE, data_dir, parse_dates=("date",))
    if columns is not None:
        keep = ["date", "sym"] + [c for c in columns if c not in ("date", "sym")]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(f"unknown factor columns: {missing}")
        df = df[keep]
    return _slice(df, symbols, start, end)


def load_fundamentals(data_dir=None, symbols=None):
    """Point-in-time annual fundamentals. Join on `avail_date` (= fy_end + 90d),
    never on `fy_end`, or you introduce lookahead."""
    df = _read(FUND_FILE, data_dir, parse_dates=("fy_end", "avail_date"))
    if symbols is not None:
        df = df[df["sym"].isin(_as_list(symbols))]
    return df.reset_index(drop=True)


def universe(data_dir=None):
    """List of symbols in the release."""
    return _read(UNIVERSE_FILE, data_dir, parse_dates=("as_of",))["symbol"].tolist()


def manifest(path=None) -> dict:
    """Release metadata: version, row counts, checksums, validation results."""
    d = data_dir(path)
    full = os.path.join(d, MANIFEST_FILE)
    if not os.path.isfile(full):
        raise DatasetNotFound(f"{MANIFEST_FILE} missing from {d!r}")
    with open(full, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_list(x):
    return [x] if isinstance(x, str) else list(x)


def _slice(df, symbols, start, end):
    if symbols is not None:
        df = df[df["sym"].isin(_as_list(symbols))]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def wide(df, value, index="date", columns="sym"):
    """Long -> wide pivot, e.g. wide(load_panel(), "close")."""
    return df.pivot(index=index, columns=columns, values=value).sort_index()


def factor_ic(factors, factor, target="fwd_excess", method="spearman"):
    """Mean cross-sectional Information Coefficient of `factor` vs `target`.

    Returns (mean_ic, n_dates). Note: consecutive dates overlap, so a naive
    t-stat on daily ICs is inflated roughly sqrt(horizon)x -- subsample to
    non-overlapping dates before testing significance.
    """
    sub = factors[["date", factor, target]].dropna()
    per_date = sub.groupby("date").apply(
        lambda g: g[factor].corr(g[target], method=method)
        if len(g) > 5 else float("nan"))
    per_date = per_date.dropna()
    return float(per_date.mean()), int(len(per_date))


__all__ = ["load_panel", "load_factors", "load_fundamentals", "universe",
           "manifest", "data_dir", "wide", "factor_ic", "DatasetNotFound",
           "__version__"]
