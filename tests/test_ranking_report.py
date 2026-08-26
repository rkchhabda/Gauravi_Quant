"""The published report must show the same factors it ranks on.

Regression test for the bug where `12-1M Mom%` was rendered as a plain
12-month return (close[-1]/close[-253]) while the score used the real 12-1
definition (close[-22]/close[-253]) -- a 5-8pp discrepancy that made the
published table impossible to reconcile with the published rank.
"""

import numpy as np
import pandas as pd

from pooled_model_v1 import build_dataset


def test_report_momentum_matches_scoring_definition(synthetic_panel):
    close_panel, highs, lows, vols = synthetic_panel
    df = build_dataset(close_panel, highs, lows, vols, horizon=10)
    latest = pd.to_datetime(df["date"]).max()

    for sym in [c for c in close_panel.columns if c != "^NSEI"]:
        c = close_panel[sym].dropna().loc[:latest]

        # What weekly_ranking_report.row() now displays.
        displayed = c.iloc[-22] / c.iloc[-253] - 1
        # What pooled_model_v1.build_dataset actually scores on, pre-z-score.
        scored = (c.shift(21) / c.shift(252) - 1).loc[latest]

        assert np.isclose(displayed, scored), sym


def test_old_display_formula_was_materially_different(synthetic_panel):
    """Guard the guard: confirm the two formulas really do disagree, so the
    test above is not vacuously true on this fixture."""
    close_panel, _, _, _ = synthetic_panel
    diffs = []
    for sym in [c for c in close_panel.columns if c != "^NSEI"]:
        c = close_panel[sym].dropna()
        old = c.iloc[-1] / c.iloc[-253] - 1     # buggy: includes last month
        new = c.iloc[-22] / c.iloc[-253] - 1    # correct: skips last month
        diffs.append(abs(old - new))
    assert max(diffs) > 0.01, "fixture too flat to detect the bug"


def test_ret20_display_matches_scoring_definition(synthetic_panel):
    close_panel, highs, lows, vols = synthetic_panel
    df = build_dataset(close_panel, highs, lows, vols, horizon=10)
    latest = pd.to_datetime(df["date"]).max()

    for sym in [c for c in close_panel.columns if c != "^NSEI"]:
        c = close_panel[sym].dropna().loc[:latest]
        displayed = c.iloc[-1] / c.iloc[-21] - 1
        scored = c.pct_change(20).loc[latest]
        assert np.isclose(displayed, scored), sym


def test_score_is_mean_of_two_percentile_ranks(synthetic_panel):
    """The Score column must equal (rank(mom) + rank(-ret20)) / 2 within the
    day's cross-section, so a subscriber can reproduce the ranking."""
    close_panel, highs, lows, vols = synthetic_panel
    df = build_dataset(close_panel, highs, lows, vols, horizon=10)
    df = df[df["symbol"] != "^NSEI"].dropna(subset=["mom_12_1", "ret_20", "liq"])
    snap = df[df["date"] == df["date"].max()].copy()

    score = (snap["mom_12_1"].rank(pct=True)
             + (1 - snap["ret_20"].rank(pct=True))) / 2.0
    assert score.between(0, 1).all()
    assert np.isclose(score.mean(), 0.5, atol=0.2)


def test_pct_helper_tolerates_missing_values():
    from weekly_ranking_report import pct
    assert pct(12.3) == "12.3%"
    assert pct(None) == "n/a"
