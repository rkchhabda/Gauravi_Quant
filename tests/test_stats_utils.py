"""Calibration tests for stats_utils.

The whole t-statistic correction in accuracy_log.md rests on one claim: that a
naive t-test on daily observations of an H-day forward return rejects a true
null far more often than its nominal rate, and that the non-overlapping test
does not. These tests verify that by simulation rather than asserting it in a
docstring.

The noise generator builds x_t = sum(eps_{t+1..t+H}) from iid eps -- exactly the
overlap structure of a daily-sampled H-day forward return, with a true mean of
zero. Any rejection is a false positive by construction.
"""

import numpy as np
import pytest

from stats_utils import format_tstats, mean_tstats, newey_west_se

H = 10


def overlapping_noise(n, horizon, rng):
    """Daily series of `horizon`-day forward sums of iid noise. True mean = 0."""
    eps = rng.normal(0.0, 1.0, n + horizon)
    csum = np.concatenate([[0.0], np.cumsum(eps)])
    return csum[horizon:horizon + n] - csum[:n]


def false_positive_rates(n_sims=400, n=500, horizon=H, seed=7):
    """Share of pure-noise samples each t-stat calls significant at 5%."""
    rng = np.random.default_rng(seed)
    rej = {"t_naive": 0, "t_hac": 0, "t_nonoverlap": 0}
    for _ in range(n_sims):
        st = mean_tstats(overlapping_noise(n, horizon, rng), horizon=horizon)
        for k in rej:
            if not np.isnan(st[k]) and abs(st[k]) >= 1.96:
                rej[k] += 1
    return {k: v / n_sims for k, v in rej.items()}


@pytest.fixture(scope="module")
def fpr():
    return false_positive_rates()


def test_naive_tstat_massively_over_rejects(fpr):
    """The statistic that produced 't=4.38' is wrong far more often than 5%."""
    assert fpr["t_naive"] > 0.35, (
        f"naive FPR {fpr['t_naive']:.1%} -- expected the large inflation that "
        f"justifies the retraction")


def test_nonoverlap_tstat_is_calibrated(fpr):
    """The test we now judge on holds its nominal 5% size."""
    assert 0.02 <= fpr["t_nonoverlap"] <= 0.09, (
        f"non-overlap FPR {fpr['t_nonoverlap']:.1%} should be ~5%")


def test_hac_is_better_than_naive_but_still_liberal(fpr):
    """Newey-West helps a lot but is not exact -- hence 'cross-check only'."""
    assert fpr["t_hac"] < fpr["t_naive"] / 3
    assert fpr["t_hac"] >= fpr["t_nonoverlap"] - 0.02


def test_naive_inflation_is_around_sqrt_horizon():
    """accuracy_log.md claims ~3.2x (sqrt(10)); check the order of magnitude."""
    rng = np.random.default_rng(11)
    ratios = [mean_tstats(overlapping_noise(500, H, rng), horizon=H)["inflation"]
              for _ in range(200)]
    median = float(np.nanmedian(ratios))
    assert 2.0 <= median <= 5.0, f"median naive/HAC inflation {median:.2f}"


def test_retracted_tstat_falls_below_significance():
    """t=4.38 divided by the measured inflation must land under 1.96.

    This is the concrete arithmetic behind 'corrected t ~ 1.4' in accuracy_log.md.
    """
    rng = np.random.default_rng(3)
    inflation = float(np.nanmedian(
        [mean_tstats(overlapping_noise(500, H, rng), horizon=H)["inflation"]
         for _ in range(200)]))
    assert 4.38 / inflation < 1.96


def test_newey_west_matches_naive_se_on_iid_data():
    """With no autocorrelation there is nothing to correct, so the two agree."""
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 4000)
    naive = x.std(ddof=1) / np.sqrt(x.size)
    assert newey_west_se(x, lags=5) == pytest.approx(naive, rel=0.15)


def test_newey_west_se_exceeds_naive_under_positive_autocorrelation():
    rng = np.random.default_rng(9)
    x = overlapping_noise(2000, H, rng)
    naive = x.std(ddof=1) / np.sqrt(x.size)
    assert newey_west_se(x, lags=2 * H - 1) > naive * 1.5


def test_effective_sample_size_divides_by_horizon():
    st = mean_tstats(np.arange(500.0), horizon=H)
    assert st["n"] == 500
    assert st["n_effective"] == pytest.approx(50.0)


def test_nans_are_dropped_not_propagated():
    x = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    st = mean_tstats(x, horizon=2)
    assert st["n"] == 4
    assert st["mean"] == pytest.approx(2.5)
    assert not np.isnan(st["t_naive"])


def test_degenerate_inputs_do_not_raise():
    for x in ([], [1.0], [2.0, 2.0], [0.0] * 50):
        st = mean_tstats(np.asarray(x, dtype=float), horizon=H)
        assert np.isnan(st["t_naive"]) or st["t_naive"] == 0.0
        assert newey_west_se(np.asarray(x, dtype=float)) is not None


def test_format_tstats_reports_not_significant_for_noise():
    rng = np.random.default_rng(13)
    st = mean_tstats(overlapping_noise(500, H, rng), horizon=H)
    line = format_tstats("noise", st, horizon=H)
    assert "NOT significant" in line or "marginal" in line
    assert "inflated" in line
