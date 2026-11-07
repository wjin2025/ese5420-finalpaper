"""Look-ahead test (professor's requirement).

Idea: a feature for day t may only use prices up to day t. So if we scramble
every price AFTER day t, nothing on or before day t may change.

How to run (from the project folder):  python -m pytest -v
"""
import numpy as np
import pandas as pd
import pytest

from src.config import SEED
from src.features import FEATURES, features_by_ticker
from src.load_data import load_prices

# Several cut-off days t spread over the sample (each is a real trading day).
CUTOFFS = ["2005-06-30", "2011-12-30", "2018-03-15", "2023-12-29"]
PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]


@pytest.fixture(scope="module")
def raw():
    """Original price file, loaded once for all tests in this file."""
    return load_prices()


@pytest.fixture(scope="module")
def original(raw):
    """The 23 features computed from the untouched prices."""
    return features_by_ticker(raw)[FEATURES]


def scramble_after(raw, cutoff, seed):
    """Copy of `raw` where every price and volume AFTER `cutoff` is changed.

    Each column of each future row is multiplied by its own random number
    between 0.5 and 2, so the future looks completely different.
    Returns the changed copy and a True/False column marking the future rows.
    """
    rng = np.random.default_rng(seed)
    changed = raw.copy()
    future = changed["Date"] > pd.Timestamp(cutoff)
    n = int(future.sum())
    for col in PRICE_COLS:
        changed.loc[future, col] = changed.loc[future, col] * rng.uniform(0.5, 2.0, n)
    new_volume = changed.loc[future, "Volume"] * rng.uniform(0.5, 2.0, n)
    changed.loc[future, "Volume"] = new_volume.round().astype("int64")
    return changed, future


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_features_ignore_future_prices(raw, original, cutoff):
    changed, future = scramble_after(raw, cutoff, SEED)
    recomputed = features_by_ticker(changed)[FEATURES]

    # Main check: every feature on or before day t is EXACTLY the same
    # (bit for bit; missing values must also stay missing in the same places).
    past = ~future
    pd.testing.assert_frame_equal(recomputed[past], original[past], check_exact=True)

    # Sanity check: the scramble really did change the features after day t.
    # Otherwise the main check would pass for a useless reason.
    assert not recomputed[future].equals(original[future])


def test_check_catches_a_leaky_feature(raw):
    """A test is only useful if it can fail.

    Build a deliberately leaky feature (tomorrow's price / today's price) and
    confirm the same scramble changes it on or before day t.
    """
    def leaky_feature(prices):
        tomorrow = prices.groupby("Ticker")["Adj Close"].shift(-1)
        return tomorrow / prices["Adj Close"]

    changed, future = scramble_after(raw, CUTOFFS[0], SEED)
    past = ~future
    assert not leaky_feature(changed)[past].equals(leaky_feature(raw)[past])
