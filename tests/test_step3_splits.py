"""Tests for STEP 3: the splits.

Authors: Jin and Qidi.

No row in both train and validation; date-grouped and walk-forward never split a
day; walk-forward trains only on earlier days with one purged day; every outer
split purges exactly one day; bootstrap draws n days per resample.
"""
import numpy as np
import pytest

from src.config import SCHEMES, N_FOLDS
from src.step2_features import build_model_table
from src.step1_load_data import load_prices
from src.step3_splits import TESTS, trading_days, split, assign_folds_and_bootstrap, inner_folds


@pytest.fixture(scope="module")
def setup():
    data = build_model_table(load_prices())
    days = trading_days(data)
    boot = assign_folds_and_bootstrap(data, days)
    return data, days, boot


@pytest.mark.parametrize("test", list(TESTS))
def test_outer_split_purges_exactly_one_day(setup, test):
    data, days, _ = setup
    tr, te = split(data, days, test)
    gap = days.searchsorted(te['Date'].min()) - days.searchsorted(tr['Date'].max())
    assert gap == 2                      # exactly one purged trading day between train and test


@pytest.mark.parametrize("test", list(TESTS))
@pytest.mark.parametrize("scheme", SCHEMES)
def test_inner_folds(setup, test, scheme):
    data, days, _ = setup
    dev_rows = len(split(data, days, test)[0])
    n_val = 0
    for tr, va in inner_folds(data, days, test, scheme):
        tr_dates, va_dates = data.loc[tr, 'Date'], data.loc[va, 'Date']
        assert tr.intersection(va).empty                              # no row is in both train and validation
        if scheme != 'shuffled':                                      # no day is split across train and validation
            assert not va_dates.isin(set(tr_dates)).any()
        if scheme == 'walk_forward':                                  # training ends before validation, one day purged
            assert days.searchsorted(va_dates.min()) - days.searchsorted(tr_dates.max()) == 2
        n_val += len(va)
    if scheme != 'walk_forward':                                      # random folds cover the whole training set
        assert n_val == dev_rows


def test_bootstrap_draws_every_resample_with_n_days(setup):
    data, days, boot = setup
    for test, (start, end) in TESTS.items():
        n = ((days >= start) & (days <= end)).sum()
        assert boot[test].shape == (1000, n)
        assert (boot[test].sum(axis=1) == n).all()      # each resample draws n days, with replacement


def test_folds_are_balanced(setup):
    data, _, _ = setup
    for col in ['fold_shuffled', 'fold_date']:
        share = data[col].value_counts(normalize=True)
        assert len(share) == N_FOLDS and np.allclose(share, 1 / N_FOLDS, atol=0.01)
