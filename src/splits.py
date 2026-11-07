"""Outer train/test splits, inner validation folds, and day-bootstrap resamples.

Author: Qidi. Copied from Qidi's Colab (cells "Step 4" and "Step 5", version of
2026-11-07). The logic is unchanged. Two mechanical changes only:
  1. `data` and `days` are passed in as arguments instead of global variables,
     so the functions can live in a .py file;
  2. the seed comes from src/config.py (SEED = 0, project rule) instead of 5420.
"""
import numpy as np
import pandas as pd

from src.config import SEED, N_FOLDS, N_BOOT, TEST_YEARS, FINAL_TEST_START, SCHEMES

K, B = N_FOLDS, N_BOOT

# 12 annual tests (2012-2023) + one final test (2024-2026, opened once at the end)
TESTS = {Y: (f'{Y}-01-01', f'{Y}-12-31') for Y in TEST_YEARS} | {'final': (FINAL_TEST_START, '2026-12-31')}


def trading_days(data):
    """All trading days in the model table, sorted."""
    return pd.DatetimeIndex(data['Date'].drop_duplicates().sort_values())


def split(data, days, test):
    """Train = all days before the test period except the last one, whose label is the first test day's close (purge)."""
    start, end = map(pd.Timestamp, TESTS[test])
    train_end = days[days < start][-2]
    return data[data['Date'] <= train_end], data[data['Date'].between(start, end)]


def assign_folds_and_bootstrap(data, days, seed=SEED):
    """Add two fold columns to `data` (in place) and return the bootstrap resamples.

    One fixed random draw is shared by every model and setting, so comparisons
    are paired (instructor's noise-control feedback).
      fold_shuffled: each ROW gets a fold 0-4 at random
      fold_date:     each trading DAY gets a fold 0-4 at random; all stocks of that day share it
      boot[test][b, d] = how many times test day d is drawn in resample b (1,000 resamples)
    """
    rng = np.random.default_rng(seed)
    data['fold_shuffled'] = rng.permutation(len(data)) % K                                       # rows at random
    data['fold_date'] = data['Date'].map(pd.Series(rng.permutation(len(days)) % K, index=days))  # whole trading days at random
    boot = {}
    for test, (start, end) in TESTS.items():
        n = ((days >= start) & (days <= end)).sum()
        boot[test] = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.int16)
    return boot


def inner_folds(data, days, test, scheme):
    """Yield (train_index, validation_index) inside the training set of `test`."""
    dev = split(data, days, test)[0]
    if scheme == 'walk_forward':                                   # each of the last 5 training years validates once
        last = dev['Date'].dt.year.max()
        for year in range(last - K + 1, last + 1):
            train_end = days[days < pd.Timestamp(f'{year}-01-01')][-2]      # purge the day before the validation year
            yield dev.index[dev['Date'] <= train_end], dev.index[dev['Date'].dt.year == year]
    else:
        fold = dev['fold_shuffled' if scheme == 'shuffled' else 'fold_date']
        for k in range(K):
            yield dev.index[fold != k], dev.index[fold == k]
