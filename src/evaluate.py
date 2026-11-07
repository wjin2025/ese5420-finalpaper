"""CV estimates, year-Y AUC with its day-bootstrap interval, and the yearly loop.

Author: Qidi. Copied from Qidi's Colab (cells "Step 6" day_bootstrap_auc,
"Step 7" and "Step 8", version of 2026-11-07). The logic is unchanged.
Mechanical changes only: `data`, `days`, `boot` are passed in as arguments
instead of global variables; the loop takes the model's name so the same
results file can later hold LightGBM rows too.

For one test year Y:
  cv_auc   = mean validation AUC over the 5 inner folds of a scheme (dev = before Y)
  test_auc = AUC on year Y of a model trained on the whole (purged) dev set
  test_lo, test_hi = 95% interval from 1,000 day-bootstrap resamples of year Y
  error    = cv_auc - test_auc
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import RESULTS_CSV, SCHEMES, TEST_YEARS
from src.features import FEATURES
from src.splits import TESTS, split, inner_folds

RESULT_COLUMNS = ['model', 'year', 'scheme', 'cv_auc', 'test_auc', 'test_lo', 'test_hi']


def day_bootstrap_auc(y, score, day_pos, counts, chunk=50):
    """AUC on every day-bootstrap resample: each row weighted by how often its trading day was drawn."""
    o = np.argsort(score, kind='mergesort')
    y, day_pos = y[o], day_pos[o]
    out = []
    for c in np.array_split(counts, max(1, len(counts) // chunk)):
        w = c[:, day_pos].astype(float)
        neg, pos = w * (1 - y), w * y
        out.append((pos * (np.cumsum(neg, axis=1) - neg)).sum(1) / (pos.sum(1) * neg.sum(1)))
    return np.concatenate(out)


def cv_auc(data, days, test, scheme, make_model):
    """CV estimate = mean validation AUC over the 5 inner folds of the test's training set."""
    aucs = []
    for tr, va in inner_folds(data, days, test, scheme):
        m = make_model().fit(data.loc[tr, FEATURES].to_numpy(), data.loc[tr, 'target'].to_numpy())
        aucs.append(roc_auc_score(data.loc[va, 'target'], m.predict_proba(data.loc[va, FEATURES].to_numpy())[:, 1]))
    return np.mean(aucs)


def test_auc(data, days, boot, test, make_model):
    """Year-Y AUC of a model trained on the whole training set, with its 95% day-bootstrap interval."""
    tr, te = split(data, days, test)
    p = make_model().fit(tr[FEATURES].to_numpy(), tr['target'].to_numpy()).predict_proba(te[FEATURES].to_numpy())[:, 1]
    y = te['target'].to_numpy()
    start, end = TESTS[test]
    day_pos = days[(days >= start) & (days <= end)].get_indexer(te['Date'])
    assert np.isclose(day_bootstrap_auc(y, p, day_pos, np.ones((1, day_pos.max() + 1))), roc_auc_score(y, p))
    return roc_auc_score(y, p), *np.percentile(day_bootstrap_auc(y, p, day_pos, boot[test]), [2.5, 97.5])


def run_years(data, days, boot, model_name, make_model, years=TEST_YEARS, path=RESULTS_CSV):
    """CV estimate of each scheme vs the year-Y AUC, for every test year.

    Results are saved after every year, so an interrupted run resumes where it stopped.
    Returns all rows of this model, with error = cv_auc - test_auc.
    """
    path.parent.mkdir(parents=True, exist_ok=True)          # create results/ if missing
    results = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=RESULT_COLUMNS)
    for Y in years:
        if ((results['model'] == model_name) & (results['year'] == Y)).any():
            continue
        auc, lo, hi = test_auc(data, days, boot, Y, make_model)
        rows = [{'model': model_name, 'year': Y, 'scheme': s, 'cv_auc': cv_auc(data, days, Y, s, make_model),
                 'test_auc': auc, 'test_lo': lo, 'test_hi': hi} for s in SCHEMES]
        results = pd.concat([results.dropna(axis=1, how='all'), pd.DataFrame(rows)], ignore_index=True)
        results.to_csv(path, index=False)
        print(f'{model_name} {Y} done', flush=True)
    results['error'] = results['cv_auc'] - results['test_auc']
    return results[results['model'] == model_name].reset_index(drop=True)
