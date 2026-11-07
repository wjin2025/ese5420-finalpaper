"""Jin's part: the LR model, the three scheme estimates, the true AUC with its
day-bootstrap 95% interval, and the results file.

For one test year Y:
  estimate  = mean of the 5 fold AUCs of a scheme, on the dev set (2000..Y-1)
  true AUC  = refit on the whole (purged) dev set, AUC on year Y
  interval  = 2.5% and 97.5% points of 1,000 day-bootstrap AUCs of year Y
  error     = estimate - true AUC
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import LR_DEFAULT, MAIN_RESULTS_CSV, SCHEMES
from src.features import FEATURES
from src.splits import TESTS, split, inner_folds

RESULT_COLUMNS = ['year', 'model', 'scheme', 'estimate', 'true_auc', 'true_lo', 'true_hi', 'error']


def make_lr(C=LR_DEFAULT['C']):
    """Standardize the 23 features, then logistic regression with an L2 penalty.

    Because both steps sit in one Pipeline, the scaler learns its means and
    standard deviations only from the rows the model is fit on (the training
    folds), never from the validation or test rows.
    """
    return Pipeline([
        ('scale', StandardScaler()),
        ('model', LogisticRegression(C=C, max_iter=1000)),   # sklearn's default penalty is L2
    ])


def fit_and_predict(make_model, train, test):
    """Fit a fresh model on `train` rows; return its P(up) for every `test` row."""
    model = make_model()
    model.fit(train[FEATURES], train['target'])
    return model.predict_proba(test[FEATURES])[:, 1]


def scheme_estimate(data, days, test, scheme, make_model):
    """AUC estimate of one scheme = mean of its 5 fold AUCs. Also returns the 5 AUCs."""
    fold_aucs = []
    for train_idx, valid_idx in inner_folds(data, days, test, scheme):
        train, valid = data.loc[train_idx], data.loc[valid_idx]
        score = fit_and_predict(make_model, train, valid)
        fold_aucs.append(roc_auc_score(valid['target'], score))
    return float(np.mean(fold_aucs)), fold_aucs


def true_auc_with_interval(data, days, test, boot, make_model):
    """Refit on the whole dev set, score the test period; bootstrap whole trading days.

    In resample b, a row of test day d gets weight boot[test][b, d] (how many
    times day d was drawn), so all stocks of a day are drawn together.
    """
    dev, test_rows = split(data, days, test)
    score = fit_and_predict(make_model, dev, test_rows)
    true_auc = roc_auc_score(test_rows['target'], score)

    start, end = TESTS[test]
    test_days = days[(days >= start) & (days <= end)]
    day_of_row = test_days.get_indexer(test_rows['Date'])      # position 0..n-1 of each row's day
    boot_aucs = [roc_auc_score(test_rows['target'], score, sample_weight=counts[day_of_row])
                 for counts in boot[test]]
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return float(true_auc), float(lo), float(hi)


def finished_runs():
    """(year, model, scheme) triples already saved in results/main.csv."""
    if not MAIN_RESULTS_CSV.exists():
        return set()
    done = pd.read_csv(MAIN_RESULTS_CSV)
    return set(zip(done['year'], done['model'], done['scheme']))


def append_results(rows):
    """Add rows to results/main.csv (write the header only if the file is new)."""
    new_file = not MAIN_RESULTS_CSV.exists()
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(MAIN_RESULTS_CSV, mode='a', header=new_file, index=False)


def run_year(data, days, boot, year, model_name='LR', make_model=make_lr):
    """All three schemes + true AUC for one test year; saves and returns the rows.

    Skips the year if all three rows are already in results/main.csv (resume).
    """
    done = finished_runs()
    if all((year, model_name, s) in done for s in SCHEMES):
        print(f'{year} {model_name}: already in {MAIN_RESULTS_CSV.name}, skipped')
        return None
    true_auc, lo, hi = true_auc_with_interval(data, days, year, boot, make_model)
    rows = []
    for scheme in SCHEMES:
        estimate, fold_aucs = scheme_estimate(data, days, year, scheme, make_model)
        rows.append({'year': year, 'model': model_name, 'scheme': scheme, 'estimate': estimate,
                     'true_auc': true_auc, 'true_lo': lo, 'true_hi': hi, 'error': estimate - true_auc})
        print(f'{year} {model_name} {scheme:13s} folds: {np.round(fold_aucs, 4)}  mean: {estimate:.4f}')
    append_results(rows)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
