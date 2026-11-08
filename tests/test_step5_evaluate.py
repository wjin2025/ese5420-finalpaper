"""Tests for STEPS 4-5: the evaluation code.

Authors: Jin and Qidi.

The fast bootstrap AUC must equal sklearn's AUC under real bootstrap weights
(days drawn 0, 1, 2, ... times), and the clipping limits must come from the
training rows only.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.step5_evaluate import day_bootstrap_auc
from src.step4_models import Clip


def test_day_bootstrap_auc_matches_sklearn_with_weights():
    rng = np.random.default_rng(0)
    n_rows, n_days = 2000, 50
    y = rng.integers(0, 2, n_rows).astype(float)
    score = rng.normal(size=n_rows) + 0.3 * y            # a weak but real signal
    day_pos = rng.integers(0, n_days, n_rows)            # which day each row belongs to
    counts = rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=20)   # 20 day-bootstrap resamples
    ours = day_bootstrap_auc(y, score, day_pos, counts)
    sklearn = [roc_auc_score(y, score, sample_weight=c[day_pos]) for c in counts]
    np.testing.assert_allclose(ours, sklearn, rtol=0, atol=1e-12)


def test_clip_learns_limits_from_training_rows_only():
    train = np.arange(100, dtype=float).reshape(-1, 1)   # values 0..99
    test = np.array([[-1000.0], [50.0], [1000.0]])
    clip = Clip().fit(train)
    out = clip.transform(test)
    lo, hi = np.quantile(train, [0.01, 0.99])
    assert out[0, 0] == lo and out[2, 0] == hi and out[1, 0] == 50.0   # extremes cut to TRAINING limits
