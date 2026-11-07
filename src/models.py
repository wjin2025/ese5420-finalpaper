"""The two models with their fixed preprocessing.

Author: Qidi. Copied from Qidi's Colab (cell "Step 6", version of 2026-11-07);
the logic is unchanged, the settings now come from src/config.py.
"""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import LR_CLIP_QUANTILES, LGBM_OTHER


class Clip(BaseEstimator, TransformerMixin):
    """Clip each feature to its 1st-99th percentile, learned on the training fold only."""
    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.quantile(X, LR_CLIP_QUANTILES, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def logistic(C):
    return make_pipeline(Clip(), StandardScaler(), LogisticRegression(C=C, max_iter=1000))   # every step fit inside the fold


def lightgbm(num_leaves):
    return LGBMClassifier(num_leaves=num_leaves, **LGBM_OTHER)                # trees need no scaling
