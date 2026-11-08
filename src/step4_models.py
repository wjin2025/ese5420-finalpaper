"""STEP 4. The two models and their preprocessing.

Authors: Jin and Qidi. First developed in the group Colab (notebooks/colab_notebook.ipynb).

  logistic(C):          clip extremes -> standardize -> L2 logistic regression (all fit inside the fold)
  lightgbm(num_leaves): gradient-boosted trees (no scaling needed)
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
