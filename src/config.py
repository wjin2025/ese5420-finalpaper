"""All fixed settings in one place: seed, test years, schemes, model configurations.

Authors: Jin and Qidi.

Everything here was fixed BEFORE any model was trained (professor's request),
so nobody can quietly change a setting after seeing which scheme wins.
Every other file reads its numbers from here.
"""
from pathlib import Path

# ---------- Reproducibility ----------
SEED = 5420  # one seed for every random step (fold assignment, bootstrap, LightGBM)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent  # project folder (one level above src/)
DATA_CSV = ROOT / "data" / "NASDAQ100_Historical_Data.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_CSV = RESULTS_DIR / "results.csv"     # one row per (model, year, scheme)

# ---------- Study design (from docs/proposal.txt) ----------
TEST_YEARS = list(range(2012, 2024))  # 12 replications: 2012, ..., 2023
SCHEMES = ("shuffled", "date_grouped", "walk_forward")
N_FOLDS = 5          # 5 folds for shuffled and date-grouped; 5 yearly folds for walk-forward
N_BOOT = 1000        # day-bootstrap resamples for the true-AUC 95% interval

# Final check window: used ONCE at the very end. Do not touch before then.
FINAL_TEST_START = "2024-01-01"
FINAL_TEST_END = "2026-02-18"

# ---------- Logistic regression ----------
LR_FIXED_C = 1                           # bias analysis: L2 penalty, C = 1
LR_GRID_C = [0.001, 0.01, 0.1, 1, 10]    # selection analysis: 5 log-spaced values
LR_CLIP_QUANTILES = [0.01, 0.99]         # clip each feature to its 1st-99th percentile, learned on the training fold only

# ---------- LightGBM ----------
LGBM_FIXED_NUM_LEAVES = 32               # bias analysis
LGBM_GRID_NUM_LEAVES = [8, 16, 32, 64, 128]   # selection analysis: same budget as LR (5 settings)
LGBM_OTHER = {                           # shared by every LightGBM model
    "n_estimators": 300, "learning_rate": 0.05, "min_child_samples": 200,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "random_state": SEED, "n_jobs": -1, "verbose": -1,
    # Computing only (does not change the model): identical results on every run.
    "deterministic": True, "force_row_wise": True,
}
