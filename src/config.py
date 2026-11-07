"""Fixed settings for the whole project.

Author: Jin (written with Claude's help; see README, AI use).

Everything here is decided BEFORE looking at any results (professor's request),
so that nobody can quietly tune the grid after seeing which scheme wins.
Other files import from here instead of typing numbers again.
"""
from pathlib import Path

# ---------- Reproducibility ----------
SEED = 0  # one seed for every random step (fold assignment, bootstrap, LightGBM)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent  # project folder (one level above src/)
DATA_CSV = ROOT / "data" / "NASDAQ100_Historical_Data.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
MAIN_RESULTS_CSV = RESULTS_DIR / "main.csv"

# ---------- Study design (from proposal.txt) ----------
TEST_YEARS = list(range(2012, 2024))  # 12 replications: 2012, ..., 2023
SCHEMES = ["shuffled", "date_grouped", "walk_forward"]
N_FOLDS = 5          # 5 folds for shuffled and date-grouped; 5 yearly folds for walk-forward
WARMUP_DAYS = 200    # first 200 rows of each ticker are dropped (SMA-200 needs them)
N_BOOT = 1000        # day-bootstrap resamples for the true-AUC 95% interval

# Final check window: used ONCE at the very end. Do not touch before then.
FINAL_TEST_START = "2024-01-01"
FINAL_TEST_END = "2026-02-18"

# ---------- Logistic regression ----------
# Default config (used to measure estimation error): L2 penalty, C = 1.
LR_DEFAULT = {"C": 1.0}
# Grid for the model-selection analysis: 5 log-spaced values of C.
LR_GRID = [{"C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0]]

# ---------- LightGBM ----------
# Settings shared by every LightGBM model: they control reproducibility and
# CPU use only, not the model itself.
LGBM_COMMON = {
    "random_state": SEED,   # LightGBM's "seed"
    "deterministic": True,  # same result on every run
    "force_row_wise": True, # LightGBM docs: pair with deterministic for exact repeats
    "verbose": -1,          # silence LightGBM's training log
    "num_threads": 6,       # M1 Pro has 6 performance cores
}
# Default config: library defaults plus the common settings above.
LGBM_DEFAULT = dict(LGBM_COMMON)
# Grid: 5 fixed configs, the same budget as the LR grid.
# Each tuple is (num_leaves, learning_rate, n_estimators).
_LGBM_GRID_TUPLES = [
    (7, 0.05, 200),
    (15, 0.05, 200),
    (31, 0.05, 200),
    (15, 0.1, 100),
    (31, 0.1, 100),
]
LGBM_GRID = [
    {**LGBM_COMMON, "num_leaves": nl, "learning_rate": lr, "n_estimators": n}
    for nl, lr, n in _LGBM_GRID_TUPLES
]
