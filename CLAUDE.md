# Group 5420 final paper: how much does shuffled CV overstate next-day stock AUC?

Read docs/proposal.txt first. It is the approved plan. Do not change the design without asking Jin.
Talk to Jin in Chinese, explain every step plainly, and keep code readable (Jin must explain every line).

## Environment
conda env `finalpaper`, Python 3.11; pandas 3.1, numpy 2.4, scikit-learn 1.9, lightgbm 4.7 (see requirements.txt). Mac M1 Pro: 6 performance cores, use num_threads=6 for LightGBM.

## Data
data/NASDAQ100_Historical_Data.csv: Ticker, Date, Open, High, Low, Close, Adj Close, Volume; 100 tickers, 2000-01-03 to 2026-02-18.

## Features
The 27 indicators documented by the Kaggle competition "Stock Market Signal: Predict Next-Day Returns"
(https://www.kaggle.com/competitions/stock-market-signal-predict-next-day-returns): return_1d/5d/10d/20d,
sma_ratio_5/10/20/50/200, ema_ratio_12/26, macd_hist, rsi_14, bb_position, volatility_10d/20d/60d,
volume_sma_ratio_10/20, daily_range, avg_range_10d, high_low_ratio, momentum_10d/20d, roc_5, roc_10, atr_14.
We only borrow the definitions; our prices are the NASDAQ-100 CSV above. Implementation: Qidi's make_features (src/features.py).
We use 23 of the 27: momentum_10d, momentum_20d, roc_5, roc_10 are exact rescalings of return_10d/20d/5d/10d and are
dropped (decided by Jin + Qidi; state this in the paper and to the professor). Model table: 491,772 rows, 50.58% up.
Look-ahead test: python -m pytest (tests/test_no_lookahead.py). Rerun it after ANY change to features.py.

## Team workflow
Each part has an owner (see proposal "Who does what"); the others review it. Qidi owns features and splits;
Jin owns models; Yixun owns error analysis, bootstrap, key figure. Help Jin review others' code, do not silently rewrite it.

## Rules (never break)
- Features at day t use only data up to day t, computed per ticker (groupby Ticker). No center=True, no shift(-1) in features.
- Label = 1 if next-day Adj Close is higher (shift(-1) within ticker); drop each ticker's last day. Secondary label for robustness: beats the day's cross-sectional median return.
- 200-day warm-up dropped per ticker.
- Every step learned from data (scaling, tuning) goes inside the folds: use sklearn Pipeline.
- For test year Y: dev = 2000..Y-1, test = Y. Purge the last dev day before refitting on the full dev set.
- Three schemes on the dev set:
  1. shuffled 5-fold over rows;
  2. date-grouped 5-fold with dates assigned to folds AT RANDOM (not contiguous blocks), so a day's stocks share a fold;
  3. expanding-window walk-forward over the last 5 dev years (one fold per year), purging the last training day.
- Scheme estimate = mean of the 5 fold AUCs (same rule for all schemes).
- True AUC in year Y gets a day-bootstrap 95% interval: resample trading days with replacement, 1,000 times.
- Estimation error = estimate - true AUC. Measured with FIXED default configs, not the tuned best.
- The 2024-01 to 2026-02 test is used ONCE at the very end, after every choice. Never touch it otherwise.
- SEED = 0 everywhere. Append each result row to results/main.csv (year, model, scheme, estimate, true_auc, true_lo, true_hi, error) so runs can resume.

## Fixed configs (src/config.py)
- LR default: L2, C=1. LR grid: C in {0.001, 0.01, 0.1, 1, 10}.
- LightGBM default: library defaults + seed=0, deterministic=True, num_threads=6.
- LightGBM grid (5 configs, fixed now): (num_leaves, learning_rate, n_estimators) = (7,0.05,200), (15,0.05,200), (31,0.05,200), (15,0.1,100), (31,0.1,100).

## Order of work
LR bias analysis -> LightGBM -> model-selection analysis -> robustness reruns (no prices < $5; median label) -> final 2024-26 check -> MLP (optional) -> simulated panel (optional).

## Layout
data/ (raw, not committed to git), docs/ (proposal), notebooks/ (Colab snapshots: qidi_colab, jin_early_colab), analysis.ipynb (main team notebook), src/, tests/, results/, figures/. README.md says who wrote what.
