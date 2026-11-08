"""STEP 2. Build the label and the 23 features for every (Ticker, Date).

Authors: Jin and Qidi. First developed in the group Colab (notebooks/colab_notebook.ipynb).

Features = the 27 indicators documented by the Kaggle competition
"Stock Market Signal: Predict Next-Day Returns", minus 4 exact copies of the
returns (momentum_10d, momentum_20d, roc_5, roc_10), so 23 features.
Label (target) = 1 if the next day's adjusted close is higher.
All prices are split- AND dividend-adjusted (one price basis).
Checked by tests/test_step2_no_lookahead.py.
"""
import numpy as np
import pandas as pd

FEATURES = ['return_1d', 'return_5d', 'return_10d', 'return_20d',
            'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50', 'sma_ratio_200',
            'ema_ratio_12', 'ema_ratio_26', 'macd_hist', 'rsi_14', 'bb_position',
            'volatility_10d', 'volatility_20d', 'volatility_60d', 'volume_sma_ratio_10', 'volume_sma_ratio_20',
            'daily_range', 'avg_range_10d', 'high_low_ratio', 'atr_14']


def make_features(d):
    """Features and label for ONE ticker. d = that ticker's rows, sorted by Date."""
    k = d['Adj Close'] / d['Close']                          # that day's adjustment factor
    c, h, l = d['Adj Close'], d['High'] * k, d['Low'] * k    # adjusted close, high, low
    v = d['Volume'].where(d['Volume'] > 0)                   # zero volume -> missing
    r = c.pct_change()
    x = pd.DataFrame(index=d.index)
    for n in (1, 5, 10, 20):
        x[f'return_{n}d'] = c.pct_change(n)
    for n in (5, 10, 20, 50, 200):
        x[f'sma_ratio_{n}'] = c / c.rolling(n).mean()
    ema12, ema26 = (c.ewm(span=s, adjust=False, min_periods=s).mean() for s in (12, 26))
    x['ema_ratio_12'], x['ema_ratio_26'] = c / ema12, c / ema26
    macd = ema12 - ema26
    x['macd_hist'] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / c
    gain, loss = (s.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() for s in (c.diff().clip(lower=0), -c.diff().clip(upper=0)))
    x['rsi_14'] = (100 * gain / (gain + loss)).mask(gain + loss == 0, 50)            # Wilder RSI; flat window -> 50
    x['bb_position'] = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std())
    for n in (10, 20, 60):
        x[f'volatility_{n}d'] = r.rolling(n).std() * 252 ** 0.5
    for n in (10, 20):
        x[f'volume_sma_ratio_{n}'] = v / v.rolling(n, min_periods=n // 2).mean()
    x['daily_range'] = (h - l) / c
    x['avg_range_10d'] = x['daily_range'].rolling(10).mean()
    x['high_low_ratio'] = h / l
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    x['atr_14'] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c       # Wilder ATR / price
    x['target'] = (c.shift(-1) > c).astype(float).where(c.shift(-1).notna())          # 1 if next day's Adj Close is higher
    return x.replace([np.inf, -np.inf], np.nan)


def features_by_ticker(raw):
    """Run make_features on each ticker separately and stack the results.

    Returns one row per row of `raw` (same index), BEFORE dropping missing
    values. The look-ahead test uses this version.
    """
    return pd.concat([make_features(d) for _, d in raw.groupby('Ticker')])


def build_model_table(raw):
    """Ticker, Date, 23 features and target; rows with any missing value dropped.

    dropna removes the 200-day warm-up, each ticker's last day (no next-day
    label) and zero-volume / flat-price days. Expected: 491,772 rows.
    """
    return raw[['Ticker', 'Date']].join(features_by_ticker(raw)).dropna()
