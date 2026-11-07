"""12-year summary and the key figure.

Author: Qidi. Copied from Qidi's Colab (cell "Step 9", version of 2026-11-07).
The logic is unchanged; it is split into two functions so it can be reused for
LightGBM. (Per the proposal, the key figure is Yixun's part to refine.)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import SCHEMES


def ci(d):
    """95% t-interval half-width across years (years treated as independent)."""
    t = stats.t.ppf(0.975, len(d) - 1)
    return t * d.std(ddof=1) / len(d) ** 0.5


def summarize(r):
    """Per-scheme mean error and paired scheme differences. r = results rows of ONE model."""
    err = r.pivot(index='year', columns='scheme', values='error')[list(SCHEMES)]
    outside = r.assign(out=(r['cv_auc'] > r['test_hi']) | (r['cv_auc'] < r['test_lo'])).pivot(index='year', columns='scheme', values='out')
    by_scheme = pd.DataFrame({'mean_error': err.mean(), 'ci_95': err.apply(ci), 'years_overstated': (err > 0).sum(),
                              'years_outside_noise': outside[list(SCHEMES)].sum()})
    pairs = {'shuffled - walk_forward': err['shuffled'] - err['walk_forward'],
             'date_grouped - walk_forward': err['date_grouped'] - err['walk_forward'],
             'shuffled - date_grouped': err['shuffled'] - err['date_grouped']}
    paired = pd.DataFrame({k: {'mean_diff': d.mean(), 'ci_95': ci(d), 'years_positive': (d > 0).sum()} for k, d in pairs.items()}).T
    return err, by_scheme, paired


def key_figure(r, title, path=None):
    """x: scheme; y: estimation error. Dots = single years, big dot = 12-year mean +/- 95% CI,
    grey band = typical year-Y noise (mean half-width of the 95% day-bootstrap interval)."""
    err = r.pivot(index='year', columns='scheme', values='error')[list(SCHEMES)]
    yearly = r.groupby('year')[['test_auc', 'test_lo', 'test_hi']].first()
    band = ((yearly['test_hi'] - yearly['test_lo']) / 2).mean()
    INK, MUTED, BLUE, BAND = '#0b0b0b', '#52514e', '#2a78d6', '#e9e8e4'
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.axhspan(-band, band, color=BAND, lw=0, zorder=0, label=f'Typical year-Y noise (95% day bootstrap, ±{band:.3f})')
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    for i, s in enumerate(SCHEMES):
        ax.scatter(i - 0.12 + np.linspace(-0.08, 0.08, len(err)), err[s], s=16, color=BLUE, alpha=0.35, lw=0, zorder=2,
                   label='One test year' if i == 0 else None)
        m, h = err[s].mean(), ci(err[s])
        ax.errorbar(i + 0.14, m, yerr=h, fmt='o', ms=7, color=BLUE, ecolor=BLUE, elinewidth=2, capsize=0, zorder=3,
                    label='Mean of 12 years ± 95% CI' if i == 0 else None)
        ax.annotate(f'{m:+.4f}', (i + 0.14, m), xytext=(8, 0), textcoords='offset points', va='center', fontsize=8, color=INK)
    ax.set_xticks(range(len(SCHEMES)), ['Shuffled 5-fold', 'Date-grouped 5-fold', 'Walk-forward'], color=INK)
    ax.set_ylabel('Estimation error (CV AUC − year-Y AUC)', color=INK)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200)
    return fig
