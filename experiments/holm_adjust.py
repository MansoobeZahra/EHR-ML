"""Holm adjustment of the 66 stacking comparisons (reviewer 2, comment 3).

Recomputes each paired AUROC difference from the saved per-fold scores, its Nadeau-Bengio
corrected t statistic (n_test/n_train = 1/4) and two-sided p-value, then applies Holm's
step-down correction over all 66 comparisons. Writes results/COMPARISON_INVENTORY.csv with
the p-values added and prints the counts used in the paper.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
EXT, EXT3 = HERE / 'out_ext', HERE / 'out_ext3'


def src(kind, key):
    if key.startswith('eicu') and kind in ('main', 'tuned', 'div', 'cat5', 'catmlp'):
        return EXT3
    return EXT


def nb_test(d, ratio=0.25):
    k = len(d); m = d.mean(); v = d.var(ddof=1)
    se = np.sqrt((1 / k + ratio) * v)
    t95 = stats.t.ppf(0.975, k - 1)
    p = 2 * stats.t.sf(abs(m / se), k - 1)
    return m, m - t95 * se, m + t95 * se, p


def holm(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (n - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


SETTINGS = ([('main', k) for k in ['mimic24', 'eicu72', 'los7_d2', 'los14_d2', 'los7_d1', 'los14_d1']] +
            [('tuned', k) for k in ['mimic24', 'eicu72', 'los7_d2', 'los14_d2']] +
            [(kind, k) for kind in ('div', 'cat5', 'catmlp') for k in ['mimic24', 'eicu72', 'los7_d2', 'los14_d2']])
rows = []
for kind, key in SETTINGS:
    f = pd.read_csv(src(kind, key) / f'{kind}_{key}_folds.csv')
    P = f.pivot(index='fold', columns='method', values='auc')
    ref = 'XGB single (tuned)' if kind == 'tuned' else 'XGB single (fixed config)'
    comps = [('Stack, XGB meta (library defaults)', 'default')]
    comps += ([('Stack, XGB meta (tuned)', 'regularized'), ('Stack, LR meta (tuned C)', 'regularized')]
              if kind == 'tuned' else
              [('Stack, XGB meta (fixed config)', 'regularized'), ('Stack, LR meta', 'regularized')])
    for mth, group in comps:
        m, lo, hi, p = nb_test((P[mth] - P[ref]).values)
        rows.append({'setting': f'{kind}:{key}', 'meta_learner': mth, 'group': group, 'reference': ref,
                     'n_folds': len(P), 'delta_points': round(100 * m, 2), 'ci_low': round(100 * lo, 2),
                     'ci_high': round(100 * hi, 2),
                     'verdict': 'worse' if hi < 0 else ('better' if lo > 0 else 'not significant'),
                     'p_value': p})
inv = pd.DataFrame(rows)
inv['p_holm'] = holm(inv.p_value)
inv['verdict_holm'] = np.where(inv.p_holm < 0.05, np.where(inv.delta_points < 0, 'worse', 'better'),
                               'not significant')
inv.round({'p_value': 5, 'p_holm': 5}).to_csv(RES / 'COMPARISON_INVENTORY.csv', index=False)

for g in ('default', 'regularized'):
    s = inv[inv.group == g]
    print(f"{g}: n={len(s)} unadjusted worse={int((s.verdict == 'worse').sum())} "
          f"holm worse={int((s.verdict_holm == 'worse').sum())} holm better={int((s.verdict_holm == 'better').sum())}")
print(inv[(inv.verdict == 'worse')][['setting', 'meta_learner', 'delta_points', 'p_value', 'p_holm', 'verdict_holm']]
      .to_string(index=False))
