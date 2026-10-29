"""Tables for the Step 27 extension results (local_run/out_ext -> markdown).

Usage: python make_summary_ext.py <out_ext_dir> <md_path>
Numbers only; interpretation is written by hand.
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT, DST = Path(sys.argv[1]), Path(sys.argv[2])
TASK_ORDER = ['mimic24', 'eicu72', 'los7_d2', 'los14_d2', 'los7_d1', 'los14_d1']
L = []


def md(df, nd=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else f'{v:.{nd}f}')
    rows = ['| ' + ' | '.join(map(str, df.columns)) + ' |', '|' + '---|' * len(df.columns)]
    rows += ['| ' + ' | '.join(map(str, r)) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(rows)


def ci(d):
    d = d.copy()
    d['delta (95% NB CI)'] = [f'{m:+.4f} ({lo:+.4f}, {hi:+.4f})'
                              for m, lo, hi in zip(d['delta'], d['CI95_low'], d['CI95_high'])]
    d['excl. 0'] = ['yes' if (lo > 0 or hi < 0) else 'no'
                    for lo, hi in zip(d['CI95_low'], d['CI95_high'])]
    return d


def block(kind, title):
    L.append(f'## {title}\n')
    for t in TASK_ORDER:
        f = OUT / f'{kind}_{t}_summary.csv'
        if not f.exists():
            continue
        s = pd.read_csv(f)
        d = pd.read_csv(OUT / f'{kind}_{t}_deltas.csv')
        v = pd.read_csv(OUT / f'{kind}_{t}_diversity.csv')
        L.append(f'### {s.task.iloc[0]}\n')
        L.append(md(s[['method', 'auc', 'auc_sd', 'auprc', 'brier', 'ece', 'slope', 'citl']]) + '\n')
        for ref in d['reference'].unique():
            dd = ci(d[d.reference == ref])
            wide = dd.pivot_table(index='method', columns='metric', values='delta (95% NB CI)',
                                  aggfunc='first').reset_index()
            ex = dd[dd.metric == 'auc'][['method', 'excl. 0', 'p_corrected_t']]
            wide = wide.merge(ex, on='method')
            L.append(f'Paired difference vs **{ref}** (AUROC and AUPRC; "excl. 0" refers to AUROC):\n')
            L.append(md(wide) + '\n')
        L.append('Level-1 correlation: ' + ', '.join(
            f'{c} {v[c].iloc[0]:.3f}' for c in v.columns if c.startswith('mean_corr')) + '\n')


block('main', 'Main comparison with extra metrics and CatBoost single (original roster, fixed config)')
block('tuned', 'Equal nested tuning budget (20 random configs, 3-fold inner CV, per outer fold)')
block('div', 'Diverse Level-1 roster (XGB, LR, kNN, RF, naive Bayes, MLP)')
block('cat5', 'CatBoost added as a fifth Level-1 learner')
block('catmlp', 'CatBoost replacing the MLP in Level-1')

L.append('## Cohort size: 10 stratified draws of n = 729 (MIMIC-IV 24 h), 3 repeats each\n')
f = OUT / 'n729_deltas.csv'
if f.exists():
    d = pd.read_csv(f)
    d = d[d.metric == 'auc']
    agg = d.groupby('method').agg(
        mean_delta=('delta', 'mean'), min_delta=('delta', 'min'), max_delta=('delta', 'max'),
        draws_CI_below_0=('CI95_high', lambda x: int((x < 0).sum())),
        draws_CI_above_0=('CI95_low', lambda x: int((x > 0).sum())),
        draws=('delta', 'size')).reset_index()
    L.append(md(agg) + '\n')
    s = pd.read_csv(OUT / 'n729_summary.csv')
    L.append('Mean over draws of AUROC / ECE:\n')
    L.append(md(s.groupby('method')[['auc', 'auprc', 'ece']].mean().reset_index()) + '\n')

L.append('## Ablation under the new protocol (fixed XGBoost, identical folds, 15 scores)\n')
for t in TASK_ORDER:
    f = OUT / f'abl_{t}.csv'
    if f.exists():
        a = pd.read_csv(f)
        L.append(f'### {a.task.iloc[0]}\n')
        L.append(md(a.drop(columns='task')) + '\n')

f = OUT / 'shap_crosscohort.json'
if f.exists():
    j = json.load(open(f))
    L.append('## SHAP, fixed-config XGBoost on the full cohorts\n')
    L.append(f'Shared features: {j["n_shared_features"]}. Spearman rank correlation of mean |SHAP|: '
             f'{j["spearman_rho"]:.3f} (95% CI {j["ci95_fisher_z"][0]:.3f} to '
             f'{j["ci95_fisher_z"][1]:.3f}), p = {j["p_value"]:.2g}.\n')
    for k in ('mimic24', 'eicu72'):
        g = pd.read_csv(OUT / f'shap_global_{k}.csv').head(10)
        L.append(f'{k} top 10:\n')
        L.append(md(g, 3) + '\n')
DST.write_text('\n'.join(L))
print('wrote', DST)
