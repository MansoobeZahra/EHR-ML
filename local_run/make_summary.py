"""Build the numeric part of results/SUMMARY.md from the run's CSV files.

Usage: python make_summary.py <out_dir> <summary_md_path>
Only numbers read from the CSVs are written; interpretation is added by hand.
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(sys.argv[1])
DST = Path(sys.argv[2])
L = []


def md(df, floatfmt=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: '' if pd.isna(v) else f'{v:.{floatfmt}f}')
    cols = [str(c) for c in df.columns]
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    rows += ['| ' + ' | '.join(str(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(rows)


def read(name, **kw):
    f = OUT / name
    return pd.read_csv(f, **kw) if f.exists() else None


def ci_text(d):
    d = d.copy()
    d['delta (95% NB CI)'] = [f'{m:+.4f} ({lo:+.4f}, {hi:+.4f})'
                              for m, lo, hi in zip(d['delta_AUROC_vs_XGB_single'],
                                                   d['CI95_low'], d['CI95_high'])]
    d['CI excludes 0'] = ['yes' if (lo > 0 or hi < 0) else 'no'
                          for lo, hi in zip(d['CI95_low'], d['CI95_high'])]
    return d[['task', 'method', 'delta (95% NB CI)', 'p_corrected_t', 'CI excludes 0', 'n_scores']]


L.append('## Step 22: controlled architecture comparison (identical outer folds)\n')
for roster in ('original', 'diverse'):
    s = read(f'step22_{roster}_summary.csv')
    d = read(f'step22_{roster}_deltas.csv')
    v = read(f'step22_{roster}_diversity.csv', index_col=0)
    if s is None:
        L.append(f'*{roster} roster: no output*\n')
        continue
    L.append(f'### Roster: {roster}\n')
    L.append(md(s[['task', 'method', 'AUROC', 'AUROC_sd', 'Brier', 'ECE']]) + '\n')
    L.append('Paired AUROC difference vs XGB single, Nadeau-Bengio corrected 95% CI:\n')
    L.append(md(ci_text(d)) + '\n')
    if v is not None:
        L.append('Level-1 output correlation (validation folds, mean):\n')
        L.append(md(v.reset_index().rename(columns={'index': 'task'}), 3) + '\n')

L.append('## Step 23: size sensitivity (5 stratified draws, n = 729)\n')
a = read('step23_size_sensitivity_summary.csv')
L.append(md(a) + '\n' if a is not None else '*no output*\n')

L.append('## Step 24: feature coverage before filtering, and coverage sweep\n')
b = read('step24_coverage_bands.csv', index_col=0)
if b is not None:
    L.append('Number of features per coverage band (pre-filter):\n')
    L.append(md(b.reset_index().rename(columns={'index': 'coverage band'}), 0) + '\n')
L.append('Histograms: ' + ', '.join(sorted(p.name for p in OUT.glob('coverage_hist*.png'))) + '\n')
sw = read('step24_coverage_sweep.csv')
L.append(md(sw) + '\n' if sw is not None else '*no sweep output*\n')

L.append('## Step 25: SHAP fold stability and nested (leak-free) top-K selection\n')
j = OUT / 'step25_shap_jaccard.json'
if j.exists():
    for k, v in json.load(open(j)).items():
        L.append(f'- {k}: top-20 SHAP set Jaccard across training folds, mean '
                 f'{v["mean_jaccard"]:.3f}, min {v["min_jaccard"]:.3f}')
    L.append('')
for nm in ('MIMIC-IV', 'eICU'):
    t = read(f'step25_nested_selection_{nm}.csv')
    if t is not None:
        L.append(f'{nm}:\n')
        L.append(md(t) + '\n')
    fq = read(f'step25_shap_frequency_{nm}.csv')
    if fq is not None:
        fq.columns = ['feature', 'share of folds in top-20']
        L.append(f'{nm} most frequently selected features (first 10):\n')
        L.append(md(fq.head(10), 2) + '\n')

L.append('## Step 26: LOS landmark tasks (features from ICU day 0 to t; only patients still in ICU at t)\n')
c = read('step26_landmark_cohorts.csv')
if c is not None:
    L.append(md(c, 3) + '\n')
rows = []
s = read('step22_original_summary.csv')
if s is not None:
    rows.append(s[s['task'].str.contains('LOS>7')])
s14 = read('step26_los14_original_summary.csv')
if s14 is not None:
    rows.append(s14)
if rows:
    r = pd.concat(rows)
    L.append(md(r[['task', 'method', 'AUROC', 'AUROC_sd', 'Brier', 'ECE']]) + '\n')
d14 = read('step26_los14_original_deltas.csv')
if d14 is not None:
    L.append('LOS>14 landmark, paired difference vs XGB single:\n')
    L.append(md(ci_text(d14)) + '\n')
L.append('Old protocol, for comparison only (7-day feature window, window chosen by maximum '
         'AUROC, stacked ensemble, one 5-fold run): LOS>7 AUROC 0.946, LOS>14 AUROC 0.918.\n')

DST.write_text('\n'.join(L))
print('wrote', DST)
