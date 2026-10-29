"""Every number quoted in the v3 text, computed from the result files.

eICU results come from out_ext2 (patient-grouped outer folds); MIMIC-IV and LOS from
out_ext. Run directly to print all facts for review. Imported by build_paper_v3.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXT, EXT2 = HERE / 'out_ext', HERE / 'out_ext2'
FINAL = HERE.parent / 'results' / 'final'
MAIN4 = ['mimic24', 'eicu72', 'los7_d2', 'los14_d2']
ALL6 = MAIN4 + ['los7_d1', 'los14_d1']
SINGLE, TSINGLE = 'XGB single (fixed config)', 'XGB single (tuned)'
DEF = 'Stack, XGB meta (library defaults)'
FIX, LRM = 'Stack, XGB meta (fixed config)', 'Stack, LR meta'
TFIX, TLRM = 'Stack, XGB meta (tuned)', 'Stack, LR meta (tuned C)'
MEAN, BEST = 'Mean of Level-1', 'Best single Level-1 (picked in-fold)'


EXT3 = HERE / 'out_ext3'      # eICU stacking units re-run with patient-grouped inner splits too


def src(kind, key):
    if kind in ('main', 'tuned', 'div', 'cat5', 'catmlp', 'nodeath') and key.startswith('eicu'):
        return EXT3
    return EXT2 if (key.startswith('eicu') or kind == 'nodeath') else EXT


def S(kind, key):
    return pd.read_csv(src(kind, key) / f'{kind}_{key}_summary.csv').set_index('method')


def D(kind, key, ref=None, metric='auc'):
    d = pd.read_csv(src(kind, key) / f'{kind}_{key}_deltas.csv')
    d = d[d.metric == metric]
    if ref:
        d = d[d.reference == ref]
    return d.set_index('method')


def V(kind, key):
    return pd.read_csv(src(kind, key) / f'{kind}_{key}_diversity.csv').iloc[0]


def pts(v, sign=True):
    """AUROC difference in points (x100), one decimal, true minus sign."""
    s = f'{100 * v:+.1f}' if sign else f'{100 * abs(v):.1f}'
    return s.replace('-', '−')


def sig(lo, hi):
    return (lo > 0) or (hi < 0)


def rng(vals, fmt='{:.3f}'):
    return fmt.format(min(vals)), fmt.format(max(vals))


F = {}
# ---------------------------------------------------------------- Table II facts
d_def = [D('main', k, SINGLE).loc[DEF] for k in MAIN4]
s_main = {k: S('main', k) for k in MAIN4}
F['def_min'], F['def_max'] = (pts(-max(r.delta for r in d_def), False),
                              pts(-min(r.delta for r in d_def), False))
rel = [-(r.delta) / s_main[k].loc[SINGLE, 'auc'] * 100 for r, k in zip(d_def, MAIN4)]
F['def_rel_min'], F['def_rel_max'] = f'{min(rel):.1f}', f'{max(rel):.1f}'
F['def_sig4'] = sum(sig(r.CI95_low, r.CI95_high) for r in d_def)
F['ece_def'] = rng([s_main[k].loc[DEF, 'ece'] for k in MAIN4])
F['ece_single'] = rng([s_main[k].loc[SINGLE, 'ece'] for k in MAIN4])
F['slope_def'] = rng([s_main[k].loc[DEF, 'slope'] for k in MAIN4], '{:.2f}')
F['slope_single'] = rng([s_main[k].loc[SINGLE, 'slope'] for k in MAIN4], '{:.2f}')
F['citl_def'] = rng([s_main[k].loc[DEF, 'citl'] for k in MAIN4], '{:+.2f}')
for name, m in (('lr', LRM), ('fix', FIX)):
    dd = [D('main', k, SINGLE).loc[m] for k in MAIN4]
    F[f'{name}_min'], F[f'{name}_max'] = pts(min(r.delta for r in dd)), pts(max(r.delta for r in dd))
    F[f'{name}_sig'] = [k for r, k in zip(dd, MAIN4) if sig(r.CI95_low, r.CI95_high)]
F['ece_reg'] = rng([s_main[k].loc[m, 'ece'] for k in MAIN4 for m in (FIX, LRM)])
for k in ('mimic24', 'eicu72'):
    r = D('main', k, SINGLE).loc[MEAN]
    F[f'mean_{k}'] = pts(r.delta); F[f'mean_{k}_sig'] = sig(r.CI95_low, r.CI95_high)
    F[f'mean_ece_{k}'] = f"{s_main[k].loc[MEAN, 'ece']:.3f}"
best = [D('main', k, SINGLE).loc[BEST].delta for k in MAIN4]
F['best_min'], F['best_max'] = pts(-max(best), False), pts(-min(best), False)
r = D('main', 'los7_d2', SINGLE, 'auprc').loc[FIX]
F['auprc_los7_fix'] = (pts(r.delta), pts(r.CI95_low), pts(r.CI95_high), sig(r.CI95_low, r.CI95_high))

# ------------------------------------------------------- all settings (22 / 44)
SETTINGS = ([('main', k, SINGLE) for k in ALL6] + [('tuned', k, TSINGLE) for k in MAIN4] +
            [(kind, k, SINGLE) for kind in ('div', 'cat5', 'catmlp') for k in MAIN4])
def_all, reg_all = [], []
for kind, k, ref in SETTINGS:
    d = D(kind, k, ref)
    def_all.append((kind, k, d.loc[DEF].delta, d.loc[DEF].CI95_low, d.loc[DEF].CI95_high))
    for m in ((TFIX, TLRM) if kind == 'tuned' else (FIX, LRM)):
        reg_all.append((kind, k, m, d.loc[m].delta, d.loc[m].CI95_low, d.loc[m].CI95_high))
F['n_settings'] = len(def_all)
F['def_all_sigworse'] = sum(hi < 0 for *_, lo, hi in def_all)
F['def_all_sigbetter'] = sum(lo > 0 for *_, lo, hi in def_all)
F['def_all_range'] = (pts(-max(d for _, _, d, _, _ in def_all), False),
                      pts(-min(d for _, _, d, _, _ in def_all), False))
F['n_reg'] = len(reg_all)
F['reg_ns'] = sum(not sig(lo, hi) for *_, lo, hi in reg_all)
F['reg_worse'] = [(kind, k, m, pts(d)) for kind, k, m, d, lo, hi in reg_all if hi < 0]
F['reg_better'] = [(kind, k, m, pts(d)) for kind, k, m, d, lo, hi in reg_all if lo > 0]
# tuned stacks against the UNTUNED single model (disclosure)
dt = D('tuned', 'mimic24', SINGLE)
F['tuned_vs_fixed_better'] = [(k, m, pts(D('tuned', k, SINGLE).loc[m].delta))
                              for k in MAIN4 for m in (TFIX, TLRM)
                              if D('tuned', k, SINGLE).loc[m].CI95_low > 0]

# ------------------------------------------------------------------ tuning facts
F['tuned_auc'] = {k: f"{S('tuned', k).loc[TSINGLE, 'auc']:.3f}" for k in MAIN4}
gain = {k: D('tuned', k, SINGLE).loc[TSINGLE] for k in MAIN4}
F['tune_gain'] = rng([100 * g.delta for g in gain.values()], '{:+.1f}')
F['tune_gain_sig'] = [k for k, g in gain.items() if sig(g.CI95_low, g.CI95_high)]
F['tuned_stack'] = {k: {m: (pts(D('tuned', k, TSINGLE).loc[m].delta),
                            sig(D('tuned', k, TSINGLE).loc[m].CI95_low,
                                D('tuned', k, TSINGLE).loc[m].CI95_high))
                        for m in (TFIX, TLRM, DEF)} for k in MAIN4}
F['tuned_def_range'] = (pts(-max(D('tuned', k, TSINGLE).loc[DEF].delta for k in MAIN4), False),
                        pts(-min(D('tuned', k, TSINGLE).loc[DEF].delta for k in MAIN4), False))

# ---------------------------------------------------------- diversity / CatBoost
F['corr'] = {k: (f"{V('main', k)['mean_corr_diff_algo_same_subset']:.2f}",
                 f"{V('div', k)['mean_corr_diff_algo_same_subset']:.2f}") for k in MAIN4}
F['corr_main_range'] = rng([V('main', k)['mean_corr_diff_algo_same_subset'] for k in MAIN4], '{:.2f}')
cat = [D('main', k, SINGLE).loc['CatBoost single (fixed config)'] for k in ALL6]
F['cat_range'] = (pts(min(r.delta for r in cat)), pts(max(r.delta for r in cat)))
F['cat_sig'] = sum(sig(r.CI95_low, r.CI95_high) for r in cat)

# ----------------------------------------------------------------- cohort size
n7 = pd.read_csv(EXT / 'n729_deltas.csv'); n7 = n7[n7.metric == 'auc']
g = n7.groupby('method')
F['n729'] = {m: (pts(g.get_group(m).delta.mean()),
                 int((g.get_group(m).CI95_high < 0).sum()), int((g.get_group(m).CI95_low > 0).sum()))
             for m in (FIX, LRM, DEF, MEAN)}

# ------------------------------------------------------------------- ablation
F['abl'] = {}
for k in MAIN4:
    a = pd.read_csv(src('abl', k) / f'abl_{k}.csv').set_index('condition')
    F['abl'][k] = {c: (pts(a.loc[c, 'delta_vs_all']), sig(a.loc[c, 'CI95_low'], a.loc[c, 'CI95_high']))
                   for c in a.index if c != 'All features'}
    F['abl'][k]['n_features'] = int(a.loc['All features', 'n_features'])

# ------------------------------------------------------------------- coverage
cm = pd.read_csv(FINAL / 'step24_coverage_sweep.csv')
cm = cm[cm.dataset == 'MIMIC-IV 24h'].set_index('threshold')
ce = pd.read_csv(EXT2 / 'cov_eicu72.csv').set_index('threshold')
F['cov_mimic'] = {t: (int(cm.loc[t, 'n_features']), pts(cm.loc[t, 'delta_vs_ref']),
                      pts(cm.loc[t, 'CI95_low']), pts(cm.loc[t, 'CI95_high']),
                      sig(cm.loc[t, 'CI95_low'], cm.loc[t, 'CI95_high'])) for t in (0.7, 0.9)}
F['cov_eicu'] = {t: (int(ce.loc[t, 'n_features']), pts(ce.loc[t, 'delta_vs_ref']),
                     pts(ce.loc[t, 'CI95_low']), pts(ce.loc[t, 'CI95_high']),
                     sig(ce.loc[t, 'CI95_low'], ce.loc[t, 'CI95_high'])) for t in (0.6, 0.7, 0.9)}

# ---------------------------------------------------------------- SHAP / selection
jm = json.load(open(FINAL / 'step25_shap_jaccard.json'))['MIMIC-IV 24h mortality']
je = json.load(open(EXT2 / 'shapstab_eicu72_jaccard.json'))
F['jac'] = {'mimic24': (f"{jm['mean_jaccard']:.2f}", f"{jm['min_jaccard']:.2f}"),
            'eicu72': (f"{je['mean_jaccard']:.2f}", f"{je['min_jaccard']:.2f}")}
sm = pd.read_csv(FINAL / 'step25_nested_selection_MIMIC-IV.csv').set_index('top_k')
se = pd.read_csv(EXT2 / 'shapstab_eicu72_selection.csv').set_index('top_k')
F['sel'] = {name: {k: (pts(t.loc[k, 'delta_vs_all']), sig(t.loc[k, 'CI95_low'], t.loc[k, 'CI95_high']))
                   for k in (10, 20, 30, 40)} for name, t in (('mimic24', sm), ('eicu72', se))}
sc = json.load(open(EXT / 'shap_crosscohort.json'))
F['spearman'] = (f"{sc['spearman_rho']:.2f}", f"{sc['ci95_fisher_z'][0]:.2f}",
                 f"{sc['ci95_fisher_z'][1]:.2f}", sc['n_shared_features'])

# ---------------------------------------------------------------- early deaths
F['nodeath'] = {}
for k in ('mimic24', 'eicu72'):
    s = pd.read_csv(src('nodeath', k) / f'nodeath_{k}_summary.csv').set_index('method')
    d = pd.read_csv(src('nodeath', k) / f'nodeath_{k}_deltas.csv')
    d = d[d.metric == 'auc'].set_index('method')
    F['nodeath'][k] = {'excluded': int(s['excluded'].iloc[0]), 'n': int(s['n'].iloc[0]),
                       'single_auc': f"{s.loc[SINGLE, 'auc']:.3f}",
                       'share_of_deaths': f"{100 * s['excluded'].iloc[0] / (s['excluded'].iloc[0] + s['prevalence'].iloc[0] * s['n'].iloc[0]):.1f}",
                       **{m: (pts(d.loc[m].delta), sig(d.loc[m].CI95_low, d.loc[m].CI95_high))
                          for m in (DEF, FIX, LRM)}}

# ------------------------------------------------- missing eICU discharge status
_s = pd.read_csv(HERE / 'out_ext4' / 'nomissing_eicu72_summary.csv').set_index('method')
_d = pd.read_csv(HERE / 'out_ext4' / 'nomissing_eicu72_deltas.csv')
_d = _d[_d.metric == 'auc'].set_index('method')
F['nomiss'] = {'excluded': int(_s['excluded'].iloc[0]), 'single_auc': f"{_s.loc[SINGLE, 'auc']:.3f}",
               **{m: (pts(_d.loc[m].delta), sig(_d.loc[m].CI95_low, _d.loc[m].CI95_high))
                  for m in (DEF, FIX, LRM)}}

if __name__ == '__main__':
    for k, v in F.items():
        print(f'{k}: {v}')
