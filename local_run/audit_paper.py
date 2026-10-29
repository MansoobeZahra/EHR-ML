"""Independent audit of the reported results (no reuse of paper_facts.py logic).

1. Split integrity: outer and inner splits as used by the engine; patient separation in eICU;
   every training row gets exactly one out-of-fold Level-1 prediction.
2. Cohort / outcome / timestamp checks against the raw tables (read-only).
3. Statistics: recompute every paired AUROC difference and Nadeau-Bengio CI from the saved
   per-fold scores, rebuild the comparison inventory, and compare with the saved summaries and
   with the numbers printed in the paper.
Writes results/AUDIT_REPORT.md and results/COMPARISON_INVENTORY.csv.
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold)

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
EXT, EXT2, EXT3 = HERE / 'out_ext', HERE / 'out_ext2', HERE / 'out_ext3'
L = ['# Audit report (independent recomputation)\n']


def say(s=''):
    print(s); L.append(s)


# ------------------------------------------------------------------ 1. splits
say('## 1. Split integrity')
d1 = pickle.load(open(HERE / 'cache' / 'data_cache.pkl', 'rb'))
d2 = pickle.load(open(HERE / 'cache' / 'data_cache2.pkl', 'rb'))
META = ['subject_id', 'hadm_id']
ei = d1['dm_eicu_72h_full']
g = ei['subject_id'].astype(str).values
y = ei['mortality'].values.astype(int)
n_viol_outer = n_viol_inner = n_partition_bad = 0
fold = 0
for r in range(3):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42 + r)
    for tr, va in cv.split(np.zeros(len(g)), y, g):
        n_viol_outer += len(set(g[tr]) & set(g[va]))
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=42 + fold).split(
            np.zeros(len(tr)), y[tr], g[tr]))
        cover = np.zeros(len(tr), int)
        for a, b in inner:
            n_viol_inner += len(set(g[tr][a]) & set(g[tr][b]))
            cover[b] += 1
        n_partition_bad += int((cover != 1).sum())
        fold += 1
say(f'- eICU: {len(g)} stays, {len(set(g))} patients, {fold} outer folds (5-fold x 3, grouped).')
say(f'- Patients shared between outer train and test: {n_viol_outer}. Shared between inner train and '
    f'inner validation: {n_viol_inner}.')
say(f'- Training rows without exactly one out-of-fold Level-1 prediction: {n_partition_bad} '
    '(the meta-learner is fitted only on these out-of-fold predictions; Level-1 models used on the '
    'test fold are refitted on the whole training fold, as in standard stacking).')
mi = d1['dm_icd_24h_hosp_qc']
say(f"- MIMIC-IV mortality matrix: {len(mi)} rows, {mi['subject_id'].nunique()} patients "
    f"(one episode per patient, so row-level folds are patient-level folds).")
for name, dm, _ in d2['los_all']:
    say(f"- {name}: {len(dm)} rows, {dm['subject_id'].nunique()} patients.")

# ------------------------------------------------------------ 2. preprocessing
say('\n## 2. What is fitted on the full matrix (before cross-validation)')
say('- Fitted inside each training fold: median imputation, scaling, all models, hyperparameter '
    'searches, SHAP ranking and top-K selection (Step 25 fits the imputer, model and SHAP on the '
    'training fold and scores the selected features on the held-out fold).')
say('- Fitted on the full matrix: 1st/99th percentile clipping thresholds and the 80% coverage filter. '
    'Neither uses the outcome. Removing clipping changes AUROC by at most 0.2 points (ablation); the '
    'coverage filter only decides which columns exist.')
say('- Tuned analysis (P3): the Level-1 XGBoost/LR hyperparameters are chosen on the whole outer '
    'training fold before the inner out-of-fold predictions are made. This does not touch the test '
    'fold but can make the meta-learner inputs slightly optimistic; the original pipeline does the same.')

# ------------------------------------------------------------ 3. outcomes
say('\n## 3. Outcomes and timestamps (raw tables, read-only)')
root = HERE.parent.parent
adm = pd.read_csv(root / 'mimic-iv-3.1' / 'hosp' / 'admissions.csv.gz',
                  usecols=['hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'])
adm = adm[adm.hadm_id.isin(mi.hadm_id)]
m = mi[['hadm_id', 'mortality']].merge(adm, on='hadm_id', how='left')
say(f"- MIMIC-IV label = hospital_expire_flag: identical for all rows: "
    f"{bool((m.mortality == m.hospital_expire_flag).all())}; deaths {int(m.mortality.sum())} "
    f"({m.mortality.mean():.1%}).")
dt = pd.to_datetime(m.deathtime).fillna(pd.to_datetime(m.dischtime))
h = (dt - pd.to_datetime(m.admittime)).dt.total_seconds() / 3600
dead = m.mortality == 1
say(f"- MIMIC-IV deaths with deathtime recorded: {int(m.deathtime[dead].notna().sum())} of {int(dead.sum())}; "
    f"deaths within 24 h of hospital admission: {int((h[dead] <= 24).sum())}; median time to death "
    f"{h[dead].median():.0f} h.")
say('- MIMIC-IV features: chartevents/labevents with charttime in [admittime, admittime + 24 h] '
    '(build_dm, window_before_days=0, window_after_days=1, anchor=admittime).')
pt = pd.read_csv(HERE.parent / 'eicu-collaborative-research-database-2.0' / 'patient.csv.gz',
                 usecols=['patientunitstayid', 'hospitaldischargestatus', 'hospitaldischargeoffset',
                          'unitdischargeoffset'])
pt = pt[pt.patientunitstayid.isin(ei.hadm_id)]
e = ei[['hadm_id', 'mortality']].merge(pt, left_on='hadm_id', right_on='patientunitstayid', how='left')
status = e.hospitaldischargestatus.astype(str).str.lower()
say(f"- eICU label = hospitaldischargestatus 'Expired': identical: "
    f"{bool(((status == 'expired').astype(int) == e.mortality).all())}; missing discharge status: "
    f"{int(e.hospitaldischargestatus.isna().sum())} stays (labelled as survivors).")
ed = e[e.mortality == 1]
say(f"- eICU deaths within 72 h of ICU admission: {int((ed.hospitaldischargeoffset <= 4320).sum())} of "
    f"{len(ed)}; median time to death {ed.hospitaldischargeoffset.median() / 60:.0f} h.")
say('- eICU features: vitalPeriodic/lab offsets in [0, 72 h] from ICU admission.')

# ------------------------------------------------------------ 4. statistics
say('\n## 4. Statistics recomputed from per-fold scores')


def nb(d, ratio):
    k = len(d); mm = d.mean(); v = d.var(ddof=1)
    se = np.sqrt((1 / k + ratio) * v); t = stats.t.ppf(0.975, k - 1)
    return mm, mm - t * se, mm + t * se


def src(kind, key):
    if key.startswith('eicu') and kind in ('main', 'tuned', 'div', 'cat5', 'catmlp'):
        return EXT3
    return EXT


SETTINGS = ([('main', k) for k in ['mimic24', 'eicu72', 'los7_d2', 'los14_d2', 'los7_d1', 'los14_d1']] +
            [('tuned', k) for k in ['mimic24', 'eicu72', 'los7_d2', 'los14_d2']] +
            [(kind, k) for kind in ('div', 'cat5', 'catmlp') for k in ['mimic24', 'eicu72', 'los7_d2', 'los14_d2']])
rows, max_dev = [], 0.0
for kind, key in SETTINGS:
    f = pd.read_csv(src(kind, key) / f'{kind}_{key}_folds.csv')
    P = f.pivot(index='fold', columns='method', values='auc')
    saved = pd.read_csv(src(kind, key) / f'{kind}_{key}_deltas.csv')
    ref = 'XGB single (tuned)' if kind == 'tuned' else 'XGB single (fixed config)'
    comps = [('Stack, XGB meta (library defaults)', 'default')]
    comps += ([('Stack, XGB meta (tuned)', 'regularized'), ('Stack, LR meta (tuned C)', 'regularized')]
              if kind == 'tuned' else
              [('Stack, XGB meta (fixed config)', 'regularized'), ('Stack, LR meta', 'regularized')])
    for mth, group in comps:
        dlt = (P[mth] - P[ref]).values
        mm, lo, hi = nb(dlt, 0.25)
        sv = saved[(saved.method == mth) & (saved.reference == ref) & (saved.metric == 'auc')].iloc[0]
        max_dev = max(max_dev, abs(mm - sv.delta), abs(lo - sv.CI95_low), abs(hi - sv.CI95_high))
        verdict = 'worse' if hi < 0 else ('better' if lo > 0 else 'not significant')
        saved_verdict = 'worse' if sv.CI95_high < 0 else ('better' if sv.CI95_low > 0 else 'not significant')
        rows.append({'setting': f'{kind}:{key}', 'meta_learner': mth, 'group': group, 'reference': ref,
                     'n_folds': len(dlt), 'delta_points': round(100 * mm, 2),
                     'ci_low': round(100 * lo, 2), 'ci_high': round(100 * hi, 2),
                     'verdict': verdict, 'verdict_saved': saved_verdict})
inv = pd.DataFrame(rows)
inv.to_csv(RES / 'COMPARISON_INVENTORY.csv', index=False)
say(f'- Recomputed {len(inv)} comparisons from fold scores (ratio n_test/n_train = 1/4); largest '
    f'difference from the saved deltas/CIs: {max_dev:.5f} AUROC (grouped folds have slightly unequal '
    f'sizes). Verdicts identical to saved: {bool((inv.verdict == inv.verdict_saved).all())}.')
dfl, reg = inv[inv.group == 'default'], inv[inv.group == 'regularized']
say(f"- Default meta-learner: {len(dfl)} settings; worse {int((dfl.verdict == 'worse').sum())}, not "
    f"significant {int((dfl.verdict == 'not significant').sum())}, better {int((dfl.verdict == 'better').sum())}; "
    f"differences {dfl.delta_points.max():+.1f} to {dfl.delta_points.min():+.1f} points.")
say(f"- Regularized/tuned meta-learners: {len(reg)} comparisons; not significant "
    f"{int((reg.verdict == 'not significant').sum())}, worse {int((reg.verdict == 'worse').sum())}, "
    f"better {int((reg.verdict == 'better').sum())}.")
say('- Worse (regularized): ' + '; '.join(f"{r.setting} {r.meta_learner} {r.delta_points:+.1f}"
                                          for r in reg[reg.verdict == 'worse'].itertuples()))
say('- Not significant (default): ' + '; '.join(f"{r.setting} {r.delta_points:+.1f}"
                                                for r in dfl[dfl.verdict == 'not significant'].itertuples()))

# compare with the paper's printed counts
txt = (HERE.parent / 'paper' / 'PAPER_TEXT_v3.txt').read_text()
m17 = re.search(r'significantly worse in (\d+) of (\d+) settings', txt)
m39 = re.search(r'not significantly different from the single model in (\d+) of (\d+) comparisons, worse in (\d+)', txt)
say(f"- Paper abstract: default worse {m17.group(1)} of {m17.group(2)}; regularized not significant "
    f"{m39.group(1)} of {m39.group(2)}, worse {m39.group(3)}.")
ok = (int(m17.group(1)) == int((dfl.verdict == 'worse').sum()) and int(m17.group(2)) == len(dfl) and
      int(m39.group(1)) == int((reg.verdict == 'not significant').sum()) and int(m39.group(2)) == len(reg) and
      int(m39.group(3)) == int((reg.verdict == 'worse').sum()))
say(f'- Paper counts match the recomputed inventory: {ok}.')

# Table II cells recomputed
say('\n### Table II recomputed from fold scores (points, 95% CI; ECE)')
for key in ['mimic24', 'eicu72', 'los7_d2', 'los14_d2']:
    f = pd.read_csv(src('main', key) / f'main_{key}_folds.csv')
    P = f.pivot(index='fold', columns='method', values='auc'); E = f.pivot(index='fold', columns='method', values='ece')
    parts = [f"{key}: single {P['XGB single (fixed config)'].mean():.3f} (ECE {E['XGB single (fixed config)'].mean():.3f})"]
    for mth in ['Stack, XGB meta (library defaults)', 'Stack, XGB meta (fixed config)', 'Stack, LR meta',
                'Mean of Level-1', 'Best single Level-1 (picked in-fold)']:
        mm, lo, hi = nb((P[mth] - P['XGB single (fixed config)']).values, 0.25)
        parts.append(f"{mth.split('(')[0].strip()}: {100 * mm:+.2f} ({100 * lo:+.2f}, {100 * hi:+.2f}), ECE {E[mth].mean():.3f}")
    say('- ' + '; '.join(parts))
(RES / 'AUDIT_REPORT.md').write_text('\n'.join(L) + '\n')
print('\nwrote results/AUDIT_REPORT.md and results/COMPARISON_INVENTORY.csv')
