"""Runner for Step 27 (extensions). Same pattern as runner.py: notebook cells
are exec'd verbatim for definitions, data come from caches, one process per unit,
skip-if-done markers, outputs in local_run/out_ext/.

  python runner2.py build2            # 1-day + 2-day landmarks, unclipped matrices
  python runner2.py check             # engine must reproduce Step 22 exactly
  python runner2.py unit <name>
  python runner2.py all [--jobs N]
"""
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / 'research02_local.ipynb'
CACHE1 = HERE / 'cache' / 'data_cache.pkl'
CACHE2 = HERE / 'cache' / 'data_cache2.pkl'
OUT = Path(os.environ.setdefault('EHRML_OUT_DIR', str(HERE / 'out_ext')))
for d in (OUT, OUT / '_done', OUT / 'logs'):
    d.mkdir(parents=True, exist_ok=True)
cells = [''.join(c['source']) for c in json.load(open(NB))['cells']]
TASKS = ['mimic24', 'eicu72', 'los7_d2', 'los14_d2']


def log(msg):
    print(f'[runner2 {time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run_cells(ns, idx):
    for i in idx:
        exec(compile(cells[i], f'<cell {i}>', 'exec'), ns)


def build2():
    """Rebuild from raw data (read-only) and cache the landmark matrices for
    t = 1 and 2 days, clipped and unclipped. Calls are the notebook's own."""
    ns = {'__name__': '__main__'}
    run_cells(ns, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14])
    exec("los_all = los_landmark_tasks(icd_cohort, landmarks=(1, 2))", ns)
    raw = {}
    for wa in (1, 2):
        dm = ns['build_dm'](ns['icd_cohort'], cov_thresh=0.80, label=f'ICD LOS landmark {wa}d',
                            window_before_days=0, window_after_days=wa, anchor='intime')
        raw[wa] = dm[dm['los'] > wa].reset_index(drop=True)
    old = pickle.load(open(CACHE1, 'rb'))['los_tasks']
    import pandas as pd
    for (n0, d0, _), (n1, d1, _) in zip(old, [t for t in ns['los_all'] if '2d' in t[0]]):
        assert n0 == n1
        pd.testing.assert_frame_equal(d0, d1)          # 2-day landmark identical to Step 26
    pickle.dump({'los_all': ns['los_all'], 'los_raw': raw}, open(CACHE2, 'wb'))
    log('build2 ok: ' + ', '.join(f'{t[0]} n={len(t[1])} prev={t[1][t[2]].mean():.3f}'
                                  for t in ns['los_all']))


def unit_ns():
    ns = {'__name__': '__main__'}
    run_cells(ns, [1, 8, 12, 14])
    src = cells[9]
    exec(src[:src.index('dm_icd_24h_hosp_qc =')], ns)            # qc_clip_features
    exec(open(HERE / 'step27_ext.py').read(), ns)
    d1 = pickle.load(open(CACHE1, 'rb'))
    d2 = pickle.load(open(CACHE2, 'rb'))
    ns.update(d1)
    # apply_cov_thresh lives in the driver cell; use its exact definition
    drv = cells[16]
    a = drv.index('def apply_cov_thresh'); b = drv.index('dm_eicu_72h_full = dm_eicu_72h.copy()')
    exec(drv[a:b], ns)
    exec("dm_eicu_72h_raw = apply_cov_thresh(dm_eicu_72h_full, 0.80)\n"
         "dm_eicu_72h_qc = qc_clip_features(dm_eicu_72h_raw, META_COLS)", ns)
    # MIMIC 24 h unclipped: same rows/order and columns as the clipped matrix
    qc = ns['dm_icd_24h_hosp_qc']
    full = ns['dm_mimic24_full'].set_index('hadm_id')
    assert qc['hadm_id'].isin(full.index).all()
    raw = full.loc[qc['hadm_id'].values].reset_index()[qc.columns]
    ns['dm_mimic24_raw'] = raw
    import pandas as pd
    pd.testing.assert_frame_equal(ns['qc_clip_features'](raw, ns['META_COLS']).reset_index(drop=True),
                                  qc.reset_index(drop=True), check_dtype=False)
    T = {'mimic24': ('MIMIC-IV 24 h mortality', qc, raw, 'mortality'),
         'eicu72': ('eICU 72 h mortality', ns['dm_eicu_72h_qc'], ns['dm_eicu_72h_raw'], 'mortality')}
    for name, dmq, tgt in d2['los_all']:
        wa = 1 if '1d' in name else 2
        key = ('los7' if 'LOS>7' in name else 'los14') + f'_d{wa}'
        dmr = d2['los_raw'][wa]
        pd.testing.assert_frame_equal(ns['qc_clip_features'](dmr, ns['META_COLS']), dmq,
                                      check_dtype=False)
        T[key] = (f'ICU LOS>{7 if "LOS>7" in name else 14} d (landmark {wa} d)', dmq, dmr, tgt)
    ns['TASKS'] = T
    return ns


def save(prefix, frames):
    for kind, df in frames.items():
        df.to_csv(OUT / f'{prefix}_{kind}.csv', index=False)


def group_by_patient(ns, dm):
    """eICU has several ICU stays per patient (12,688 stays, 10,842 patients). Replace the
    outer RepeatedStratifiedKFold used by all step functions with a patient-grouped version
    (StratifiedGroupKFold, reshuffled per repeat), so no patient is in both training and
    test folds. Only the fold assignment changes."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    groups = dm['subject_id'].astype(str).values

    class GroupedRSKF:
        def __init__(self, n_splits=5, n_repeats=3, random_state=None):
            self.n_splits, self.n_repeats, self.random_state = n_splits, n_repeats, random_state

        def split(self, X, y=None, groups_=None):
            assert len(X) == len(groups), 'grouping vector does not match the matrix'
            for r in range(self.n_repeats):
                cv = StratifiedGroupKFold(self.n_splits, shuffle=True,
                                          random_state=(self.random_state or 0) + r)
                for tr, va in cv.split(np.zeros(len(groups)), np.asarray(y), groups):
                    assert not set(groups[tr]) & set(groups[va])
                    yield tr, va

    ns['RepeatedStratifiedKFold'] = GroupedRSKF
    ns['GROUPS'] = groups                      # inner splits and tuning searches grouped too
    log(f'patient-grouped outer folds: {len(groups)} rows, {len(set(groups))} patients')


def early_death_mask(key, dm):
    """True for episodes whose in-hospital death falls inside the feature window
    (MIMIC-IV: 24 h after hospital admission; eICU: 72 h after ICU admission).
    Read-only use of the raw tables."""
    import pandas as pd
    root = HERE.parent.parent
    if key == 'mimic24':
        a = pd.read_csv(root / 'mimic-iv-3.1' / 'hosp' / 'admissions.csv.gz',
                        usecols=['hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'])
        a = a[a.hadm_id.isin(dm.hadm_id)]
        t_death = pd.to_datetime(a.deathtime).fillna(pd.to_datetime(a.dischtime))
        hours = (t_death - pd.to_datetime(a.admittime)).dt.total_seconds() / 3600
        early = set(a.hadm_id[(a.hospital_expire_flag == 1) & (hours <= 24)])
    else:
        p = pd.read_csv(HERE.parent / 'eicu-collaborative-research-database-2.0' / 'patient.csv.gz',
                        usecols=['patientunitstayid', 'hospitaldischargeoffset', 'hospitaldischargestatus'])
        p = p[p.patientunitstayid.isin(dm.hadm_id)]
        early = set(p.patientunitstayid[(p.hospitaldischargestatus.astype(str).str.lower() == 'expired') &
                                        (p.hospitaldischargeoffset <= 72 * 60)])
    return dm.hadm_id.isin(early).values


def run_unit(name):
    ns = unit_ns()
    T = ns['TASKS']
    kind, _, key = name.partition(':')
    if key.startswith('eicu'):
        group_by_patient(ns, T[key][1])
    if kind == 'nodeath':
        label, dmq, dmr, tgt = T[key]
        m = early_death_mask(key, dmq)
        log(f'nodeath {key}: excluding {m.sum()} of {len(m)} episodes '
            f'({m.sum() / max(1, dmq[tgt].sum()):.1%} of deaths)')
        dmq = dmq[~m].reset_index(drop=True)
        if key.startswith('eicu'):
            group_by_patient(ns, dmq)
        res, ntr, nte, div = ns['compare'](dmq, tgt, roster='original', n_repeats=3)
        s, d, f = ns['summarise'](res, ntr, nte, 'XGB single (fixed config)',
                                  label + ' (early deaths excluded)', 'original')
        s['n'] = len(dmq); s['excluded'] = int(m.sum()); s['prevalence'] = dmq[tgt].mean()
        save(f'nodeath_{key}', {'summary': s, 'deltas': d, 'folds': f})
        return
    if kind == 'nomissing':
        # eICU stays with no recorded hospital discharge status are labelled survivors by the
        # original pipeline; this check drops them (read-only use of patient.csv).
        import pandas as pd
        label, dmq, dmr, tgt = T[key]
        p = pd.read_csv(HERE.parent / 'eicu-collaborative-research-database-2.0' / 'patient.csv.gz',
                        usecols=['patientunitstayid', 'hospitaldischargestatus'])
        known = set(p.patientunitstayid[p.hospitaldischargestatus.notna()])
        m = ~dmq.hadm_id.isin(known).values
        log(f'nomissing {key}: dropping {m.sum()} stays with missing discharge status')
        dmq = dmq[~m].reset_index(drop=True)
        group_by_patient(ns, dmq)
        res, ntr, nte, div = ns['compare'](dmq, tgt, roster='original', n_repeats=3)
        s, d, f = ns['summarise'](res, ntr, nte, 'XGB single (fixed config)',
                                  label + ' (missing discharge status excluded)', 'original')
        s['n'] = len(dmq); s['excluded'] = int(m.sum())
        save(f'nomissing_{key}', {'summary': s, 'deltas': d, 'folds': f})
        return
    if kind == 'cov':
        sw = ns['coverage_sweep'](ns['dm_eicu_72h_full'], 'mortality', name='eICU 72h', n_repeats=3,
                                  thresholds=(0.5, 0.6, 0.7, 0.8, 0.9))
        sw.to_csv(OUT / 'cov_eicu72.csv', index=False)
        return
    if kind == 'shapstab':
        r = ns['shap_stability_and_nested_selection'](T[key][1], 'mortality', n_repeats=2)
        r['selection'].to_csv(OUT / f'shapstab_{key}_selection.csv', index=False)
        json.dump({'mean_jaccard': r['mean_jaccard'], 'min_jaccard': r['min_jaccard']},
                  open(OUT / f'shapstab_{key}_jaccard.json', 'w'))
        return
    if kind in ('main', 'div', 'cat5', 'catmlp', 'tuned'):
        label, dmq, _, tgt = T[key]
        roster = {'main': 'original', 'div': 'diverse', 'cat5': 'cat5',
                  'catmlp': 'cat_for_mlp', 'tuned': 'original'}[kind]
        tuned = kind == 'tuned'
        res, ntr, nte, div = ns['compare'](dmq, tgt, roster=roster, n_repeats=3, tuned=tuned,
                                           with_cat_single=(kind == 'main'))
        ref = 'XGB single (tuned)' if tuned else 'XGB single (fixed config)'
        s, d, f = ns['summarise'](res, ntr, nte, ref, label, roster + ('+tuned' if tuned else ''))
        if tuned:   # also against the fixed-config single model
            _, d2, _ = ns['summarise'](res, ntr, nte, 'XGB single (fixed config)', label,
                                       roster + '+tuned')
            d = pd_concat([d, d2])
        dv = div.to_frame().T; dv.insert(0, 'task', label); dv.insert(1, 'roster', roster)
        save(f'{kind}_{key}', {'summary': s, 'deltas': d, 'folds': f, 'diversity': dv})
    elif kind == 'abl':
        label, dmq, dmr, tgt = T[key]
        a = ns['ablation_scores'](dmq, dmr, tgt, n_repeats=3)
        a.insert(0, 'task', label)
        a.to_csv(OUT / f'abl_{key}.csv', index=False)
    elif kind == 'n729':
        import pandas as pd
        label, dmq, _, tgt = T['mimic24']
        S, D = [], []
        for dr in range(10):
            sub = ns['stratified_subsample'](dmq, tgt, 729, seed=1000 + dr)
            res, ntr, nte, _ = ns['compare'](sub, tgt, roster='original', n_repeats=3, verbose=False)
            s, d, _ = ns['summarise'](res, ntr, nte, 'XGB single (fixed config)', label, 'original')
            s.insert(0, 'draw', dr); d.insert(0, 'draw', dr)
            S.append(s); D.append(d)
            log(f'n729 draw {dr + 1}/10 done')
        save('n729', {'summary': pd.concat(S), 'deltas': pd.concat(D)})
    elif kind == 'shap':
        shap_unit(ns)
    else:
        raise ValueError(name)


def pd_concat(x):
    import pandas as pd
    return pd.concat(x, ignore_index=True)


def shap_unit(ns):
    """Global SHAP for the fixed-config XGBoost fitted on each full cohort, and
    the Spearman rank correlation of mean |SHAP| over features shared by both."""
    import numpy as np
    import pandas as pd
    import shap
    from scipy import stats
    from sklearn.impute import SimpleImputer
    out = {}
    for key in ('mimic24', 'eicu72'):
        label, dmq, _, tgt = ns['TASKS'][key]
        feats = ns['_feat_cols'](dmq)
        imp = SimpleImputer(strategy='median').fit(dmq[feats].values.astype(float))
        X = imp.transform(dmq[feats].values.astype(float))
        m = ns['_xgb'](ns['RANDOM_STATE']).fit(X, dmq[tgt].values.astype(int))
        sv = np.asarray(shap.TreeExplainer(m).shap_values(X))
        imp_abs = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
        std_names = ns['standardise_cols'](pd.DataFrame(columns=feats), ns['META_COLS']).columns
        # standardise_cols maps 'platelets' -> 'plateletss' (alias applied twice), so eICU
        # platelets never matched MIMIC 'platelet' -> 'platelets'. Corrected here.
        std_names = [c.replace('plateletss', 'platelets') for c in std_names]
        out[key] = pd.DataFrame({'feature': imp_abs.index,
                                 'feature_std': [dict(zip(feats, std_names))[f] for f in imp_abs.index],
                                 'mean_abs_shap': imp_abs.values})
        out[key].to_csv(OUT / f'shap_global_{key}.csv', index=False)
    a = out['mimic24'].set_index('feature_std')['mean_abs_shap']
    b = out['eicu72'].set_index('feature_std')['mean_abs_shap']
    shared = sorted(set(a.index) & set(b.index))
    rho, p = stats.spearmanr(a[shared], b[shared])
    n = len(shared)
    z, se = np.arctanh(rho), 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    json.dump({'n_shared_features': n, 'spearman_rho': float(rho), 'p_value': float(p),
               'ci95_fisher_z': [float(lo), float(hi)], 'shared_features': shared},
              open(OUT / 'shap_crosscohort.json', 'w'), indent=1)
    log(f'shap: {n} shared features, Spearman rho {rho:.3f} (95% CI {lo:.3f} to {hi:.3f})')


UNITS = ([f'tuned:{t}' for t in ('eicu72', 'mimic24', 'los7_d2', 'los14_d2')] +
         [f'{k}:{t}' for k in ('catmlp', 'cat5', 'div', 'main') for t in ('eicu72',)] +
         ['n729'] +
         [f'{k}:{t}' for k in ('catmlp', 'cat5', 'div', 'main')
          for t in ('mimic24', 'los7_d2', 'los14_d2')] +
         [f'main:{t}' for t in ('los7_d1', 'los14_d1')] +
         [f'abl:{t}' for t in TASKS] + ['shap'])


UNITS2 = (['tuned:eicu72', 'catmlp:eicu72', 'cat5:eicu72', 'div:eicu72', 'main:eicu72',
           'nodeath:eicu72', 'nodeath:mimic24', 'abl:eicu72', 'cov:eicu72', 'shapstab:eicu72'])


def check():
    """compare() with the original roster must reproduce Step 22 (MIMIC 24 h)."""
    import numpy as np
    import pandas as pd
    ns = unit_ns()
    label, dmq, _, tgt = ns['TASKS']['mimic24']
    res, ntr, nte, div = ns['compare'](dmq, tgt, roster='original', n_repeats=3, verbose=False)
    old = pd.read_csv(HERE / 'out_full' / 'step22_original_summary.csv')
    old = old[old.task.str.startswith('MIMIC-IV 24h')].set_index('method')['AUROC']
    name = {'XGB single (fixed config)': 'XGB single (all features)'}
    rows = []
    for m, d in res.items():
        o = old.get(name.get(m, m), np.nan)
        rows.append((m, np.mean(d['auc']), o, abs(np.mean(d['auc']) - o)))
    df = pd.DataFrame(rows, columns=['method', 'new', 'step22', 'absdiff'])
    print(df.to_string(index=False))
    assert df['absdiff'].max() < 5e-4, 'engine does not reproduce Step 22'
    log('check passed: max |diff| = %.2e' % df['absdiff'].max())


def run_all(jobs):
    t0 = time.time()
    threads = str(max(1, 24 // jobs))
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS=threads,
               LOKY_MAX_CPU_COUNT=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    units = UNITS2 if os.environ.get('EHRML_UNITS2') == '1' else UNITS
    if os.environ.get('EHRML_UNITS3') == '1':      # eICU units that use inner splits
        units = ['tuned:eicu72', 'div:eicu72', 'cat5:eicu72', 'catmlp:eicu72', 'main:eicu72', 'nodeath:eicu72']
    todo = [u for u in units if not (OUT / '_done' / u.replace(':', '_')).exists()]
    log(f'{len(todo)} units to run, jobs={jobs}, threads={threads}')
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            u = todo.pop(0)
            p = subprocess.Popen([sys.executable, '-u', __file__, 'unit', u], env=env,
                                 stdout=open(OUT / 'logs' / f'{u.replace(":", "_")}.log', 'w'),
                                 stderr=subprocess.STDOUT)
            running[u] = (p, time.time())
            log(f'start {u}')
        time.sleep(15)
        for u, (p, ts) in list(running.items()):
            if p.poll() is not None:
                del running[u]
                if p.returncode == 0:
                    (OUT / '_done' / u.replace(':', '_')).touch()
                    log(f'done  {u} in {time.time() - ts:.0f}s')
                else:
                    log(f'FAIL  {u} exit {p.returncode}')
    log(f'all finished in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    os.chdir(HERE)
    cmd = sys.argv[1]
    if cmd == 'build2':
        build2()
    elif cmd == 'check':
        check()
    elif cmd == 'unit':
        run_unit(sys.argv[2])
    elif cmd == 'all':
        run_all(int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 8)
