# =============================================================================
# STEPS 23-26 (reviewer response). Requires Step 22 cell to have been run
# (uses XGB_CFG, _xgb, nb_ci, arch_compare) plus get_subsets, META_COLS,
# RANDOM_STATE from earlier cells.
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


def stratified_subsample(dm, target_col, n, seed=0):
    if n >= len(dm):
        return dm.copy()
    keep, _ = train_test_split(np.arange(len(dm)), train_size=n,
                               stratify=dm[target_col].values, random_state=seed)
    return dm.iloc[np.sort(keep)].reset_index(drop=True)


def xgb_fold_scores(dm, target_col, cols=None, n_splits=5, n_repeats=3, seed=None):
    """Fixed-config XGBoost, AUROC on every outer fold (same folds for any `cols`,
    because the folds depend only on rows and labels)."""
    seed = RANDOM_STATE if seed is None else seed
    cols = cols if cols is not None else _feat_cols(dm)
    X = dm[cols].values.astype(float)
    y = dm[target_col].values.astype(int).ravel()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out, ntr, nte = [], [], []
    for tr, va in rskf.split(X, y):
        m = make_pipeline(SimpleImputer(strategy='median'), _xgb(seed)).fit(X[tr], y[tr])
        out.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        ntr.append(len(tr)); nte.append(len(va))
    return np.array(out), float(np.mean(ntr)), float(np.mean(nte))


# ------------------------------------------------------------------ STEP 23 --
def size_sensitivity(dm, target_col, n_target=729, n_draws=5, n_repeats=1,
                     roster='original'):
    """Re-run the stack-vs-single comparison on random stratified subsamples of
    the cohort at the ORIGINAL study's size, to test whether the finding depends
    on the replication cohort being ~3.5x larger."""
    rows = []
    for d in range(n_draws):
        sub = stratified_subsample(dm, target_col, n_target, seed=1000 + d)
        print(f'--- size sensitivity draw {d + 1}/{n_draws} (n={len(sub)}) ---')
        s, dl, _, _ = arch_compare(sub, target_col, roster=roster,
                                   n_repeats=n_repeats, verbose=False)
        s = s.set_index('method')
        base = s.loc['XGB single (all features)', 'AUROC']
        for m in s.index:
            rows.append({'draw': d, 'method': m, 'AUROC': s.loc[m, 'AUROC'],
                         'delta_vs_XGB_single': s.loc[m, 'AUROC'] - base,
                         'ECE': s.loc[m, 'ECE']})
    R = pd.DataFrame(rows)
    agg = (R.groupby('method')
             .agg(AUROC_mean=('AUROC', 'mean'), AUROC_sd=('AUROC', 'std'),
                  delta_mean=('delta_vs_XGB_single', 'mean'),
                  delta_sd=('delta_vs_XGB_single', 'std'),
                  draws_below_single=('delta_vs_XGB_single', lambda x: int((x < 0).sum())),
                  ECE_mean=('ECE', 'mean')).reset_index())
    print(agg.round(4).to_string(index=False))
    return R, agg


# ------------------------------------------------------------------ STEP 24 --
def coverage_report(cov_series, name, bands=(0, .5, .6, .7, .8, .9, 1.0001)):
    """Print and plot the per-feature coverage distribution (pre-filter)."""
    cov = pd.Series(cov_series).dropna()
    cnt = pd.cut(cov, bins=list(bands), right=False).value_counts().sort_index()
    print(f'\nCoverage distribution, {name} (n features = {len(cov)})')
    print(cnt.to_string())
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.hist(cov.values, bins=np.linspace(0, 1, 21), color='#2E8B8B', edgecolor='white')
    for t in (0.7, 0.8, 0.9):
        ax.axvline(t, color='#B0431F', ls='--', lw=1)
    ax.set_xlabel('Fraction of episodes with a recorded value')
    ax.set_ylabel('Number of features')
    ax.set_title(f'Feature coverage, {name}', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    try:
        plt.savefig(OUT_DIR / f'coverage_hist_{name.replace(" ", "_")}.png', dpi=300)
    except NameError:
        pass
    plt.show()
    return cnt


def coverage_sweep(dm_full, target_col, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9),
                   ref=0.8, n_repeats=3, name=''):
    """dm_full must be the UNFILTERED feature matrix. Columns are filtered by
    coverage computed on it; rows are identical for every threshold, so the CV
    folds are identical and the comparison is paired."""
    feats = _feat_cols(dm_full)
    cov = dm_full[feats].notna().mean()
    scores, ns, ntr, nte = {}, {}, None, None
    for t in thresholds:
        cols = cov[cov >= t].index.tolist()
        if not cols:
            continue
        s, ntr, nte = xgb_fold_scores(dm_full, target_col, cols, n_repeats=n_repeats)
        scores[t], ns[t] = s, len(cols)
    rows = []
    for t, s in scores.items():
        if ref in scores and t != ref:
            m, lo, hi, p = nb_ci(s - scores[ref], ntr, nte)
        else:
            m, lo, hi, p = (0.0, 0.0, 0.0, np.nan)
        rows.append({'dataset': name, 'threshold': t, 'n_features': ns[t],
                     'AUROC': s.mean(), 'delta_vs_ref': m,
                     'CI95_low': lo, 'CI95_high': hi, 'p_corrected_t': p})
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    return out


# ------------------------------------------------------------------ STEP 25 --
def shap_stability_and_nested_selection(dm, target_col, topk=20, ks=(10, 20, 30, 40),
                                        n_splits=5, n_repeats=2, shap_rows=600, seed=None):
    """(a) SHAP feature-ranking stability across training folds.
       (b) Feature selection done INSIDE each training fold (ranking from the
           training fold only), so the selection never sees the validation fold.
       This replaces the earlier ranking that was computed once on all data."""
    import shap
    seed = RANDOM_STATE if seed is None else seed
    feats = _feat_cols(dm)
    X = dm[feats].values.astype(float)
    y = dm[target_col].values.astype(int).ravel()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    tops, res = [], {'all': []}
    for k in ks:
        res[k] = []
    ntr, nte = [], []
    rng = np.random.RandomState(seed)
    for f, (tr, va) in enumerate(rskf.split(X, y)):
        ntr.append(len(tr)); nte.append(len(va))
        imp = SimpleImputer(strategy='median').fit(X[tr])
        Xtr, Xva = imp.transform(X[tr]), imp.transform(X[va])
        m = _xgb(seed).fit(Xtr, y[tr])
        samp = rng.choice(len(tr), size=min(shap_rows, len(tr)), replace=False)
        sv = shap.TreeExplainer(m).shap_values(Xtr[samp])
        if isinstance(sv, list):
            sv = sv[-1]
        imp_abs = np.abs(np.asarray(sv)).mean(axis=0)
        order = np.argsort(-imp_abs)
        tops.append(set(order[:topk]))
        res['all'].append(roc_auc_score(y[va], m.predict_proba(Xva)[:, 1]))
        for k in ks:
            sel = order[:min(k, len(order))]
            mk = _xgb(seed).fit(Xtr[:, sel], y[tr])
            res[k].append(roc_auc_score(y[va], mk.predict_proba(Xva[:, sel])[:, 1]))
        print(f'  fold {f + 1}/{n_splits * n_repeats} done', flush=True)
    # stability
    jac = [len(a & b) / len(a | b) for a, b in combinations(tops, 2)]
    freq = pd.Series(0, index=feats, dtype=float)
    for s in tops:
        for i in s:
            freq.iloc[i] += 1
    freq = (freq / len(tops)).sort_values(ascending=False)
    print(f'\nTop-{topk} SHAP set stability across {len(tops)} training folds: '
          f'mean Jaccard = {np.mean(jac):.3f} (min {np.min(jac):.3f})')
    print('Selection frequency (share of folds with the feature in the top set):')
    print(freq.head(topk).round(2).to_string())
    # nested-selection performance
    rows = []
    base = np.array(res['all'])
    for k in ks:
        a = np.array(res[k])
        m_, lo, hi, p = nb_ci(a - base, np.mean(ntr), np.mean(nte))
        rows.append({'top_k': k, 'AUROC': a.mean(), 'AUROC_all_features': base.mean(),
                     'delta_vs_all': m_, 'CI95_low': lo, 'CI95_high': hi,
                     'p_corrected_t': p})
    sel_df = pd.DataFrame(rows)
    print('\nNested (leak-free) top-K selection vs all features:')
    print(sel_df.round(4).to_string(index=False))
    return {'mean_jaccard': float(np.mean(jac)), 'min_jaccard': float(np.min(jac)),
            'freq': freq, 'selection': sel_df}


# ------------------------------------------------------------------ STEP 26 --
def los_landmark_tasks(cohort, landmarks=(1, 2)):
    """Leakage-safe LOS tasks. At landmark day t (ICU admission + t days) we use
    ONLY data from [0, t] and keep ONLY patients still in the ICU at t
    (ICU LOS > t), so the outcome is not already determined and the feature
    window never overlaps the horizon being predicted."""
    tasks = []
    for wa in landmarks:
        dm = build_dm(cohort, cov_thresh=0.80, label=f'ICD LOS landmark {wa}d',
                      window_before_days=0, window_after_days=wa, anchor='intime')
        n0 = len(dm)
        dm = dm[dm['los'] > wa].reset_index(drop=True)
        print(f'  landmark {wa}d: {n0} -> {len(dm)} episodes still in ICU; '
              f'LOS>7 prevalence {dm["los_gt7"].mean():.1%}, '
              f'LOS>14 prevalence {dm["los_gt14"].mean():.1%}')
        dmq = qc_clip_features(dm, META_COLS)
        tasks.append((f'MIMIC LOS>7 (landmark {wa}d)', dmq, 'los_gt7'))
        tasks.append((f'MIMIC LOS>14 (landmark {wa}d)', dmq, 'los_gt14'))
    return tasks
