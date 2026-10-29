# =============================================================================
# STEP 22: CONTROLLED ARCHITECTURE COMPARISON (reviewer response)
# -----------------------------------------------------------------------------
# Addresses reviewer points:
#   * tuning parity            -> every XGBoost in this step (single model,
#                                 Level-1 member, Level-2 meta-learner) uses ONE
#                                 fixed config; no grid search anywhere.
#   * meta-learner unspecified -> five Level-2 options compared on IDENTICAL
#                                 Level-1 outputs: XGB (library defaults, as in
#                                 the original run), XGB (same fixed config),
#                                 logistic regression, simple mean, best-single.
#   * Level-1 diversity        -> diversity table + an optional 'diverse' roster.
#   * effect sizes / CIs       -> paired deltas with Nadeau-Bengio corrected
#                                 95% CIs (valid for overlapping CV folds).
#   * stacked vs single were scored under different CV protocols before; here
#     every method is scored on the SAME outer folds.
# Requires from earlier cells: get_subsets, META_COLS, RANDOM_STATE.
# =============================================================================
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# One shared, fixed, regularised config for every XGBoost used in this step.
XGB_CFG = dict(n_estimators=150, max_depth=3, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8,
               eval_metric='logloss', verbosity=0, n_jobs=-1)


def _feat_cols(dm):
    """Numeric, non-meta columns only (robust to stray string columns)."""
    return [c for c in dm.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(dm[c])]


def _xgb(seed):
    return XGBClassifier(random_state=seed, **XGB_CFG)


def _make_l1(algo, seed):
    """Level-1 model factory. Imputation (and scaling where needed) is fitted
    inside each pipeline, i.e. on training data only."""
    imp = SimpleImputer(strategy='median')
    if algo == 'XGB':
        return make_pipeline(imp, _xgb(seed))
    if algo == 'LGBM':
        return make_pipeline(imp, LGBMClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1,
            random_state=seed))
    if algo == 'LR':
        return make_pipeline(imp, StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=1000))
    if algo == 'MLP':
        return make_pipeline(imp, StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 32), max_iter=300, early_stopping=True,
            random_state=seed))
    if algo == 'RF':
        return make_pipeline(imp, RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed))
    if algo == 'KNN':
        return make_pipeline(imp, StandardScaler(),
                             KNeighborsClassifier(n_neighbors=25))
    if algo == 'GNB':
        return make_pipeline(imp, StandardScaler(), GaussianNB())
    raise ValueError(algo)


ROSTERS = {
    'original': ['XGB', 'LR', 'LGBM', 'MLP'],            # as in the replicated pipeline
    'diverse':  ['XGB', 'LR', 'KNN', 'RF', 'GNB', 'MLP'],  # adds non-boosted families
}


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def nb_ci(deltas, n_train, n_test, alpha=0.05):
    """Nadeau-Bengio corrected resampled t-interval for the mean of paired
    per-fold differences (accounts for overlapping training sets)."""
    d = np.asarray(deltas, float)
    k = len(d)
    m = d.mean()
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / k + n_test / n_train) * var)
    if se == 0 or not np.isfinite(se):
        return m, m, m, np.nan
    t = stats.t.ppf(1 - alpha / 2, k - 1)
    p = 2 * (1 - stats.t.cdf(abs(m / se), k - 1))
    return m, m - t * se, m + t * se, p


def arch_compare(dm, target_col, roster='original', n_splits=5, n_repeats=3,
                 inner_splits=3, seed=None, verbose=True):
    """Score single models and several Level-2 options on identical folds.

    Returns (summary_df, delta_df, diversity_df, raw_dict).
    delta_df compares every method with 'XGB single (all features)'.
    """
    seed = RANDOM_STATE if seed is None else seed
    feat_cols = _feat_cols(dm)
    X_all = dm[feat_cols].values.astype(float)
    y = dm[target_col].values.astype(int).ravel()
    subsets = {k: v.values.astype(float) for k, v in get_subsets(dm[feat_cols]).items()}
    algos = ROSTERS[roster]
    keys = [(a, s) for s in subsets for a in algos]
    key_names = [f'{a}_{s}' for a, s in keys]

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    methods = ['XGB single (all features)', 'LR single (all features)',
               'Stack, XGB meta (library defaults)',
               'Stack, XGB meta (fixed config)', 'Stack, LR meta',
               'Mean of Level-1', 'Best single Level-1 (picked in-fold)']
    res = {m: {'auc': [], 'brier': [], 'ece': []} for m in methods}
    div_rows = []
    n_tr_list, n_te_list = [], []
    t0 = time.time()

    for f, (tr, va) in enumerate(rskf.split(X_all, y)):
        ytr, yva = y[tr], y[va]
        n_tr_list.append(len(tr)); n_te_list.append(len(va))
        inner = list(StratifiedKFold(inner_splits, shuffle=True,
                                     random_state=seed + f).split(np.zeros(len(tr)), ytr))
        M_tr = np.zeros((len(tr), len(keys)))
        M_va = np.zeros((len(va), len(keys)))
        for j, (algo, sub) in enumerate(keys):
            Xs = subsets[sub]
            Xtr_s, Xva_s = Xs[tr], Xs[va]
            for itr, iva in inner:
                m = _make_l1(algo, seed).fit(Xtr_s[itr], ytr[itr])
                M_tr[iva, j] = m.predict_proba(Xtr_s[iva])[:, 1]
            m = _make_l1(algo, seed).fit(Xtr_s, ytr)
            M_va[:, j] = m.predict_proba(Xva_s)[:, 1]

        preds = {}
        # single models on all features
        preds['XGB single (all features)'] = _make_l1('XGB', seed).fit(
            X_all[tr], ytr).predict_proba(X_all[va])[:, 1]
        preds['LR single (all features)'] = _make_l1('LR', seed).fit(
            X_all[tr], ytr).predict_proba(X_all[va])[:, 1]
        # Level-2 options on identical Level-1 outputs
        meta_default = XGBClassifier(eval_metric='logloss', verbosity=0,
                                     random_state=seed, n_jobs=-1)  # library defaults
        preds['Stack, XGB meta (library defaults)'] = meta_default.fit(
            M_tr, ytr).predict_proba(M_va)[:, 1]
        preds['Stack, XGB meta (fixed config)'] = _xgb(seed).fit(
            M_tr, ytr).predict_proba(M_va)[:, 1]
        preds['Stack, LR meta'] = LogisticRegression(C=1.0, max_iter=1000).fit(
            _logit(M_tr), ytr).predict_proba(_logit(M_va))[:, 1]
        preds['Mean of Level-1'] = M_va.mean(axis=1)
        oof_auc = [roc_auc_score(ytr, M_tr[:, j]) for j in range(M_tr.shape[1])]
        preds['Best single Level-1 (picked in-fold)'] = M_va[:, int(np.argmax(oof_auc))]

        for mth, p in preds.items():
            res[mth]['auc'].append(roc_auc_score(yva, p))
            res[mth]['brier'].append(brier_score_loss(yva, p))
            res[mth]['ece'].append(_ece(yva, p))

        # diversity of Level-1 outputs on the validation fold
        C = np.corrcoef(M_va.T)
        same_algo, diff_algo = [], []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                (ai, si), (aj, sj) = keys[i], keys[j]
                if ai == aj:
                    same_algo.append(C[i, j])
                elif si == sj:
                    diff_algo.append(C[i, j])
        iu = np.triu_indices(len(keys), 1)
        div_rows.append({'fold': f, 'mean_corr_all': np.nanmean(C[iu]),
                         'mean_corr_same_algo_diff_subset': np.nanmean(same_algo),
                         'mean_corr_diff_algo_same_subset': np.nanmean(diff_algo)})
        if verbose:
            print(f'  fold {f + 1}/{n_splits * n_repeats} done '
                  f'({time.time() - t0:.0f}s)', flush=True)

    # ---- summaries -----------------------------------------------------------
    n_tr, n_te = float(np.mean(n_tr_list)), float(np.mean(n_te_list))
    rows, drows = [], []
    ref = np.array(res['XGB single (all features)']['auc'])
    for mth in methods:
        a = np.array(res[mth]['auc'])
        rows.append({'method': mth, 'roster': roster,
                     'AUROC': a.mean(), 'AUROC_sd': a.std(ddof=1),
                     'Brier': np.mean(res[mth]['brier']),
                     'ECE': np.mean(res[mth]['ece'])})
        if mth != 'XGB single (all features)':
            m, lo, hi, p = nb_ci(a - ref, n_tr, n_te)
            drows.append({'method': mth, 'roster': roster,
                          'delta_AUROC_vs_XGB_single': m, 'CI95_low': lo,
                          'CI95_high': hi, 'p_corrected_t': p,
                          'n_scores': len(a)})
    summary = pd.DataFrame(rows)
    deltas = pd.DataFrame(drows)
    diversity = pd.DataFrame(div_rows).drop(columns='fold').mean().to_frame('value')
    return summary, deltas, diversity, res


def run_step22(tasks, roster='original', n_repeats=3, out_prefix='step22'):
    """tasks: list of (name, dm_qc, target_col)."""
    all_sum, all_del, all_div = [], [], {}
    for name, dm, tgt in tasks:
        label = name.replace('\n', ' ')
        print('=' * 70); print(f'STEP 22 | {label} | roster={roster}'); print('=' * 70)
        s, d, v, _ = arch_compare(dm, tgt, roster=roster, n_repeats=n_repeats)
        s.insert(0, 'task', label); d.insert(0, 'task', label)
        all_sum.append(s); all_del.append(d); all_div[label] = v['value']
        print(s.round(4).to_string(index=False))
        print(d.round(4).to_string(index=False))
        print(v.round(3).T.to_string())
    S = pd.concat(all_sum); D = pd.concat(all_del); V = pd.DataFrame(all_div).T
    try:
        S.to_csv(OUT_DIR / f'{out_prefix}_{roster}_summary.csv', index=False)
        D.to_csv(OUT_DIR / f'{out_prefix}_{roster}_deltas.csv', index=False)
        V.to_csv(OUT_DIR / f'{out_prefix}_{roster}_diversity.csv')
    except NameError:
        pass
    return S, D, V
