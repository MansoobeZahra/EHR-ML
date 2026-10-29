# =============================================================================
# STEP 27: extensions requested on 2026-10-22 (reviewer response, second pass)
# -----------------------------------------------------------------------------
# One evaluation engine, compare(), used for every new experiment so that all
# methods share identical outer folds. With tuned=False and the 'original'
# roster it repeats arch_compare (Step 22) exactly, which is checked before use.
# Additions over arch_compare:
#   * metrics: AUPRC, calibration slope and calibration-in-the-large
#   * CatBoost as a single model and as a Level-1 learner
#   * an equal, nested hyperparameter-search budget (tuned=True)
# Requires the Step 22 and Steps 23-26 cells (XGB_CFG, _xgb, _make_l1, _ece,
# _logit, nb_ci, _feat_cols, get_subsets, RANDOM_STATE, META_COLS).
# =============================================================================
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, StratifiedKFold)

# Patient identifiers aligned with the rows of the matrix being evaluated (set by the runner
# for eICU, which has several ICU stays per patient). When set, every inner split (Level-1
# out-of-fold predictions and all hyperparameter searches) is grouped by patient as well.
GROUPS = None


def inner_splits(y, groups, k, seed):
    z = np.zeros(len(y))
    if groups is None:
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(z, y))
    out = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(z, y, groups))
    for a, b in out:
        assert not set(groups[a]) & set(groups[b])
    return out
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# CatBoost: the fixed configuration of the original run (Step 16).
CAT_CFG = dict(iterations=150, depth=4, learning_rate=0.05, verbose=0)

# Equal search budget for every tuned XGBoost (single model, Level-1 reference,
# Level-2 meta-learner): 20 random configurations, 3-fold inner CV, AUROC.
TUNE_ITER = 20
XGB_SPACE = {'n_estimators': [100, 200, 300, 500],
             'max_depth': [2, 3, 4, 5, 6],
             'learning_rate': [0.01, 0.03, 0.05, 0.1],
             'subsample': [0.6, 0.8, 1.0],
             'colsample_bytree': [0.5, 0.8, 1.0],
             'min_child_weight': [1, 3, 5],
             'reg_lambda': [1, 5, 10]}
LR_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

ROSTERS.update({
    'cat5': ['XGB', 'LR', 'LGBM', 'MLP', 'CAT'],     # CatBoost added as a fifth learner
    'cat_for_mlp': ['XGB', 'LR', 'LGBM', 'CAT'],     # CatBoost replaces the MLP
})


def _threads():
    return int(os.environ.get('OMP_NUM_THREADS', '1'))


def _cat(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_seed=seed, thread_count=_threads(), **CAT_CFG)


# Thread counts must be explicit: n_jobs=-1 makes LightGBM use all 24 cores even
# when parallel units share the machine, which slows each fit 30-100x under load.
# Only the thread count changes; the model configuration is Step 22's.
XGB_CFG['n_jobs'] = _threads()


def make_l1(algo, seed, xgb_params=None, lr_C=None):
    """_make_l1 from Step 22, plus CatBoost and optional tuned XGB/LGBM/LR."""
    imp = SimpleImputer(strategy='median')
    if algo == 'LGBM' and xgb_params is None:
        return make_pipeline(imp, LGBMClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=_threads(),
            random_state=seed))
    if algo == 'CAT':
        return make_pipeline(imp, _cat(seed))
    if xgb_params is not None and algo == 'XGB':
        cfg = dict(XGB_CFG); cfg.update(xgb_params)
        return make_pipeline(imp, XGBClassifier(random_state=seed, **cfg))
    if xgb_params is not None and algo == 'LGBM':
        # as in the original pipeline, the tuned XGBoost values are passed to LightGBM
        p = {k: xgb_params[k] for k in ('n_estimators', 'max_depth', 'learning_rate',
                                        'subsample', 'colsample_bytree')}
        return make_pipeline(imp, LGBMClassifier(subsample_freq=1, verbose=-1,
                                                 n_jobs=_threads(), random_state=seed, **p))
    if lr_C is not None and algo == 'LR':
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(imp, StandardScaler(), LogisticRegression(C=lr_C, max_iter=1000))
    return _make_l1(algo, seed)


def tune_xgb_params(X, y, seed, groups=None):
    """Nested random search (inside the outer training fold only)."""
    pipe = make_pipeline(SimpleImputer(strategy='median'), _xgb(seed))
    space = {f'xgbclassifier__{k}': v for k, v in XGB_SPACE.items()}
    rs = RandomizedSearchCV(pipe, space, n_iter=TUNE_ITER, scoring='roc_auc',
                            cv=inner_splits(y, groups, 3, seed),
                            random_state=seed, n_jobs=1, refit=False)
    rs.fit(X, y)
    return {k.split('__', 1)[1]: v for k, v in rs.best_params_.items()}


def tune_lr_C(X, y, seed, logit_input=False, groups=None):
    best, best_c = -1, None
    splits = inner_splits(y, groups, 3, seed)
    for C in LR_C_GRID:
        sc = []
        for tr, va in splits:
            if logit_input:
                m = LogisticRegression(C=C, max_iter=1000).fit(_logit(X[tr]), y[tr])
                p = m.predict_proba(_logit(X[va]))[:, 1]
            else:
                m = make_l1('LR', seed, lr_C=C).fit(X[tr], y[tr])
                p = m.predict_proba(X[va])[:, 1]
            sc.append(roc_auc_score(y[va], p))
        if np.mean(sc) > best:
            best, best_c = np.mean(sc), C
    return best_c


def calib_slope_intercept(y, p):
    """Logistic recalibration: slope from y ~ logit(p); calibration-in-the-large
    from y ~ offset(logit(p)) (intercept only, Newton iterations)."""
    lp = _logit(p)
    slope = LogisticRegression(penalty=None, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]
    a = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(a + lp)))
        g, h = np.sum(y - q), np.sum(q * (1 - q))
        a += g / max(h, 1e-12)
        if abs(g) < 1e-9:
            break
    return slope, a


METRICS = ('auc', 'auprc', 'brier', 'ece', 'slope', 'citl')


def compare(dm, target_col, roster='original', n_splits=5, n_repeats=3, inner_k=3,
            tuned=False, with_cat_single=False, seed=None, verbose=True):
    """Score single models and Level-2 options on identical outer folds.

    tuned=False reproduces arch_compare (Step 22) and adds metrics.
    tuned=True gives every XGBoost the same nested random-search budget:
      single XGBoost on all features; the Level-1 XGBoost on the reference subset
      (vitals_avg, reused for all subsets and passed to LightGBM, as in the
      original pipeline); the Level-2 XGBoost on the Level-1 outputs. Logistic
      regression C is grid-searched for Level-1 and Level-2.
    Returns (scores: dict method -> dict metric -> list, n_tr, n_te, diversity).
    """
    seed = RANDOM_STATE if seed is None else seed
    G = GROUPS if (GROUPS is not None and len(GROUPS) == len(dm)) else None
    feat_cols = _feat_cols(dm)
    X_all = dm[feat_cols].values.astype(float)
    y = dm[target_col].values.astype(int).ravel()
    subsets = {k: v.values.astype(float) for k, v in get_subsets(dm[feat_cols]).items()}
    algos = ROSTERS[roster]
    keys = [(a, s) for s in subsets for a in algos]
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    res, div_rows, ntr, nte = {}, [], [], []
    t0 = time.time()
    for f, (tr, va) in enumerate(rskf.split(X_all, y)):
        ytr, yva = y[tr], y[va]
        ntr.append(len(tr)); nte.append(len(va))
        gtr = G[tr] if G is not None else None
        inner = inner_splits(ytr, gtr, inner_k, seed + f)
        xgb_p = lr_c = None
        if tuned:
            ref = subsets['vitals_avg'] if 'vitals_avg' in subsets else X_all
            xgb_p = tune_xgb_params(ref[tr], ytr, seed + f, gtr)
            lr_c = tune_lr_C(ref[tr], ytr, seed + f, groups=gtr)
        M_tr = np.zeros((len(tr), len(keys)))
        M_va = np.zeros((len(va), len(keys)))
        for j, (algo, sub) in enumerate(keys):
            Xs = subsets[sub]
            Xtr_s, Xva_s = Xs[tr], Xs[va]
            for itr, iva in inner:
                m = make_l1(algo, seed, xgb_p, lr_c).fit(Xtr_s[itr], ytr[itr])
                M_tr[iva, j] = m.predict_proba(Xtr_s[iva])[:, 1]
            m = make_l1(algo, seed, xgb_p, lr_c).fit(Xtr_s, ytr)
            M_va[:, j] = m.predict_proba(Xva_s)[:, 1]

        preds = {}
        preds['XGB single (fixed config)'] = _make_l1('XGB', seed).fit(
            X_all[tr], ytr).predict_proba(X_all[va])[:, 1]
        if not tuned:
            preds['LR single (all features)'] = _make_l1('LR', seed).fit(
                X_all[tr], ytr).predict_proba(X_all[va])[:, 1]
        if with_cat_single:
            preds['CatBoost single (fixed config)'] = make_l1('CAT', seed).fit(
                X_all[tr], ytr).predict_proba(X_all[va])[:, 1]
        meta_default = XGBClassifier(eval_metric='logloss', verbosity=0,
                                     random_state=seed, n_jobs=_threads())  # library defaults
        preds['Stack, XGB meta (library defaults)'] = meta_default.fit(
            M_tr, ytr).predict_proba(M_va)[:, 1]
        if tuned:
            p_single = tune_xgb_params(X_all[tr], ytr, seed + f, gtr)
            preds['XGB single (tuned)'] = make_l1('XGB', seed, p_single).fit(
                X_all[tr], ytr).predict_proba(X_all[va])[:, 1]
            p_meta = tune_xgb_params(M_tr, ytr, seed + f, gtr)
            cfg = dict(XGB_CFG); cfg.update(p_meta)
            preds['Stack, XGB meta (tuned)'] = XGBClassifier(random_state=seed, **cfg).fit(
                M_tr, ytr).predict_proba(M_va)[:, 1]
            c_meta = tune_lr_C(M_tr, ytr, seed + f, logit_input=True, groups=gtr)
            preds['Stack, LR meta (tuned C)'] = LogisticRegression(C=c_meta, max_iter=1000).fit(
                _logit(M_tr), ytr).predict_proba(_logit(M_va))[:, 1]
        else:
            preds['Stack, XGB meta (fixed config)'] = _xgb(seed).fit(
                M_tr, ytr).predict_proba(M_va)[:, 1]
            preds['Stack, LR meta'] = LogisticRegression(C=1.0, max_iter=1000).fit(
                _logit(M_tr), ytr).predict_proba(_logit(M_va))[:, 1]
        preds['Mean of Level-1'] = M_va.mean(axis=1)
        oof_auc = [roc_auc_score(ytr, M_tr[:, j]) for j in range(M_tr.shape[1])]
        preds['Best single Level-1 (picked in-fold)'] = M_va[:, int(np.argmax(oof_auc))]

        for mth, p in preds.items():
            d = res.setdefault(mth, {k: [] for k in METRICS})
            d['auc'].append(roc_auc_score(yva, p))
            d['auprc'].append(average_precision_score(yva, p))
            d['brier'].append(brier_score_loss(yva, p))
            d['ece'].append(_ece(yva, p))
            s, a = calib_slope_intercept(yva, p)
            d['slope'].append(s); d['citl'].append(a)

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
        div_rows.append({'mean_corr_all': np.nanmean(C[iu]),
                         'mean_corr_same_algo_diff_subset': np.nanmean(same_algo),
                         'mean_corr_diff_algo_same_subset': np.nanmean(diff_algo)})
        if verbose:
            print(f'  fold {f + 1}/{n_splits * n_repeats} done ({time.time() - t0:.0f}s)',
                  flush=True)
    return res, float(np.mean(ntr)), float(np.mean(nte)), pd.DataFrame(div_rows).mean()


def summarise(res, n_tr, n_te, ref, task, roster, extra=None):
    """Summary rows (means) and paired deltas vs `ref` with Nadeau-Bengio CIs."""
    rows, drows, folds = [], [], []
    for mth, d in res.items():
        r = {'task': task, 'roster': roster, 'method': mth}
        for k in METRICS:
            r[k] = float(np.mean(d[k]))
        r['auc_sd'] = float(np.std(d['auc'], ddof=1))
        if extra:
            r.update(extra)
        rows.append(r)
        for i in range(len(d['auc'])):
            folds.append({'task': task, 'roster': roster, 'method': mth, 'fold': i,
                          **{k: d[k][i] for k in METRICS}})
        if mth != ref:
            for k in ('auc', 'auprc'):
                m, lo, hi, p = nb_ci(np.array(d[k]) - np.array(res[ref][k]), n_tr, n_te)
                drows.append({'task': task, 'roster': roster, 'method': mth, 'reference': ref,
                              'metric': k, 'delta': m, 'CI95_low': lo, 'CI95_high': hi,
                              'p_corrected_t': p, 'n_scores': len(d[k])})
    return pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(folds)


def ablation_scores(dm_qc, dm_raw, target_col, n_repeats=3, seed=None):
    """Fixed-config XGBoost on identical folds for each ablation condition.
    dm_raw must hold the same rows in the same order as dm_qc, before clipping."""
    feats = _feat_cols(dm_qc)
    conds = {'All features': feats,
             'Vitals only': [c for c in feats if c.startswith('vitals_')],
             'Labs only': [c for c in feats if c.startswith('labs_')],
             'Mean aggregate only': [c for c in feats if c.endswith('_avg')]}
    out, ntr, nte = {}, None, None
    for name, cols in conds.items():
        s, ntr, nte = xgb_fold_scores(dm_qc, target_col, cols, n_repeats=n_repeats, seed=seed)
        out[name] = (s, len(cols))
    s, ntr, nte = xgb_fold_scores(dm_raw, target_col, feats, n_repeats=n_repeats, seed=seed)
    out['No clipping'] = (s, len(feats))
    base = out['All features'][0]
    rows = []
    for name, (s, k) in out.items():
        if name == 'All features':
            m = lo = hi = 0.0; p = np.nan
        else:
            m, lo, hi, p = nb_ci(s - base, ntr, nte)
        rows.append({'condition': name, 'n_features': k, 'AUROC': s.mean(),
                     'delta_vs_all': m, 'CI95_low': lo, 'CI95_high': hi, 'p_corrected_t': p})
    return pd.DataFrame(rows)
