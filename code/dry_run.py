import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
exec(open('test_harness.py').read())
exec(open('step22_arch.py').read())
exec(open('step23_26.py').read())

def qc_clip_features(dm, meta_cols, q_low=0.01, q_high=0.99):
    dm = dm.copy()
    for c in [c for c in dm.columns if c not in meta_cols]:
        if pd.api.types.is_numeric_dtype(dm[c]):
            dm[c] = dm[c].clip(dm[c].quantile(q_low), dm[c].quantile(q_high))
    return dm

def _mk(n, seed):
    d = make_dm(n, seed=seed)
    d['los'] = np.random.RandomState(seed).gamma(2.0, 2.5, len(d))
    d['los_gt7'] = (d['los'] > 7).astype(int); d['los_gt14'] = (d['los'] > 14).astype(int)
    d['subject_id'] = np.arange(len(d)); d['cohort'] = 'x'
    return d

def build_dm(cohort, cov_thresh=0.8, label='', window_before_days=None, window_after_days=None, anchor='intime'):
    d = _mk(420, 3)
    rng = np.random.RandomState(5)
    for i, c in enumerate([c for c in d.columns if c not in META_COLS]):
        d.loc[rng.rand(len(d)) < 0.03 + 0.4 * (i % 7) / 7, c] = np.nan
    feats = [c for c in d.columns if c not in META_COLS]
    cov = d[feats].notna().mean()
    keep = cov[cov >= cov_thresh].index.tolist()
    return d[[c for c in META_COLS if c in d.columns] + keep]

icd_cohort = None
dm_icd_24h_hosp_qc = qc_clip_features(build_dm(None, 0.8), META_COLS)
base = _mk(700, 9)
rng = np.random.RandomState(2)
for i, c in enumerate([c for c in base.columns if c not in META_COLS]):
    base.loc[rng.rand(len(base)) < 0.02 + 0.45 * (i % 9) / 9, c] = np.nan
dm_eicu_72h = base.copy()
COVERAGE_PREFILTER = {72: base[[c for c in base.columns if c not in META_COLS]].notna().mean()}

src = open('driver.py').read().replace("QUICK            = False", "QUICK            = True") \
        .replace("EICU_N           = 5000", "EICU_N           = 300")
exec(src)
print("DRIVER DRY RUN OK")
