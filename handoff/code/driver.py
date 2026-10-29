# =============================================================================
# DRIVER: runs every reviewer-response experiment. Edit the flags, then run.
# =============================================================================
QUICK            = False   # True = 1 repeat instead of 3 (about 3x faster, wider CIs)
STEP22_REPEATS   = 1 if QUICK else 3
EICU_N           = 5000    # stratified eICU subsample for the expensive Level-1 runs
RUN_DIVERSE      = True    # also run the 'diverse' Level-1 roster (MIMIC tasks)
LANDMARKS        = (2,)    # LOS landmark days; (1, 2) runs both

# ---- 0. Restore the original 80% coverage filter on the main eICU matrix ------
def apply_cov_thresh(dm, thresh, meta_cols=META_COLS):
    keep = [c for c in dm.columns if c in meta_cols or dm[c].notna().mean() >= thresh]
    return dm[keep]

dm_eicu_72h_full = dm_eicu_72h.copy()              # built at 50% coverage
dm_eicu_72h      = apply_cov_thresh(dm_eicu_72h_full, 0.80)
n_feat = len([c for c in dm_eicu_72h.columns if c not in META_COLS])
print(f'eICU 72h matrix at 80% coverage: {n_feat} features '
      f'(the original run reported 55)')
dm_eicu_72h_qc = qc_clip_features(dm_eicu_72h, META_COLS)

# ---- 1. Leakage-safe LOS landmark tasks ---------------------------------------
print('\nBuilding LOS landmark tasks ...')
los_tasks = los_landmark_tasks(icd_cohort, landmarks=LANDMARKS)

# ---- 2. STEP 22: controlled architecture comparison ---------------------------
eicu_sub = stratified_subsample(dm_eicu_72h_qc, 'mortality', EICU_N, seed=7)
tasks_main = [
    ('MIMIC-IV 24h-window in-hospital mortality', dm_icd_24h_hosp_qc, 'mortality'),
    (f'eICU 72h-window in-hospital mortality (n={len(eicu_sub)} subsample)', eicu_sub, 'mortality'),
] + [t for t in los_tasks if 'LOS>7' in t[0]]
S_orig, D_orig, V_orig = run_step22(tasks_main, roster='original', n_repeats=STEP22_REPEATS)

if RUN_DIVERSE:
    tasks_div = [tasks_main[0]] + [t for t in los_tasks if 'LOS>7' in t[0]]
    S_div, D_div, V_div = run_step22(tasks_div, roster='diverse', n_repeats=STEP22_REPEATS)

# ---- 3. STEP 23: does the finding depend on cohort size? ----------------------
print('\n' + '=' * 70); print('STEP 23: cohort-size sensitivity (n = 729, original study size)')
R23, A23 = size_sensitivity(dm_icd_24h_hosp_qc, 'mortality', n_target=729,
                            n_draws=5, n_repeats=1)
try:
    R23.to_csv(OUT_DIR / 'step23_size_sensitivity_draws.csv', index=False)
    A23.to_csv(OUT_DIR / 'step23_size_sensitivity_summary.csv', index=False)
except Exception as e:
    print('could not save step 23:', e)

# ---- 4. STEP 24: coverage evidence (replaces the unsupported eICU claim) ------
print('\n' + '=' * 70); print('STEP 24: feature-coverage distribution and sweep')
cov_eicu = COVERAGE_PREFILTER[72]            # coverage BEFORE any filtering
coverage_report(cov_eicu, 'eICU 72h')
dm_mimic24_full = build_dm(icd_cohort, cov_thresh=0.0, label='ICD 24h, unfiltered',
                           window_before_days=0, window_after_days=1, anchor='admittime')
feats_m = [c for c in dm_mimic24_full.columns if c not in META_COLS]
coverage_report(dm_mimic24_full[feats_m].notna().mean(), 'MIMIC-IV 24h')
sw_m = coverage_sweep(dm_mimic24_full, 'mortality', name='MIMIC-IV 24h', n_repeats=STEP22_REPEATS)
sw_e = coverage_sweep(dm_eicu_72h_full, 'mortality', name='eICU 72h', n_repeats=STEP22_REPEATS,
                      thresholds=(0.5, 0.6, 0.7, 0.8, 0.9))
pd.concat([sw_m, sw_e]).to_csv(OUT_DIR / 'step24_coverage_sweep.csv', index=False)

# ---- 5. STEP 25: SHAP stability + leak-free feature selection -----------------
print('\n' + '=' * 70); print('STEP 25: SHAP fold stability and nested feature selection')
shap_res = {}
for nm, dmx in [('MIMIC-IV 24h mortality', dm_icd_24h_hosp_qc),
                ('eICU 72h mortality', eicu_sub)]:
    print(f'\n--- {nm} ---')
    shap_res[nm] = shap_stability_and_nested_selection(dmx, 'mortality',
                                                       n_repeats=2 if not QUICK else 1)
    shap_res[nm]['selection'].to_csv(
        OUT_DIR / f'step25_nested_selection_{nm.split()[0]}.csv', index=False)
    shap_res[nm]['freq'].head(30).to_csv(
        OUT_DIR / f'step25_shap_frequency_{nm.split()[0]}.csv')

# ---- 6. Summary of produced files --------------------------------------------
import glob, os
print('\nFiles to send back for the paper revision:')
for f in sorted(glob.glob(str(OUT_DIR / 'step2[2-5]*')) + glob.glob(str(OUT_DIR / 'coverage_hist*'))):
    print('  ', os.path.basename(f))
