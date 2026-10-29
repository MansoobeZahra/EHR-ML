import json, copy
nb = json.load(open('/mnt/user-data/uploads/research02__2_.ipynb'))
cells = nb['cells']
def code_cell(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.splitlines(keepends=True)}
def md_cell(src): return {"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)}
prereq_idx = [2,4,6,8,10,11,13,15,16,18]
new = [md_cell(
"# EHR-ML reviewer-response experiments (standalone)\n\n"
"Rebuilds only the data matrices (no multi-hour baseline runs), then runs the experiments requested by the reviewers:\n\n"
"* **Step 22** controlled stacking-vs-single comparison: one fixed config for every XGBoost, five Level-2 options on identical Level-1 outputs, optional diverse Level-1 roster, paired folds, Nadeau-Bengio corrected 95% CIs.\n"
"* **Step 23** the same comparison at the original study's cohort size (n = 729).\n"
"* **Step 24** per-feature coverage distribution and a coverage sweep on matrices that are not pre-filtered at 80%.\n"
"* **Step 25** SHAP ranking stability across training folds and leak-free (nested) feature selection.\n"
"* **Step 26** leakage-safe LOS landmark tasks (features only from [0, t], patients still in ICU at t).\n\n"
"Run all cells top to bottom. Send back the `step2*.csv` files and the coverage PNGs listed at the end.")]
for i in prereq_idx: new.append(copy.deepcopy(cells[i]))
new.append(code_cell(
"def qc_clip_features(dm, meta_cols, q_low=0.01, q_high=0.99):\n    dm = dm.copy()\n"
"    feature_cols = [c for c in dm.columns if c not in meta_cols]\n    for c in feature_cols:\n"
"        if pd.api.types.is_numeric_dtype(dm[c]):\n            lo = dm[c].quantile(q_low)\n            hi = dm[c].quantile(q_high)\n"
"            dm[c] = dm[c].clip(lo, hi)\n    return dm\n\n"
"dm_icd_24h_hosp_qc = qc_clip_features(dm_icd_24h_hosp, META_COLS)\nprint('MIMIC 24h matrix (QC clipped):', dm_icd_24h_hosp_qc.shape)\n"))
src30 = ''.join(cells[30]['source'])
pat = "keep_cols = coverage[coverage >= 0.80].index.tolist()"
first = src30.index(pat); second = src30.index(pat, first + 10)
patched = (src30[:second] + "COVERAGE_PREFILTER[window_hours] = coverage.copy()   # reviewer response: keep pre-filter coverage\n    "
           + "keep_cols = coverage[coverage >= EICU_BUILD_COV].index.tolist()" + src30[second+len(pat):])
patched = patched.replace("dm_eicu_24h = build_eicu_window_matrix(24)\n","").replace("dm_eicu_48h = build_eicu_window_matrix(48)\n","")
patched = ("# Reviewer response: build the eICU window matrix at a LOW (50%) coverage threshold and keep the\n"
           "# pre-filter coverage, so the coverage sweep can see features with 50-80% coverage.\n"
           "# Only the 72h window is needed here.\nCOVERAGE_PREFILTER = {}\nEICU_BUILD_COV = 0.50\n\n") + patched
new.append(code_cell(patched))
new.append(md_cell("## Step 22: controlled architecture comparison\nDefinitions only (fast). The experiments run in the driver cell at the end."))
new.append(code_cell(open('step22_arch.py').read()))
new.append(md_cell("## Steps 23-26: helper functions"))
new.append(code_cell(open('step23_26.py').read()))
new.append(md_cell("## Driver: run all reviewer-response experiments\nSet `QUICK = True` first for a fast pass (1 repeat)."))
new.append(code_cell(open('driver.py').read()))
nb2 = copy.deepcopy(nb); nb2['cells'] = new
json.dump(nb2, open('research02_revision.ipynb','w'), indent=1)
code = [''.join(c['source']) for c in new if c['cell_type']=='code']
open('assembled.py','w').write('\n\n# ====CELL====\n'.join(code))
print('cells', len(new), 'code', len(code))
