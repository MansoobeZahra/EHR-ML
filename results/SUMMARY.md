# Reviewer-response experiments: local run summary

Run date: 2026-10-22. Machine: 24 CPUs, 62 GB RAM, CPU only. Python 3.10.12, xgboost 2.1.4, lightgbm 4.7.0, scikit-learn 1.7.2, shap 0.49.1, pandas 2.3.3, numpy 2.2.6.
Code: `local_run/research02_local.ipynb` (research02_revision.ipynb with only paths, chunk sizes and Kaggle lines changed), run through `local_run/runner.py`. Full run: 3 repeats of 5-fold CV (15 paired scores per method). Step 22 (eICU task) and Step 25 were re-run on the full eICU cohort (12,677 episodes) on 2026-10-22 at the user's request; the final result set used for the paper is `results/final/` (full-eICU rows replace the n = 5,000 subsample rows). Raw outputs: `results/full_run/` (subsample run) and `results/full_run_eicu_all/`. Log: `results/run.log`.

## Second pass (Step 27, 2026-10-22): revised verdict

All second-pass experiments use one engine (`local_run/step27_ext.py`) that was first checked against Step 22: it reproduces all seven Step 22 MIMIC-IV AUROC values with zero difference. Every comparison uses identical outer folds (5-fold CV x 3 repeats, 15 paired scores) and Nadeau-Bengio corrected 95% CIs. Tables: `results/SUMMARY_ext_tables.md`; raw files: `results/ext/`.

1. Default-XGBoost meta-learner (the original pipeline's choice): significantly worse than the single XGBoost model in 17 of 22 task/roster/tuning settings (AUROC -2.1 to -6.5 points), never better. Calibration slope 0.29 to 0.57 and calibration-in-the-large +0.30 to +1.55 (overconfident and miscalibrated), versus slope 0.80 to 1.15 for single XGBoost.
2. Regularised or tuned meta-learner (fixed-config XGBoost, logistic regression, tuned XGBoost, tuned logistic regression): never significantly better than a single XGBoost with the same tuning budget (0 of 44 comparisons), not significantly different in 38 of 44, significantly worse in 6 of 44 (eICU 72 h under equal tuning: -0.9 points; LOS>7 at the 1-day landmark: -2.6 and -3.2 points; diverse roster LR meta on LOS>7 at 2 days: -2.6 points; CatBoost-for-MLP roster, eICU, fixed XGB meta: -0.3 points).
3. Equal tuning budget (R1.1, R3): every XGBoost (single model, Level-1 reference, meta-learner) received the same nested random search (20 configurations, 3-fold inner CV, inside each outer training fold; the stack additionally had its Level-1 tuned). Tuned single XGBoost vs tuned stacks: MIMIC-IV 24 h -0.0002 and +0.0045 (CIs include 0); eICU 72 h -0.0093 and -0.0086 (CIs exclude 0, stack worse); LOS>7 and LOS>14 at 2 days: CIs include 0. Tuning gave the single model +0.2 to +1.2 points.
4. Level-1 diversity (R1.3, R3), now on both cohorts and all landmark tasks: the diverse roster (kNN, RF, naive Bayes) and CatBoost as fifth learner or in place of the MLP did not make any stack beat the single model. Standalone CatBoost was indistinguishable from XGBoost on all six tasks (-0.7 to +0.6 points, all CIs include 0).
5. Cohort size (R1.5), redone with 10 draws x 3 repeats and per-draw CIs: at n = 729 the regularised stacks had mean differences of -0.8 (fixed XGB meta) and -1.3 points (LR meta) with no draw whose CI excluded 0; the default-XGB meta stack averaged -3.0 points (CI below 0 in 1 of 10 draws; the comparison is underpowered at this size). This REPLACES the first-pass Step 23 statement that learned meta-learners "lose more" at small size, which came from 5 draws with 1 repeat and no CIs and does not hold up.
6. Ablation and preprocessing under the new protocol (fixed XGBoost, identical folds, CIs): keeping only vitals or only labs lowers AUROC on every task (vitals-only on LOS>14 is the exception, -1.6, CI includes 0); mean-only aggregation lowers AUROC on all four tasks (-0.7 to -6.0 points); removing QC clipping has no effect (-0.0 to +0.2 points, all CIs include 0).
7. LOS: 1-day landmark (2,067 patients still in ICU) gives XGBoost AUROC 0.778 (LOS>7) and 0.768 (LOS>14); 2-day landmark 0.782 and 0.795. The leakage-safe estimate does not depend on the choice t = 2.
8. AUPRC leads to the same conclusions, and is stricter on LOS>7: there some stacks have a significant AUPRC deficit while their AUROC deficit is not significant (for example the fixed-config XGB meta at 2 days, AUPRC -4.2 points, CI -8.0 to -0.4). LOS>14 AUPRC is 0.35 at 2 days (prevalence 11.6%) and 0.21 at 1 day (8.2%).
9. SHAP, fixed-config XGBoost on the full cohorts: lactate (last) leads MIMIC-IV (mean |SHAP| 0.34); heart rate (last) leads eICU (0.31). Rank agreement over the 55 shared features: Spearman rho 0.51 (95% CI 0.28 to 0.68).

New headline (decision tree, between branches 1 and 2): the large stacking penalty in the original replication is caused by the untuned default-XGBoost meta-learner; with a regularised or equally tuned meta-learner the stack matches a single XGBoost model in most settings, is worse in a few, and is never better.

### Errors found in this pass (must be reflected in the paper)

- Fix I was itself wrong. `standardise_cols` maps eICU "Platelets" to "plateletss", so platelets never matched MIMIC-IV, which produced the old "50 shared features". In fact all 55 eICU features have a counterpart in the MIMIC-IV matrix (11 measurements x 5 aggregates). Paper v1/v2 says "50 of which are shared"; this must become 55.
- First-pass Step 23 conclusion (n = 729) is replaced by item 5 above.
- First-pass verdict item 3 said the stack was never worse with a regularised meta-learner. With the additional settings it is worse in 6 of 44 comparisons; the paper must say "matches in most settings" rather than "matches".
- Performance note: Step 22's LightGBM uses n_jobs = -1 (all 24 cores), which slowed fits 30 to 100 times while another training job shared the machine. The Step 27 engine sets explicit thread counts; results are identical (checked above).

## Fourth pass (2026-10-26): eICU inner splits grouped by patient

- The eICU stacking analyses (main, tuned, diverse, both CatBoost rosters, early deaths) were re-run with every inner split also grouped by patient: Level-1 out-of-fold predictions and all hyperparameter searches (`step27_ext.py`: `GROUPS`, `inner_splits`). Results: `results/ext3/`. MIMIC-IV results are unchanged; the engine check still gives zero difference from Step 22.
- Effect: default-meta stack on eICU −2.7 points (was −2.2). Regularized/tuned meta-learners are not significantly different in 39 of 44 comparisons (was 38) and worse in 5 (was 6): the CatBoost-roster loss on eICU disappears. Early deaths, eICU: default −4.7, fixed −0.6 (CI excludes 0), LR −0.4.
- Ablation, coverage sweep and SHAP selection have no inner splits; their grouped results in `results/ext2/` stand.

## Third pass (2026-10-26): eICU grouped by patient, early-death sensitivity

- eICU has 12,677 stays from 10,835 patients in the 72 h matrix. All eICU analyses (main, tuned, diverse, both CatBoost rosters, ablation, coverage sweep, SHAP stability and selection) were re-run with outer folds grouped by patient (StratifiedGroupKFold per repeat; `local_run/runner2.py`, `group_by_patient`). Results: `results/ext2/`. MIMIC-IV has one episode per patient, so it is unaffected.
- Effect: single XGBoost on eICU 0.8532 (was 0.8529); default-meta stack -2.2 points (was -2.3); conclusions unchanged. With equal tuning, both tuned stacks are now significantly worse on eICU (-0.9 and -0.7). Nested selection on eICU: 20 features now lose 0.3 points (CI excluding zero), 30 features are within 0.1. Top-20 SHAP Jaccard on eICU 0.85 (min 0.74).
- Early deaths (death inside the feature window) excluded: MIMIC-IV 140 of 2,533 episodes, eICU 1,230 of 12,677 stays. Single-model AUROC drops to 0.800 and 0.772; the default meta-learner stays significantly worse (-3.7, -4.1); regularized stacks within 0.7 points (only fixed-config XGB meta on eICU, -0.7, excludes zero).
- All numbers quoted in the paper are computed by `local_run/paper_facts.py`; output in `results/PAPER_FACTS.txt`.

## Sanity checks against the original Kaggle run (all pass)

| Check | Original | Local |
|---|---|---|
| MIMIC ICD cohort (variant E), episodes | 2,545 | 2,545 |
| MIMIC 24h matrix: rows / features at 80% | 2,533 / 120 | 2,533 / 120 |
| eICU ICD cohort: episodes / patients / mortality | 12,688 / ~10,842 / 22.4% | 12,688 / 10,842 / 22.4% |
| eICU 72h matrix features at 80% | 55 | 55 |
| XGB single, MIMIC 24h mortality AUROC (old: 5x5 CV) | 0.8333 | 0.8337 |
| XGB single, MIMIC 24h, 80% coverage (Step 24 sweep vs old Step 18) | 0.8345 | 0.8344 |

## Verdict (decision tree in CLAUDE.md)

1. The "stacking is the bottleneck / sets a performance ceiling" claim does not survive. With a regularised meta-learner (logistic regression, or XGBoost with the same fixed config) the stack is statistically indistinguishable from single XGBoost on all four tasks (original roster: deltas -0.016 to +0.004 AUROC, every 95% Nadeau-Bengio CI includes 0).
2. The old gap is reproduced only with the original meta-learner, XGBoost with library defaults: -0.023 to -0.039 AUROC, CI excludes 0 in 4 of 6 task/roster pairs, and ECE 0.06 to 0.18 versus 0.02 to 0.06 for single XGBoost. So the gap and the poor calibration come from the untuned meta-learner, not from stacking per se (decision-tree branch 1).
3. What survives, softened: no stacking variant beat single XGBoost on any task (no CI lies above 0), and a more diverse Level-1 roster (kNN, RF, naive Bayes, MLP) did not change this (with the diverse roster the LR-meta stack was even significantly worse on LOS>7, -0.027). A single fixed-config XGBoost is as good as stacking here and simpler. "Independent of base learner" must not be claimed.
4. The old LOS results (0.946 / 0.918) are not defensible. Under the leakage-safe landmark design (features from ICU day 0 to 2, patients still in ICU at day 2, n = 1,470) single XGBoost reaches 0.782 (LOS>7) and 0.795 (LOS>14).
5. The old MIMIC coverage-threshold effect (p = .034 and p < .001, Wilcoxon) is not significant with corrected CIs; eICU does have features with 60 to 80% coverage (10 of 85), and adding them gives a small but CI-supported gain (+0.003 to +0.007). Leak-free SHAP selection: top-20 and top-30 match all features; top-10 loses 1.2 to 2.3 AUROC points.

## Full eICU cohort (12,677 episodes), Step 22 original roster and Step 25

| Method | AUROC | Delta vs XGB single (95% NB CI) | ECE |
|---|---|---|---|
| XGB single | 0.8529 | reference | 0.019 |
| LR single | 0.8320 | -0.0209 (-0.0263, -0.0155) | 0.015 |
| Stack, XGB meta (library defaults) | 0.8295 | -0.0234 (-0.0305, -0.0163) | 0.064 |
| Stack, XGB meta (fixed config) | 0.8499 | -0.0030 (-0.0062, +0.0001) | 0.013 |
| Stack, LR meta | 0.8502 | -0.0027 (-0.0062, +0.0009) | 0.016 |
| Mean of Level-1 | 0.8337 | -0.0192 (-0.0239, -0.0145) | 0.103 |
| Best single Level-1 (in-fold) | 0.7937 | -0.0592 (-0.0680, -0.0503) | 0.018 |

Level-1 correlation, different algorithms on the same subset: 0.86. Step 25 on full eICU: top-10 -0.0128 (CI -0.0172, -0.0084), top-20 -0.0017 (CI includes 0), top-30 -0.0005, top-40 +0.0001; top-20 SHAP Jaccard mean 0.82 (min 0.74).

## Other findings worth reporting

- Level-1 diversity is low: different algorithms on the same feature subset correlate at 0.72 to 0.90 (original roster) and 0.70 to 0.83 (diverse roster).
- Step 23, n = 729 (5 draws, 1 repeat each, no CIs by design): every learned-meta stack scores below single XGBoost in at least 4 of 5 draws (fixed-config meta -0.015, LR meta -0.022, default meta -0.043). Only the plain mean of Level-1 matches single XGBoost (+0.002). At the original study's size the learned meta-learner loses more, consistent with meta-learner overfitting.
- SHAP top-20 stability across training folds: mean Jaccard 0.61 (min 0.38) for MIMIC-IV 24h, 0.82 (min 0.74) for eICU 72h. Lactate, SpO2, heart rate, temperature and GCS verbal are in the MIMIC top-20 in every fold.
- eICU: all 55 features kept at 80% have at least 90% coverage, so 80% and 90% give identical matrices. The old sentence "no columns between 70 and 90%" was wrong: 5 features lie in 70 to 80%.

## Problems, failures and differences from the original run

1. My own earlier mistake: in the v1/v2 paper (fix E) I rewrote the coverage sentence as "lowering it from 80% to 70% improved AUROC (p = .034) and raising it to 90% reduced it (p < .001)", using RESULTS.md. With corrected CIs neither change is significant (+0.0033, CI -0.0042 to +0.0108; -0.0079, CI -0.0211 to +0.0053). That sentence must change in Phase 4.
2. Step 22 uses fixed Level-1 configs (LR C = 1, MLP 32x32, LGBM and XGB fixed), not the tuned Level-1 and 2*n_features MLP of the original pipeline. Its "Stack, XGB meta (library defaults)" scores 0.8011 on MIMIC 24h versus 0.8132 for the original stacked run. The comparison is fair within Step 22 but is not a re-run of the original stack.
3. eICU Step 22 and Step 25 were first run on the 5,000 subsample (XGB single 0.8415) and then, after approval, on all 12,677 episodes (XGB single 0.8529 versus 0.8530 in the old run; 140 s). The output files label the full run "n=12677 subsample" because of the driver's label string; it is the full cohort. On the full cohort the mean of Level-1 is significantly below single XGBoost (-0.019) and its ECE is 0.103.
4. Not run: eICU 24h and 48h windows, the 1-day LOS landmark (LANDMARKS = (2,)), and eICU LOS. The LOS>14 landmark scoring (unit s26_los14) is my addition; the notebook builds that task but never scores it.
5. At 50%, 60% and 70% the MIMIC sweep gives the same 155 features (no MIMIC feature has 50 to 70% coverage), so those three rows are identical.
6. Software differs from Kaggle (local xgboost 2.1.4; xgboost 3.2 failed with shap 0.49.1). Parallel units ran with 2 threads each; this can change results in the last decimals only.
7. Runtime was far below the CLAUDE.md estimate: data build 9 minutes, full experiment run 3.3 minutes.
8. Process issues: the first build was killed when a session ended (restarted, logged in run.log), and the project folder was moved mid-task to `Desktop/Speech to sign/Colacare`; the local notebook now derives paths from its own location.


# Full tables (generated by local_run/make_summary.py from results/final)

## Step 22: controlled architecture comparison (identical outer folds)

#### Roster: original

| task | method | AUROC | AUROC_sd | Brier | ECE |
|---|---|---|---|---|---|
| MIMIC-IV 24h-window in-hospital mortality | XGB single (all features) | 0.8337 | 0.0158 | 0.1347 | 0.0392 |
| MIMIC-IV 24h-window in-hospital mortality | LR single (all features) | 0.8291 | 0.0151 | 0.1385 | 0.0446 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, XGB meta (library defaults) | 0.8011 | 0.0182 | 0.1617 | 0.1199 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, XGB meta (fixed config) | 0.8286 | 0.0161 | 0.1376 | 0.0365 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, LR meta | 0.8322 | 0.0188 | 0.1354 | 0.0397 |
| MIMIC-IV 24h-window in-hospital mortality | Mean of Level-1 | 0.8339 | 0.0166 | 0.1453 | 0.0807 |
| MIMIC-IV 24h-window in-hospital mortality | Best single Level-1 (picked in-fold) | 0.7717 | 0.0233 | 0.1572 | 0.0439 |
| MIMIC LOS>7 (landmark 2d) | XGB single (all features) | 0.7815 | 0.0284 | 0.1724 | 0.0550 |
| MIMIC LOS>7 (landmark 2d) | LR single (all features) | 0.7696 | 0.0238 | 0.1839 | 0.0890 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (library defaults) | 0.7429 | 0.0276 | 0.2197 | 0.1748 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (fixed config) | 0.7658 | 0.0252 | 0.1783 | 0.0645 |
| MIMIC LOS>7 (landmark 2d) | Stack, LR meta | 0.7762 | 0.0228 | 0.1762 | 0.0707 |
| MIMIC LOS>7 (landmark 2d) | Mean of Level-1 | 0.7674 | 0.0211 | 0.1857 | 0.0918 |
| MIMIC LOS>7 (landmark 2d) | Best single Level-1 (picked in-fold) | 0.7659 | 0.0196 | 0.1744 | 0.0457 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | XGB single (all features) | 0.8529 | 0.0086 | 0.1080 | 0.0190 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | LR single (all features) | 0.8320 | 0.0081 | 0.1130 | 0.0152 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Stack, XGB meta (library defaults) | 0.8295 | 0.0092 | 0.1190 | 0.0641 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Stack, XGB meta (fixed config) | 0.8499 | 0.0085 | 0.1082 | 0.0133 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Stack, LR meta | 0.8502 | 0.0085 | 0.1083 | 0.0163 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Mean of Level-1 | 0.8337 | 0.0083 | 0.1387 | 0.1033 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Best single Level-1 (picked in-fold) | 0.7937 | 0.0104 | 0.1356 | 0.0180 |

Paired AUROC difference vs XGB single, Nadeau-Bengio corrected 95% CI:

| task | method | delta (95% NB CI) | p_corrected_t | CI excludes 0 | n_scores |
|---|---|---|---|---|---|
| MIMIC-IV 24h-window in-hospital mortality | LR single (all features) | -0.0046 (-0.0170, +0.0077) | 0.4347 | no | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, XGB meta (library defaults) | -0.0326 (-0.0538, -0.0114) | 0.0053 | yes | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, XGB meta (fixed config) | -0.0051 (-0.0184, +0.0082) | 0.4233 | no | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, LR meta | -0.0016 (-0.0122, +0.0090) | 0.7576 | no | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Mean of Level-1 | +0.0002 (-0.0060, +0.0065) | 0.9353 | no | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Best single Level-1 (picked in-fold) | -0.0620 (-0.0831, -0.0408) | 0.0000 | yes | 15 |
| MIMIC LOS>7 (landmark 2d) | LR single (all features) | -0.0120 (-0.0318, +0.0079) | 0.2174 | no | 15 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (library defaults) | -0.0387 (-0.0727, -0.0046) | 0.0288 | yes | 15 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (fixed config) | -0.0158 (-0.0430, +0.0115) | 0.2347 | no | 15 |
| MIMIC LOS>7 (landmark 2d) | Stack, LR meta | -0.0053 (-0.0276, +0.0169) | 0.6144 | no | 15 |
| MIMIC LOS>7 (landmark 2d) | Mean of Level-1 | -0.0142 (-0.0338, +0.0054) | 0.1432 | no | 15 |
| MIMIC LOS>7 (landmark 2d) | Best single Level-1 (picked in-fold) | -0.0157 (-0.0400, +0.0087) | 0.1899 | no | 15 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | LR single (all features) | -0.0209 (-0.0263, -0.0155) | 0.0000 | yes | 15 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Stack, XGB meta (library defaults) | -0.0234 (-0.0305, -0.0163) | 0.0000 | yes | 15 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Stack, XGB meta (fixed config) | -0.0030 (-0.0062, +0.0001) | 0.0599 | no | 15 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Stack, LR meta | -0.0027 (-0.0062, +0.0009) | 0.1332 | no | 15 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Mean of Level-1 | -0.0192 (-0.0239, -0.0145) | 0.0000 | yes | 15 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | Best single Level-1 (picked in-fold) | -0.0592 (-0.0680, -0.0503) | 0.0000 | yes | 15 |

Level-1 output correlation (validation folds, mean):

| task | mean_corr_all | mean_corr_same_algo_diff_subset | mean_corr_diff_algo_same_subset |
|---|---|---|---|
| MIMIC-IV 24h-window in-hospital mortality | 0.611 | 0.603 | 0.899 |
| MIMIC LOS>7 (landmark 2d) | 0.421 | 0.431 | 0.715 |
| eICU 72h-window in-hospital mortality (all 12,677 episodes) | 0.381 | 0.358 | 0.858 |

#### Roster: diverse

| task | method | AUROC | AUROC_sd | Brier | ECE |
|---|---|---|---|---|---|
| MIMIC-IV 24h-window in-hospital mortality | XGB single (all features) | 0.8337 | 0.0158 | 0.1347 | 0.0392 |
| MIMIC-IV 24h-window in-hospital mortality | LR single (all features) | 0.8291 | 0.0151 | 0.1385 | 0.0446 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, XGB meta (library defaults) | 0.8048 | 0.0173 | 0.1631 | 0.1254 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, XGB meta (fixed config) | 0.8316 | 0.0176 | 0.1365 | 0.0423 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, LR meta | 0.8223 | 0.0200 | 0.1385 | 0.0375 |
| MIMIC-IV 24h-window in-hospital mortality | Mean of Level-1 | 0.8284 | 0.0165 | 0.1457 | 0.0719 |
| MIMIC-IV 24h-window in-hospital mortality | Best single Level-1 (picked in-fold) | 0.7753 | 0.0198 | 0.1561 | 0.0422 |
| MIMIC LOS>7 (landmark 2d) | XGB single (all features) | 0.7815 | 0.0284 | 0.1724 | 0.0550 |
| MIMIC LOS>7 (landmark 2d) | LR single (all features) | 0.7696 | 0.0238 | 0.1839 | 0.0890 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (library defaults) | 0.7487 | 0.0246 | 0.2192 | 0.1783 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (fixed config) | 0.7677 | 0.0204 | 0.1776 | 0.0631 |
| MIMIC LOS>7 (landmark 2d) | Stack, LR meta | 0.7550 | 0.0229 | 0.1836 | 0.0723 |
| MIMIC LOS>7 (landmark 2d) | Mean of Level-1 | 0.7621 | 0.0198 | 0.1850 | 0.0863 |
| MIMIC LOS>7 (landmark 2d) | Best single Level-1 (picked in-fold) | 0.7631 | 0.0168 | 0.1772 | 0.0544 |

Paired AUROC difference vs XGB single, Nadeau-Bengio corrected 95% CI:

| task | method | delta (95% NB CI) | p_corrected_t | CI excludes 0 | n_scores |
|---|---|---|---|---|---|
| MIMIC-IV 24h-window in-hospital mortality | LR single (all features) | -0.0046 (-0.0170, +0.0077) | 0.4347 | no | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, XGB meta (library defaults) | -0.0289 (-0.0487, -0.0091) | 0.0074 | yes | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, XGB meta (fixed config) | -0.0021 (-0.0138, +0.0096) | 0.7066 | no | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Stack, LR meta | -0.0114 (-0.0333, +0.0105) | 0.2832 | no | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Mean of Level-1 | -0.0053 (-0.0143, +0.0038) | 0.2342 | no | 15 |
| MIMIC-IV 24h-window in-hospital mortality | Best single Level-1 (picked in-fold) | -0.0584 (-0.0737, -0.0430) | 0.0000 | yes | 15 |
| MIMIC LOS>7 (landmark 2d) | LR single (all features) | -0.0120 (-0.0318, +0.0079) | 0.2174 | no | 15 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (library defaults) | -0.0329 (-0.0668, +0.0010) | 0.0560 | no | 15 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (fixed config) | -0.0138 (-0.0424, +0.0148) | 0.3167 | no | 15 |
| MIMIC LOS>7 (landmark 2d) | Stack, LR meta | -0.0265 (-0.0513, -0.0018) | 0.0376 | yes | 15 |
| MIMIC LOS>7 (landmark 2d) | Mean of Level-1 | -0.0194 (-0.0383, -0.0005) | 0.0447 | yes | 15 |
| MIMIC LOS>7 (landmark 2d) | Best single Level-1 (picked in-fold) | -0.0185 (-0.0465, +0.0096) | 0.1803 | no | 15 |

Level-1 output correlation (validation folds, mean):

| task | mean_corr_all | mean_corr_same_algo_diff_subset | mean_corr_diff_algo_same_subset |
|---|---|---|---|
| MIMIC-IV 24h-window in-hospital mortality | 0.572 | 0.581 | 0.830 |
| MIMIC LOS>7 (landmark 2d) | 0.414 | 0.428 | 0.696 |

### Step 23: size sensitivity (5 stratified draws, n = 729)

| method | AUROC_mean | AUROC_sd | delta_mean | delta_sd | draws_below_single | ECE_mean |
|---|---|---|---|---|---|---|
| Best single Level-1 (picked in-fold) | 0.7559 | 0.0173 | -0.0573 | 0.0180 | 5 | 0.0779 |
| LR single (all features) | 0.7919 | 0.0142 | -0.0213 | 0.0085 | 5 | 0.1083 |
| Mean of Level-1 | 0.8150 | 0.0065 | 0.0018 | 0.0141 | 1 | 0.0957 |
| Stack, LR meta | 0.7911 | 0.0302 | -0.0220 | 0.0228 | 4 | 0.0816 |
| Stack, XGB meta (fixed config) | 0.7979 | 0.0176 | -0.0153 | 0.0078 | 5 | 0.0851 |
| Stack, XGB meta (library defaults) | 0.7707 | 0.0198 | -0.0425 | 0.0087 | 5 | 0.1606 |
| XGB single (all features) | 0.8132 | 0.0191 | 0.0000 | 0.0000 | 0 | 0.0833 |

### Step 24: feature coverage before filtering, and coverage sweep

Number of features per coverage band (pre-filter):

| coverage band | eICU 72h | MIMIC-IV 24h |
|---|---|---|
| [0.0, 0.5) | 20 | 15 |
| [0.5, 0.6) | 0 | 0 |
| [0.6, 0.7) | 5 | 0 |
| [0.7, 0.8) | 5 | 35 |
| [0.8, 0.9) | 0 | 15 |
| [0.9, 1.0) | 55 | 105 |

Histograms: coverage_hist_MIMIC-IV_24h.png, coverage_hist_eICU_72h.png

| dataset | threshold | n_features | AUROC | delta_vs_ref | CI95_low | CI95_high | p_corrected_t |
|---|---|---|---|---|---|---|---|
| MIMIC-IV 24h | 0.5000 | 155 | 0.8377 | 0.0033 | -0.0042 | 0.0108 | 0.3665 |
| MIMIC-IV 24h | 0.6000 | 155 | 0.8377 | 0.0033 | -0.0042 | 0.0108 | 0.3665 |
| MIMIC-IV 24h | 0.7000 | 155 | 0.8377 | 0.0033 | -0.0042 | 0.0108 | 0.3665 |
| MIMIC-IV 24h | 0.8000 | 120 | 0.8344 | 0.0000 | 0.0000 | 0.0000 |  |
| MIMIC-IV 24h | 0.9000 | 105 | 0.8265 | -0.0079 | -0.0211 | 0.0053 | 0.2210 |
| eICU 72h | 0.5000 | 65 | 0.8596 | 0.0067 | 0.0032 | 0.0102 | 0.0010 |
| eICU 72h | 0.6000 | 65 | 0.8596 | 0.0067 | 0.0032 | 0.0102 | 0.0010 |
| eICU 72h | 0.7000 | 60 | 0.8558 | 0.0029 | 0.0013 | 0.0045 | 0.0018 |
| eICU 72h | 0.8000 | 55 | 0.8529 | 0.0000 | 0.0000 | 0.0000 |  |
| eICU 72h | 0.9000 | 55 | 0.8529 | 0.0000 | 0.0000 | 0.0000 |  |

### Step 25: SHAP fold stability and nested (leak-free) top-K selection

- MIMIC-IV 24h mortality: top-20 SHAP set Jaccard across training folds, mean 0.612, min 0.379
- eICU 72h mortality: top-20 SHAP set Jaccard across training folds, mean 0.820, min 0.739

MIMIC-IV:

| top_k | AUROC | AUROC_all_features | delta_vs_all | CI95_low | CI95_high | p_corrected_t |
|---|---|---|---|---|---|---|
| 10 | 0.8096 | 0.8329 | -0.0233 | -0.0382 | -0.0083 | 0.0064 |
| 20 | 0.8288 | 0.8329 | -0.0041 | -0.0175 | 0.0094 | 0.5138 |
| 30 | 0.8312 | 0.8329 | -0.0017 | -0.0088 | 0.0054 | 0.5980 |
| 40 | 0.8336 | 0.8329 | 0.0007 | -0.0087 | 0.0100 | 0.8763 |

MIMIC-IV most frequently selected features (first 10):

| feature | share of folds in top-20 |
|---|---|
| vitals_heart_rate_avg | 1.00 |
| vitals_gcs_verbal_avg | 1.00 |
| vitals_temperature_f_avg | 1.00 |
| vitals_temperature_f_min | 1.00 |
| vitals_spo2_min | 1.00 |
| vitals_spo2_last | 1.00 |
| labs_lactate_last | 1.00 |
| vitals_spo2_avg | 0.90 |
| labs_lactate_avg | 0.90 |
| vitals_gcs_eye_last | 0.90 |

eICU:

| top_k | AUROC | AUROC_all_features | delta_vs_all | CI95_low | CI95_high | p_corrected_t |
|---|---|---|---|---|---|---|
| 10 | 0.8403 | 0.8531 | -0.0128 | -0.0172 | -0.0084 | 0.0001 |
| 20 | 0.8515 | 0.8531 | -0.0017 | -0.0041 | 0.0008 | 0.1649 |
| 30 | 0.8526 | 0.8531 | -0.0005 | -0.0021 | 0.0011 | 0.4675 |
| 40 | 0.8532 | 0.8531 | 0.0001 | -0.0007 | 0.0010 | 0.7650 |

eICU most frequently selected features (first 10):

| feature | share of folds in top-20 |
|---|---|
| vitals_Heart Rate_avg | 1.00 |
| vitals_SpO2_avg | 1.00 |
| vitals_Heart Rate_max | 1.00 |
| vitals_SpO2_min | 1.00 |
| vitals_Heart Rate_last | 1.00 |
| vitals_SpO2_last | 1.00 |
| vitals_Respiratory Rate_last | 1.00 |
| labs_Hemoglobin_avg | 1.00 |
| labs_Glucose_last | 1.00 |
| labs_Potassium_last | 1.00 |

### Step 26: LOS landmark tasks (features from ICU day 0 to t; only patients still in ICU at t)

| task | n | prevalence |
|---|---|---|
| MIMIC LOS>7 (landmark 2d) | 1470 | 0.319 |
| MIMIC LOS>14 (landmark 2d) | 1470 | 0.116 |

| task | method | AUROC | AUROC_sd | Brier | ECE |
|---|---|---|---|---|---|
| MIMIC LOS>7 (landmark 2d) | XGB single (all features) | 0.7815 | 0.0284 | 0.1724 | 0.0550 |
| MIMIC LOS>7 (landmark 2d) | LR single (all features) | 0.7696 | 0.0238 | 0.1839 | 0.0890 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (library defaults) | 0.7429 | 0.0276 | 0.2197 | 0.1748 |
| MIMIC LOS>7 (landmark 2d) | Stack, XGB meta (fixed config) | 0.7658 | 0.0252 | 0.1783 | 0.0645 |
| MIMIC LOS>7 (landmark 2d) | Stack, LR meta | 0.7762 | 0.0228 | 0.1762 | 0.0707 |
| MIMIC LOS>7 (landmark 2d) | Mean of Level-1 | 0.7674 | 0.0211 | 0.1857 | 0.0918 |
| MIMIC LOS>7 (landmark 2d) | Best single Level-1 (picked in-fold) | 0.7659 | 0.0196 | 0.1744 | 0.0457 |
| MIMIC LOS>14 (landmark 2d) | XGB single (all features) | 0.7952 | 0.0273 | 0.0899 | 0.0360 |
| MIMIC LOS>14 (landmark 2d) | LR single (all features) | 0.7563 | 0.0274 | 0.1080 | 0.0784 |
| MIMIC LOS>14 (landmark 2d) | Stack, XGB meta (library defaults) | 0.7627 | 0.0353 | 0.1064 | 0.0923 |
| MIMIC LOS>14 (landmark 2d) | Stack, XGB meta (fixed config) | 0.7932 | 0.0297 | 0.0921 | 0.0410 |
| MIMIC LOS>14 (landmark 2d) | Stack, LR meta | 0.7997 | 0.0191 | 0.0923 | 0.0411 |
| MIMIC LOS>14 (landmark 2d) | Mean of Level-1 | 0.7905 | 0.0276 | 0.0944 | 0.0629 |
| MIMIC LOS>14 (landmark 2d) | Best single Level-1 (picked in-fold) | 0.7902 | 0.0318 | 0.0913 | 0.0279 |

LOS>14 landmark, paired difference vs XGB single:

| task | method | delta (95% NB CI) | p_corrected_t | CI excludes 0 | n_scores |
|---|---|---|---|---|---|
| MIMIC LOS>14 (landmark 2d) | LR single (all features) | -0.0389 (-0.0882, +0.0103) | 0.1118 | no | 15 |
| MIMIC LOS>14 (landmark 2d) | Stack, XGB meta (library defaults) | -0.0325 (-0.0736, +0.0086) | 0.1123 | no | 15 |
| MIMIC LOS>14 (landmark 2d) | Stack, XGB meta (fixed config) | -0.0021 (-0.0358, +0.0317) | 0.8970 | no | 15 |
| MIMIC LOS>14 (landmark 2d) | Stack, LR meta | +0.0044 (-0.0308, +0.0397) | 0.7903 | no | 15 |
| MIMIC LOS>14 (landmark 2d) | Mean of Level-1 | -0.0047 (-0.0299, +0.0205) | 0.6945 | no | 15 |
| MIMIC LOS>14 (landmark 2d) | Best single Level-1 (picked in-fold) | -0.0050 (-0.0448, +0.0349) | 0.7926 | no | 15 |

Old protocol, for comparison only (7-day feature window, window chosen by maximum AUROC, stacked ensemble, one 5-fold run): LOS>7 AUROC 0.946, LOS>14 AUROC 0.918.
