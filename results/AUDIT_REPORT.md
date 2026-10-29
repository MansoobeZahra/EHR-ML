# Audit report (independent recomputation)

## 1. Split integrity
- eICU: 12677 stays, 10835 patients, 15 outer folds (5-fold x 3, grouped).
- Patients shared between outer train and test: 0. Shared between inner train and inner validation: 0.
- Training rows without exactly one out-of-fold Level-1 prediction: 0 (the meta-learner is fitted only on these out-of-fold predictions; Level-1 models used on the test fold are refitted on the whole training fold, as in standard stacking).
- MIMIC-IV mortality matrix: 2533 rows, 2533 patients (one episode per patient, so row-level folds are patient-level folds).
- MIMIC LOS>7 (landmark 1d): 2067 rows, 2067 patients.
- MIMIC LOS>14 (landmark 1d): 2067 rows, 2067 patients.
- MIMIC LOS>7 (landmark 2d): 1470 rows, 1470 patients.
- MIMIC LOS>14 (landmark 2d): 1470 rows, 1470 patients.

## 2. What is fitted on the full matrix (before cross-validation)
- Fitted inside each training fold: median imputation, scaling, all models, hyperparameter searches, SHAP ranking and top-K selection (Step 25 fits the imputer, model and SHAP on the training fold and scores the selected features on the held-out fold).
- Fitted on the full matrix: 1st/99th percentile clipping thresholds and the 80% coverage filter. Neither uses the outcome. Removing clipping changes AUROC by at most 0.2 points (ablation); the coverage filter only decides which columns exist.
- Tuned analysis (P3): the Level-1 XGBoost/LR hyperparameters are chosen on the whole outer training fold before the inner out-of-fold predictions are made. This does not touch the test fold but can make the meta-learner inputs slightly optimistic; the original pipeline does the same.

## 3. Outcomes and timestamps (raw tables, read-only)
- MIMIC-IV label = hospital_expire_flag: identical for all rows: True; deaths 704 (27.8%).
- MIMIC-IV deaths with deathtime recorded: 703 of 704; deaths within 24 h of hospital admission: 140; median time to death 96 h.
- MIMIC-IV features: chartevents/labevents with charttime in [admittime, admittime + 24 h] (build_dm, window_before_days=0, window_after_days=1, anchor=admittime).
- eICU label = hospitaldischargestatus 'Expired': identical: True; missing discharge status: 104 stays (labelled as survivors).
- eICU deaths within 72 h of ICU admission: 1230 of 2835; median time to death 95 h.
- eICU features: vitalPeriodic/lab offsets in [0, 72 h] from ICU admission.

## 4. Statistics recomputed from per-fold scores
- Recomputed 66 comparisons from fold scores (ratio n_test/n_train = 1/4); largest difference from the saved deltas/CIs: 0.00000 AUROC (grouped folds have slightly unequal sizes). Verdicts identical to saved: True.
- Default meta-learner: 22 settings; worse 17, not significant 5, better 0; differences -2.0 to -6.5 points.
- Regularized/tuned meta-learners: 44 comparisons; not significant 39, worse 5, better 0.
- Worse (regularized): main:los7_d1 Stack, XGB meta (fixed config) -2.6; main:los7_d1 Stack, LR meta -3.2; tuned:eicu72 Stack, XGB meta (tuned) -0.9; tuned:eicu72 Stack, LR meta (tuned C) -0.7; div:los7_d2 Stack, LR meta -2.6
- Not significant (default): main:los14_d2 -3.2; div:los7_d2 -3.0; div:los14_d2 -3.2; cat5:los14_d2 -3.0; catmlp:los14_d2 -2.9
- Paper abstract: default worse 17 of 22; regularized not significant 39 of 44, worse 5.
- Paper counts match the recomputed inventory: True.

### Table II recomputed from fold scores (points, 95% CI; ECE)
- mimic24: single 0.834 (ECE 0.039); Stack, XGB meta: -3.26 (-5.38, -1.14), ECE 0.120; Stack, XGB meta: -0.51 (-1.84, +0.82), ECE 0.037; Stack, LR meta: -0.16 (-1.22, +0.90), ECE 0.040; Mean of Level-1: +0.02 (-0.60, +0.65), ECE 0.081; Best single Level-1: -6.20 (-8.31, -4.08), ECE 0.044
- eicu72: single 0.853 (ECE 0.020); Stack, XGB meta: -2.71 (-3.51, -1.91), ECE 0.066; Stack, XGB meta: -0.32 (-0.82, +0.17), ECE 0.017; Stack, LR meta: -0.27 (-0.67, +0.13), ECE 0.018; Mean of Level-1: -1.97 (-2.57, -1.38), ECE 0.104; Best single Level-1: -5.96 (-6.86, -5.07), ECE 0.020
- los7_d2: single 0.782 (ECE 0.055); Stack, XGB meta: -3.87 (-7.27, -0.46), ECE 0.175; Stack, XGB meta: -1.58 (-4.30, +1.15), ECE 0.064; Stack, LR meta: -0.53 (-2.76, +1.69), ECE 0.071; Mean of Level-1: -1.42 (-3.38, +0.54), ECE 0.092; Best single Level-1: -1.57 (-4.00, +0.87), ECE 0.046
- los14_d2: single 0.795 (ECE 0.036); Stack, XGB meta: -3.25 (-7.36, +0.86), ECE 0.092; Stack, XGB meta: -0.21 (-3.58, +3.17), ECE 0.041; Stack, LR meta: +0.44 (-3.08, +3.97), ECE 0.041; Mean of Level-1: -0.47 (-2.99, +2.05), ECE 0.063; Best single Level-1: -0.50 (-4.48, +3.49), ECE 0.028
