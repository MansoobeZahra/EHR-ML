# Verified results from the executed Kaggle run (notebooks/research02_EXECUTED_with_outputs_incl_MICE.ipynb)
All AUROC. "Stacked" = full two-layer EHR-ML ensemble, ONE 5-fold CV run (seed 42), mean +- sd over folds.
"XGB-only" etc. = 5x5 RepeatedStratifiedKFold (25 scores). These two protocols are NOT the same folds (see CLAUDE.md, known issues).

## A. Baseline (stacked, 5-fold)
| Task | AUROC | AvgPrec | BalAcc | F1 | Acc | MCC |
|---|---|---|---|---|---|---|
| MIMIC-IV in-hospital mortality, first 24h of hospital data (n=2533) | 0.8132+-0.0136 | 0.6913 | 0.7031 | 0.5700 | 0.7971 | 0.4581 |
| eICU 72h-window mortality (n=12677) | 0.8279+-0.0091 | 0.6838 | 0.7119 | 0.5770 | 0.8467 | 0.5104 |
| MIMIC LOS>7 (feature window WA=7 days, n=2545) | 0.9464+-0.0122 | 0.7953 | 0.8190 | 0.7195 | 0.9014 | 0.6618 |
| MIMIC LOS>14 (WA=7 days) | 0.9177+-0.0253 | 0.4182 | 0.6197 | 0.3186 | 0.9273 | 0.2957 |
| MIMIC all-admission mortality (exploratory, uses whole stay, leakage-prone) | 0.9828+-0.0062 | 0.9634 | 0.9287 | 0.9042 | 0.9477 | 0.8687 |
eICU windows (stacked): 24h 0.7425 (n=12669), 48h 0.7909 (n=12677), 72h 0.8279 (n=12677).
Cohorts: MIMIC ICD 2,545 episodes (mortality 27.9%, LOS>7 18.4%, LOS>14 6.7%); eICU ICD 12,688 episodes, mortality 22.4%. Original paper: 729 and 11,146.
Class balance in the 24h mortality matrix: pos rate 27.79%.

## B. SAPS-II benchmark (24h mortality)
SAPS-II proxy (logistic regression on SAPS-II variables): 0.794 (paper 0.73). EHR-ML on SAPS-II variables only: 0.728 (paper 0.79). EHR-ML full: 0.813 mean-of-folds
(pooled out-of-fold AUROC 0.808 in Table 2 code, 0.8082 in calibration step; paper 0.85). Ordering differs from the original study (see paper text).

## C. Inter-source (72h eICU window) stacked AUROC, rows = train, cols = validate
| | MIMIC | eICU |
|---|---|---|
| MIMIC | 0.813 | 0.807 |
| eICU | 0.756 | 0.828 |
Paper: MIMIC->MIMIC 0.85, MIMIC->eICU 0.84, eICU->MIMIC 0.83, eICU->eICU 0.89. 50 features are shared between the MIMIC matrix and the 55-feature eICU matrix.

## D. LOS window sweep (stacked) WA = feature window length in days from ICU admission
| WA | LOS>7 | LOS>14 |
|---|---|---|
| 1 | 0.7627 | 0.7440 |
| 2 | 0.8361 | 0.8387 |
| 3 | 0.8787 | 0.8329 |
| 4 | 0.9033 | 0.8708 |
| 5 | 0.9232 | 0.8845 |
| 6 | 0.9344 | 0.9021 |
| 7 | 0.9464 | 0.9177 |
The notebook picked WA=7 as "best" (max AUROC; paper reports 0.96 / 0.95). AUROC rises as the window approaches the 7-day horizon: leakage red flag.

## E. Ablation (Step 15), 5x5 repeated CV, Wilcoxon vs XGB-only. Delta in AUROC vs XGB-only; *** p<.001
| Task | Stacked | XGB-only | Vitals only | Labs only | Mean-agg only | No clipping |
|---|---|---|---|---|---|---|
| MIMIC 24h mort | 0.8132 | 0.8333+-0.0157 | 0.8000 (-0.0333 ***) | 0.7786 (-0.0547 ***) | 0.8281 (-0.0052, p=.0009) | 0.8336 (+0.0003, p=.54) |
| eICU 72h mort | 0.8279 | 0.8530+-0.0087 | 0.7923 (-0.0607 ***) | 0.8064 (-0.0466 ***) | 0.7932 (-0.0598 ***) | 0.8529 (-0.0001, p=.26) |
| eICU 48h mort | 0.7909 | 0.8251+-0.0081 | 0.7650 (-0.0600 ***) | 0.7768 (-0.0482 ***) | 0.7745 (-0.0506 ***) | 0.8252 (+0.0001, p=.72) |
| eICU 24h mort | 0.7425 | 0.7781+-0.0093 | 0.7288 (-0.0494 ***) | 0.7224 (-0.0558 ***) | 0.7441 (-0.0341 ***) | 0.7775 (-0.0006, p=.043) |
| MIMIC LOS>7 | 0.9464 | 0.9522+-0.0093 | 0.9317 (-0.0205 ***) | 0.9171 (-0.0352 ***) | 0.9032 (-0.0491 ***) | 0.9521 (-0.0002, p=.38) |
| MIMIC LOS>14 | 0.9177 | 0.9382+-0.0156 | 0.9133 (-0.0249 ***) | 0.8963 (-0.0418 ***) | 0.9038 (-0.0343 ***) | 0.9378 (-0.0004, p=.28) |
Feature counts: MIMIC 24h 120 (vitals 35, labs 85, mean-agg 24); eICU 55 (15, 40, 11); LOS 155 (35, 120, 31).
Stacked minus XGB-only: -0.0201, -0.0251, -0.0342, -0.0357, -0.0058, -0.0205 (0.6 to 3.6 points; up to 4.6% relative).

## F. CatBoost
Standalone vs XGB-only (Step 16a): MIMIC24h 0.8371 vs 0.8333 (+0.0037, p=.0125 *); eICU72 0.8512 vs 0.8530 (p=.0001); eICU48 0.8211 vs 0.8251; eICU24 0.7748 vs 0.7781;
LOS>7 0.9507 vs 0.9522 (p=.016 *); LOS>14 0.9372 vs 0.9382 (ns, p=.264). Standalone CatBoost beats the STACKED ensemble on all six tasks.
CatBoost fixed config: iterations=150, depth=4, lr=0.05.
Ensemble variants (single 5-fold, no significance tests): original [XGB,LR,LGBM,MLP]: 0.8132 / 0.8279 / 0.9464 (MIMIC24h / eICU72 / LOS>7);
+CatBoost as 5th Level-1 (Step 16b): 0.8126 / 0.8318 / 0.9425; MLP replaced by CatBoost (Step 21): 0.8147 / 0.8242 / 0.9427.

## G. Preprocessing (Step 17), standalone XGB, 5x5 CV, tasks MIMIC24h / eICU72 / LOS>7
Baseline (QC clip 1-99th pct, median): 0.8345 / 0.8529 / 0.9526. No clipping: 0.8335 / 0.8531 / 0.9520. IQR x1.5: 0.8344 / 0.8531 / 0.9516.
P1-P99 clip is IDENTICAL to baseline by construction. KNN k=5: 0.8334 / 0.8518 (p=.019) / 0.9525. MICE: 0.8369 (p=.071) / 0.8531 (p=.31) / skipped (runtime).
MICE removed from the latest notebook (no benefit, 2-4x runtime).

## H. Coverage threshold (Step 18)
MIMIC 24h mortality: 70% -> 155 features AUROC 0.8374 (p=.034); 80% -> 120, 0.8345; 90% -> 105, 0.8260 (p=.0004).
"MIMIC LOS>7" in this step: 0.8000 / 0.7885 / 0.7825 BUT it was built with a 1-day window anchored at admittime, not the WA=7 matrix used elsewhere (inconsistent).
eICU 72h: 55 features and 0.8529 at all thresholds. INVALID as evidence: matrix was pre-filtered at 80% (see REVIEWS.md).

## I. Calibration (Step 19), pooled out-of-fold predictions, 5-fold. Brier / ECE (10 bins)
| Task | Stacked (pooled AUROC) | XGB-only | CatBoost |
|---|---|---|---|
| MIMIC 24h | 0.162 / 0.124 (0.8082) | 0.135 / 0.023 (0.8334) | 0.134 / 0.019 (0.8360) |
| eICU 72h | 0.121 / 0.068 (0.8276) | 0.108 / 0.013 (0.8529) | 0.109 / 0.017 (0.8513) |
| MIMIC LOS>7 | 0.076 / 0.058 (0.9456) | 0.066 / 0.012 (0.9508) | 0.067 / 0.021 (0.9482) |
XGB-only calibration used XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, ...) (make_xgb_calib).

## J. SHAP-ranked selection (Step 20). LEAKY: ranking computed once on all data
| Task | All | Top10 | Top20 | Top30 | Top40 |
|---|---|---|---|---|---|
| MIMIC 24h | 0.8345 | 0.8168 | 0.8379 | 0.8411 | 0.8395 |
| eICU 72h | 0.8533 | 0.8396 | 0.8525 | 0.8534 | 0.8532 |
| LOS>7 | 0.9523 | 0.9384 | 0.9505 | 0.9527 | 0.9543 |
| LOS>14 | 0.9372 | 0.9369 | 0.9454 | 0.9454 | 0.9467 |

## K. SHAP (Step 14), representative XGB, mean |SHAP|, 2,000-row sample
MIMIC 24h: labs_lactate_last 0.33, vitals_spo2_last 0.15, vitals_gcs_verbal_last 0.15, vitals_gcs_verbal_avg 0.15, vitals_temperature_f_min 0.13, vitals_spo2_min 0.12, labs_bun_last 0.11.
eICU 72h: vitals_heart_rate_last 0.31, labs_wbc_last 0.24, labs_bun_last 0.23, vitals_spo2_min 0.22, vitals_spo2_last 0.18, labs_plateletss_last 0.16, labs_potassium_last 0.16.
LOS>7: gcs_verbal_avg 0.44, gcs_verbal_last 0.42, labs_ph_max 0.42. LOS>14: gcs_verbal_last 0.65, gcs_verbal_avg 0.59, labs_ph_max 0.40.
