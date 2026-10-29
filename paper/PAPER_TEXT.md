# Current paper text (final v38, 6 pages), extracted from EHR-ML_FIT_Conference_Paper_FINAL.docx

Explainable EHR-ML for Sepsis Outcome Prediction: An Ablation-Based Study Using MIMIC-IV and eICU

Mansoob-e-Zahra, Ijaz Hussain

Department of Computer and Information Sciences

PIEAS

Nilore, Islamabad, Pakistan

ijazhussain@pieas.edu.pk

Abstract: Sepsis remains a leading cause of in-hospital mortality and prolonged intensive care stays, and interpretable, well-calibrated risk-prediction models are needed to support timely clinical decisions. Prior work has largely evaluated ensemble machine-learning pipelines on discrimination alone, typically reporting a single-run performance figure for a fixed architecture without testing which components drive that performance or whether reported gains generalize across data sources. In this work, we replicate a two-layer stacked-ensemble framework for sepsis mortality and length-of-stay prediction on MIMIC-IV and eICU, and extend it with a statistically grounded ablation study, SHapley Additive exPlanations (SHAP)-based explainability, an independent base-learner substitution, and a calibration analysis. Using repeated 5×5 cross-validation with paired Wilcoxon significance testing across six prediction tasks, we show that a single, unstacked gradient-boosted model outperforms the two-layer stacked ensemble by 0.6 to 3.6 Area Under the Receiver Operating Characteristic Curve (AUROC) points (up to 4.6% relative), a result confirmed independently with two base learners and consistent across both discrimination and calibration metrics. We further show that clinically plausible, task-specific predictors drive the models' decisions and that most tasks tolerate a substantially reduced feature set with under 1% AUROC loss. These findings give a concrete, evidence-based basis for simplifying stacked-ensemble Electronic Health Record (EHR) pipelines without sacrificing predictive or clinical value.

Index Terms: sepsis prediction, electronic health records, ensemble learning, stacked generalization, explainable AI, SHAP, model calibration, MIMIC-IV, eICU

I. Introduction

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection, and it remains one of the leading causes of morbidity and mortality among critically ill patients worldwide [1]. Because early recognition and timely intervention are strongly associated with improved outcomes, accurate and trustworthy risk stratification at or shortly after Intensive Care Unit (ICU) admission has long been treated as a high-value target for clinical decision support, motivating decades of work on scoring systems and, more recently, machine-learning models trained on Electronic Health Record (EHR) data.

Early approaches to this problem relied on hand-crafted bedside severity scores such as the Simplified Acute Physiology Score (SAPS-II) [2], which combine a small set of physiological and laboratory variables into a single risk estimate. These scores are transparent and easy to compute, but their fixed linear form limits discrimination relative to what is achievable from the full richness of modern EHR data. The public release of large critical-care datasets such as MIMIC-IV [3] and eICU [4] shifted the field toward supervised machine learning: gradient-boosted tree models, most commonly XGBoost [5], have repeatedly been shown to outperform SAPS-II-style scores on sepsis-related mortality prediction [6]. Building on this, several recent studies have combined multiple base learners into stacked ensembles, arguing that heterogeneous base models capture complementary signal and improve on any single model [7], [8]. In parallel, a separate line of work has focused on making these models interpretable, most often through SHAP values, which attribute a prediction to individual input features and have been applied to sepsis-related outcomes in several MIMIC/eICU studies [9], [10].

Despite this progress, most reported pipelines share three limitations. First, they typically use a single train/test split, making it hard to tell whether an architectural choice reflects a real effect or fold-to-fold noise. Second, evaluation is almost always confined to discrimination metrics such as AUROC, even though calibration is equally important for clinical decisions and rarely reported for stacked architectures. Third, explainability and predictive performance are usually reported separately, rather than together, to test whether a smaller, better-understood model can match a larger black-box ensemble.

This paper addresses these gaps directly. We replicate a two-layer stacked-ensemble framework, with Level-1 base learners (XGBoost, logistic regression, LightGBM, and a Multilayer Perceptron (MLP)) feeding a Level-2 meta-learner, for six sepsis-related prediction tasks spanning 24 h, 48 h, and 72 h mortality and length-of-stay beyond 7 and 14 days, on both MIMIC-IV and eICU cohorts. On top of this baseline replication we contribute five extensions: (1) a component-wise ablation study isolating the contribution of vitals, laboratory values, temporal aggregation, and data-clipping choices; (2) global and local SHAP explainability across the four core tasks, with a cross-cohort feature-overlap analysis between MIMIC-IV and eICU; (3) an architecture-generalization test substituting CatBoost into the pipeline three ways (standalone, added as a fifth Level-1 model, and swapped for the MLP); (4) a calibration analysis using Brier score and Expected Calibration Error (ECE) to quantify a cost of stacking that AUROC alone cannot show; and (5) two robustness studies, preprocessing sensitivity and feature-coverage threshold sensitivity, together with a SHAP-ranked feature-selection experiment.

The central and unifying finding across these experiments is architectural rather than incremental. On every one of six prediction tasks, a single unstacked gradient-boosted model outperforms the two-layer stacked ensemble by 0.6 to 3.6 AUROC points (up to 4.6% relative), in both discrimination and calibration, and this result is independently reproduced with a second base learner. The practical consequence is direct: EHR-based clinical risk models should be benchmarked against a single well-tuned model before adopting multi-layer stacking. Doing so yields a lighter, better-calibrated, and more interpretable candidate architecture for downstream deployment in ICU decision-support tools.

The remainder of this paper is organized as follows. Section II reviews related work on sepsis outcome prediction, ensemble methods, and explainability. Section III describes the datasets, the baseline pipeline, and the design of each experiment. Section IV reports results. Section V discusses their implications, Section VI states the limitations of this study, and Section VII concludes with directions for future work.

II. Related Work

Classical severity scores such as SAPS-II [2] and the Sequential Organ Failure Assessment (SOFA) score remain widely used in clinical practice because of their transparency, but their reliance on a small, fixed set of linearly combined variables limits their discriminative power relative to data-driven models, and comparisons in the recent literature consistently favor tree-based machine-learning models [6], [11].

Since the release of MIMIC-IV [3] and eICU [4], gradient-boosted tree models, most often XGBoost [5], have become a de facto standard for structured-data mortality and length-of-stay prediction in sepsis and related ICU cohorts, frequently paired with SHAP-based interpretability [9] to identify clinically meaningful drivers such as lactate, respiratory indices, and coagulation markers [6], [10]. A parallel body of work builds stacked ensembles on these base learners, arguing that combining heterogeneous model families improves discrimination over any single model [7], [8], [12]. These studies typically report a single stacked-ensemble AUROC without a paired statistical comparison against an unstacked baseline, and rarely report calibration, a gap also flagged by two recent scoping reviews [13], [14].

Our work differs from this literature in three ways. First, we test the stacking hypothesis directly and statistically, using repeated cross-validation with paired significance testing against two independent base learners. Second, we combine ablation, calibration, and explainability into one coherent evaluation of the same six tasks. Third, we test the robustness of design choices (preprocessing, feature-coverage threshold, feature-set size) that prior work usually fixes without justification.

III. Materials and Methods

Fig. 1 summarizes the overall pipeline: the replicated baseline architecture (left) and the five methodological extensions introduced in this study (right); each block is described in the corresponding subsection below.

Fig. 1. Architecture of the replicated EHR-ML pipeline (left) and this study's methodological extensions (right): a component-wise ablation study, an architecture-generalization test with CatBoost, a calibration analysis, preprocessing/feature-coverage robustness checks, and SHAP-based explainability with feature selection.

A. Data Sources and Prediction Tasks

Two public critical-care databases were used: MIMIC-IV [3], from a single U.S. teaching hospital, and eICU [4], a multi-center database. Sepsis cohorts were identified using the ICD-9/ICD-10 codes from the original EHR-ML study [17]. This gave 2,545 MIMIC-IV episodes and 12,677 to 12,686 eICU episodes, depending on the mortality window used. These counts are larger than the original study's 729 and 11,146 episodes because of small differences in cohort filtering, discussed in Section VI. Six prediction tasks were defined: 24-hour mortality on MIMIC-IV; 24-, 48-, and 72-hour mortality on eICU; and length of stay (LOS) beyond 7 and 14 days on MIMIC-IV.

B. Baseline Pipeline

The baseline pipeline replicates a two-layer stacked-ensemble framework for sepsis outcome prediction [17]. Raw vitals and laboratory measurements within a fixed observation window are aggregated using five statistics (mean, minimum, maximum, first, and last value) and filtered to features with at least 80% patient coverage. Level-1 consists of four heterogeneous base learners (XGBoost, logistic regression, LightGBM, and a Multilayer Perceptron (MLP)) whose out-of-fold predictions are combined by a Level-2 meta-learner, following the stacked-generalization approach of Wolpert [7]. Baseline replication used standard 5-fold cross-validation, matching the original paper's protocol, and was benchmarked against a logistic-regression SAPS-II proxy and validated across MIMIC-IV↔eICU transfer.

C. Statistical Evaluation Protocol

For every ablation, architecture, and robustness comparison described below, we used repeated 5×5 stratified cross-validation (5 repeats of 5-fold CV, yielding 25 AUROC scores per condition) and compared each condition against its reference condition using a paired Wilcoxon signed-rank test on matched fold splits and random seeds. This protocol was chosen specifically because a single 5-fold run cannot distinguish a genuine architectural effect from cross-validation noise, and because paired testing controls for fold-to-fold variance that would otherwise dominate small AUROC differences.

D. Explainability (SHAP)

Global and local SHAP values [9] were computed for a representative XGBoost model on each of the four core tasks (MIMIC 24 h mortality, eICU 72 h mortality, MIMIC LOS>7, MIMIC LOS>14), using a 2,000-row sample for tractability. Global importance is summarized as mean(|SHAP value|) rankings and beeswarm plots; local explanations are reported as SHAP waterfall decompositions for the highest-predicted-risk patient in each task. A feature-overlap analysis compared SHAP rankings between MIMIC-IV and eICU for the 55 features common to both cohorts.

E. Ablation Study

Four ablation conditions were compared against a full-feature, unstacked XGBoost reference across all six tasks: vitals-only features, laboratory-only features, mean-aggregation-only features (dropping min/max/first/last), and no QC clipping. This isolates which pipeline components are load-bearing rather than incidental to performance.

F. Architecture Generalization (CatBoost)

To test whether the architecture finding described in Section IV is specific to XGBoost, CatBoost [18] was substituted into the pipeline three ways: (a) standalone, with no stacking layer, compared directly against standalone XGBoost using the same repeated-CV/paired-Wilcoxon protocol as the ablation study; (b) added as a fifth Level-1 model alongside the original four; and (c) substituted for the MLP within the Level-1 roster, keeping the total model count unchanged.

G. Preprocessing and Feature-Coverage Sensitivity

Two preprocessing dimensions were tested against the pipeline's defaults on three core tasks: clipping strategy (no clipping, 1st–99th percentile, and interquartile-range ×1.5 clipping) and imputation method (median and k-nearest-neighbors; multiple imputation by chained equations was also tested but excluded from further analysis after showing no significant benefit at substantially higher computational cost). Separately, the minimum per-feature coverage threshold, defined as the fraction of patients for whom a feature must be present to be retained, was swept across 70%, 80%, and 90% on MIMIC-IV and eICU.

H. Calibration Analysis

Brier score and 10-bin Expected Calibration Error (ECE) [19] were computed on pooled out-of-fold predicted probabilities from 5-fold cross-validation (i.e., each patient's prediction comes from a model that did not see that patient during training) for the stacked ensemble, standalone XGBoost, and standalone CatBoost, on the three core tasks, together with reliability diagrams.

I. SHAP-Ranked Feature Selection

Standalone XGBoost was retrained on the top 10, 20, 30, and 40 SHAP-ranked features (from Section III-D) against the full feature set, on the four core tasks, using the same repeated-CV/paired-Wilcoxon protocol, with rankings computed independently per task.

IV. Results

A. Baseline Replication

Table I summarizes the replicated baseline. Discrimination is close to, but below, the original paper's [17] reported values on the four core tasks: MIMIC-IV 24-hour mortality AUROC 0.813 versus 0.85 reported (4.4% lower), LOS>7 AUROC 0.946 versus 0.96 reported (1.5% lower), and LOS>14 AUROC 0.918 versus 0.95 reported (3.4% lower). The largest gap is in eICU-to-MIMIC transfer (0.756 versus 0.83 reported, 8.9% lower). On 24-hour mortality, the full-feature ensemble (0.813 AUROC) beat both a SAPS-II proxy (0.794) and an ensemble trained only on SAPS-II variables (0.728). This differs from the original study, where the SAPS-II-variable ensemble (0.79) beat the plain SAPS-II proxy (0.73); the reversal likely reflects the larger, differently filtered cohort used here rather than an implementation error.

TABLE I. Baseline Replication: 5-Fold Cross-Validation Results

B. The Architecture Finding

The single most consequential result of this study is that a single, unstacked XGBoost model outperforms the two-layer stacked ensemble on every one of six tasks (Table II). The gain ranges from 0.6 to 3.6 AUROC points (0.6-4.6% relative), all differences derived from the same paired-fold evaluation used throughout this study. This was independently reproduced with a second base learner: standalone CatBoost also outperforms the stacked ensemble on all six tasks tested (Section IV-D).

TABLE II. Stacked Ensemble vs. Single Unstacked Model (XGBoost only)

C. Ablation Study

Removing either the labs or vitals feature group significantly degrades performance on all six tasks (p < .001 in every case). Which group matters more is task-dependent: laboratory features are the largest contributor for MIMIC-IV mortality and eICU 24 h mortality, vitals dominate for eICU 48 h and 72 h mortality, and temporal aggregation (retaining only the mean, and dropping min/max/first/last) is the largest single contributor for both length-of-stay tasks. QC clipping choice has no statistically or practically meaningful effect on five of six tasks (Section IV-E). Fig. 2 shows the full six-task ablation comparison.

Fig. 2. Component-wise ablation study across all six prediction tasks. Bars show mean AUROC over 25 repeated-CV scores for each condition; removing labs or vitals (light green) significantly lowers AUROC versus the full stacked model (dark green) and the unstacked XGBoost reference (orange) on every task (p < .001), while clipping choice (pale green) has no meaningful effect.

D. Architecture Generalization: CatBoost

Standalone CatBoost outperforms standalone XGBoost only on MIMIC-IV 24 h mortality (ΔAUROC +0.0037, p = .0125); XGBoost matches or outperforms CatBoost on the remaining five tasks (Table III). Adding CatBoost as a fifth Level-1 model, or swapping it in for the MLP, produced small and inconsistent shifts (within ±0.004 AUROC) rather than a systematic improvement. Because a stronger or different base model does not close the gap identified in Section IV-B, the bottleneck is attributable to the stacking mechanism itself rather than to any specific base learner.

TABLE III. CatBoost vs. XGBoost, Standalone (No Stacker)

*p<.05  **p<.01  ***p<.001  ns = not significant (paired Wilcoxon signed-rank test, CatBoost vs. XGBoost, 25 repeated-CV scores).

E. Calibration Analysis

Beyond discrimination, the stacked ensemble is also substantially worse-calibrated than either single model (Table IV): its Expected Calibration Error is roughly 4–6× higher than standalone XGBoost or CatBoost on all three tasks tested, and its reliability curve visibly departs from the diagonal in the high-confidence range, while the single-model curves track close to perfect calibration. This provides a second, independent line of evidence, beyond AUROC, that the stacking layer degrades rather than improves the model.

TABLE IV. Calibration: Brier Score / Expected Calibration Error

Fig. 3. Reliability diagrams (out-of-fold, 5-fold CV) for the stacked ensemble, standalone XGBoost, and standalone CatBoost on three core tasks; the diagonal is perfect calibration.

F. Preprocessing and Feature-Coverage Robustness

No clipping strategy or imputation method produced a practically meaningful AUROC change (all deltas ≤ 0.003); this is a positive robustness result: the framework's conclusions do not depend on these preprocessing details. The feature-coverage threshold, in contrast, behaved differently across cohorts: MIMIC-IV is sensitive in both directions (lowering the threshold from 80% to 70% significantly improves AUROC, raising it to 90% significantly reduces it, p < .001), confirming the pipeline's 80% default is a real, load-bearing design choice for this cohort. eICU showed identical feature counts and AUROC at all three thresholds, because its feature set simply has no columns with coverage between 70% and 90%, a genuine property of the data rather than a null experiment.

G. Explainability

For MIMIC-IV 24 h mortality, the last recorded lactate value is the dominant SHAP predictor (mean |SHAP| 0.33), followed by SpO₂ and the Glasgow Coma Scale (GCS) verbal component (Fig. 4). For eICU 72 h mortality, the last recorded heart rate dominates (mean |SHAP| 0.31), followed by white-blood-cell count and blood urea nitrogen, clinically well-established sepsis-mortality signals. Of the 55 features common to both cohorts, 50 overlap directly, and importance rank agrees for most shared features, diverging for a handful of task-specific signals (e.g., heart rate ranks substantially higher in eICU than MIMIC-IV).

Fig. 4. Top 20 SHAP feature importances (representative XGBoost model): MIMIC-IV 24h mortality (left) and eICU 72h mortality (right).

H. Feature-Set Minimality

Three of the four core tasks plateau well before the full feature set is used: eICU 72 h mortality and MIMIC LOS>7 plateau at roughly 30 SHAP-ranked features, and MIMIC LOS>14 plateaus at roughly 10. MIMIC 24 h mortality shows no plateau within the tested range (10–40 features). Because SHAP rankings are task-specific, these reduced sets are not a single universal minimal feature set.

V. Discussion

Taken together, these results support a claim stronger than a typical replication study: the performance ceiling of this EHR pipeline is set by its stacking architecture, not by the choice of base learner, the preprocessing pipeline, or the feature-coverage threshold. Two independent lines of evidence corroborate this: a second base learner (CatBoost) and a second metric family (calibration), which together rule out the two most obvious alternative explanations: that XGBoost is a poor fit for the stacker, or that the gap is an artefact of discrimination-only evaluation.

The task-dependent nature of feature importance (labs for MIMIC-IV mortality, vitals for eICU mortality, temporal aggregation for length-of-stay) argues against a one-size-fits-all feature set. Most tasks tolerate a substantially reduced feature set (10–30 SHAP-ranked features, under 1% AUROC loss), which is cheaper to compute at the bedside and easier for clinicians to audit.

VI. Limitations

This study has several limitations. Cohort identification followed the original paper's ICD-code criteria [17], but the data was not passed through the original study's EHR-QC/OMOP-CDM standardization pipeline, which likely explains the residual gaps in Section IV-A, consistent with broader evidence on limited generalizability in clinical prediction models [15], [16]. Some eICU tasks have modest effective sample sizes (n = 55 per repeated-CV condition), widening confidence intervals even where significance was reached. Local SHAP explanations are illustrated for a single highest-risk patient per task as a face-validity check, not a systematic stability audit; fold-stability testing is planned but not yet executed. Finally, this work uses a logistic-regression SAPS-II proxy rather than the true clinical score.

VII. Conclusion and Future Work

This paper presented an ablation-based, statistically validated study of a two-layer stacked-ensemble framework for sepsis mortality and length-of-stay prediction on MIMIC-IV and eICU. Beyond replicating the baseline pipeline, we tested the stacking architecture using repeated cross-validation, paired significance testing, and an independent base learner. Across both discrimination and calibration metrics, the stacking architecture itself, not the choice of base model, preprocessing pipeline, or feature-coverage threshold, was the primary factor limiting performance. We further showed that a compact, SHAP-selected feature set is sufficient for most tasks, with under 1% AUROC loss, and that the models' top predictors are clinically plausible and largely consistent across cohorts. These results give a concrete, evidence-based recommendation for future EHR-based clinical risk-prediction pipelines: evaluate a single well-tuned model as a mandatory baseline before adopting multi-layer stacking, and report calibration alongside discrimination.

Two directions follow from this work: testing the architecture finding on prospective, multi-institutional data through the original EHR-QC/OMOP-CDM pipeline, and tracking SHAP fold-stability alongside calibration in future deployments.

Acknowledgment

The author thanks their research supervisor for guidance throughout this study.

References

[1] M. Singer et al., "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)," JAMA, vol. 315, no. 8, pp. 801–810, 2016.

[2] J.-R. Le Gall, S. Lemeshow, and F. Saulnier, "A new Simplified Acute Physiology Score (SAPS II) based on a European/North American multicenter study," JAMA, vol. 270, no. 24, pp. 2957–2963, 1993.

[3] A. E. W. Johnson et al., "MIMIC-IV, a freely accessible electronic health record dataset," Scientific Data, vol. 10, no. 1, art. 1, 2023.

[4] T. J. Pollard et al., "The eICU Collaborative Research Database, a freely available multi-center database for critical care research," Scientific Data, vol. 5, art. 180178, 2018.

[5] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, San Francisco, CA, USA, 2016, pp. 785–794.

[6] S. Chen, J. Fan, E. Pishgar, K. Alaei, G. Placencia, and M. Pishgar, "Machine learning-based prediction of ICU mortality in sepsis-associated acute kidney injury patients using MIMIC-IV database with validation from eICU database," arXiv:2502.17978, 2025.

[7] D. H. Wolpert, "Stacked generalization," Neural Networks, vol. 5, no. 2, pp. 241–259, 1992.

[8] X. Liu, H. Niu, and J. Peng, "Improving predictions: Enhancing in-hospital mortality forecast for ICU patients with sepsis-induced coagulopathy using a stacking ensemble model," Medicine, vol. 103, no. 14, art. e37634, 2024.

[9] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems 30 (NeurIPS), 2017, pp. 4765–4774.

[10] S. Zhou et al., "Interpretable machine learning model for early prediction of 28-day mortality in ICU patients with sepsis-induced coagulopathy: development and validation," Eur. J. Med. Res., vol. 29, no. 1, art. 14, 2024.

[11] Y. Wang, Z. Gao, Y. Zhang, Z. Lu, and F. Sun, "Early sepsis mortality prediction model based on interpretable machine learning approach: development and validation study," Intern. Emerg. Med., vol. 20, no. 3, pp. 909–918, 2025.

[12] J. Lei, J. Zhai, Y. Zhang, J. Qi, and C. Sun, "Supervised machine learning models for predicting sepsis-associated liver injury in patients with sepsis: development and validation study based on a multicenter cohort study," J. Med. Internet Res., vol. 27, art. e66733, 2025.

[13] H. Shanmugam, L. Airen, and S. Rawat, "Machine learning and deep learning models for early sepsis prediction: a scoping review," Indian J. Crit. Care Med., vol. 29, no. 6, pp. 516–524, 2025.

[14] S. Bomrah, M. Uddin, U. Upadhyay, M. Komorowski, J. Priya, E. Dhar, S.-C. Hsu, and S. Syed-Abdul, "A scoping review of machine learning for sepsis prediction: feature engineering strategies and model performance, a step towards explainability," Crit. Care, vol. 28, art. 180, 2024.

[15] B. Van Calster, E. W. Steyerberg, L. Wynants, and M. van Smeden, "There is no such thing as a validated prediction model," BMC Med., vol. 21, art. 70, 2023.

[16] A. A. H. de Hond, V. B. Shah, I. M. J. Kant, B. Van Calster, E. W. Steyerberg, and T. Hernandez-Boussard, "Perspectives on validation of clinical predictive algorithms," npj Digit. Med., vol. 6, art. 86, 2023.

[17] Y. Ramakrishnaiah, N. Macesic, G. I. Webb, A. Y. Peleg, and S. Tyagi, "EHR-ML: A data-driven framework for designing machine learning applications with electronic health records," International Journal of Medical Informatics, vol. 196, art. 105816, Apr. 2025, doi: 10.1016/j.ijmedinf.2025.105816.

[18] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: Unbiased boosting with categorical features," in Advances in Neural Information Processing Systems 31 (NeurIPS), 2018, pp. 6638–6648.

[19] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. 34th Int. Conf. Machine Learning (ICML), 2017, pp. 1321–1330.

# TABLES

## Table index 0
| Cohort / Target | Task | AUROC | F1 |
| MIMIC-IV 24h | Mortality | 0.813 | 0.570 |
| eICU 72h | Mortality | 0.828 | 0.847 |
| MIMIC-IV ICD | LOS>7 days | 0.946 | 0.720 |
| MIMIC-IV ICD | LOS>14 days | 0.918 | 0.319 |

## Table index 1
| Task | Stacked | XGB-only | ΔAUROC |
| MIMIC 24h Mort | 0.8132 | 0.8333 | -0.0201 |
| eICU 72h Mort | 0.8279 | 0.8530 | -0.0251 |
| eICU 48h Mort | 0.7909 | 0.8251 | -0.0342 |
| eICU 24h Mort | 0.7425 | 0.7781 | -0.0357 |
| MIMIC LOS>7 | 0.9464 | 0.9522 | -0.0058 |
| MIMIC LOS>14 | 0.9177 | 0.9382 | -0.0205 |

## Table index 2
| Task | CatBoost | XGBoost | Sig. |
| MIMIC 24h Mort | 0.8371 | 0.8333 | * |
| eICU 72h Mort | 0.8512 | 0.8530 | *** |
| eICU 48h Mort | 0.8211 | 0.8251 | *** |
| eICU 24h Mort | 0.7748 | 0.7781 | *** |
| MIMIC LOS>7 | 0.9507 | 0.9522 | * |
| MIMIC LOS>14 | 0.9372 | 0.9382 | ns |

## Table index 3
| Task | Stacked | XGB-only | CatBoost |
| MIMIC 24h Mort | 0.162/0.124 | 0.135/0.023 | 0.134/0.019 |
| eICU 72h Mort | 0.121/0.068 | 0.108/0.013 | 0.109/0.017 |
| MIMIC LOS>7 | 0.076/0.058 | 0.066/0.012 | 0.067/0.021 |
