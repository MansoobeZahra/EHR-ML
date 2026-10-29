# Text, tables and figures of paper v3 (exec'd by build_paper_v3.py, which provides the
# XML helpers). Every number comes from paper_facts.F or is read from the result files.
import sys as _sys
_sys.path.insert(0, str(HERE))
from paper_facts import F, S as FS, D as FD, MAIN4, SINGLE, TSINGLE, DEF, FIX, LRM, TFIX, TLRM, MEAN, BEST, pts

DASH = '−'


def p2(v):
    return f'{100 * v:+.2f}'.replace('-', DASH)


def cip(r, newline=False):
    return f'{p2(r.delta)}{chr(10) if newline else " "}({p2(r.CI95_low)}, {p2(r.CI95_high)})'


KEYS = [('mimic24', 'MIMIC-IV mortality (24 h data)'), ('eicu72', 'eICU mortality (72 h data)'),
        ('los7_d2', 'LOS>7 (day-2 landmark)'), ('los14_d2', 'LOS>14 (day-2 landmark)')]

# --------------------------------------------------------------------- Table I
T2_METHODS = [('Stack, default XGBoost meta*', DEF), ('Stack, fixed-config. XGBoost meta', FIX),
              ('Stack, LR meta', LRM), ('Mean of Level-1 outputs', MEAN), ('Best Level-1 model', BEST)]
t2_rows = [['Single XGBoost (AUROC)'] + sum(
    [[f"{FS('main', k).loc[SINGLE, 'auc']:.3f} (ref.)", f"{FS('main', k).loc[SINGLE, 'ece']:.3f}"]
     for k, _ in KEYS], [])]
for lab, m in T2_METHODS:
    row = [lab]
    for k, _ in KEYS:
        row += [cip(FD('main', k, SINGLE).loc[m]), f"{FS('main', k).loc[m, 'ece']:.3f}"]
    t2_rows.append(row)
TABLE2 = table([1700] + [1650, 445] * 4,
               [[('Method', 1)] + [(n, 2) for _, n in KEYS],
                [('', 1)] + [('ΔAUROC, points (95% CI)', 1), ('ECE', 1)] * 4],
               t2_rows)

# -------------------------------------------------------------------- Table II
t3_rows = []
for k, n in KEYS:
    d = FD('tuned', k, TSINGLE)
    t3_rows.append([n.replace(' mortality', '').replace(' landmark', ''), f"{FS('tuned', k).loc[TSINGLE, 'auc']:.3f}",
                    cip(d.loc[DEF], True), cip(d.loc[TFIX], True), cip(d.loc[TLRM], True)])
TABLE3 = table([950, 650, 1100, 1100, 1100],
               [[('Task', 1), ('Tuned single XGBoost', 1), ('Stack, default XGBoost meta', 1),
                 ('Stack, tuned XGBoost meta', 1), ('Stack, tuned LR meta', 1)]],
               t3_rows)

TABLE1 = table([2700, 1100, 1100],
               [[('Task (protocol P1)', 1), ('AUROC', 1), ('F1', 1)]],
               [['MIMIC-IV mortality (24 h data)', '0.813', '0.570'],
                ['eICU mortality (72 h data)', '0.828', '0.577']])

# ----------------------------------------------------------------- shorthand facts
f = F
tw = f['tuned_stack']
nd = f['nodeath']
ab = f['abl']
worse = f['reg_worse']
assert len(worse) == 5 and not f['reg_better'] and f['def_all_sigbetter'] == 0
assert [w[:2] for w in worse] == [('main', 'los7_d1'), ('main', 'los7_d1'), ('tuned', 'eicu72'), ('tuned', 'eicu72'), ('div', 'los7_d2')]
assert not f['lr_sig'] and not f['fix_sig'] and f['cat_sig'] == 0
assert f['tune_gain_sig'] == ['eicu72'] and f['tuned_vs_fixed_better'] == [('mimic24', TLRM, '+0.8')]
assert tw['eicu72'][TFIX][1] and tw['eicu72'][TLRM][1] and not tw['mimic24'][TFIX][1]
assert not tw['mimic24'][TLRM][1] and not tw['los7_d2'][TFIX][1] and not tw['los14_d2'][TLRM][1]
assert f['sel']['eicu72'][20][1] and not f['sel']['mimic24'][20][1] and not f['sel']['eicu72'][30][1]
assert not f['sel']['mimic24'][30][1] and f['sel']['mimic24'][10][1] and f['sel']['eicu72'][10][1]
assert nd['mimic24'][DEF][1] and nd['eicu72'][DEF][1] and nd['eicu72'][FIX][1]
assert not nd['mimic24'][FIX][1] and not nd['mimic24'][LRM][1] and not nd['eicu72'][LRM][1]
assert [k for k in MAIN4 if not ab[k]['Vitals only'][1]] == ['los14_d2']
assert all(ab[k]['Labs only'][1] and ab[k]['Mean aggregate only'][1] and not ab[k]['No clipping'][1]
           for k in MAIN4)
assert f['cov_eicu'][0.6][4] and f['cov_eicu'][0.7][4] and not f['cov_mimic'][0.7][4] and not f['cov_mimic'][0.9][4]
assert f['nomiss'][DEF][1] and not f['nomiss'][FIX][1] and not f['nomiss'][LRM][1]
assert f['n729'][DEF][1] == 1 and f['n729'][FIX][1:] == (0, 0) and f['n729'][LRM][1:] == (0, 0)
labs = [float(ab[k]['Labs only'][0].replace(DASH, '-')) for k in MAIN4]
means = [float(ab[k]['Mean aggregate only'][0].replace(DASH, '-')) for k in MAIN4]
noclip = max(abs(float(ab[k]['No clipping'][0].replace(DASH, '-'))) for k in MAIN4)
neg = lambda s: s.replace(DASH, '')    # magnitude of a negative difference in points

ABSTRACT = (
    "Sepsis risk models must be discriminative and well "
    "calibrated. Stacked ensembles are proposed for sepsis outcome prediction from Electronic "
    "Health Record (EHR) data, but are seldom compared with a single model on identical folds, with equal per-model "
    "tuning and a stated meta-learner. We replicate the EHR-ML stacking framework on MIMIC-IV (2,545 "
    "episodes) and eICU (12,688 ICU stays) sepsis cohorts and re-test it on identical patient-grouped "
    "cross-validation folds, reporting paired differences with Nadeau\u2013Bengio corrected 95% "
    "confidence intervals. With the original meta-learner, eXtreme Gradient Boosting (XGBoost) at library "
    f"defaults, the stack scores {f['def_min']} to {f['def_max']} Area Under the Receiver Operating "
    f"Characteristic Curve (AUROC) points ({f['def_rel_min']}% to {f['def_rel_max']}%) below a single XGBoost "
    f"model, is significantly worse in {f['def_all_sigworse']} of {f['n_settings']} settings and is poorly "
    f"calibrated (calibration slope {f['slope_def'][0]} to {f['slope_def'][1]}). With a regularized or equally "
    f"tuned meta-learner, the stack is not significantly different from the comparably configured single model in {f['reg_ns']} of "
    f"{f['n_reg']} comparisons (two meta-learners in {f['n_settings']} settings), worse in {len(worse)} and never significantly better, also with more diverse "
    "base learners or without early deaths. Landmark length-of-stay tasks reach "
    "AUROC 0.77 to 0.80, against 0.94 to 0.95 with the original 7-day window, which overlaps the outcome and "
    "covers all patients, and "
    "leak-free SHapley Additive exPlanations (SHAP) selection keeps AUROC within 0.2 points with 30 features. "
    "In the tested implementation, the stacking penalty reflects the untuned meta-learner rather than stacking "
    "itself; stacked models should report and regularize their meta-learner and be compared with a "
    "single well-configured model.")
INDEX = ("sepsis, electronic health records, stacked generalization, meta-learner, model calibration, "
         "explainable AI, SHAP, MIMIC-IV, eICU")

B = []


def add(*xs):
    B.extend(xs)


add(lead_para('Abstract: ', ABSTRACT), lead_para('Index Terms: ', INDEX))

add(h_sec('I. Introduction'),
    para("Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection, "
         "and it remains one of the leading causes of morbidity and mortality among critically ill patients "
         "worldwide [1], so accurate and well-calibrated risk stratification early in an Intensive Care Unit "
         "(ICU) or hospital stay is a valuable aid to clinical decisions."),
    para("Early approaches relied on bedside severity scores such as the Simplified Acute Physiology Score II "
         "(SAPS II) [2], which combine a small set of physiological and laboratory variables into one risk "
         "estimate. These scores are transparent, but their fixed form limits discrimination compared with what "
         "full Electronic Health Record (EHR) data allow. The release of large critical-care databases such as "
         "MIMIC-IV [3] and eICU [4] moved the field toward supervised machine learning, and gradient-boosted "
         "trees, most often eXtreme Gradient Boosting (XGBoost) [5], are now a common choice for sepsis-related "
         "mortality prediction on these data [6]. Several studies combine multiple base learners into stacked "
         "ensembles [7], [8]. In parallel, "
         "SHapley Additive exPlanations (SHAP) [9] are widely used to explain such models on sepsis outcomes "
         "[6], [10]."),
    para("Three gaps can undermine such comparisons: a stacked ensemble and a single model may be evaluated on "
         "different folds or tuning budgets without reporting the meta-learner; calibration is often omitted; "
         "and length-of-stay feature windows or feature-selection rankings can overlap the outcome period or the "
         "test folds."),
    para("This paper replicates the EHR-ML framework, a two-layer stacked ensemble whose Level-1 base learners "
         "(XGBoost, logistic regression, LightGBM [11] and a Multilayer Perceptron (MLP)) feed a Level-2 "
         "meta-learner, on MIMIC-IV and eICU sepsis cohorts, and re-tests its central design choice under a "
         "controlled protocol. We contribute: (1) a comparison of stacking and single models on identical folds "
         "with five Level-2 options, an equal per-model tuning budget and four Level-1 rosters; (2) calibration analysis "
         "with Expected Calibration Error (ECE) and calibration slope; (3) sensitivity analyses for cohort size, "
         "early deaths, feature coverage and preprocessing; (4) leakage-safe landmark length-of-stay (LOS) "
         "tasks; and (5) SHAP explainability with fold stability and leak-free feature selection."),
    para("In a controlled re-implementation, the stacking penalty comes from the default XGBoost meta-learner "
         "of the original pipeline rather than from stacking itself, and no stacking variant significantly "
         "outperforms a single XGBoost model."))

add(h_sec('II. Related Work'),
    para("Severity scores such as SAPS II [2] and the Sequential Organ Failure Assessment (SOFA) score are "
         "transparent, but data-driven models often discriminate better: an XGBoost model outperformed "
         "conventional scoring systems for 28-day mortality in sepsis-induced coagulopathy [10], and "
         "interpretable gradient-boosted models predict early sepsis mortality [12]."),
    para("On MIMIC-IV and eICU, gradient-boosted trees are often combined with SHAP to identify drivers of "
         "mortality; in sepsis-associated acute kidney injury, an XGBoost model validated on eICU ranked lactate "
         "and respiratory rate among its main predictors [6]. Stacked ensembles gave the best discrimination "
         "among the models compared in one study [8], whereas in a multicenter study stacked and single "
         "gradient-boosted models reached similar internal discrimination [13]. Two recent scoping "
         "reviews describe the wide range of models, features and evaluation choices in machine-learning "
         "sepsis prediction [14], [15]."),
    para("Unlike these studies, we compare stacking with a single model on identical folds under the same "
         "configuration or tuning budget, name the meta-learner, and report calibration and corrected confidence "
         "intervals with discrimination."))

add(h_sec('III. Materials and Methods'),
    para("Fig. 1 summarizes the replicated pipeline (left) and the analyses added in this study (right)."),
    SECT_END_2COL,
    figure(FIG / 'v3_fig1_architecture.png', 4.8, 'rIdV3F1'),
    caption('Fig. 1.', 'The replicated EHR-ML pipeline (left) and the analyses added in this study (right). '
            'CV: cross-validation; LR: logistic regression; kNN: k-nearest neighbors; RF: random forest; '
            'NB: naive Bayes; CI: confidence interval.', 'fig1'),
    SECT_END_1COL)

add(h_sub('A. Data, Cohorts and Prediction Windows'),
    para("We used MIMIC-IV v3.1 [3], from a single U.S. teaching hospital, and eICU v2.0 [4], a multicenter "
         "database, both obtained through PhysioNet under credentialed access. Adult sepsis episodes were "
         "identified with the International Classification of Diseases (ICD) codes of the original EHR-ML "
         "study [16] (ICD-9 995.91, 995.92, 785.52; ICD-10 A41.9, R65.20, R65.21; eICU: the ICD-9 codes). In "
         "MIMIC-IV we kept admissions with sepsis as the principal diagnosis, the first ICU stay per admission, "
         "ICU admission within 48 h of hospital admission and the first sepsis admission per patient, which "
         "gave 2,545 episodes from 2,545 patients (in-hospital mortality 27.9%), 3.5 times the 729 episodes of "
         "[16]. eICU gave 12,688 ICU stays from 10,842 patients (mortality 22.4%). All mortality outcomes are "
         "in-hospital death at any time during the admission. MIMIC-IV mortality is predicted from vitals and "
         "laboratory values recorded in the first 24 h after hospital admission (2,533 episodes with data) and "
         "eICU mortality from the first 72 h of the ICU stay (12,677 stays from 10,835 patients). The 24 h and "
         "72 h refer only to the feature window, not to a prediction horizon; we write MIMIC-IV mortality (24 h "
         "data) and eICU mortality (72 h data). Patients who die inside the feature window remain in the cohort "
         f"({nd['mimic24']['share_of_deaths']}% of MIMIC-IV and {nd['eicu72']['share_of_deaths']}% of eICU "
         "deaths), so a model can use measurements taken shortly before death; the tasks measure discrimination "
         "from early-admission data, not early warning, and Section III-D tests excluding these deaths. eICU "
         "death is a hospital discharge status of Expired; "
         f"{f['nomiss']['excluded']} stays with no recorded status are treated as survivors, as in the "
         "replicated pipeline, and are checked separately. LOS tasks use MIMIC-IV only and predict an ICU "
         "stay longer than 7 or 14 days at a landmark t: features come only from the first t days of the ICU "
         "stay, and only patients still in the ICU at t are included (t = 2: 1,470 episodes, prevalence 31.9% "
         "and 11.6%; t = 1: 2,067 episodes, 22.7% and 8.2%)."),
    h_sub('B. Replicated Pipeline and Original Protocol'),
    para("Vitals and laboratory values in the observation window are aggregated with five statistics (mean, "
         "minimum, maximum, first and last value), giving ten feature subsets; values are clipped at the 1st "
         "and 99th percentiles, and features recorded in fewer than 80% of episodes are removed (120 features "
         "for MIMIC-IV mortality, 55 for eICU, 120 and 145 for the 1- and 2-day LOS landmarks, where coverage "
         "was computed on all 2,545 episodes before the landmark restriction). Level-1 trains XGBoost, logistic "
         "regression (LR), LightGBM and an MLP on each subset (40 models) with inner 5-fold out-of-fold "
         "predictions, and a Level-2 meta-learner combines their probabilities, following stacked "
         "generalization [7]. The original pipeline, which we first ran as published with a single 5-fold "
         "cross-validation (CV), tuned XGBoost by stepwise grid search (3-fold) on one reference subset (vitals "
         "mean) inside each training fold and reused these values for every subset and for LightGBM; LR was "
         "grid-searched, the MLP was fixed (two hidden layers of twice the number of inputs, early stopping), "
         "and the meta-learner was XGBoost with library defaults (100 trees, depth 6, learning rate 0.3), "
         "without tuning. Median imputation was fitted within folds."),
    h_sub('C. Controlled Comparison'),
    para("We use three evaluation protocols and label every result with one of them. P1 is the original "
         "protocol (single 5-fold CV, tuned Level-1, default meta-learner; Section IV-A only). P2 is the controlled "
         "protocol used for all other results unless stated otherwise: identical outer folds (5-fold CV repeated "
         "three times, 15 paired scores per method) and fixed model configurations (Table I, the ablation and the "
         "coverage sweep). P3 is P2 with an equal per-model nested search budget for every XGBoost (Section IV-B). In eICU all splits, outer "
         "and inner, including every hyperparameter search, are grouped by patient, so no patient contributes "
         "to both the data used to fit a model and the data used to evaluate it. In P2, every "
         "XGBoost model, whether single, Level-1 or Level-2, uses one fixed configuration (150 trees, depth 3, "
         "learning rate 0.05, row and column subsampling 0.8), taken from the standalone model of the original "
         "pipeline. Level-1 LR (C = 1), LightGBM (same values) and MLP (two hidden "
         "layers of 32 units, early stopping) are also fixed, and Level-1 out-of-fold predictions come from "
         "3-fold inner CV within each training fold. On identical Level-1 outputs we compare five Level-2 "
         "options: the default XGBoost meta-learner of the original pipeline, the fixed-configuration XGBoost, "
         "LR on logit-transformed probabilities, the unweighted mean of Level-1 outputs, and the best Level-1 "
         "model chosen inside the training fold. We call the fixed-configuration XGBoost and LR meta-learners "
         "regularized, since both constrain Level-2 capacity relative to the defaults. The reference is a "
         "single XGBoost on all features. We report the Area Under the Receiver Operating Characteristic Curve "
         "(AUROC), the Area Under the Precision-Recall Curve (AUPRC), ECE with 10 equal-width bins [17], and "
         "the calibration slope and calibration-in-the-large from logistic recalibration of the outcome on the "
         "logit of the prediction in each test fold. Paired differences are given with Nadeau–Bengio "
         "corrected 95% confidence intervals (CIs) [18], which account for overlapping training sets; a "
         "difference whose CI excludes zero is called significant. CIs are not adjusted for multiple "
         "comparisons, so we rely on patterns that repeat across tasks and settings."),
    h_sub('D. Robustness Analyses'),
    para("Equal per-model search budget: in each outer training fold the single XGBoost, the Level-1 XGBoost (tuned on "
         "the reference subset, reused for all subsets and passed to LightGBM, as in the original pipeline) and "
         "the Level-2 XGBoost each received the same nested random search of 20 configurations over discrete "
         "grids (100 to 500 trees, depth 2 to 6, learning rate 0.01 to 0.1, row and column subsampling, minimum "
         "child weight, L2 penalty; 3-fold inner CV), and LR C was grid-searched on the reference subset and on the Level-1 outputs. This equalizes the search "
         "per XGBoost model, not the total computational budget: with several tuned models and a fixed MLP, "
         "the stack received more search in total than the single model. Level-1 diversity: "
         "besides the original roster we used a diverse roster that replaces LightGBM with k-nearest neighbors "
         "(kNN), random forest (RF) and Gaussian naive Bayes (NB), and two rosters with CatBoost [19] (150 iterations, depth 4, learning rate 0.05) as a fifth "
         "learner or in place of the MLP; standalone CatBoost was also scored. The tuning, roster and CatBoost "
         "analyses cover the four main tasks; the 1-day landmark uses the original roster with fixed "
         "configurations. Cohort size: the comparison was repeated on ten stratified subsamples of 729 "
         "MIMIC-IV episodes, each with three repeats. Early deaths: the comparison was repeated after removing "
         f"episodes with in-hospital death inside the feature window ({nd['mimic24']['excluded']} MIMIC-IV "
         f"episodes and {nd['eicu72']['excluded']:,} eICU stays). Ablation: the single XGBoost was retrained "
         "with vitals only, laboratory values only, mean aggregates only and without clipping. Coverage: the "
         "threshold was swept from 50% to 90% on unclipped matrices built without the 80% filter (eICU: with a "
         "50% pre-filter). Software: Python 3.10 with scikit-learn 1.7.2 [20], XGBoost 2.1.4, LightGBM 4.7.0, "
         "CatBoost 1.2.10 and SHAP 0.49.1."),
    h_sub('E. Explainability and Feature Selection'),
    para("Two SHAP [9] analyses are kept separate. The descriptive global importance (Section IV-D) uses one "
         "fixed-configuration XGBoost fitted on each full cohort, with mean |SHAP| (log-odds) over all its rows; "
         "it is not used for feature selection. Rank agreement between cohorts was measured with Spearman "
         f"correlation over the {f['spearman'][3]} features present in both matrices. Stability and selection "
         "use P2 folds with two repeats (10 folds, eICU grouped by patient): in each training fold an XGBoost is "
         "fitted on the full training data, features are ranked by mean |SHAP| on 600 training rows drawn at "
         "random (fixed seed, both cohorts), and the Jaccard overlap of the top-20 sets is computed across folds. "
         "XGBoost is then retrained on the top 10, 20, 30 and 40 features of that fold and scored on the "
         "held-out fold, so the ranking never sees test data."))

tl = tw['los7_d2']; t14 = tw['los14_d2']
add(h_sec('IV. Results'),
    h_sub('A. Replication Under the Original Protocol'),
    para("Under the original protocol P1 (single 5-fold CV, folds not grouped by patient), the stack reached AUROC 0.813 on "
         "MIMIC-IV mortality (24 h data), against 0.85 reported in [16] on a different, 729-episode cohort (4.3% lower), and 0.828 on eICU "
         "mortality (72 h data). The original LOS results (AUROC 0.946 and 0.918) used a 7-day feature window chosen "
         "by maximum AUROC, which overlaps the outcome period; they are replaced by the landmark tasks in "
         "Section IV-C."),
    SECT_END_2COL,
    caption('TABLE I.', 'Controlled Comparison (Protocol P2): AUROC Difference From the Single XGBoost '
            'Model (Points, 95% Nadeau–Bengio CI) and ECE', 'tableI', table=True),
    TABLE2,
    footnote('P2: 15 paired scores per method (5-fold CV repeated 3 times; eICU splits grouped by patient); one '
             'fixed XGBoost configuration for all models; original Level-1 roster; eICU: 12,677 stays; LOS '
             'tasks: MIMIC-IV at the 2-day landmark. *Meta-learner of the original pipeline. A negative '
             'difference means the method is worse than the single XGBoost.'),
    SECT_END_1COL,
    h_sub('B. Stacking Versus a Single Model'),
    para(f"Table I gives the controlled comparison. With the default meta-learner the stack is {f['def_min']} "
         f"to {f['def_max']} AUROC points below the single XGBoost model ({f['def_rel_min']}% to "
         f"{f['def_rel_max']}% relative), with CIs excluding zero on {['no', 'one', 'two', 'three', 'four'][f['def_sig4']]} of four tasks, and it is "
         f"poorly calibrated: ECE {f['ece_def'][0]} to {f['ece_def'][1]} against {f['ece_single'][0]} to "
         f"{f['ece_single'][1]}, calibration slope {f['slope_def'][0]} to {f['slope_def'][1]} against "
         f"{f['slope_single'][0]} to {f['slope_single'][1]}, and calibration-in-the-large {f['citl_def'][0]} to "
         f"{f['citl_def'][1]}, so its predictions are too extreme and, on average, too low. Replacing only the "
         f"meta-learner removes most of the gap: with LR the difference is {f['lr_min']} to {f['lr_max']} "
         f"points and with the fixed-configuration XGBoost {f['fix_min']} to {f['fix_max']} points, all CIs "
         f"including zero, and ECE ({f['ece_reg'][0]} to {f['ece_reg'][1]}) is close to that of the single "
         f"model; the mean and best-model combiners do not beat it (Table I). AUPRC leads to the same conclusions (on LOS>7 the fixed-configuration XGBoost "
         f"meta-learner also loses {neg(f['auprc_los7_fix'][0])} AUPRC points, CI excluding zero)."),
    para("When every XGBoost receives the same per-model search budget (P3), tuning "
         f"changed the single model's AUROC by {f['tune_gain'][0]} to {f['tune_gain'][1]} points, significantly "
         f"only on eICU (AUROC {f['tuned_auc']['mimic24']} on MIMIC-IV and {f['tuned_auc']['eicu72']} on "
         f"eICU). The tuned stacks were not significantly different on MIMIC-IV ({p2(FD('tuned', 'mimic24', TSINGLE).loc[TFIX].delta)} and "
         f"{p2(FD('tuned', 'mimic24', TSINGLE).loc[TLRM].delta)} points) or on either LOS task, but were {neg(tw['eicu72'][TFIX][0])} (CI {pts(FD('tuned', 'eicu72', TSINGLE).loc[TFIX].CI95_low)} to {pts(FD('tuned', 'eicu72', TSINGLE).loc[TFIX].CI95_high)}) and "
         f"{neg(tw['eicu72'][TLRM][0])} points (CI {pts(FD('tuned', 'eicu72', TSINGLE).loc[TLRM].CI95_low)} to {pts(FD('tuned', 'eicu72', TSINGLE).loc[TLRM].CI95_high)}) lower on eICU, and the default meta-learner "
         f"remained {f['tuned_def_range'][0]} to {f['tuned_def_range'][1]} points lower on every task. Against "
         f"the untuned single model, the tuned LR stack was {f['tuned_vs_fixed_better'][0][2]} points better on "
         "MIMIC-IV (CI excluding zero), but it was not significantly better than the tuned single model."),
    para(f"Over all {f['n_settings']} task, roster and "
         f"tuning settings, the default meta-learner was significantly worse than the single model in "
         f"{f['def_all_sigworse']} and never better ({f['def_all_range'][0]} to {f['def_all_range'][1]} points "
         f"lower). The two regularized or tuned meta-learners, each against the single model of the same protocol, were not "
         f"significantly different in {f['reg_ns']} of "
         f"{f['n_reg']} comparisons, significantly worse in {len(worse)} and never significantly better; the "
         f"losses were eICU under P3 ({tw['eicu72'][TFIX][0]} and {tw['eicu72'][TLRM][0]} points), "
         f"LOS>7 at the 1-day landmark ({worse[0][3]} and {worse[1][3]}) and the diverse roster with an LR "
         f"meta-learner on LOS>7 ({worse[4][3]}). "
         "The diverse roster lowered the correlation between algorithms on the same subset "
         f"(MIMIC-IV {f['corr']['mimic24'][0]} to {f['corr']['mimic24'][1]}) without making any stack better, "
         "and standalone CatBoost was not "
         f"significantly different from XGBoost on any of six tasks ({f['cat_range'][0]} to "
         f"{f['cat_range'][1]} points). At n = 729 (ten draws) the regularized stacks averaged "
         f"{f['n729'][FIX][0]} and {f['n729'][LRM][0]} points (no draw excluding zero) and the default "
         f"meta-learner {f['n729'][DEF][0]} (one draw excluding zero); this size is underpowered. Removing deaths inside the feature window lowered "
         f"single-model AUROC to {nd['mimic24']['single_auc']} (MIMIC-IV) and {nd['eicu72']['single_auc']} "
         "(eICU) and also lowered prevalence, suggesting that early deaths are comparatively easy to predict, but the "
         "comparison was unchanged: the default "
         f"meta-learner remained {neg(nd['mimic24'][DEF][0])} and {neg(nd['eicu72'][DEF][0])} points lower, "
         f"and the regularized stacks were within {neg(nd['mimic24'][FIX][0])} points (only the "
         f"fixed-configuration XGBoost on eICU, {nd['eicu72'][FIX][0]}, had a CI excluding zero). Excluding the "
         f"{f['nomiss']['excluded']} eICU stays without a recorded discharge status also left the comparison "
         f"unchanged (single XGBoost {f['nomiss']['single_auc']}; default meta-learner "
         f"{f['nomiss'][DEF][0]} points; regularized {f['nomiss'][FIX][0]} and {f['nomiss'][LRM][0]}, CIs "
         "including zero)."),
    h_sub('C. Leakage-Safe Length of Stay'),
    para("At the 2-day landmark the single XGBoost reached AUROC 0.782 for LOS>7 and 0.795 for LOS>14 (AUPRC "
         "0.598 and 0.352; prevalence 31.9% and 11.6%); the 1-day landmark gave 0.778 and 0.768. In the original "
         "five-repeat runs, the same model with the 7-day window reached 0.952 and 0.938. This gap mixes removed "
         "post-landmark information with a different population (patients still in the ICU), so it is not a "
         "pure leakage estimate; on a fixed population, however, stacked AUROC for LOS>7 rose steadily from "
         "0.763 (1-day window) to 0.946 (7-day window), as expected when the window overlaps the outcome."),
    SECT_END_2COL,
    figure(FIG / 'v3_fig4_shap.png', 6.2, 'rIdV3F4'),
    caption('Fig. 2.', 'Top 8 features by mean |SHAP value| for the fixed-configuration XGBoost fitted on each '
            'full cohort (descriptive, not used for selection): MIMIC-IV mortality (24 h data, left) and eICU '
            'mortality (72 h data, right).', 'fig2'),
    SECT_END_1COL,
    h_sub('D. Explainability'),
    para("Fig. 2 shows global SHAP importance. On MIMIC-IV the last lactate value dominates (mean |SHAP| 0.34), "
         "followed by the last Glasgow Coma Scale (GCS) verbal score and peripheral oxygen saturation "
         f"(SpO₂); on eICU the last heart rate leads (0.31), followed by white-blood-cell count. Rank agreement over the {f['spearman'][3]} shared features is "
         f"moderate (Spearman ρ = {f['spearman'][0]}, 95% CI {f['spearman'][1]} to {f['spearman'][2]}), "
         "and the two cohorts use different window anchors (hospital versus ICU admission). Top-20 sets are "
         f"moderately stable across training folds on MIMIC-IV (mean Jaccard {f['jac']['mimic24'][0]}) and more "
         f"stable on eICU ({f['jac']['eicu72'][0]}). With nested selection, 30 features stay within "
         f"{neg(f['sel']['mimic24'][30][0])} points of the full set on both cohorts (CIs including zero); 20 "
         f"features lose {neg(f['sel']['mimic24'][20][0])} points on MIMIC-IV (CI including zero) and "
         f"{neg(f['sel']['eicu72'][20][0])} on eICU (CI excluding zero), and 10 features lose "
         f"{neg(f['sel']['mimic24'][10][0])} and {neg(f['sel']['eicu72'][10][0])} points."),
    figure(FIG / 'v3_fig3_coverage.png', 3.2, 'rIdV3F3'),
    caption('Fig. 3.', 'Per-feature coverage before filtering for the MIMIC-IV 24 h and eICU 72 h feature windows; dashed lines '
            'mark 70%, 80% and 90%.', 'fig3'),
    h_sub('E. Ablation, Preprocessing and Feature Coverage'),
    para(f"In the ablation (P2), keeping only vitals lowered AUROC by {neg(ab['mimic24']['Vitals only'][0])} "
         f"and {neg(ab['eicu72']['Vitals only'][0])} points on the mortality tasks and by "
         f"{neg(ab['los7_d2']['Vitals only'][0])} points on LOS>7 (CIs excluding zero), with a smaller, "
         f"non-significant loss on LOS>14 ({neg(ab['los14_d2']['Vitals only'][0])} points). Keeping only "
         f"laboratory values lowered AUROC on every task ({-max(labs):.1f} to {-min(labs):.1f} points), and "
         f"using mean aggregates alone lowered it by {-max(means):.1f} to {-min(means):.1f} points. Removing "
         f"clipping changed AUROC by at most {noclip:.1f} points (all CIs including zero)."),
    para("Fig. 3 shows feature coverage before filtering: 10 of 85 eICU features have 60% to 80% coverage and "
         f"all 55 retained ones at least 90%. Lowering the eICU threshold to 70% ({f['cov_eicu'][0.7][0]} features) or 60% "
         f"({f['cov_eicu'][0.6][0]} features) raised AUROC by {f['cov_eicu'][0.7][1]} and "
         f"{f['cov_eicu'][0.6][1]} points (CIs {f['cov_eicu'][0.7][2]} to {f['cov_eicu'][0.7][3]} and "
         f"{f['cov_eicu'][0.6][2]} to {f['cov_eicu'][0.6][3]}), while on MIMIC-IV the 70% and 90% thresholds "
         f"changed AUROC by {f['cov_mimic'][0.7][1]} and {f['cov_mimic'][0.9][1]} points with CIs including zero. "
         "The 80% default is adequate for MIMIC-IV and slightly conservative for eICU."))
assert abs(float(f['sel']['eicu72'][30][0].replace(DASH, '-'))) <= abs(float(f['sel']['mimic24'][30][0].replace(DASH, '-')))
assert float(neg(nd['eicu72'][FIX][0])) <= float(neg(nd['mimic24'][FIX][0])) + 1e-9 and \
    float(neg(nd['eicu72'][LRM][0])) <= float(neg(nd['mimic24'][FIX][0])) + 1e-9 and \
    float(neg(nd['mimic24'][LRM][0])) <= float(neg(nd['mimic24'][FIX][0])) + 1e-9

add(h_sec('V. Discussion'),
    para("The controlled comparison does not support the first analysis's claim that stacking itself limits "
         "performance. The large, consistent loss appears with the default XGBoost meta-learner (100 trees, "
         "depth 6, learning rate 0.3), whose inputs are strongly correlated; its "
         "calibration slope well below one indicates over-confident predictions, consistent with overfitting at "
         "Level-2. With LR or the fixed-configuration XGBoost the stack is not significantly different from the "
         "single model in most settings, but it was never significantly better, even with less correlated base "
         "learners, a matched per-model tuning budget or CatBoost. The original pipeline also tuned its Level-1 models, "
         "which we did not re-run with other meta-learners, so our attribution applies to the controlled "
         "re-implementation."),
    para("In these cohorts a single well-configured gradient-boosted model was an adequate, simpler choice; "
         "other cohorts, feature sets and stacking designs need external testing. A stacked alternative should "
         "report its meta-learner, regularize or tune it with the same per-model budget as the baseline, and be "
         "compared on the same folds with calibration."))

add(h_sec('VI. Limitations'),
    para("Sepsis was identified from ICD codes, as in [16], rather than by Sepsis-3 criteria [1], and the data "
         "did not pass through the original EHR-QC pipeline with the Observational Medical Outcomes Partnership "
         "Common Data Model (OMOP-CDM), which may explain the gaps in Section IV-A, consistent with the limited "
         "transportability of clinical prediction models [21], [22]. The controlled comparison used fixed Level-1 configurations except in the P3 "
         "analysis; clipping thresholds and coverage filters were computed on the full matrix (neither uses "
         "the outcome, and removing clipping did not change AUROC). The cohort-size analysis and LOS tasks used MIMIC-IV only, and all "
         "results are retrospective and internal."))

add(h_sec('VII. Conclusion'),
    para("This study replicated the EHR-ML stacked ensemble for sepsis mortality and length-of-stay prediction "
         "on MIMIC-IV and eICU and re-tested it on identical folds with corrected confidence intervals. In a "
         f"controlled re-implementation, the {f['def_min']} to {f['def_max']} point AUROC loss and the poor "
         "calibration of the stack were produced by its default XGBoost meta-learner; with a regularized or "
         "per-model-tuned meta-learner the stack was not significantly different from a comparably configured single XGBoost model in "
         "most settings and was never significantly better. Future work will validate these results through the EHR-QC/OMOP-CDM pipeline and on external, "
         "multi-institutional data."))
