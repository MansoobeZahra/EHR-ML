# Stacked Ensembles Versus a Single Gradient-Boosted Model for Sepsis Outcome Prediction

Code and results for the paper:

> Mansoob-e-Zahra and Ijaz Hussain, "Stacked Ensembles Versus a Single Gradient-Boosted Model for Sepsis Outcome
> Prediction: A Controlled and Explainable Replication on MIMIC-IV and eICU," accepted at the *2026 International
> Conference on Computing, Intelligence and Innovation (ICCII 2026)*, Riphah International University and IEEE
> Islamabad Section, Islamabad, Pakistan, 23 to 24 December 2026. IEEE Xplore link: to be added after publication.

Department of Computer and Information Sciences, Pakistan Institute of Engineering and Applied Sciences (PIEAS),
Nilore, Islamabad, Pakistan.

## What the study does

The study replicates the EHR-ML stacked ensemble for sepsis outcome prediction
([Ramakrishnaiah et al., 2025](https://doi.org/10.1016/j.ijmedinf.2025.105816);
[original code](https://github.com/ryashpal/EHR-ML)) on MIMIC-IV v3.1 (2,545 sepsis episodes) and eICU v2.0
(12,688 ICU stays). It then re-tests the pipeline's central design choice on identical, patient-grouped
cross-validation folds, reporting paired differences with Nadeau-Bengio corrected 95% confidence intervals.

Main results:

- **Default meta-learner.** With the original Level-2 meta-learner (XGBoost at library defaults), the stack scores
  2.7 to 3.9 AUROC points below a single XGBoost model and is poorly calibrated (calibration slope 0.30 to 0.55).
- **Regularized meta-learner.** With a regularized or equally tuned meta-learner, the stack is not significantly
  different from a comparably configured single model in 39 of 44 comparisons and never significantly better. Its
  5 losses do not survive Holm adjustment.
- **Length of stay.** Leakage-safe landmark length-of-stay tasks reach AUROC 0.77 to 0.80. With the original 7-day
  feature window, which overlaps the outcome, the same model reaches 0.95 and 0.94.
- **Feature selection.** Nested, leak-free SHAP feature selection keeps AUROC within 0.2 points with 30 features.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/original_pipeline_kaggle.ipynb` | The replicated EHR-ML pipeline as first run on Kaggle (original protocol P1), with outputs. |
| `experiments/research02_local.ipynb` | Cohort extraction, feature construction and the Step 22 to 26 analyses (controlled comparison, cohort size, coverage, SHAP stability and nested selection, landmark LOS tasks). |
| `experiments/runner.py` | Runs the notebook: data build, then one process per experiment unit (Steps 22 to 26). |
| `experiments/step27_ext.py` | The evaluation engine for all controlled comparisons: identical folds, five Level-2 options, equal tuning budget, extra rosters, calibration metrics. |
| `experiments/runner2.py` | Runs the Step 27 units: tuning, rosters, CatBoost, landmarks, early deaths, ablation, coverage and SHAP. |
| `experiments/holm_adjust.py` | Holm adjustment of the 66 stacking comparisons; writes `results/COMPARISON_INVENTORY.csv`. |
| `experiments/paper_facts.py` | Recomputes every number quoted in the paper from the result files (`results/PAPER_FACTS.txt`). |
| `experiments/make_figures.py` | Paper figures (`figures/`). |
| `experiments/out_ext*/` | Per-fold scores (`*_folds.csv`), summaries (`*_summary.csv`), paired differences with CIs (`*_deltas.csv`) and Level-1 correlations (`*_diversity.csv`). |
| `results/final/` | Step 22 to 26 outputs used in the paper. |
| `results/COMPARISON_INVENTORY.csv` | All 66 stacking comparisons: setting, meta-learner, reference, difference, 95% CI, p-value and Holm-adjusted p-value. |

Result folders:

| Folder | Contents |
|---|---|
| `out_ext` | MIMIC-IV and LOS results. |
| `out_ext2` | eICU results with patient-grouped outer folds, the early-death and coverage analyses, and eICU SHAP stability. |
| `out_ext3` | eICU stacking results with patient-grouped inner splits as well. These are the eICU numbers used in the paper. |
| `out_ext4` | eICU sensitivity analysis without stays that lack a discharge status. |

File names follow `<setting>_<task>_<kind>.csv`.
- Settings: `main` (P2, original roster), `tuned` (P3), `div` (diverse roster), `cat5` (CatBoost added), `catmlp` (CatBoost replaces MLP), `nodeath`, `abl`.
- Tasks: `mimic24`, `eicu72`, `los7_d2`, `los14_d2`, `los7_d1`, `los14_d1`.

## Data

MIMIC-IV v3.1 and eICU v2.0 are credentialed datasets from [PhysioNet](https://physionet.org/). They are **not**
included, and no patient-level data or derived feature matrices are stored in this repository. To reproduce the
results you need your own PhysioNet credentialed access. The extraction reads the raw `hosp`/`icu` tables of
MIMIC-IV and the `patient`, `diagnosis`, `vitalPeriodic` and `lab` tables of eICU.

## Reproducing the results

```bash
pip install -r requirements.txt            # Python 3.10
export MIMIC_DIR=/path/to/mimic-iv-3.1
export EICU_DIR=/path/to/eicu-collaborative-research-database-2.0
cd experiments

python runner.py build                     # cohorts and feature matrices (cached locally in experiments/cache/)
python runner.py all --jobs 4              # Steps 22 to 26 -> out_full/
python runner2.py build2                   # landmark and unclipped matrices
python runner2.py check                    # the Step 27 engine must reproduce Step 22 exactly
python runner2.py all --jobs 4             # Step 27 units -> out_ext/
EHRML_UNITS2=1 EHRML_OUT_DIR=out_ext2 python runner2.py all --jobs 4   # eICU, patient-grouped
EHRML_UNITS3=1 EHRML_OUT_DIR=out_ext3 python runner2.py all --jobs 4   # eICU, grouped inner splits
EHRML_OUT_DIR=out_ext4 python runner2.py unit nomissing:eicu72

python holm_adjust.py                      # multiple-comparison adjustment
python paper_facts.py                      # all numbers quoted in the paper
```

All folds, subsamples and models use fixed random seeds. The paper's runs used a CPU-only workstation (Ubuntu
22.04, 24 cores, 62 GB RAM); the full set of experiments takes a few hours. Because XGBoost 3.x is incompatible with
SHAP 0.49.1, keep the versions in `requirements.txt`.

## Citation

```bibtex
@inproceedings{zahra2026stacked,
  author    = {Mansoob-e-Zahra and Ijaz Hussain},
  title     = {Stacked Ensembles Versus a Single Gradient-Boosted Model for Sepsis Outcome Prediction:
               A Controlled and Explainable Replication on {MIMIC-IV} and {eICU}},
  booktitle = {2026 International Conference on Computing, Intelligence and Innovation (ICCII)},
  address   = {Islamabad, Pakistan},
  year      = {2026},
  note      = {Accepted; to appear in IEEE Xplore}
}
```

## Contact

Ijaz Hussain, ijazhussain@pieas.edu.pk
Mansoob-e-Zahra, mansoobezehra@gmail.com
