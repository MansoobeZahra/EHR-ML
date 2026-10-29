"""Figures for the revised paper (v3). Output: Res_02/figures/v3_*.png

Fig. 1  updated architecture diagram (replicated pipeline + this study's analyses)
Fig. 2  ablation under the controlled protocol (out_ext/abl_*.csv)
Fig. 4  global SHAP of the fixed-config XGBoost on the full cohorts (out_ext/shap_global_*.csv)
Fig. 3  per-feature coverage before filtering (from local_run/cache, Step 24 inputs)
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 400})

INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']  # reference categorical order


# ----------------------------------------------------------------- Fig. 2 ----
# Ablation under the controlled protocol: paired AUROC difference vs all features with
# Nadeau-Bengio 95% CIs (fixed-config XGBoost, identical folds; eICU folds grouped by patient).
import pandas as pd
import os
EXT = HERE / 'out_ext'
EXT2 = HERE / 'out_ext2'
tasks = ['MIMIC-IV\nmortality', 'eICU\nmortality', 'LOS>7\n(day 2)', 'LOS>14\n(day 2)']
keys = ['mimic24', 'eicu72', 'los7_d2', 'los14_d2']
conds = ['Vitals only', 'Labs only', 'Mean aggregate only', 'No clipping']
marks = ['o', 's', 'D', '^']


def abl(k):
    f = EXT2 / f'abl_{k}.csv' if (EXT2 / f'abl_{k}.csv').exists() else EXT / f'abl_{k}.csv'
    return pd.read_csv(f).set_index('condition')


fig, ax = plt.subplots(figsize=(3.45, 2.25))
w = 0.19
for i, k in enumerate(keys):
    t = abl(k)
    for j, c in enumerate(conds):
        d, lo, hi = 100 * t.loc[c, ['delta_vs_all', 'CI95_low', 'CI95_high']].values
        xx = i + (j - 1.5) * w
        ax.errorbar(xx, d, yerr=[[d - lo], [hi - d]], fmt=marks[j], ms=4.2, color=S[j],
                    ecolor=S[j], elinewidth=1.1, capsize=2, zorder=3,
                    label=c if i == 0 else None)
ax.axhline(0, color=INK2, lw=0.8, zorder=2)
ax.set_xticks(range(len(tasks)), tasks, fontsize=7.4, color=INK)
ax.tick_params(axis='y', labelsize=7.4, colors=INK2)
ax.tick_params(axis='x', length=0)
ax.set_ylabel('\u0394AUROC vs all features (points)', fontsize=7.6, color=INK)
ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
ax.legend(fontsize=6.9, ncol=2, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.25), handlelength=1.2, columnspacing=1.0)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / 'v3_fig2_ablation.png')
plt.close(fig)

# ----------------------------------------------------------------- Fig. 4 ----
# Global SHAP (mean |SHAP|), fixed-config XGBoost fitted on each full cohort (Step 27).
NAMES = {'lactate': 'Lactate', 'gcs_verbal': 'GCS verbal', 'gcs_eye': 'GCS eye',
         'gcs_motor': 'GCS motor', 'spo2': 'SpO$_2$', 'temperature_f': 'Temperature',
         'heart_rate': 'Heart rate', 'resp_rate': 'Respiratory rate', 'bun': 'BUN',
         'wbc': 'WBC', 'potassium': 'Potassium', 'platelets': 'Platelets',
         'plateletss': 'Platelets', 'creatinine': 'Creatinine', 'hemoglobin': 'Hemoglobin',
         'glucose': 'Glucose', 'sodium': 'Sodium', 'anion_gap': 'Anion gap',
         'bicarbonate': 'Bicarbonate', 'calcium': 'Calcium', 'hematocrit': 'Hematocrit',
         'inr': 'INR', 'magnesium': 'Magnesium', 'phosphate': 'Phosphate', 'ptt': 'PTT'}
AGG = {'avg': 'mean', 'min': 'min', 'max': 'max', 'first': 'first', 'last': 'last'}


def nice(std):
    body = std.split('_', 1)[1]
    concept, agg = body.rsplit('_', 1)
    return f'{NAMES.get(concept, concept)} ({AGG[agg]})'


fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.65))
for ax, key, title in ((axes[0], 'mimic24', 'MIMIC-IV mortality (24 h data)'),
                       (axes[1], 'eicu72', 'eICU mortality (72 h data)')):
    g = pd.read_csv(EXT / f'shap_global_{key}.csv').head(8).iloc[::-1]
    ax.barh([nice(s) for s in g.feature_std], g.mean_abs_shap, color=S[0], height=0.7, zorder=3)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.tick_params(axis='y', labelsize=7.6, colors=INK, length=0)
    ax.tick_params(axis='x', labelsize=7.2, colors=INK2)
    ax.set_xlabel('Mean |SHAP value| (log-odds)', fontsize=7.6, color=INK)
    ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
fig.tight_layout(pad=0.3, w_pad=2.0)
fig.savefig(FIG / 'v3_fig4_shap.png')
plt.close(fig)

# ----------------------------------------------------------------- Fig. 3 ----
data = pickle.load(open(HERE / 'cache' / 'data_cache.pkl', 'rb'))
META = ['subject_id', 'hadm_id', 'stay_id', 'anchor_date', 'mortality', 'los_gt7', 'los_gt14',
        'los_days', 'los', 'cohort', 'patientunitstayid', 'uniquepid',
        'hospitaldischargestatus', 'unitdischargeoffset']
dm = data['dm_mimic24_full']
cov_m = dm[[c for c in dm.columns if c not in META]].notna().mean().values
cov_e = data['COVERAGE_PREFILTER'][72].values
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6), sharey=False)
for ax, cov, name in ((axes[0], cov_m, f'MIMIC-IV 24 h ({len(cov_m)} features)'),
                      (axes[1], cov_e, f'eICU 72 h ({len(cov_e)} features)')):
    ax.hist(cov, bins=np.linspace(0, 1, 21), color=S[0], edgecolor='white', lw=0.6, zorder=3)
    for t in (0.7, 0.8, 0.9):
        ax.axvline(t, color=INK2, ls='--', lw=0.7, zorder=4)
    ax.set_title(name, fontsize=8.0, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 0.8, 1.0], ['0', '0.5', '0.8', '1'])
    ax.tick_params(labelsize=7.6, colors=INK2)
    ax.grid(axis='y', color=GRID, lw=0.5, zorder=0)
axes[0].set_ylabel('Features', fontsize=8.0, color=INK)
fig.supxlabel('Share of episodes with a recorded value', fontsize=8.0, color=INK, y=0.02)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(FIG / 'v3_fig3_coverage.png')
plt.close(fig)

# ----------------------------------------------------------------- Fig. 1 ----
LEFT = ['#dbe7f7', '#2a78d6']
RIGHT = ['#fde3d7', '#eb6834']


def box(ax, x, y, w, h, title, sub, col, fs=9.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.012',
                                fc=col[0], ec=col[1], lw=1.0))
    if sub:
        ax.text(x + w / 2, y + h * 0.68, title, ha='center', va='center', fontsize=fs,
                color=INK, weight='bold')
        ax.text(x + w / 2, y + h * 0.29, sub, ha='center', va='center', fontsize=fs - 1.2,
                color=INK2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs,
                color=INK, weight='bold')


def arrow(ax, x, y0, y1):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, mutation_scale=8))


fig = plt.figure(figsize=(7.0, 3.05))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.245, 0.955, 'Replicated EHR-ML pipeline', ha='center', fontsize=10.5,
        weight='bold', color=LEFT[1])
ax.text(0.745, 0.955, 'Analyses added in this study', ha='center', fontsize=10.5,
        weight='bold', color=RIGHT[1])

L = [('MIMIC-IV and eICU sepsis cohorts', 'ICD codes, adults, in-hospital death, ICU LOS'),
     ('Feature extraction', 'Vitals + labs, 5 aggregates, coverage \u2265 80%'),
     ('Level-1: 40 models', 'XGBoost, LR, LightGBM, MLP \u00d7 10 feature subsets'),
     ('Level-2 meta-learner', 'XGBoost, library defaults (original pipeline)'),
     ('Original protocol', 'Single 5-fold CV; mortality and LOS tasks')]
R = [('Controlled comparison', 'Same folds; one XGBoost setup; eICU by patient'),
     ('Five Level-2 options', 'XGBoost default, XGBoost fixed, LR, mean, best'),
     ('Equal per-model search budget', 'Same nested search for each XGBoost model'),
     ('Level-1 diversity and cohort size', 'kNN, RF, NB and CatBoost rosters; n = 729'),
     ('Calibration and corrected CIs', 'ECE, calibration slope; Nadeau\u2013Bengio CIs'),
     ('Leakage-safe LOS, early deaths', '1- and 2-day landmarks; early-death check'),
     ('Coverage, ablation, preprocessing', 'Pre-filter coverage and threshold sweep'),
     ('SHAP explainability', 'Fold stability, nested top-K selection')]
bw, bx_l, bx_r = 0.43, 0.03, 0.53
h_l, g_l = 0.148, 0.0325
y = 0.905
for i, (t, s) in enumerate(L):
    box(ax, bx_l, y - h_l, bw, h_l, t, s, LEFT, fs=9.6)
    if i < len(L) - 1:
        arrow(ax, bx_l + bw / 2, y - h_l, y - h_l - g_l)
    y -= h_l + g_l
h_r, g_r = 0.0975, 0.0137
y = 0.905
for i, (t, s) in enumerate(R):
    box(ax, bx_r, y - h_r, bw, h_r, t, s, RIGHT, fs=9.2)
    y -= h_r + g_r
fig.savefig(FIG / 'v3_fig1_architecture.png')
plt.close(fig)
print('figures written to', FIG)
