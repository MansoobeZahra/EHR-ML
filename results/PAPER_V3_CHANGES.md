# Paper v3: what changed and why

Files: `EHR-ML_revision_v3_rewrite.docx` and `EHR-ML_revision_v3_rewrite_LO73.pdf` (rendered with LibreOffice 7.3). Plain text: `paper/PAPER_TEXT_v3.txt`.

Build: `local_run/build_paper_v3.py` (formatting taken from v2) with the text in `local_run/paper_v3_text.py`. Every number in the text and tables comes from `local_run/paper_facts.py`, which reads the result files; the output is in `results/PAPER_FACTS.txt`. The builder stops if any stated direction (for example "CI excludes zero") disagrees with the data.

Length: 6 pages in LibreOffice 7.3 (5,003 words including references), with a small margin on page 6. The original FINAL paper needs 7 pages in 7.3 but 6 in LibreOffice 24.2, so v3 should fit 6 pages in Word. Confirm in Word.

## Reviewer comments and where v3 answers them

| Reviewer point | v3 section | Evidence used |
|---|---|---|
| R1.1, R3: tuning parity | III-C, III-D, Table III | One fixed setup for every XGBoost; an equal nested tuning budget for single, Level-1 and meta XGBoost; the stack's larger total search is disclosed |
| R1.2, R3: meta-learner not named | III-B, III-C, Table II | Default XGBoost meta-learner named with its settings; five Level-2 options compared |
| R1.3, R3: Level-1 diversity | III-D, IV-B | Diverse roster (kNN, RF, NB) and two CatBoost rosters on four tasks; correlations reported |
| R1.4: confidence intervals; unpaired Table II | III-C, Tables II and III, Fig. 2 | Identical folds, Nadeau–Bengio 95% CIs; multiple comparisons not adjusted (stated) |
| R1.5: cohort 3.5x larger | III-A, IV-B | 10 draws of n = 729 with 3 repeats; stated as underpowered |
| R1.6: eICU coverage claim not shown | IV-D, Fig. 3 | Pre-filter coverage histograms and a 50–90% sweep with CIs |
| R1.7: SHAP stability | III-E, IV-E, Fig. 4 | Fold-wise Jaccard, nested top-K selection, cross-cohort Spearman |
| R2: windows and boundaries | III-A, IV-B, IV-C | Exact windows and anchors; 1- and 2-day LOS landmarks; early-death sensitivity |
| R2: "stacking ceiling" claim | Abstract, I, V, VII | Removed. New headline: the default meta-learner produces the penalty; regularized or tuned stacks are not significantly different in 38 of 44 comparisons and never significantly better |
| R2: statistics | III-C | Wilcoxon on overlapping folds replaced; eICU folds grouped by patient |

## Changes made after the independent audit (26 Oct)

- **eICU patient grouping.** 12,677 stays come from 10,835 patients. All eICU analyses were re-run with patient-grouped outer folds; the conclusions did not change.
- **Early-death sensitivity added.** The default-meta penalty persists (−3.7, −4.1).
- **Transfer result (0.756) removed.** It came from a single XGBoost, not the stack, and used 50 features because of the platelet bug.
- **LOS comparison reworded.** The 0.12–0.16 gap mixes leakage with a population change; this is now said explicitly.
- **Wording softened.**
  - "Matches" became "not significantly different".
  - "Never better" became "never significantly better".
  - The attribution is limited to the controlled re-implementation.
  - "Unregularised" was replaced by the actual default settings.
- **Disclosed:** the tuned LR stack beat the untuned single model on MIMIC-IV (+0.8), but not the equally tuned single model.
- **Feature selection updated.** The claim moved from 20 to 30 features, because on eICU 20 features lose 0.3 points (significant).
- **Numbers corrected:**
  - Relative loss is 2.6–4.9%.
  - The original-paper gap is 4.3%.
  - The 1-day landmark matrix has 120 features.
  - LOS AUROC is 0.77–0.80.
  - Step 25 uses 10 folds.
- **Tables:**
  - Differences are now shown in AUROC points with two decimals.
  - Table II gained a best Level-1 row.
  - Table III gained a default-meta column.
  - Table I gained a footnote on the F1 threshold and folds.
- **Figures:**
  - Fig. 1 now shows the tuning, CatBoost, landmark and early-death analyses; its unexplained arrow is gone and abbreviations are defined in the caption.
  - Fig. 2 is now a paired-difference plot with CIs.
  - Fig. 3 has larger type.
  - Fig. 4's axis label says log-odds.
- **Methods details added:**
  - The search grids.
  - The scope of each analysis.
  - The LOS tasks are MIMIC-IV only.
  - The landmark coverage filter was computed before the landmark restriction.
  - The coverage sweep used unclipped matrices.
  - How calibration slope, calibration-in-the-large and ECE are computed.
  - Where the fixed configuration came from.
  - The data-access statement.
  - Software versions.
- **Limitations added:**
  - ICD-coded sepsis rather than Sepsis-3.
  - Inner CV not grouped by patient.
  - No multiplicity adjustment.
  - Differing window anchors between the cohorts.
- **Style:**
  - US spelling.
  - Acronyms defined in the body.
  - "SAPS II" used consistently.
  - En dash in Nadeau–Bengio.

## Second audit round (26 Oct, after re-audit; all numbers re-derived independently and confirmed)

- **SAPS II estimate types corrected.** The proxy (0.794) is a pooled out-of-fold AUROC; the SAPS II-variable ensemble (0.728) is a mean over folds.
- **LOS comparison is now like-for-like.** It compares against the same single model on the 7-day window (0.952 and 0.938; 0.14 to 0.17 higher). The abstract notes that this window used all patients.
- **Early-death sensitivity:**
  - Now reports the share of deaths excluded (MIMIC-IV 19.9%, eICU 43.4%).
  - Now uses "suggesting", since prevalence also changes.
  - The abstract says this analysis gives the same pattern, separate from the 38/44 count.
- **Wording:**
  - ECE "is close to" the single model's, not "returns to the level of" it.
  - "Not tuned on these data" was dropped from the fixed-configuration description (it can't be verified from the repository).
  - kNN, RF and NB are spelled out in the text.
  - "Electronic Health Record (EHR) data" now in both places.
- **Feature selection:** stated for both 30 features (within 0.2 points) and 20 features (within 0.4), so the choice of 30 does not look post hoc.
- **Figures:**
  - The Fig. 1 first box mentions patient grouping.
  - Fig. 3 tick labels are thinned for legibility.

## References (22, numbered by first citation, all hyperlinked)

Checked against Crossref, arXiv, PMLR or NeurIPS:
- [3], [4], [5], [6], [7], [8], [10], [12], [13], [15], [16], [17], [18], [19], [21], [22] match.
- [14] author order corrected to Rawat, Shanmugam, Airen.
- Crossref holds only partial records for [1] and [2]. The printed details are the standard ones, and four verified papers cite [1] with the same details.
- [9] is not in Crossref (NeurIPS).
- [11] LightGBM and [20] scikit-learn were added, because both packages are used.

Claims checked against abstracts:
- [6] is cited only for XGBoost with SHAP, with lactate and respiratory rate as top predictors.
- [10] is cited for "XGBoost beat conventional scores".
- [8] reports stacking as best of ten models.
- [13] reports similar internal discrimination for stacked and single models.
- The calibration claim on [14] and [15] was removed.

## Still open before submission
1. Riphah template, page limit, blind-review and reference-style rules are not yet known.
2. Confirm 6 pages in Word.
3. Title changed (user choice, 26 Oct) to "Stacked Ensembles Versus a Single Gradient-Boosted Model for Sepsis Outcome Prediction: A Controlled and Explainable Replication on MIMIC-IV and eICU" (docx and document metadata). Use the same title in the submission system.
4. No code-availability statement.
5. All validation is internal and retrospective; there is no external validation under the new protocol.

## Final check (26 Oct): references, links, length

- **References:** all 22 checked against Crossref, arXiv, NeurIPS, PMLR or JMLR; title, authors, venue and year match. Page ranges for the NeurIPS papers [9], [11] and [19] are not printed on the proceedings pages; the standard values are kept.
- **Links in the docx:**
  - 35 in-text citations and 8 Fig./Table mentions are internal links, all resolving.
  - All 22 reference entries link to their DOI or official proceedings page.
- **Links in the PDF:** LibreOffice 7.3 drops every link on PDF export (the original 24.2 export kept them). `local_run/add_pdf_links.py` re-adds them: 43 internal links, each verified to land on the right target, and 72 external line-links covering the 22 URLs. A PDF saved from Word ("Save as PDF") keeps the docx links natively.
- **Abstract:** shortened from 307 to 249 words to fit the usual IEEE 250-word limit; all findings kept.
- **One-step rebuild:** `local_run/make_final.sh <unzipped v2 dir>` regenerates the figures, docx and linked PDF.

## External review comments (26 Oct) and what was done

1. **Window versus outcome.**
   - Tasks are renamed throughout the text, tables and figures as "MIMIC-IV mortality (24 h data)" and "eICU mortality (72 h data)".
   - Methods state that the outcome is in-hospital death at any time, and that 24 h and 72 h are the feature window, not a horizon.
   - Early deaths inside the window are covered by the existing sensitivity analysis.
2. **eICU inner CV not grouped.** Fixed in code. Inner splits and all tuning searches are now grouped by patient, and the eICU stacking analyses were re-run (results/ext3). The headline is unchanged; the eICU default-meta loss is larger (−2.7), and the regularized stacks are worse in 5 of 44 comparisons instead of 6. The Limitations sentence about ungrouped inner splits was removed.
3. **Conclusion too strong.**
   - The abstract now says "in the tested implementation".
   - The Discussion says "in these cohorts" and that external testing is needed.
4. **Protocols.**
   - Methods define P1 (original), P2 (controlled) and P3 (equal tuning).
   - Table I, Table II, Table III and Fig. 2 are labelled with their protocol.
5. **SHAP.**
   - Methods separate the descriptive global SHAP (Fig. 4: full-cohort model, all rows, not used for selection) from the fold-wise stability and selection analysis.
   - The selection analysis uses 600 random training rows per fold with a fixed seed, in both cohorts.
6. **LOS claim.** Already precise (IV-C states the gap is not a pure leakage estimate). The original 0.952/0.938 values are now labelled as the original repeated-CV runs.

To stay at 6 pages, these were removed: the Related Work roadmap, the section-roadmap sentence and the SAPS II side comparison. The weaknesses paragraph, contributions list, tuning grid, robustness paragraph and Discussion were shortened. No result or caveat was removed.

Final state: 6 pages in LibreOffice 7.3 (4,908 words); abstract 250 words.
- Docx: all 46 internal links resolve, and all 22 reference entries link to their DOI or proceedings page.
- PDF: 46 internal links, each verified to land on its target, plus external reference links.

## Verification round (26 Oct, second external review)

Independent audit (`local_run/audit_paper.py` → `results/AUDIT_REPORT.md`, `results/COMPARISON_INVENTORY.csv`):
- **Stacking:** the meta-learner is fitted only on inner out-of-fold Level-1 predictions; every training row has exactly one.
- **eICU patient separation:** no patient appears on both sides of any outer or inner split (0 overlaps); all tuning searches are grouped too. MIMIC-IV and LOS have one episode per patient.
- **Fitted on the full matrix:** clipping thresholds and coverage filters only. Neither uses the outcome; disclosed in Limitations.
- **Fitted inside folds:** imputation, models, searches and SHAP-based selection. The 30-feature result uses fold-specific rankings scored on held-out folds.
- **Outcomes:**
  - MIMIC-IV label = hospital_expire_flag, and eICU label = discharge status "Expired"; both verified against the raw tables.
  - 104 eICU stays have no recorded status and are treated as survivors. Disclosed, and a check without them changes nothing (single 0.853; default −2.8; regularized −0.3, −0.2).
  - Deaths inside the window: 140 of 704 (MIMIC-IV) and 1,230 of 2,835 (eICU). The text now says the tasks measure discrimination from early data, not early warning.
- **Statistics:** all 66 comparisons were recomputed from per-fold scores with identical results. 17/22 (default worse) and 39/44 not significant plus 5 worse (regularized) match the abstract and text. All Table I cells reproduce.

Manuscript changes in this round:
- "Equal tuning budget" became "equal per-model search budget", stating that the stack received more total search.
- The outcome-versus-window definition and the missing-status handling are stated.
- Non-breaking hyphens in compound terms fix text extraction ("metalearner", "length-ofstay" no longer occur).
- To keep 6 pages:
  - The ablation figure was removed (its numbers and CIs are in the text).
  - Table I (original-protocol baseline) was folded into the text, as was the P3 table, with eICU CIs added.
  - Results subsections D/E were swapped so figures appear in order (Fig. 2 SHAP, Fig. 3 coverage).
  - Some restating sentences were cut.

Final state:
- 6 pages in LibreOffice 7.3, which renders longer than Word; abstract 250 words.
- Docx: 39 internal links resolve, and all 22 references link out.
- PDF: 39 internal links verified, plus external reference links.

## Round 5 (external review, 2026-10-27): wording only, no new experiments
- Abstract, Results IV-B, Conclusion: "39 of 44" now says what a comparison is (two regularized/tuned meta-learners x 22 settings, each vs the single model of the same protocol: fixed vs fixed under P2, tuned vs tuned under P3; confirmed in COMPARISON_INVENTORY.csv: 36 + 8). "Never significantly better" now refers to the comparably configured single model (the tuned LR stack's +0.8 vs the untuned single model is already reported in IV-B).
- Abstract opening shortened ("Sepsis risk models must be...") and "regularized or equally tuned meta-learner" to keep the abstract under 250 words (now 246).
- IV-A: 0.85 of [16] is now marked as coming from a different, 729-episode cohort. 4.3% checked: (0.85 - 0.8132) / 0.85.
- Checked, no change: relative losses 3.2% to 4.9% use the single-model AUROC as denominator (paper_facts.py line 69); figure and table numbering consistent; early-warning disclaimer already present.
- Backup of the previous deliverables: results/round4_backup/. Still 6 pages in LibreOffice 7.3, audit counts match.
