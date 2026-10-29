# EHR-ML sepsis replication: stacked ensemble vs single XGBoost

Replication and controlled re-test of the EHR-ML stacked ensemble on MIMIC-IV 3.1 and eICU 2.0 sepsis cohorts.

- `main_complete`: original Kaggle notebook.
- `EHR-ML_revision_v3_rewrite.docx` / `_LO73.pdf`: current revised paper (earlier versions alongside).
- `local_run/`: local experiment code (`step27_ext.py` evaluation engine, `runner2.py` units, `paper_facts.py`, `audit_paper.py`, paper build scripts).
- `results/`: per-fold scores, paired deltas, summaries, `AUDIT_REPORT.md`, `COMPARISON_INVENTORY.csv`, change log.
- `code/`, `notebooks/`: revision experiment code and the executed original notebook.

Data: MIMIC-IV and eICU are credentialed PhysioNet datasets and are not included. No patient-level data or derived feature matrices are stored in this repository.
