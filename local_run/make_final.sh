#!/bin/bash
# Final paper build: docx (from v2 templates) -> LibreOffice PDF -> PDF links added.
set -e
cd "$(dirname "$0")"
R="$(cd .. && pwd)"
TPL="${1:?usage: make_final.sh <unzipped_v2_dir>}"
W="$(mktemp -d)"
../.venv/bin/python make_figures.py
../.venv/bin/python build_paper_v3.py "$TPL" "$R/EHR-ML_revision_v3_rewrite.docx"
cp "$R/EHR-ML_revision_v3_rewrite.docx" "$W/v3.docx"
timeout 300 soffice --headless --convert-to pdf --outdir "$W" "$W/v3.docx" >/dev/null 2>&1
../.venv/bin/python add_pdf_links.py "$W/v3.pdf" "$R/EHR-ML_revision_v3_rewrite_LO73.pdf"
pdfinfo "$R/EHR-ML_revision_v3_rewrite_LO73.pdf" | grep Pages
