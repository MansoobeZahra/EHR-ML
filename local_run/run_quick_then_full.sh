#!/bin/bash
# Smoke run (QUICK=1, 1 repeat), then the full run (3 repeats) only if no unit failed.
cd "$(dirname "$0")"
PY=../.venv/bin/python
EHRML_QUICK=1 $PY -u runner.py all --jobs 9
if grep -q 'FAIL' ../results/run.log 2>/dev/null; then echo "[chain] smoke run had failures; full run NOT started"; exit 1; fi
echo "[chain $(date +%T)] smoke run clean; starting full run"
EHRML_QUICK=0 $PY -u runner.py all --jobs 9
