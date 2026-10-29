"""Make local_run/research02_local.ipynb from research02_revision.ipynb.

Only paths, Kaggle-specific lines and chunk sizes change. Every edit is an
exact string replacement that must match once, so nothing else can drift.
"""
import copy
import json
from pathlib import Path

# Paths follow this file, so the project folder can be moved: re-run this script.
ROOT = Path(__file__).resolve().parent.parent          # .../Colacare/Res_02
SRC = ROOT / 'research02_revision.ipynb'
DST = ROOT / 'local_run' / 'research02_local.ipynb'

MIMIC = str(ROOT.parent / 'mimic-iv-3.1')
EICU = str(ROOT / 'eicu-collaborative-research-database-2.0')
OUTD = str(ROOT / 'local_run' / 'out')

# Local find_real_file: official PhysioNet downloads are flat .csv.gz files.
FIND_OLD = '''def find_real_file(root, name):
    hits = [p for p in Path(root).rglob(name) if p.is_file()]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]'''
FIND_NEW = '''def find_real_file(root, name):
    # LOCAL: also accept the official .csv.gz download layout.
    hits = [p for p in Path(root).rglob(name) if p.is_file()]
    if not hits:
        hits = [p for p in Path(root).rglob(name + ".gz") if p.is_file()]
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]'''

EDITS = {
    3: [
        ('DATA_ROOT = Path("/kaggle/input/datasets/mansoobezehra")\n', ''),
        ('MIMIC_DIR = DATA_ROOT / "mimic-iv-31/mimic-iv-3.1"',
         f'MIMIC_DIR = Path("{MIMIC}")  # LOCAL'),
        ('EICU_DIR = DATA_ROOT / "eicu-dir" / "eicu-collaborative-research-database-2.0"',
         f'EICU_DIR = Path("{EICU}")  # LOCAL'),
        ("""# Auto-fallback: scan all available input datasets
if not MIMIC_DIR.exists():
    print('WARNING: Specified MIMIC_DIR not found. Scanning /kaggle/input/ ...')
    for d in sorted(Path('/kaggle/input').iterdir()):
        if any(x in d.name.lower() for x in ['mimic', 'physio']):
            MIMIC_DIR = d
            print(f'  -> Using MIMIC_DIR: {MIMIC_DIR}')
            break

if not EICU_DIR.exists():
    print('WARNING: Specified EICU_DIR not found. Scanning /kaggle/input/ ...')
    for d in sorted(Path('/kaggle/input').iterdir()):
        if 'eicu' in d.name.lower():
            EICU_DIR = d
            print(f'  -> Using EICU_DIR: {EICU_DIR}')
            break
""", "assert MIMIC_DIR.exists() and EICU_DIR.exists()  # LOCAL: no Kaggle fallback scan\n"),
        ("OUT_DIR  = Path('/kaggle/working/ehr_ml_output')",
         f"import os\nOUT_DIR  = Path(os.environ.get('EHRML_OUT_DIR', '{OUTD}'))  # LOCAL"),
    ],
    # 62 GB RAM: larger chunks, same filtering logic.
    4: [("chunksize=1500000):", "chunksize=5_000_000):  # LOCAL: was 1,500,000")],
    8: [(FIND_OLD, FIND_NEW)],
    12: [
        ('EICU_DIR = Path("/kaggle/input/datasets/mansoobezehra/eicu-dir/eicu-collaborative-research-database-2.0")',
         f'EICU_DIR = Path("{EICU}")  # LOCAL'),
        (FIND_OLD, FIND_NEW),
        ("chunk_size = 250_000", "chunk_size = 2_000_000  # LOCAL: was 250,000"),
        ("pd.read_csv(lab_path, usecols=lab_usecols, chunksize=250_000)",
         "pd.read_csv(lab_path, usecols=lab_usecols, chunksize=2_000_000)"),
        ("pd.read_csv(vital_path, usecols=vital_usecols, chunksize=250_000)",
         "pd.read_csv(vital_path, usecols=vital_usecols, chunksize=2_000_000)"),
        ("pd.read_csv(lab_path, usecols=lab_usecols_local, chunksize=250_000)",
         "pd.read_csv(lab_path, usecols=lab_usecols_local, chunksize=2_000_000)"),
    ],
    # Driver: QUICK and EICU_N come from the environment so the same file runs
    # the smoke test and the full run. Defaults are the notebook's values.
    18: [
        ("QUICK            = False", "import os\nQUICK            = os.environ.get('EHRML_QUICK', '0') == '1'  # LOCAL"),
        ("EICU_N           = 5000", "EICU_N           = int(os.environ.get('EHRML_EICU_N', '5000'))  # LOCAL"),
    ],
}

# Cells 1 (pip install) and 2 (git clone EHR-ML, never imported later) are dropped.
DROP = {1, 2}

nb = json.load(open(SRC))
out = copy.deepcopy(nb)
cells = []
for i, c in enumerate(nb['cells']):
    if i in DROP:
        assert c['cell_type'] == 'code'
        continue
    c = copy.deepcopy(c)
    if i in EDITS:
        s = ''.join(c['source'])
        for old, new in EDITS[i]:
            n = s.count(old)
            assert n >= 1, (i, old[:60])
            if old.startswith('pd.read_csv') or old == FIND_OLD:
                assert n == 1, (i, old[:60], n)
            s = s.replace(old, new)
        c['source'] = s.splitlines(keepends=True)
    cells.append(c)
out['cells'] = cells
json.dump(out, open(DST, 'w'), indent=1)
print('wrote', DST, len(cells), 'cells')
