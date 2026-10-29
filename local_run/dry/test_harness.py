import numpy as np, pandas as pd
from sklearn.datasets import make_classification
RANDOM_STATE = 42
META_COLS = ['subject_id','hadm_id','stay_id','anchor_date','mortality','los_gt7','los_gt14','los_days','los','cohort','patientunitstayid','uniquepid','hospitaldischargestatus','unitdischargeoffset']
from pathlib import Path
OUT_DIR = Path(__import__('os').path.abspath('dry_out')); OUT_DIR.mkdir(exist_ok=True)  # LOCAL

def get_subsets(dm):
    subsets = {}
    for prefix in ['vitals','labs']:
        for agg in ['avg','min','max','first','last']:
            cols = [c for c in dm.columns if c.startswith(f'{prefix}_') and c.endswith(f'_{agg}')]
            if cols:
                subsets[f'{prefix}_{agg}'] = dm[cols].copy()
    return subsets

def make_dm(n=500, seed=0):
    X, y = make_classification(n_samples=n, n_features=40, n_informative=10,
                               n_redundant=10, weights=[0.75, 0.25], random_state=seed)
    names = []
    for p in ['vitals','labs']:
        for v in range(2):
            for a in ['avg','min','max','first','last']:
                names.append(f'{p}_v{v}_{a}')
    X = X[:, :len(names)]
    dm = pd.DataFrame(X, columns=names)
    # inject missingness
    mask = np.random.RandomState(seed).rand(*dm.shape) < 0.08
    dm = dm.mask(mask)
    dm['mortality'] = y
    dm['hadm_id'] = np.arange(n)
    return dm
