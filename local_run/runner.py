"""Resumable local runner for local_run/research02_local.ipynb.

The notebook logic is executed verbatim (cell sources and driver sections are
exec'd as text). The only additions are orchestration: a data cache after the
build, one process per independent experiment unit, skip-if-done markers, and
a few extra output files for values the notebook only prints.

Usage:
  python runner.py build                 # data build + sanity numbers, then cache
  python runner.py unit <name>           # one experiment unit (reads cache)
  python runner.py all [--jobs N]        # build if needed, then all units
  python runner.py merge                 # merge per-task Step 22 parts

Environment: EHRML_QUICK=1 for the smoke test, EHRML_OUT_DIR for outputs.
"""
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / 'research02_local.ipynb'
CACHE = Path(os.environ.get('EHRML_CACHE', HERE / 'cache' / 'data_cache.pkl'))  # derived matrices only, stays local
CHECKS = CACHE.parent / 'build_checks.json'
QUICK = os.environ.get('EHRML_QUICK', '0') == '1'
OUT = Path(os.environ.setdefault(
    'EHRML_OUT_DIR', str(HERE / ('out_quick' if QUICK else 'out_full'))))
OUT.mkdir(parents=True, exist_ok=True)
DONE = OUT / '_done'
DONE.mkdir(exist_ok=True)
LOGS = OUT / 'logs'
LOGS.mkdir(exist_ok=True)

cells = [''.join(c['source']) for c in json.load(open(NB))['cells']]
DRIVER = cells[16]


def driver_sections():
    """Split the driver cell at its '# ---- N.' markers. Key 'head' = flags."""
    secs, key, buf = {}, 'head', []
    for line in DRIVER.splitlines(keepends=True):
        if line.startswith('# ---- '):
            secs[key] = ''.join(buf)
            key, buf = line[7], []
        buf.append(line)
    secs[key] = ''.join(buf)
    return secs


SEC = driver_sections()


def run_cells(ns, idx):
    for i in idx:
        t0 = time.time()
        print(f'[runner] cell {i} start', flush=True)
        exec(compile(cells[i], f'<cell {i}>', 'exec'), ns)
        print(f'[runner] cell {i} done in {time.time() - t0:.0f}s', flush=True)


def log(msg):
    print(f'[runner {time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------- build
def build():
    ns = {'__name__': '__main__'}
    t0 = time.time()
    run_cells(ns, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14])   # data, helpers, step defs
    exec(SEC['head'] + SEC['0'] + SEC['1'], ns)       # 80% eICU filter, LOS landmarks
    # The one Step 24 statement that needs the raw-event tables (verbatim).
    s4 = SEC['4']
    a = s4.index('dm_mimic24_full = build_dm(')
    b = s4.index("anchor='admittime')", a) + len("anchor='admittime')")
    exec(s4[a:b], ns)

    ec = ns['eicu_cohort']
    checks = {
        'mimic_icd_cohort_episodes': int(len(ns['icd_cohort'])),
        'mimic_icd_cohort_patients': int(ns['icd_cohort']['subject_id'].nunique()),
        'mimic_24h_matrix_rows': int(len(ns['dm_icd_24h_hosp'])),
        'mimic_24h_matrix_features': int(len([c for c in ns['dm_icd_24h_hosp'].columns
                                              if c not in ns['META_COLS']])),
        'eicu_cohort_episodes': int(len(ec)),
        'eicu_cohort_patients': int(ec['uniquepid'].nunique()),
        'eicu_cohort_mortality': float(ec['mortality'].mean()),
        'eicu_72h_rows_50pct_build': int(len(ns['dm_eicu_72h_full'])),
        'eicu_72h_features_at_80pct': int(ns['n_feat']),
        'eicu_72h_features_at_50pct': int(len([c for c in ns['dm_eicu_72h_full'].columns
                                               if c not in ns['META_COLS']])),
        'los_landmark_tasks': {t[0]: [int(len(t[1])), float(t[1][t[2]].mean())]
                               for t in ns['los_tasks']},
        'mimic24_unfiltered_features': int(len([c for c in ns['dm_mimic24_full'].columns
                                                if c not in ns['META_COLS']])),
        'build_seconds': round(time.time() - t0),
    }
    keep = ['dm_icd_24h_hosp_qc', 'dm_eicu_72h_full', 'COVERAGE_PREFILTER',
            'los_tasks', 'dm_mimic24_full']
    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump({k: ns[k] for k in keep}, f)
    json.dump(checks, open(CHECKS, 'w'), indent=1)
    log('build checks: ' + json.dumps(checks))


# --------------------------------------------------------------------- units
def unit_ns():
    """Definitions only (cells 1, 8, 12, 14) plus the cached matrices, then the
    driver flags and section 0 verbatim (rebuilds dm_eicu_72h_qc)."""
    ns = {'__name__': '__main__'}
    run_cells(ns, [1, 8, 12, 14])
    ns['qc_clip_features'] = _qc_clip(ns)
    data = pickle.load(open(CACHE, 'rb'))
    ns.update(data)
    ns['dm_eicu_72h'] = data['dm_eicu_72h_full']      # what the build cell returns (50%)
    exec(SEC['head'] + SEC['0'], ns)
    exec("eicu_sub = stratified_subsample(dm_eicu_72h_qc, 'mortality', EICU_N, seed=7)", ns)
    return ns


def _qc_clip(ns):
    """qc_clip_features exactly as cell 9 defines it (cell 9 also runs a line
    that needs the uncached raw matrix, so only the def is taken)."""
    src = cells[9]
    exec(src[:src.index('dm_icd_24h_hosp_qc =')], ns)
    return ns['qc_clip_features']


def tasks_main_src():
    s2 = SEC['2']
    a = s2.index('tasks_main = [')
    b = s2.index("if 'LOS>7' in t[0]]", a) + len("if 'LOS>7' in t[0]]")
    return s2[a:b]


def run_unit(name):
    ns = unit_ns()
    exec(tasks_main_src(), ns)
    if name.startswith('s22orig_'):
        k = int(name.split('_')[1])
        exec(f"run_step22([tasks_main[{k}]], roster='original', n_repeats=STEP22_REPEATS, "
             f"out_prefix='step22part{k}')", ns)
    elif name.startswith('s22div_'):
        k = int(name.split('_')[1])
        exec("tasks_div = [tasks_main[0]] + [t for t in los_tasks if 'LOS>7' in t[0]]", ns)
        exec(f"run_step22([tasks_div[{k}]], roster='diverse', n_repeats=STEP22_REPEATS, "
             f"out_prefix='step22part{k}')", ns)
    elif name == 's23':
        exec(SEC['3'], ns)
    elif name == 's24':
        s4 = SEC['4']
        a = s4.index('dm_mimic24_full = build_dm(')
        b = s4.index("anchor='admittime')", a) + len("anchor='admittime')")
        src = s4[:a] + '# dm_mimic24_full loaded from cache (built verbatim in build)' + s4[b:]
        # also keep the coverage band counts the notebook only prints
        src = src.replace("coverage_report(cov_eicu, 'eICU 72h')",
                          "_c1 = coverage_report(cov_eicu, 'eICU 72h')")
        src = src.replace("coverage_report(dm_mimic24_full[feats_m].notna().mean(), 'MIMIC-IV 24h')",
                          "_c2 = coverage_report(dm_mimic24_full[feats_m].notna().mean(), 'MIMIC-IV 24h')")
        exec(src, ns)
        import pandas as pd
        pd.DataFrame({'eICU 72h': ns['_c1'], 'MIMIC-IV 24h': ns['_c2']}).to_csv(
            OUT / 'step24_coverage_bands.csv')
    elif name == 's25':
        exec(SEC['5'], ns)
        json.dump({k: {'mean_jaccard': v['mean_jaccard'], 'min_jaccard': v['min_jaccard']}
                   for k, v in ns['shap_res'].items()},
                  open(OUT / 'step25_shap_jaccard.json', 'w'), indent=1)
    elif name == 's26_los14':
        # ADDITION (not in the notebook driver): the LOS>14 landmark task is built
        # by los_landmark_tasks but never scored there. Same arch_compare, original roster.
        exec("run_step22([t for t in los_tasks if 'LOS>14' in t[0]], roster='original', "
             "n_repeats=STEP22_REPEATS, out_prefix='step26_los14')", ns)
    else:
        raise ValueError(name)
    if name == 's26_los14':
        import pandas as pd
        rows = [{'task': t[0], 'n': len(t[1]), 'prevalence': t[1][t[2]].mean()}
                for t in ns['los_tasks']]
        pd.DataFrame(rows).to_csv(OUT / 'step26_landmark_cohorts.csv', index=False)


UNITS = ['s22orig_1', 's22orig_0', 's22orig_2', 's22div_0', 's22div_1',
         's26_los14', 's23', 's24', 's25']   # slowest first


def merge():
    import pandas as pd
    for roster, n in (('original', 3), ('diverse', 2)):
        for kind in ('summary', 'deltas', 'diversity'):
            parts = []
            for k in range(n):
                f = OUT / f'step22part{k}_{roster}_{kind}.csv'
                if f.exists():
                    # diversity tables are indexed by task (run_step22 writes the index)
                    parts.append(pd.read_csv(f, index_col=0 if kind == 'diversity' else None))
            if parts:
                pd.concat(parts).to_csv(OUT / f'step22_{roster}_{kind}.csv',
                                        index=(kind == 'diversity'))
    log('merged step22 parts')


# ----------------------------------------------------------------- orchestrate
def run_all(jobs):
    t0 = time.time()
    log(f'QUICK={QUICK} OUT={OUT} jobs={jobs}')
    if not CACHE.exists():
        log('build: start')
        r = subprocess.run([sys.executable, '-u', __file__, 'build'],
                           stdout=open(LOGS / 'build.log', 'w'), stderr=subprocess.STDOUT)
        log(f'build: exit {r.returncode} after {time.time() - t0:.0f}s')
        if r.returncode:
            sys.exit(r.returncode)
    else:
        log('build: cache present, skipped')
    log('build checks: ' + CHECKS.read_text().replace('\n', ' '))
    if os.environ.get('EHRML_BUILD_ONLY') == '1':
        return
    threads = str(max(1, 24 // jobs))
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS=threads, LOKY_MAX_CPU_COUNT=threads,
               MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
    units = os.environ['EHRML_UNITS'].split(',') if os.environ.get('EHRML_UNITS') else UNITS
    todo = [u for u in units if not (DONE / u).exists()]
    log(f'units to run: {todo} (skipped done: {[u for u in UNITS if u not in todo]})')
    running = {}
    while todo or running:
        while todo and len(running) < jobs:
            u = todo.pop(0)
            p = subprocess.Popen([sys.executable, __file__, 'unit', u], env=env,
                                 stdout=open(LOGS / f'{u}.log', 'w'), stderr=subprocess.STDOUT)
            running[u] = (p, time.time())
            log(f'start {u} (threads={threads})')
        time.sleep(20)
        for u, (p, ts) in list(running.items()):
            if p.poll() is not None:
                del running[u]
                if p.returncode == 0:
                    (DONE / u).touch()
                    log(f'done  {u} in {time.time() - ts:.0f}s')
                else:
                    log(f'FAIL  {u} exit {p.returncode}; see {LOGS / (u + ".log")}')
    merge()
    log(f'all finished in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    os.chdir(HERE)
    cmd = sys.argv[1]
    if cmd == 'build':
        build()
    elif cmd == 'unit':
        run_unit(sys.argv[2])
    elif cmd == 'merge':
        merge()
    elif cmd == 'all':
        jobs = int(sys.argv[sys.argv.index('--jobs') + 1]) if '--jobs' in sys.argv else 8
        run_all(jobs)
