# Evaluation of the manual passive reading (shearWaveProcessing)

What it is: a re-runnable evaluation of the manual passive-SWE reading
(`shearWaveProcessing/scripts/passive_manual.py`). It answers seven questions: reproducibility
within and between acquisitions, M-line placement, septal thickness, event placement, slope
fitting, and the quality metric.

**Findings:** [REPORT_2026-10-05.md](REPORT_2026-10-05.md) (snapshot `20261005_1530`).

Written on 2026-10-05 in `D:\Luuk van Knippenberg\Claude\passive_manual_eval` (the repo was in use);
moved here on 2026-10-06. It only *reads* the `swp` package, configs, view labels and Z:. Paths come
from `common.py` (the enclosing repo, or env `SWP_REPO`; `SWP_RAW_DATA` for the data root).
Tracked in git: scripts, reports, `results/<snapshot>/` figures, summaries and tables. Not tracked
(regenerable): `snapshots/` (~0.4 GB each), `scratch/`, and the heavy caches listed below.

**Changes made from this evaluation (2026-10-06):** registered line pre-loads, the slider start
and the automatic tilt, and the scoring rule. See `docs/passive_manual.md`, "2026-10-06".

## Re-running (e.g. when the reading is complete)

Use `D:\Luuk van Knippenberg\envs\zea_latest\python.exe`, from this folder:

```
python run_all.py                          # new snapshot of Z:/raw_data + every analysis (~35 min)
python run_all.py --snapshot 20261005_1530 # same analyses on an existing snapshot
python run_all.py --skip 03b,04            # skip the slow image steps
```

Each step can also run on its own: `python 0X_*.py [--snapshot <stamp>]` (default: newest snapshot).
Results go to `results/<snapshot>/<step>/` as `summary.json`, CSV tables and PNG figures.
- **Same snapshot = same numbers.** The only randomness is fixed seeds: the bootstrap, the
  sensitivity sample and the figure jitter.
- **New snapshot = the current state of the reading.**
- Write a new `REPORT_<date>.md` by comparing the `summary.json` files.

| step | script | what | reads Z: images? | time |
|---|---|---|---|---|
| 00 | `00_snapshot.py` | copies every `output/swp_passive_manual/` (json/jsonl/npz, archives; no PNGs) to `snapshots/<stamp>/data`, builds `tables/` (folders, lines, windows, review, log, retest) | no (json/npz only) | 1 min |
| 01 | `01_features.py` | space-time features per window (`results/<snap>/features.csv`) | no | 1 min |
| 02 | `02_reproducibility.py` | points 1 + 2: within-acquisition pairs, REML subject/acquisition/beat, re-reading noise | no | 10-15 min (bootstrap) |
| 03 | `03_mline.py` | point 3: line geometry, event-line and general-line prediction by registration / wall search | yes (buffer 4) | 3 min |
| 03b | `03b_mline_sensitivity.py` | point 3: speed change for ±1/±2 mm, ±5° lines (re-processes IQ, worker code path) | yes (IQ) | ~25 min |
| 04 | `04_septum.py` | point 4: wall thickness on buffers 1/3/4, beats, adjacent frames, AVC/MVC frames | yes (B-mode) | 5 min |
| 05 | `05_events.py` | point 5: automatic vs reviewed windows, timing priors, label outliers | no | < 1 min |
| 06 | `06_slopes.py` | point 6: automatic speed estimators vs hand slopes | no | 1-2 min |
| 07 | `07_quality.py --sheets` | point 7: current vs new quality metric (LOSO), outliers + contact sheets, drift | sheets read slope PNGs | < 1 min |

Helpers: `common.py` (paths, snapshot loader), `stlib.py` (space-time line sampling, best line,
lag profile), `stats.py` (REML nested variance components, cluster bootstrap), `septum.py` (wall
edges). `scratch/` holds exploratory plots only.

Caches (delete to recompute): `results/<snap>/03_mline/raw.json`, `04_septum/measurements.csv`,
`06_slopes/estimates.csv`. Earlier method versions are kept as `*_v1*`.

## Caveats when re-running

- **The snapshot copies only records whose files exist at that moment.** The tables keep only
  records whose hashes chain up (general → windows → review → events → space-times → slopes),
  as `store.state` does. A folder half-way through a prompt simply has fewer slopes.
- **`retest.csv`** (same beat read twice) comes from the `archive_*_redetect_valves` folders of the
  27 first folders. If those folders are re-read again, the pairs change.
- **The quality model and the outlier list are fitted on the snapshot.** Expect them to change as
  the reading grows; the LOSO numbers are the honest ones.
- **The 03b sensitivity runs the real worker pipeline** on 24 windows. If `configs/passive_manual.yaml`
  changes, so do its views.
