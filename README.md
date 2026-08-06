# shearWaveProcessing

Unified cardiac shear-wave elastography (SWE) processing, combining the acquisition/conversion
half (from `SWI/Zea`) and the visualization half (from `iq2sws`) into one modular pipeline.

Three stages, each runnable on its own if its inputs already exist:

1. **convert** — Verasonics `.mat` → zea `.hdf5` (converted RF buffers). No GPU needed.
2. **beamform** — RF → complex IQ, B-mode GIFs, and per-measurement shear-wave IQ, saved with
   the **ARF push location** and all scan parameters needed downstream.
3. **viz** — IQ → shear-wave **space-time plots** (+ speed), for **active** SWE (buffer 2 tracking,
   M-line drawn on the co-registered buffer-5 B-mode) and **passive** SWE (buffer 4), each with its
   own config/recipe.

## The 6-buffer sequence (S5-1 SWI Widebeam)

| Buffer | Role |
|--------|------|
| 1 | widebeam B-mode (orientation) |
| **2** | **active shear-wave**: reference + ARF push + tracking |
| 3 | focused B-mode |
| **4** | ultrafast diverging-wave B-mode (**passive** SWE source) |
| 5 | widebeam B-mode, one frame per SW measurement (co-registered with buffer 2) |
| 6 | long widebeam for strain |

## Install

Runs under the group's `zea` conda environment (Python 3.12, `KERAS_BACKEND=torch`):

```
D:\Luuk van Knippenberg\envs\zea_latest\python.exe -m pip install -e .
```

`zea` (the tue-bmd toolbox) must be importable; it provides the Verasonics converter and the
beamformer used by stages 1–2. Stage 3 needs only numpy/scipy/h5py/matplotlib/pyyaml.

## Usage

```
python run.py convert  <measurement folder>
python run.py beamform <measurement folder> [--no-gifs] [--overwrite]
python run.py viz      <measurement folder> --config configs/active.yaml  [--meas N] [--phantom]
python run.py viz      <measurement folder> --config configs/passive.yaml
python run.py all      <measurement folder> --config configs/active.yaml [--phantom]
```

`<measurement folder>` contains the Verasonics `CombinedData.mat` (+ any `RF_data_*.bin`).
Stages 1–2 write into `<folder>/output/`; stage 3 reads that IQ and writes montages + space-time
HDF5 into `<folder>/output/swp_active` (or `swp_passive`).

Stages 1 and 2 share one RF read, so `beamform` also produces the converted files — run `convert`
separately only if you want the zea database copies without beamforming. Because each stage skips
inputs that already exist, you can **start from stage 3** whenever the IQ files are present.

If the folder has only the runtime `AcquisitionParametersAndECG.mat` (the dynamic parameters saved
during acquisition) and no `CombinedData.mat`, stages 1–2 build `CombinedData.mat` first by merging
that file with the constant base config (`swp.acquisition.ensure_combined_data`). This currently
runs the ported MATLAB merge (`src/swp/acquisition/matlab/make_combined_data.m`) via `matlab
-batch`, so MATLAB must be available (found on `PATH`, or set `SWP_MATLAB`); the base config
directory defaults to `D:\Luuk van Knippenberg\SWI\Base config files` (override with
`SWP_BASE_CONFIG_DIR`). A pure-Python v7.3 writer will replace the MATLAB step later.

## Phantom measurements

`--phantom` (on `viz` / `all`) adapts the **active** recipe for a phantom: there is no anatomical
M-line and no natural (buffer-4) shear wave, and the ARF push produces a wave that is **symmetric
about the focal point**, so the M-line is simply a **horizontal line through the push focal depth**
(`mline.type: horizontal_push`, depth from the stored `push_z`) — no interactive drawing and no
buffer-5 B-mode. The processing recipe (bands, filters, stored-focus `r0`, outward-directional) is
unchanged, so `r0` sits at the centre of the line and the wave travels outward both ways. Phantom
acquisitions also have fewer frames (no cardiac dependency), which the pipeline handles transparently.

```
python run.py beamform <phantom folder> --no-gifs
python run.py viz      <phantom folder> --config configs/active.yaml --phantom
```

`scripts/phantom_voltage_montage.py --root "<Phantom parent folder>"` assembles a side-by-side
montage of the per-folder space-time maps across a push-voltage sweep, labelled with the **delivered**
push voltage read from the data. Run `scripts/check_push_voltage.py --root "<Phantom parent folder>"`
first to confirm the sweep actually varied — the push TPC profile can silently clamp the voltage at
its `highVoltageLimit`. Full workflow + the clamp lesson: [docs/phantom_voltage_sweep.md](docs/phantom_voltage_sweep.md).

## Interactive method-exploration GUI (`swp_gui/`)

A Streamlit app to **experiment visually with every step from IQ/RF to the space-time plot** — pick a
folder + push, then tune each stage and see the space-time plot, B-mode + M-line, metrics, and a
built-in **no-push control** (the same recipe run on the pre-push reference, which reveals whether a
recipe images the ARF wave or cardiac motion). Every step's source code is viewable inline.

```
pip install -e ".[gui]"          # adds streamlit
KERAS_BACKEND=torch  D:\Luuk van Knippenberg\envs\zea_latest\python.exe -m streamlit run swp_gui/app.py
```

The four filter stages are **ordered, add/remove chains of methods** (e.g. polynomial detrend → band-pass;
or a spatial + slow-time IQ pre-filter), with per-step and global **reset**. The M-line is shown as the
**resampled constant-arc-length spline** (the exact curve used for processing) with its **offset lines**
and anchors, and the acquisition constants (f0/PRF/c/dz) are shown read-only.

Stages (all tunable, code viewable): **1** IQ pre-filter (spatial/slow-time low-pass, SVD clutter,
**bulk-motion compensation**) · **2** displacement (Loupas / Kasai / complex-IQ xcorr / **RF cross-
correlation on a fine local re-beamform**; frame-to-frame or vs-reference; displacement/velocity/
acceleration; **adaptive axial kernel from RF coherence**; lateral kernel) · **3** cardiac-motion removal
(**quality mask**, band-pass, high-pass, polynomial, SVD-on-displacement, **reference poly-extrapolation /
adaptive high-pass / reference-subspace projection**, phase-unwrap, axial-strain) · **4** spatial
(Gaussian, median, **bilateral, non-local means, Perona–Malik anisotropic diffusion, coherence-enhancing
tensor diffusion**) · **5** temporal (moving mean/median, **Savitzky–Golay**) · **6** M-line offset
averaging (mean/median) · **7** directional (outward/left/right) · **8** speed (TOF-RANSAC / Radon /
TOF-xcorr) + quality metrics. Recipes download as YAML.

The **RF cross-correlation** estimator (and the "use fine grid" toggle) re-beamform buffer 2 locally on
a fine axial grid around the M-line (`src/swp/acquisition/finegrid.py`); the data is NS200BW real RF, so
this is genuine fine-grid RF tracking. New processing methods live in `src/swp/viz/` (reusable outside
the GUI): `filters/experimental.py`, `estimators/rf_ncc.py`. Background + references:
[docs/literature_review.md](docs/literature_review.md).

## M-line selection

The septal M-line is drawn interactively (ported from `SWI/Zea/swi_mline.py`): click points on the
B-mode in any order, drag to adjust, Enter to finish. It is saved as `output/mlines/*.npz`
(`points` (k,2)=(x,z) m + `n_samples`) and **reused automatically** on later runs, so batch
processing is unattended once the lines exist. Active lines are drawn on the buffer-5 frame for each
push; passive lines on the buffer-4 stream itself.

## Visualization recipe

The active recipe is the settled `iq2sws` in-vivo consensus (Loupas, relative-to-reference
displacement, drop frame 0, band-pass 120–700 Hz, Gaussian smooth σz 0.6 / σx 1.2 mm, temporal
moving-mean 3, outward-directional), ranked by `origin_coherence`; the montage shows three
band-pass bands side by side (a line present in all three is real). The wave origin `r0` is anchored
on the **stored push location** (`focus.mode: stored`), not a hard-coded assumption.

**Passive SWE is an experimental framework**: same core, `frame_to_frame` displacement (no reference),
a starting band-pass recipe in `configs/passive.yaml` that still needs tuning against real passive
data, and origin handling left as a TODO.

## Layout

```
run.py                 stage driver (convert / beamform / viz / all)
configs/               active.yaml, passive.yaml
scripts/               phantom_voltage_montage.py (cross-folder voltage-sweep montage),
                       check_push_voltage.py (delivered push-voltage sweep check)
docs/                  HANDOFF.md, phantom_voltage_sweep.md (phantom sweep runbook)
src/swp/
  acquisition/         stages 1-2, ported from SWI/Zea (beamform, sequence, gifs, txsettings, scanparams);
                       combined.py + matlab/make_combined_data.m build CombinedData.mat from the runtime .mat;
                       pushvoltage.py reads the delivered ARF push voltage (TPC profile) from a folder
  mline/               interactive M-line selection, ported from SWI/Zea/swi_mline.py
  viz/                 stage 3 core, ported from iq2sws (io/core/estimators/filters/speed/viz/metrics/pipeline/runconfig)
```
