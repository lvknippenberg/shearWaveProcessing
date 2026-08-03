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
python run.py viz      <measurement folder> --config configs/active.yaml  [--meas N]
python run.py viz      <measurement folder> --config configs/passive.yaml
python run.py all      <measurement folder> --config configs/active.yaml
```

`<measurement folder>` contains the Verasonics `CombinedData.mat` (+ any `RF_data_*.bin`).
Stages 1–2 write into `<folder>/output/`; stage 3 reads that IQ and writes montages + space-time
HDF5 into `<folder>/output/swp_active` (or `swp_passive`).

Stages 1 and 2 share one RF read, so `beamform` also produces the converted files — run `convert`
separately only if you want the zea database copies without beamforming. Because each stage skips
inputs that already exist, you can **start from stage 3** whenever the IQ files are present.

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
src/swp/
  acquisition/         stages 1-2, ported from SWI/Zea (beamform, sequence, gifs, txsettings, scanparams)
  mline/               interactive M-line selection, ported from SWI/Zea/swi_mline.py
  viz/                 stage 3 core, ported from iq2sws (io/core/estimators/filters/speed/viz/metrics/pipeline/runconfig)
```
