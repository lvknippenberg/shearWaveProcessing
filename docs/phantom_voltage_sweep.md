# Phantom push-voltage sweep — runbook

How to acquire and analyse a phantom ARF shear-wave **push-voltage sweep** (one measurement per
push voltage), and how to read the result. Written after the 2026-07-30 sweep, whose lesson is in
§4.

## 0. TL;DR

```bash
# in the zea env (KERAS_BACKEND=torch)
cd "D:/Luuk van Knippenberg/Github/shearWaveProcessing"

# 1. sanity-check the delivered push voltages BEFORE processing
python scripts/check_push_voltage.py --root "<Phantom parent folder>"

# 2. per folder: build CombinedData.mat (auto) -> beamform -> phantom viz
for f in "<Phantom parent folder>"/*/ ; do
  python run.py beamform "$f" --no-gifs
  python run.py viz      "$f" --config configs/active.yaml --phantom
done

# 3. cross-folder montage vs delivered voltage
python scripts/phantom_voltage_montage.py --root "<Phantom parent folder>"
```

Each measurement folder holds the runtime `AcquisitionParametersAndECG.mat` + `RF_data_*.bin`.
Outputs land in `<folder>/output/` (IQ) and `<folder>/output/swp_active/` (space-time PNG + HDF5);
the montage is written to `<Phantom parent folder>/phantom_voltage_montage.png`.

## 1. What "phantom" changes

A phantom has **no anatomical M-line** and **no natural shear wave** (buffer 4 is not processed),
and the ARF push makes a wave that is **symmetric about the focal point**. The `--phantom` flag
(on `viz` / `all`) therefore:

- uses `mline.type: horizontal_push` — a horizontal M-line through the **stored push focal depth**
  (`push_z`), with the origin `r0` at the push `x` (the line centre); the outward-directional filter
  then keeps the wave travelling **both ways** out of the focus;
- needs **no interactive M-line drawing** and **no buffer-5 B-mode**;
- otherwise reuses the **default active recipe** unchanged (`configs/active.yaml`: bands, filters,
  stored-focus `r0`, outward-directional).

Phantom acquisitions also have **fewer frames** (no cardiac gating); this is handled transparently
(`source` is auto-detected as `phantom`).

## 2. `CombinedData.mat` is built automatically

The scanner saves only the dynamic `AcquisitionParametersAndECG.mat`. The beamformer needs the
merged constant+dynamic workspace `CombinedData.mat` (MATLAB v7.3). If it is missing, `find_mat`
builds it via `swp.acquisition.ensure_combined_data`, which currently runs the ported MATLAB merge
`src/swp/acquisition/matlab/make_combined_data.m` through `matlab -batch`.

- MATLAB must be available (found on `PATH`, or set env `SWP_MATLAB` to `matlab.exe`).
- Base config dir defaults to `D:\Luuk van Knippenberg\SWI\Base config files`
  (override with env `SWP_BASE_CONFIG_DIR`). It must contain
  `S5_1_SWI_PulseInversion_P15-xx_runtime.mat` (in-vivo), `..._runtime_2frames.mat` (phantom),
  and `NonzeroRFcolumns.mat`.
- Phantom vs in-vivo is detected from `RF_frames(1)` (`==2` → phantom, 2-frame base config).
- A pure-Python v7.3 writer to drop the MATLAB dependency is a TODO (zea's reader is h5py-only, so
  the file must be v7.3; scipy only writes v5). The MATLAB output was validated to be identical to a
  hand-made reference for every variable the reader touches (all 237 TX elements, RF_rows/cols/frames,
  SW.Nframes, NonzeroRFcolumns).

## 3. Reading the montage

`phantom_voltage_montage.py` shows displacement (top) and velocity (bottom) for the recipe band
(`bp120-700`), one column per measurement, **labelled with the delivered push voltage read from the
data**. The dashed line is the push origin `r0` (line centre); a real shear wave is the pair of
wavefronts fanning **outward** from `r0` (a "∧" opening downward in time). Panels share one colour
scale per row (`--per-panel-clim` to auto-scale each), so a genuinely stronger wave reads as stronger
contrast. `origin_coherence` (`oc`) in each title is the symmetric-origin score; higher = cleaner.

At **low push voltages (≤30 V) the wave is weak and noise-dominated** (wave/noise RMS < 1), so expect
a clear "clearer with voltage" trend only once the delivered voltage actually climbs well past 30 V.

## 4. The lesson from the 2026-07-30 sweep: the push voltage was clamped

The intended sweep was **20→50 V in 5 V steps** across 7 folders. The **delivered** push voltage
(TPC profile 5, read from the data) was **20, 25, 30, 30, 30, 30, 30**: the push profile's
`highVoltageLimit = maxHighVoltage = 30 V` **silently clamped** everything above 30 V, so
measurements 4–7 were all really 30 V (near-replicates). That is why no "clearer with voltage" trend
appears beyond 30 V — the higher voltages were never delivered.

**Before re-acquiring:** raise the push TPC profile's `highVoltageLimit` / `maxHighVoltage` above the
top of the intended sweep. **After acquiring:** run
`python scripts/check_push_voltage.py --root <folder>` — it prints the delivered voltage per
measurement and warns when any sits at the profile ceiling or when the sweep has repeats. Only trust
the sweep once it reports distinct, monotonically increasing voltages.

## 5. Files

| Path | Role |
|------|------|
| `run.py viz --phantom` | phantom space-time per measurement (horizontal push-focus M-line) |
| `src/swp/acquisition/combined.py` + `matlab/make_combined_data.m` | build `CombinedData.mat` |
| `src/swp/acquisition/pushvoltage.py` | read the delivered push voltage from a folder / `.mat` |
| `scripts/check_push_voltage.py` | pre/post-flight sweep check (delivered voltage per measurement) |
| `scripts/phantom_voltage_montage.py` | cross-folder montage vs delivered voltage |
