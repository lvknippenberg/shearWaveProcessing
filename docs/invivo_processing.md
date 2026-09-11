# In-vivo processing runbook (`Z:\raw_data`)

How to turn a raw in-vivo SWE measurement folder into zea HDF5 + IQ + real-time B-mode GIFs,
reproducibly. Stages 1–2 only; stage 3 (`viz` / `passive`) needs interactive M-line drawing and is
covered in the README.

## TL;DR

```powershell
$env:KERAS_BACKEND="torch"
$env:SWP_BASE_CONFIG_DIR="Z:\Base config files"
$py = "D:\Luuk van Knippenberg\envs\zea_latest\python.exe"

# 0. audit first - seconds, reads no RF, writes nothing
& $py scripts\process_raw_data.py --root "Z:\raw_data" --check

# 1. one subject
& $py scripts\process_raw_data.py --root "Z:\raw_data" --subject C000000001

# 2. the whole study (skips folders already done)
& $py scripts\process_raw_data.py --root "Z:\raw_data"
```

A single folder, equivalently: `python run.py beamform "<folder>"`.

**No base config needs to be named.** It is auto-selected and then asserted — see below.

## The base config, and why it is asserted

The acquisition saves only the *dynamic* parameters (`AcquisitionParametersAndECG.mat` +
`RF_data_*.bin`). The *constant* parameters — including `Resource.RcvBuffer`, the map from RF bytes
to frames — come from a **base config** in `Z:\Base config files`, and the two are merged into
`CombinedData.mat` by `matlab -batch make_combined_data(...)`.

The in-vivo base config **changed between acquisition campaigns**:

| base config | `numFrames` | folders in this study |
|---|---|---|
| `S5_1_SWI_PulseInversion_P1-6_runtime.mat` | `[90, 20, 26, 926, 20, 268]` | 9 |
| `S5_1_SWI_PulseInversion_P11-14_runtime.mat` | `[106, 22, 32, 1112, 22, 318]` | 4 |
| `S5_1_SWI_PulseInversion_P15-xx_runtime.mat` | `[106, 24, 32, 1112, 24, 318]` | 31 |

Using the wrong one slices **every** RF buffer out of the wrong region. Nothing downstream
complains — beamforming succeeds and the B-modes look plausible — so this is a silent-corruption
failure mode. `make_combined_data.m` originally hardcoded the newest campaign (`P15-xx`) for any
in-vivo folder, which is wrong for **13 of the 44 folders** in this study.

It is now handled in `swp.acquisition.combined`:

1. **Auto-selection** — every candidate in the base config dir is checked by comparing its
   `Resource.RcvBuffer.numFrames`/`rowsPerFrame` against the folder's own `RF_frames`/`RF_rows`.
   Exactly one in-vivo config matches, so the right campaign is picked without being named.
2. **Assertion** — the merged `CombinedData.mat` carries *both* halves, so it is validated before
   it is ever read (`validate_combined_data`). This runs on every path, including an
   **already-existing** `CombinedData.mat` — a file built by an earlier run with the old default is
   rejected, not silently reused. A mismatch raises `BaseConfigMismatch` with both layouts printed.
3. **Phantoms defer** — every `BaseConfig_10frames_*` shares one `RcvBuffer`, so the layout cannot
   choose between them; the discriminator is the push setting in the *file name*. When several
   match, selection falls back to `make_combined_data.m`'s naming rule. The merged file is still
   validated.

Override with `--base-config` / `SWP_BASE_CONFIG` if you must — it is still checked, since naming
one by hand is exactly where the wrong campaign gets picked.

### If `--check` reports `NO MATCH`

The campaign's base config is missing from `SWP_BASE_CONFIG_DIR`. Find the runtime `.mat` saved
with that acquisition campaign and drop it in; do **not** force a near-miss with `--base-config`.

### If `--check` reports `STALE`

That folder's `CombinedData.mat` was built with a base config that does not describe it (likely
before this check existed). Rebuild: `scripts\process_raw_data.py --folder "<folder>" --overwrite`.

## What gets written

Into `<folder>\output\`:

| file | what |
|---|---|
| `converted\CombinedData_buffer<k>.hdf5` | stage 1 — converted RF, the zea database copy |
| `CombinedData_buffer<k>_iq.hdf5` | stage 2 — beamformed complex IQ + timestamps |
| `CombinedData_buffer2_meas<m>_iq.hdf5` | per-push shear-wave IQ + reference IQ + push location |
| `CombinedData_buffer<k>_iq.gif` | real-time B-mode |

Roughly 4.8 GB per measurement folder, so budget ~210 GB for all 44.

## Buffers

| Buffer | Role | Real-time GIF |
|---|---|---|
| 1 | widebeam B-mode, ~88 Hz | every 2nd frame @ 50 fps |
| 2 | active shear-wave (push + tracking, 3.7 kHz) | slow motion — ~16 ms is unwatchable in real time |
| 3 | focused B-mode, ~25 Hz | all frames @ 25 fps |
| 4 | ultrafast diverging-wave, ~926 Hz (passive source) | sub-sampled ~50/926 @ 50 fps |
| 5 | B-mode, one frame per push — **rate is `SW.ActualFPS` (~18 Hz)**, not `Bmode_WB.ActualFPS` | all frames @ 20 fps |
| 6 | strain | often **not saved** — no `RF_data_6.bin` |

A missing `RF_data_k.bin` means that buffer was not dumped; that is not an error. Check with
`RF_rows[k] * RF_cols[k] * RF_frames[k] * 2` bytes against the file size.

## Notes

* GPU: pick an idle one with `CUDA_VISIBLE_DEVICES=<n>` (`nvidia-smi`) — the box is shared. The
  beamformer sizes its patches to the GPU and retries on OOM, so it completes regardless.
* Runtime: ~10 min per folder, dominated by buffer 4 (926 frames) and the serial lzf read-back of
  the converted files.
* Failures in a batch are reported per folder and do not stop the run; see the closing SUMMARY.
