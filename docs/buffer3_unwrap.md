# Buffer 3 is a circular buffer: unwrapping it (2026-09-25)

Buffer 3 is the focused orientation loop at the start of every acquisition. Its stored frames are
**rotated**: frame 0 of the file is not the oldest frame. Nothing marks where the rotation starts,
so every buffer-3 timestamp and cardiac phase used before 2026-09-25 is wrong, sometimes by up to
half a heartbeat. Buffers 1, 2, 4, 5 and 6 are not affected.

The fix is `swp.acquisition.unwrap`. It puts the frames of the converted RF file and of the IQ file
in chronological order, and it stores the true frame times. It runs automatically after
beamforming, and `scripts/unwrap_buffer3.py` retrofits folders beamformed earlier.

## How it was found

- While drawing passive M-lines, the buffer-3 frame labelled "R-peak" plainly showed another
  cardiac phase than buffers 1 and 4 (C000000001, C000000005).
- The study-wide fits (`study/analysis/buffer3_timing_fit.py`, `buffer3_frame_order.py`) pointed
  the same way. Compared with the correctly timed buffer 1, buffer 3 matched far better once its
  frame order was rotated:
  - 0 of 27 folders with 32 frames were best in the stored order;
  - on 39 of those folders, the same fit on correctly ordered buffer-4 frames (the control) never
    rotated anything (largest gain 0.005), while buffer 3 did (median gain 0.052).
- A dedicated scanner recording (`Z:\2026_09_25 LastFrame check`) closed the case:
  - the RF on disk equals the full-workspace `RcvData{3}` byte for byte, so saving and conversion
    are faithful;
  - the frames jump once, between stored slots 8 and 9 (neighbour correlation 0.54 against a
    median of 0.91), while the wrap from slot 32 back to slot 1 is continuous.

## Why the saved head is useless

The sequence (`SWI/Widebeam/SetUp_SWI_Widebeam.m`) writes frame *i* of every pass of the focused
live loop into receive slot *i* and jumps back to slot 1. The button that starts the acquisition
(`set&Run` with a new start event) leaves the loop wherever the hardware happens to be.

VSX would record the head in `Resource.RcvBuffer(3).lastFrame`, and zea's converter unwraps with
it (`VerasonicsFile.get_raw_data_order`: first frame = `(lastFrame + 1) mod N`). But the
Sequence Programming Manual (p57) says the field *"is set by the system when exiting a sequence"*,
and the Tutorial (p64) says *"when a script freezes or exits"*. Verasonics' own `saveRF.m` refuses
to save unless VSX is frozen.

`SaveRFData.m` calls `copyBuffers` mid-sequence. The value it gets back is `numFrames` for every
buffer, in every folder. The merge of `Resource` and `Resource_func` in `SaveRFData.m` is **not**
the cause: `Resource_func` already carries `numFrames`. For buffers 1, 2, 4, 5 and 6, which are
written in one complete pass, `numFrames` happens to be correct. For buffer 3 it is not.

## The three candidate methods, and the answer

The trigger log is correct: R-peaks, plus one frame trigger per buffer-3 frame, fired on the
frame's first transmit. The candidates:

| method | how | verdict |
|---|---|---|
| **trigger count** | L triggers in the live run: the loop wrote slots 1…N cyclically and the last trigger belongs to the frame that was being acquired when the loop was left (aborted mid-frame, never transferred), so the oldest stored slot is `(L − 1) mod N` | **exact**, but only where the log still holds the run's start |
| **buffer-1 similarity** | each stored frame gets a trigger time and cardiac phase under each candidate head and is compared with the buffer-1 frame at the same phase; the head with the most similar pairs wins; the shift is fixed at −1 (mid-frame abort) | **accurate** where the trigger count is missing and the ECG is trustworthy |
| frame continuity | the least similar cyclic neighbour pair of stored frames is the wrap; computed on the IQ and on beamforming-free RF (incoherent channel-sum A-lines) | **rejected** |

Calibration against the trigger count (`study/analysis/buffer3_unwrap_calibration.py`, all 145
complete-run folders: 105 with 26 frames, 40 with 32; 119 with a trustworthy ECG):

| method | exact | within one slot |
|---|---|---|
| buffer-1 similarity, shift fixed −1 | 89 % | 95 % |
| … accepted at margin ≥ 0.006 | **98.9 %** (75 % of folders accepted; the one miss is one slot) | 100 % |
| buffer-1 similarity, shift free | 70 % | 85 % |
| IQ continuity | 39 % | 63 % |
| RF continuity (18-folder subset of the validation sample) | 56 % | 72 % |

Three findings from the calibration:

- **The loop is always left mid-frame.** The trigger-count head `(L − 1) mod N` and the buffer-1
  head agree exactly in 89 % of folders, and every disagreement but one
  has a buffer-1 margin ≤ 0.005. A "clean stop between frames" never shows up as more
  than noise.
- **Freeing the ±1 shift hurts the buffer-1 method.** A one-slot rotation and a one-frame shift
  are nearly indistinguishable, so leaving both free lets the fit swap one for the other.
- **Continuity fails on in-vivo data.** Fast cardiac motion produces neighbour pairs as
  dissimilar as the wrap itself. The check dataset, where it worked, was a fortunate case.

Where the trigger log keeps the start of the live run: 145 of 341 ready folders (43 %). In the
rest, the circular log (1500 or 2000 triggers, shared with every buffer) had already overwritten
it; most 32-frame acquisitions are in this group.

## What the helper does

Decision rule (`swp.acquisition.unwrap.estimate`):

1. Trigger count, if the run start is in the log.
2. Otherwise buffer-1 similarity, if the ECG is trustworthy and the margin is ≥ 0.006.
3. Otherwise **ambiguous**: stored order kept, `buffer_unwrapped = false`. These are for review.

The buffer-1 method also runs where the trigger count is available, and its agreement is stored
(`buffer1_agrees`). Every folder therefore carries a running cross-check.

Stored in **both** `output/converted/*_buffer3.hdf5` and `output/*_buffer3_iq.hdf5`:

| dataset | meaning |
|---|---|
| per-frame data (`raw_data`, `values`) | reordered: frame 0 is the oldest |
| `custom/buffer_unwrapped` | true once the frames are chronological |
| `custom/unwrap_status` | `unwrapped`, `chronological` (head already in slot 0), `ambiguous` or `no-data` |
| `custom/unwrap_first_frame` | the original (.mat) slot that was acquired first; −1 if unknown |
| `custom/unwrap_shift_frames` | −1 (the last trigger has no stored frame) |
| `custom/unwrap_method`, `unwrap_details`, `unwrap_time`, `unwrap_version` | how it was decided (JSON details: trigger count, buffer-1 head and margin, agreement) |
| `custom/frame_time_ms` | chronological frame-**centre** times on the trigger-log clock (trigger + half a frame: the trigger fires on the first of 73 lines, a frame lasts 39.4 ms) |
| `custom/frame_rpeak_phase_ms` | time since the preceding logged R-peak, per frame |
| IQ `timestamps` | the frame-centre times relative to frame 0 |
| `/provenance_unwrap` | provenance stamp of the unwrap |

Safety properties:

- **Atomic.** A reordered copy is written next to the file and swapped in with `os.replace`, so an
  interrupted run never leaves a half-permuted file.
- **Idempotent.** A file carrying `custom/unwrap_status` is never permuted again.
- **Mixed states are handled.** If one file is flagged and the other is not, the stored head is
  reused. A converted file reconverted from the .mat is re-aligned. An IQ re-beamformed from the
  unwrapped RF is left as is.
- Flag-only changes (ambiguous, or head already in slot 0) are written in place, without copying
  the ~1 GB RF file.
- The buffer-3 GIF is re-rendered after a reorder.

## Running it

```
python scripts/unwrap_buffer3.py --root "Z:/raw_data" --dry-run    # estimate only, write nothing
python scripts/unwrap_buffer3.py --root "Z:/raw_data"              # rewrite (safe to re-run)
python scripts/unwrap_buffer3.py --folder "<folder>"
```

- One CSV row per folder goes to `study/logs/buffer3_unwrap_<date>.csv`.
- A folder is only touched once its beamforming is complete (the buffer-4 GIF exists).
- `beamform.process_folder(..., unwrap=True)` (the default) runs it after every new beamform, so
  newly converted folders come out chronological.
- Cost: ~6 s to estimate. Rewriting a ~1 GB converted file over the NAS takes a minute or two.
- Large runs belong on the Linux server (`docs/linux_server.md`).

## Applied so far

**The 44 September folders (2026-09-25, from Windows, 24 min;**
`study/logs/buffer3_unwrap_september_20260925.csv`**):**

| result | folders |
|---|---|
| resolved by trigger count | 12 (C000000001 in the first test run; one already chronological) |
| resolved by buffer-1 similarity | 25 |
| **ambiguous** (stored order kept) | 7 |

- The 7 ambiguous folders:
  - no trustworthy ECG: C000000014, 17, 21, 30;
  - buffer-1 margin below 0.006: C000000036, 39, 45.
- Cross-check where both ran: buffer 1 matched the trigger count in 7 of 8 folders. The miss
  (C000000035) is one slot off, at margin 0.0036, below the acceptance threshold anyway.
- The rest of the study: `docs/TODO.md` item 15.

**Buffer-3 reconstruction variants.** Files such as REFoCUS adjoint, Tikhonov/TSVD, incoherent,
deconvolution and pfield-normalised (`*_buffer3_<variant>_iq.hdf5`) come from the same stored-order
RF, so they carry the same rotation. `unwrap_buffer3` then calls `unwrap_variants`, which applies
the head stored in the main file (never re-estimated), with the same flags and frame times, and
re-renders the variant's GIF. In the September folders: 42 reordered, 8 flagged only.

**R-peak-aligned montage** (`study/analysis/buffer3_rpeak_montage.py` →
`study/montages/all_buffer3_refocus_rpeak_aligned.gif`, stills `_R.png` / `_mid.png`). This
replaces `all_buffer3_refocus_adaptive.gif`, which started every tile at stored slot 0.
- **White (34 subjects):** resampled onto a common phase axis (fraction of each subject's RR),
  so every tile starts at the R-peak and one loop is one beat.
- **Orange (3):** unwrapped, but with no valid ECG. They play in chronological order and cannot
  be phase-aligned.
- **Red (7):** ambiguous, in stored order.

Why the old montage looked roughly synchronised: in the 12 September folders with an exact trigger
count, stored slot 0 fell at 0.14-0.71 of the RR, clustered mid-cycle (8 of 12 at 0.36-0.63). The
tiles started within about half a beat of each other, not at one phase.

## Consumers

- **M-line tool** (`swp.manual.frames`):
  - for folders with a resolved unwrap, the buffer-3 frame is chosen by its stored frame-centre
    phase;
  - for ambiguous or not-yet-unwrapped folders, by anatomy match to buffer 1;
  - without a valid ECG, buffer 1 is chosen by anatomy match to buffer 4.
- **Anything else that read buffer-3 frame times or phases before 2026-09-25 is suspect.** This
  includes `report/passive_methods/passive_methods_v3.pdf`: its finding that buffer-3 lines are
  "2–3 beats away, 3.1 mm off" used the wrong frame assignment.

## For future acquisitions

Either change fixes it at the source:

- **Freeze before saving.** `copyBuffers` from the freeze state, as Verasonics' `saveRF.m` does.
  Then check whether `lastFrame` is correct.
- **Make the stop deterministic.** Instead of `set&Run`, redirect the loop's jump-back `SeqControl`
  to the widebeam start, so the loop always finishes its pass. Then `lastFrame = N` is true.

Either way, verify it with a recording that has a visible event (e.g. lifting the probe) just
before the loop is left.

## Files

| file | purpose |
|---|---|
| `src/swp/acquisition/unwrap.py` | estimators, decision, atomic rewrite |
| `scripts/unwrap_buffer3.py` | retrofit command |
| `tests/test_unwrap.py` | reorder correctness, idempotency, mixed states, trigger count |
| `study/analysis/buffer3_unwrap_calibration.py` → `study/logs/buffer3_unwrap_calibration.csv` | methods against the exact trigger count |
| `study/analysis/buffer3_unwrap_validation.py` → `study/logs/buffer3_unwrap_validation.csv`, `study/montages/buffer3_unwrap/` | RF / IQ / buffer-1 on a stratified sample |
| `study/analysis/buffer3_timing_fit.py`, `buffer3_frame_order.py` | the first evidence |
| `study/analysis/buffer3_rpeak_montage.py` → `study/montages/all_buffer3_refocus_rpeak_aligned*` | R-peak-aligned montage of every subject from the unwrapped frame times |
