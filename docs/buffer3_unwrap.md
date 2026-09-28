# Buffer 3 is a circular buffer: unwrapping it (2026-09-25)

Buffer 3 is the focused orientation loop at the start of every acquisition. Its stored frames are
**rotated**: frame 0 of the file is not the oldest frame. Nothing marks where the rotation starts,
so every buffer-3 timestamp and cardiac phase used before 2026-09-25 is wrong, sometimes by up to
half a heartbeat. Buffers 1, 2, 4, 5 and 6 are not affected.

The fix is `swp.acquisition.unwrap`. It puts the frames of the converted RF file and of the IQ file
in chronological order, and it stores the true frame times. It runs automatically after
beamforming, and `scripts/unwrap_buffer3.py` retrofits folders beamformed earlier.

> **2026-09-28 review - read this first.** A trigger-log segmentation bug makes the head (and the
> buffer-3 frame times) one slot off in ~31 % of folders, and the calibration below was run against
> that partly wrong reference. With it corrected, continuity is NOT useless and a combined
> estimator is ~98 % exact. See "2026-09-28: boundary bug and a better estimator" at the end.
> Numbers in the sections in between are the original 09-25 ones.

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

VERSION 2 (2026-09-28; VERSION 1 used buffer-1 similarity alone at margin ≥ 0.006):

1. Trigger count, if the parsed log (`triggerlog.parse_sequence`) holds the run start.
2. Otherwise, with a trustworthy ECG, the combined score (buffer-1 similarity + expected motion +
   0.5 × continuity, each z-scored) at margin ≥ 1.3.
3. Otherwise continuity alone at margin ≥ 0.055 (the only method without ECG).
4. Otherwise **ambiguous**: stored order kept, `buffer_unwrapped = false`. These are for review.

The combined score and continuity also run where the trigger count is available, and their
agreement is stored (`combined_agrees`, `continuity_agrees`): a running cross-check per folder.

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
- **Idempotent.** A file flagged by the current `unwrap_version` is never permuted again; a file
  flagged by an older version is re-estimated from its reconstructed stored order and rotated from
  its current head to the new one.
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

**The 48 second acquisitions (2026-09-28, from Windows, 4 workers, 9 min;**
`study/logs/buffer3_unwrap_second_acq_20260928.csv`, first vs second per subject in
`buffer3_unwrap_first_vs_second_20260928.csv`**):**

| result | 1st acquisition (44 + C1) | 2nd acquisition (48) |
|---|---|---|
| trigger count | 12 | 10 (C001-005, 009, 017, 028, 031, 037) |
| buffer-1 similarity | 25 | 30 |
| ambiguous | 7 | 8 (C008, 014, 015, 022, 034, 047: margin < 0.006; C012, 048: no valid ECG) |

- **The hope that later acquisitions are countable more often does not hold.** Paired over 44
  subjects the trigger count gained 5 and lost 7. Buffer 1 agreed with the trigger count in 8 of 8
  folders where both ran.
- **Why: the trigger log's budget, not the operator.** The log holds 2000 entries in C1-C4 and 1500
  from C5 on, and the SW acquisition *after* the live loop fills 1096 of them (26-frame campaign,
  C1-C10) or 1305-1314 (32-frame campaign, C11 on). That leaves the focused live view ~903 triggers
  (**35.6 s**, C1-C4), ~403 (**15.9 s**, C5-C10) or ~185 (**7.3 s**, C11 on). Any longer live view
  loses its start, first acquisition or not (`study/analysis/trigger_log_budget.py` ->
  `study/logs/trigger_log_budget_20260928.csv`: every truncated run fills exactly that budget).
  Countable: C1-C4 8/8, C5-C10 6/10, C11 on 8/74.
- So a larger trigger log (or clearing it when the live loop starts) would make every acquisition
  exact; the other source fixes are under "For future acquisitions".
- These were VERSION 1 results; see the 2026-09-28 section for the corrected (VERSION 2) re-run.

**Buffer-3 reconstruction variants.** Files such as REFoCUS adjoint, Tikhonov/TSVD, incoherent,
deconvolution and pfield-normalised (`*_buffer3_<variant>_iq.hdf5`) come from the same stored-order
RF, so they carry the same rotation. `unwrap_buffer3` then calls `unwrap_variants`, which applies
the head stored in the main file (never re-estimated), with the same flags and frame times, and
re-renders the variant's GIF. In the September folders: 42 reordered, 8 flagged only.

**Playback montages** (`study/analysis/buffer_playback_montage.py` + `buffer_playback_pair.py` →
`study/montages/playback/acq<N>_buffer<1|3>_<stored|unwrapped>.gif` and
`acq<N>_buffer3_stored_vs_unwrapped.gif`). Every tile plays all of its frames in file order; tiles
are deliberately **not** synchronised to the R-peak or to each other - buffers 1 and 3 are not
ECG-triggered, and the unwrap's purpose is continuous playback without the wrap jump. They replace
the 2026-09-25 R-peak-aligned montage (`buffer3_rpeak_montage.py`, removed 2026-09-28), which
resampled every tile onto an R-peak phase axis and so re-introduced jumps and repeated frames.

Why the old stored-order montage looked roughly synchronised: in the 12 September folders with an exact trigger
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
| `study/analysis/buffer_playback_montage.py`, `buffer_playback_pair.py` → `study/montages/playback/` | playback montages of the N-th acquisition, stored vs unwrapped |
| `study/analysis/triggerlog_structure_check.py` | the whole trigger log parsed into the acquisition sequence, all folders |
| `study/analysis/unwrap_explore_*.py`, `unwrap_combined_calibration.py` | 2026-09-28 estimator comparison, synthetic known-shift test, calibration |

## 2026-09-28: boundary bug and a better estimator

Exploration of all 724 folders (`study/analysis/unwrap_explore_{cache,eval,anchor,coverage}.py`,
logs `study/logs/unwrap_explore_*_20260928.*`). Reference = trigger-count head (243 folders, 210
with a valid ECG). It is a reference derived from the log, not an independent measurement; the only
fully independent check is continuity (images only), and that check is what exposed the bug.

**The boundary bug.** Buffer 1 starts 35-60 ms after the live loop's last trigger. When that gap
falls within the 39.4 +- 3 ms tolerance of `triggerlog._runs`, buffer 1's FIRST trigger is merged
into the buffer-3 run (seen directly: run end index == buffer-1 block start index). Then the run
length L is one too long, the trigger-count head is one slot too far, and `buffer_timing(3)` puts
every buffer-3 frame one frame (39 ms) late - which drags the buffer-1 method one slot off too (a
timing shift and a rotation trade off). Signature: gap `t1[0] - t3[-1]` ~ 0 ms instead of 35-60 ms.
**226 of 724 folders.** Evidence: where the wrap is visible (>300 ms phase jump), continuity put
the break exactly at the reference in 57 folders and one slot earlier in 22; 21 of those 22 have
the ~0 ms gap, 0 of the 57 do. In the -1 folders slot c-1 correlates normally with c (1.01 of
median) and the break is c-2 -> c-1 (-0.68) - an old frame, not a hybrid.
Of the 77 folders already unwrapped, 21 have the bug and 20 of those are now estimated one slot off.
Fix: end the buffer-3 run before the buffer-1 block's first trigger.

**Methods (reference corrected), exact on 210 folders:**

| method | exact | notes |
|---|---|---|
| continuity (low-pass log-envelope, de-meaned) | 73 % (92 % when the wrap jumps > 300 ms of phase, ~45 % below 200 ms) | images only; the 09-25 rejection (39 %) was mostly the bug. Works without ECG (58-67 %) |
| buffer-1 similarity (current) | 89 %; 98.7 % at margin >= 0.006 | margin separates well: bottom quintile 52 % exact |
| expected-motion match (buffer 3's own similarity matrix vs buffer 1's at the same phases) | 83 % | each buffer compared only with itself |
| anchor at the R-peak (single buffer-1 frame -> best buffer-3 slot -> head) | 27 % | one frame cannot localise: the corr peak over slots is flat (median 0.05-0.18 above the runner-up at any phase); votes over all frames 80 % |
| time adjacency (newest buffer-3 frame ~ buffer-1 frame 0, no ECG) | 33 % | confused with the same phase one beat earlier |
| **combined: z(buffer-1, per-frame normalised, circular phase) + z(expected motion) + 0.5 z(continuity)** | **98 %** (26 fr 99 %, 32 fr 98 %; the 4 misses are gross, not +-1) | weights fixed a priori; features chosen as the best of 4, so expect slightly less out of sample |

Why ambiguous folders were many: 32-frame campaign margins are lower (median 0.0096 vs 0.0117 on
the reference set) and continuity was not used. Current rule study-wide: trigger 243, buffer-1 353,
ambiguous 128 (97 low margin, 31 no valid ECG). The combined score resolves all but ~6 of the 450
ECG-valid folders without a trigger count at its 98 % point; for the 31 without ECG only continuity
applies (accept at high margin: its top quintile is 100 % exact).

**Fix (in `src`, 2026-09-28; data NOT yet re-unwrapped).** `triggerlog.parse_sequence` assigns every
trigger to its mode from the known sequence, parsed BACKWARDS from the unmistakable buffer-4 block
(n4 triggers at 1.08 ms): buffer 1 = the n1 triggers before it, the live loop = everything before
buffer 1 that keeps the loop cadence; after buffer 4: active SW, 4 triggers per push (reference,
push, tracking, B-mode). `buffer_timing` and `unwrap.trigger_count_head` use it; the old
period-only segmentation remains only as a fallback for logs that do not parse. All 724 logs parse
(`study/analysis/triggerlog_structure_check.py` -> `study/logs/triggerlog_structure_20260928.*`):
loop -> buffer 1 gap 22-61 ms everywhere, exactly 4 SW triggers per push everywhere, the loop start
(where kept) follows a >= 4.3 s pause. Verified against the old code on all 724: buffers 1 and 4
unchanged; buffer 3 one frame (39-40 ms) earlier in 227 folders; trigger count available in 244
(head -1 in 79 of them). Regression tests in `tests/test_unwrap.py`.

A second log bug surfaced: the µs timestamps are a 32-bit counter (wraps every 71.6 min) and
`read_log` sorted them, scrambling logs that span an overflow (C000000031 12-09-49 and the 3 other
folders whose SW block did not come out at 4 triggers per push). `read_log` now unwraps the counter
in log order and puts both columns on one clock before sorting.

**From `Claude/Buffer3_circular/Plan.txt`** (`study/analysis/unwrap_explore_plan.py`): its
synthetic test with a KNOWN shift - buffer 1 subsampled to the loop cadence and rotated, a true
ground truth independent of the trigger log - gives continuity 73 % exact, 95 % when the wrap jumps
> 300 ms of phase, ~35 % below 100 ms: the same as against the corrected trigger-count reference,
which supports that reference. Its identifiability point holds and is computable in advance: the
wrap's phase jump (n T mod RR) from the log says when an image-only method cannot work. Its
trajectory-prediction score (linear extrapolation across the boundary) is worse than plain
neighbour similarity (65 vs 73 %): cardiac motion is not linear over 40 ms.

**VERSION 2 applied to all 724 SW folders (2026-09-28, from Windows, 4 workers, 2 h 20 min, 0
failures; `study/logs/buffer3_unwrap_v2_20260928.csv`, summary `buffer3_unwrap_v2_report_20260928.log`,
runner `unwrap_v2_all_20260928.sh`):**

| result | 26 frames (C1-C10) | 32 frames (C11 on) | all |
|---|---|---|---|
| trigger count | 106 | 138 | 244 |
| combined | 14 | 408 | 422 |
| continuity only | 0 | 5 | 5 |
| **ambiguous** | 12 | 41 | **53** (VERSION 1 rule: 128) |

- Ambiguous: 28 without a trustworthy ECG, 25 below the combined margin (both also below the
  continuity threshold).
- Where the trigger count exists the combined estimator agrees in 208/211 (98.6 %; in-sample, the
  same set it was calibrated on) and continuity alone in 166/244.
- The 77 folders unwrapped by VERSION 1: 57 unchanged, 19 moved one slot, 1 two slots, none became
  ambiguous - as predicted from the boundary bug.
- **Playback check** (`study/analysis/unwrap_v2_playback_check.py`): the weakest frame-to-frame link
  INSIDE the playback (loop point excluded), relative to the median link. Resolved folders
  (671): sequences with a jump inside (link < 0) 450 -> 15; median weakest link -0.37 -> 0.48
  (ordinary cardiac motion); 538 smoother, 0 worse. Ambiguous (53): unchanged, 24 still jump.
- Montages: `study/montages/playback/` (1st and 2nd acquisition; buffer 1 as stored, buffer 3
  stored vs unwrapped).
- New beamforms unwrap with VERSION 2 automatically once the server clone is updated.

**Visual check with every possible head** (`study/analysis/all_heads_gifs.py` -> one GIF per head +
a grid, `study/montages/playback/<subject>_<folder>_all_heads/`). C000000029, 2nd acquisition:
only the chosen head (11) plays without a jump inside the sequence (weakest link +0.60 of the
median); all 31 others contain the stored 10 -> 11 link at -0.81. Its last 32 loop triggers are
exactly 39.4 ms apart (no dropped frames). A remaining visible jump at the GIF **loop point** is
not an unwrap error: 32 frames = 1.26 s is ~1.5 beats (RR 865 ms), so each loop steps the heart
back ~257 ms of its cycle; in montages, 26-frame tiles also hold their last frame for 6 steps.
