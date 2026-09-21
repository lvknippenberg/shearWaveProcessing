# ECG gating, buffer timing and cardiac event labels

What is R-peak gated in the S5-1 SWI sequence, how to place any buffer frame on the cardiac cycle
from the trigger log, and how passive burst windows are labelled MVC / AVC / AK. Established
2026-09-17 on all 44 folders of `Z:\raw_data`. Code: `src/swp/acquisition/triggerlog.py`.

## What is gated (from `CombinedData.mat`)

`SeqControl(15)` is `pause` (argument 17), i.e. wait for the external ECG trigger. Exactly three
events carry it, each directly before an acquisition block:

| event | `Event.info` | acquires |
|---|---|---|
| 7726 | "Wait for trigger input diverging waves" | **buffer 4** (passive source) |
| 12358 | "Wait for trigger input SW" | **buffer 2** (active SWE; buffer 5 is interleaved, one frame per push) |
| 15327 | "Wait for trigger input strain" | **buffer 6** |

**Buffers 1 (widebeam) and 3 (focused) are not gated.** `SW.WaitForRpeak = 1` in these acquisitions.

The sequence order is: focused live imaging (orientation; buffer 3 keeps its last 26 frames) ->
widebeam B-mode (buffer 1) -> **R-peak** -> diverging waves (buffer 4) -> **R-peak** -> active SW
(buffers 2 + 5) -> **R-peak** -> strain (buffer 6).

## The trigger log

`AcquisitionParametersAndECG.mat` -> `ECG_data_raw`: a text dump of four columns written one after
another. Two formats occur:

| acquisitions | headers | unit |
|---|---|---|
| C000000001-C000000004 | `Time`, `Signal`, `ECG_trigger`, `Vera_trigger` | ms |
| C000000005 onward | `Time_us`, `Signal_V`, `ECG_trigger_us`, `Vera_trigger_us` | µs |

`ECG_trigger` = R-peak times; `Vera_trigger` = one timestamp per Verasonics frame (a circular log,
unsorted). **`ECG_trigger` is the hardware R-peak detection the acquisition itself triggers on.**
The pipeline reads it directly and never re-detects R-peaks, so event phases are measured against
exactly the timing the sequence used. Before trying to improve on it, read *Do not re-detect
R-peaks from the `Signal` trace* below. Consecutive frame triggers at a buffer's frame period form
that buffer's block:
buffer 4 = 926 triggers ~1.08 ms apart; buffer 1 = 90 triggers ~11.34 ms apart; buffer 3 = the last
26 of a long ~39.4 ms run (live imaging). `buffer_timing(folder, buffer)` finds the block and gives
every frame's time and phase (ms since the preceding R-peak).

Frame periods check out against the sequence: buffer 3 = 73 transmits x 2 (pulse inversion) x 270 µs
= 39.42 ms (`Bmode_FC.ActualFPS` 25.37); buffer 4 = 4 x 270 µs = 1.08 ms.

## Timing across the study (all 44 folders)

- **Buffer 4 frame 0 is exactly on an R-peak in 44/44** (offset 0 ms).
- **Buffer 1 frame 0 falls anywhere in the cycle** (+10 to +1134 ms after an R-peak), but buffer 1
  spans more than a beat, so **every folder has a buffer-1 frame within ±5 ms of an R-peak**
  (`BufferTiming.nearest_rpeak_frame()`; C000000001: frame 71, -4 ms).
- Buffer 3 frame 0 is likewise at an arbitrary phase (C000000001: ~+200 ms, mid-systole).

Consequence: an M-line drawn on buffer 3 or buffer 1 **frame 0** shows the anatomy at a different
cardiac phase than buffer-4 frame 0. Use the R-peak frame (single line) or a phase-matched frame per
event (see `docs/passive_mlines.md`).

## Do not re-detect R-peaks from the `Signal` trace

**Proposed, tested and withdrawn twice - 2026-09-18 and again 2026-09-21. Read this section before
touching the `Signal` trace again.**

The logged `Signal` waveform is a **secondary, lossy display stream** and is not a reference for
timing. In C000000023 it is sampled at 100 Hz, takes only 51 distinct values, and **61 % of its
samples are exactly zero**.

### Why, from the acquisition hardware

`SWI/Arduino/ArduinoReadECG_request_micros.ino` records the two channels by completely different
routes, and only one of them is a timing instrument:

| | `ECG_trigger` | `Signal` |
|---|---|---|
| source | hardware interrupt, pin 2 (`ISR_ECG`, RISING) | `analogRead(A0)` inside `loop()` |
| rate | every R-peak, as it happens | **100 Hz** (`sampling_frequency`), 10 ms per sample |
| signal path | the monitor's **trigger output** | the monitor's **analog output** |

The analog output carries the monitor's own display filtering, so it lags the trigger output by an
unknown, device- and setting-dependent amount, and it is quantised to 10 ms on top of that.
Comparing the two measures **monitor latency**, not trigger accuracy.

### What the second attempt found, and why it was wrong

On 2026-09-21 the waveform was re-detected across all 44 folders and appeared to show a cleanly
bimodal trigger offset: 15 folders on the R-peak, 20 folders about 55 ms early, with an *empty gap*
between the groups. This was reported as "only 10 of 44 acquisitions have a trustworthy trigger".
**It is an artefact** - the bimodality is consistent with two monitor or filter settings in use
across the study, i.e. two analog-path delays, not a good and a bad detector. Two further mistakes
compounded it: the delay was quoted against the *previous* R-peak (797 ms in C000000023) when the
*next* one was 83 ms away, which makes correct gating look broken; and buffer-4's start was called
"83 ms early" when it simply sits on whichever fiducial that monitor emitted.

Measured on the triggers alone, **the passive block starts within 6 microseconds of a recorded
R-peak in every folder of the study**. There is no gating problem.

Checked against it (2026-09-18), of the 12 `ECG_trigger` R-peaks that fall inside the logged trace,
10 land on a clear deflection (amplitude 0.21-0.48) and **2 land on a sample of exactly zero with
no deflection at all within ±30 ms** - one of which is a beat that plays no part in triggering. The
gaps therefore belong to the trace, not to the detector.

This matters because the trace can appear to contradict the trigger by tens of milliseconds. In
C000000023 the nearest tall deflection to buffer-4 frame 0 is 83 ms later, which looks like an
early trigger and is not: the same thing happens on a non-triggering beat. **A "correction" derived
this way would shift every event phase in the recording and turn an MVC into an atrial kick.** It
was proposed and then withdrawn during the 2026-09-18 session; see `study/SESSION_LOG.md`.

Use `ECG_trigger`. If better timing is genuinely needed, record a higher-quality ECG rather than
post-processing this trace.

## ECG quality problems

Checked around buffer 4 (±5 s):

| folder | problem | effect |
|---|---|---|
| C000000005, C000000012 | "R-peaks" every 240 ms (250 bpm), perfectly regular - a periodic trigger, not an ECG | gating of buffers 2/4/6 is arbitrary; labels `?` |
| C000000007, C000000014, C000000017 | a fixed-rate pulse train mixed with real beats (17-39 of 49 intervals at 246-280 ms) | labels `?`; `rrcheck` status `unusable` |
| C000000010, C000000017, C000000021, C000000030, C000000039, C000000041 | a missed or extra trigger elsewhere in the log | filtered out; labels usable |

`clean_r_peaks()` takes the median of plausible intervals (400-1600 ms) as reference RR (none if fewer
than 3 or less than half the intervals are plausible) and drops triggers arriving < 0.6 x RR after
the last kept one.

## Automatic RR assessment (runs on every passive workflow)

`swp.acquisition.rrcheck.assess_rr` assesses the trigger record whenever ECG data is present.
`swp.passive.check_ecg_quality` calls it at the start of `process_passive`, prints the result, and
writes it to `<outdir>/ecg_rr_check.json` so a batch run can be audited afterwards. The batch view
over a whole tree is `python study/analysis/ecg_check_study.py` -> `study/logs/ecg_check.csv`.

The principle: **a single wrong R-peak is correctable, uncertainty about the R-peaks is not.**
Dropping one spurious trigger out of ~50 leaves the reference RR and every event phase intact, so
that is reported and not warned about. Anything that leaves the R-peaks themselves in doubt warns,
because a phase error relabels an MVC as an AK and silently falsifies the speed analysis.

| status | meaning | `trustworthy` |
|---|---|---|
| `ok` | no short/long intervals, CV <= 0.15, no outliers | yes |
| `corrected` | exactly one spurious trigger discarded by `clean_r_peaks` | yes |
| `warn` | >1 trigger discarded, a missed beat, CV > 0.15, or an RR >30 % off the median | **no** |
| `unusable` | fixed-rate pulse train (>30 % of intervals < 400 ms) or < 3 plausible intervals | **no** |
| `no-ecg` | no trigger log at all | **no** |

Conditions are evaluated independently and the worst one sets the status - a record whose single
spurious trigger was correctable can still warn on high variability.

Across the 44 beamformed folders (2026-09-21): **24 `ok`, 15 `warn`, 5 `unusable`**; cardiac
phases are trustworthy in 24/44. Gating is exact everywhere (max 6 us). The `unusable` five are
C000000005, C000000007, C000000012, C000000014, C000000017 - all fixed-rate or near-empty trigger
records.

## Event labels

`label_event()` / `label_buffer4_events()` label an event from its time since the preceding R-peak
(`phase`), time to the next R-peak and HR (from the reference RR). First match wins:

| label | rule |
|---|---|
| **MVC** | phase <= 150 ms |
| **AK** (atrial kick / atrial contraction) | <= 200 ms before the next R-peak (estimated from RR if not logged) |
| **AVC** | within ±120 ms of QS2 = 546 - 2.1·HR ms (Weissler; from Q, ~R) |
| other | none of the above (mostly mid-diastole) |
| `?` | no usable ECG, or the beat containing the event is outside 0.75-1.35 x reference RR |

Applied to the 36 processed folders (`python scripts/passive_study.py label --root Z:/raw_data`,
table in `study/logs/passive_window_labels.csv`): **MVC 47, AVC 31, AK 24, other 16, ? 15.** Pattern is
consistent: MVC at 33-143 ms, AVC at 340-530 ms, AK shortly before the next R. C000000022 has two
windows labelled AVC (350 and 501 ms), both inside the tolerance - check by eye.

C000000001 (RR ~700 ms, R-peaks at 0 and 695 ms in buffer-4 time): win0 50 ms **MVC**, win1 441 ms
**AVC**, win2 610 ms **AK** (85 ms before the next R), win3 782 ms **MVC** (+87 ms, next beat).

Labels are stored in `passive_windows.json` (`windows[i].label` + `window_phases` with the timing
behind each), in every row of `passive_speeds.json`, and in the montage titles.

## Caveats

- **Detection is an energy ranking, not a physiological search.** `detect_line_bursts` keeps the
  four largest bursts along the M-line and only then labels them. Nothing prevents a large
  non-valvular event (rapid filling, respiratory motion) from displacing a genuine valve closure
  out of the list, and nothing guarantees that both MVC and AVC are among the four. C000000023
  happens to catch both; that is not by construction. Making the detector phase-aware - searching
  the expected MVC / AVC windows rather than ranking on energy - would remove the risk.
- `Vera_trigger` is interpreted as one pulse per stored frame and buffer 3 as the **last** 26 frames
  of its live run; both fit every folder (block lengths match `RF_frames`) but are inferences.
- Timestamps are whole ms (early format), so frame times carry ~±1 ms.
- QS2 is a population regression; the ±120 ms tolerance is deliberately wide. Labels are a
  screening aid, not a replacement for looking at the valves.
