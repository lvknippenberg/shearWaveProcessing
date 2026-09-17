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
unsorted). Consecutive frame triggers at a buffer's frame period form that buffer's block:
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

## ECG quality problems

Checked around buffer 4 (±5 s):

| folder | problem | effect |
|---|---|---|
| C000000005, C000000012 | "R-peaks" every 240 ms (250 bpm), perfectly regular - a periodic trigger, not an ECG | gating of buffers 2/4/6 is arbitrary; labels `?` |
| C000000007, C000000014 | spurious extra triggers (230-260 ms intervals among normal beats) inside the buffer-4 beat | labels `?` |
| C000000010, C000000017, C000000021, C000000030, C000000039, C000000041 | a missed or extra trigger elsewhere in the log | filtered out; labels usable |

`clean_r_peaks()` takes the median of plausible intervals (400-1600 ms) as reference RR (none if fewer
than 3 or less than half the intervals are plausible) and drops triggers arriving < 0.6 x RR after
the last kept one.

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

- `Vera_trigger` is interpreted as one pulse per stored frame and buffer 3 as the **last** 26 frames
  of its live run; both fit every folder (block lengths match `RF_frames`) but are inferences.
- Timestamps are whole ms (early format), so frame times carry ~±1 ms.
- QS2 is a population regression; the ±120 ms tolerance is deliberately wide. Labels are a
  screening aid, not a replacement for looking at the valves.
