# Does an MVC event need its own M-line? (2026-10-02)

The general M-line is drawn on the buffer-4 frame nearest an R-peak, and the MVC follows the
R-peak by only a few tens of ms. Two questions:

1. Is a drawing prompt for the MVC line worth it?
2. Does the MVC of the *other* beat (the recording holds about 1.5 beats, so usually two MVCs) need
   its own line? Probe or breathing motion between the beats would certainly need one.

**Implemented:** an MVC window at most R+50 ms reuses the general line without a prompt. The
buffer-4 anatomy must have moved at most 1 mm across the line (and 3 mm in total) since the
general line's frame. Otherwise the prompt appears. See "The rule" at the end of this page and
[passive_manual.md](passive_manual.md) step 4.

Data: all folders of the manual study on Z:\raw_data (43 folders, 77 MVC windows, current reviewed
windows). Scripts and tables:

| script (study/analysis/) | table (study/logs/) | what |
|---|---|---|
| `passive_event_vs_general.py` | `passive_event_vs_general.csv` | drawn MVC / AVC lines vs the general line, and MVC vs MVC in one recording |
| `passive_event_vs_general_st.py` | `passive_event_vs_general_st.csv` | the five space-times on both lines, speed sensitivity |
| `passive_mvc_beat_motion.py` | `passive_mvc_beat_motion.csv` | anatomy motion in buffer 4: general frame → each MVC, MVC beat 1 → beat 2 |

## 1. The drawn lines

**Timing.** MVC is at R+36 ms median (IQR 30-46). 69 of 72 MVCs are before R+80 ms. The
general-line frame is in the same beat for 38 MVCs and in the other beat for 34 (the frame
nearest an R-peak can be at either end of the ~1 s recording).

**How often the line was changed.** 40 MVC prompts opened with the general line pre-loaded, and
**33 (83 %) were accepted unchanged**. That share is the same at every phase (R+0-30: 16/19,
R+30-50: 12/15, R+50-80: 3/4). The other 37 opened with an old September line drawn
independently on buffer 1.

**Distances** (mm, median [IQR]):

| | mean distance | midpoint shift | angle |
|---|---|---|---|
| MVC line vs general line, all changed (44) | 1.3 [0.8-1.8] | 2.0 | 2.2° |
| MVC line vs general line, edited from it (7) | 1.4 [0.7-1.8] | 2.2 | 1.9° |
| two MVC lines of one recording, both drawn independently (10) | 0.5 | 1.2 | 0.9° |
| two MVC lines of one recording, all non-trivial pairs (25) | 0.7 [0.5-1.7] | 1.6 | 2.2° |
| AVC line vs general line (44), for scale | 3.5 [2.0-5.0] | 6.5 | 11.2° |

A 2° angle changes a speed by 0.06 % (cos).

**Does it change the space-time?** For the 44 changed MVC lines with a hand slope, the five
space-times were recomputed on the general line, using the worker's code. The event-line
space-times reproduce the stored ones with r = 1.000.

- Correlation of the two space-times on common ground: velocity **0.97** median (IQR
  0.93-0.99), displacement **0.99**. The 7 cases below 0.9 are the lines 2-3.6 mm away
  (Spearman rho distance vs correlation = -0.77).
- Best speed through the reader's own anchor point: median ratio general/event 1.00, |change|
  median 9 %. The same estimator differs from the hand speed itself by 30 %.
- Tracking score of the reader's line: median ratio 1.00, |change| 5 %.
- Hand speeds of the two MVC beats of one recording: 13 % apart with the identical line (n = 8),
  18 % with different lines (n = 15). The difference is not significant (Mann-Whitney p = 0.44).

## 2. The other beat, and probe motion

The buffer-4 envelope (9-frame average, as in the editor) at the general frame is registered
onto the one at each MVC. The registration is the line transfer's rigid one, in a box around the
general line, with an ensemble of box sizes and a known-shift check. This measures the motion in
the images, independent of the reader. The shift is split into the part **across** the
general line, which takes a line off the septum, and the part **along** it, which only slides
the line along the septum. In PLAX the septum is near-horizontal, and much of the motion is
lateral, along the line.

Shift in mm, median [IQR], 90th percentile (trusted registrations; 100 % same beat, 94 % other):

| | n | across the line | along the line | total |
|---|---|---|---|---|
| general frame → MVC, same beat | 38 | **0.32** [0.24-0.58], p90 0.83 | 0.50, p90 1.70 | 0.86, p90 1.77 |
| general frame → MVC, other beat | 32 | **0.44** [0.28-1.47], p90 2.27 | 1.06, p90 2.51 | 1.70, p90 3.10 |
| MVC beat 1 → MVC beat 2 (same phase) | 32 | 0.24 [0.12-0.88], p90 1.95 | 0.24, p90 1.37 | 0.51, p90 2.63 |

- **Same beat: the anatomy stays put.** 29 of 30 MVCs at most R+50 ms move ≤ 1 mm across the
  line.
- **Other beat: usually the same, with a tail.** The median is similar, but about 1 in 3 move
  1-2.3 mm across the line. That is breathing or probe motion between the beats. Two
  registrations failed outright (C13 10-11-11: an 11.6 mm lateral jump, corr0 0.26; C7: no ECG
  phase). They are probe-motion recordings and need their own line.
- **The reader follows the motion across the line, not the total.** The distance of the drawn
  MVC line to the general line rises with the shift across the line (Spearman rho 0.40,
  p = 0.01). It does not rise with the total shift (rho 0.23, p = 0.13). The 33 lines accepted
  unchanged had a median of 0.38 mm across the line.
- **MVC beat 1 vs beat 2** (both drawn independently): 0.5 mm, 0.9° apart, as close as redrawing
  one event twice. So the second MVC does not need its own line unless the images moved.

## The rule (configs/passive_manual.yaml `events.reuse_general`)

An event window takes the general line **without a prompt** when all of these hold:

- its label is MVC;
- it is at most **R+50 ms** (`max_phase_ms`). Above that, the drawn lines start to differ: R+50-80
  changed-line median 1.5 mm, R+80+ 3.2 mm;
- the registration general frame → event frame is **trusted**, and the anatomy moved at most
  **1.0 mm across** the line (`max_perp_mm`) and at most **3.0 mm in total** (`max_shift_mm`);
- the window has no answer yet. `--redo event --window i` always opens the editor, and
  `session --no-reuse` switches the rule off.

The same check covers both the same beat and the other beat, so a beat with probe or breathing
motion still gets a prompt. On the 77 MVC windows analysed here:

- **45 (58 %) would have been reused**: 29 of 38 same-beat and 16 of 34 other-beat MVCs.
- The rest would be prompted: 15 above R+50 ms, 10 that moved > 1 mm across the line, 1 not
  trusted, 1 over 3 mm in total, and 5 without an ECG phase.
- At the prompted windows (phase ≤ 50 ms), you had changed 9 of 12 lines (median 1.5 mm). At
  the reusable windows, the lines that were changed moved a median 1.0 mm, mostly old September
  pre-loads.

The check costs 1-6 s per window (two buffer-4 reads and the registration). It runs in the
session's background prefetch. `events.json` keeps the registration (`auto_reuse`: phase,
shift across the line and in total, dx / dz, frames, checks). A prompted MVC keeps the failed check (`reuse_check`),
and the export gains `line_reused_general`.

AVC stays a prompt: its line lies 3.5 mm and 11° from the general line.
