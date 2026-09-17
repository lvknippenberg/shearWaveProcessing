# Session log — 2026-09-11/14

Chronological record of what was done, decided, and got wrong. `README.md` in this folder is the
state; this is the history. Repo: `D:\Luuk van Knippenberg\Github\shearWaveProcessing`.

## Commits

| commit | what |
|---|---|
| `3ec3088` | base-config auto-selection + assertion; batch script + `--check`; real-time GIFs; buffer-5 frame-rate fix |
| `69f04ae` | passive: honour `directional: false`; widen the speed search to 20 m/s |
| `cdb02fa` | `pfield` enabled for the focused buffer only |
| `a4ab9bf` | scan-param append made robust to network-share locks; `verify_outputs.py` |
| `d27b502` | buffer-3 striation investigation documented; `incoherent_bmode.py` |
| `33d44c5` | fix options recorded in the striations doc |
| `7bf1c40` | `refocus_bmode.py` |
| `7959b83` | five-way reconstruction comparison; REFoCUS adjoint is best |
| `cc7edc0` | cine M-line selector; `draw_passive_mlines.py`; `gif_montage.py`; §3a periodicity correction |
| `9537e5f` | **GIF display: per-acquisition adaptive levels + gamma2** |
| `783d3f8` | M-line selector uses the same display as the GIFs |
| `7aeb1be` | base-config matching: phantom numFrames, v7 files, TX field mismatch |
| `06a204f` | **resolution phantom closes the striation investigation** |
| `4146d63` | in-vivo check: neither field-normalisation map helps buffer 3 |
| `8690017` | phantom: all 8 reconstructions + frame_montage.py |
| `87b47d5` | **scorecard: all 8 x both datasets; phantom flatters SVD inversions** |
| `f9ba889` | CenterTransmit.mat: beam vs region geometry (period argument later withdrawn) |
| `cc44d33` | **CORRECTION: ripple measured about the wrong centre, understated ~10x; + nearest-k ladder** |
| `320ea45` | two mechanisms separated: beam non-uniformity (minor here) vs interference (dominant) |
| `f83305c` | REFoCUS +6% compute; only buffer 3 affected (buffer 1 clean, REFoCUS null test) |
| `79ea567` | REFoCUS adjoint becomes the buffer-3 default; study/ moved into the repo |
| `f6d6742` | `--buffers` flag for the batch script |
| (fix) | **Refocus must run BEFORE Demodulate** — first version ran it on baseband IQ |

## What was done

1. **Whole study processed** — 44/44 folders, 0 failed, ~9.5 h. 1188 IQ + 1188 GIFs, 512 GB.
   Verified clean; 3 files repaired.
2. **Base config made safe.** It changed per acquisition campaign and the merge hardcoded the
   newest, which was wrong for **13 of 44** folders — silent corruption, since beamforming
   succeeds and the B-modes look plausible. Now auto-selected by matching `RcvBuffer` against
   `RF_frames`/`RF_rows`, and the merged file is asserted before it is ever read.
3. **Buffer-3 radial striations** investigated at length — see
   `docs/focused_bmode_striations.md`. Mechanism identified, REFoCUS adjoint found to fix it,
   swept across all 44 (median 13x reduction). **Not adopted** pending a resolution phantom.
4. **Buffer 1 checked** with the same method: genuinely clean; REFoCUS does *not* transfer
   (introduces an 8° artefact at 10x runtime).
5. **Display reworked** — per-acquisition adaptive levels + gamma2, now the pipeline default.

## Decisions, and why

* **`pfield` for buffer 3 only.** Focused: +3.5 dB at p90. Widebeam: *loses* field of view
  (56.6 → 50.2% above -50 dB). Diverging: unchanged. Blanket-enabling would trade a real
  buffer-1 loss for a gain only buffer 3 sees.
* **REFoCUS `adjoint`, not the SVD methods.** H is 73x80 — under-determined — so inverting
  harder amplifies noise: `tsvd` is *worse than doing nothing* (4.46% vs 3.77%) and 11x slower.
  The matched filter, which does not attempt a full inversion, wins.
* **Incoherent compounding rejected** despite the best-looking ripple metrics: it scores well
  only because it is washed out (dynamic range 18.7 dB). Caught by looking at the images.
* **Per-acquisition display, not per-set.** White points span 8x and ranges 50–61 dB across
  subjects; no single setting serves them. But aggregate the *range* per set when comparing
  subjects — per-subject ranges artificially boost the contrast of poor-window acquisitions.
* **+0 dB brightness** (user pick): higher gains mostly lifted noise.

## Mistakes made, and what fixed them

Recorded because each one changed a conclusion.

1. **"Buffer 1/3/4 all show the same radial structure, so it is not a focused artefact"** —
   wrong. The metric was single-frame and speckle-dominated. **Frame-averaging** (speckle falls
   as 1/√N, a fixed artefact does not) showed a 20x enrichment in the focused buffer alone.
2. **"The periodicity is exactly `rayDelta` (1.111°)"** — wrong. That came from summing power in
   a window around an assumed frequency. A **peak fit** puts it at 1.264°, the one-way beam
   width. The same error invented a "4° artefact" in buffer 1 that does not exist.
   → **Fit the peak; never sum a window around a frequency you assumed.**
3. **F-number 7.97** — wrong, it is 3.93. `Trans.ElementPos` is in **mm** while `TX.focus`/
   `PData` are in **wavelengths**; reading it as wavelengths halved the aperture. This inverted
   the conclusion: beam overlap is 1.12x, not 2.28x, which turned the argument *against* the
   hypothesis into the hypothesis.
4. **Saturation metric read 0.00% at every brightness** — gamma2 caps output at 0.97, so an
   8-bit `>250` test could never fire. Clipping happens at the normalisation, before the curve.
   A metric that reports "fine" regardless of input is worse than none.
5. **Backslashes in bash heredocs** collapsed repeatedly (`\r` became a carriage return, breaking
   a source file and this README's paths). There is a memory note about exactly this.
   → **Write files with the Write/Edit tool, never a heredoc, when they contain backslashes.**
6. **Killed a run and immediately restarted it** — the dying process still held a file handle and
   the new run died on a sharing violation. Check the process is gone first.

## Phantom outcome (2026-09-14)

The resolution phantom closed the striation investigation and **reversed two conclusions**:

* **REFoCUS degrades resolution** — 1.87 mm vs standard's 1.40 mm on 79 point targets (34%
  worse). The in-vivo number suggesting the opposite was an anatomy-scale correlation length, not
  a PSF. The caveat was stated when the number was first reported; the phantom confirms it
  mattered.
* **The residual survives envelope summing** (3.1x down, not 28x). The earlier figure came from a
  *fraction* of ripple power while the totals differed 20.6 / 17.5 / 8.1% — the third time in this
  session a normalised metric misled.

Normalising by a transmit-field map does not work: zea's `compute_pfield` returns magnitudes
only, so its compound is flat (no-op); a synthesised complex field has the right periodicity at
the 2nd harmonic (1.260 deg) but is anti-correlated with the image (-0.68, robust to mirroring
and delay-sign flips), because `|sum A|` is the point-scatterer sensitivity while the ripple is
measured in speckle, which follows the flat `sum|A|^2`.

**Verdict: keep standard.** ~2% angular ripple on an orientation buffer is not worth 34% of
lateral resolution.

## The pipeline-order bug (2026-09-14)

Worth recording because of how it passed review. `Refocus` declares `input_data_type=RAW_DATA` but
does not enforce it, so placing it after `Demodulate` produced a plausible image instead of an
error — and the verification I ran ("are the striations gone?") was satisfied by the wrong image,
because the wrong image is also smooth. A 44-folder batch completed and was reported as verified
before the equivalence check against the independent `scripts/refocus_bmode.py` caught it.

**Lesson:** "the artefact is gone" is not a correctness check when the failure mode also removes
the artefact. Compare against a known-good independent implementation. That check now reads
log-correlation 1.00000 on all 44.

## Open items

1. ~~Resolution phantom~~ **DONE (2026-09-14)** — both questions answered; see the section above
   and README §4a. The striation investigation is closed.
2. **Passive processing for all 44** — superseded by the 2026-09-17 session below: 36 folders
   processed with buffer-3 frame-0 lines, which turned out to be at the wrong cardiac phase; redraw
   on buffer 1 (`docs/passive_mlines.md`).
3. ~~GIF re-render~~ **DONE** — 44/44, exit 0.
4. **Buffer-1 REFoCUS** — dropped; the phantom shows REFoCUS costs 34% of lateral resolution even
   where it is well conditioned.
5. ~~Deconvolution lead~~ **CLOSED (same day)** — on the phantom it improved PSF
   1.40 → 1.07 mm and CNR 13.8 → 18.2 dB, but in vivo on C000000001 it imprints the map’s own
   1.260° texture (ripple 2.69 → 14.67%). Not adopted; see README §4a.

# Session log — 2026-09-17: passive M-lines, ECG timing, Linux server

Docs: `docs/passive_mlines.md`, `docs/ecg_timing.md`, `docs/linux_server.md`.

## What was done

1. **Passive study run, three times over.** Buffer-4 cine general lines (aborted: the septum moves too
   much for one line) -> buffer-4 stills (aborted: septum not visible in most folders) -> one line per
   folder on **buffer 3 frame 0**: 36 drawn, 8 skipped, all 36 processed unattended
   (`scripts/passive_study.py`, logs `passive_study_draw.log` / `passive_study_process.log`).
2. **Buffer 3 frame 0 is the wrong phase.** The user's valve frames (MVC 7/21, AVC 12) were 552 ms
   apart against an ECG RR of 675-740 ms. The trigger log showed buffer 3 is the tail of the live
   focused run (C000000001 frame 0 ~+200 ms after an R-peak) while buffer 4 starts exactly on an
   R-peak. Confirmed in `CombinedData.mat`: `SeqControl(15)` (pause for the ECG trigger) precedes only
   buffers 4, 2 and 6. `src/swp/acquisition/triggerlog.py`; buffer 4 at the R-peak in 44/44, buffer-1
   frame 0 anywhere (+10 to +1134 ms), an R-peak buffer-1 frame within ±5 ms in every folder.
3. **C000000001 redone on buffer 1**: R-peak-frame single line, then per-event lines on
   phase-matched frames (`draw-events`), then full/left/right halves
   (`study/analysis/passive_mline_split.py`). Montages gained a B-mode + M-line column.
4. **Automatic labels** MVC/AVC/AK/other/? for all windows (`passive_study.py label`,
   `study/logs/passive_window_labels.csv`), with an ECG plausibility check after finding logs that are
   not an ECG (C000000005/12: 240 ms periodic trigger) or have spurious triggers (C000000007/14).
5. **Linux server.** `scripts/linux_validation.py` (run one folder into `output_linux`, compare
   dataset by dataset). First server run: FAIL (buffer-4 displacement proxy corr 0.21). A Windows
   rerun on the same commit was bit-identical to the reference, so the server environment was the
   cause: its dev container mounted zea `44208e0b` (older fork `main`, lacking ~20 upstream commits incl.
   curved-probe support #516). New image + container from zea `8c2699fd`: torch bit-identical, JAX
   relRMS 4-7e-5, PASS. Runbook `docs/linux_server.md`; phase check `study/analysis/iq_phase_check.py`.

## Decisions, and why

* **One line per event, drawn on buffer 1 at the event's phase.** The image is good enough to see the
  septum, and phase matching keeps the anatomy consistent with the buffer-4 data being sampled.
* **Label rather than drop the AK window.** Its burst energy equals an MVC's, so it is real motion; it
  just has no measurable wave on any segment.
* **Rebuild the image rather than patch the container.** The Dockerfile installs from zea's `uv.lock`, so
  the image at `8c2699fd` gets exactly that commit's dependencies (it also needs `hdf5plugin`,
  `numcodecs`, `fsspec[http]`, `tyro>=1`). A standalone `docker run` container is not stopped by VS Code
  (`shutdownAction: stopContainer` in the dev container).

## Mistakes made, and what fixed them

1. **"The acquisition is R-peak gated, so frame 0 of every buffer matches"** — wrong for buffers 1 and
   3. Caught by the user's valve frames not fitting the ECG RR; settled from the trigger log and the
   `SeqControl` table rather than assumed.
2. **Blaming JAX for the server mismatch** was the first hypothesis; the Windows same-commit control
   run is what showed the environment (zea version) was the cause. Control runs before conclusions.
3. **The comparison report said FAIL on a passing run** — a description tag (`[delay-and-sum]`) and
   intentionally skipped converted files were counted as differences. Fixed in `linux_validation.py`
   (`note`, skip converted when not written).
4. **Heredocs with backslashes, again** (three times): `\n` in f-strings became real newlines and broke
   `passive.py` / `passive_study.py` until caught by `py_compile`. Edit/Write tools only for code with
   escapes.
5. **Speed search bounds read as results**: 20 % of the automatic study speeds sit at 1.0 or 20 m/s.
   They mean "no front found", not a measurement.
