# SWE study processing — 2026-09-11/14

> Chronology, decisions and the mistakes that changed conclusions: **SESSION_LOG.md**.

Outputs from processing the whole `Z:\raw_data` in-vivo SWE study (44 measurement folders), the
buffer-3 radial-striation investigation, and the reconstruction comparisons.

* **Code**: `D:\Luuk van Knippenberg\Github\shearWaveProcessing` (pushed through `f83305c`)
* **Processed data**: stays next to the raw data in `Z:\raw_data\C*\*\output\`
* **This folder**: deliverables, figures, logs, one-off analysis scripts


## 0. Buffer 3 default changed — REFoCUS adjoint (2026-09-14) — **APPLIED to all 44**

`BufferSpec(3).refocus = "adjoint"`. `run.py beamform` writes the focused buffer with
retrospective transmit beamforming instead of plain DAS, and **all 44 folders have been
re-run** (`study/logs/rerun_buffer3_refocus.log`, 44 ok / 0 failed, ~1.2 min each).

| | ripple @ transmit pitch (C000000001) | lateral −6 dB (phantom) | beamform time |
|---|---|---|---|
| plain DAS (old default) | 15.55% | 1.40 mm | 47.9 s / 26 frames |
| **REFoCUS adjoint (new)** | **3.82%** | 1.87 mm (+34%) | 50.8 s (**+6%**) |

Across the study, apex-referenced ripple at the 1.111° transmit pitch is now **median 5.72%
(min 2.71%, max 15.42%)**. The spread is real and subject-dependent — REFoCUS helps a lot on some
acquisitions and barely on others — so C000000001's 3.82% is better than typical, not
representative. Anything quoted from C1 alone should be read that way.

Every file records its reconstruction in the HDF5 root `description` attribute
(`… [REFoCUS adjoint]`); all 44 verified. The filename does not carry it.

Only buffer 3 — buffer 1's widebeams show no reduction under the same test, and buffers 2/4 feed
the displacement estimators, whose phase must not be touched (§5g of the striations doc).

### The first attempt was wrong — what to check if this pipeline changes again

`Refocus` declares `input_data_type=RAW_DATA`: it inverts the transmit encoding from the transmit
delays and must see **raw RF, before `Demodulate`**. The first implementation put it *after*
`Demodulate`, so it ran on baseband IQ. That does **not** raise — it produces a plausible image
(envelope scaled 15×, log-correlation 0.62 against the correct one), and it passed both a shape
smoke test and a full 44-folder batch whose verification only asked "is the artefact gone". It
was: the wrong image is also smooth.

**The check that catches it** is equivalence against `scripts/refocus_bmode.py`, which is
independent and known good. Correct pipeline vs reference, all 44 folders: log-correlation
**1.00000** (min and median), envelope ratio 1.0000. Run that comparison after any change to
`build_beamform_pipeline`.

## 1. Whole-study processing — DONE

44/44 folders converted → beamformed → real-time GIFs. **0 failed.** ~9.5 h, 8–15 min/folder.

| | |
|---|---|
| converted zea RF | 220 files |
| IQ files | 1188 |
| GIFs | 1188 |
| on disk | 512 GB (on Z:) |
| verification | **44/44 clean** (`verify_outputs.py --repair` fixed 3 files) |

Base configs auto-selected and asserted per folder: **9 × P1-6, 4 × P11-14, 31 × P15-xx**,
0 mismatches. Under the old hardcoded default, **13 of 44** would have been silently mis-sliced.

Reproduce / extend:

```
python scripts/process_raw_data.py --root "Z:\raw_data" --check   # audit, seconds
python scripts/process_raw_data.py --root "Z:\raw_data"           # process, skips done
python scripts/verify_outputs.py   --root "Z:\raw_data" --repair  # check + fix
```

## 2. Buffer-3 radial striations — CLOSED, no fix adopted

Full record: `shearWaveProcessing/docs/focused_bmode_striations.md`.

**Ruled out** (each with the measurement): beam gaps; per-line transmit gain (`sum|Apod|` std
0.000%); delay quantisation (1.82° of phase); "too focused" (it is F/3.93); that summing all 73
transmits is the problem (using *fewer* is monotonically worse); `pfield`.

**Measured mechanism:** an isolated spectral peak at **1.264°** carrying 17.3% of ripple power,
focused-buffer-only, concentrated at the focal depth (45–85 mm) and gone by 105–145 mm. Note it
sits at the **one-way −6 dB beam width (1.247°)**, *not* the 1.111° line spacing.

**Reconstruction comparison** (same RF, 26 frames):

| reconstruction | ripple @ spacing | lat. corr. | dyn. range | time |
|---|---|---|---|---|
| coherent all-73 (current) | 3.77% | 24.45 mm | 32.9 dB | 62 s |
| incoherent (envelope) | 0.31% | 38.64 mm | **18.7 dB** | 155 s |
| **REFoCUS adjoint** | **0.16%** | **21.29 mm** | **33.7 dB** | **69 s** |
| REFoCUS tikhonov | 1.60% | 24.45 mm | 28.7 dB | 793 s |
| REFoCUS tsvd | 4.46% | 26.03 mm | 21.1 dB | 788 s |

**REFoCUS adjoint wins on every axis _in this in-vivo table_** at the same cost as the normal
beamform — but every column above is a surrogate, and the phantom PSF measurement (§4a) disagrees.
Incoherent compounding scores well only because it is washed out. The SVD inversions are worse
and 11× slower, as the under-determined 73×80 encoding predicts.

**NOT ADOPTED — and the resolution phantom (2026-06-18) reversed the case for it.** On 79 point
targets REFoCUS measures **1.87 mm lateral vs standard's 1.40 mm (34% WORSE)**. The in-vivo
lateral-correlation figure that suggested it *improved* resolution was an anatomy-scale proxy,
not a PSF. Full results: `docs/focused_bmode_striations.md` §5a. **Investigation CLOSED: keep
standard.**

## 3. Buffer 1 — clean, and REFoCUS does *not* transfer

Buffer 1 is 21 widebeam transmits, ±40° in **4.0°** steps. It is **2.9× under-sampled** by the
λ/D coherent-compounding criterion (far worse than buffer 3's 0.79×) yet has **no distinct
spectral peak** at its own 4.0° spacing — only broad 18° shading. Its 9.4° diverging opening
angle gives 2.36× beam overlap, which is what protects it. So overlap, not λ/D, predicts the
artefact.

REFoCUS on buffer 1 (H is 21×80, far more under-determined):

| | peaks | lat. corr. | dyn. range | time |
|---|---|---|---|---|
| coherent 21-tx | 18.0° | 27.60 mm | 31.4 dB | 70 s |
| REFoCUS adjoint | 18.0°, **8.00° (16%)** | 24.45 mm | 35.9 dB | **746 s** |

Better resolution and +4.5 dB dynamic range, but it **introduces** an 8.00° artefact —
exactly 2× the transmit step, the signature of a poorly-conditioned decode — at **10× runtime**.
**Not recommended for buffer 1.** If revisited, tuned `tikhonov` regularisation is the lever.

### But it was not clean of CLUTTER — and that is now fixed (2026-09-21)

"Clean" above means *free of transmit-lattice artefacts*, and that still holds. It says nothing
about off-axis clutter, which is what buffer 1 actually suffers from. Each of the 21 widebeams
opens only ~9.3–11.6° from a virtual source 123 mm behind the array, so at 100 mm depth about
**5 of the 21 transmits insonify a given pixel** — and the beamformer compounded all 21. In vivo
the other 16 contribute as much amplitude as the 5 that did reach it; on the resolution phantom,
identical geometry but no reverberating chest wall, they are 18 dB down.

`BufferSpec.tx_window = ("rect", 1.0)` now restricts each transmit to its own cone:
**+2.6 to +5.9 dB dynamic range, +0.4 to +1.4 dB dark-region contrast**, for +0.49 % lateral
−6 dB width and +12 % beamforming time. Applied to **buffers 1 and 5** — both use the
`Bmode_WB` widebeam transmit, and all 44 buffer-5 files were scanned and confirmed identical
(21 tx, ±40°, −123.2 mm) before adoption. All 44 folders re-run for both buffers.

Note this is the *opposite* conclusion to buffer 3, where all 73 focused beams do overlap each
pixel and using fewer blurs the image — which is why it is a per-buffer field, not a global
switch. Full investigation, including the two metric bugs that nearly reversed the answer:
`docs/widebeam_bmode_reconstruction.md`.

## 4. Montages — `study/montages/`

Four families, each answering a different question. **Use `*_best.gif` for looking at data.**

> **Buffer-1 montages come in two reconstructions now (2026-09-21).**
> `all_buffer1_txwin_best.gif` is the **current** one — the 44 folders rebuilt with
> `BufferSpec.tx_window = ("rect", 1.0)`. It is deliberately the **same 1360x876 geometry and
> 60 frames** as the all-21 montages (same `--title` strip), so the two can be flipped or
> overlaid directly. Every `all_buffer1*` file WITHOUT `txwin` is the
> older all-21-transmit compound, kept deliberately as the reference for what changed. They are
> the same 44 subjects in the same 8x6 order, so the two play tile-for-tile against each other.
> Do **not** put the two on a shared-reference montage: the cone sums ~5 transmits instead of 21,
> so its absolute level is legitimately lower and a shared scale would read that as a quality
> difference (the same trap as REFoCUS, flagged in `shared_norm_montage.py`).

| family | normalisation | use for |
|---|---|---|
| **`*_best.gif`** | per-acquisition levels + **gamma2** | **best visibility — the default choice** |
| `*_adaptive.gif` | one scale per buffer set | fair exposure, subjects comparable within a set |
| `*_shared.gif` | one absolute scale for everything | the only set where brightness has cross-subject *and* cross-buffer meaning |
| `all_buffer1.gif`, `*_standard.gif`, `*_refocus.gif` | per-clip max (original pipeline) | the renders as first delivered |

Sets: `all_buffer1*`, `all_buffer3_standard*`, `all_buffer3_refocus*` — all 44 subjects, 8x6,
real time, loop-synchronised (shorter clips black-padded so every tile restarts together).
Also `recon_compare.gif` (buffer 3: standard / incoherent / REFoCUS) and
`buffer1_refocus_compare.gif`.

**Incoherent-sum coverage** (checked 2026-09-14): before today the only incoherent reconstruction
anywhere was buffer 3 of C000000001 (a tile inside `recon_compare.gif`) plus the phantom. Buffers
1 and 4 of C000000001 were added; `incoherent_allbuffers.gif` now shows all three B-mode buffers,
standard on the top row and incoherent below.
There is deliberately **no all-44 incoherent set**: it would take ~7 h and the phantom measures
incoherent compounding at a 7.48 mm lateral PSF against standard's 1.40 mm, so it is not a
reconstruction anyone should image with — it exists to isolate the interference term, nothing more.

`c1_field_correction.gif` — C1 buffer 3: standard / ÷ pfield compound / ÷ harmonic field at
γ 0.5 and 1.0. The two right-hand tiles are visibly cross-hatched; that is the map stamping its
own 1.260° periodicity on the image, and it is why the deconvolution lead was dropped (§4a).

`c1_buffer3_methods_frame18.png` — **all eight buffer-3 reconstructions frozen on one frame**,
4x2, full tile resolution. The best single view of the striation question: a fine radial pattern
is far easier to judge still than tracked through a moving loop. All eight GIFs are natively
529x382, so nothing is rescaled — the incoherent tile only looks zoomed because its wide PSF
smears energy past the sector edge. Rebuild with `study/analysis/frame_montage.py`.

`phantom_buffer3_methods_frame1.png` — **the same 4x2 on the resolution phantom**, same order,
same tile size, so the two files can be flipped between. Frame 1 because the phantom clip is 2
frames (a static target). Generating it required filling in the 5 reconstructions the phantom
lacked: REFoCUS tikhonov/tsvd and the three field-normalised variants. Here the wire targets make
the trade visible rather than inferred — the incoherent tile smears each wire into an arc, and the
÷ harmonic tiles keep the wires sharp while cross-hatching the background, which is exactly the
split the in-vivo images could not show.

* `curves/` — buffer 1 under all six tone curves + `curve_comparison.png` (3 subjects x 6 curves)
* `brightness/` — buffer 1 at +0/+3/+6/+9/+12 dB. **+0 dB chosen**; above it the extra brightness
  mostly lifts noise. Clipping stays under 1% even at +12 dB.

Measured level offsets, for the record (the `*_best` renders deliberately discard these):
buffer 3 standard is **-23.9 dB** vs buffer 1; REFoCUS is **+18.5 dB** vs standard, which is
`H^H` ramp decode gain, **not** image quality.

Build more with `scripts/gif_montage.py` (tile existing GIFs) or
`scripts/shared_norm_montage.py` (re-render from IQ with explicit levels/curve/gain; caches
downscaled tiles in `study/analysis/tilecache/` so re-rendering is seconds).

## 4a. Resolution phantom — investigation CLOSED

`D:\swp_res\Resolution phantom\DefaultPatient_SW_data_18-June-2026_13-52-51` (79 point targets,
30–105 mm; focused sequence identical to in-vivo on all 7 parameters).

| reconstruction | lateral −6 dB | axial | CNR | ripple |
|---|---|---|---|---|
| **standard** | **1.40 mm** | **0.97 mm** | **13.8 dB** | 2.02% |
| REFoCUS adjoint | 1.87 mm (+34%) | 1.03 mm | 12.9 dB | 0.59% |
| REFoCUS tikhonov | 1.84 mm (+31%) | 1.06 mm | 12.6 dB | 0.14% |
| REFoCUS tsvd | 1.78 mm (+27%) | 1.05 mm | 12.4 dB | 0.14% |
| incoherent | 7.48 mm (+434%) | 2.66 mm | 6.7 dB | 0.65% |
| ÷ pfield / ÷ synth field | — | no-op / anti-phase | — | — |

**Mechanism confirmed** — but see the CORRECTION banner in `docs/focused_bmode_striations.md`:
the ripple was measured about the array origin instead of the virtual apex 12.1 mm behind it, so
every ripple figure here understates the artefact ~10x. Apex-referenced, standard is **15.41%
(1.24 dB) at exactly the 1.111° transmit pitch**, REFoCUS adjoint 1.54%. Rankings unchanged.
(old text: ripple 1.284° on phantom vs 1.264° in vivo vs 1.247° beam width; the
synthesised 2nd-harmonic field peaks at 1.260°), but **every fix costs more than the artefact**.
Keep standard.

**Processing note:** needs the `D:\swp_res` junction (original path is 209 chars, overruns
MAX_PATH) and the *same session's elasticity-phantom* `CombinedData.mat` as base config —
`S5_1_SWI_Luuk.mat` is a setup-time workspace missing `Receive.startSample`/`endSample`, and
`BaseConfig_10frames_*` has only half the buffer-2 `Receive` entries a 20-push run needs.

```
python run.py beamform "D:/swp_res/Resolution phantom/DefaultPatient_SW_data_18-June-2026_13-52-51" \
  --base-config-dir "D:/Luuk van Knippenberg/SWI/Base config files" \
  --base-config "D:/swp_res/Elasticity phantom/DefaultPatient_SW_data_18-June-2026_14-05-44/CombinedData.mat"
```

**Deconvolution lead — chased, and it does NOT transfer in vivo.** On the phantom, dividing by the
synthesised harmonic field at γ 0.5–1.0 improved the PSF 1.40 → **1.07 mm** and CNR 13.8 → **18.2
dB**. Applied to C000000001 buffer 3 (`study/analysis/c1_field_correct.py`, `docs/…striations.md` §5b)
it instead **imprints the field's own texture**: ripple 2.69 → 14.67%, and although the lateral
correlation collapses 9.07 → 1.18 mm (which reads like a 7.7× resolution gain), the ripple peak
moves onto the map's own 1.260° and speckle SNR drops 0.74 → 0.45. In vivo there are no point
targets to tell sharpening from stamping apart. See `study/montages/c1_field_correction.gif` — the
γ 0.5 and γ 1.0 tiles are visibly cross-hatched. **Not adopted.** Revisiting it would need a real
nonlinear-propagation harmonic field, not the squared-linear stand-in (whose ripple is ~6× too deep).

**The pfield compound is a no-op in vivo too** — 13.8 dB of span, image ripple unchanged to three
digits (2.69 → 2.69%), −6 dB of dynamic range. Inherent: `compute_pfield` returns magnitudes, so
its only compound is the incoherent `Σ|A|`, while the striations live in `|ΣA|`.

## 5. State / what is running

### Buffer-3 REFoCUS sweep — DONE
44/44 ok, 0 failed. `CombinedData_buffer3_refocus-adjoint_iq.hdf5` + `.gif` in every folder.
Ripple at the 1.111 deg transmit spacing across the study: **min 0.07%, median 0.30%, max 1.84%**
vs the coherent baseline 3.77% — a **13x median reduction**, only 1 folder above 1%
(C000000037 at 1.84%, isolated: its neighbours are 0.28% and 0.15%).

### GIF re-render with the new display — DONE (44/44, exit 0)
`study/logs/rerender_gifs.log`. All 1188 GIFs now carry the per-acquisition levels + gamma2.
GIF-only (reads existing IQ, no re-beamforming, nothing else touched); ~45 GB of reads, so it
takes a while. To re-render after a display change — or if a run is interrupted — just run it
again; it simply overwrites:

```
PYTHONPATH=src bash "<repo>/study/logs/rerender_gifs.sh"
```

or per folder: `python -m swp.acquisition.gifs "<folder>/output"`.

### Passive general M-lines — 1/44, STOPPED
Stopped on request (windows kept popping up). Nothing lost; each line is saved the moment it is
drawn and drawn folders are skipped. Resume any time:

```
python scripts/draw_passive_mlines.py --root "Z:/raw_data"            # resume
python scripts/draw_passive_mlines.py --root "Z:/raw_data" --dry-run  # what is left
```

Opens the buffer-4 B-mode as a **looping cine** (whole buffer in a 5 s loop = 5x slow motion,
starting at frame 0 = the R-peak, where the gated acquisition is at end-diastole and sharpest).
The selector now uses the **same display as the GIFs** (adaptive levels + gamma2) - it previously
used the old darker rendering, which is much of why the M-lines were hard to draw.

~34 s load per folder. This covers 1 of the ~5 M-lines per folder; the per-window ones are drawn
during `run.py passive` and cannot be front-loaded. See `docs/passive_mlines.md`.

## 6. Files

* `study/analysis/` — one-off scripts + figures from the investigation. `buf3_per_tx.npy` (118 MB) is
  the expensive artefact: all 73 transmits of buffer 3 frame 0 beamformed **separately**, so
  further composite-rule experiments run on CPU with no re-beamforming.
  `buffer1_own_spacing.py` is the frame-averaged angular-spectrum tool — **use a peak fit, not a
  windowed sum**; the windowed version produced a spurious "buffer 1 has a 4° artefact".
* Phantom chain (§4a), in order: `phantom_psf.py` (target detection + lateral/axial −6 dB + CNR + angular ripple; the other three import it), `pfield_normalise.py` (zea `compute_pfield`, `downsample=1`, `norm=False`), `pfield_evaluate.py`, `tx_field_normalise.py` (synthesised complex Rayleigh–Sommerfeld field, fundamental + 2nd harmonic), `txfield_correct.py` (the γ sweep — this is the one holding the deconvolution lead).
* `study/logs/` — `batch.log` (the 44-folder run), `verify.log`, `refocus*.log`, `incoherent.log`,
  `passive*.log`, `draw_passive_mlines*.log`, and the `queued_*.sh` GPU-chaining runners.

## 7. Open items

**For you to decide / do:**

1. ~~Re-run buffer 3 for all 44~~ **DONE (2026-09-14)** — 44 ok, 0 failed; all verified against
   the independent reference implementation. See §0.
2. **Passive processing for all 44** — the next stage. 1/44 general M-lines drawn; resume with
   `python scripts/draw_passive_mlines.py --root "Z:/raw_data"`, then `run.py passive` per folder,
   which prompts for up to 4 more M-lines each.
3. **80 lines instead of 73?** — measured answer in §8. Short version: not worth it.
4. ~~Incoherent-sum GIFs~~ **DONE for C000000001** — buffers 1/3/4, standard vs incoherent, in
   `study/montages/incoherent_allbuffers.gif`. No all-44 set, deliberately (~7 h, and the phantom puts
   incoherent compounding at a 7.48 mm lateral PSF — a diagnostic, not an imaging mode).
5. ~~Deconvolution lead~~ **CLOSED** — chased on C1; it imprints the field texture in vivo (§4a).

## 8. Would 80 lines instead of 73 help?

**No — marginal, and it costs frame rate.**

| | pitch | beams within the 2.3° beam width | frame rate |
|---|---|---|---|
| 73 lines over 80° | 1.1111° | 2.07 | 25.4 Hz |
| 80 lines over 80° | 1.0127° | 2.27 | 23.2 Hz (**−9%**) |

The artefact is already **saturated** in the number of overlapping transmits. From the nearest-k
ladder (§5e of the striations doc), ripple is 14.99% at k=6, 14.88% at k=12 and 14.97% at k=73 —
flat once about six transmits contribute. Going from 2.07 to 2.27 beams of overlap moves nothing;
it is a 9% change in a parameter the artefact stopped responding to long before.

The one thing 80 lines *would* change is that the REFoCUS encoding matrix becomes **80×80 —
square** — instead of 73×80 under-determined. That sounds like it should matter, but the phantom
already measured all three inversions as equivalent (adjoint 1.87 mm, tikhonov 1.84, tsvd
1.78), so rank deficiency is not what limits them — and `adjoint` does not invert anything at all,
so it cannot benefit. The new default would be unaffected.

**Caveat on the evidence:** the decimation test can only make the pitch *coarser* than acquired,
so the extrapolation to a finer pitch rests on the saturation argument rather than a direct
measurement. If you want certainty, one acquisition at 80 lines would settle it.
