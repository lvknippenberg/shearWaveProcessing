# Radial striations in the focused B-mode (buffer 3)

Investigation log for the fine radial lines visible in the buffer-3 GIF, which the widebeam
(buffer 1) and diverging-wave (buffer 4) B-modes do not show. Records the sequence geometry as
read from the data, what was **ruled out** and by which measurement, and the hypothesis that
currently survives. Status: **open — awaiting resolution-phantom data.**

All numbers below are read from `CombinedData.mat` for
`Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54` unless stated.

## 1. The sequence, as acquired

From `Bmode_FC` and `Trans`:

| quantity | value | source |
|---|---|---|
| probe centre frequency | 3.125 MHz → λ = **0.4928 mm** | `Trans.frequency` |
| element pitch | **0.254 mm** | `Trans.spacingMm` |
| elements / active aperture | 80 / 79 → **20.07 mm** | `Trans.numelements`, `TX.Apod` |
| transmit focus | 160 λ = **78.85 mm** | `Bmode_FC.txFocus` |
| requested F-number | **3.5** | `Bmode_FC.txFNum` |
| aperture-limited F-number | **3.93** (F/3.5 at 78.85 mm needs a 22.5 mm aperture; only 20.07 mm exists) | derived |
| lines | **73** over ±40° | `Bmode_FC.na`, `.theta` |
| line spacing | 0.019393 rad = **1.111°** | `Bmode_FC.rayDelta` |
| sector apex | 24.57 λ = **12.11 mm behind the face** | `Bmode_FC.radius` |
| receive window | 396.8 λ = **195.5 mm** along the beam | `Receive.endDepth` |

> **Watch the units.** `Trans.ElementPos` is in **mm**, while `TX.Origin` / `TX.focus` /
> `PData` are in **wavelengths**. Reading `ElementPos` as wavelengths halves the aperture and
> doubles the F-number — an error that ran through the first pass of this investigation and
> produced a spurious "2.28× beam overlap, therefore fine" conclusion.

### Angular sampling

```
one-way -6 dB beam width at focus  = 0.886 * lambda * F#  = 1.72 mm = 1.247 deg
line spacing                                              =           1.111 deg
one-way overlap                                           =           1.12x
two-way (dynamic receive focusing), approx one-way/sqrt(2) =          ~0.88 deg
```

So the scan is **marginally sampled one-way (1.12×) and under-sampled two-way** (~0.88° beam vs
1.111° spacing) at the focal depth. Away from the focus the beams broaden and the sampling
becomes comfortable again.

## 2. Ruled out

| hypothesis | measurement | verdict |
|---|---|---|
| Gaps between transmit beams — pixels no beam illuminates | one-way beam width vs line spacing | **No gaps**, but the margin is only 1.12× (not the 2.28× first computed from a halved aperture) |
| Per-line transmit gain variation | `sum|TX.Apod|` across the 73 lines: std **0.000%**, all 79 elements active, no odd/even split | ruled out |
| Transmit delay quantisation | `TX.Delay` step = 0.005 λ = **1.82°** of phase | negligible, ruled out |
| "The beam is too focused" | F/3.93, beam 1.72 mm wide at focus | wrong framing — the issue is angular *sampling*, not focus tightness |
| Coherently summing all 73 transmits (vs only those that insonified a pixel) | per-transmit reconstruction + composite rules, line-spacing power in the 45–85 mm band: all-73 **2.27%**, ang3 2.84%, ang2 3.15%, ang1 4.44%, nearest-beam **10.31%** | **backwards** — using *fewer* beams is monotonically worse; full compounding is the best coherent rule |
| `enable_pfield` transmit-field weighting would fix it | pfield on vs off, buffer 3 | improves signal (+3.5 dB p90) but does **not** remove the striations |
| The structure is not focused-specific at all (first conclusion, **wrong**) | single-frame ripple metric showed buffers 1/3/4 all ≈2.25× excess | the metric was speckle-dominated; see §3 |

## 3. The measurement that actually separates it

Single-frame angular ripple is swamped by speckle (22–38% RMS). **Frame-averaging** separates
them: speckle falls as 1/√N, a fixed reconstruction artefact does not.

Fraction of angular-ripple power at the 1.111° line spacing, frame-averaged:

| depth band | buffer 1 (widebeam, N=90) | buffer 3 (focused, N=26) |
|---|---|---|
| 45–85 mm (spans the 78.85 mm focus) | 0.25% | **5.06%** |
| 105–145 mm | 0.04% | 0.02% |

A **20× enrichment at exactly the line spacing**, focused-only, concentrated around the focal
depth and absent at depth. Visible directly as a fine oscillation on the focused angular profile
that the widebeam profile does not have.

## 3a. Where the power actually sits (peak fit, not a windowed sum)

Scanning the whole frame-averaged angular spectrum rather than probing one assumed frequency
(`analysis/buffer1_own_spacing.py` in the Claude working folder):

| buffer | transmit step | distinct spectral peaks (45–85 mm) |
|---|---|---|
| 1 widebeam (21 tx) | 4.0° | **18.0° only** — no peak at its own 4.0° step |
| 3 focused (73 tx) | 1.111° | 12.0° **and 1.264°** (17.3% of ripple power) |

Two things follow:

* **Buffer 1 is genuinely clean of transmit-periodic structure.** It has no distinct peak at its
  own 4.0° transmit spacing; its only peak is broad 18° shading (sector illumination / anatomy).
  This holds even though buffer 1 is **2.9× under-sampled** by the coherent plane-wave
  compounding criterion (4.0° step vs λ/D = 1.39°) — far worse than buffer 3's 0.79×. Its
  9.4° diverging-wave opening angle (2.36× overlap) evidently protects it.
* **Buffer 3's artefact periodicity is the beam width, not the sampling step.** 1.264° measured
  vs 1.247° one-way −6 dB beam width (1.4% apart) vs 1.111° line spacing (14% apart). A pure
  sampling/aliasing effect would land on the line spacing; landing on the beam width points at
  the beam pattern itself. *Caveat:* the 1.247° figure is a derived estimate (0.886 λ F#), so
  agreement at the 1–2% level should not be over-read.

**Methodological lesson:** measure the *peak*, not the power in a window around an assumed
frequency. The windowed metric reported a spurious 5.87% "artefact at 4°" for buffer 1 that a
peak fit shows is simply the skirt of its low-frequency shading.

## 4. Current hypothesis

**The focused sector scan is angularly under-sampled near its focus, so the compounded transmit
field scallops at the line spacing.**

This fits every surviving observation:

* the periodicity is **1.264°** (measured: isolated spectral peak at 0.791 cycles/deg, 17.3% of
  ripple power). Note this matches the **one-way −6 dB beam width (1.247°)**, *not* the line
  spacing `rayDelta` (1.111°) — a 14% offset, well outside the spectral resolution. An earlier
  draft claimed it sat exactly at `rayDelta`; that was read off a ±12% power window rather than
  a peak fit, and is wrong. See §3a;
* it is focused-only — the widebeam and diverging-wave buffers insonify the whole sector per
  transmit and have no per-line structure to scallop (0.25% / 0.04%);
* it peaks in the 45–85 mm band and vanishes by 105–145 mm, matching where the beams are
  narrowest (at the focus) and where they have broadened again;
* two-way beam width (~0.88°) is *below* the 1.111° line spacing at the focus, so the field
  between adjacent beam axes genuinely dips there;
* it is a *coherent* effect: incoherent (envelope) compounding reduces the line-spacing power
  from 2.27% to **0.08%**, a 28× drop (`scripts/incoherent_bmode.py`).

The effect is modest in absolute terms — a few percent of the angular ripple power. The bulk of
what the eye sees in the GIF is still speckle and anatomy at 22–38% ripple.

### A quantitative check, and what it implies

Summing the beam patterns over lines and measuring the ripple across one line period predicts
the scalloping from geometry alone, with nothing fitted. **Which beam pattern applies depends on
the reconstruction**:

| reconstruction | relevant pattern | overlap | predicted ripple |
|---|---|---|---|
| scanline (Verasonics): tx beam × dynamic receive on the *same* line | two-way, 0.882° | 0.79× | **42.4%** (−3.74 dB) |
| full DAS (ours): every pixel uses the full receive aperture from *every* transmit | one-way, 1.247° | 1.12× | **4.5%** (−0.39 dB) |

Measured in our images (frame-averaged, 45–85 mm): **~5%**. The one-way prediction matches, and
the two-way figure explains why the same sequence shows far stronger lines under a conventional
scanline reconstruction.

### Open question

Whether the residual is transmit-field **scalloping** (a multiplicative sensitivity dip) or
**interference** intrinsic to coherently summing transmits sampled at 1.111°. Both predict
line-spacing periodicity and both scale with beam overlap, so the numerical agreement above does
not by itself settle it.

The evidence currently favours **interference**: a multiplicative sensitivity dip would survive
envelope summing, and it does not (2.27% → 0.08% under incoherent compounding). If that holds,
dividing out a computed transmit-sensitivity map cannot fix it.

**Discriminator for the phantom:** does the residual survive incoherent compounding? A uniform
scatterer removes anatomy and shadowing, so the sensitivity pattern can be measured directly
against a known-flat target, and its depth profile compared with the predicted beam-width curve.

## 4a. What would fix it

Conditional on the hypothesis. The two families trade against different resources.

### In processing (works on existing data, no re-acquisition)

| approach | effect | cost |
|---|---|---|
| **Incoherent (envelope) compounding** — `scripts/incoherent_bmode.py` | line-spacing power 2.27% → **0.08%** (28×), demonstrated | discards coherent gain: softer lateral resolution, lower speckle contrast |
| **REFoCUS / retrospective transmit beamforming** | invert the transmit encoding to synthesise a full synthetic-aperture dataset with uniform transmit coverage, then re-beamform | non-trivial to implement; needs per-transmit channel RF + `TX.Delay` + `TX.Apod` — **all of which we already store** |
| **Transmit-field normalisation** (divide by the compounded pfield) | cheap to try | predicted **not to work**: you cannot normalise away an interference pattern. Worth one confirming test, not worth building on |

### In acquisition

| change | overlap | cost | predicted ripple (our recon) |
|---|---|---|---|
| 73 → **91 lines** | two-way 0.79 → 0.99× | frame rate 25.4 → **20.4 FPS** | — |
| 79 → **63 elements** (F/3.93 → F/4.93) | one-way 1.12 → 1.41× | lateral resolution 1.72 → **2.15 mm** | 4.53% → **0.35%** |
| 79 → 55 elements (F/5.64) | one-way → 1.61× | lateral resolution → 2.46 mm | → 0.04% |

The aperture route is counter-intuitive and cheap: **reducing** the transmit aperture widens the
beam, improving angular sampling at **zero frame-rate cost**, paying ~25% lateral resolution at
the focus instead. The line-count route keeps resolution and pays frame rate.

### Measured: all five reconstructions, same RF (buffer 3, C000000001, 26 frames)

| reconstruction | line-spacing power | ripple RMS | lat. corr. | dyn. range | time |
|---|---|---|---|---|---|
| coherent all-73 (current pipeline) | 3.77% | 17.6% | 24.45 mm | 32.9 dB | 62 s |
| incoherent (envelope sum) | 0.31% | 14.4% | 38.64 mm | **18.7 dB** | 155 s |
| **REFoCUS adjoint** | **0.16%** | 19.4% | **21.29 mm** | **33.7 dB** | **69 s** |
| REFoCUS tikhonov | 1.60% | 16.8% | 24.45 mm | 28.7 dB | 793 s |
| REFoCUS tsvd | 4.46% | 20.1% | 26.03 mm | 21.1 dB | 788 s |

**REFoCUS `adjoint` is the best option on every axis** - 24x less line-spacing structure than the
current pipeline, *better* lateral correlation, best dynamic range, at essentially the same cost
as the normal beamform (69 s vs 62 s for 26 frames).

Two findings that the metrics alone would have got wrong, and which only looking at the images
revealed:

* **Incoherent compounding is not viable.** Its low line-spacing power and low ripple come from
  being washed out, not from being better: dynamic range collapses to 18.7 dB and the sector is
  uniformly grey with anatomy barely separable from background. The metrics rewarded smoothing.
* **The SVD inversions confirm the rank-deficiency.** `tikhonov` shows fine amplified noise
  throughout and `tsvd` is *worse than doing nothing* (4.46% vs the 3.77% baseline), both with
  visible texture and 11x slower. With H under-determined at 73 x 80, inverting harder amplifies
  noise - which is why `adjoint` (the matched filter, which does not attempt a full inversion) is
  the right choice here.

**Caveat before adopting:** one dataset, and the lateral-correlation column is an anatomy-scale
proxy (~24 mm), not a true resolution measure. Resolution-phantom point targets are what would
justify switching buffer 3 to REFoCUS in the pipeline.

### Or: use buffer 1

Buffer 1 is a widebeam B-mode of the same anatomy at 88 FPS with no line structure (0.25% vs
5.06%). If buffer 3 exists for resolution or penetration, the striations are the price of the
tighter beam; if it exists for orientation, buffer 1 already does that job cleanly.

## 5. Separately: the Verasonics reconstruction regions

Distinct from our pipeline (which does not use them), and the mechanism seen previously with the
default Verasonics reconstruction.

`PData(3).Region` is 73 regions, one per line, each a **sector of a circle**: angular width
0.1164 rad = **6.67°**, radius **324.57 λ** about the apex 24.57 λ behind the face.

* Inside the arc, coverage is **uniformly 6 regions per pixel at every depth** — constant angle,
  6.67 / 1.111 = 6. There is no angular gap and no coverage-driven line structure there.
* The arc is a **finite radius**, reaching `324.57 − 24.57 = 300 λ = 147.8 mm` on axis — the
  sequence's nominal end depth. At the sector edges the same *z* is further from the apex
  (`ρ = √(x² + (z + 12.11 mm)²)`), so the arc cuts in from the sides:

```
   z=100mm  covered to |x|=114mm  (sector half-width  81mm) -> 100%
   z=120mm  covered to |x|= 90mm  (                   97mm) ->  93%
   z=140mm  covered to |x|= 50mm  (                  113mm) ->  44%
```

  Predicted 74% of pixels uncovered at z = 145 mm inside ±39°; **measured 76.3%** from the
  explicit `PixelsLA` index lists.
* This is **not** a data limit: `Receive.endDepth` is 396.8 λ = 195.5 mm along the beam, so the
  samples for those deep corners exist. The regions simply stop at 300 λ, leaving the deep
  corners of PData un-reconstructed.

Note also that region-based reconstruction sits at the *few-beams-per-pixel* end, which §2
measured as intrinsically more line-prone (6 beams ≈ 2.84%, 1 beam 10.31%, vs 2.27% for full
compounding). So a Verasonics-reconstructed image of this sequence should show *stronger* line
structure than ours, plus a coverage boundary past ~110–120 mm at the edges.

## 6. Reproducing

| script | what |
|---|---|
| `scripts/incoherent_bmode.py <folder> --buffer 3` | envelope-compounded reconstruction + GIF, written alongside the normal output |

The per-transmit reconstruction, composite-rule comparison, region-coverage map and
frame-averaged ripple metric were run as one-off analyses; the numbers above are the record. The
frame-averaged line-spacing power is the metric worth re-running on new data — single-frame
ripple is too speckle-dominated to be informative.

## 7. Practical impact

For the shear-wave processing: **none**. Buffer 3 is a focused B-mode for orientation and never
feeds the displacement estimators. Buffer 2 (active tracking) and buffer 4 (passive) both use
unfocused transmits and show no line-spacing structure.

`enable_pfield=True` is enabled for buffer 3 (see `BufferSpec.pfield`) because it improves signal
recovery, not because it addresses the striations — it does not.
