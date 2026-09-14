# Radial striations in the focused B-mode (buffer 3)

Investigation log for the fine radial lines visible in the buffer-3 GIF, which the widebeam
(buffer 1) and diverging-wave (buffer 4) B-modes do not show. Records the sequence geometry as
read from the data, what was **ruled out** and by which measurement, and the hypothesis that
currently survives. Status: **REOPENED (2026-09-14)** — the mechanism is confirmed and the method ranking holds, but the artefact is ~10x larger than reported (see the CORRECTION below), so the cost/benefit of REFoCUS adjoint needs revisiting. See §5e.

> ## CORRECTION (2026-09-14, after S5d) - read before using any ripple number below
>
> **Every angular-ripple figure in S2-S5d was measured about the wrong centre and understates the
> artefact by about 10x.** The metric resampled the image about the ARRAY ORIGIN (`theta =
> arctan2(x, z)`), but the sector is steered from a **virtual apex 12.1 mm behind the array face**
> (`CenterTransmit.mat`: `Region.Shape.Position(3) = -24.57 lambda`). A structure periodic at
> 1.1111 deg in apex-angle is not periodic in origin-angle - over the 45-85 mm band the ratio runs
> 1.15-1.19 - so the peak was smeared across periods and its amplitude spread over neighbouring
> bins.
>
> Re-measured about the apex (`analysis/apex_referenced_ripple.py`), the standard reconstruction's
> line-spacing ripple is **15.41%, not 2.02%**, and its peak sits at **1.112 deg - exactly the
> 1.1111 deg transmit pitch**. The decimation test confirms it: pitch 2.222 deg -> peak 2.144 deg,
> pitch 3.333 deg -> 3.159 deg.
>
> What this changes:
> * **S5d's central argument is WITHDRAWN.** It argued the image peak was *not* a harmonic of the
>   transmit pitch and concluded on spectral grounds that no amplitude mechanism could produce it.
>   The peak IS at the pitch; that argument was an artefact of the geometry error. The *conclusion*
>   that the beam/region geometry is not the cause still stands, but now on magnitude alone - the
>   region-truncation prediction is 0.76% against a measured 15.41%, i.e. 20x too small (both
>   apex-referenced, so directly comparable).
> * **The "artefact is only 0.12-0.23 dB, too small to care" framing in S5c is wrong.** 15.41% is
>   **1.24 dB** of envelope modulation. That is visible, which is why it was noticed in the first
>   place, and it makes REFoCUS adjoint a more serious option than S5c concluded.
> * **Rankings are unaffected.** All reconstructions were measured the same wrong way, so their
>   order is unchanged; only the magnitudes move. Apex-referenced: standard 15.41%, REFoCUS adjoint
>   1.54%, tikhonov 1.46%, tsvd 1.39%, incoherent 1.84% - a ~10x reduction rather than the ~3.4x
>   reported in S5a.
>
> This is the fourth time in this investigation a metric rather than the data gave the wrong
> answer. The others: single-frame ripple (speckle-dominated), a windowed power sum read as a peak,
> and an anatomy-scale correlation length read as a PSF.

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

## 5a. Resolution-phantom results — and the conclusions they REVERSED

Phantom: `D:\swp_res\Resolution phantom\DefaultPatient_SW_data_18-June-2026_13-52-51`
(2026-06-18, 79 point targets over 30-105 mm). Its focused sequence is **identical to in-vivo**
on all seven parameters (`na`, `txFocus`, `txFNum`, `rayDelta`, `theta`, `aperture`, `radius`),
so the results transfer.

### Mechanism: confirmed

The standard reconstruction's ripple peaks at **1.284 deg** on the phantom, against **1.264 deg**
in vivo and a one-way beam width of **1.247 deg**. Three independent measurements agree, on data
with no anatomy or rib shadowing to confuse the spectrum.

The synthesised **2nd-harmonic** complex transmit field peaks at **1.260 deg**; the fundamental
at 2.430 deg, i.e. twice the period, exactly as harmonic doubling predicts. So the scalloping is
transmit-related and lives in the harmonic band the images are formed in.

### REFoCUS DEGRADES resolution — reversing §4a

Measured on point targets, median over 79 targets:

| reconstruction | lateral -6 dB | axial -6 dB | target CNR | ripple |
|---|---|---|---|---|
| **standard (coherent all-73)** | **1.40 mm** | **0.97 mm** | **13.8 dB** | 2.02% |
| REFoCUS adjoint | **1.87 mm (+34%)** | 1.03 mm | 12.9 dB | 0.59% (3.4x) |
| REFoCUS tikhonov | 1.84 mm (+31%) | 1.06 mm | 12.6 dB | 0.14% |
| REFoCUS tsvd | 1.78 mm (+27%) | 1.05 mm | 12.4 dB | 0.14% |
| incoherent (envelope) | 7.48 mm (+434%) | 2.66 mm | 6.7 dB | 0.65% (3.1x) |

All three REFoCUS inversions land in the same place - 27-34% wider laterally - so the penalty is
inherent to the under-determined 73x80 inversion, not to the choice of regulariser. Note also that
the 2.835 deg "peak" for every non-standard reconstruction is the bottom of the search band: once
the line-spacing term is removed there is no peak left to find, which is the correct outcome and
not a competing artefact.

**The in-vivo lateral-correlation figure said REFoCUS IMPROVED resolution (21.29 vs 24.45 mm).
That was wrong** - an anatomy-scale correlation length is not a PSF, as flagged at the time. On
real point targets REFoCUS is wider at every depth. The trade is 3.4x less ripple for 34% of
lateral resolution.

### The residual largely SURVIVES envelope summing — reversing §4's "open question"

Using **absolute** ripple amplitude (the fraction-of-power alone misleads, since the totals are
20.6 / 17.5 / 8.1%): standard **2.02%**, incoherent **0.65%** - only **3.1x** down, not the 28x a
single in-vivo frame suggested. A multiplicative dip survives envelope summing, so this points at
amplitude scalloping rather than pure interference.

### Normalising by the transmit field does NOT work

| correction map | ripple @ line spacing | its own peak | result |
|---|---|---|---|
| zea `compute_pfield` (returns float32 **magnitudes**, so only `sum|A|`) | 0.02% | 1.063 deg | ~100x too smooth: **no-op** |
| synthesised coherent `|sum A|`, fundamental | 2.43% | 2.430 deg | wrong period |
| synthesised coherent `|sum A|`, 2nd harmonic | 13.05% | **1.260 deg** | right period, but **anti-correlated with the image (-0.68)** |

The anti-correlation is robust - mirroring the map or negating `t0_delays` leaves it at -0.68 -
so it is not a sign error. The reason is that `|sum A|` is the sensitivity for a **point
scatterer**, while the ripple is measured in **speckle**, whose mean intensity follows the
*incoherent* compound `sum|A|^2` - and that one is flat. Neither model predicts the speckle
ripple with the right sign, so there is no correction map to divide by.

### Verdict

**Keep the standard reconstruction.** Every method that reduces the ripple costs resolution, the
one that would not cost resolution cannot be built from the available field models, and the
artefact is ~2% angular ripple on a buffer that exists for orientation and never feeds the
shear-wave estimators.

### Loose end worth a look later

Dividing by the harmonic map at gamma 0.5-1.0 **improves** the PSF (1.40 -> **1.07 mm**, -24%)
and CNR (13.8 -> **18.2 dB**) while *increasing* ripple. It is behaving as a deconvolution rather
than a flat-field correction. Unrelated to the striations, but a real resolution-enhancement lead.

## 5b. In-vivo check of the two field-normalisation maps (C000000001, 2026-09-14)

The phantom left one loose end worth chasing: dividing by the synthesised harmonic field made the
ripple *worse* but made point targets *sharper* (1.40 -> 1.07 mm, CNR +4.4 dB), behaving like a
crude deconvolution. Both maps were therefore rebuilt on C000000001's buffer-3 grid and applied
in vivo. Script: `analysis/c1_field_correct.py` in the working folder.

| C1 buffer 3, 26 frames | ripple @ spacing | its peak | lat. corr | speckle SNR | dyn. range |
|---|---|---|---|---|---|
| **standard** | **2.69%** | 1.284 deg | 9.07 mm | **0.74** | 51.3 dB |
| / pfield compound | 2.69% | 1.284 deg | 7.49 mm | 0.70 | 45.2 dB |
| / synth harmonic, g=0.25 | 2.99% | 1.260 deg | 6.70 mm | 0.69 | 48.2 dB |
| / synth harmonic, g=0.50 | 6.01% | 1.260 deg | 5.52 mm | 0.62 | 47.1 dB |
| / synth harmonic, g=1.00 | 14.67% | 1.260 deg | 1.18 mm | 0.45 | 55.0 dB |

Both phantom results reproduce in vivo:

* **The pfield compound is a no-op.** It spans 13.8 dB across the sector, but its own ripple at the
  line spacing is 0.02% against the image's 2.69%, and dividing by it leaves the image ripple
  unchanged to three digits. It costs 6 dB of dynamic range for nothing. This is inherent, not a
  tuning problem: `compute_pfield` returns magnitudes, so the only compound available is the
  incoherent `sum|A|`, which is smooth by construction. The striations live in `|sum A|`.
* **Dividing by the harmonic field monotonically worsens the striations**, 2.69% -> 14.67%.

The interesting column is `lat. corr`, which collapses 9.07 -> 1.18 mm and *looks* like a 7.7x
resolution gain. It is not. The ripple peak moves off the image's 1.284 deg onto the field map's
own **1.260 deg** at every gamma - the map is imprinting its texture on the image. Speckle SNR
falling 0.74 -> 0.45 is the same effect seen a second way: added multiplicative texture. A
correlation length cannot separate "resolved finer anatomy" from "stamped with a finer pattern";
on the phantom the wire targets could, which is the whole reason the phantom was needed.

**This is the third time in this investigation that a normalised or proxy metric pointed the wrong
way** (after the single-frame ripple fraction and the in-vivo REFoCUS correlation length). The
visual check is unambiguous and took seconds: `montages/c1_field_correction.gif`.

**Conclusion unchanged: keep the standard reconstruction.** The deconvolution lead is real on the
phantom but is not separable from field-texture imprinting in vivo, so it is not a route to a
cleaner in-vivo B-mode. If it is ever revisited, it needs a physically generated harmonic field
(a nonlinear propagation model) rather than the squared-linear stand-in used here, whose ripple is
~6x too deep.

## 5c. Scorecard: all 8 reconstructions, one consistent metric set

Earlier tables in this doc mixed two different "ripple" numbers - a FRACTION of ripple power in the
line-spacing band, and an ABSOLUTE ripple amplitude. They are not comparable. `analysis/
method_scorecard.py` recomputes everything as **absolute amplitude**, over a **30-105 mm** band on
the phantom and 45-85 mm in vivo (so these do not match the S5a numbers, which used 45-85 mm on
both - the ripple concentrates at the focal depth, so a wider band dilutes it).

It also adds the metric the earlier tables lacked. `rip_line` only sees the 1.111 deg band, so a
reconstruction can score well there and still show obvious BROAD radial streaks. The discriminator
for "radial lines" is not amplitude but **depth persistence**: speckle decorrelates with range, a
transmit-field streak does not. `persist` correlates the near-half and far-half angular profiles.

**Phantom** (79 point targets):

| method | rip_line | ang_rms | persist | lateral | CNR | dyn.rng |
|---|---|---|---|---|---|---|
| **standard** | 1.41% | 17.7% | 0.034 | **1.40 mm** | 13.8 dB | 58.3 dB |
| incoherent | 0.59% | 7.8% | 0.074 | 7.48 mm | 6.7 dB | 36.6 dB |
| REFoCUS adjoint | 0.99% | 16.2% | 0.036 | 1.87 mm | 12.9 dB | 61.7 dB |
| REFoCUS tikhonov | 1.03% | 16.2% | 0.040 | 1.84 mm | 12.6 dB | 54.2 dB |
| REFoCUS tsvd | 1.05% | 16.1% | 0.034 | 1.78 mm | 12.4 dB | 50.1 dB |
| / pfield | 1.41% | 17.6% | 0.031 | 1.41 mm | 13.6 dB | 54.6 dB |
| / harmonic g=0.5 | 4.95% | 29.2% | 0.168 | 1.14 mm | 16.7 dB | 52.6 dB |
| / harmonic g=1.0 | 11.36% | 51.0% | 0.281 | **1.07 mm** | **18.2 dB** | 51.5 dB |

**In vivo C000000001** (no point targets, so no PSF):

| method | rip_line | ang_rms | persist | spk SNR | dyn.rng |
|---|---|---|---|---|---|
| **standard** | 2.69% | 15.0% | 0.533 | 0.74 | 51.3 dB |
| incoherent | 0.50% | 8.7% | 0.729 | 1.73 | 36.1 dB |
| **REFoCUS adjoint** | **0.56%** | 16.8% | 0.666 | 0.69 | 51.5 dB |
| REFoCUS tikhonov | 1.50% | 13.1% | 0.472 | 0.85 | 43.8 dB |
| REFoCUS tsvd | **3.26%** | 15.1% | 0.221 | 1.38 | **36.2 dB** |
| / pfield | 2.69% | 14.9% | 0.534 | 0.70 | 45.2 dB |
| / harmonic g=0.5 | 6.01% | 28.3% | 0.542 | 0.62 | 47.1 dB |
| / harmonic g=1.0 | 14.67% | 54.4% | 0.477 | 0.45 | 55.0 dB |

Three things these two tables show that no single-dataset table could:

**1. The harmonic division really does sharpen - the PSF gain is not a metric artefact.** 1.40 ->
1.07 mm (-24%) and CNR +4.4 dB, on 79 wire targets. But it buys that by stamping depth-persistent
radial structure: `persist` rises 8x (0.034 -> 0.281) and `ang_rms` triples. And the mechanism
should be treated with suspicion - dividing by a map with 24 dB of span and ~1.26 deg structure
narrows a point target partly by suppressing its shoulders, which shrinks the -6 dB width without
necessarily improving two-point separability. On a phantom of isolated wires that reads as pure
gain; in vivo, where nearly every pixel is speckle, the same operation is just texture.

**2. The phantom FLATTERS the SVD inversions.** tikhonov and tsvd look interchangeable with adjoint
on the phantom (1.78-1.87 mm, rip_line ~1.0%). In vivo they separate hard: tsvd's line ripple is
**3.26%, worse than the standard reconstruction it was meant to fix**, and its dynamic range
collapses 51.3 -> 36.2 dB. That is textbook rank-deficient inversion amplifying noise, and the
phantom cannot show it because its SNR is far higher than a heart at 100 mm. **Any future
phantom-only verdict on a regularised inversion should be distrusted for this reason.**

**3. `persist` is only meaningful on the phantom.** In vivo every method scores 0.22-0.73 because
anatomy is itself depth-persistent - a bright ridge spans many radii. The phantom's speckle region
is the only place the metric isolates the transmit field.

### Verdict

**Keep standard.** On the phantom its striation is 1.41% absolute envelope modulation - **0.12 dB** -
and does not persist through depth (0.034). In vivo it is 2.69%, or **0.23 dB**. Nothing else on
either table is worth 27-434% of lateral resolution to remove an artefact that small.

**If it ever must be removed, use REFoCUS adjoint and nothing else.** It is the only alternative
that cuts the line ripple (2.69 -> 0.56% in vivo) while holding dynamic range (51.5 vs 51.3 dB).
The price is measured and real: 34% wider laterally on the phantom.

## 5d. The simulated centre beam vs its region - and why no field map can fix this

The user supplied `CenterTransmit.mat`: the Verasonics-simulated field of the centre beam (region
37 of 73) together with the `TransmitPData` defining which pixels that transmit reconstructs. This
is the direct measurement the "too focused" hypothesis needed, and it both **confirms the geometry
and rules it out as the cause**. Scripts: `analysis/center_transmit_vs_region.py`,
`region_truncation_test.py`, `region_truncation_figure.py`. Figure: `montages/region_vs_beam.png`.

### The geometry claim is correct

Region 37 is a `SectorFT` with a virtual apex 12.1 mm behind the array and a full angle of
**6.667 deg = exactly 6.00x the 1.111 deg line spacing**. Every pixel is therefore reconstructed
from 6 transmits, not 1. Against that, the simulated beam:

| depth | beam half-max | beam -20 dB | region | region / beam | region px inside the beam |
|---|---|---|---|---|---|
| 20 mm | 13.80 mm | 23.41 mm | 3.70 mm | 0.27 | 100% |
| 40 mm | 8.38 mm | 18.97 mm | 5.67 mm | 0.68 | 100% |
| 60 mm | **3.20 mm** | 12.32 mm | 8.13 mm | 2.54 | 38% |
| 80 mm | 3.70 mm | 9.61 mm | 10.60 mm | **2.87** | 34% |
| 100 mm | 5.17 mm | 16.76 mm | 13.06 mm | 2.52 | 39% |
| 140 mm | 11.58 mm | 34.25 mm | 17.49 mm | 1.51 | 65% |

So yes - **from ~45 mm down, the beam is narrower than the region it reconstructs**, by up to
2.9x at 80 mm, and only about a third of each region's pixels sit within the beam's half-max
width. The region edge lands near the beam's **-20 dB** contour (at 80 mm: 9.61 vs 10.60 mm).
Above ~45 mm the relationship inverts and the beam is wider than its region.

(The profile is a Verasonics magnitude/intensity map, so the absolute dB calibration of the
"half-max" column is not certain. Nothing below depends on it - the -20 dB column brackets it, and
the spectral argument is independent of beam width entirely.)

### But it produces the wrong artefact, by a factor of 3.5 and at the wrong period

Replicating the simulated beam at all 73 steering angles and compounding it two ways - freely, and
truncated to each transmit's own +/-3.333 deg region, which is what the beamformer does:

| compound | ripple rms | peak period |
|---|---|---|
| untruncated (free field) | 0.38% | **1.111 deg** |
| region-truncated | 0.76% | **1.111 deg** |
| **measured image (phantom)** | 16.98% (2.02% at the line spacing) | **1.284 deg** |

Region truncation **does** do something: it doubles the sensitivity ripple and discards 36% of the
compounded energy (a smooth 0.63-0.72 efficiency loss across depth - that costs SNR, not texture).
But the result is 0.76% modulation where the image shows 2.02% at the line spacing, and both
predictions peak at **1.111 deg while the image peaks at 1.284 deg**.

### Why that is decisive, not just suggestive

The 73 transmits form a **regular 1.111 deg lattice**. Any per-transmit *amplitude* effect -
beam narrowness, region truncation, apodisation, element dropout, a pfield weighting, anything
that scales a transmit's contribution without touching its phase - is a function sampled on that
lattice, so it can only produce angular structure at 1/1.111 deg/cycle and its harmonics. The
measured image peak at 1.284 deg is **not a harmonic of 1.111 deg** (0.779 vs 0.900, 1.800,
2.700 cyc/deg), and the separation is ~8 spectral bins, well beyond the resolution of the
estimate.

**An amplitude mechanism cannot put energy at 1.284 deg. So the striations are not amplitude
scalloping, and no magnitude map - Verasonics TXPD, zea `pfield`, or a synthesised
Rayleigh-Sommerfeld field - contains the information needed to remove them.** That is why every
normalisation attempt in S5a and S5b failed, and it explains the sign of the failure too: the
correction maps carry their own structure at a *different* period, so dividing by them adds a
second pattern instead of cancelling the first (measured anti-correlation -0.68).

This also retires an earlier loose end. S2 noted the image peak sits near the one-way beam width
rather than the line spacing and left that unexplained; it is now clear the peak simply is not on
the transmit lattice at all, which is a positive statement about what the mechanism *cannot* be.

### What this implies

* **The residual is cross-transmit phase interference**, consistent with everything else measured:
  it drops 3.1x (not to zero) under envelope summing, and it survives every magnitude correction.
* **The only reconstruction-side fix is one that changes the phase relationship**, which is exactly
  what REFoCUS does - it inverts the encoding rather than compensating a weight. That is why it is
  the one method that reduced the ripple without blurring (S5c), and it still costs 34% laterally.
* **The 6x region overlap is doing its job.** Using fewer transmits per pixel is monotonically
  worse (S2: single nearest beam = 10.31%), so the wide regions are a mitigation, not the cause.
* **Acquisition-side**, the lever is the transmit pitch relative to the beam, not the F-number. But
  at 2.02% (0.12 dB on the phantom) the artefact does not justify a sequence change.

## 5e. Would reconstructing each pixel from fewer transmits help?

Direct test, all 73 transmits of the phantom beamformed separately and recomposited nearest-k
(`analysis/nearest_k_composite.py`, re-measured apex-referenced in `nearest_k_apex.py`). k=6 is
what the pipeline already does, since the 6.667 deg region on a 1.111 deg lattice covers exactly 6
transmits - and it reproduces the stored standard reconstruction, which validates the composite
rule.

| composite | ripple @ pitch | peak | lateral -6 dB | CNR |
|---|---|---|---|---|
| nearest-1 (classical line imaging) | **7.41%** | 1.091 deg | 2.05 mm | 12.0 dB |
| nearest-2 | 12.40% | 1.091 deg | 1.62 mm | 13.3 dB |
| nearest-3 | 13.02% | 1.112 deg | 1.57 mm | 13.2 dB |
| **nearest-6 (= region rule)** | 14.99% | 1.112 deg | 1.44 mm | 13.6 dB |
| nearest-12 | 14.88% | 1.112 deg | 1.39 mm | 13.7 dB |
| nearest-73 | 14.97% | 1.112 deg | **1.32 mm** | 13.9 dB |
| REFoCUS adjoint | **1.54%** | 3.001 deg | 1.87 mm | 12.9 dB |
| incoherent | 1.84% | 3.001 deg | 7.48 mm | 6.7 dB |

**It is a clean monotonic trade, and nearest-1 sits at the wrong end of it.** More transmits per
pixel = better resolution (2.05 -> 1.32 mm) and more ripple (7.41 -> 14.97%). Using one transmit
per pixel halves the artefact; it does not remove it.

**And nearest-1 is strictly dominated by REFoCUS adjoint** - 7.41% / 2.05 mm versus 1.54% /
1.87 mm, worse on *both* axes. There is no operating point at which nearest-1 is the right choice.

Note nearest-1 still peaks at the transmit pitch even though it has no cross-transmit interference
by construction. So the residual is not purely an interference term between transmits: a single
focused transmit reconstructing a 1.111 deg strip already imprints the pitch, because the strip
samples the beam's own lateral profile off-axis. Compounding then *adds* to it rather than
averaging it away - which is the opposite of the intuition that started this section.

### Multiplying instead of dividing by the field

With the period confirmed to match, the S5b anti-correlation (-0.68) raised an obvious question:
if the map is anti-phase, does *multiplying* work where dividing failed? Tested: multiplying gives
15.41% -> 12.21% at gamma 0.5 while lateral goes 1.40 -> 1.52 mm. Dividing gives 21.53% and
1.14 mm. So the pair behaves as a mild blur/deconvolution, not as a scalloping correction in
either direction. Still not a fix.

## 5f. Two separate mechanisms, and which one is actually in our images

Two corrections to how S5d and S5e were framed.

### The pipeline does not use the Verasonics regions

`read_swi_meta` reads `PData.Origin`, `PDelta` and `Size` - the field of view and pixel pitch -
and **never touches `PData.Region`**. zea beamforms every pixel from **all 73 transmits**, with an
optional per-transmit `pfield` weight (`BufferSpec.pfield`, on for buffer 3). So the
region-truncation model in S5d describes what the *Verasonics* reconstruction does, not ours; our
reconstruction is the *untruncated* column of that table. And the nearest-k ladder in S5e is a
diagnostic, not a description of the pipeline - the pipeline is "nearest-73".

### The single-transmit ripple IS the non-uniform beam - confirmed quantitatively

nearest-1 has no cross-transmit term of any kind, yet it still shows ripple at the transmit pitch
(7.41%). The only mechanism available is the beam being non-uniform across its own 1.111 deg
strip: a pixel at the strip edge sits 0.556 deg off the beam axis and is insonified more weakly.

That is a phase-free geometric prediction and `CenterTransmit.mat` makes it directly
(`analysis/mosaic_prediction.py`) - mosaic the simulated beam in +/-0.556 deg strips and measure:

| effective sensitivity | predicted ripple @ pitch | peak |
|---|---|---|
| sqrt(profile) | 2.63% | 1.111 deg |
| profile (fundamental) | 3.83% | 1.111 deg |
| **profile^2 (2nd harmonic)** | **6.05%** | **1.111 deg** |
| profile^3 | 8.45% | 1.111 deg |

**Measured nearest-1: 7.41% at 1.112 deg.** The 2nd-harmonic row is the physically right one -
buffer 3 transmits at 1.95 MHz and demodulates at 3.9 MHz, and harmonic amplitude goes roughly as
the square of the fundamental - and it lands within 20% of the measurement, at exactly the right
period, from a model with no free parameters and no phase. **So yes: in a one-transmit-per-pixel
reconstruction the striations are simply the transmit beam profile, and they would be correctable
by a sensitivity map.**

### But that is not the reconstruction we use

Our pipeline sums all 73 transmits per pixel, so there are no strips and no strip edges - the
mechanism above does not apply to it. For the all-73 compound the same phase-free model predicts
**0.37%** against **14.97%** measured: 40x short. The coherent sum is not smoothing the
single-beam non-uniformity away, it is adding an interference term roughly twice as large
(7.41% -> 14.97%).

The measured reconstructions agree: incoherent compounding, which keeps the amplitudes and discards
the phase, gives 1.84% - the same order as the 0.37% prediction and ~8x below the coherent 14.97%.
That ratio is the interference term.

**So the artefact has two contributors and our images are dominated by the wrong one for
normalisation purposes.** Amplitude non-uniformity is real, predictable and correctable, but it is
a minor term in an all-transmit coherent sum. What dominates is cross-transmit phase interference,
which no magnitude map contains the information to undo - and that, not the period argument
withdrawn in the CORRECTION banner, is why every normalisation attempt in S5a/S5b failed.

## 5g. Cost of REFoCUS, and which buffers are affected at all

### Beamforming time

GPU beamform only, C000000001 buffer 3, 26 frames x 73 transmits onto a 382x529 grid, each method
given an untimed one-frame warm-up so kernel autotuning does not land on one stopwatch
(`analysis/timing_refocus.py`):

| method | total | per frame |
|---|---|---|
| standard (`enable_pfield=True`, as the pipeline runs it) | **47.9 s** | 1840 ms |
| REFoCUS adjoint | **50.8 s** | 1955 ms |

**REFoCUS adjoint costs +6%.** It is essentially free because `adjoint` is a matched filter -
one `H^H` multiply - not an inversion. The SVD variants are the expensive ones (S2: ~11x).
Buffer 1 for scale: 90 frames, 21 transmits, 49.8 s standard (554 ms/frame).

Note REFoCUS expands the transmit axis from n_tx to **n_el virtual transmits** (73 -> 80 here), so
the patch budget has to be planned for n_el. Budgeting for n_tx OOMs on a 24 GB card.

### Only buffer 3 has the artefact

Transmit geometry, read from the converted parameters (`polar_angles` is already in the sector-apex
convention - buffer 3's 1.1111 deg matches `CenterTransmit.mat`'s region geometry exactly):

| buffer | transmits | angular span | **pitch** | transmit focus |
|---|---|---|---|---|
| 1 widebeam | 21 | -40..+40 deg | **4.0000 deg** | -123.2 mm (virtual source behind array) |
| 3 focused | 73 | -40..+40 deg | **1.1111 deg** | +78.9 mm |
| 4 diverging | 2 | -6..+6 deg | 12.000 deg | -12.3 mm |

Measured apex-referenced, each buffer at its own pitch, with the same function that produced the
15.41% figure - and with REFoCUS as a **null test**, since REFoCUS removes per-transmit structure
and should therefore suppress a genuine lattice artefact and leave image content alone:

| | @ its pitch | @ 2x pitch | standard -> REFoCUS @pitch |
|---|---|---|---|
| **buffer 3 focused** | **15.55%** | 2.46% | **15.55% -> 3.82%** (4x down) |
| buffer 1 widebeam | 4.08% | 13.47% | 4.08% -> 4.98% (**no reduction**) |
| buffer 4 diverging | n/a - two transmits are not a lattice | | |

**Buffer 1 is clean, and the null test is what proves it.** It does carry angular structure - 13.47%
at 8 deg, twice its transmit pitch - but REFoCUS does not reduce any of it (13.47 -> 15.08% at
8 deg, 4.08 -> 4.98% at 4 deg). A transmit-lattice artefact cannot survive REFoCUS; buffer 3's
drops 4x under the identical test. So buffer 1's 8 deg content is anatomy and speckle at that
angular scale, not a reconstruction artefact. The original "buffer 1 is clean" conclusion in S3
was reached with the broken origin-referenced metric, so it needed re-deriving - but it survives.

Why only buffer 3: the mechanism needs **many overlapping beams on a fine regular lattice**.
Buffer 1's 21 widebeams come from a virtual source 123 mm behind the array, so each beam is broad
and its amplitude varies little across a 4 deg step; buffer 4 has two transmits, and two is not a
lattice. Buffer 3's 73 focused beams at 1.111 deg are the only case with both a fine pitch and
beams narrow enough to vary across it.

**Practical consequence:** REFoCUS is worth considering for buffer 3 alone, where it costs +6%
compute and 34% lateral resolution (S5a) to take the artefact from 15.55% to 3.82%. There is
nothing for it to fix in buffers 1 or 4, and buffer 2 (active tracking) and 4 feed the displacement
estimators, whose phase must not be touched.

## 6. Reproducing

| script | what |
|---|---|
| `scripts/incoherent_bmode.py <folder> --buffer 3` | envelope-compounded reconstruction + GIF, written alongside the normal output |
| `analysis/phantom_psf.py` (working folder) | point-target PSF/CNR + angular ripple per reconstruction |
| `analysis/c1_field_correct.py` (working folder) | builds both field maps on any folder's grid and applies them (S5b) |
| `analysis/frame_montage.py` (working folder) | tiles one frame from N GIFs into a still PNG - the fastest way to re-judge striations |
| `analysis/method_scorecard.py` (working folder) | all 8 reconstructions x both datasets on one consistent metric set (S5c) |
| `analysis/center_transmit_vs_region.py` (working folder) | beam width vs region width from `CenterTransmit.mat` (S5d) |
| `analysis/region_truncation_test.py` (working folder) | compounds the simulated beam over 73 angles, with/without region truncation (S5d) |
| `analysis/apex_referenced_ripple.py` (working folder) | **the corrected ripple metric - use this one** |
| `analysis/nearest_k_composite.py` / `nearest_k_apex.py` (working folder) | per-transmit stack + nearest-k ladder (S5e) |
| `analysis/ripple_metric_control.py` (working folder) | synthetic-speckle control for the ripple metric |
| `analysis/mosaic_prediction.py` (working folder) | predicts the nearest-1 ripple from the simulated beam alone (S5f) |
| `analysis/timing_refocus.py` (working folder) | standard vs REFoCUS beamforming time (S5g) |
| `analysis/buffer_lattice_ripple.py` (working folder) | per-buffer lattice ripple with REFoCUS as a null test (S5g) |

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
