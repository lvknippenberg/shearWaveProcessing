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

## 4. Current hypothesis

**The focused sector scan is angularly under-sampled near its focus, so the compounded transmit
field scallops at the line spacing.**

This fits every surviving observation:

* the periodicity is exactly `rayDelta` (1.111°), not some other scale;
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

### Open question

Whether the residual is genuinely transmit-field scalloping or interference intrinsic to
coherently summing transmits sampled at 1.111°. Both predict line-spacing periodicity; they
differ in how the amplitude should scale with focal depth and F-number.

**Next step: resolution-phantom data** — a uniform scattering medium removes anatomy and
shadowing, so the residual angular ripple can be measured against a known-flat target, and its
depth profile compared against the predicted beam-width curve.

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
