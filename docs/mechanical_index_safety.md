# Mechanical index & acoustic safety: transmit voltage / elements vs MI

What MI to expect for our S5-1 ARF push, how it scales with transmit voltage and aperture, and the
publicly available anchor values. Written 2026-08-10. **Bottom line: our current 30 V push is estimated
at MI ≈ 0.4–0.8 — well under the FDA 1.9 cap and *low*, consistent with the underpowered-push finding;
there is large headroom to strengthen it (more elements + higher voltage) before MI 1.9.**

> **Update 2026-08-10 — direct hydrophone measurements now exist and *confirm* the estimate, after a
> calibration fix.** Our own HGL-0400 measurements had over-read MI by ~2× (intensities ~4×) because the
> preamp output was measured into a **high-Z scope** while the calibration is referenced to **50 Ω**
> (§8). After correcting, the **S5-1 push at 30 V measures MI ≈ 0.69** at the (past-focus) probe point —
> right in the 0.4–0.8 estimate band. The corrected analysis code lives in the acquisition repo,
> **`SWI/Mechanical index/`** (`CalculateSafety.m`). See
> **§8** (measurements + fix) and **§9** (should we raise the imaging / passive-SWE voltage?).

## 1. Definition and regulatory limits

```
MI = p_r.3 / sqrt(f_c)      (p_r.3 = derated peak rarefactional pressure [MPa]; f_c [MHz])
```

- `p_r.3` = **peak rarefactional (negative) pressure**, derated at 0.3 dB/cm/MHz, at the location of
  maximum `p_r.3`.
- **FDA limit: MI ≤ 1.9** (Track 3 diagnostic); IEC 62359 defines the measurement. Research above 1.9
  needs IRB/ethics cover, not an FDA clearance.
- MI = **mechanical/cavitation** risk only. Our pushes are long (667–800 µs), so the **thermal indices
  (TIS/TIB/TIC) and I_spta** are the *other* limit and are usually the binding one for ARFI — check them
  too, not just MI.

## 2. Verasonics voltage convention

The Verasonics "transmit voltage" (TPC-profile HV) is the **amplitude** of the bipolar drive, so each
element sees **±V = 2V peak-to-peak**. So **30 V push = 60 V peak-to-peak** at the element (as observed).
When comparing to systems/papers that quote Vpp, halve their number. Piezo surface pressure is ~linear in
drive amplitude, so **pressure ∝ V** in the linear regime (nonlinear saturation flattens the top end).
The S5-1 element HV limit is typically ~50 V on a standard TPC.

## 3. Scaling rules

In the linear regime, at fixed focal geometry:

```
MI  ∝  V  ×  N  ×  1/sqrt(f_c)          (focused push)
```

plus a strong **F-number** dependence (looser focus / higher F# → sharply lower focal pressure) and
**nonlinear saturation** at high focal pressure (so linear extrapolation *overestimates* MI at the top —
conservative for safety).

| Transmit type | MI vs **voltage** | MI vs **# elements (aperture)** |
|---|---|---|
| **Focused** (ARF push) | ∝ V, flattening near saturation | **∝ N** (coherent focal gain) until saturation |
| **Unfocused — plane wave** | ∝ V | **~independent of N** (on-axis peak ≈ 2× element face pressure; more elements push the last near-field max *deeper*, not higher) |
| **Unfocused — diverging** (phased-array cardiac B-mode) | ∝ V | ~independent of N, and *lower* than a plane wave |

**Pulse length does NOT change MI** (peak pressure unchanged) — it raises thermal / I_spta instead, so a
longer push is "free" on MI. This is why aperture (not pulse) is the dominant *clarity* lever in our
phantom sweep, and why unfocused imaging transmits stay at low MI regardless of element count.

## 4. Publicly available anchor values (measured)

| Source | Probe / mode | f | Aperture | Focal depth | Voltage | **MI** | I_spta.3 |
|---|---|---|---|---|---|---|---|
| Bouchard/Vos harmonic SWE (PMC3947393) | **P4-2** push | 2 MHz | **all 64 el** | 45 mm | (not reported) | **1.60** | 226 mW/cm² |
| " | P4-2 | 2 | 64 | 60 mm | — | **1.11** | 99 |
| " | P4-2 | 2 | 64 | 65 mm | — | **1.00** | 80 |
| " | P4-2 | 2 | 64 | 70 mm | — | **0.90** | 63 |
| Caenen 2023 (Sci Rep 13:17660) | P4-2 cardiac | ~2 | full, F/1 | ~60 mm | **50–60 V** | **2.2** (research, >1.9) | Isppa.3 403 W/cm² |
| Deng 2021 (PMC8290933) | L7-4/L11-4v/L12-5 ARFI | 5–8 | 128–256 | 38–58 mm | — | ~0.5–1.2 (p_r 1.5–2.65 MPa) | — |
| Typical diagnostic B-mode / plane-wave | various | 2–8 | — | — | low | 0.3–1.3 | — |

Notes: the P4-2 study **ran near the limit** — it had to cut push PWM to **27 %** at shallow depths
(<45 mm) to stay under MI 1.9. **No study publishes a universal voltage→MI table**: MI and I_spta (the
*regulated* quantities) are reported directly, because the voltage→MI mapping is probe/geometry-specific.

## 5. Estimate for our sequence at 30 V

Our base push: **S5-1, 2.25 MHz, 41 elements, F/≈4.3, focus ~45 mm, 30 V**. Two independent anchorings:

- From **Caenen** (P4-2 full aperture, ~55 V, MI 2.2 @ 60 mm): scale by elements (41/64), voltage (30/55),
  √(2/2.25) → **≈ 0.7**.
- From the **P4-2 under-limit paper** (MI 1.6 @ 45 mm, full aperture, likely near probe max ~75–90 V):
  scale by elements and 30/~80 V → **≈ 0.35**.

→ **Best estimate: MI ≈ 0.4–0.8 at 30 V (central ~0.5–0.6)** — well under 1.9, and *low*, which matches
our finding that this push is underpowered (weak ARF, swamped by cardiac motion). The **loose F/4.3 focus
at 41 elements is the main reason it's low**; a full-aperture tight push at the same 30 V would be several
times higher.

## 6. Headroom — MI vs (elements, voltage)

Anchoring MI(41 el, 30 V) ≈ 0.55 and scaling ∝ N·V (conservative — ignores saturation, which lowers the
top-right; and holds focal geometry fixed):

| | 30 V | 40 V | 50 V |
|---|---|---|---|
| **41 el** | ~0.55 | ~0.73 | ~0.92 |
| **61 el** | ~0.82 | ~1.09 | ~1.37 |
| **79 el** | ~1.06 | ~1.42 | ~1.77 |

So the recommended **61–79 elements at 40–50 V → MI ~1.1–1.8**: roughly **2–3× stronger ARF while still
under 1.9**. That is the quantitative headroom to fix the underpowered in-vivo push (see
`docs/phantom_parameter_sweep.md` for the clarity-per-MI trade-off → recommend 61 el + ~800 µs pulse).
**Thermal / I_spta will likely bind before MI** for the long push — check TIS/TIB.

## 7. How to get an absolute number

There is **no S5-1-specific published voltage→MI curve** (Philips measures it per-preset for FDA
clearance; it is not in datasheets — the S5-1 is a 1.3–3.2 MHz PureWave sector array). For a trustworthy
absolute MI of *our* custom sequence: **one hydrophone measurement** at a known (V, N), then scale with
§3: `MI ≈ MI_ref · (V/V_ref) · (N/N_ref) · sqrt(f_ref/f)` (until saturation). The estimates here are
**±~50 %** regime guides, not a substitute for calibration.

## 8. Direct hydrophone measurements + calibration correction (2026-08-10)

We measured our own transmit sequences (S5-1, plus L11-5 / L12-3 for method validation) with an **Onda
HGL-0400** hydrophone + **AH-2010-025** preamp on a digital oscilloscope, and converted V→MI in
`CalculateSafety.m` (in the acquisition repo, `SWI/Mechanical index/`). Initial results were
**2–3× above** the Verasonics/PULS-e reference
values. Root cause found (two issues; both now fixed in the code):

1. **Scope input impedance (factor ~2 on MI, ~4 on intensities).** The AH-2010 has a **50 Ω output
   impedance**, and its 20 dB gain — hence the Onda combined system sensitivity `M_L` — is referenced to
   a **50 Ω load**. A 50 Ω source into a **high-Z (1 MΩ)** scope delivers ~2× the voltage it delivers
   into 50 Ω (a resistive divider), so every pressure/MI came out ~2× high. *Fingerprint:* the AH-2010
   clips at 4 Vpp (2 V peak) into 50 Ω, yet our "saturation" ceiling was ~4.8 V peak → we were on high-Z.
   (That ~4.8 V ceiling is the **preamp** clipping, not the hydrophone.)
2. **Method.** The hydrophone certificate is **open-circuit** (`ELECTRICAL_LOAD OpenCircuit`), so Onda's
   loaded-sensitivity formula (`HydroCalMethod` Eq. 2a) *requires* the capacitive-divider term
   `C_H/(C_H+C_A)≈0.667` → our **`ParallelCircuit`** method is correct; the old `Simple` method dropped it
   and under-reads 1.5×. **Use `ParallelCircuit`.**

**Corrected MI (`ParallelCircuit`, high-Z / ÷2, derated).** `MI old` = same but without the ÷2 (what we
had reported before):

| Sequence (measured point) | f | depth | 20 V | 30 V | 40 V | MI old @30 V |
|---|---|---|---|---|---|---|
| S5-1 **push** (@59 mm, *past* the 40 mm focus) | 2.25 MHz | 5.9 cm | 0.55 | **0.69** | (sat.) | 1.39 |
| S5-1 **widebeam PI** (imaging/tracking, @40 mm) | 1.9531 MHz | 4.0 cm | 0.48 | **0.71** | 0.91 | 1.41 |
| L11-5 **focused PI** (validation, @15 mm) | 4.4643 MHz | 1.5 cm | 0.59 | **0.63*** | — | 1.25 |

\* L11-5 20–30 V points are **at/near preamp saturation** (Vmin pinned ≈4.45 V), so those MI are soft.

**Validation.** (a) The corrected S5-1 push (0.69 at the past-focus point; higher at the 40 mm focus, see
the 30 V MI map) lands in the independent **0.4–0.8 literature estimate** of §5 — the estimate and the
measurement now agree. (b) L11-5 focused corrected ≈0.63 vs the Verasonics reference **0.40** at 30 V:
the correction cut the gap from ~2.5–3× to **~1.5×**, the residual being cross-lab spread (Onda cal
±1 dB, our saturation extrapolation, 400 µm hydrophone spatial-averaging/directivity — none applied —
and a different probe/system than Verasonics).

**Code (acquisition repo `SWI/Mechanical index/`; see its `README.md`).** `CalculateSafety.m` now takes a
`ScopeImpedance_Ohm` arg (÷2 at 1 MΩ, default 1e6 + warning) and defaults `MI_method="ParallelCircuit"`.
To rescale old numbers without
re-running: **Parallel-method MI ÷2** (unsaturated points), Simple ×0.75, **intensities ÷4**; saturated
points need re-measurement/extrapolation, not ÷2.

## 9. Should we raise the imaging / passive-SWE transmit voltage?

Short answer: **likely yes for SNR, but MI is not the limit that matters here — recompute I_spta.3 and
probe heating with the corrected code first.**

- **MI has huge headroom.** The imaging/tracking transmit (widebeam PI, and the Buffer-4 diverging waves
  used for **passive** SWE) is at **MI ≈ 0.5–0.7 at 20–30 V** (corrected) vs the 1.9 cap. And because the
  old numbers were ~2×/4× high, the *real* margin on every index is larger than we thought.
- **The binding limit for imaging/passive is I_spta.3 and thermal, not MI.** Passive/tracking runs
  **continuously at high PRF for ~1 s per acquisition over many cycles**, so the *temporal-average*
  intensity **I_spta.3** (cardiac limit **430 mW/cm²**) and **probe surface heating** (TIS/TIC) bind
  first — and both scale with **V²** (I_spta.3 ∝ pii·PRF ∝ V²·PRF). So doubling voltage ≈ 4× I_spta.3.
  MI (∝ V) is the *easy* index; the intensity/thermal budget is the real gate.
- **Why raising it helps passive SWE specifically.** Passive detects small *natural*-wave displacements →
  it is **SNR-limited** (the same displacement-SNR wall that dominates the whole project). Higher tracking
  voltage → higher echo SNR → cleaner displacement → cleaner passive wavefronts. Same benefit for active
  *tracking* (distinct from the push strength).

**Recommendation / how to do it safely:**
1. **Recompute I_spta.3 and I_sppa.3 for the imaging + passive (Buffer-4) transmits** with the corrected
   `CalculateSafety.m` at the candidate voltages, at the real PRFs (tracking 3704 Hz, passive Buffer-4
   ≈925.9 Hz) and the 1 s continuous duration. Check against **430 mW/cm² (cardiac I_spta.3)** and
   190 W/cm² (I_sppa.3).
2. **Check probe heating** (TIS/TIC or a surface-temperature measurement, as in the PULS-e C5-2 report) —
   continuous high-PRF transmit is where the thermal limit actually bites, not the short push.
3. If both clear, **raise the tracking/imaging voltage** as an SNR lever for active tracking *and*
   passive; expect displacement-SNR (not MI) to be the payoff.
4. **Mind saturation when re-measuring:** the preamp clips at ~2 V peak into 50 Ω (~4 V into 1 MΩ), so at
   higher transmit voltages either add an inline **attenuator** (Onda `CombineCal` supports it) or
   extrapolate from the **low-voltage linear region**. Don't trust points near the ceiling.

## 10. TODO / next steps (safety)

1. **Confirm the ~2.0 impedance factor experimentally.** One unsaturated signal measured at **1 MΩ** then
   with a **50 Ω feed-through** → expect a clean 2:1 ratio. Turns "assume 2.00" into a measured number;
   set `Rout`/`Rref` (or pass the measured `ScopeImpedance_Ohm`) if it differs.
2. **Finish the corrected safety table (user is redoing the Excel).** Needs the two missing transmit
   frequencies: **L12-3 widebeam** and **L11-5 widebeam PI** (only L11-5 *focused* = 4.4643 MHz is on the
   sheet). Then report corrected MI + I_sppa.3 + I_spta.3 for all sequences.
3. **Re-anchor the push element/voltage headroom (§6) to the corrected *focal* MI** (from the 30 V MI
   map, not the past-focus 59 mm point). The §6 table was scaled from the 0.55 *estimate*; the measured
   focal MI may be a bit higher (measured 0.69 already at a past-focus point), so the 79 el / 50 V corner
   could reach ~1.9 sooner than the table implies — recompute before committing to a stronger push.
4. **I_spta.3 + thermal budget for imaging/passive** before raising voltage (§9).
5. **Optional: hydrophone spatial-averaging (directivity) correction** — the largest remaining source of
   the ~1.5× validation residual; a 400 µm tip under-reads focal peaks (Deng 2021 method).

## Sources

- Bouchard/Vos et al., *Improved Shear Wave Motion Detection Using Pulse-Inversion Harmonic Imaging with a
  Phased Array Transducer* — https://pmc.ncbi.nlm.nih.gov/articles/PMC3947393/ (P4-2 MI vs depth table)
- Deng et al. 2021, *Hydrophone Spatial Averaging Correction… ARFI and Pulsed Doppler Waveforms* —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8290933/ (linear-array ARFI pressures)
- Deng et al. 2017, *Ultrasonic Shear Wave Elasticity Imaging Sequencing and Data Processing Using a
  Verasonics Research Scanner*, IEEE UFFC —
  https://verasonics.com/wp-content/uploads/2017/10/Deng-2017-UFFC_Ultrasonic-Shear-Wave-Elasticity-Imaging-Sequencing-and-Data-Processing-Using-a-Verasonics-Research-Scanner.pdf
- Caenen et al., *Sci. Rep.* 2023;13:17660 (P4-2 cardiac SWE, MI 2.2 at 50–60 V) — see
  `docs/caenen_vs_invivo_acquisition.md`
- Philips S5-1 transducer (1.3–3.2 MHz sector array) —
  https://www.usa.philips.com/healthcare/product/HC989605412081/s5-1-broadband-sector-array-transducer
