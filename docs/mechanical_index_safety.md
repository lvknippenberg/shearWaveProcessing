# Mechanical index & acoustic safety: transmit voltage / elements vs MI

What MI to expect for our S5-1 ARF push, how it scales with transmit voltage and aperture, and the
publicly available anchor values. Written 2026-08-10. **Bottom line: our current 30 V push is estimated
at MI ≈ 0.4–0.8 — well under the FDA 1.9 cap and *low*, consistent with the underpowered-push finding;
there is large headroom to strengthen it (more elements + higher voltage) before MI 1.9.**

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
