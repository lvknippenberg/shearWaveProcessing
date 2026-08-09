# Acquisition comparison: our in-vivo (Luuk 40 V) vs the Caenen pig ARF-SWE data

Why the Caenen pig data shows a clean ~2 m/s shear wave at nearly every push while our 40 V in-vivo
human data is cardiac-motion-dominated. Parameters are read directly from the data (our
`AcquisitionParametersAndECG.mat`/`CombinedData.mat` `SW`/`Trans` structs; Caenen `..._1_info.mat` `P`
struct + exported params) and from their paper: **Caenen et al., "Continuous shear wave measurements for
dynamic cardiac stiffness evaluation in pigs", Sci Rep 2023, 13:17660** (+ the dataset `Readme.txt`).

## Side-by-side

| Parameter | **Our in-vivo (Luuk 40 V)** | **Caenen (example file 20200616)** | Caenen paper (representative) |
|---|---|---|---|
| Subject / view | Human (Luuk), transthoracic | Pig, transthoracic **PLAX** | 13 pigs, transthoracic PLAX, ECG |
| Scanner / probe | Verasonics, **S5-1** phased (80 el, pitch 0.254 mm, BW 1–6 MHz) | Verasonics Vantage 256, **P4-2** phased (128 el) | P4-2 / P4-2V |
| Imaging mode | 2nd-harmonic (NS200BW), 1 angle | **Pulse-inversion** harmonic, 1 angle | PI harmonic, sliding window |
| Demod / center freq | 3.906 MHz (2× 1.95) | 4.032 MHz (2× 2.02); opFreq 3.05 MHz | 2 MHz tx |
| **Push center freq** | 2.25 MHz | 2.0 MHz | 2 MHz |
| **Push duration** | 1500 cyc → **667 µs** | 1600 cyc → **800 µs** | 800 µs |
| **Push F-number** | ≈ **4.3** (41 el → 10.4 mm aperture @ 44.6 mm) | **1** (`push_Fnum = 1`) | tight focus |
| **Push focal depth** | **44.6 mm** (`FocusZ`), off-axis x = 5.1 mm | **25.3 mm** (mid-septal), on-axis x = 0 | mid-septal wall (≈ 60 mm in the MI example) |
| **Push voltage / MI** | ~40 V (delivered; probe max 50 V) | 50 V | 50–60 V, **MI = 2.2**, Isppa.3 = 403 W/cm² |
| **Tracking PRF** | **3 704 Hz** (PRI 270 µs) | **8 774 Hz** (PRI 114 µs) | **≥ 5.6 kHz**, diverging-wave |
| Reference frames | ~39 (~10.5 ms pre-push) | 82 compounded (41 × PI) | 20 (ultrafast) |
| Tracking frames | 59 (15.7 ms) | 164 (18.6 ms) | — |
| Axial pixel dz | 394 µm (~1 λ) | 253 µm (0.5 λ) | — |
| **Cardiac sync** | **`WaitForRpeak = 0` — free-running**, 20 fps, 24 pushes; ECG recorded | **R-peak-triggered**, push-rep 34 Hz, 52 pushes/cycle; ECG | **R-peak-triggered**; ECG on oscilloscope |
| Motion filter | band-pass ~80–500 Hz (this sweep) | 6th-order **75–750 Hz** Butterworth | 75–750 Hz; wave content 50–500 Hz |
| Displacement estimator | **Loupas** (autocorr) | (paper: **Kasai**) | Kasai |

## Interpretation — via the three feasibility factors the paper itself names

The paper states SWE feasibility depends on **(i) the amplitude of the induced vibration, (ii) the image
quality in which it is tracked, and (iii) the tracking algorithm's ability to cope with tissue motion** —
and that transthoracic cardiac SWE is hard because the phased-array footprint limits excitation strength
and image quality, and because fast wall motion challenges both tracking and push settings. Even Caenen
only reached a **54 % success rate (7/13 pigs)**.

**(i) Vibration amplitude — the dominant gap (acquisition-limited).** Caenen's push is far more
concentrated: **F/1 vs our ≈ F/4.3**, at **25.3 mm vs our 44.6 mm**, at **50–60 V vs our 40 V**. A
tighter F-number concentrates the beam energy (ARF ∝ intensity), a shallower focus suffers less
attenuation, and higher voltage adds power — so Caenen displaces the wall by many microns while our
loose, deep, lower-voltage push produces a much smaller vibration. Our small S5-1 footprint (80 el,
10.4 mm push aperture) simply cannot form an F/1 push at 44.6 mm. This alone can put the shear wave
below the ~20–40 µm of cardiac motion — matching our finding that the in-vivo plot is bulk motion.

**(ii) Image / tracking quality.** Caenen images the wall at **25 mm** with **0.5 λ axial sampling**;
we track at **~45–50 mm** with **1 λ sampling**. Depth costs round-trip attenuation and SNR, so even an
equal push would be tracked more noisily in our data.

**(iii) Motion handling.** Caenen tracks at **8.8 kHz vs our 3.7 kHz** — 2.4× finer temporal sampling of
both the wave and the confounding wall motion — and **triggers on the R-peak** so every push has a known
cardiac phase (and can be placed to sweep the cycle deliberately). Our sequence is **free-running**
(`WaitForRpeak = 0`), so pushes land at arbitrary phases, many during fast systolic/early-diastolic
motion. Notably, our *processing* is not the weak point here: we use **Loupas**, which the paper
explicitly lists as a desired upgrade over their **Kasai** estimator.

## Bottom line

Our in-vivo failure is not one thing — it is **acquisition-limited first** (an underpowered, deep,
loose-F/number, ungated push on a small-footprint probe, tracked at half the frame rate and twice the
depth), which then leaves the true ARF vibration buried under in-band cardiac motion that our current
filter does not remove. The Caenen data succeeds because its **push is much stronger and shallower, the
frame rate higher, and the acquisition ECG-referenced** — and even then only ~half their pigs worked.

Concrete implications, in priority order:
1. **Strengthen and shallow the push:** lowest achievable F-number, focus on the nearest wall segment,
   push voltage to the MI limit — the single biggest lever (matches the paper's "new transducer with a
   better mechanical focus for shear-wave excitation").
2. **Raise the tracking PRF** toward ≥ 5.6 kHz (fewer tracking samples per frame / diverging-wave).
3. **ECG-gate / trigger on the R-peak** and select diastasis (quiet-phase) pushes.
4. **Then** apply the reference-trained cardiac-motion filter (our next-step-2) to remove residual
   in-band wall motion — the paper's own recommended "optimization of filters to suppress background
   motion". Only after (1)–(4) can we judge whether any residual SNR limit remains.

Our processing (Loupas + the speed-scan / per-lobe detectors) already **works on a real cardiac ARF wave**
(the Caenen sweep), so the in-vivo gap is an acquisition + motion-suppression problem, not a pipeline one.

*Sources: data structs above; Caenen et al., Sci Rep 2023, 13:17660 (methods: R-peak trigger, 20 ref
frames, 2 MHz/800 µs push to mid-septal wall, ≥5.6 kHz diverging-wave tracking, 75–750 Hz filter,
MI 2.2 @ 60 mm/60 V, 54 % success); dataset Readme (pig, PLAX, push-rep 34 Hz, PI harmonic).*
