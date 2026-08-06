# Literature review — IQ/RF → shear-wave space-time processing

A short, practically-oriented review of the processing choices exposed in the method-exploration
GUI (`swp_gui/`), organised by pipeline stage. Each stage lists the methods, what the literature
says, and how it maps to the code. Written 2026-08 for the cardiac ARF-SWE pipeline; the driving
question is recovering the (few-µm) ARF shear wave from under (tens-of-µm) cardiac wall motion —
see the finding that the in-vivo active plots were dominated by cardiac motion, not the push.

## 0. Context: ARF vs natural cardiac shear waves

Cardiac SWE either **actively** induces a wave with a focused acoustic-radiation-force (ARF) push
and tracks it at high frame rate, or reads the **natural** waves from valve closure. A direct
comparison found natural SWE more feasible in the heart, with ARF waves easily masked by physiological
motion — consistent with our in-vivo finding.
- Strachinaru, Bosch, Vos et al., *A direct comparison of natural and acoustic-radiation-force-induced
  cardiac mechanical waves*, Sci. Rep. 2020. https://www.nature.com/articles/s41598-020-75401-1
- Vos et al. / Petrescu et al., *Ultrasound Shear Wave Elastography in Cardiology*, JACC: Cardiovasc.
  Imaging 2024. https://www.jacc.org/doi/10.1016/j.jcmg.2023.12.007

## 1. IQ/RF pre-filtering (before displacement)  — GUI stage 1

- **Spatial / slow-time low-pass** — mild speckle / electronic-noise denoise before tracking; keep
  axial smoothing small to avoid biasing phase estimates. (`iq_spatial_lowpass`, `iq_slowtime_lowpass`.)
- **SVD spatiotemporal clutter filter** — remove the dominant low-rank slow-time subspace (bulk tissue)
  from the IQ ensemble; outperforms temporal high-pass for flow/wave discrimination and is the standard
  ultrafast-Doppler clutter filter. (`svd_clutter`.)
  - Demené et al., *Spatiotemporal Clutter Filtering of Ultrafast Ultrasound Data Highly Increases
    Doppler and fUltrasound Sensitivity*, IEEE TMI 2015. https://pubmed.ncbi.nlm.nih.gov/25955583/
  - Adaptive/localized SVD variants threshold singular values by spatial coherence.

## 2. Displacement estimation  — GUI stage 2

- **Loupas 2-D autocorrelation** — phase-based; ~as accurate as normalized cross-correlation (NCC) but
  much faster; degrades for broadband signals and axial windows < ~0.5 λ. (`loupas`.)
- **Kasai 1-D autocorrelation** — classic phase-shift estimator at a fixed demod frequency. (`kasai`.)
- **Cross-correlation (NCC) on RF** — the precision benchmark; robust to large displacement (no phase
  wrapping) and to broadband speckle, but needs many samples per wavelength → a **fine axial grid**.
  Hybrid *autocorrelation-guided cross-correlation* (coarse phase → narrow NCC search) is best-of-both.
  (`xcorr` on complex IQ; `rf_ncc` on the fine-grid RF from `swp.acquisition.finegrid`.)
  - Pinton, Dahl, Trahey, *Rapid tracking of small displacements with ultrasound*, IEEE UFFC 2006.
    https://pubmed.ncbi.nlm.nih.gov/16846143/
  - *Autocorrelation-guided cross-correlation in ultrasound SWE*, WO2015173709.
    https://patents.google.com/patent/WO2015173709A1/en
- **Quantity: displacement / velocity / acceleration** — the choice changes the estimated speed;
  acceleration is least biased but noisiest. (`quantity`.)
  - Petrescu et al., *Comparing Myocardial Shear Wave Propagation Velocity Estimation Methods Based on
    Tissue Displacement, Velocity and Acceleration Data*, Ultrasound Med. Biol. 2022.
    https://www.sciencedirect.com/science/article/abs/pii/S0301562922004367

**Which Loupas parameters are tunable, and which are fixed by the data.** Free tuning knobs (in the
GUI): axial kernel length, lateral kernel width, local-frequency correction on/off, and adaptive
kernel. Fixed by the Verasonics acquisition and shown read-only (must match the data, *not* tuned):
centre/transmit frequency `f0`, demodulation frequency (phase→displacement scaling), PRF, speed of
sound `c`, and the axial pitch `dz` (which encodes the effective sampling rate `Fs ≈ c/2/dz`).
- *Complex-correlation weighting* is already built into Loupas: taking the angle of the spatially-
  averaged **complex** correlation weights each sample by its magnitude, so high-SNR samples dominate —
  no separate knob needed.
- *Adaptive kernel from local RF coherence* (**yes, implemented**): size the axial window per pixel by
  the local coherence `C = |⟨P⟩| / ⟨|P|⟩` of the slow-time correlation `P` — small window where coherent
  (preserve resolution), larger where noisy (reduce variance). The GUI blends the small- and large-kernel
  displacements by `C`. (`loupas(..., adaptive_kernel=True)`.)
- *Sub-sample / interpolation* is intrinsic to phase estimators (Loupas/Kasai are continuous); it matters
  for the RF-NCC estimator (parabolic peak interpolation), which is where it is applied.

## 3. Cardiac-motion removal  — GUI stage 3 (the crux)

Fixed band-pass is the baseline but a poor fit when bulk motion is large and wraps the phase estimator.
Options in the GUI, roughly increasing in how much they *measure* rather than *assume* the motion:
- **Temporal band-pass / high-pass** — assume the wave lives above a corner; simple, but a fixed corner
  either passes bulk motion (low) or attenuates slow diastolic waves (high). (`temporal_bandpass`,
  `temporal_highpass`.)
- **Polynomial detrend** — subtract a per-pixel low-order slow-time polynomial; no cutoff assumption but
  still model-based. (`polynomial_drift`.)
- **SVD clutter on the displacement movie** — the Demené filter applied to displacement; removes the
  coherent whole-wall subspace directly. (`svd_clutter_field`.)
- **Reference (pre-push) polynomial extrapolation** — fit cardiac motion on the reference frames (pure
  motion, no wave), extrapolate onto the tracking window, subtract. *Measures* the motion; the fit never
  sees the wave. Giannantonio-style. (`reference_motion_comp`, with a velocity-anchored linear variant.)
- **Reference-adaptive high-pass** — per-pixel corner scales with the reference cardiac-motion strength;
  a reference-informed answer to the fixed-cutoff problem. (`adaptive_highpass`.)
- **Reference-subspace projection [novel]** — build the dominant SVD subspace of the *reference* motion
  and project it out of the tracking field; the ARF wave (absent from the reference) is orthogonal to it.
  A data-driven, reference-trained clutter filter — the strongest realisation of "use the reference to
  estimate cardiac motion better than a band-pass." (`reference_subspace_projection`.)
- **Phase unwrap (temporal)** — large motion wraps phase estimators; unwrapping along slow time restores
  continuous trajectories so a motion filter can actually remove them. (`phase_unwrap_temporal`.)
- **Axial strain (curl-like)** — d/dz of displacement is invariant to spatially-uniform bulk translation;
  a construction-by-design rejection of bulk motion (at the cost of amplifying noise). (`axial_strain`.)
- **Bulk-motion compensation** — estimate the *global* per-frame tissue translation (axial+lateral) by
  sub-pixel phase cross-correlation to the reference and undo it on the IQ *before* displacement
  estimation. The simplest global-rigid model; affine / optical-flow background estimation are the
  natural extensions. An IQ-space step. (`bulk_motion_compensation`.)
- **Quality masking** — down-weight (soft confidence weight) or remove pixels with low B-mode intensity
  or poor slow-time coherence, where the phase estimate is unstable — often more valuable than extra
  filtering, and the basis of a confidence map. (`quality_mask`.)

## 4–5. Spatial & temporal smoothing  — GUI stages 4–5

Spatial: Gaussian, edge-preserving **median**, **bilateral** (weights by spatial + value similarity),
**non-local means** (averages similar patches — good for speckle texture), **Perona–Malik anisotropic
diffusion** (diffuse within, not across, wavefront edges), and **coherence-enhancing / tensor-guided
diffusion** (Weickert — smooth *along* the local wavefront orientation, sharpening coherent lines).
(`spatial_smooth`, `spatial_median`, `bilateral_denoise`, `nlm_denoise`, `aniso_diffusion`,
`coherence_diffusion`.)
Temporal: moving **mean**, outlier-robust moving **median**, and **Savitzky–Golay** (sliding polynomial;
suppresses jitter while preserving the transient's timing/amplitude better than a moving average).
(`temporal_moving_mean/median`, `savgol_temporal`.) Keep kernels small to avoid blurring the wavefront
(which biases the speed).

## 6. M-line offset averaging  — GUI stage 6

Sample several parallel copies of the M-line offset along its normal and combine (mean/median) — a cheap,
wavefront-preserving denoiser tailored to the anatomy. (`sample_along_mline`, `mline_offsets/agg`.)

## 7. Directional filtering  — GUI stage 7

Decompose the (k, ω) wavefield to keep a single propagation direction — separates outward vs inward /
left vs right waves, suppressing reflections, standing waves, and (importantly here) **non-propagating
spatially-uniform bulk-motion bands**. (`outward_spacetime`, `directional_spacetime`.)
- Manduca et al., *Spatio-temporal directional filtering for improved inversion of MR elastography
  images*, Med. Image Anal. 2003. https://pubmed.ncbi.nlm.nih.gov/14561551/
- Ultrasound SWE adaptations separate LR/RL waves before speed inversion.

## 8. Speed estimation & quality  — GUI stage 8

- **Time-of-flight + RANSAC** — robust line fit of arrival time vs distance. (`ttp_ransac`.)
- **Slant-stack / Radon-sum** — search trajectories, no peak-picking; robust to outliers; our variant
  removes the per-time spatial mean to reject the flat bulk band and returns a **signed** speed.
  (`radon` / `slant_stack_speed`.)
  - McLaughlin, Renzi et al., *Robust estimation of time-of-flight shear wave speed using a Radon sum
    transformation*, IEEE UFFC 2010. https://pmc.ncbi.nlm.nih.gov/articles/PMC3412360/
- **Local phase-velocity imaging (LPVI)** — frequency-domain phase velocity, dispersion-aware (not yet in
  the GUI; candidate addition).
  - Kijanka & Urban, *Fast Local Phase Velocity-Based Imaging*, IEEE UFFC 2019.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC7123440/

**Quality metrics** (to judge "did we capture a wave, not motion"): `origin_coherence` (symmetric
outward-from-r0, early-onset), `wavefront_visibility` (outward energy fraction), and the slant-stack
semblance. All are permissive toward coherent bulk motion, which is exactly why the GUI's **no-push
control** (same recipe on the pre-push reference) is the decisive check.

## Candidate additions (not yet implemented)
- True (non-reconstructed) RF beamforming for `rf_ncc` (skip demodulation in the fine beamform).
- f–k (velocity-passband) directional filtering with a tunable phase-speed window.
- Local phase-velocity / two-point phase-difference speed.
- ECG-gated selection of quiet-cardiac-phase pushes.
