"""Declarative registry of the IQ/RF -> space-time pipeline stages, methods, and tunable params.

One source of truth that drives (a) the Streamlit widgets, (b) the ``PipelineConfig`` / ``Step``
objects handed to :func:`swp.viz.pipeline.run_pipeline`, and (c) the "view the code of this step"
panels (each :class:`Method` points at the actual callable, shown via ``inspect.getsource``).

Params carry a ``scale`` so the widget can show friendly units (um, mm) while the function receives
SI: ``function_kwarg = widget_value * scale``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

# real implementations (also used for the code viewer) --------------------------------
from swp.viz.filters.experimental import (iq_spatial_lowpass, iq_slowtime_lowpass,
                                          iq_slowtime_highpass, svd_clutter_field,
                                          reference_subspace_projection,
                                          phase_unwrap_temporal, bulk_motion_compensation,
                                          aniso_diffusion, coherence_diffusion, bilateral_denoise,
                                          nlm_denoise, quality_mask, savgol_temporal)
from swp.viz.filters.clutter import svd_clutter
from swp.viz.filters.spatial import (spatial_smooth, spatial_median, temporal_bandpass,
                                     temporal_moving_mean, temporal_moving_median)
from swp.viz.filters.motion import (polynomial_drift, temporal_highpass,
                                    reference_motion_compensation, adaptive_highpass, axial_strain)
from swp.viz.filters.directional import outward_spacetime, directional_spacetime
from swp.viz.estimators.loupas import loupas_displacement
from swp.viz.estimators.kasai import kasai_displacement
from swp.viz.estimators.xcorr import xcorr_displacement
from swp.viz.estimators.rf_ncc import rf_ncc_displacement
from swp.viz.speed.tof import ttp_ransac_speed, tof_xcorr_speed, slant_stack_speed
from swp.viz.mline.mline import sample_along_mline
# NOTE: swp.acquisition.finegrid is intentionally NOT imported here - it pulls in zea/torch
# (~10 s + CUDA init), which the coarse (default) path never needs. It is imported lazily in
# core.load_acq only when the fine grid / RF cross-correlation is actually used.


@dataclass
class Param:
    arg: str                       # function/Step kwarg name (SI)
    label: str
    kind: str                      # "float" | "int" | "bool" | "select"
    default: object
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.1
    scale: float = 1.0             # function value = widget value * scale
    options: Optional[list] = None
    help: str = ""


@dataclass
class Method:
    name: str                      # pipeline step / estimator name; "none" => stage skipped
    label: str
    fn: Optional[Callable]         # for the code viewer
    params: List[Param] = field(default_factory=list)
    ref: bool = False              # needs the pre-push reference frames
    help: str = ""


NONE = Method("none", "none (skip)", None, [])

# ----------------------------------------------------------------------------- Stage 1
IQ_METHODS: List[Method] = [
    NONE,
    Method("iq_spatial_lowpass", "spatial low-pass (Gaussian)", iq_spatial_lowpass, [
        Param("sigma_z_m", "sigma z (um)", "float", 200, 10, 1000, 10, 1e-6),
        Param("sigma_x_m", "sigma x (um)", "float", 400, 10, 2000, 10, 1e-6),
    ], help="Denoise speckle in the IQ before tracking (keep small to avoid biasing the phase)."),
    Method("iq_slowtime_lowpass", "slow-time low-pass (Butterworth)", iq_slowtime_lowpass, [
        Param("fc_hz", "cutoff (Hz)", "float", 900, 100, 1852, 50),
        Param("order", "order", "int", 2, 1, 6, 1),
    ], help="Suppress high-frequency inter-frame noise; keep the corner above the wave frequency."),
    Method("iq_slowtime_highpass", "slow-time high-pass (Butterworth)", iq_slowtime_highpass, [
        Param("fc_hz", "corner (Hz)", "float", 40, 5, 400, 5),
        Param("order", "order", "int", 2, 1, 6, 1),
    ], help="Remove low-frequency slow-time clutter/bulk motion from the IQ before displacement "
            "estimation; keep the corner below the shear-wave band."),
    Method("svd_clutter", "SVD clutter (IQ)", svd_clutter, [
        Param("n_remove", "remove low ranks", "int", 1, 0, 20, 1),
        Param("n_high_remove", "remove high ranks", "int", 0, 0, 20, 1),
    ], help="Demene spatiotemporal clutter filter on the IQ ensemble (removes bulk-tissue subspace)."),
    Method("bulk_motion_compensation", "bulk motion comp (global rigid)", bulk_motion_compensation, [
        Param("max_shift_m", "max shift (mm)", "float", 2.0, 0.2, 5.0, 0.1, 1e-3),
        Param("upsample", "sub-pixel upsample", "int", 10, 1, 50, 1),
    ], help="Estimate & undo global tissue translation per frame vs the reference (phase "
            "correlation) - removes bulk cardiac motion before the axial phase estimator sees it."),
]

# ----------------------------------------------------------------------------- Stage 2
ESTIMATOR_METHODS: List[Method] = [
    Method("loupas", "Loupas (2-D autocorrelation)", loupas_displacement, [
        Param("kernel_z_m", "axial kernel (mm)", "float", 1.0, 0.2, 4.0, 0.1, 1e-3),
        Param("kernel_x_m", "lateral kernel (mm)", "float", 0.0, 0.0, 3.0, 0.1, 1e-3),
        Param("local_frequency", "local centre freq", "bool", True),
        Param("adaptive_kernel", "adaptive kernel (RF coherence)", "bool", False),
        Param("kernel_z_max_m", "adaptive max axial (mm)", "float", 2.5, 0.5, 6.0, 0.1, 1e-3),
    ], help="Phase-based; fast; ~NCC accuracy except very small windows / broadband. Adaptive kernel "
            "grows the axial window where local RF coherence is low."),
    Method("kasai", "Kasai (1-D autocorrelation)", kasai_displacement, [
        Param("kernel_z_m", "axial kernel (mm)", "float", 1.0, 0.2, 4.0, 0.1, 1e-3),
        Param("kernel_x_m", "lateral kernel (mm)", "float", 0.0, 0.0, 3.0, 0.1, 1e-3),
    ], help="Classic phase-shift estimator (fixed demod frequency)."),
    Method("xcorr", "complex-IQ cross-correlation", xcorr_displacement, [
        Param("window_m", "window (mm)", "float", 1.5, 0.5, 4.0, 0.1, 1e-3),
        Param("max_disp_m", "max disp (mm)", "float", 0.3, 0.05, 1.0, 0.05, 1e-3),
    ], help="Envelope NCC for the integer lag + phase sub-sample; benefits from the fine grid."),
    Method("rf_ncc", "RF cross-correlation (fine grid)", rf_ncc_displacement, [
        Param("window_m", "window (mm)", "float", 1.0, 0.3, 3.0, 0.1, 1e-3),
        Param("max_disp_m", "max disp (mm)", "float", 0.2, 0.05, 1.0, 0.05, 1e-3),
    ], help="Windowed NCC on the reconstructed fine-grid RF; robust to large motion (no phase wrap). "
            "Requires the fine grid (auto-enabled)."),
]

# ----------------------------------------------------------------------------- Stage 3
MOTION_METHODS: List[Method] = [
    NONE,
    Method("quality_mask", "quality mask (B-mode / coherence)", quality_mask, [
        Param("min_db", "min B-mode (dB)", "float", -40, -80, 0, 2),
        Param("min_coherence", "min coherence", "float", 0.0, 0.0, 0.95, 0.05),
        Param("soft", "soft (confidence weight)", "bool", True),
    ], help="Down-weight/remove pixels with low B-mode intensity or poor RF coherence (unstable "
            "phase). Put it first in the chain."),
    Method("temporal_bandpass", "temporal band-pass", temporal_bandpass, [
        Param("f_lo", "f low (Hz)", "float", 120, 0, 600, 10),
        Param("f_hi", "f high (Hz)", "float", 700, 100, 1800, 25),
        Param("order", "order", "int", 2, 1, 6, 1),
    ], help="Baseline fixed-band motion rejection."),
    Method("temporal_highpass", "temporal high-pass", temporal_highpass, [
        Param("fc_hz", "corner (Hz)", "float", 80, 5, 400, 5),
        Param("order", "order", "int", 2, 1, 6, 1),
    ]),
    Method("polynomial_drift", "polynomial detrend", polynomial_drift, [
        Param("order", "order", "int", 2, 0, 6, 1),
        Param("fit_frac", "fit fraction", "float", 1.0, 0.1, 1.0, 0.05),
    ], help="Subtract a per-pixel low-order slow-time polynomial (no cutoff assumption)."),
    Method("svd_clutter_field", "SVD clutter (displacement)", svd_clutter_field, [
        Param("n_remove", "remove low ranks", "int", 1, 0, 20, 1),
        Param("n_high_remove", "remove high ranks", "int", 0, 0, 20, 1),
    ], help="Demene clutter filter on the displacement movie (coherent whole-wall motion)."),
    Method("reference_motion_comp", "reference poly-extrapolation", reference_motion_compensation, [
        Param("order", "order", "int", 2, 0, 4, 1),
        Param("use_last_frac", "use last fraction", "float", 1.0, 0.1, 1.0, 0.05),
        Param("anchor", "velocity-anchored", "bool", False),
    ], ref=True, help="Fit cardiac motion on the pre-push reference, extrapolate onto tracking, subtract."),
    Method("adaptive_highpass", "reference adaptive high-pass", adaptive_highpass, [
        Param("base_fc", "base corner (Hz)", "float", 40, 5, 200, 5),
        Param("gain", "gain (Hz)", "float", 60, 0, 300, 10),
        Param("max_fc", "max corner (Hz)", "float", 180, 40, 500, 10),
    ], ref=True, help="Per-pixel corner scales with the reference cardiac-motion strength."),
    Method("reference_subspace_projection", "reference-subspace projection [novel]",
           reference_subspace_projection, [
        Param("n_components", "components", "int", 3, 1, 15, 1),
        Param("basis", "basis", "select", "spatial", options=["spatial", "temporal"]),
    ], ref=True, help="Learn the cardiac-motion subspace from the reference and project it out of "
                      "the tracking field (the ARF wave is orthogonal to it)."),
    Method("phase_unwrap_temporal", "phase unwrap (temporal)", phase_unwrap_temporal, [
        Param("ambiguity_um", "ambiguity (um, 0=auto)", "float", 0, 0, 800, 10),
    ], help="Unwrap large phase-wrapped displacement along slow time so motion filters can remove it."),
    Method("axial_strain", "axial strain (curl-like)", axial_strain, [
        Param("smooth", "axial smooth", "int", 3, 1, 9, 1),
    ], help="d/dz of displacement: translation-invariant, rejects spatially-uniform bulk motion."),
]

# ----------------------------------------------------------------------------- Stage 4
SPATIAL_METHODS: List[Method] = [
    NONE,
    Method("spatial_smooth", "Gaussian", spatial_smooth, [
        Param("sigma_z_m", "sigma z (um)", "float", 600, 50, 2000, 50, 1e-6),
        Param("sigma_x_m", "sigma x (um)", "float", 1200, 50, 3000, 50, 1e-6),
    ]),
    Method("spatial_median", "median", spatial_median, [
        Param("size_z_m", "size z (um)", "float", 400, 50, 2000, 50, 1e-6),
        Param("size_x_m", "size x (um)", "float", 800, 50, 3000, 50, 1e-6),
    ]),
    Method("bilateral_denoise", "bilateral (edge-preserving)", bilateral_denoise, [
        Param("sigma_color_um", "sigma value (um)", "float", 3.0, 0.2, 20.0, 0.2),
        Param("sigma_spatial_px", "sigma spatial (px)", "float", 2.0, 0.5, 8.0, 0.5),
    ], help="Averages neighbours weighted by spatial + value similarity; keeps wavefront edges."),
    Method("nlm_denoise", "non-local means", nlm_denoise, [
        Param("h_um", "strength h (um)", "float", 3.0, 0.2, 20.0, 0.2),
        Param("patch_size", "patch size (px)", "int", 5, 3, 11, 2),
        Param("patch_distance", "search dist (px)", "int", 6, 3, 15, 1),
    ], help="Denoise by averaging similar patches across the frame (repetitive speckle texture)."),
    Method("aniso_diffusion", "anisotropic diffusion (Perona-Malik)", aniso_diffusion, [
        Param("n_iter", "iterations", "int", 5, 1, 30, 1),
        Param("kappa_um", "gradient scale (um)", "float", 5.0, 0.5, 30.0, 0.5),
        Param("gamma", "step size", "float", 0.2, 0.05, 0.24, 0.01),
    ], help="Edge-preserving diffusion: smooths within, not across, wavefront edges."),
    Method("coherence_diffusion", "coherence-enhancing (tensor) diffusion", coherence_diffusion, [
        Param("n_iter", "iterations", "int", 5, 1, 20, 1),
        Param("sigma", "gradient sigma (px)", "float", 1.0, 0.4, 4.0, 0.2),
        Param("rho", "tensor sigma (px)", "float", 3.0, 0.5, 8.0, 0.5),
        Param("gamma", "step size", "float", 0.15, 0.02, 0.24, 0.01),
    ], help="Weickert tensor-guided diffusion: smooths ALONG the local wavefront orientation, "
            "sharpening coherent line structures."),
]

# ----------------------------------------------------------------------------- Stage 5
TEMPORAL_METHODS: List[Method] = [
    NONE,
    Method("temporal_moving_mean", "moving mean", temporal_moving_mean, [
        Param("window", "window (frames)", "int", 3, 1, 15, 1),
    ]),
    Method("temporal_moving_median", "moving median", temporal_moving_median, [
        Param("window", "window (frames)", "int", 3, 1, 15, 1),
    ]),
    Method("savgol_temporal", "Savitzky-Golay", savgol_temporal, [
        Param("window", "window (frames)", "int", 7, 3, 21, 2),
        Param("polyorder", "poly order", "int", 3, 1, 5, 1),
    ], help="Sliding polynomial fit: suppresses jitter while preserving the shear-wave transient's "
            "timing/amplitude better than a moving average."),
]

# ----------------------------------------------------------------------------- Stage 7
DIRECTIONAL_METHODS: List[Method] = [
    NONE,
    Method("outward", "outward (symmetric about r0)", outward_spacetime, [], help="ARF: keep waves "
           "travelling away from r0 on both sides."),
    Method("leftward", "leftward (-r)", directional_spacetime, [], help="Keep -r-travelling waves."),
    Method("rightward", "rightward (+r)", directional_spacetime, [], help="Keep +r-travelling waves."),
]

# ----------------------------------------------------------------------------- Stage 8
SPEED_METHODS_UI: List[Method] = [
    Method("ttp_ransac", "time-to-peak + RANSAC", ttp_ransac_speed, []),
    Method("radon", "slant-stack (tau-p / Radon)", slant_stack_speed, []),
    Method("tof_xcorr", "time-of-flight cross-correlation", tof_xcorr_speed, []),
]

QUANTITIES = ["displacement", "velocity", "acceleration"]
MODES = ["relative_to_reference", "frame_to_frame"]


def by_name(methods: List[Method], name: str) -> Method:
    for m in methods:
        if m.name == name:
            return m
    return methods[0]


def step_params(method: Method, values: dict) -> dict:
    """Build the SI kwargs dict for a Step from widget values (applying each param's scale)."""
    out = {}
    for p in method.params:
        v = values.get(p.arg, p.default)
        out[p.arg] = (v * p.scale) if p.kind == "float" and p.scale != 1.0 else v
    return out
