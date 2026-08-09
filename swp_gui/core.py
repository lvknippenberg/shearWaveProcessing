"""Streamlit-independent compute layer for the method-exploration GUI.

Pure functions the app wraps with caching/widgets: data discovery, coarse-vs-fine acquisition
loading, turning a :class:`Recipe` (the UI selections) into a
:class:`~swp.viz.pipeline.PipelineConfig`, running the pipeline (and the pre-push "no-push"
control), computing metrics, and rendering the B-mode + space-time figures. Kept free of any
``streamlit`` import so it can be unit-/headless-tested.
"""
from __future__ import annotations

import dataclasses
import glob
import os
from dataclasses import dataclass, field

import numpy as np

from swp.viz.pipeline import run_pipeline, PipelineConfig, Step, PipelineResult
from swp.viz.io.loader import load_acquisition
from swp.viz.io import load_mline as load_mline_npz
from swp.viz.mline.mline import mline_from_points, horizontal_mline, MLine
from swp.viz.core.acquisition import Acquisition
from swp.viz.core.geometry import PushFocus
from swp.viz.metrics import origin_coherence, wavefront_visibility, slant_stack_speed
# swp.acquisition.finegrid (which imports zea/torch, ~10 s + CUDA) is imported lazily inside
# load_acq() so the coarse path - and GUI boot - never pays that cost.

import registry as reg


# ============================================================ data discovery
def _has_active_iq(folder: str) -> bool:
    return bool(glob.glob(os.path.join(folder, "output", "CombinedData_buffer2_meas*_iq.hdf5")))


def list_measurement_folders(root: str):
    """Sub-folders of ``root`` that look like SWI measurements (have beamformed IQ or a runtime .mat)."""
    if not root or not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        f = os.path.join(root, name)
        if not os.path.isdir(f):
            continue
        if _has_active_iq(f) or os.path.exists(os.path.join(f, "AcquisitionParametersAndECG.mat")) \
                or glob.glob(os.path.join(f, "*.mat")):
            out.append(f)
    # also allow root itself to be a single measurement folder
    if not out and (_has_active_iq(root) or glob.glob(os.path.join(root, "*.mat"))):
        out = [root]
    return out


def list_measurements(folder: str):
    files = glob.glob(os.path.join(folder, "output", "CombinedData_buffer2_meas*_iq.hdf5"))
    idx = []
    for f in files:
        b = os.path.basename(f)
        try:
            idx.append(int(b.split("meas")[1].split("_")[0]))
        except (IndexError, ValueError):
            pass
    return sorted(idx)


def stored_iq_path(folder, meas):
    return os.path.join(folder, "output", f"CombinedData_buffer2_meas{meas}_iq.hdf5")


def mline_npz_path(folder, meas):
    return os.path.join(folder, "output", "mlines", f"active_meas{meas}_mline.npz")


def bmode_iq_path(folder):
    return os.path.join(folder, "output", "CombinedData_buffer5_iq.hdf5")


# ============================================================ the recipe (UI state)
Steps = list          # ordered list of (method_name, params_dict); params in widget units


@dataclass
class Recipe:
    """UI selections. The four filter stages are **ordered lists of steps** ``[(method, params), ...]``
    (so methods can be chained, e.g. polynomial detrend -> band-pass). Each ``params`` dict holds
    widget-unit values keyed by the function kwarg; an empty dict falls back to each param's registry
    default (see :func:`registry.step_params`). Length params convert to SI by the param ``scale``."""
    # 1 IQ pre-filter (chain)
    iq_steps: Steps = field(default_factory=list)
    # 2 displacement
    estimator: str = "loupas"
    est_params: dict = field(default_factory=dict)
    mode: str = "relative_to_reference"
    quantity: str = "displacement"
    drop_first: int = 1
    use_fine: bool = False
    ppw: float = 8.0
    margin_mm: float = 3.0
    # 3 motion removal (chain)
    motion_steps: Steps = field(default_factory=lambda: [("temporal_bandpass", {})])
    # 4 spatial (chain)
    spatial_steps: Steps = field(default_factory=lambda: [("spatial_smooth", {})])
    # 5 temporal (chain)
    temporal_steps: Steps = field(default_factory=lambda: [("temporal_moving_mean", {})])
    # 6 M-line averaging
    offsets: int = 7
    offset_step_mm: float = 0.8
    mline_agg: str = "mean"
    mline_source: str = "auto"          # "auto" | "manual" | "horizontal_push"
    # 7 directional
    directional: str = "outward"
    # 8 speed
    speed: str = "ttp_ransac"


# ============================================================ acquisition loading
def load_acq(folder, meas, recipe: Recipe, mline_points=None):
    """Coarse stored IQ, or a fine-grid re-beamform around the M-line if requested (or forced by
    the rf_ncc estimator). Returns an :class:`Acquisition` (rf_ncc gets the RF view)."""
    fine = recipe.use_fine or recipe.estimator == "rf_ncc"
    if fine:
        from swp.acquisition.finegrid import finegrid_acquisition, as_rf   # lazy: pulls in zea/torch
        acq = finegrid_acquisition(folder, meas, points_xz=mline_points, ppw=recipe.ppw,
                                   margin_m=recipe.margin_mm * 1e-3, verbose=False)
        if recipe.estimator == "rf_ncc":
            acq = as_rf(acq)
        return acq
    return load_acquisition(stored_iq_path(folder, meas))


def load_mline_for(folder, meas, acq: Acquisition, recipe: Recipe) -> MLine:
    """Load the M-line for this measurement: saved manual .npz, or a horizontal line at the push
    depth (phantom), chosen by ``recipe.mline_source`` ('auto' picks npz if present)."""
    npz = mline_npz_path(folder, meas)
    src = recipe.mline_source
    if src == "horizontal_push" or (src == "auto" and not os.path.exists(npz)):
        depth = acq.push_z if acq.push_z is not None else float(np.median(acq.z))
        return horizontal_mline(acq.x, depth, 250)
    pts, ns = load_mline_npz(npz)
    return mline_from_points(pts, ns)


def mline_points_for(folder, meas):
    """The raw (x,z) anchor points of the saved M-line, or None (used to size the fine ROI)."""
    npz = mline_npz_path(folder, meas)
    if os.path.exists(npz):
        pts, _ = load_mline_npz(npz)
        return pts
    return None


def _bmode_h5py(path, frame, dynamic_range=(-50, 0)):
    """Read one buffer-5 B-mode frame + extent with **plain h5py** (no zea import - keeps the GUI
    boot/coarse path zea-free). Mirrors the zea beamformed_data layout."""
    import h5py
    bf = "tracks/track_0/data/beamformed_data"
    with h5py.File(path, "r") as f:
        vals = f[f"{bf}/values"]
        n = vals.shape[0]
        frame = int(np.clip(frame, 0, n - 1))
        v = np.asarray(vals[frame])                       # (nz, nx, 2)
        coords = np.asarray(f[f"{bf}/coordinates"])       # (nz, nx, 3)
    env = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2).astype(np.float32)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(env / (env.max() + 1e-12) + 1e-12)
    lo, hi = dynamic_range
    u8 = np.clip((db - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    xs = coords[0, :, 0]; zs = coords[:, 0, 2]
    extent = (xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3)
    return u8, extent


def bmode_for_display(folder, meas, acq: Acquisition):
    """(img_u8, extent_mm) for the B-mode backdrop: the co-registered buffer-5 frame if present
    (what the M-line was drawn on), else the tracking envelope of ``acq``. Uses h5py, not zea."""
    path = bmode_iq_path(folder)
    if os.path.exists(path):
        try:
            return _bmode_h5py(path, meas)
        except Exception:
            pass
    env = np.abs(acq.iq[0])
    db = 20 * np.log10(env / (env.max() + 1e-12) + 1e-6)
    u8 = np.clip((db + 50) / 50 * 255, 0, 255).astype(np.uint8)
    return u8, acq.extent_mm()


# ============================================================ recipe -> PipelineConfig
def _step(method_list, name, params):
    if name == "none":
        return None
    m = reg.by_name(method_list, name)
    return Step(name, reg.step_params(m, params))


def build_focus(acq: Acquisition) -> PushFocus:
    px = acq.push_x if acq.push_x is not None else 0.0
    pz = acq.push_z if acq.push_z is not None else float(np.median(acq.z))
    ix = int(np.argmin(np.abs(acq.x - px)))
    iz = int(np.argmin(np.abs(acq.z - pz)))
    return PushFocus(x=float(acq.x[ix]), z=float(acq.z[iz]), ix=ix, iz=iz)


def _steps_to_filters(step_list, methods):
    out = []
    for name, params in step_list:
        s = _step(methods, name, params)
        if s:
            out.append(s)
    return out


def to_config(recipe: Recipe, acq: Acquisition) -> PipelineConfig:
    iq_filters = _steps_to_filters(recipe.iq_steps, reg.IQ_METHODS)
    field_filters = (_steps_to_filters(recipe.motion_steps, reg.MOTION_METHODS)
                     + _steps_to_filters(recipe.spatial_steps, reg.SPATIAL_METHODS)
                     + _steps_to_filters(recipe.temporal_steps, reg.TEMPORAL_METHODS))

    directional = recipe.directional != "none"
    dmode = {"outward": "outward", "leftward": "leftward", "rightward": "rightward"}.get(
        recipe.directional, "outward")

    est_method = reg.by_name(reg.ESTIMATOR_METHODS, recipe.estimator)
    est_params = reg.step_params(est_method, recipe.est_params)
    # bool params must stay bool (step_params leaves them; loupas local_frequency etc.)

    return PipelineConfig(
        estimator=recipe.estimator,
        estimator_params=est_params,
        mode=recipe.mode,
        quantity=recipe.quantity,
        iq_filters=iq_filters,
        field_filters=field_filters,
        directional=directional,
        directional_mode=dmode,
        speed=recipe.speed,
        drop_first=recipe.drop_first,
        mline_offsets=recipe.offsets,
        mline_offset_step_m=recipe.offset_step_mm * 1e-3,
        mline_agg=recipe.mline_agg,
        push_x_m=float(acq.push_x) if acq.push_x is not None else 0.0,
    )


# ============================================================ SVD spectrum (GUI tuning aid)
def svd_spectrum(acq: Acquisition, recipe: Recipe):
    """Singular-value spectrum for the active SVD-clutter step, for the GUI cutoff-tuning plot.

    Returns ``dict(S, nr, nh, domain)`` for the first ``svd_clutter`` (IQ, computed on the tracking IQ
    ensemble) or ``svd_clutter_field`` (displacement, on the Loupas displacement field) in the recipe,
    else ``None``. ``nr``/``nh`` are the current remove-low / remove-high cutoffs. The Casorati layout
    matches the filter, so the rank axis and the kept range are exactly what the filter uses."""
    def _S(vol):
        vol = np.asarray(vol)
        return np.linalg.svd(vol.reshape(vol.shape[0], -1), compute_uv=False)

    try:
        for name, params in recipe.iq_steps:
            if name == "svd_clutter":
                return dict(S=_S(acq.iq), nr=int(params.get("n_remove", 1)),
                            nh=int(params.get("n_high_remove", 0)), domain="IQ")
        for name, params in recipe.motion_steps:
            if name == "svd_clutter_field":
                from swp.viz.estimators import loupas_displacement
                ref = acq.ref_iq if recipe.mode == "relative_to_reference" else None
                est = loupas_displacement(acq.iq, dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod,
                                          prf=acq.prf, mode=recipe.mode, reference=ref)
                fld = est.displacement[recipe.drop_first:]
                return dict(S=_S(fld), nr=int(params.get("n_remove", 1)),
                            nh=int(params.get("n_high_remove", 0)), domain="displacement")
    except Exception:  # noqa: BLE001 - a tuning aid must never break the run
        return None
    return None


# ============================================================ running
def run_recipe(acq: Acquisition, mline: MLine, cfg: PipelineConfig, nopush: bool = False):
    """Run the pipeline. ``nopush=True`` feeds the pre-push reference frames as the tracking
    ensemble (same recipe) - the control that reveals whether a recipe images the wave or cardiac
    motion."""
    focus = build_focus(acq)
    if nopush:
        if acq.ref_iq is None:
            return None
        n_ref = acq.ref_iq.shape[0]
        t_ref = acq.t_ref if acq.t_ref is not None else (np.arange(n_ref) / acq.prf)
        acq = dataclasses.replace(acq, iq=acq.ref_iq, t=np.asarray(t_ref, float))
    return run_pipeline(acq, mline, cfg, focus=focus)


def recipe_label(recipe: Recipe) -> str:
    """Compact one-line summary of the active methods (for labelling history/comparison panels)."""
    def steps(lst):
        real = [n for n, _ in lst if n != "none"]
        return "+".join(real) if real else "none"
    parts = []
    if recipe.iq_steps:
        parts.append("iq:" + steps(recipe.iq_steps))
    parts.append(f"{recipe.estimator}/{recipe.mode.split('_')[0]}/{recipe.quantity[:4]}")
    parts.append("mo:" + steps(recipe.motion_steps))
    parts.append("sp:" + steps(recipe.spatial_steps))
    parts.append("tp:" + steps(recipe.temporal_steps))
    parts.append(f"off{recipe.offsets}")
    if recipe.directional != "none":
        parts.append(recipe.directional)
    return " | ".join(parts)


def data_params(acq: Acquisition) -> dict:
    """Acquisition constants that are fixed by the Verasonics data (NOT tuning knobs) - shown
    read-only in the GUI so it's clear they match the acquisition. ``Fs_equiv`` is the effective
    axial sampling rate implied by the beamforming pitch (c / 2 / dz)."""
    return {
        "center freq f0 [MHz]": acq.f0 / 1e6 if acq.f0 else float("nan"),
        "demod freq [MHz]": acq.f_demod / 1e6 if acq.f_demod else float("nan"),
        "PRF [Hz]": acq.prf,
        "speed of sound c [m/s]": acq.c,
        "axial pitch dz [um]": acq.dz * 1e6,
        "lateral pitch dx [um]": acq.dx * 1e6,
        "Fs_equiv = c/2/dz [MHz]": (acq.c / (2 * acq.dz)) / 1e6 if acq.dz else float("nan"),
        "grid": f"{acq.grid} ({acq.nz}x{acq.nx})",
        "frames (track/ref)": f"{acq.n_frames}/{0 if acq.ref_iq is None else acq.ref_iq.shape[0]}",
    }


def compute_metrics(res: PipelineResult) -> dict:
    oc, left, right = origin_coherence(res.st, res.r0, return_sides=True)
    sem, c_signed = slant_stack_speed(res.st, res.r0)
    vis = wavefront_visibility(res.st, res.r0)
    sp = res.speed
    amp = float(np.nanpercentile(np.abs(res.st.data), 95))
    unit = 1e6 if res.st.quantity == "displacement" else 1e3
    return {
        "origin_coherence": oc, "oc_left": left, "oc_right": right,
        "outward_fraction": vis,
        "slant_semblance": sem, "slant_speed": c_signed,
        "c_pos": sp.c_pos, "c_neg": sp.c_neg, "c_mean": sp.c_mean, "speed_quality": sp.quality,
        "amp_p95": amp * unit,
        "amp_unit": "um" if res.st.quantity == "displacement" else "mm/s",
    }
