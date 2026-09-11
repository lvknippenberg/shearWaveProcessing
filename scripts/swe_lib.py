"""Shared helpers for the 2026-08 acquisition-parameter campaign (tasks 1-4).

One place for: folder discovery + acquisition-parameter reading, the two settled processing
recipes, space-time construction straight from the beamformed buffer-2 IQ, and the
speed-free quality scores (origin coherence + mirror symmetry).

Recipes
-------
``REC_INVIVO``  the in-vivo consensus / report recipe (configs/active.yaml view A):
                band-pass 120-700 Hz -> Gaussian 0.6 x 1.2 mm -> moving-mean 3 -> 7 offsets
                (0.8 mm) -> outward.  Report Fig. "phantom voltage montage" uses this.
``REC_PHANTOM`` the phantom/low-SNR robust recipe id438 (docs/phantom_parameter_sweep.md):
                band-pass 80-500 Hz -> spatial median 0.90 x 2.64 mm -> moving-median 5 ->
                9 offsets (1.09 mm) -> outward.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.io as sio

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "swp_gui", "scripts"):
    if os.path.join(_ROOT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _p))

import core                                                     # noqa: E402
from swp.viz.estimators import loupas_displacement              # noqa: E402
from swp.viz.filters import FIELD_FILTERS, svd_clutter          # noqa: E402
from swp.viz.filters.context import FilterCtx                   # noqa: E402
from swp.viz.filters.directional import outward_spacetime       # noqa: E402
from swp.viz.speed.spacetime import build_spacetime, SpaceTime  # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing               # noqa: E402
from swp.viz.metrics import origin_coherence                    # noqa: E402
from swp.viz.core.geometry import robust_clim                   # noqa: E402
from detect_v import symmetric_v_score, roi_contrast            # noqa: E402

REC_INVIVO = {
    "iq": "none",
    "motion": [("temporal_bandpass", {"f_lo": 120, "f_hi": 700, "order": 2})],
    "spatial": [("spatial_smooth", {"sigma_z_m": 0.6e-3, "sigma_x_m": 1.2e-3})],
    "temporal": [("temporal_moving_mean", {"window": 3})],
    "offsets": 7, "step_m": 0.8e-3, "tag": "bp120-700 gauss mean3",
}
REC_PHANTOM = {
    "iq": "none",
    "motion": [("temporal_bandpass", {"f_lo": 80, "f_hi": 500, "order": 2})],
    "spatial": [("spatial_median", {"size_z_m": 0.90e-3, "size_x_m": 2.64e-3})],
    "temporal": [("temporal_moving_median", {"window": 5})],
    "offsets": 9, "step_m": 1.09e-3, "tag": "bp80-500 median median5",
}

_REC_PHANTOM_ML = core.Recipe(mline_source="horizontal_push")
_REC_MANUAL_ML = core.Recipe(mline_source="auto")


# ---------------------------------------------------------------- acquisition metadata
def acq_params(folder):
    """(elements, cycles, commanded push V, PRI us, focal depth mm) from the runtime .mat."""
    m = sio.loadmat(os.path.join(folder, "AcquisitionParametersAndECG.mat"),
                    squeeze_me=True, struct_as_record=False)
    SW, TPC = m["SW"], m["TPC"]
    out = {"el": int(SW.nb_push_elmts), "cyc": int(SW.pushCycle),
           "V": float(TPC[4].hv), "folder": folder}
    for k, attr in (("PRI_us", "PRI_us"), ("Nframes", "Nframes"), ("push_depth", "push_depth")):
        if hasattr(SW, attr):
            out[k] = float(getattr(SW, attr))
    return out


def measurement_folders(root):
    """All measurement subfolders of `root` that hold a runtime .mat, in acquisition order."""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "AcquisitionParametersAndECG.mat")):
            out.append(p)
    return out


def n_pushes(folder):
    import glob
    return len(glob.glob(os.path.join(folder, "output", "CombinedData_buffer2_meas*_iq.hdf5")))


# ---------------------------------------------------------------- space-time
_ACQ_CACHE = {}


def load_push(folder, meas, phantom=True, cache=True):
    """(acq, mline, r0) for one push. Loupas estimator is built by `spacetime`."""
    key = (folder, meas, phantom)
    if cache and key in _ACQ_CACHE:
        return _ACQ_CACHE[key]
    rec = _REC_PHANTOM_ML if phantom else _REC_MANUAL_ML
    acq = core.load_acq(folder, meas, rec)
    ml = core.load_mline_for(folder, meas, acq, rec)
    r0 = _r0_lateral_crossing(ml, float(acq.push_x))
    if cache:
        _ACQ_CACHE[key] = (acq, ml, r0)
    return acq, ml, r0


def estimator(acq, iq_cfg="none", nopush=False, ref_split=10):
    """Loupas displacement, relative to the pre-push reference.

    ``nopush=True`` builds the **control**: the first ``ref_split`` pre-push frames act as the
    reference and the *remaining* pre-push frames are tracked against them, so the recipe runs on
    cardiac/clutter motion alone with **no data leakage** (reference and tracked frames are
    disjoint). A recipe that produces a V here is imaging motion, not the ARF wave. ``ref_split``
    trades reference averaging against control length -- keep it small so the control window stays
    long enough for the moveout to develop.
    """
    iq, ref = acq.iq, acq.ref_iq
    if nopush:
        if ref is None or ref.shape[0] < 6:
            raise ValueError("no usable pre-push reference for the no-push control")
        k = int(np.clip(ref_split, 3, ref.shape[0] - 5))
        iq, ref = ref[k:], ref[:k]
    if iq_cfg.startswith("svd"):
        iq = svd_clutter(iq, n_remove=int(iq_cfg[3:]))
    return loupas_displacement(iq, dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod,
                               prf=acq.prf, mode="relative_to_reference", reference=ref)


def spacetime(est, acq, ml, r0, recipe, quantity="displacement", directional=True, n_t=None):
    """Post-estimator pipeline -> SpaceTime (identical to scripts/sweep_extract.spacetime_for).

    ``n_t`` (used by the no-push control, whose ensemble is shorter) overrides the time base
    length so the control shares the push's frame indexing.
    """
    prf = acq.prf
    t = acq.t if n_t is None else np.arange(n_t) / prf
    if quantity == "velocity":
        f_all, t_all = est.velocity, 0.5 * (t[:-1] + t[1:])
    elif quantity == "acceleration":
        f_all, t_all = np.diff(est.velocity, axis=0) * prf, t[1:-1]
    else:
        f_all, t_all = est.displacement, t
    fld, times = f_all[1:], t_all[1:]                       # drop_first = 1
    ctx = FilterCtx(dz=acq.dz, dx=acq.dx, prf=prf, t=times, x=acq.x, z=acq.z,
                    f_demod=acq.f_demod, c=acq.c)
    for name, params in recipe["motion"] + recipe["spatial"] + recipe["temporal"]:
        fld = FIELD_FILTERS[name](fld, ctx, **params)
    st = build_spacetime(fld, acq.z, acq.x, ml, times, quantity=quantity,
                         n_offsets=recipe["offsets"], offset_step_m=recipe["step_m"], agg="mean")
    data = outward_spacetime(st.data, st.r, r0) if directional else st.data
    return SpaceTime(data, st.r, st.t, st.quantity)


def st_for(folder, meas, recipe, quantity="displacement", phantom=True, directional=True,
           nopush=False, ref_split=10):
    acq, ml, r0 = load_push(folder, meas, phantom=phantom)
    est = estimator(acq, recipe.get("iq", "none"), nopush=nopush, ref_split=ref_split)
    n_t = est.displacement.shape[0] if nopush else None
    return spacetime(est, acq, ml, r0, recipe, quantity, directional, n_t=n_t), r0


# ---------------------------------------------------------------- scoring / drawing
def truncate(st, n_t):
    """First ``n_t`` time samples of a SpaceTime (used to score a push and its shorter no-push
    control over an identical window)."""
    n = min(int(n_t), st.data.shape[0])
    return SpaceTime(st.data[:n], st.r, st.t[:n], st.quantity)


def scores(st, r0, d_max=16e-3, cmin=1.0, t0_max=3.0e-3):
    """(origin_coherence, mirror-symmetry) - both speed-free-ish quality proxies.

    ``d_max`` / ``cmin`` / ``t0_max`` bound the slant-stack. The defaults suit the full ~15 ms
    push window; a short window (e.g. the no-push control, ~5 ms) must use a smaller ``d_max``
    and a higher ``cmin``, or the moveout runs off the end of the record and the normalisation
    breaks (origin coherence can then exceed 1).
    """
    try:
        oc = float(origin_coherence(st, r0, cmin=cmin, d_max=d_max, t0_max=t0_max))
    except Exception:                                        # noqa: BLE001
        oc = float("nan")
    try:
        sym = float(symmetric_v_score(st, r0, d_max=d_max, cmin=cmin, t0_max=t0_max)[0])
    except Exception:                                        # noqa: BLE001
        sym = float("nan")
    return oc, sym


def draw(ax, st, r0, clim=None, unit=None):
    """imshow one space-time (r in mm, t in ms, RdBu_r, dashed r0). Returns the used clim."""
    unit = unit if unit is not None else (1e3 if st.quantity == "velocity" else 1e6)
    r_mm, t_ms = st.r * 1e3, st.t * 1e3
    img = st.data * unit
    if clim is None:
        rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
        clim = (robust_clim(st.data, rc, 97) or np.percentile(np.abs(st.data), 99) or 1.0) * unit
    ax.imshow(img, extent=(r_mm[0], r_mm[-1], t_ms[-1], t_ms[0]), cmap="RdBu_r",
              vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    ax.axvline(r0 * 1e3, color="0.2", ls="--", lw=0.8, alpha=0.7)
    ax.tick_params(labelsize=7)
    return clim
