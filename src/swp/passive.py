"""Passive-SWE burst-window workflow.

Natural (valve-closure) shear waves in the buffer-4 ultrafast stream are transient bursts, so
processing the whole ~1 s cine as one space-time is not the right unit. This workflow, matching
the intended clinical reading, is:

  1. Draw a **general M-line** on the buffer-4 B-mode (once).
  2. Compute the displacement along it over the whole recording and **detect bursts** in the
     along-line energy -> candidate mitral / aortic valve-closure events (fixed 100 ms windows).
  3. For **each** window, draw a **fresh M-line** on the B-mode frame at that event (the wall
     moves through the cardiac cycle, so each event gets its own line) and process only the
     100 ms around it with the passive recipe.
  4. Return a **montage** of the resulting space-time plots.

Every M-line is saved as ``<output>/mlines/*.npz`` and reused on re-runs (draw once, batch after).
M-line selection is interactive (matplotlib); a saved line is loaded without a prompt.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import replace

import numpy as np

from .viz import runconfig as rc
from .viz.core.acquisition import Acquisition
from .viz.io import load_acquisition, load_mline
from .viz.mline import mline_from_points
from .viz.pipeline import run_pipeline
from .viz.metrics import origin_coherence
from .viz.viz import spacetime_montage
from .mline.select import (
    load_bmode_frame, select_mline, save_mline, draw_mline_on_bmode,
    detect_line_bursts, plot_bursts,
)

# Motion-comp steps that the burst-overview pass keeps (it uses the full config recipe minus
# the directional filter); see run.py MOTION_FILTERS for the active-side equivalent.


def _middle_frame(path: str) -> int:
    import h5py
    with h5py.File(path, "r") as f:
        n = f["tracks/track_0/data/beamformed_data/values"].shape[0]
    return n // 2


def _frame_at_time(t_axis: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(t_axis) - t)))


def _stride_acq(acq: Acquisition, s: int) -> Acquisition:
    """Spatially decimate an acquisition by ``s`` (for the fast burst-overview pass only)."""
    if s <= 1:
        return acq
    coords = acq.coords[::s, ::s, :] if acq.coords is not None else None
    return dataclasses.replace(
        acq, iq=acq.iq[:, ::s, ::s], x=acq.x[::s].copy(), z=acq.z[::s].copy(),
        dz=acq.dz * s, dx=acq.dx * s, coords=coords,
    )


def _ensure_mline(npz_path: str, bmode_path: str, frame: int, title: str,
                  n_samples: int = 250, redraw: bool = False):
    """Load a saved M-line if present (and not ``redraw``), else draw one interactively + save."""
    if os.path.exists(npz_path) and not redraw:
        print(f"  [M-line] reuse {os.path.basename(npz_path)}")
        pts, ns = load_mline(npz_path)
        return mline_from_points(pts, ns or n_samples)
    bmode_u8, coords, n = load_bmode_frame(bmode_path, frame)
    frame = int(np.clip(frame, 0, n - 1))
    mline = select_mline(bmode_u8, coords, n_samples=n_samples, title=title)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    save_mline(npz_path, mline)
    draw_mline_on_bmode(bmode_u8, coords, mline, npz_path.replace(".npz", ".png"), title=title)
    print(f"  [M-line] drawn ({mline.length * 1e3:.1f} mm) -> {os.path.basename(npz_path)} (+ .png)")
    pts, ns = load_mline(npz_path)
    return mline_from_points(pts, ns or n_samples)


def process_passive(folder, config="configs/passive.yaml", window_ms=100.0, max_events=4,
                    pad_ms=20.0, overview_stride=2, redraw=False):
    """Run the passive burst-window workflow on one measurement folder. Returns the montage path.

    Args:
        folder: measurement folder (uses ``<folder>/output`` written by the beamform stage).
        config: passive config (frame_to_frame recipe).
        window_ms: width of the window placed around each detected burst (default 100 ms).
        max_events: maximum number of burst windows to keep.
        pad_ms: extra time each side of the window fed to the filters (cropped conceptually).
        overview_stride: spatial decimation for the fast burst-detection pass (1 = full res).
        redraw: force re-drawing every M-line even if a saved one exists.
    """
    cfg = rc.load_config(config)
    output_dir = os.path.join(folder, "output")
    cfg["data"]["root"] = output_dir
    iq_path = rc.hdf5_path(cfg, 0)
    if not os.path.exists(iq_path):
        raise SystemExit(f"passive IQ not found: {iq_path}\nRun the beamform stage first "
                         f"(python run.py beamform \"{folder}\").")
    bmode_path = os.path.join(output_dir, cfg["data"].get("bmode") or os.path.basename(iq_path))
    outdir = rc.outdir(cfg, output_dir)
    os.makedirs(outdir, exist_ok=True)
    mlines_dir = os.path.join(output_dir, "mlines")

    print(f"passive workflow on {os.path.basename(iq_path)}")
    acq = load_acquisition(iq_path)
    print(f"  {acq.summary()}")
    base = rc.build_pipeline_config(cfg, acq=acq)
    print(f"  recipe: mode={base.mode} filters={[s.name for s in base.field_filters]} "
          f"directional={base.directional}")

    # --- 1) general M-line (interactive if not saved) ---
    gen_npz = os.path.join(mlines_dir, "passive_general_mline.npz")
    gen_mline = _ensure_mline(gen_npz, bmode_path, _middle_frame(bmode_path),
                              "GENERAL passive M-line (buffer 4) -- for burst detection",
                              n_samples=cfg["mline"].get("n_samples", 250), redraw=redraw)

    # --- 2) along-line displacement over the whole recording -> burst detection ---
    print("  computing displacement overview for burst detection "
          f"(spatial stride {overview_stride}) ...")
    ov_cfg = replace(base, directional=False)
    ov = run_pipeline(_stride_acq(acq, overview_stride), gen_mline, ov_cfg, focus=None)
    D_st = np.asarray(ov.st.data).T                    # (n_s, n_t) as the burst detector expects
    t_s = np.asarray(ov.st.t)
    windows, energy = detect_line_bursts(D_st, t_s, window_ms=window_ms, max_events=max_events)
    bursts_png = os.path.join(outdir, "passive_bursts.png")
    plot_bursts(energy, t_s, windows, bursts_png,
                title=f"passive along-M-line energy -- {len(windows)} burst window(s) "
                      f"({int(window_ms)} ms), candidate valve closures")
    print(f"  detected {len(windows)} burst window(s) -> {os.path.basename(bursts_png)}")
    for i, w in enumerate(windows):
        print(f"    #{i}: peak {w.t_peak*1e3:.0f} ms, window [{w.t0*1e3:.0f}, {w.t1*1e3:.0f}] ms")
    if not windows:
        raise SystemExit("no bursts detected; try a different general M-line or lower thresholds.")

    # --- 3) per-window: fresh M-line + process the 100 ms around the event ---
    pad_s = pad_ms * 1e-3
    results, titles = [], []
    for i, w in enumerate(windows):
        frame = _frame_at_time(acq.t, w.t_peak)
        # Keyed by window INDEX (not time span) so reuse survives recipe/band tweaks that
        # jitter the detected window edges. If the number/order of detected windows changes,
        # re-draw with --redraw so a saved line is not reused for a different event.
        npz = os.path.join(mlines_dir, f"passive_win{i}_mline.npz")
        ml = _ensure_mline(npz, bmode_path, frame,
                           f"passive M-line -- window {i} @ {w.t_peak*1e3:.0f} ms "
                           f"[{w.t0*1e3:.0f}-{w.t1*1e3:.0f} ms]",
                           n_samples=cfg["mline"].get("n_samples", 250), redraw=redraw)
        i0 = _frame_at_time(acq.t, w.t0 - pad_s)
        i1 = _frame_at_time(acq.t, w.t1 + pad_s) + 1
        acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
        res = run_pipeline(acq_w, ml, base, focus=None)
        oc = origin_coherence(res.st, res.r0)
        results.append(res)
        titles.append(f"win{i}  {w.t_peak*1e3:.0f} ms\n[{w.t0*1e3:.0f}-{w.t1*1e3:.0f} ms]  oc={oc:.2f}")
        print(f"    window #{i}: space-time {res.st.data.shape} oc={oc:.3f}")

    # --- 4) montage ---
    montage = os.path.join(outdir, "passive_windows_montage.png")
    spacetime_montage(results, montage, ncols=min(len(results), 4), panel_titles=titles,
                      suptitle=f"Passive SWE -- {len(results)} valve-closure window(s), "
                               f"{int(window_ms)} ms each  (fresh M-line per window)")
    print(f"done -> {montage}")
    return montage
