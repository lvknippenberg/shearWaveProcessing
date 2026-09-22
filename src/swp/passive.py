"""Passive-SWE burst-window workflow.

Natural (valve-closure) shear waves in the buffer-4 ultrafast stream are transient bursts, so
processing the whole ~1 s cine as one space-time is not the right unit. This workflow, matching
the intended clinical reading, is:

  1. Draw a **general M-line** on the buffer-4 B-mode (once).
  2. Compute the displacement along it over the whole recording and **detect bursts** in the
     along-line energy -> candidate mitral / aortic valve-closure events (fixed 100 ms windows).
  3. For **each** window, draw a **fresh M-line** on the B-mode around that event (the wall
     moves through the cardiac cycle, so each event gets its own line) and process only the
     100 ms around it with the passive recipe.
  4. Return a **montage** of the resulting space-time plots.

Every M-line is saved as ``<output>/mlines/*.npz`` and reused on re-runs (draw once, batch after).
M-line selection is interactive (matplotlib); a saved line is loaded without a prompt. Lines are
drawn in the GIF display (adaptive levels + gamma2) on a **single frame**: frame 0 for the general
line, the first frame of the window for a window line. The septum moves too much over the cardiac
cycle for one line to fit a looping cine; ``cine=True`` still offers that view.

The steps are split so a study can be done in two sittings (``scripts/passive_study.py``):
:func:`draw_passive_mlines` (interactive: general line -> detection -> window lines, detection
cached to ``swp_passive/passive_windows.json``) and :func:`process_passive_windows` (unattended).
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import replace

import numpy as np

from .viz import runconfig as rc
from .viz.core.acquisition import Acquisition
from .viz.io import load_acquisition, load_mline
from .viz.mline import mline_from_points
from .viz.pipeline import run_pipeline, Step
from .viz.metrics import passive_coherence, slant_stack_speed
from .viz.viz import spacetime_montage, plot_spacetime
from .viz.filters.directional import directional_spacetime
from .viz.speed.spacetime import SpaceTime
from .mline.select import (
    BurstWindow, fit_spline, select_mline_cine, save_mline, draw_mline_on_bmode, cine_u8_from_iq,
    detect_line_bursts, plot_bursts,
)

# Upper bound of the slant-stack speed search. Well above physiological shear-wave speed on
# purpose: a window with no propagating wavefront rails at the bound, and a bound inside the
# plausible range (the old 6 m/s) made that failure look like a real measurement.
SPEED_CMAX = 20.0

WINDOWS_JSON = "passive_windows.json"
GENERAL_NPZ = "passive_general_mline.npz"
MONTAGE_PNG = "passive_windows_montage.png"

# Cine playback for drawing. General: the whole buffer as a 5 s loop (as
# scripts/draw_passive_mlines.py). Window: the event +/- WINDOW_CONTEXT_MS as a ~3 s loop.
GENERAL_CINE = dict(duration=5.0, fps=25.0)
WINDOW_CINE = dict(duration=3.0, fps=20.0)
WINDOW_CONTEXT_MS = 100.0


class SkipLine(Exception):
    """The user closed a selector window without drawing (skip this line)."""


def _build_views(cfg, acq):
    """Build the list of ``(name, PipelineConfig)`` processing *views* from ``cfg['run']['views']``.

    Each view fully overrides quantity / field_filters / directional / M-line sampling, so a single
    run produces one space-time plot per view per window (mirroring the active side's multi-band
    confirmation view, but with completely different recipes). Falls back to the single main
    ``pipeline`` config when no views are defined.
    """
    views = rc.build_views(cfg, acq)
    if views:
        return views
    return [(cfg["pipeline"].get("quantity", "displacement"),
             rc.build_pipeline_config(cfg, acq=acq))]


def _middle_frame(path: str) -> int:
    import h5py
    with h5py.File(path, "r") as f:
        n = f["tracks/track_0/data/beamformed_data/values"].shape[0]
    return n // 2


def _ensure_mline(npz_path: str, bmode_path: str, frame: int, title: str,
                  n_samples: int = 250, redraw: bool = False):
    """Load a saved M-line if present (and not ``redraw``), else draw one on a STILL frame + save.

    Kept for the search/exploration scripts; the study workflow draws on a cine instead.
    """
    from .mline.select import load_bmode_frame, select_mline

    if os.path.exists(npz_path) and not redraw:
        print(f"  [M-line] reuse {os.path.basename(npz_path)}")
        return _load_line(npz_path, n_samples)
    bmode_u8, coords, n = load_bmode_frame(bmode_path, frame)
    mline = select_mline(bmode_u8, coords, n_samples=n_samples, title=title)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    save_mline(npz_path, mline)
    draw_mline_on_bmode(bmode_u8, coords, mline, npz_path.replace(".npz", ".png"), title=title)
    print(f"  [M-line] drawn ({mline.length * 1e3:.1f} mm) -> {os.path.basename(npz_path)} (+ .png)")
    return _load_line(npz_path, n_samples)


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


# ------------------------------------------------------------------ paths / config
def _paths(folder, config):
    """Resolve the config and the passive paths of one measurement folder."""
    cfg = rc.load_config(config)
    output_dir = os.path.join(folder, "output")
    cfg["data"]["root"] = output_dir
    iq_path = rc.hdf5_path(cfg, 0)
    outdir = rc.outdir(cfg, output_dir)
    return cfg, dict(output=output_dir, iq=iq_path, outdir=outdir,
                     mlines=os.path.join(output_dir, "mlines"),
                     general=os.path.join(output_dir, "mlines", GENERAL_NPZ),
                     windows_json=os.path.join(outdir, WINDOWS_JSON),
                     montage=os.path.join(outdir, MONTAGE_PNG))


def _window_npz(mlines_dir, i):
    # Keyed by window INDEX (not time span) so reuse survives recipe/band tweaks that jitter the
    # detected window edges. If the number/order of detected windows changes, re-draw.
    return os.path.join(mlines_dir, f"passive_win{i}_mline.npz")


def load_acq(folder, config="configs/passive.yaml"):
    """Load the buffer-4 acquisition of a folder (the ~1-2 GB read; safe to run in a thread)."""
    _, p = _paths(folder, config)
    if not os.path.exists(p["iq"]):
        raise FileNotFoundError(f"passive IQ not found: {p['iq']} - run the beamform stage first.")
    return load_acquisition(p["iq"])


# ------------------------------------------------------------------ interactive M-lines
def _load_line(npz, n_samples):
    pts, ns = load_mline(npz)
    return mline_from_points(pts, ns or n_samples)


def _draw_line(npz, frames_u8, coords, title, n_samples, fps, reference=None, labels=None,
               fallback_points=None):
    """Prompt for an M-line on a cine/still, save ``.npz`` + a ``.png`` record. Raises SkipLine.

    With ``fallback_points`` (e.g. the general line), ENTER without clicking saves that line
    instead. Returns ``(line, reused_fallback)``.
    """
    try:
        ml = select_mline_cine(frames_u8, coords, n_samples=n_samples, title=title, fps=fps,
                               reference_lines=reference, frame_labels=labels,
                               allow_empty=fallback_points is not None)
    except ValueError as exc:                      # closed without >= 2 points
        raise SkipLine(str(exc)) from exc
    reused = ml is None
    if reused:
        ml = fit_spline(np.asarray(fallback_points), n_samples=n_samples)
        title = title + "  [= general M-line]"
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    save_mline(npz, ml)
    # Record frame: the one with the event (window) / the first (general) cine frame.
    rec = frames_u8[len(frames_u8) // 2] if labels is not None else frames_u8[0]
    draw_mline_on_bmode(rec, coords, ml, npz.replace(".npz", ".png"), title=title)
    print(f"  [M-line] {'general line reused' if reused else 'drawn'} ({ml.length * 1e3:.1f} mm)"
          f" -> {os.path.basename(npz)} (+ .png)")
    return _load_line(npz, n_samples), reused


def still_frame(acq: Acquisition, k: int, note: str = ""):
    """One frame ``k`` as a 1-frame stack for the selector -> (frames_u8, labels).

    The drawing default. The septum moves too much over the cardiac cycle for one line to fit a
    cine, so each line is drawn on a single frame: frame 0 for the general line, the first frame
    of the event window for a window line.
    """
    k = int(np.clip(k, 0, acq.iq.shape[0] - 1))
    return cine_u8_from_iq(acq.iq[k:k + 1]), [f"frame {k}, t = {acq.t[k] * 1e3:.0f} ms{note}"]


def general_cine(acq: Acquisition):
    """Whole buffer, evenly strided to a GENERAL_CINE loop -> (frames_u8, labels, slowdown)."""
    n = acq.iq.shape[0]
    n_show = int(round(GENERAL_CINE["duration"] * GENERAL_CINE["fps"]))
    stride = max(1, int(round(n / n_show)))
    idx = np.arange(0, n, stride)
    labels = [f"t = {acq.t[k] * 1e3:5.0f} ms" for k in idx]
    slowdown = GENERAL_CINE["duration"] / (n / acq.prf)
    return cine_u8_from_iq(acq.iq[idx]), labels, slowdown


def window_cine(acq: Acquisition, w: BurstWindow):
    """The event window +/- WINDOW_CONTEXT_MS, strided to a WINDOW_CINE loop."""
    ctx = WINDOW_CONTEXT_MS * 1e-3
    i0 = _frame_at_time(acq.t, w.t0 - ctx)
    i1 = _frame_at_time(acq.t, w.t1 + ctx) + 1
    n_show = int(round(WINDOW_CINE["duration"] * WINDOW_CINE["fps"]))
    stride = max(1, int(round((i1 - i0) / n_show)))
    idx = np.arange(i0, i1, stride)
    labels = [f"t = {acq.t[k] * 1e3:5.0f} ms" + ("  << EVENT" if w.t0 <= acq.t[k] <= w.t1 else "")
              for k in idx]
    slowdown = WINDOW_CINE["duration"] / ((i1 - i0) / acq.prf)
    return cine_u8_from_iq(acq.iq[idx]), labels, slowdown


# ------------------------------------------------------------------ detection
def _detect_phase_windows(acq, D_st, t_s, window_ms, max_events, folder=None):
    """Phase-aware burst search -> windows, or None when there is no usable R-peak record.

    **R-peaks only - the ECG waveform is never read.** The expected MVC and AVC intervals are
    derived entirely from the ``ECG_trigger`` timestamps: the R-peak times themselves, and the
    heart rate from their median interval, which sets the Weissler QS2. The logged ``Signal``
    trace plays no part (``docs/ecg_timing.md`` explains why it must not).

    Bridges the trigger log into :func:`swp.mline.select.detect_phase_windows`, expressing the
    R-peaks relative to buffer-4 frame 0, which is the clock ``t_s`` is on. Returns None (so the
    caller falls back to energy ranking) when the trigger record is a fixed-rate pulse train or
    too sparse, since searching a cardiac phase that does not exist would be worse than not
    searching at all.
    """
    from .acquisition.triggerlog import buffer_timing, clean_r_peaks
    from .mline.select import detect_phase_windows

    if folder is None:
        folder = getattr(acq, "meta", {}).get("folder")
    if not folder:
        return None
    bt = buffer_timing(folder, 4)
    if bt is None:
        return None
    kept, rr_ms = clean_r_peaks(bt.r_peaks_ms)
    if kept is None or not np.isfinite(rr_ms or np.nan):
        return None
    r_rel = (np.asarray(kept, float) - bt.t0_ms) * 1e-3          # -> buffer-4 clock, seconds
    span = (float(t_s[0]) - 1.5, float(t_s[-1]) + 1.5)
    r_rel = r_rel[(r_rel >= span[0]) & (r_rel <= span[1])]
    if r_rel.size < 1:
        return None
    windows, _ = detect_phase_windows(D_st, t_s, r_rel, rr_ms * 1e-3,
                                      window_ms=window_ms, max_events=max_events)
    return windows or None


def detect_windows(acq, gen_mline, cfg, outdir, window_ms=100.0, max_events=4, overview_stride=2,
                   folder=None):
    """Along-line displacement over the whole recording -> burst windows (+ overview figures)."""
    base = rc.build_pipeline_config(cfg, acq=acq)
    # Burst detection runs on a FIXED band (detect.band), independent of the processing band, so
    # tuning pipeline.field_filters does not move the detected windows (keeps index M-lines valid).
    detect_band = cfg.get("detect", {}).get("band", [5.0, 150.0])
    smoothing = [s for s in base.field_filters if s.name != "temporal_bandpass"]
    ov_filters = [Step("temporal_bandpass", dict(f_lo=detect_band[0], f_hi=detect_band[1]))] + smoothing
    print("  computing displacement overview for burst detection "
          f"({detect_band[0]}-{detect_band[1]} Hz, spatial stride {overview_stride}) ...")
    ov_cfg = replace(base, directional=False, field_filters=ov_filters)
    ov = run_pipeline(_stride_acq(acq, overview_stride), gen_mline, ov_cfg, focus=None)
    D_st = np.asarray(ov.st.data).T                    # (n_s, n_t) as the burst detector expects
    t_s = np.asarray(ov.st.t)
    # Detection mode. "energy" (default) keeps the four largest bursts and labels them
    # afterwards; "phase" searches the expected MVC/AVC phase windows instead, so a large
    # non-valvular burst cannot displace a genuine closure. Phase mode needs a usable R-peak
    # record and falls back to energy when there is none.
    #
    # NOTE: switching modes changes which windows are detected, and the per-event M-lines are
    # keyed by window INDEX - so it invalidates every stored per-event line for that folder and
    # they must be redrawn. That is why the default is unchanged.
    mode = str(cfg.get("detect", {}).get("mode", "energy")).lower()
    windows = None
    if mode == "phase":
        windows = _detect_phase_windows(acq, D_st, t_s, window_ms, max_events, folder)
        if windows is None:
            print("  [detect] no usable R-peak record - falling back to energy ranking")
    if windows is None:
        windows, energy = detect_line_bursts(D_st, t_s, window_ms=window_ms,
                                             max_events=max_events)
    else:
        from .mline.select import energy_along_line
        energy, _ = energy_along_line(D_st, 30)
        print(f"  [detect] phase-aware: " + ", ".join(
            f"{w.expect or 'energy'}@{w.t_peak * 1e3:.0f}ms" for w in windows))
    os.makedirs(outdir, exist_ok=True)
    bursts_png = os.path.join(outdir, "passive_bursts.png")
    plot_bursts(energy, t_s, windows, bursts_png,
                title=f"passive along-M-line energy -- {len(windows)} burst window(s) "
                      f"({int(window_ms)} ms), candidate valve closures")
    print(f"  detected {len(windows)} burst window(s) -> {os.path.basename(bursts_png)}")

    # --- full-window space-time plot (the whole recording along the general M-line) ---
    # Honours `pipeline.directional`: with it off (the tuned passive default) the overview is
    # left unfiltered, matching the per-window views below. This previously applied
    # `directional_mode` unconditionally, so the overview was leftward-filtered while the
    # windows were not - and the k-omega directional filter is exactly what the passive search
    # found to inject reverberation banding and bias the apparent speed high
    # (docs/passive_search.md), so the two figures disagreed by construction.
    mode = base.directional_mode if base.directional else None
    keep = ("neg" if mode in ("leftward", "neg")
            else "pos" if mode in ("rightward", "pos") else None)
    full = np.asarray(ov.st.data)
    r0_full = float(ov.st.r[-1] if keep == "neg" else ov.st.r[0])
    if keep is not None:
        full = directional_spacetime(full, keep)
    full_st = SpaceTime(full, ov.st.r, ov.st.t, ov.st.quantity)
    full_png = os.path.join(outdir, "passive_full_spacetime.png")
    plot_spacetime(full_st, full_png, r0_mm=r0_full * 1e3,
                   title=f"passive full-window space-time -- general M-line, "
                         f"{detect_band[0]}-{detect_band[1]} Hz"
                         f"{'' if keep is None else f', directional {mode}'}")
    print(f"  full-window space-time -> {os.path.basename(full_png)}")
    for i, w in enumerate(windows):
        print(f"    #{i}: peak {w.t_peak*1e3:.0f} ms, window [{w.t0*1e3:.0f}, {w.t1*1e3:.0f}] ms")
    return windows


def _detect_key(gen_mline, cfg, window_ms, max_events, overview_stride):
    """Everything detection depends on - a cached window list is valid only if this matches."""
    return dict(general_points=np.round(np.asarray(gen_mline.points), 7).tolist(),
                detect_band=list(cfg.get("detect", {}).get("band", [5.0, 150.0])),
                # the detection MODE is part of the key: switching it changes which windows are
                # found, and the per-event M-lines are keyed by window index, so a cached window
                # list from the other mode must not be reused.
                detect_mode=str(cfg.get("detect", {}).get("mode", "energy")).lower(),
                window_ms=window_ms, max_events=max_events, overview_stride=overview_stride)


def read_windows(json_path):
    """-> (state dict, [BurstWindow]) or (None, None) when there is no cache."""
    if not os.path.exists(json_path):
        return None, None
    with open(json_path) as f:
        st = json.load(f)
    return st, [BurstWindow(**w) for w in st.get("windows", [])]


def _write_windows(json_path, st):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, json_path)


def _archive_window_lines(mlines_dir):
    import glob
    import shutil
    import time

    old = glob.glob(os.path.join(mlines_dir, "passive_win*_mline.*"))
    if not old:
        return
    dest = os.path.join(mlines_dir, "archive_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dest, exist_ok=True)
    for f in old:
        shutil.move(f, os.path.join(dest, os.path.basename(f)))
    print(f"  [M-line] {len(old)} stale window-line file(s) archived -> {os.path.basename(dest)}/")


def _cached_windows(p, gen_mline, cfg, key):
    st, windows = read_windows(p["windows_json"])
    if st is not None and st.get("key") == key:
        print(f"  [windows] reuse {WINDOWS_JSON} ({len(windows)} window(s))")
        return st, windows
    return None, None


# ------------------------------------------------------------------ the two phases
def draw_passive_mlines(folder, config="configs/passive.yaml", acq=None, window_ms=100.0,
                        max_events=4, overview_stride=2, redraw=False, label="", cine=False):
    """Interactive phase: general M-line -> burst detection -> one M-line per window.

    Writes ``swp_passive/passive_windows.json`` recording the detected windows and which were
    drawn / skipped. Returns that state dict. Closing the general-line window without drawing
    marks the whole folder skipped; closing a window-line selector skips just that window.
    Lines are drawn on a still frame (``cine=False``, default) or on a looping cine.
    """
    cfg, p = _paths(folder, config)
    n_samples = cfg["mline"].get("n_samples", 250)
    if acq is None:
        acq = load_acq(folder, config)
    coords = acq.coords
    head = f"{label}{os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}"

    # --- 1) general M-line ---
    if os.path.exists(p["general"]) and not redraw:
        print(f"  [M-line] reuse {GENERAL_NPZ}")
        gen = _load_line(p["general"], n_samples)
    else:
        if cine:
            frames, labels, slow = general_cine(acq)
            what = f"whole buffer, {slow:.1f}x slow motion"
        else:
            frames, labels = still_frame(acq, 0)
            what = "first frame"
        title = (f"{head}\nGENERAL passive M-line (buffer 4) - {what}"
                 f"  |  close window = skip folder")
        try:
            gen, _ = _draw_line(p["general"], frames, coords, title, n_samples,
                                GENERAL_CINE["fps"], labels=labels)
        except SkipLine:
            st = dict(skipped_general=True)
            _write_windows(p["windows_json"], st)
            print("  general M-line SKIPPED -> folder marked skipped")
            return st

    # --- 2) detection (cached) ---
    key = _detect_key(gen, cfg, window_ms, max_events, overview_stride)
    st, windows = (None, None) if redraw else _cached_windows(p, gen, cfg, key)
    if st is None:
        # Fresh detection: any existing window lines belong to an earlier detection (a different
        # general line or settings) and would silently be reused for the wrong event - archive them.
        _archive_window_lines(p["mlines"])
        windows = detect_windows(acq, gen, cfg, p["outdir"], window_ms, max_events,
                                 overview_stride, folder=folder)
        st = dict(key=key, windows=[dataclasses.asdict(w) for w in windows], drawn=[], skipped=[])
        _write_windows(p["windows_json"], st)

    # --- 3) per-window M-lines on the window's first frame (or a cine around the event) ---
    ref = [("general M-line", gen.x, gen.z)]
    for i, w in enumerate(windows):
        npz = _window_npz(p["mlines"], i)
        if not redraw and (os.path.exists(npz) or i in st["skipped"]):
            if os.path.exists(npz) and i not in st["drawn"]:
                st["drawn"].append(i)
            print(f"  [M-line] window {i}: {'reuse' if os.path.exists(npz) else 'skipped earlier'}")
            continue
        if cine:
            frames, labels, slow = window_cine(acq, w)
            what = f"+/-{WINDOW_CONTEXT_MS:.0f} ms, {slow:.1f}x slow"
        else:
            frames, labels = still_frame(acq, _frame_at_time(acq.t, w.t0), " (window start)")
            what = "first frame of the window"
        title = (f"{head}\nwindow {i + 1}/{len(windows)}: event @ {w.t_peak*1e3:.0f} ms "
                 f"[{w.t0*1e3:.0f}-{w.t1*1e3:.0f} ms], {what}\n"
                 f"ENTER without clicking = use general line  |  close = skip window")
        try:
            _, reused = _draw_line(npz, frames, coords, title, n_samples, WINDOW_CINE["fps"],
                                   reference=ref, labels=labels, fallback_points=gen.points)
            st["drawn"] = sorted(set(st["drawn"]) | {i})
            fg = set(st.get("from_general", []))
            st["from_general"] = sorted(fg | {i} if reused else fg - {i})
            st["skipped"] = sorted(set(st["skipped"]) - {i})
        except SkipLine:
            st["skipped"] = sorted(set(st["skipped"]) | {i})
            print(f"  [M-line] window {i}: SKIPPED")
        _write_windows(p["windows_json"], st)
    return st


def _row_bmode(p, st, i, w, ml, acq):
    """B-mode frame the window's M-line was drawn on (for the montage), or None.

    Per-event lines record their frame (``window_mlines``); a single line records its source in
    ``passive_general_mline.json``; otherwise the buffer-4 frame at the window start is shown.
    """
    from .mline.select import _grid_axes, load_bmode_frame

    try:
        info = (st.get("window_mlines") or {}).get(str(i))
        src_json = os.path.join(p["mlines"], MLINE_SOURCE_JSON)
        if info is None and os.path.exists(src_json):
            with open(src_json) as f:
                info = json.load(f)
        if info is not None and "buffer" in info:
            buf, k = int(info["buffer"]), int(info["frame"])
            img, coords, _ = load_bmode_frame(os.path.join(p["output"], bmode_file(buf)), k)
            label = f"buffer {buf} frame {k}"
        else:
            k = _frame_at_time(acq.t, w.t0)
            img, coords, label = cine_u8_from_iq(acq.iq[k:k + 1])[0], acq.coords, f"buffer 4 frame {k}"
        xs, zs = _grid_axes(coords)
        return dict(img=img, extent=[xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3],
                    x=ml.x * 1e3, z=ml.z * 1e3,
                    title=f"win{i} {w.t_peak * 1e3:.0f} ms: {label}\nM-line {ml.r[-1] * 1e3:.0f} mm")
    except Exception as exc:                              # noqa: BLE001 - a figure extra only
        print(f"    (no B-mode panel for window {i}: {exc})")
        return None


def _single_wave_speed(st, sem, c, line):
    """A one-directional :class:`SpeedResult` carrying the signed slant-stack moveout.

    Passive waves are one-sided (see :func:`swp.viz.metrics.passive_coherence`), so the whole line
    is one fit: ``t_pred_pos`` is the fitted wavefront over every sample and ``t_pred_neg`` is empty.
    The panel then draws one straight moveout instead of a symmetric pair about r0.
    """
    from .viz.speed.tof import SpeedResult

    r = st.r
    t_line = line[1] if line is not None else np.full(r.size, np.nan)
    return SpeedResult("radon-signed", float(c), float("nan"), float(sem), 0.0,
                       r, np.asarray(t_line, float), np.full(r.size, np.nan))


def process_passive_windows(folder, config="configs/passive.yaml", acq=None, pad_ms=20.0):
    """Unattended phase: process every drawn window with every view -> montage path (or None)."""
    cfg, p = _paths(folder, config)
    n_samples = cfg["mline"].get("n_samples", 250)
    st, windows = read_windows(p["windows_json"])
    if st is None or st.get("skipped_general"):
        print("  no drawn windows (not drawn yet, or folder skipped)")
        return None
    todo = [(i, w) for i, w in enumerate(windows) if os.path.exists(_window_npz(p["mlines"], i))]
    if not todo:
        print("  every window skipped - nothing to process")
        return None
    if acq is None:
        acq = load_acq(folder, config)

    # Each window is rendered with every configured view (3 different high-performing recipes), so the
    # shear-wave propagation can be confirmed across independent processing choices: a real wave shows
    # the same slope in all three, noise does not. Layout: rows = windows, columns = views.
    views = _build_views(cfg, acq)
    print(f"  {len(views)} view(s) per window: " + " | ".join(n for n, _ in views))
    pad_s = pad_ms * 1e-3
    results, titles, speeds, row_bmodes = [], [], [], []
    for i, w in todo:
        ml = _load_line(_window_npz(p["mlines"], i), n_samples)
        row_bmodes.append(_row_bmode(p, st, i, w, ml, acq))
        i0 = _frame_at_time(acq.t, w.t0 - pad_s)
        i1 = _frame_at_time(acq.t, w.t1 + pad_s) + 1
        acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
        for vname, vcfg in views:
            res = run_pipeline(acq_w, ml, vcfg, focus=None)
            # Signed-Radon (slant-stack) speed on the raw band-passed M-mode = wave CENTRE, both
            # directions.
            #
            # remove_flat=False is deliberate and was re-tested on C000000001 (2026-09-11):
            # turning it ON rails 10 of 12 window/view fits at the opposite bound (+/-cmin) and
            # collapses semblance (e.g. win3 0.95 -> 0.19). The cause is geometric - these
            # M-lines are ~42-49 mm while a 3 m/s wave at ~20 Hz has lambda ~ 175 mm, so the
            # line spans only ~0.3 of a wavelength and a GENUINE wave is nearly spatially
            # uniform along it. Subtracting the per-time spatial mean therefore removes the
            # signal along with the bulk motion. (Corollary: this line length is marginal for
            # these speeds; the slant-stack is fitting a fraction of a cycle.)
            sem, c, line = slant_stack_speed(res.st, res.r0, cmin=1.0, cmax=SPEED_CMAX,
                                             remove_flat=False, return_line=True)
            # Overlay the SINGLE fitted moveout across the whole line, replacing the pipeline's
            # default two-sided ttp_ransac fit. That estimator splits the line at r0 and fits each
            # side separately, which is the ARF picture: a push radiates outward from a focus. A
            # valve-closure wave enters at one end and crosses in one direction, so the two-lobed
            # fit is wrong here - and it is only ever computed over 2-14 mm either side of r0.
            res = dataclasses.replace(res, speed=_single_wave_speed(res.st, sem, c, line))
            results.append(res)
            tag = f" {w.label}" if w.label else ""
            titles.append(f"win{i}{tag} {w.t_peak*1e3:.0f} ms  [{vname}]\n"
                          f"c={abs(c):.1f} m/s (semblance {sem:.2f})")
            speeds.append(dict(window=i, label=w.label, t_peak_ms=w.t_peak * 1e3, view=vname,
                               speed_m_s=float(c), semblance=float(sem),
                               mline_length_mm=float(ml.r[-1] * 1e3)))
            print(f"    window #{i} [{vname}]: space-time {res.st.data.shape} c={c:.2f} sem={sem:.3f}")

    # --- montage: rows = windows, cols = views ---
    for row, k in zip(row_bmodes, range(0, len(results), len(views))):
        if row is not None:
            row["r0_mm"] = results[k].r0 * 1e3
    spacetime_montage(results, p["montage"], ncols=len(views), panel_titles=titles, transpose=True,
                      row_bmodes=row_bmodes, show_r0=False,
                      suptitle=f"Passive SWE -- {len(todo)} window(s) x {len(views)} views "
                               f"(M-mode: x=time, y=along-line; columns = recipes; "
                               f"black = fitted wavefront, one direction across the whole line; "
                               f"left: B-mode + M-line, yellow = r 0)")
    with open(os.path.join(p["outdir"], "passive_speeds.json"), "w") as f:
        json.dump(speeds, f, indent=1)
    print(f"done -> {p['montage']}")
    return p["montage"]


# ------------------------------------------------------------------ single-line (B-mode at the R-peak)
# Most buffer-4 frames do not show the septum clearly, so the study draws ONE line per folder on a
# B-mode frame and uses it for detection and every window. Buffer 4 is R-peak triggered (its frame
# 0 sits on an R-peak in every folder of the study), but the B-mode buffers are not - buffer 1's
# frame 0 falls anywhere in the cycle. So the line is drawn on the B-mode frame nearest an R-peak,
# found from the trigger log (swp.acquisition.triggerlog): same cardiac phase as buffer-4 frame 0.
# All grids carry per-pixel coordinates in metres, so the line maps across as is.
BMODE_BUFFER = 1
MLINE_SOURCE_JSON = "passive_general_mline.json"


def bmode_file(buffer):
    return f"CombinedData_buffer{buffer}_iq.hdf5"


def draw_bmode_mline(folder, config="configs/passive.yaml", buffer=BMODE_BUFFER, frame="rpeak",
                     label=""):
    """Prompt for the single passive M-line on one frame of a B-mode buffer (default buffer 1).

    ``frame="rpeak"`` picks the frame nearest an R-peak from the trigger log (falls back to 0 with
    a warning when the log cannot place the buffer); an int uses that frame. The line is saved as
    the general line (``mlines/passive_general_mline.npz`` + ``.png``) with its source recorded in
    ``passive_general_mline.json``. Closing the window without drawing marks the folder skipped.
    Returns True if drawn.
    """
    from .mline.select import load_bmode_frame

    cfg, p = _paths(folder, config)
    n_samples = cfg["mline"].get("n_samples", 250)
    source = dict(buffer=int(buffer), frame_request=str(frame))
    if frame == "rpeak":
        from .acquisition.triggerlog import buffer_timing
        bt = buffer_timing(folder, buffer)
        if bt is None:
            print(f"  WARNING: buffer {buffer} not found in the trigger log - using frame 0")
            frame, note = 0, "R-peak unknown"
        else:
            frame, off = bt.nearest_rpeak_frame()
            source.update(rpeak_offset_ms=round(off, 1), frame0_phase_ms=float(bt.phase_ms()[0]))
            note = f"{off:+.0f} ms from R-peak"
    else:
        frame, note = int(frame), "fixed frame"
    source["frame"] = int(frame)
    bmode_u8, coords, n = load_bmode_frame(os.path.join(p["output"], bmode_file(buffer)), frame)
    head = f"{label}{os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}"
    title = (f"{head}\npassive M-line on buffer {buffer}, frame {frame}/{n} ({note})  |  "
             f"close window = skip folder")
    try:
        _draw_line(p["general"], bmode_u8[None], coords, title, n_samples, 1.0,
                   labels=[f"buffer {buffer}, frame {frame} ({note})"])
    except SkipLine:
        _write_windows(p["windows_json"], dict(skipped_general=True))
        print("  M-line SKIPPED -> folder marked skipped")
        return False
    with open(os.path.join(p["mlines"], MLINE_SOURCE_JSON), "w") as f:
        json.dump(source, f, indent=1)
    if os.path.exists(p["windows_json"]):             # a stale skip / detection from an old line
        os.remove(p["windows_json"])
    return True


def _mline_source(p):
    path = os.path.join(p["mlines"], MLINE_SOURCE_JSON)
    if not os.path.exists(path):
        return "buffer3_frame0"                       # lines drawn before the source was recorded
    with open(path) as f:
        s = json.load(f)
    return f"buffer{s['buffer']}_frame{s['frame']}"


def process_single_line(folder, config="configs/passive.yaml", acq=None, window_ms=100.0,
                        max_events=4, overview_stride=2, pad_ms=20.0):
    """Unattended: detect bursts along the single line, reuse it for every window, process."""
    cfg, p = _paths(folder, config)
    n_samples = cfg["mline"].get("n_samples", 250)
    gen = _load_line(p["general"], n_samples)
    if acq is None:
        acq = load_acq(folder, config)
    key = dict(_detect_key(gen, cfg, window_ms, max_events, overview_stride),
               mline_source=_mline_source(p))
    st, windows = _cached_windows(p, gen, cfg, key)
    if st is None:
        _archive_window_lines(p["mlines"])
        windows = detect_windows(acq, gen, cfg, p["outdir"], window_ms, max_events,
                                 overview_stride, folder=folder)
        st = dict(key=key, windows=[dataclasses.asdict(w) for w in windows])
    if label_windows(folder, st):
        _write_windows(p["windows_json"], st)
        windows = [BurstWindow(**w) for w in st["windows"]]
    if st.get("window_mlines"):
        # Per-event lines were drawn for this detection (draw_event_mlines): keep them.
        print(f"  [M-line] using {len(st['window_mlines'])} per-event line(s)")
    else:
        # Every window uses the single line: write it under each window name so the per-window
        # machinery (and anyone reading mlines/) sees exactly what was processed.
        for i in range(len(windows)):
            npz = _window_npz(p["mlines"], i)
            np.savez(npz, points=np.asarray(gen.points, float), n_samples=n_samples)
        st.update(drawn=list(range(len(windows))), skipped=[], from_general=list(range(len(windows))))
        _write_windows(p["windows_json"], st)
    if not windows:
        print("  no bursts detected along the line")
        return None
    return process_passive_windows(folder, config, acq=acq, pad_ms=pad_ms)


def label_windows(folder, st):
    """Label each window of a windows-state dict (MVC / AVC / AK / other) from the trigger log.

    Sets ``windows[i]["label"]`` and stores the timing behind it under ``window_phases``
    (see :func:`swp.acquisition.triggerlog.label_event`). Returns True if anything changed.
    """
    from .acquisition.triggerlog import label_buffer4_events

    wins = st.get("windows") or []
    if not wins:
        return False
    labels = label_buffer4_events(folder, [w["t_peak"] for w in wins])
    if labels is None:
        return False
    changed = False
    for w, lab in zip(wins, labels):
        if w.get("label") != lab["label"]:
            w["label"] = lab["label"]
            changed = True
    if st.get("window_phases") != labels:
        st["window_phases"] = labels
        changed = True
    return changed


def event_bmode_frames(folder, windows, buffer=BMODE_BUFFER, at="t_peak"):
    """For each burst window: the B-mode frame at the same cardiac phase as the event.

    The event time in buffer 4 (``at``: ``"t_peak"`` or ``"t0"``) is placed on the trigger-log
    clock via buffer 4's frame-0 trigger, turned into a phase (ms since the preceding R-peak), and
    matched to the B-mode frame with the closest phase since *its* preceding R-peak. Returns a list
    of dicts (frame, event_phase_ms, frame_phase_ms), or None when the log cannot place a buffer.
    """
    from .acquisition.triggerlog import buffer_timing

    b4, bb = buffer_timing(folder, 4), buffer_timing(folder, buffer)
    if b4 is None or bb is None:
        return None
    ph_b = bb.phase_ms()
    out = []
    for w in windows:
        t_abs = b4.t0_ms + getattr(w, at) * 1e3
        prev = b4.r_peaks_ms[b4.r_peaks_ms <= t_abs + 0.5]
        phase = t_abs - prev.max() if prev.size else getattr(w, at) * 1e3
        k = int(np.nanargmin(np.abs(ph_b - phase)))
        out.append(dict(frame=k, event_phase_ms=round(float(phase), 1),
                        frame_phase_ms=round(float(ph_b[k]), 1)))
    return out


def draw_event_mlines(folder, config="configs/passive.yaml", buffer=BMODE_BUFFER, at="t_peak",
                      label=""):
    """Prompt for one M-line per detected window, each on the phase-matched B-mode frame.

    Needs the detection of the current general line (run :func:`process_single_line` first).
    The general line is overlaid dashed; ENTER without clicking reuses it for that window, closing
    the window skips that window. Lines go to ``mlines/passive_win<i>_mline.npz``; the frame
    choice is recorded under ``window_mlines`` in ``passive_windows.json``, which also stops
    :func:`process_single_line` from overwriting them. Returns the state dict.
    """
    from .mline.select import load_bmode_frame

    cfg, p = _paths(folder, config)
    n_samples = cfg["mline"].get("n_samples", 250)
    st, windows = read_windows(p["windows_json"])
    if st is None or not windows:
        raise SystemExit("no detected windows - run process (single line) first")
    gen = _load_line(p["general"], n_samples)
    frames = event_bmode_frames(folder, windows, buffer, at)
    if frames is None:
        raise SystemExit(f"trigger log cannot place buffer {buffer} / 4 for {folder}")
    head = f"{label}{os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}"
    ref = [("general M-line", gen.x, gen.z)]
    wm = {}
    for i, (w, fr) in enumerate(zip(windows, frames)):
        k = fr["frame"]
        bmode_u8, coords, n = load_bmode_frame(os.path.join(p["output"], bmode_file(buffer)), k)
        note = (f"event @ {w.t_peak * 1e3:.0f} ms = R+{fr['event_phase_ms']:.0f} ms; "
                f"frame R+{fr['frame_phase_ms']:.0f} ms")
        title = (f"{head}\nwindow {i + 1}/{len(windows)}: buffer {buffer} frame {k}/{n} ({note})\n"
                 f"ENTER without clicking = use general line  |  close = skip window")
        npz = _window_npz(p["mlines"], i)
        try:
            _, reused = _draw_line(npz, bmode_u8[None], coords, title, n_samples, 1.0,
                                   reference=ref, labels=[f"buffer {buffer}, frame {k}"],
                                   fallback_points=gen.points)
            wm[str(i)] = dict(buffer=buffer, **fr, from_general=bool(reused))
        except SkipLine:
            if os.path.exists(npz):
                os.remove(npz)
            wm[str(i)] = dict(buffer=buffer, **fr, skipped=True)
            print(f"  [M-line] window {i}: SKIPPED")
    st.update(window_mlines=wm,
              drawn=[int(i) for i, v in wm.items() if not v.get("skipped")],
              skipped=[int(i) for i, v in wm.items() if v.get("skipped")],
              from_general=[int(i) for i, v in wm.items() if v.get("from_general")])
    _write_windows(p["windows_json"], st)
    return st


def check_ecg_quality(folder, write=True):
    """Assess the R-peak record and warn if cardiac phases cannot be trusted -> :class:`RRCheck`.

    Called automatically at the start of the passive workflow whenever ECG data is present. A
    single spurious trigger is corrected silently by ``clean_r_peaks``; anything that leaves the
    R-peaks in doubt prints a WARNING, because a wrong phase relabels MVC as AK and falsifies the
    speed analysis downstream. The result is also written to ``<outdir>/ecg_rr_check.json`` so a
    batch run can be audited afterwards.
    """
    from .acquisition.rrcheck import assess_rr

    chk = assess_rr(folder)
    for line in chk.report():
        print(line)
    if write:
        try:
            _, p = _paths(folder, "configs/passive.yaml")
            os.makedirs(p["outdir"], exist_ok=True)
            with open(os.path.join(p["outdir"], "ecg_rr_check.json"), "w") as f:
                json.dump(chk.as_row(), f, indent=1)
        except Exception:                                            # noqa: BLE001
            pass          # the check is advisory; never let logging it break a run
    return chk


def process_passive(folder, config="configs/passive.yaml", window_ms=100.0, max_events=4,
                    pad_ms=20.0, overview_stride=2, redraw=False, cine=False):
    """Run the whole passive burst-window workflow on one folder (draw, then process).

    Args:
        folder: measurement folder (uses ``<folder>/output`` written by the beamform stage).
        config: passive config (frame_to_frame recipe).
        window_ms: width of the window placed around each detected burst (default 100 ms).
        max_events: maximum number of burst windows to keep.
        pad_ms: extra time each side of the window fed to the filters (cropped conceptually).
        overview_stride: spatial decimation for the fast burst-detection pass (1 = full res).
        redraw: force re-drawing every M-line (and re-detecting) even if saved.
    """
    acq = load_acq(folder, config)
    print(f"passive workflow on {folder}\n  {acq.summary()}")
    check_ecg_quality(folder)
    st = draw_passive_mlines(folder, config, acq=acq, window_ms=window_ms, max_events=max_events,
                             overview_stride=overview_stride, redraw=redraw, cine=cine)
    if not st.get("skipped_general") and not st.get("windows"):
        raise SystemExit("no bursts detected; try a different general M-line or lower thresholds.")
    return process_passive_windows(folder, config, acq=acq, pad_ms=pad_ms)
