"""Screening passive event windows by a propagation score on the general M-line.

Validated on the manual study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md):

* The slant-stack semblance of the default velocity view, computed on the GENERAL line in a detected
  window, predicts the reader's confidence >= 2 with AUC 0.91 (97 windows) - nearly as well as the
  same score on the drawn event line (0.94), but available before any event line is drawn.
* The energy detector searched a second AVC after the end of the ~1 s recording (7 of the 30
  zero-scored windows; peaks on masked, zero energy).

What is used (detect.picker: energy, the default of configs/passive_manual.yaml):

1. drop a search window that lies mostly outside the usable recording
   (``detect.min_inside``, in :func:`swp.mline.select.detect_phase_windows`);
2. keep picking the time by displacement energy;
3. score each picked window here (:func:`screen_windows`); below ``screen_min`` it is marked
   ``screened`` and no event line is asked. At the energy-picked times this is the validated use:
   on the 97 scored windows a 0.3 cut removes 12 of the 30 zero-scored ones for 1 usable.

What was tried and is NOT the default (detect.picker: semblance, :func:`pick_windows`): picking the
time of highest semblance inside each phase window. The dry run on the same 27 folders
(study/analysis/passive_detector_v2_check.py) kept only 34 of the 57 usable windows, screened 3 of
102 (the maximum over a 150-240 ms search window inflates the score), and 41 of 99 asked windows
sat on near-vertical bands (automatic speed >= 6 m/s) - semblance without flat-band removal rewards
in-phase motion. It stays available for further work, not for the study.
"""
from __future__ import annotations

import numpy as np

HALF_S = 0.050          # half the 100 ms event window
PAD_S = 0.020           # context either side, as the worker processes a window (worker.PAD_S)
EDGE_S = 0.030          # centres this close to the record ends are not used (filter transients)
STEP_S = 0.005          # spacing of the score track
MIN_SEP_S = 0.150       # as detect_phase_windows


def window_scores(v, r, t, tc, half=HALF_S, pad=PAD_S):
    """Scores of the window centred on ``tc`` of a (n_t, n_r) space-time ``v``.

    sem    slant-stack semblance (as processed.json 'auto', swp.viz.metrics.slant_stack_speed);
    c      its signed speed [m/s];
    burst  RMS inside the window / RMS in the pads.
    """
    from .viz.metrics import slant_stack_speed
    from .viz.speed.spacetime import SpaceTime

    m = (t >= tc - half - pad) & (t <= tc + half + pad)
    inside = (t >= tc - half) & (t <= tc + half)
    if m.sum() < 20:
        return dict(sem=np.nan, c=np.nan, burst=np.nan)
    sem, c = slant_stack_speed(SpaceTime(v[m], r, t[m], "velocity"), None, cmin=1.0, cmax=20.0,
                               remove_flat=False)
    pads = m & ~inside
    rms_pad = np.sqrt(np.mean(v[pads] ** 2)) if pads.any() else np.nan
    burst = np.sqrt(np.mean(v[inside] ** 2)) / (rms_pad + 1e-30)
    return dict(sem=float(sem), c=float(c), burst=float(burst))


def screen_track(v, r, t, step=STEP_S, edge=EDGE_S):
    """The score track over the whole recording -> dict(t, sem, c, burst) (arrays, seconds)."""
    v, r, t = np.asarray(v, float), np.asarray(r, float), np.asarray(t, float)
    tc = np.arange(t[0] + edge, t[-1] - edge + 1e-9, step)
    rows = [window_scores(v, r, t, x) for x in tc]
    return dict(t=tc, sem=np.array([q["sem"] for q in rows]), c=np.array([q["c"] for q in rows]),
                burst=np.array([q["burst"] for q in rows]))


def general_line_track(acq, gen_mline, cfg, view="velocity gauss"):
    """Run ``view`` (a run.views entry of ``cfg``) along the general line over the whole
    recording and return its :func:`screen_track`."""
    from .passive import _build_views
    from .viz.pipeline import run_pipeline

    views = dict(_build_views(cfg, acq))
    if view not in views:
        raise KeyError(f"screen view {view!r} not in run.views ({list(views)})")
    res = run_pipeline(acq, gen_mline, views[view], focus=None)
    return screen_track(np.asarray(res.st.data), np.asarray(res.st.r), np.asarray(res.st.t))


def screen_windows(windows, track, screen_min=0.3):
    """Score already-picked windows from a :func:`screen_track` (nearest track centre, <= 2.5 ms
    away at the default step) and mark those below ``screen_min`` as screened. In place; returns
    ``windows``."""
    t = np.asarray(track["t"], float)
    for w in windows:
        k = int(np.argmin(np.abs(t - w.t_peak)))
        s = float(track["sem"][k])
        w.screen = s if np.isfinite(s) else None
        w.burst = float(track["burst"][k]) if np.isfinite(track["burst"][k]) else None
        w.screened = bool(w.screen is None or w.screen < screen_min)
    return windows


def _inside_fraction(lo, hi, span):
    return max(0.0, min(hi, span[1]) - max(lo, span[0])) / (hi - lo)


def pick_windows(track, r_peaks_s=None, rr_s=None, window_ms=100.0, max_events=4,
                 min_inside=0.5, screen_min=0.3, min_separation_s=MIN_SEP_S, energy=None):
    """Event windows from a :func:`screen_track` -> [BurstWindow] sorted in time.

    ``r_peaks_s`` on the track's clock (buffer-4 frame 0), ``rr_s`` the reference RR interval; with
    no usable R-peaks every window is a top-up. ``energy`` = optional (t_s, e) of the detection
    overview, only to fill the legacy ``score`` field (peak energy).
    """
    from .mline.select import BurstWindow, _detect_peaks, phase_targets

    t = np.asarray(track["t"], float)
    sem = np.nan_to_num(np.asarray(track["sem"], float), nan=-1.0)
    burst = np.asarray(track["burst"], float)
    span = (float(t[0]), float(t[-1])) if t.size else (0.0, 0.0)
    half = 0.5 * window_ms * 1e-3

    found = []                                            # (expect, index into the track)
    if r_peaks_s is not None and len(r_peaks_s) and rr_s and np.isfinite(rr_s):
        for name, lo, hi in phase_targets(r_peaks_s, rr_s):
            if _inside_fraction(lo, hi, span) < min_inside:
                continue                                   # e.g. a 2nd AVC after the recording
            idx = np.where((t >= lo) & (t <= hi))[0]
            if idx.size < 3:
                continue
            k = idx[int(np.argmax(sem[idx]))]
            if any(abs(t[k] - t[q]) < min_separation_s for _, q in found):
                continue
            found.append((name, k))
        found.sort(key=lambda nk: -sem[nk[1]])
        found = found[:max_events]
    if len(found) < max_events:                            # top up with the best remaining score
        taken = [k for _, k in found]
        peaks = _detect_peaks(np.clip(sem, 0, None), t, min_separation_s, max_events * 3,
                              prominence_frac=0.1)
        for k in sorted(peaks, key=lambda q: -sem[q]):     # strongest first (returned in time order)
            if len(found) >= max_events:
                break
            if all(abs(t[k] - t[q]) >= min_separation_s for q in taken):
                found.append(("", int(k)))
                taken.append(int(k))
    found.sort(key=lambda nk: t[nk[1]])

    windows = []
    for name, k in found:
        e_pk = float("nan")
        if energy is not None:
            et, ee = (np.asarray(a, float) for a in energy)
            e_pk = float(ee[int(np.argmin(np.abs(et - t[k])))])
        s = float(sem[k]) if sem[k] >= 0 else float("nan")
        windows.append(BurstWindow(t_peak=float(t[k]), t0=float(max(span[0] - EDGE_S, t[k] - half)),
                                   t1=float(min(span[1] + EDGE_S, t[k] + half)), score=e_pk,
                                   expect=name, screen=s, burst=float(burst[k]),
                                   screened=bool(not np.isfinite(s) or s < screen_min)))
    return windows
