"""Automatic MVC / AVC windows on the general line's whole-recording space-time.

Fitted to the reader's ROIs of 2026-10-01 (scripts/passive_roi.py; 27 folders, 42 MVC + 24 AVC
ROIs in the 23 with a usable ECG; study/analysis/passive_roi_auto.py, leave-one-subject-out):

1. **Search windows** from the R-peaks (as ``swp.mline.select.phase_targets``): MVC R+0-150 ms,
   AVC QS2 +- 120 ms (Weissler QS2 = 546 - 2.1 HR). Kept when at least half inside the recording.
2. **Burst:** the peak of the short-time (20 ms) along-line velocity energy inside the search
   window, weighted by a Gaussian prior (sigma 45 ms) around the typical timing (MVC R + 43 ms,
   AVC QS2 centre - 28 ms). The valve closure is a sharp, short burst; scores averaged over the
   whole 100-120 ms window smear it out and drift to broader activity next to it.
3. **Window:** a fixed 120 ms, centred on the burst + a small offset (MVC +6 ms, AVC -2 ms, the
   reader's ROI centre relative to the burst), shifted inside the recording at the ends.
4. **Presence:** the slant-stack semblance of the window (``passive_screen.window_scores``);
   below 0.3 the window is ``screened`` (the 3 AVC search windows the reader left empty scored
   0.21-0.24, every marked one >= 0.31; every MVC >= 0.37).

Agreement (leave-one-subject-out): the 120 ms window fully contains 24/24 AVC and 41/42 MVC ROIs,
centre error median 3 ms (AVC) and 12 ms (MVC, where the record start pins the first window).
The one miss (C000000021) is an ROI 187 ms after the R-peak, at the end of the MVC search window.

Needs R-peaks: folders without a usable ECG get no automatic windows (mark them by hand,
``passive_roi.py session --click-ms 120``). numpy only, apart from the semblance.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

WINDOW_S = 0.120
PEAK_S = 0.020
SIGMA_S = 0.045
DELAY_S = {"MVC": 0.043, "AVC": -0.028}      # prior: MVC from the R-peak, AVC from the QS2 centre
OFFSET_S = {"MVC": 0.006, "AVC": -0.002}     # window centre - burst peak
MIN_INSIDE = 0.5
SCREEN_MIN = 0.3


def expected_windows(r_peaks_s, rr_s, mvc_max_ms=150.0, avc_tol_ms=120.0):
    """[(name, lo_s, hi_s)]: a zea-free copy of ``swp.mline.select.phase_targets`` (whose module
    imports zea; tests/test_manual.py checks they agree)."""
    hr = 60.0 / rr_s if rr_s and np.isfinite(rr_s) and rr_s > 0 else np.nan
    qs2 = (546.0 - 2.1 * hr) * 1e-3 if np.isfinite(hr) else np.nan
    out = []
    for r in np.asarray(r_peaks_s, float):
        out.append(("MVC", r, r + mvc_max_ms * 1e-3))
        if np.isfinite(qs2):
            out.append(("AVC", r + qs2 - avc_tol_ms * 1e-3, r + qs2 + avc_tol_ms * 1e-3))
    return out


def short_energy(v, t, peak_s=PEAK_S):
    """Along-line mean squared velocity, box-smoothed over ``peak_s``."""
    e = np.mean(np.asarray(v, float) ** 2, axis=1)
    return uniform_filter1d(e, max(1, int(round(peak_s / np.median(np.diff(t))))), mode="nearest")


def valve_windows(v, r, t, r_peaks_s, rr_s, window_s=WINDOW_S, peak_s=PEAK_S, sigma_s=SIGMA_S,
                  delay_s=None, offset_s=None, min_inside=MIN_INSIDE, screen_min=SCREEN_MIN,
                  score=True):
    """Automatic MVC / AVC windows of a (n_t, n_r) velocity space-time ``v`` (t in s, buffer-4
    clock; r in m) -> list of dicts sorted in time:

    label, search_lo, search_hi, t_burst, t0, t1, t_mid (s), sem, c (slant-stack semblance and
    signed speed of the window; NaN when ``score`` is False), screened.
    """
    delay_s = {**DELAY_S, **(delay_s or {})}
    offset_s = {**OFFSET_S, **(offset_s or {})}
    t = np.asarray(t, float)
    rp = np.asarray(r_peaks_s, float)
    if rp.size == 0 or not rr_s:
        return []
    es = short_energy(v, t, peak_s)
    out = []
    for name, lo, hi in expected_windows(rp, rr_s):
        a, b = max(lo, t[0]), min(hi, t[-1])
        if (b - a) / (hi - lo) < min_inside:
            continue
        anchor = rp[rp <= lo + 1e-9].max() if name == "MVC" else 0.5 * (lo + hi)
        m = (t >= a) & (t <= b)
        prior_c = float(np.clip(anchor + delay_s[name], a, b))
        s = es[m] * np.exp(-0.5 * ((t[m] - prior_c) / sigma_s) ** 2)
        t_burst = float(t[m][int(np.argmax(s))])
        w0 = float(np.clip(t_burst + offset_s[name] - window_s / 2, t[0], t[-1] - window_s))
        rec = dict(label=name, search_lo=lo, search_hi=hi, t_burst=t_burst, t0=w0, t1=w0 + window_s,
                   t_mid=w0 + window_s / 2, sem=np.nan, c=np.nan, screened=False)
        if score:
            from .passive_screen import window_scores
            q = window_scores(np.asarray(v, float), np.asarray(r, float), t, rec["t_mid"], half=window_s / 2)
            rec.update(sem=q["sem"], c=q["c"], screened=bool(not np.isfinite(q["sem"]) or q["sem"] < screen_min))
        out.append(rec)
    return sorted(out, key=lambda w: w["t0"])
