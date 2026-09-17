"""Buffer timing relative to the ECG R-peaks, from the trigger log in the runtime .mat.

``AcquisitionParametersAndECG.mat`` carries ``ECG_data_raw``: a text dump with four columns
written one after another - ``Time``, ``Signal``, ``ECG_trigger`` (R-peak times, ms) and
``Vera_trigger`` (one timestamp per Verasonics frame trigger, ms, a circular log). Consecutive
frame triggers at a buffer's frame period form that buffer's acquisition block, so the block can
be placed against the R-peaks and a buffer frame can be matched to a cardiac phase.

Only buffers with a distinctive cadence are located: the widebeam cine (buffer 1, ~11.3 ms),
the diverging-wave stream (buffer 4, ~1.08 ms) and the focused cine (buffer 3, ~39.4 ms, the
LAST ``Nframes`` of the live run). Timestamps are whole milliseconds, so matching uses a
tolerance and the last block of the right length is taken.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUNTIME_MAT = "AcquisitionParametersAndECG.mat"
BUFFER_STRUCT = {1: "Bmode_WB", 3: "Bmode_FC", 4: "Bmode_DW"}


@dataclass
class BufferTiming:
    buffer: int
    n_frames: int
    frame_ms: float            # nominal frame period (1000 / ActualFPS)
    t0_ms: float               # trigger time of frame 0 (log clock)
    r_peaks_ms: np.ndarray     # R-peak times (log clock), sorted
    source: str                # "block" (a matched trigger block) or "tail" (end of a live run)

    def frame_times(self):
        return self.t0_ms + np.arange(self.n_frames) * self.frame_ms

    def phase_ms(self):
        """Per frame: time since the preceding R-peak (NaN before the first logged R-peak)."""
        t = self.frame_times()
        out = np.full(t.shape, np.nan)
        for i, ti in enumerate(t):
            prev = self.r_peaks_ms[self.r_peaks_ms <= ti + 0.5]
            if prev.size:
                out[i] = ti - prev.max()
        return out

    def nearest_rpeak_frame(self):
        """(frame index, signed offset ms) of the frame closest to any logged R-peak."""
        t = self.frame_times()
        d = t[:, None] - self.r_peaks_ms[None, :]
        i, j = np.unravel_index(np.argmin(np.abs(d)), d.shape)
        return int(i), float(d[i, j])


def read_log(folder):
    """-> (r_peaks_ms sorted, frame_triggers_ms sorted, params dict) or None if absent."""
    import scipy.io as sio

    path = Path(folder) / RUNTIME_MAT
    if not path.is_file():
        return None
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "ECG_data_raw" not in m:
        return None
    lines = str(m["ECG_data_raw"]).splitlines()
    # Two formats: early acquisitions label columns in ms ("ECG_trigger"), later ones in µs
    # ("ECG_trigger_us"). Everything is returned in ms.
    heads, scale = {}, {}
    for i, name in enumerate(lines):
        key = name.strip()
        base = key[:-3] if key.endswith("_us") else key.rsplit("_V", 1)[0] if key.endswith("_V") else key
        if base in ("Time", "Signal", "ECG_trigger", "Vera_trigger"):
            heads[base] = i
            scale[base] = 1e-3 if key.endswith("_us") else 1.0
    order = sorted(heads.values())

    def column(name):
        start = heads[name] + 1
        nxt = [i for i in order if i > heads[name]]
        stop = nxt[0] if nxt else len(lines)
        return np.array([float(x) for x in lines[start:stop] if x.strip()], float) * scale[name]

    if "ECG_trigger" not in heads or "Vera_trigger" not in heads:
        return None
    r = np.sort(column("ECG_trigger"))
    trig = np.sort(column("Vera_trigger"))
    params = {}
    for b, s in BUFFER_STRUCT.items():
        if s in m:
            params[b] = dict(fps=float(m[s].ActualFPS), n=int(m[s].Nframes))
    return r, trig, params


def _runs(trig, period, tol):
    d = np.diff(trig)
    ok = np.abs(d - period) <= tol
    runs, i = [], 0
    while i < ok.size:
        if ok[i]:
            j = i
            while j < ok.size and ok[j]:
                j += 1
            runs.append((i, j))              # triggers i..j inclusive
            i = j
        else:
            i += 1
    return runs


MVC_MAX_MS = 150.0        # MVC: within this long after an R-peak
AK_BEFORE_R_MS = 200.0    # atrial kick: within this long before the next R-peak
AVC_TOL_MS = 120.0        # AVC: within this of the Weissler QS2 estimate


def median_rr_ms(r_peaks_ms):
    rr = np.diff(np.sort(np.asarray(r_peaks_ms, float)))
    rr = rr[(rr > 300) & (rr < 2000)]
    return float(np.median(rr)) if rr.size else float("nan")


def label_event(t_ms, r_peaks_ms, rr_ms=None):
    """Cardiac label of an event at log time ``t_ms``.

    Rules (first match): ``MVC`` <= 150 ms after an R-peak; ``AK`` (atrial kick / atrial
    contraction) <= 200 ms before the next R-peak (estimated from the median RR if not logged);
    ``AVC`` within 120 ms of the end of systole, QS2 ~ 546 - 2.1 HR ms (Weissler; measured from Q,
    ~R); otherwise ``other``. Returns a dict with the label and the timing it was based on.
    """
    r = np.sort(np.asarray(r_peaks_ms, float))
    rr = median_rr_ms(r) if rr_ms is None else rr_ms
    prev, nxt = r[r <= t_ms + 0.5], r[r > t_ms + 0.5]
    phase = t_ms - prev.max() if prev.size else float("nan")
    to_next = (nxt.min() - t_ms) if nxt.size else (rr - phase if np.isfinite(rr) else float("nan"))
    hr = 60000.0 / rr if np.isfinite(rr) and rr > 0 else float("nan")
    qs2 = 546.0 - 2.1 * hr if np.isfinite(hr) else float("nan")
    if np.isfinite(phase) and phase <= MVC_MAX_MS:
        label = "MVC"
    elif np.isfinite(to_next) and to_next <= AK_BEFORE_R_MS:
        label = "AK"
    elif np.isfinite(phase) and np.isfinite(qs2) and abs(phase - qs2) <= AVC_TOL_MS:
        label = "AVC"
    else:
        label = "other"
    return dict(label=label, phase_ms=round(float(phase), 1), to_next_r_ms=round(float(to_next), 1),
                rr_ms=round(float(rr), 1), hr_bpm=round(float(hr), 1), qs2_ms=round(float(qs2), 1),
                next_r_logged=bool(nxt.size))


RR_PLAUSIBLE_MS = (400.0, 1600.0)     # 37.5-150 bpm
BEAT_TOL = (0.75, 1.35)               # beat containing an event, relative to the reference RR


def clean_r_peaks(r_peaks_ms):
    """-> (kept R-peaks, reference RR ms) or (None, None) when the log holds no usable ECG.

    Some logs are not an ECG at all (a regular 240 ms trigger) and some contain spurious extra
    triggers between beats. The reference RR is the median of the plausible intervals; a trigger
    arriving sooner than 0.6x that after the last kept one is dropped.
    """
    r = np.sort(np.asarray(r_peaks_ms, float))
    rr = np.diff(r)
    plaus = rr[(rr >= RR_PLAUSIBLE_MS[0]) & (rr <= RR_PLAUSIBLE_MS[1])]
    if plaus.size < 3 or plaus.size < 0.5 * rr.size:
        return None, None
    ref = float(np.median(plaus))
    kept = [r[0]]
    for t in r[1:]:
        if t - kept[-1] >= 0.6 * ref:
            kept.append(t)
    return np.array(kept), ref


def label_buffer4_events(folder, t_s):
    """Labels for events at buffer-4 times ``t_s`` (seconds from buffer-4 frame 0), or None.

    Events are labelled only when the ECG log is usable (:func:`clean_r_peaks`) and the beat
    containing the event has a plausible length; otherwise the label is ``"?"`` with a ``reason``.
    """
    b4 = buffer_timing(folder, 4)
    if b4 is None:
        return None
    r, ref = clean_r_peaks(b4.r_peaks_ms)
    out = []
    for t in t_s:
        t_abs = b4.t0_ms + t * 1e3
        if r is None:
            out.append(dict(label="?", reason="no usable ECG in the trigger log"))
            continue
        lab = label_event(t_abs, r, ref)
        prev, nxt = r[r <= t_abs + 0.5], r[r > t_abs + 0.5]
        if prev.size and nxt.size:
            beat = nxt.min() - prev.max()
            if not BEAT_TOL[0] * ref <= beat <= BEAT_TOL[1] * ref:
                lab.update(label="?", reason=f"beat {beat:.0f} ms vs reference RR {ref:.0f} ms")
        out.append(lab)
    return out


def buffer_timing(folder, buffer):
    """Locate ``buffer`` (1, 3 or 4) in the trigger log. Returns :class:`BufferTiming` or None."""
    log = read_log(folder)
    if log is None or buffer not in log[2]:
        return None
    r, trig, params = log
    n, frame_ms = params[buffer]["n"], 1000.0 / params[buffer]["fps"]
    tol = max(1.5, 0.08 * frame_ms)
    runs = [(i, j) for i, j in _runs(trig, frame_ms, tol) if j - i + 1 >= n]
    if not runs:
        return None
    i, j = runs[-1]
    if j - i + 1 == n:
        return BufferTiming(buffer, n, frame_ms, float(trig[i]), r, "block")
    # a longer run (live imaging): the stored frames are its last n
    return BufferTiming(buffer, n, frame_ms, float(trig[j - n + 1]), r, "tail")
