"""B-mode frames of buffers 1, 3 and 4, synchronised on the ECG.

Only buffer 4 (and 2/5/6) is R-peak triggered; buffers 1 and 3 run in other heartbeats. The trigger
log in the runtime .mat places every buffer's frames against the logged R-peaks
(:mod:`swp.acquisition.triggerlog`), so each buffer contributes the frame at the SAME CARDIAC PHASE:

* general line: the frame nearest an R-peak in each buffer (buffer 4: frame 0 in the study);
* event line:   buffer 4 at the event itself; buffers 1 and 3 at the frame whose time since its
  preceding R-peak matches the event's.

Same phase is not same position: the heart sits differently in different beats (median ~0.8 mm
for buffer 1, ~3 mm for buffer 3, docs/passive_mlines.md), which is what the line transfer in
``line_gui`` corrects.

Buffer 4 (diverging waves, low SNR) is shown as the envelope averaged over a few frames around the
target (default 9 frames = ~10 ms); the anatomy hardly moves over that span and speckle noise drops.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import h5py
import numpy as np

try:                                  # zea HDF5 outputs use an hdf5plugin compression filter
    import hdf5plugin                 # noqa: F401
except ImportError:                   # pragma: no cover
    pass

_BF = "tracks/track_0/data/beamformed_data"
BUFFERS = (1, 3, 4)
AVG4 = 9                              # buffer-4 frames averaged for display / registration


@dataclass
class Panel:
    buffer: int
    env: np.ndarray            # (z, x) envelope, averaged over n_avg frames
    x_mm: np.ndarray           # ascending
    z_mm: np.ndarray           # ascending
    frame: int                 # centre frame
    n_avg: int
    n_frames: int
    phase_ms: float            # time since the preceding R-peak (NaN if unknown)
    note: str                  # how the frame was chosen

    @property
    def extent(self):
        return [self.x_mm[0], self.x_mm[-1], self.z_mm[-1], self.z_mm[0]]

    def u8(self):
        from ._light import display_8bit
        return display_8bit(self.env)


def n_frames(path):
    with h5py.File(path, "r") as f:
        return int(f[f"{_BF}/values"].shape[0])


def read_env(path, frame, n_avg=1):
    """Mean envelope of ``n_avg`` frames centred on ``frame`` -> (env, x_mm, z_mm, frame, n)."""
    with h5py.File(path, "r") as f:
        d = f[f"{_BF}/values"]
        n = d.shape[0]
        frame = int(np.clip(frame, 0, n - 1))
        i0 = int(np.clip(frame - n_avg // 2, 0, max(n - n_avg, 0)))
        v = np.asarray(d[i0:i0 + n_avg], dtype=np.float32)
        coords = np.asarray(f[f"{_BF}/coordinates"])
    env = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2).mean(axis=0)
    x = coords[0, :, 0] * 1e3
    z = coords[:, 0, 2] * 1e3
    if x[0] > x[-1]:
        x, env = x[::-1], env[:, ::-1]
    if z[0] > z[-1]:
        z, env = z[::-1], env[::-1]
    return np.ascontiguousarray(env), x.copy(), z.copy(), frame, n


def buffer4_times(path):
    with h5py.File(path, "r") as f:
        return np.asarray(f[f"{_BF}/timestamps"], float)


@lru_cache(maxsize=64)
def _timing(folder, buffer):
    from ._light import triggerlog
    try:
        return triggerlog().buffer_timing(folder, buffer)
    except Exception:                                   # noqa: BLE001 - unreadable log
        return None


def buffer3_record(folder):
    """The unwrap record of buffer 3 (swp.acquisition.unwrap) from its IQ file, or None.

    ``status`` 'unwrapped'/'chronological': frames are in acquisition order and ``frame_time_ms``
    holds their frame-centre times on the trigger-log clock; otherwise ('ambiguous', or never
    unwrapped) the stored order is unknown and buffer 3 is chosen by anatomy.
    """
    import os
    from .store import Paths
    path = Paths(folder).bmode(3)
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        if "custom/unwrap_status" not in f:
            return None
        rec = dict(status=f["custom/unwrap_status"][()].decode()
                   if isinstance(f["custom/unwrap_status"][()], bytes) else str(f["custom/unwrap_status"][()]))
        if "custom/frame_time_ms" in f:
            rec["frame_time_ms"] = np.asarray(f["custom/frame_time_ms"], float)
    return rec


def centre_times(folder, buffer):
    """(frame-centre times ms on the trigger-log clock, R-peaks ms) of one buffer, or (None, None).

    The trigger fires on the FIRST transmit of a frame, so a frame's centre is half a frame period
    later (buffer 1: ~5.7 ms, buffer 3: ~19.7 ms, buffer 4: ~0.5 ms). Buffer 3 uses the times
    stored by the unwrap; without a (resolved) unwrap its frame order is unknown -> (None, None).
    """
    bt = _timing(str(folder), buffer)
    if bt is None:
        return None, None
    if buffer == 3:
        rec = buffer3_record(str(folder))
        if not rec or rec["status"] not in ("unwrapped", "chronological") or "frame_time_ms" not in rec:
            return None, np.asarray(bt.r_peaks_ms)
        return rec["frame_time_ms"], np.asarray(bt.r_peaks_ms)
    return bt.frame_times() + bt.frame_ms / 2, np.asarray(bt.r_peaks_ms)


def _phase(t, r):
    prev = r[r <= t + 0.5]
    return float(t - prev.max()) if prev.size else float("nan")


def rpeak_targets(folder):
    """{buffer: (frame, phase_ms, note)} - the frame whose centre is nearest an R-peak."""
    out = {}
    for b in BUFFERS:
        t, r = centre_times(str(folder), b)
        if t is None or r is None or not r.size:
            why = "buffer 3 order unknown" if b == 3 and r is not None else "no trigger log"
            out[b] = (0, float("nan"), f"{why}: frame 0")
            continue
        d = t[:, None] - r[None, :]
        k, j = np.unravel_index(np.argmin(np.abs(d)), d.shape)
        out[b] = (int(k), float(d[k, j]), f"R-peak {d[k, j]:+.0f} ms")
    return out


def event_targets(folder, t_peak_s, t4):
    """{buffer: (frame, phase_ms, note)} for an event at buffer-4 time ``t_peak_s``."""
    k4 = int(np.argmin(np.abs(np.asarray(t4) - t_peak_s)))
    b4 = _timing(str(folder), 4)
    if b4 is None:
        return {4: (k4, float("nan"), "event (no trigger log)"),
                1: (0, float("nan"), "no trigger log: frame 0"),
                3: (0, float("nan"), "no trigger log: frame 0")}
    r = np.asarray(b4.r_peaks_ms)
    phase = _phase(b4.t0_ms + t_peak_s * 1e3, r)
    out = {4: (k4, phase, f"event, R+{phase:.0f} ms")}
    for b in (1, 3):
        t, rr = centre_times(str(folder), b)
        if t is None or not np.isfinite(phase):
            why = "buffer 3 order unknown" if b == 3 and rr is not None else "no trigger log"
            out[b] = (0, float("nan"), f"{why}: frame 0")
            continue
        ph = np.array([_phase(ti, rr) for ti in t])
        if not np.isfinite(ph).any():
            out[b] = (0, float("nan"), "no R-peak before this buffer: frame 0")
            continue
        k = int(np.nanargmin(np.abs(ph - phase)))
        out[b] = (k, float(ph[k]), f"R+{ph[k]:.0f} ms ({ph[k] - phase:+.0f} vs event)")
    return out


def frame_phase(folder, buffer, frame):
    t, r = centre_times(str(folder), buffer)
    if t is None:
        return float("nan")
    return _phase(t[int(np.clip(frame, 0, len(t) - 1))], r)


def load_panel(folder, path, buffer, frame, note, phase_ms=float("nan"), n_avg=None):
    n_avg = (AVG4 if buffer == 4 else 1) if n_avg is None else n_avg
    env, x, z, k, n = read_env(path, frame, n_avg)
    return Panel(buffer, env, x, z, k, n_avg, n, phase_ms, note)


def load_panels(folder, paths, targets):
    """{buffer: Panel or None} for ``targets`` from :func:`rpeak_targets` / :func:`event_targets`."""
    import os
    ok, why = ecg_check(str(folder))
    out = {}
    for b in (4, 1, 3):                     # 4 first: without a valid ECG, buffer 1 is matched to it
        path = paths.bmode(b)
        if not os.path.exists(path):
            out[b] = None
            continue
        k, ph, note = targets[b]
        if not ok and b == 4:
            note = f"NO VALID ECG ({why}) - {note}"
        if not ok and b == 1 and out.get(4) is not None:
            try:
                k, c1 = match_frame(path, out[4])
                note, ph = f"NO VALID ECG: matched to buffer 4 (r {c1:.2f})", float("nan")
            except Exception as exc:                            # noqa: BLE001
                note = f"NO VALID ECG, match failed ({exc}): {note}"
        elif b == 3 and (not ok or not _buffer3_timed(folder)) and out.get(1) is not None:
            # buffer 3 not (resolvably) unwrapped, or no valid ECG: choose it by anatomy
            rec = buffer3_record(str(folder))
            why3 = ("NO VALID ECG" if not ok else
                    f"buffer 3 {rec['status'] if rec else 'not unwrapped'}")
            try:
                k3, c3 = match_frame(path, out[1])
                note = f"{why3}: matched to buffer 1 (r {c3:.2f})"
                k, ph = k3, float("nan")
            except Exception as exc:                            # noqa: BLE001 - fall back to the log
                note += f" (match failed: {exc})"
        out[b] = load_panel(folder, path, b, k, note, ph)
    return {b: out.get(b) for b in BUFFERS}


@lru_cache(maxsize=64)
def ecg_check(folder):
    """(trustworthy, reason) of the R-peak record (swp.acquisition.rrcheck). A fixed-rate pulse
    train or a sparse record makes every "R-peak" / "R+x ms" frame choice meaningless."""
    from ._light import rrcheck
    try:
        c = rrcheck().assess_rr(folder)
        return bool(c.trustworthy), f"{c.status}: {c.quality}"
    except Exception as exc:                                    # noqa: BLE001
        return False, f"check failed: {exc}"


def _buffer3_timed(folder):
    rec = buffer3_record(str(folder))
    return bool(rec) and rec["status"] in ("unwrapped", "chronological") and "frame_time_ms" in rec


# ------------------------------------------------------------------ frames by anatomy
# Buffer 3 is the focused LIVE loop, stored in a circular receive buffer whose head VSX does not
# record. swp.acquisition.unwrap puts it in chronological order and stores its frame times; until a
# folder is unwrapped (or when the unwrap is 'ambiguous') buffer 3 is chosen by ANATOMY - the frame
# that looks most like the correctly timed buffer-1 frame. Without a valid ECG, buffer 1 is chosen
# the same way against buffer 4. (docs/buffer3_unwrap.md)
_GX = np.arange(-25, 25.01, 0.8)
_GZ = np.arange(20, 100.01, 0.8)


def _anatomy(env, x_mm, z_mm):
    from ._light import transfer
    T = transfer()
    a = T.anatomy(T.resample(env, x_mm, z_mm, _GX, _GZ), 0.8)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-12)


def match_frame(path, ref: Panel):
    """(frame, correlation) of the frame in ``path`` whose anatomy best matches panel ``ref``."""
    with h5py.File(path, "r") as f:
        v = np.asarray(f[f"{_BF}/values"], dtype=np.float32)
        coords = np.asarray(f[f"{_BF}/coordinates"])
    env = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    x, z = coords[0, :, 0] * 1e3, coords[:, 0, 2] * 1e3
    if x[0] > x[-1]:
        x, env = x[::-1], env[:, :, ::-1]
    if z[0] > z[-1]:
        z, env = z[::-1], env[:, ::-1]
    r = _anatomy(ref.env, ref.x_mm, ref.z_mm)
    c = np.array([float((_anatomy(e, x, z) * r).sum()) for e in env])
    k = int(np.argmax(c))
    return k, float(c[k])
