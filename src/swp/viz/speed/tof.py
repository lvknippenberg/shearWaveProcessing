"""Time-of-flight shear-wave-speed estimators on the M-line space-time plot.

Each estimates speed separately on the two sides of the push origin ``r0`` (the wave travels
outward in both directions) and returns predicted wavefront arrival times for plotting.
Speeds are constrained to a physical range (default 1-10 m/s).  See docs/literature_review.md
sec. 5.1: slant-stack/Radon (robust, no peak-picking), TOF cross-correlation, and TTP+RANSAC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .spacetime import SpaceTime

CMIN, CMAX = 1.0, 10.0


@dataclass
class SpeedResult:
    method: str
    c_pos: float
    c_neg: float
    quality_pos: float
    quality_neg: float
    r: np.ndarray
    t_pred_pos: np.ndarray
    t_pred_neg: np.ndarray

    @property
    def c_mean(self) -> float:
        vals = [c for c in (self.c_pos, self.c_neg)
                if np.isfinite(c) and CMIN <= c <= CMAX]
        return float(np.mean(vals)) if vals else np.nan

    @property
    def quality(self) -> float:
        return float(max(self.quality_pos, self.quality_neg))

    def label(self) -> str:
        def fmt(c):
            return f"{c:.2f}" if np.isfinite(c) else "--"
        return (f"{self.method}: {fmt(self.c_pos)}/{fmt(self.c_neg)} m/s "
                f"(mean {fmt(self.c_mean)}, q={self.quality:.2f})")


def _sides(st: SpaceTime, r0: float, d_min: float = 0.0, d_max: float = np.inf):
    """Boolean masks for the +/- sides of the origin, restricted to a distance window.

    The distance window excludes the near-origin static push (``d_min``) and the attenuated,
    noise-dominated far field (``d_max``) -- essential for stable TOF on real data.
    """
    r = st.r
    d = np.abs(r - r0)
    win = (d >= d_min) & (d <= d_max)
    return (r >= r0) & win, (r < r0) & win, d


# --------------------------------------------------------------------------- #
# Slant-stack / Radon-sum
# --------------------------------------------------------------------------- #
def _temporal_envelope(traces: np.ndarray) -> np.ndarray:
    """Polarity-independent temporal envelope (|Hilbert| along time), demeaned per column.

    Demeaning removes the near-static push/relaxation (a temporal-DC feature) so the
    slant-stack locks onto the *propagating* wavefront rather than the trivial c->min shift.
    """
    from scipy.signal import hilbert
    env = np.abs(hilbert(traces, axis=0))
    return env - env.mean(axis=0, keepdims=True)


def _slant_side(traces: np.ndarray, d: np.ndarray, t: np.ndarray,
                speeds: np.ndarray) -> Tuple[float, float, float]:
    """Return (best_speed, quality, arrival_time_at_origin) for one side.

    Each column is reduced to a demeaned temporal envelope, aligned by -d/c, and summed;
    the speed maximising the coherent stack energy wins.  Column selection (distance window)
    is applied by the caller.
    """
    if traces.shape[1] < 4:
        return np.nan, 0.0, np.nan
    T, R = traces.shape
    dt = t[1] - t[0]
    base = np.arange(T)
    env = _temporal_envelope(traces)
    energy = np.full(speeds.size, -np.inf)
    stacks = np.zeros((speeds.size, T))
    for i, c in enumerate(speeds):
        shift = d / c / dt                     # frames to advance each column
        # only stack columns whose wavefront actually arrives within the recording
        valid = shift <= (T - 1)
        if valid.sum() < 4:
            continue
        stacked = np.zeros(T)
        for j in np.where(valid)[0]:
            stacked += np.interp(base + shift[j], base, env[:, j])
        stacked /= valid.sum()                 # mean: fair across candidate speeds
        stacks[i] = stacked
        energy[i] = stacked.max() - np.median(stacked)   # coherent-peak contrast
    if not np.any(np.isfinite(energy)):
        return np.nan, 0.0, np.nan
    ibest = int(np.argmax(energy))
    emed = np.median(energy[np.isfinite(energy)])
    quality = float((energy[ibest] - emed) / (abs(energy[ibest]) + 1e-20))
    t0 = float(t[np.argmax(stacks[ibest])])
    return float(speeds[ibest]), quality, t0


def slant_stack_speed(st: SpaceTime, r0: float, n_speeds: int = 181,
                      d_min: float = 2.0e-3, d_max: float = 14.0e-3) -> SpeedResult:
    speeds = np.linspace(CMIN, CMAX, n_speeds)
    pos, neg, dist = _sides(st, r0, d_min, d_max)
    r = st.r
    res = {}
    for name, mask in (("pos", pos), ("neg", neg)):
        if mask.sum() < 4:
            res[name] = (np.nan, 0.0, np.nan)
            continue
        res[name] = _slant_side(st.data[:, mask], dist[mask], st.t, speeds)
    c_pos, q_pos, t0_pos = res["pos"]
    c_neg, q_neg, t0_neg = res["neg"]
    t_pred_pos = np.where(pos, t0_pos + dist / c_pos, np.nan) if np.isfinite(c_pos) else np.full(r.size, np.nan)
    t_pred_neg = np.where(neg, t0_neg + dist / c_neg, np.nan) if np.isfinite(c_neg) else np.full(r.size, np.nan)
    return SpeedResult("radon", c_pos, c_neg, q_pos, q_neg, r, t_pred_pos, t_pred_neg)


# --------------------------------------------------------------------------- #
# TOF cross-correlation of adjacent time profiles
# --------------------------------------------------------------------------- #
def _theil_speed(d: np.ndarray, tarr: np.ndarray) -> Tuple[float, float, float]:
    """Robust fit tarr = t0 + d/c; return (c, quality=R^2-ish, t0)."""
    from scipy.stats import theilslopes
    good = np.isfinite(tarr) & np.isfinite(d)
    if good.sum() < 4:
        return np.nan, 0.0, np.nan
    slope, intercept, lo, hi = theilslopes(tarr[good], d[good])
    if not np.isfinite(slope) or abs(slope) < 1e-9:
        return np.nan, 0.0, np.nan
    c = 1.0 / slope
    pred = intercept + slope * d[good]
    ss_res = np.sum((tarr[good] - pred) ** 2)
    ss_tot = np.sum((tarr[good] - tarr[good].mean()) ** 2) + 1e-20
    quality = float(max(0.0, 1.0 - ss_res / ss_tot))
    return float(c), quality, float(intercept)


def _xcorr_lag(a: np.ndarray, b: np.ndarray, dt: float) -> float:
    """Sub-sample lag (s) of b relative to a by parabolic-refined cross-correlation."""
    a = a - a.mean(); b = b - b.mean()
    n = a.size
    xc = np.correlate(b, a, mode="full")
    lags = np.arange(-n + 1, n)
    k = int(np.argmax(xc))
    if 0 < k < xc.size - 1:
        ym, y0, yp = xc[k - 1], xc[k], xc[k + 1]
        denom = ym - 2 * y0 + yp
        delta = 0.5 * (ym - yp) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    return (lags[k] + delta) * dt


def tof_xcorr_speed(st: SpaceTime, r0: float, d_min: float = 2.0e-3,
                    d_max: float = 14.0e-3) -> SpeedResult:
    pos, neg, dist = _sides(st, r0, d_min, d_max)
    r = st.r
    dt = st.t[1] - st.t[0]
    out = {}
    for name, mask in (("pos", pos), ("neg", neg)):
        idx = np.where(mask)[0]
        if idx.size < 4:
            out[name] = (np.nan, 0.0, np.nan, None); continue
        order = idx[np.argsort(dist[idx])]     # ascending distance from origin
        ref = st.data[:, order[0]]
        tarr = np.full(r.size, np.nan)
        tarr[order[0]] = 0.0
        for j in order[1:]:
            tarr[j] = _xcorr_lag(ref, st.data[:, j], dt)
        c, q, t0 = _theil_speed(dist[mask], tarr[mask])
        out[name] = (c, q, t0, tarr)
    c_pos, q_pos, t0_pos, tp = out["pos"]
    c_neg, q_neg, t0_neg, tn = out["neg"]
    # predicted line uses fitted c and the origin-nearest column time
    t_pred_pos = (t0_pos + dist / c_pos) if np.isfinite(c_pos) else np.full(r.size, np.nan)
    t_pred_neg = (t0_neg + dist / c_neg) if np.isfinite(c_neg) else np.full(r.size, np.nan)
    t_pred_pos = np.where(pos, t_pred_pos, np.nan)
    t_pred_neg = np.where(neg, t_pred_neg, np.nan)
    return SpeedResult("tof_xcorr", c_pos, c_neg, q_pos, q_neg, r, t_pred_pos, t_pred_neg)


# --------------------------------------------------------------------------- #
# Lateral time-to-peak + robust (RANSAC-like) fit
# --------------------------------------------------------------------------- #
def ttp_ransac_speed(st: SpaceTime, r0: float, use_abs: bool = True,
                     d_min: float = 2.0e-3, d_max: float = 14.0e-3) -> SpeedResult:
    pos, neg, dist = _sides(st, r0, d_min, d_max)
    r = st.r
    t = st.t
    # demean each column in time so the static push/relaxation does not win the peak
    detr = st.data - st.data.mean(axis=0, keepdims=True)
    sig = np.abs(detr) if use_abs else detr
    tpeak = t[np.argmax(sig, axis=0)]          # (R,) peak time per position
    out = {}
    for name, mask in (("pos", pos), ("neg", neg)):
        if mask.sum() < 4:
            out[name] = (np.nan, 0.0, np.nan); continue
        out[name] = _theil_speed(dist[mask], tpeak[mask])
    c_pos, q_pos, t0_pos = out["pos"]
    c_neg, q_neg, t0_neg = out["neg"]
    t_pred_pos = np.where(pos, t0_pos + dist / c_pos, np.nan) if np.isfinite(c_pos) else np.full(r.size, np.nan)
    t_pred_neg = np.where(neg, t0_neg + dist / c_neg, np.nan) if np.isfinite(c_neg) else np.full(r.size, np.nan)
    return SpeedResult("ttp_ransac", c_pos, c_neg, q_pos, q_neg, r, t_pred_pos, t_pred_neg)


SPEED_METHODS = {
    "radon": slant_stack_speed,
    "tof_xcorr": tof_xcorr_speed,
    "ttp_ransac": ttp_ransac_speed,
}
