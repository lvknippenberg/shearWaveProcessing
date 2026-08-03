"""Load the in-vivo ECG.mat (v7.3) timing file.

The file stores one vector ``ECG_data_values`` containing four NaN-separated segments, in
order: (1) ECG time axis [us], (2) rectified ECG signal (above baseline only), (3) ECG
R-peak times [us], (4) ultrasound event times [us].  All times share a common microsecond
clock.  The ultrasound events mix several acquisition types (focused B-mode, weakly-focused
B-mode, ultrafast B-mode, and the shear-wave sub-events: reference-frame / push / tracking /
paired B-mode starts); the precise per-event typing is not encoded here (see repo notes).
"""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class ECG:
    t: np.ndarray            # ECG time axis [s]
    signal: np.ndarray       # rectified ECG amplitude (above baseline)
    r_peaks: np.ndarray      # sorted R-peak times [s]
    us_events: np.ndarray    # sorted ultrasound event times [s]

    @property
    def rr(self) -> np.ndarray:
        return np.diff(self.r_peaks)

    @property
    def hr_bpm(self) -> float:
        return 60.0 / np.median(self.rr)

    def cardiac_phase(self, event_t):
        """Fractional cardiac phase in [0,1) of event time(s) [s]: (t - last R) / RR."""
        event_t = np.atleast_1d(np.asarray(event_t, float))
        out = np.full(event_t.shape, np.nan)
        for i, te in enumerate(event_t):
            prev = self.r_peaks[self.r_peaks <= te]
            nxt = self.r_peaks[self.r_peaks > te]
            if prev.size and nxt.size:
                out[i] = (te - prev[-1]) / (nxt[0] - prev[-1])
        return out

    def detect_sw_pushes(self, gap_lo: float = 600e-6, gap_hi: float = 800e-6) -> np.ndarray:
        """Shear-wave push-transmit times [s], in measurement order.

        The ultrasound event stream follows a fixed pattern in which each SW measurement
        (which follows the ultrafast-B-mode block) has its push transmit immediately followed
        by a ~700 us gap marking the start of the tracking frames.  Push events are therefore
        those whose forward gap falls in ``[gap_lo, gap_hi]``.  For the example data this
        returns exactly 24 pushes (meas0..23) at ~50 ms spacing.
        """
        gaps = np.diff(self.us_events)
        idx = np.where((gaps >= gap_lo) & (gaps <= gap_hi))[0]
        return self.us_events[idx]

    def push_cardiac_phases(self, **kw) -> np.ndarray:
        """Cardiac phase (0-1) of each detected SW push, i.e. per measurement in order."""
        return self.cardiac_phase(self.detect_sw_pushes(**kw))


def load_ecg(path: str, clock_unit_s: float = 1e-6) -> ECG:
    """Load ECG.mat; ``clock_unit_s`` converts the raw clock to seconds (default: microseconds)."""
    with h5py.File(path, "r") as f:
        a = np.array(f["ECG_data_values"]).squeeze().astype(float)
    nan_idx = np.where(np.isnan(a))[0]
    segs = [a[nan_idx[i] + 1:nan_idx[i + 1]] for i in range(len(nan_idx) - 1)]
    t_axis, ecg_sig, r_peaks, us = segs[0], segs[1], segs[2], segs[3]
    # R-peaks occasionally carry a leading out-of-order marker; sort defensively.
    r_peaks = np.unique(r_peaks)
    us = np.unique(us)
    t0 = t_axis.min()
    return ECG(
        t=(t_axis - t0) * clock_unit_s,
        signal=ecg_sig,
        r_peaks=(r_peaks - t0) * clock_unit_s,
        us_events=(us - t0) * clock_unit_s,
    )
