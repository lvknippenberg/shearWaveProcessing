"""Cardiac timing of the in-vivo ARF pushes, from the ECG/trigger record.

The runtime `AcquisitionParametersAndECG.mat` stores `ECG_data_raw`: a text table with four named
sections concatenated one after the other --

    Time_us          ECG sample times      (1000 samples, 10 ms apart)
    Signal_V         ECG amplitude         (1000 samples)
    ECG_trigger_us   detected R-peaks      (the whole session, not just this acquisition)
    Vera_trigger_us  every Verasonics acquisition trigger

The shear-wave block is the last run of Verasonics triggers: 24 measurements, 4 triggers each,
repeating on an exact 50 ms grid (20 Hz). Only the *relative* push times matter for pairing pushes
across beats, and those are exact; the absolute offset between a trigger and the ARF pulse inside
its frame is a constant and cancels.

`push_phases()` returns, per push, the time since the preceding R-peak and the RR interval it sits
in -- which is what lets pushes from different beats be matched by cardiac phase.
"""
from __future__ import annotations

import os

import numpy as np
import scipy.io as sio

PUSH_PERIOD_MS = 50.0


def _raw_text(mat_path):
    """`ECG_data_raw` as text, from either a v7 runtime .mat or a v7.3 workspace/CombinedData."""
    try:
        return str(sio.loadmat(mat_path, squeeze_me=True,
                               struct_as_record=False)["ECG_data_raw"])
    except NotImplementedError:                     # v7.3 -> HDF5, char array stored as uint16
        import h5py
        import numpy as _np
        with h5py.File(mat_path, "r") as f:
            return "".join(chr(c) for c in _np.array(f["ECG_data_raw"]).squeeze().ravel())


def ecg_path(folder):
    """The runtime .mat if present, else the merged CombinedData.mat (which carries the same
    record for acquisitions whose runtime file was lost)."""
    runtime = os.path.join(folder, "AcquisitionParametersAndECG.mat")
    return runtime if os.path.isfile(runtime) else os.path.join(folder, "CombinedData.mat")


def read_ecg(mat_path):
    """Parse `ECG_data_raw` into {section: array}. Times are converted to milliseconds."""
    raw = _raw_text(mat_path)
    secs, cur = {}, None
    for line in str(raw).split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            secs[cur].append(float(line))
        except (ValueError, KeyError):
            cur = line
            secs[cur] = []
    out = {k: np.asarray(v, dtype=float) for k, v in secs.items()}
    for k in ("Time_us", "ECG_trigger_us", "Vera_trigger_us"):
        if k in out:
            out[k.replace("_us", "_ms")] = out.pop(k) / 1e3
    return out


def push_times(ecg, n_push):
    """Push trigger times [ms]. The shear-wave block is the trailing 4*n_push triggers."""
    vt = ecg["Vera_trigger_ms"]
    blk = vt[-4 * n_push:]
    t = blk[1::4]                                  # one trigger per measurement, exact 50 ms grid
    if len(t) != n_push:                           # fall back to a synthetic grid
        t = blk[0] + PUSH_PERIOD_MS * np.arange(n_push)
    return t


def push_phases(mat_path, n_push):
    """Per push: (t_ms, beat index, phase since R-peak [ms], RR of that beat [ms])."""
    ecg = read_ecg(mat_path)
    t = push_times(ecg, n_push)
    rp = ecg["ECG_trigger_ms"]
    prev = np.array([rp[rp <= x].max() if (rp <= x).any() else np.nan for x in t])
    nxt = np.array([rp[rp > x].min() if (rp > x).any() else np.nan for x in t])
    beat = np.array([int((rp <= x).sum()) for x in t])
    return dict(t=t, beat=beat - beat.min(), phase=t - prev, rr=nxt - prev,
                frac=(t - prev) / (nxt - prev), ecg=ecg)


def cross_beat_pairs(ph, max_phase_diff_ms=40.0):
    """Pairs (i, j) of pushes in DIFFERENT beats whose cardiac phase agrees within a tolerance."""
    pairs = []
    n = len(ph["t"])
    for i in range(n):
        for j in range(i + 1, n):
            if ph["beat"][i] == ph["beat"][j]:
                continue
            d = abs(ph["phase"][i] - ph["phase"][j])
            if d <= max_phase_diff_ms:
                pairs.append((i, j, d))
    pairs.sort(key=lambda p: p[2])
    used_i, used_j, keep = set(), set(), []
    for i, j, d in pairs:                          # greedy: each push used at most once
        if i in used_i or j in used_j or i in used_j or j in used_i:
            continue
        used_i.add(i)
        used_j.add(j)
        keep.append((i, j, d))
    return sorted(keep)


def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--n-push", type=int, default=24)
    ap.add_argument("--tol", type=float, default=40.0)
    a = ap.parse_args()
    ph = push_phases(ecg_path(a.folder), a.n_push)
    rr = ph["rr"]
    print(f"HR over the push window: {60000 / np.nanmedian(rr):.1f} bpm "
          f"(RR {np.nanmin(rr):.0f}-{np.nanmax(rr):.0f} ms)")
    print(f"{'push':>4} {'t_rel':>8} {'beat':>5} {'phase':>8} {'RR':>7} {'phase %':>8}")
    for i in range(a.n_push):
        print(f"{i:4d} {ph['t'][i]-ph['t'][0]:8.1f} {ph['beat'][i]:5d} {ph['phase'][i]:8.1f} "
              f"{ph['rr'][i]:7.1f} {100*ph['frac'][i]:8.1f}")
    pairs = cross_beat_pairs(ph, a.tol)
    print(f"\n{len(pairs)} cross-beat pairs within {a.tol:.0f} ms of phase:")
    for i, j, d in pairs:
        print(f"  push {i:2d} (beat {ph['beat'][i]}, {ph['phase'][i]:6.1f} ms)  <->  "
              f"push {j:2d} (beat {ph['beat'][j]}, {ph['phase'][j]:6.1f} ms)   dphase {d:5.1f} ms")


if __name__ == "__main__":
    _cli()
