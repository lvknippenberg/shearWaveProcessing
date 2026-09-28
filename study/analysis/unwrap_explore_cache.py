"""Feature cache for exploring buffer-3 head estimators (2026-09-28). Read-only on the data.

Per measurement folder (all of Z:/raw_data whose beamforming is complete) one npz with:
  A1, A3        anatomy vectors (swp.acquisition.unwrap: log-envelope band-passed 1-6 mm on the
                0.8 mm PLAX grid, standardised) of buffer 1 and of buffer 3 in ORIGINAL STORED
                SLOT ORDER (already-unwrapped files are rotated back with unwrap_first_frame)
  L1, L3        low-pass log-envelope (Gaussian 1.5 mm, no high-pass) on a 1.6 mm grid, same order
  t1            buffer-1 frame trigger times (ms, log clock)
  t3            buffer-3 chronological frame trigger times: the last n triggers of the live run
  r             logged R-peaks (ms), frame_ms3, frame_ms1
  tc_first      exact head from the trigger count (-1 when the run start is lost), run_length
  ecg_ok, unwrap_first (head applied to the files, -1 = none), unwrap_method

    python study/analysis/unwrap_explore_cache.py [--workers 4]
-> study/analysis/unwrap_cache/<subject>__<folder>.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
os.environ.setdefault("KERAS_BACKEND", "torch")
CACHE = _REPO / "study" / "analysis" / "unwrap_cache"
_BF = "tracks/track_0/data/beamformed_data"
LX = np.arange(-25, 25.01, 1.6)
LZ = np.arange(20, 100.01, 1.6)


def features(iq_path):
    import h5py
    import hdf5plugin  # noqa: F401
    from scipy.ndimage import gaussian_filter
    from swp.acquisition import unwrap as U
    from swp.mline.transfer import resample
    with h5py.File(iq_path, "r") as f:
        v = np.asarray(f[f"{_BF}/values"], np.float32)
        c = np.asarray(f[f"{_BF}/coordinates"])
    x, z = c[0, :, 0] * 1e3, c[:, 0, 2] * 1e3
    env = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    if x[0] > x[-1]:
        x, env = x[::-1], env[:, :, ::-1]
    if z[0] > z[-1]:
        z, env = z[::-1], env[:, ::-1]
    A = np.stack([U._anatomy(e, x, z) for e in env]).astype(np.float16)
    Ls = []
    for e in env:
        d = resample(e, x, z, LX, LZ)
        d = 20 * np.log10(d / np.nanmax(d) + 1e-6)
        d = np.where(np.isfinite(d), d, np.nanmedian(d))
        d = gaussian_filter(d, 1.5 / 1.6)
        Ls.append(((d - d.mean()) / (d.std() + 1e-12)).ravel())
    return A, np.stack(Ls).astype(np.float16)


def one(folder):
    from swp.acquisition import unwrap as U
    from swp.acquisition.triggerlog import buffer_timing
    folder = Path(folder)
    dst = CACHE / f"{folder.parent.name}__{folder.name}.npz"
    if dst.exists():
        return dst.name, "cached"
    out = folder / "output"
    _, iq3 = U.buffer_files(out)
    _, iq1 = U.buffer_files(out, buffer=1)
    if iq3 is None or iq1 is None:
        return dst.name, "no iq"
    b1, b3 = buffer_timing(str(folder), 1), buffer_timing(str(folder), 3)
    if b1 is None or b3 is None:
        return dst.name, "no timing"
    A1, L1 = features(iq1)
    A3, L3 = features(iq3)
    rec = U.read_flag(iq3)
    first = int(rec["first_frame"]) if rec and rec.get("buffer_unwrapped") else -1
    if first >= 0:                                   # rotate back to the ORIGINAL stored order
        n = len(A3)
        idx = [(s - first) % n for s in range(n)]
        A3, L3 = A3[idx], L3[idx]
    tc = U.trigger_count_head(str(folder))
    try:
        ecg = U._ecg_ok(str(folder))
    except Exception:                                                # noqa: BLE001
        ecg = False
    np.savez(dst, A1=A1, A3=A3, L1=L1, L3=L3, t1=b1.frame_times(), t3=b3.frame_times(),
             r=np.asarray(b3.r_peaks_ms), frame_ms1=b1.frame_ms, frame_ms3=b3.frame_ms,
             tc_first=tc["c"] if tc else -1, run_length=tc["run_length"] if tc else -1,
             ecg_ok=ecg, unwrap_first=first, unwrap_method=(rec or {}).get("method", ""),
             n1_ok=len(A1) == b1.n_frames, n3_ok=len(A3) == b3.n_frames)
    return dst.name, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from process_raw_data import find_measurement_folders
    CACHE.mkdir(parents=True, exist_ok=True)
    folders = [f for f in find_measurement_folders(a.root) if "_sw_data_" in f.name.lower()]
    print(f"{len(folders)} SW folders", flush=True)
    t0, counts = time.perf_counter(), {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, str(f)): f for f in folders}
        for k, fu in enumerate(as_completed(futs)):
            try:
                name, st = fu.result()
            except Exception:                                        # noqa: BLE001
                name, st = str(futs[fu]), "FAILED " + traceback.format_exc().splitlines()[-1]
            counts[st.split()[0]] = counts.get(st.split()[0], 0) + 1
            print(f"[{k + 1}/{len(folders)}] {st:8s} {name}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    print(counts)


if __name__ == "__main__":
    main()
