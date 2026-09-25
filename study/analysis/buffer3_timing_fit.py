"""Fit the buffer-3 frame timing against buffer 1: slot rotation r and time offset delta.

Buffer 1 (widebeam) is acquired once, as a block, and its frame times from the trigger log are
trusted. Buffer 3 (focused live loop) has two suspected timing errors (see
buffer3_frame_order.py and the sequence, SetUp_SWI_Widebeam.m):

* rolling shutter: the trigger-out fires on the FIRST of 73 lines (x2 pulse inversion x 270 us =
  39.4 ms per frame), so the middle of the frame is ~19.7 ms after the logged time; buffer 1 (21
  transmits, 11.3 ms per frame) is likewise ~5.7 ms late, so relative to buffer 1 the expected
  offset is ~ +14 ms;
* circular-buffer rotation: if the live loop stopped after slot s < N, slot q was acquired at the
  chronological trigger index (q - s - 1) mod N, not q - 1.

For each (r = s, delta), every buffer-3 slot is assigned a time, hence a cardiac phase (ms since the
preceding logged R-peak), and paired with the buffer-1 frame of the nearest phase. The score is the
mean anatomy correlation of those pairs. The best (r, delta) is reported next to the current
assumption (r = N, delta = 0). Anatomy images: swp.mline.transfer.anatomy on a common 0.8 mm grid,
x in +-25 mm, z 20-100 mm.

    python study/analysis/buffer3_timing_fit.py [--limit N]
-> study/logs/buffer3_timing_fit.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from swp.manual import store as S                     # noqa: E402
from swp.manual._light import transfer, triggerlog    # noqa: E402

import h5py                                           # noqa: E402
import hdf5plugin                                     # noqa: E402,F401

_BF = "tracks/track_0/data/beamformed_data"
GX = np.arange(-25, 25.01, 0.8)
GZ = np.arange(20, 100.01, 0.8)
DELTAS = np.arange(-60, 80.1, 2.0)


def anatomy_stack(path, frames=None, n_avg=1):
    """Anatomy images of every frame, or of ``frames`` (each the mean envelope of n_avg frames)."""
    T = transfer()
    with h5py.File(path, "r") as f:
        d = f[f"{_BF}/values"]
        if frames is None:
            v = np.asarray(d, np.float32)
        else:
            v = np.stack([np.sqrt((np.asarray(d[k:k + n_avg], np.float32) ** 2).sum(-1)).mean(0)
                          for k in frames])[..., None]
            v = np.concatenate([v, np.zeros_like(v)], -1)        # envelope as I, Q = 0
        c = np.asarray(f[f"{_BF}/coordinates"])
    x, z = c[0, :, 0] * 1e3, c[:, 0, 2] * 1e3
    env = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    if x[0] > x[-1]:
        x, env = x[::-1], env[:, :, ::-1]
    if z[0] > z[-1]:
        z, env = z[::-1], env[:, ::-1]
    out = []
    for e in env:
        a = T.anatomy(T.resample(e, x, z, GX, GZ), 0.8)
        a = (a - a.mean()) / (np.linalg.norm(a - a.mean()) + 1e-12)
        out.append(a.ravel())
    return np.array(out)


def phase_of(t_ms, r_peaks):
    out = np.full(np.shape(t_ms), np.nan)
    for i, t in enumerate(np.atleast_1d(t_ms)):
        prev = r_peaks[r_peaks <= t]
        if prev.size:
            out[i] = t - prev.max()
    return out


CONTROL_STRIDE = 37      # buffer-4 frames (1.08 ms) per control frame: ~40 ms, as buffer 3


def fit(folder, control=False):
    """``control=True``: the same fit on buffer 4 subsampled to buffer-3 spacing - frames whose
    order and timing are known to be right - to measure how much 'rotation gain' the search finds
    in correctly ordered data."""
    tl = triggerlog()
    p = S.Paths(folder)
    b1, b3 = tl.buffer_timing(folder, 1), tl.buffer_timing(folder, 4 if control else 3)
    if b1 is None or b3 is None:
        return None
    A1 = anatomy_stack(p.bmode(1))
    if control:
        idx = np.arange(0, b3.n_frames - 9, CONTROL_STRIDE)
        A3 = anatomy_stack(p.bmode(4), frames=idx, n_avg=9)
        t_all = b3.frame_times()
        b3 = tl.BufferTiming(4, len(idx), b3.frame_ms * CONTROL_STRIDE, float(t_all[0]), b3.r_peaks_ms, "control")
    else:
        A3 = anatomy_stack(p.bmode(3))
    n1, n3 = len(A1), len(A3)
    if n1 != b1.n_frames or n3 != b3.n_frames:
        return None
    M = A3 @ A1.T                                          # (n3, n1) correlation
    ph1 = b1.phase_ms()
    ok1 = np.isfinite(ph1)
    T3 = b3.frame_times()
    rp = np.asarray(b3.r_peaks_ms)

    def score(s, delta):
        chrono = (np.arange(1, n3 + 1) - s - 1) % n3
        ph3 = phase_of(T3[chrono] + delta, rp)
        vals = []
        for q in range(n3):
            if not np.isfinite(ph3[q]):
                continue
            d = np.where(ok1, np.abs(ph1 - ph3[q]), np.inf)
            f = int(np.argmin(d))
            if d[f] < 15:
                vals.append(M[q, f])
        return float(np.mean(vals)) if len(vals) >= n3 // 2 else np.nan

    grid = np.array([[score(s, d) for d in DELTAS] for s in range(1, n3 + 1)])
    s_best, d_best = np.unravel_index(np.nanargmax(grid), grid.shape)
    no_rot = grid[n3 - 1]
    # the frame the M-line tool shows at the R-peak: its assumed vs best-fit phase
    k, off = b3.nearest_rpeak_frame()
    chrono = (np.arange(1, n3 + 1) - (s_best + 1) - 1) % n3
    true_ph = float(phase_of(np.atleast_1d(T3[chrono[k]] + DELTAS[d_best]), rp)[0])
    # assumption-free: phase of the buffer-1 frame that looks most like it
    f_best = int(np.argmax(np.where(ok1, M[k], -np.inf)))
    return dict(n_frames=n3, rr_ms=float(np.median(np.diff(rp))) if rp.size > 1 else np.nan,
                span_ms=n3 * b3.frame_ms,
                best_stop_slot=int(s_best + 1), best_delta_ms=float(DELTAS[d_best]),
                best_score=round(float(grid[s_best, d_best]), 3),
                norot_best_delta_ms=float(DELTAS[np.nanargmax(no_rot)]),
                norot_best_score=round(float(np.nanmax(no_rot)), 3),
                current_score=round(float(no_rot[np.argmin(np.abs(DELTAS))]), 3),
                ceiling=round(float(M.max(axis=1).mean()), 3),
                stop_is_return_point=((s_best + 1 - 1) % 5 == 0),
                b3_source=b3.source, rpeak_frame=k, rpeak_frame_assumed_ms=round(off, 1),
                rpeak_frame_fit_phase_ms=round(true_ph, 1),
                rpeak_frame_b1match_phase_ms=round(float(ph1[f_best]), 1),
                rpeak_frame_b1match_corr=round(float(M[k, f_best]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="run the fit on buffer 4 subsampled to buffer-3 spacing (known-correct order)")
    ap.add_argument("--only-n", type=int, default=0, help="only folders whose buffer 3 has this many frames")
    ap.add_argument("--every", type=int, default=1, help="take every k-th folder")
    a = ap.parse_args()
    folders = [f for f in S.find_folders(a.root) if S.ready(S.Paths(f))]
    if a.only_n:
        from swp.manual.frames import n_frames
        folders = [f for f in folders if n_frames(S.Paths(f).bmode(3)) == a.only_n]
    folders = folders[::a.every]
    if a.limit:
        folders = folders[:a.limit]
    rows = []
    for k, f in enumerate(folders):
        try:
            r = fit(f, control=a.control)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[{k + 1}] {f}: {exc}")
            continue
        if r is None:
            continue
        r = dict(folder=f"{Path(f).parent.name}/{Path(f).name}", **r)
        rows.append(r)
        print(f"[{k + 1}/{len(folders)}] {r['folder'][:38]:38s} N={r['n_frames']} RR {r['rr_ms']:.0f} "
              f"best stop {r['best_stop_slot']:2d} ({'ret' if r['stop_is_return_point'] else '---'}) "
              f"delta {r['best_delta_ms']:+.0f} score {r['best_score']:.3f} | no-rot delta "
              f"{r['norot_best_delta_ms']:+.0f} {r['norot_best_score']:.3f} | current {r['current_score']:.3f}"
              f" | ceiling {r['ceiling']:.3f} | R-frame fit R+{r['rpeak_frame_fit_phase_ms']:.0f}"
              f" b1-match R+{r['rpeak_frame_b1match_phase_ms']:.0f}", flush=True)
    out = _REPO / "study" / "logs" / ("buffer3_timing_fit_control.csv" if a.control else "buffer3_timing_fit.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
