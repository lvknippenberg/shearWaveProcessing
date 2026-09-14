"""Buffers 1 / 3 / 4 beamformed with and without enable_pfield, with timing.

Timing is split the way it actually costs in a run:
  prepare  - pipeline build + prepare_parameters. With pfield this is where the transmit-field
             simulation happens; zea caches it to ~/.cache/zea/cached_funcs keyed on geometry,
             so it is paid once per buffer geometry (cold) and reloaded after (warm).
  frame    - steady-state per-frame beamforming, which is what scales with the buffer's
             frame count (90 / 26 / 926 for buffers 1 / 3 / 4).
"""
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO = r"D:\Luuk van Knippenberg\Github\shearWaveProcessing"
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import zea
from zea import File, init_device
from zea.ops import Beamform, Cast, Demodulate

from swp.acquisition.beamform import apply_grid, plan_patches_and_chunk, _ensure_cpu_t_peak
from swp.acquisition.sequence import read_swi_meta

FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54"
MAT = os.path.join(FOLDER, "CombinedData.mat")
CONV = os.path.join(FOLDER, "output", "converted")
OUT = os.path.dirname(os.path.abspath(__file__))
N_TIME = 3                       # frames used for the steady-state per-frame timing

init_device(verbose=True)
meta = read_swi_meta(MAT)

BUFFERS = [(1, "widebeam", 90), (3, "focused", 26), (4, "diverging", 926)]
results = {}
timing = []

for matlab, role, n_frames_total in BUFFERS:
    path = os.path.join(CONV, f"CombinedData_buffer{matlab}.hdf5")
    with File(path) as fh:
        raw = np.asarray(fh.data.raw_data[0:N_TIME])
        params = fh.load_parameters()
    grid = meta.grids[matlab - 1]
    apply_grid(params, grid)
    _ensure_cpu_t_peak(params)
    n_tx, n_el = raw.shape[1], raw.shape[3]
    num_patches, _ = plan_patches_and_chunk(params, n_tx, n_el)
    print(f"\nbuffer {matlab} ({role}): raw {raw.shape}, num_patches={num_patches}")

    for pf in (False, True):
        ops = [Cast(dtype="float32"), Demodulate(),
               Beamform(beamformer="delay_and_sum", num_patches=num_patches, enable_pfield=pf)]
        t0 = time.perf_counter()
        pipe = zea.Pipeline(ops, with_batch_dim=True, jit_options=None)
        bf_in = pipe.prepare_parameters(params)
        t_prep = time.perf_counter() - t0

        # warm-up (first call allocates / compiles kernels), then time per frame
        _ = pipe(data=raw[0:1], **bf_in)[pipe.output_key]
        per = []
        for k in range(N_TIME):
            t1 = time.perf_counter()
            iq = pipe(data=raw[k:k + 1], **bf_in)[pipe.output_key]
            iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)
            per.append(time.perf_counter() - t1)
            if k == 0:
                results[(matlab, pf)] = np.sqrt(iq[0, ..., 0] ** 2 + iq[0, ..., 1] ** 2)
        t_frame = float(np.median(per))
        timing.append((matlab, role, pf, t_prep, t_frame, n_frames_total))
        print(f"  pfield={str(pf):5s}  prepare {t_prep:7.2f}s   per-frame {t_frame:6.3f}s   "
              f"-> whole buffer ({n_frames_total} fr) {t_prep + t_frame * n_frames_total:7.1f}s")

print("\n" + "=" * 92)
print(f"{'buffer':>7s} {'role':10s} {'pfield':>7s} {'prepare':>9s} {'per-frame':>10s} "
      f"{'frames':>7s} {'buffer total':>13s}")
print("=" * 92)
tot = {False: 0.0, True: 0.0}
for matlab, role, pf, tp, tf, nf in timing:
    total = tp + tf * nf
    tot[pf] += total
    print(f"{matlab:7d} {role:10s} {str(pf):>7s} {tp:8.2f}s {tf:9.3f}s {nf:7d} {total:12.1f}s")
print("-" * 92)
print(f"{'':7s} {'ALL THREE':10s} {'OFF':>7s} {'':9s} {'':10s} {'':7s} {tot[False]:12.1f}s")
print(f"{'':7s} {'':10s} {'ON':>7s} {'':9s} {'':10s} {'':7s} {tot[True]:12.1f}s"
      f"   ({tot[True] / max(tot[False], 1e-9):.2f}x)")

# ---- montage ----
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for col, (matlab, role, nf) in enumerate(BUFFERS):
    for row, pf in enumerate((False, True)):
        e = results[(matlab, pf)]
        d = 20 * np.log10(e / e.max() + 1e-12)
        ins = e > 0
        frac = (d[ins] > -50).mean() * 100
        p90 = np.percentile(d[ins], 90)
        tp, tf = next((t[3], t[4]) for t in timing if t[0] == matlab and t[2] == pf)
        ax = axes[row, col]
        ax.imshow(d, cmap="gray", vmin=-50, vmax=0, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"buffer {matlab} ({role})  pfield={pf}\n"
                     f"{tf:.3f} s/frame (+{tp:.1f} s prepare)  |  "
                     f">-50 dB: {frac:.1f}%  p90 {p90:.1f} dB", fontsize=9)
plt.suptitle("enable_pfield off (top) vs on (bottom) - frame 0, -50..0 dB", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pfield_montage.png"), dpi=95)
print("\nwrote pfield_montage.png")
