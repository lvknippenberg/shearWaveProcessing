"""Does reconstructing each pixel from fewer transmits remove the striations?

S5d showed the artefact is cross-transmit phase interference, not amplitude scalloping. The
obvious follow-up: if interference BETWEEN transmits is the problem, use fewer of them. The
extreme is nearest-1 - each pixel from the single transmit whose beam axis is closest - which is
classical line-by-line focused imaging and has no cross-transmit interference at all by
construction.

This beamforms all 73 transmits separately on the phantom and then composites them nearest-k for
k = 1, 2, 3, 6, 12, 73. Note k=6 is what the pipeline already does: each region spans 6.667 deg on
a 1.111 deg lattice, so exactly 6 transmits cover any pixel. Reproducing the standard
reconstruction's ripple at k=6 is the validation that the composite rule is right.

Measured on the phantom, so lateral PSF on 79 wire targets comes along for free - the question is
not only whether the striations go but what the resolution does, and nearest-1 is the case where
resolution should be BEST near the focus (a single focused beam) and worst away from it.
"""
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "src"))

import numpy as np
from pathlib import Path

import zea
from zea import File, init_device
from zea.ops import Beamform, Cast, Demodulate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from swp.acquisition.beamform import _ensure_cpu_t_peak, apply_grid, plan_patches_and_chunk
from swp.acquisition.sequence import read_swi_meta

import phantom_psf as P   # its module-level analysis prints the reference table; harmless here

ROOT = r"D:\swp_res\Resolution phantom\DefaultPatient_SW_data_18-June-2026_13-52-51"
APEX_M = -0.0121                 # virtual apex, 12.1 mm behind the array (from CenterTransmit.mat)
STEER_DEG = np.arange(73) * 1.1111111 - 40.0
KS = (1, 2, 3, 6, 12, 73)
STACK = os.path.join(HERE, "phantom_per_tx.npy")


def per_transmit_stack():
    """(n_tx, nz, nx) complex64, frame-averaged coherently (the phantom is static)."""
    if os.path.isfile(STACK):
        print(f"loading cached {os.path.basename(STACK)}")
        return np.load(STACK)
    init_device(verbose=False)
    meta = read_swi_meta(Path(ROOT) / "CombinedData.mat")
    with File(os.path.join(ROOT, "output", "converted", "CombinedData_buffer3.hdf5")) as fh:
        raw = np.asarray(fh.data.raw_data[:])
        params = fh.load_parameters()
    apply_grid(params, meta.grids[2])
    _ensure_cpu_t_peak(params)
    n_frames, n_tx = raw.shape[0], raw.shape[1]
    num_patches, _ = plan_patches_and_chunk(params, 1, raw.shape[3])
    pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                         Beamform(beamformer="delay_and_sum", num_patches=num_patches,
                                  enable_pfield=False)],
                        with_batch_dim=True, jit_options=None)
    out = None
    t0 = time.perf_counter()
    for i in range(n_tx):
        params.set_transmits([i])
        bf_in = pipe.prepare_parameters(params)
        iq = pipe(data=raw[:, i:i + 1], **bf_in)[pipe.output_key]
        iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)
        c = (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64).mean(axis=0)
        if out is None:
            out = np.zeros((n_tx,) + c.shape, np.complex64)
        out[i] = c
        if i % 20 == 0:
            print(f"    transmit {i}/{n_tx} ({time.perf_counter() - t0:.0f}s)")
    print(f"  per-transmit beamform done in {time.perf_counter() - t0:.0f}s -> {out.shape}")
    np.save(STACK, out)
    return out


stack = per_transmit_stack()
env_std, co = P.load_env(P.RECONS[0][0])
targets = P.find_targets(env_std, *P.axes_mm(co)[::-1][::-1])

# angle of every pixel about the virtual apex, and its distance to each transmit axis
x, z = co[..., 0], co[..., -1]
pix_deg = np.degrees(np.arctan2(x, z - APEX_M))
order = np.argsort(np.abs(pix_deg[None, :, :] - STEER_DEG[:, None, None]), axis=0)

print(f"\nstack {stack.shape}, grid {pix_deg.shape}, "
      f"pixel angles {pix_deg.min():.1f} to {pix_deg.max():.1f} deg")
print(f"\n{'composite':28s} {'ripple':>8s} {'peak':>9s} {'lateral':>9s} {'axial':>8s} {'CNR':>7s}")
print("-" * 74)


def report(env, lab):
    rows = P.measure(env, co, targets)
    frac, rms, pk = P.angular_ripple(env, co)
    amp = np.sqrt(frac / 100.0) * rms
    if rows:
        lat = np.median([r["lat"] for r in rows])
        ax = np.median([r["ax"] for r in rows])
        cnr = np.median([r["cnr"] for r in rows])
        print(f"{lab:28s} {amp:7.2f}% {pk:8.3f}d {lat:8.2f}mm {ax:7.2f}mm {cnr:6.1f}dB")
    else:
        print(f"{lab:28s} {amp:7.2f}% {pk:8.3f}d   (no measurable targets)")


nz, nx = pix_deg.shape
ii, jj = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
acc = np.zeros((nz, nx), np.complex64)
for k in range(1, max(KS) + 1):
    acc += stack[order[k - 1], ii, jj]        # add the k-th nearest transmit for every pixel
    if k in KS:
        report(np.abs(acc), f"nearest-{k}" + ("  (= region rule)" if k == 6 else ""))

report(np.abs(np.sum(stack, axis=0)), "all-73, no region mask")
report(np.sum(np.abs(stack), axis=0), "all-73 incoherent")
report(env_std, "stored standard (reference)")
