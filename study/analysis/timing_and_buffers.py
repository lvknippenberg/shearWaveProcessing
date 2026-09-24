"""Beamforming time, standard vs REFoCUS, and whether buffers 1/4 carry the same artefact.

Timing is the GPU beamform only - reading the converted RF off the network share dominates the
wall clock of a real run and is identical for both methods, so including it would hide the
difference. Each method is run twice and the second run reported, so CUDA context setup and
kernel autotuning do not land on one method and not the other.

The buffer question has to be re-asked with the CORRECTED apex-referenced metric: the earlier
"buffer 1 is clean" conclusion was reached with the origin-referenced version, which understates
a line-pitch artefact ~10x and can smear it away entirely. Each buffer is measured at ITS OWN
transmit pitch, derived from the delays rather than assumed.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "src"))

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

import zea
from zea import File, init_device
from zea.ops import Beamform, Cast, Demodulate, Refocus

from swp.acquisition.beamform import (_ensure_cpu_t_peak, apply_grid, beamform_frames,
                                      plan_patches_and_chunk)
from swp.acquisition.sequence import SPEC_BY_INDEX, read_swi_meta

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54"
OUT = os.path.join(FOLDER, "output")
APEX_M = -0.0121          # buffer-3 virtual apex, from CenterTransmit.mat


def steer_angles_deg(params):
    """Steering angle per transmit, from the delay ramp across the aperture."""
    t0 = np.asarray(params.t0_delays, np.float64)
    ap = np.asarray(params.tx_apodizations, np.float64)
    xe = np.asarray(params.probe_geometry, np.float64)[:, 0]
    c = float(params.sound_speed)
    out = []
    for k in range(t0.shape[0]):
        on = ap[k] > 0
        if on.sum() < 4:
            out.append(np.nan)
            continue
        slope = np.polyfit(xe[on], t0[k][on], 1)[0]       # s per m
        out.append(np.degrees(np.arcsin(np.clip(-slope * c, -1, 1))))
    return np.asarray(out)


def ripple_at(env, co, pitch_deg, apex_m=APEX_M, r_lo=0.045, r_hi=0.085, n_th=2001, th_max=30.0):
    x, z = co[..., 0], co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-th_max, th_max, n_th))
    r = np.linspace(r_lo - apex_m, r_hi - apex_m, 200)
    R, TH = np.meshgrid(r, th, indexing="ij")
    jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
    ii = np.clip((((R * np.cos(TH) + apex_m) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
    pol = env[ii, jj].astype(np.float64)
    pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
    prof = pol.mean(axis=0)
    dth = np.degrees(th[1] - th[0])
    trend = uniform_filter1d(prof, int(round(10.0 / dth)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    s = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    f = np.fft.rfftfreq(rip.size, d=dth)
    tot = s[(f > 0.05) & (f < 5)].sum()
    fl = 1.0 / pitch_deg
    band = (f > fl * 0.88) & (f < fl * 1.12)
    sel = (f > 1.0 / (4 * pitch_deg)) & (f < 2.5)
    return np.sqrt(s[band].sum() / tot) * rip.std() * 100, 1.0 / f[sel][np.argmax(s[sel])], \
        rip.std() * 100


def load_env(name):
    with h5py.File(os.path.join(OUT, name), "r") as f:
        g = f["tracks/track_0/data/beamformed_data"]
        iq = np.asarray(g["values"][:])
        co = np.asarray(g["coordinates"])
    return np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2).mean(axis=0), co


init_device(verbose=False)
meta = read_swi_meta(Path(FOLDER) / "CombinedData.mat")

# ---------------------------------------------------------------- transmit geometry per buffer
print(f"{'buffer':22s} {'n_tx':>5s} {'span (deg)':>12s} {'pitch (deg)':>12s}")
print("-" * 56)
pitches = {}
for b in (1, 3, 4):
    conv = os.path.join(OUT, "converted", f"CombinedData_buffer{b}.hdf5")
    with File(conv) as fh:
        params = fh.load_parameters()
    ang = steer_angles_deg(params)
    good = ang[np.isfinite(ang)]
    d = np.diff(np.sort(good))
    pitch = float(np.median(d)) if d.size else np.nan
    pitches[b] = pitch
    print(f"{SPEC_BY_INDEX[b - 1].name:22s} {len(ang):5d} "
          f"{good.min():5.1f} to {good.max():5.1f} {pitch:12.4f}")

# ---------------------------------------------------------------- ripple per buffer, corrected
print(f"\n{'buffer (apex-referenced, own pitch)':38s} {'@pitch':>9s} {'peak':>9s} {'rms':>8s}")
print("-" * 68)
for b, fn in ((1, "CombinedData_buffer1_iq.hdf5"),
              (3, "CombinedData_buffer3_iq.hdf5"),
              (4, "CombinedData_buffer4_iq.hdf5")):
    env, co = load_env(fn)
    amp, pk, rms = ripple_at(env, co, pitches[b])
    print(f"{SPEC_BY_INDEX[b - 1].name + f'  (pitch {pitches[b]:.3f} deg)':38s} "
          f"{amp:8.2f}% {pk:8.3f}d {rms:7.1f}%")
for b, fn in ((1, "CombinedData_buffer1_refocus-adjoint_iq.hdf5"),
              (3, "CombinedData_buffer3_refocus-adjoint_iq.hdf5")):
    try:
        env, co = load_env(fn)
    except OSError:
        continue
    amp, pk, rms = ripple_at(env, co, pitches[b])
    print(f"{f'  buffer {b} REFoCUS adjoint':38s} {amp:8.2f}% {pk:8.3f}d {rms:7.1f}%")

# ---------------------------------------------------------------- timing
print("\n--- beamforming time, GPU only (2nd run reported) ---")
for b in (3, 1):
    conv = os.path.join(OUT, "converted", f"CombinedData_buffer{b}.hdf5")
    with File(conv) as fh:
        raw = np.asarray(fh.data.raw_data[:])
        params = fh.load_parameters()
    apply_grid(params, meta.grids[b - 1])
    spec = SPEC_BY_INDEX[b - 1]
    n_f, n_tx = raw.shape[0], raw.shape[1]
    print(f"\nbuffer {b} ({spec.name}): {n_f} frames x {n_tx} transmits, "
          f"grid {tuple(np.asarray(params.grid).shape[:2])}, pfield={spec.pfield}")

    for _ in range(2):
        t0 = time.perf_counter()
        beamform_frames(raw, params, enable_pfield=spec.pfield)
        t_std = time.perf_counter() - t0
    print(f"  standard (as pipeline, pfield={spec.pfield}) {t_std:7.1f} s   "
          f"{t_std / n_f * 1000:6.0f} ms/frame")

    _ensure_cpu_t_peak(params)
    num_patches, chunk = plan_patches_and_chunk(params, n_tx, raw.shape[3])
    for _ in range(2):
        pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(), Refocus(method="adjoint"),
                             Beamform(beamformer="delay_and_sum", num_patches=num_patches,
                                      enable_pfield=False)],
                            with_batch_dim=True, jit_options=None)
        bf_in = pipe.prepare_parameters(params)
        t0 = time.perf_counter()
        for s in range(0, n_f, max(chunk, 1)):
            pipe(data=raw[s:s + max(chunk, 1)], **bf_in)
        t_ref = time.perf_counter() - t0
    print(f"  REFoCUS adjoint                            {t_ref:7.1f} s   "
          f"{t_ref / n_f * 1000:6.0f} ms/frame   ({t_ref / t_std:.2f}x standard)")
