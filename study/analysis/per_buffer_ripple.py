"""Does the line artefact exist in buffers 1 and 4, measured correctly?

The earlier "buffer 1 is clean" conclusion used the origin-referenced metric, which understates a
line-pitch artefact ~10x, so it has to be re-asked. Two things must be right per buffer:

* **the apex.** Buffer 3's virtual apex (-12.1 mm) is specific to its focused sector. Buffers 1 and
  4 use different transmit geometry, so the apex is SWEPT here and chosen by the data - the value
  that maximises the spectral peak - rather than assumed. For buffer 3 the sweep should land on
  -12 mm on its own, which is the check that the procedure works.
* **the pitch, in the same frame.** A delay-ramp steering angle is an origin-referenced plane-wave
  angle; the region geometry is apex-referenced. They differ by the same 1.13 factor that caused
  the original error, so the measured peak is compared against the pitch converted into whichever
  frame the measurement is made in.

Buffer 4 has 2 transmits. Two transmits are not a lattice, so a "pitch" there is meaningless and
any peak found is just the lowest frequency in the search band - reported, but not interpretable.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "src"))

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

from zea import File
from swp.acquisition.sequence import SPEC_BY_INDEX, read_swi_meta

FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54"
OUT = os.path.join(FOLDER, "output")
APEX_SWEEP_MM = np.arange(-30, 6, 1.0)


def load_env(name):
    with h5py.File(os.path.join(OUT, name), "r") as f:
        g = f["tracks/track_0/data/beamformed_data"]
        iq = np.asarray(g["values"][:])
        co = np.asarray(g["coordinates"])
    return np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2).mean(axis=0), co


def ripple_spec(env, co, apex_m, r_lo=0.045, r_hi=0.085, n_th=2001, th_max=30.0):
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
    return f, s, rip.std() * 100


def concentration(f, s, f_lo, f_hi, tol=0.12):
    """How concentrated the ripple is into its single strongest period, in [f_lo, f_hi].

    Assumption-free objective for the apex sweep: the right apex is the one that makes a periodic
    structure line up into ONE sharp peak. Using a target period instead would bake in the
    delay-ramp steering angle, which is a plane-wave approximation and ~5%% off the true focused
    sector geometry - that error picked the wrong apex on the first attempt.
    """
    sel = (f > f_lo) & (f < f_hi)
    if not sel.any():
        return 0.0, np.nan
    fpk = f[sel][np.argmax(s[sel])]
    band = (f > fpk * (1 - tol)) & (f < fpk * (1 + tol))
    tot = s[(f > 0.05) & (f < 5)].sum()
    return s[band].sum() / max(tot, 1e-30), 1.0 / fpk


def steer_pitch_deg(params):
    t0 = np.asarray(params.t0_delays, np.float64)
    ap = np.asarray(params.tx_apodizations, np.float64)
    xe = np.asarray(params.probe_geometry, np.float64)[:, 0]
    c = float(params.sound_speed)
    ang = []
    for k in range(t0.shape[0]):
        on = ap[k] > 0
        if on.sum() < 4:
            continue
        ang.append(np.degrees(np.arcsin(np.clip(-np.polyfit(xe[on], t0[k][on], 1)[0] * c, -1, 1))))
    ang = np.sort(np.asarray(ang))
    return (float(np.median(np.diff(ang))) if ang.size > 1 else np.nan), ang


meta = read_swi_meta(Path(FOLDER) / "CombinedData.mat")
print(f"{'buffer':24s} {'n_tx':>5s} {'pitch_origin':>13s} {'best apex':>10s} "
      f"{'peak':>9s} {'pitch_apex':>11s} {'amp@peak':>10s} {'rms':>7s}")
print("-" * 96)

for b, fn in ((1, "CombinedData_buffer1_iq.hdf5"),
              (3, "CombinedData_buffer3_iq.hdf5"),
              (4, "CombinedData_buffer4_iq.hdf5"),
              (1, "CombinedData_buffer1_refocus-adjoint_iq.hdf5"),
              (3, "CombinedData_buffer3_refocus-adjoint_iq.hdf5")):
    try:
        env, co = load_env(fn)
    except OSError:
        continue
    with File(os.path.join(OUT, "converted", f"CombinedData_buffer{b}.hdf5")) as fh:
        params = fh.load_parameters()
    pitch_o, ang = steer_pitch_deg(params)
    n_tx = len(ang)

    # search band: periods from half to three times the delay-ramp pitch estimate
    f_lo, f_hi = 1.0 / (3.0 * pitch_o), 1.0 / (0.5 * pitch_o)
    best = None
    for a_mm in APEX_SWEEP_MM:
        f, s, rms = ripple_spec(env, co, a_mm * 1e-3)
        conc, pk_deg = concentration(f, s, f_lo, f_hi)
        if best is None or conc > best[0]:
            best = (conc, a_mm, pk_deg, rms)
    p, a_mm, pk, rms = best
    mid = 0.065
    pitch_a = pitch_o * mid / (mid - a_mm * 1e-3)
    tag = "REFoCUS" if "refocus" in fn else SPEC_BY_INDEX[b - 1].name
    note = "  (2 tx: no lattice)" if n_tx < 3 else ""
    print(f"{f'b{b} {tag}' + note:24s} {n_tx:5d} {pitch_o:12.3f}d {a_mm:9.0f}mm "
          f"{pk:8.3f}d {pitch_a:10.3f}d {np.sqrt(p) * rms:9.2f}% {rms:6.1f}%")
