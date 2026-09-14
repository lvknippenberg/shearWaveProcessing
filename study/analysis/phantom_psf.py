"""Resolution-phantom analysis: point-target PSF and angular ripple, per reconstruction.

Answers the two questions the in-vivo data could not:

1. **Does REFoCUS preserve resolution?** Measured on wire targets as lateral/axial -6 dB width,
   versus depth. This replaces the anatomy-scale correlation proxy (~24 mm) used in vivo, which
   was never a resolution measure.
2. **Scalloping or interference?** A multiplicative transmit-field dip would SURVIVE envelope
   summing; an interference pattern would not. Measured as angular ripple at the 1.111 deg line
   spacing in a uniform speckle region - and a phantom has no anatomy or rib shadowing to
   confuse it.

Targets are found automatically as local maxima well above the surrounding speckle, then matched
across reconstructions by position so the same wire is compared each time.
"""
import os

import h5py
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter1d

F = r"D:\swp_res\Resolution phantom\DefaultPatient_SW_data_18-June-2026_13-52-51\output"
OUT = os.path.dirname(os.path.abspath(__file__))

RECONS = [
    ("CombinedData_buffer3_iq.hdf5", "standard (coherent all-73)"),
    ("CombinedData_buffer3_refocus-adjoint_iq.hdf5", "REFoCUS adjoint"),
    ("CombinedData_buffer3_refocus-tikhonov_iq.hdf5", "REFoCUS tikhonov"),
    ("CombinedData_buffer3_refocus-tsvd_iq.hdf5", "REFoCUS tsvd"),
    ("CombinedData_buffer3_incoh_iq.hdf5", "incoherent (envelope)"),
]
LINE_SPACING_DEG = 1.1111


def load_env(name):
    with h5py.File(os.path.join(F, name), "r") as f:
        g = f["tracks/track_0/data/beamformed_data"]
        iq = np.asarray(g["values"][:])
        co = np.asarray(g["coordinates"])
    env = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2).mean(axis=0)   # 2 static frames -> average
    return env, co


def axes_mm(co):
    z = co[:, 0, -1] * 1e3
    x = co[0, :, 0] * 1e3
    return x, z


def find_targets(env, x, z, z_range=(30, 105), prominence_db=12.0, min_sep_px=14):
    """Local maxima that stand well above their local background = wire targets."""
    db = 20 * np.log10(env / env.max() + 1e-12)
    local_max = maximum_filter(db, size=min_sep_px) == db
    bg = uniform_filter1d(uniform_filter1d(db, 61, axis=0), 61, axis=1)
    zz = z[:, None] * np.ones_like(db)
    ok = local_max & (db - bg > prominence_db) & (zz > z_range[0]) & (zz < z_range[1])
    iz, ix = np.nonzero(ok)
    return [(int(a), int(b)) for a, b in zip(iz, ix)]


def fwhm(profile, step_mm, peak_idx, drop_db=6.0):
    """-drop_db width of a peak, linearly interpolated, in mm. None if it does not close."""
    p = profile / (profile[peak_idx] + 1e-20)
    thr = 10 ** (-drop_db / 20.0)
    left = right = None
    for i in range(peak_idx, 0, -1):
        if p[i] < thr:
            left = i + (thr - p[i]) / (p[i + 1] - p[i] + 1e-20)
            break
    for i in range(peak_idx, len(p) - 1):
        if p[i] < thr:
            right = i - (thr - p[i]) / (p[i - 1] - p[i] + 1e-20)
            break
    if left is None or right is None:
        return None
    return (right - left) * step_mm


def measure(env, co, targets):
    x, z = axes_mm(co)
    dx, dz = abs(x[1] - x[0]), abs(z[1] - z[0])
    rows = []
    for iz, ix in targets:
        lat = fwhm(env[iz], dx, ix)
        ax = fwhm(env[:, ix], dz, iz)
        if lat is None or ax is None:
            continue
        # contrast of the target over nearby speckle
        z0, z1 = max(iz - 40, 0), min(iz + 40, env.shape[0])
        x0, x1 = max(ix - 40, 0), min(ix + 40, env.shape[1])
        patch = env[z0:z1, x0:x1]
        rows.append(dict(z=z[iz], x=x[ix], lat=lat, ax=ax,
                         cnr=20 * np.log10(env[iz, ix] / (np.median(patch) + 1e-20))))
    return rows


def angular_ripple(env, co, r_lo=0.045, r_hi=0.085, n_th=1501, th_max=34.0):
    """Fraction of angular-ripple power at the line spacing, in a speckle band."""
    x, z = co[..., 0], co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-th_max, th_max, n_th))
    r = np.linspace(r_lo, r_hi, 200)
    R, TH = np.meshgrid(r, th, indexing="ij")
    jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
    ii = np.clip(((R * np.cos(TH) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
    pol = env[ii, jj]
    pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
    prof = pol.mean(axis=0)
    dth = np.degrees(th[1] - th[0])
    trend = uniform_filter1d(prof, int(round(10.0 / dth)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    spec = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    freq = np.fft.rfftfreq(rip.size, d=dth)
    tot = spec[(freq > 0.05) & (freq < 5)].sum()
    fl = 1.0 / LINE_SPACING_DEG
    band = (freq > fl * 0.9) & (freq < fl * 1.1)
    # also report where the peak actually is (lesson: fit the peak, do not trust a window)
    sel = (freq > 0.3) & (freq < 2.0)
    peak_deg = 1.0 / freq[sel][np.argmax(spec[sel])]
    return spec[band].sum() / tot * 100, rip.std() * 100, peak_deg


# ---- reference targets from the standard reconstruction ----
env0, co0 = load_env(RECONS[0][0])
x0ax, z0ax = axes_mm(co0)
targets = find_targets(env0, x0ax, z0ax)
print(f"point targets detected (standard recon): {len(targets)}")
depths = sorted({round(z0ax[t[0]]) for t in targets})
print(f"  depths spanned: {min(depths):.0f} - {max(depths):.0f} mm")
print()

print(f"{'reconstruction':28s} {'n':>3s} {'lateral -6dB':>14s} {'axial -6dB':>12s} "
      f"{'target CNR':>11s}")
print("-" * 74)
results = {}
for name, lab in RECONS:
    try:
        env, co = load_env(name)
    except (OSError, KeyError):
        print(f"{lab:28s}  (not available)")
        continue
    rows = measure(env, co, targets)
    if not rows:
        print(f"{lab:28s}  (no measurable targets)")
        continue
    lat = np.array([r["lat"] for r in rows])
    ax = np.array([r["ax"] for r in rows])
    cnr = np.array([r["cnr"] for r in rows])
    results[lab] = rows
    print(f"{lab:28s} {len(rows):3d} {np.median(lat):11.2f} mm {np.median(ax):9.2f} mm "
          f"{np.median(cnr):8.1f} dB")

print()
print(f"{'reconstruction':28s} {'@1.111deg':>10s} {'ripple RMS':>11s} {'actual peak':>12s}")
print("-" * 66)
for name, lab in RECONS:
    try:
        env, co = load_env(name)
    except (OSError, KeyError):
        continue
    p, rms, pk = angular_ripple(env, co)
    print(f"{lab:28s} {p:9.2f}% {rms:10.1f}% {pk:9.3f} deg")

np.save(os.path.join(OUT, "phantom_psf_results.npy"), results, allow_pickle=True)
print("\nsaved phantom_psf_results.npy")
