"""Score all 8 buffer-3 reconstructions on one consistent metric set, phantom and in vivo.

Earlier tables in this study mixed two different "ripple" numbers - a FRACTION of ripple power in
the line-spacing band, and an ABSOLUTE ripple amplitude - which are not comparable and have
already misled once. Everything here is absolute.

It also adds the metric the others were missing. ``rip_line`` only looks at the 1.111 deg band, so
a reconstruction can score well on it while showing obvious BROAD radial streaks - which is
exactly what REFoCUS tsvd looks like by eye. The discriminator for "radial lines" is not amplitude
but DEPTH PERSISTENCE: speckle decorrelates as you move in range, a transmit-field streak does
not. So the polar band is split into a near and a far half and their angular ripple profiles are
correlated. Near 0 = speckle. Near 1 = a genuine radial line burned through the whole image.

Columns
    rip_line   absolute ripple amplitude at the 1.111 deg line spacing, %
    ang_rms    total angular ripple RMS of the depth-averaged profile, % (speckle dominates it)
    persist    near/far angular-profile correlation - the radial-streak metric
    lat/CNR    point-target PSF and contrast (phantom only; there are no point targets in vivo)
    spk_snr    envelope mean/std - falls when a multiplicative texture is added
    dyn_rng    99.9th pct over 5th pct, dB
"""
import os
import sys

import h5py
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter1d

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)

LINE_SPACING_DEG = 1.1111

DATASETS = [
    ("phantom", r"D:\swp_res\Resolution phantom\DefaultPatient_SW_data_18-June-2026_13-52-51\output",
     (0.030, 0.105), True),
    ("in vivo C1", r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54\output",
     (0.045, 0.085), False),
]
METHODS = [
    ("CombinedData_buffer3_iq.hdf5", "standard (coherent all-73)"),
    ("CombinedData_buffer3_incoh_iq.hdf5", "incoherent (envelope)"),
    ("CombinedData_buffer3_refocus-adjoint_iq.hdf5", "REFoCUS adjoint"),
    ("CombinedData_buffer3_refocus-tikhonov_iq.hdf5", "REFoCUS tikhonov"),
    ("CombinedData_buffer3_refocus-tsvd_iq.hdf5", "REFoCUS tsvd"),
    ("CombinedData_buffer3_pfieldnorm_iq.hdf5", "/ pfield compound"),
    ("CombinedData_buffer3_deconv-g050_iq.hdf5", "/ harmonic field g=0.5"),
    ("CombinedData_buffer3_deconv-g100_iq.hdf5", "/ harmonic field g=1.0"),
]


def load_env(folder, name):
    with h5py.File(os.path.join(folder, name), "r") as f:
        g = f["tracks/track_0/data/beamformed_data"]
        iq = np.asarray(g["values"][:])
        co = np.asarray(g["coordinates"])
    return np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2).mean(axis=0), co


def polar(env, co, r_lo, r_hi, n_th=1501, th_max=34.0, n_r=200):
    x, z = co[..., 0], co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-th_max, th_max, n_th))
    r = np.linspace(r_lo, r_hi, n_r)
    R, TH = np.meshgrid(r, th, indexing="ij")
    jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
    ii = np.clip(((R * np.cos(TH) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
    pol = env[ii, jj]
    return pol / (np.median(pol, axis=1, keepdims=True) + 1e-12), np.degrees(th[1] - th[0])


def ripple_profile(pol, dth):
    """Depth-average, remove the 10 deg trend, return the fractional angular ripple."""
    prof = pol.mean(axis=0)
    trend = uniform_filter1d(prof, int(round(10.0 / dth)) | 1, mode="nearest")
    return (prof - trend) / (trend + 1e-12)


def metrics(env, co, band):
    pol, dth = polar(env, co, *band)
    rip = ripple_profile(pol, dth)

    spec = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    freq = np.fft.rfftfreq(rip.size, d=dth)
    tot = spec[(freq > 0.05) & (freq < 5)].sum()
    fl = 1.0 / LINE_SPACING_DEG
    frac = spec[(freq > fl * 0.9) & (freq < fl * 1.1)].sum() / tot
    ang_rms = rip.std() * 100
    rip_line = np.sqrt(frac) * ang_rms

    # depth persistence: speckle decorrelates with range, a transmit-field streak does not
    h = pol.shape[0] // 2
    near = ripple_profile(pol[:h], dth)
    far = ripple_profile(pol[h:], dth)
    persist = float(np.corrcoef(near, far)[0, 1])

    b = env[env.shape[0] // 4: env.shape[0] * 3 // 4]
    b = b[b > 0]
    ins = env[env > 0]
    return dict(rip_line=rip_line, ang_rms=ang_rms, persist=persist,
                spk_snr=float(b.mean() / (b.std() + 1e-20)),
                dyn_rng=float(20 * np.log10(np.percentile(ins, 99.9) / np.percentile(ins, 5))))


# ---- point targets, phantom only -------------------------------------------------
def find_targets(env, z, z_range=(30, 105), prominence_db=12.0, min_sep_px=14):
    db = 20 * np.log10(env / env.max() + 1e-12)
    local_max = maximum_filter(db, size=min_sep_px) == db
    bg = uniform_filter1d(uniform_filter1d(db, 61, axis=0), 61, axis=1)
    zz = z[:, None] * np.ones_like(db)
    ok = local_max & (db - bg > prominence_db) & (zz > z_range[0]) & (zz < z_range[1])
    return list(zip(*[a.tolist() for a in np.nonzero(ok)]))


def fwhm(profile, step_mm, peak_idx, drop_db=6.0):
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
    return None if left is None or right is None else (right - left) * step_mm


def psf(env, co, targets):
    x = co[0, :, 0] * 1e3
    z = co[:, 0, -1] * 1e3
    dx, dz = abs(x[1] - x[0]), abs(z[1] - z[0])
    lat, cnr = [], []
    for iz, ix in targets:
        w = fwhm(env[iz], dx, ix)
        if w is None or fwhm(env[:, ix], dz, iz) is None:
            continue
        z0, z1 = max(iz - 40, 0), min(iz + 40, env.shape[0])
        x0, x1 = max(ix - 40, 0), min(ix + 40, env.shape[1])
        lat.append(w)
        cnr.append(20 * np.log10(env[iz, ix] / (np.median(env[z0:z1, x0:x1]) + 1e-20)))
    if not lat:
        return np.nan, np.nan, 0
    return float(np.median(lat)), float(np.median(cnr)), len(lat)


# ---- run --------------------------------------------------------------------------
all_rows = {}
for dsname, folder, band, do_psf in DATASETS:
    print(f"\n=== {dsname} ===")
    targets = None
    if do_psf:
        env0, co0 = load_env(folder, METHODS[0][0])
        targets = find_targets(env0, co0[:, 0, -1] * 1e3)
        print(f"{len(targets)} point targets from the standard reconstruction")
    print(f"{'method':26s} {'rip_line':>9s} {'ang_rms':>8s} {'persist':>8s} "
          f"{'lat':>7s} {'CNR':>7s} {'spk_snr':>8s} {'dyn_rng':>8s}")
    print("-" * 92)
    for fn, lab in METHODS:
        try:
            env, co = load_env(folder, fn)
        except (OSError, KeyError):
            print(f"{lab:26s}  (not available)")
            continue
        m = metrics(env, co, band)
        if do_psf:
            m["lat"], m["cnr"], m["n"] = psf(env, co, targets)
        all_rows[(dsname, lab)] = m
        lat = f"{m['lat']:6.2f}m" if do_psf else "     - "
        cnr = f"{m['cnr']:6.1f}d" if do_psf else "     - "
        print(f"{lab:26s} {m['rip_line']:8.2f}% {m['ang_rms']:7.1f}% {m['persist']:8.3f} "
              f"{lat} {cnr} {m['spk_snr']:8.2f} {m['dyn_rng']:7.1f}dB")

np.save(os.path.join(OUT, "method_scorecard.npy"), all_rows, allow_pickle=True)
print("\nsaved method_scorecard.npy")
