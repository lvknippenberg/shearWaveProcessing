"""Metrics for comparing buffer-1 reconstructions.

Deliberately biased towards measures that CANNOT be gamed by smoothing or by a change of
dynamic range, because the striations investigation was repeatedly misled by metrics that
rewarded blur (docs/focused_bmode_striations.md S5a/S5b/S5c):

* ``gcnr``  - generalised CNR (overlap of two ROI histograms).  Invariant under ANY
  monotonic transform of the envelope, so log compression, gain and gamma cannot move it.
* ``speckle_snr`` - mean/std of the envelope in speckle; 1.91 for fully developed speckle.
  Rises when signal is added coherently, rises when the image is blurred - so it is only
  read together with the resolution numbers.
* ``fwhm`` / ``measure_targets`` - -6 dB widths on the phantom's wire targets.  The only
  honest resolution measure available (an anatomy-scale correlation length is not one).
* ``noise_gain_db`` - level in a signal-free region relative to a tissue region.  This is
  the metric the whole experiment is about: off-axis transmits add noise but no signal.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter1d


def env_of(iq_complex):
    return np.abs(iq_complex)


def axes_mm(coords):
    """(x_mm (nx,), z_mm (nz,)) of a zea coordinate grid."""
    return coords[0, :, 0] * 1e3, coords[:, 0, -1] * 1e3


def roi_mask(coords, xlim_mm, zlim_mm):
    x, z = axes_mm(coords)
    return ((z[:, None] >= zlim_mm[0]) & (z[:, None] <= zlim_mm[1]) &
            (x[None, :] >= xlim_mm[0]) & (x[None, :] <= xlim_mm[1]))


def sector_mask(coords, half_deg=40.0, z_min_mm=5.0, apex_mm=0.0):
    """Pixels inside the transmitted sector (the only ones that carry data)."""
    x, z = axes_mm(coords)
    X, Z = np.meshgrid(x, z)
    ang = np.degrees(np.arctan2(X, Z - apex_mm))
    return (np.abs(ang) <= half_deg) & (Z >= z_min_mm)


# ------------------------------------------------------------------ contrast
def gcnr(a, b, bins=256):
    """Generalised CNR between two envelope samples: 1 - histogram overlap.

    Invariant under any monotonic transform of the envelope, so it cannot be inflated by
    log compression, gain, or a tone curve - unlike CNR or contrast ratio.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, edges, density=False)
    hb, _ = np.histogram(b, edges, density=False)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(1.0 - np.minimum(ha, hb).sum())


def contrast_db(a, b):
    """20 log10(mean a / mean b) - plain envelope contrast between two regions."""
    return float(20 * np.log10((np.mean(a) + 1e-20) / (np.mean(b) + 1e-20)))


def cnr(a, b):
    """|mu_a - mu_b| / sqrt(var_a + var_b)."""
    a, b = np.asarray(a), np.asarray(b)
    return float(abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var() + 1e-20))


def speckle_snr(a):
    a = np.asarray(a)
    return float(a.mean() / (a.std() + 1e-20))


def dynamic_range_db(env, mask=None, hi_pct=99.9, lo_pct=1.0):
    v = env[mask] if mask is not None else env.ravel()
    v = v[v > 0]
    if v.size == 0:
        return float("nan")
    return float(20 * np.log10(np.percentile(v, hi_pct) / max(np.percentile(v, lo_pct), 1e-20)))


# -------------------------------------------------------------- resolution
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


def find_targets(env, coords, z_range=(25, 110), prominence_db=12.0, min_sep_px=14):
    """Wire targets: local maxima standing well above their local background."""
    x, z = axes_mm(coords)
    db = 20 * np.log10(env / env.max() + 1e-12)
    local_max = maximum_filter(db, size=min_sep_px) == db
    bg = uniform_filter1d(uniform_filter1d(db, 61, axis=0), 61, axis=1)
    zz = z[:, None] * np.ones_like(db)
    ok = local_max & (db - bg > prominence_db) & (zz > z_range[0]) & (zz < z_range[1])
    iz, ix = np.nonzero(ok)
    return [(int(a), int(b)) for a, b in zip(iz, ix)]


def measure_targets(env, coords, targets):
    x, z = axes_mm(coords)
    dx, dz = abs(x[1] - x[0]), abs(z[1] - z[0])
    rows = []
    for iz, ix in targets:
        lat = fwhm(env[iz], dx, ix)
        ax = fwhm(env[:, ix], dz, iz)
        if lat is None or ax is None:
            continue
        z0, z1 = max(iz - 40, 0), min(iz + 40, env.shape[0])
        x0, x1 = max(ix - 40, 0), min(ix + 40, env.shape[1])
        patch = env[z0:z1, x0:x1]
        rows.append(dict(z=z[iz], x=x[ix], lat=lat, ax=ax,
                         cnr=20 * np.log10(env[iz, ix] / (np.median(patch) + 1e-20))))
    return rows


def target_summary(rows):
    if not rows:
        return dict(n=0, lat=np.nan, ax=np.nan, cnr=np.nan)
    return dict(n=len(rows),
                lat=float(np.median([r["lat"] for r in rows])),
                ax=float(np.median([r["ax"] for r in rows])),
                cnr=float(np.median([r["cnr"] for r in rows])))


# -------------------------------------------------------- coverage / uniformity
def sector_coverage(env, coords, db_floor=-50.0, ref_pct=99.9):
    """Fraction of in-sector pixels above ``db_floor`` relative to the clip's bright end.

    The field-of-view metric used for the pfield comparison in ``BufferSpec.pfield``.
    """
    m = sector_mask(coords)
    v = env[m]
    ref = np.percentile(v[v > 0], ref_pct)
    return float((20 * np.log10(v / ref + 1e-12) > db_floor).mean() * 100)


def depth_profile(env, coords, n_bins=30, half_deg=30.0):
    """Mean in-sector envelope vs depth, for penetration curves."""
    x, z = axes_mm(coords)
    X, Z = np.meshgrid(x, z)
    m = (np.abs(np.degrees(np.arctan2(X, Z))) <= half_deg) & (Z > 5)
    edges = np.linspace(z.min(), z.max(), n_bins + 1)
    idx = np.digitize(Z, edges) - 1
    prof, ctr = [], []
    for b in range(n_bins):
        sel = m & (idx == b)
        prof.append(env[sel].mean() if sel.any() else np.nan)
        ctr.append(0.5 * (edges[b] + edges[b + 1]))
    return np.array(ctr), np.array(prof)
