"""Composite the 73 per-transmit reconstructions different ways and measure the streaks.

Rules compared:
  all      - coherent sum of all 73 (what the pipeline does today)
  angN     - coherent sum weighted by a Gaussian in |theta_pixel - theta_tx|, width N beams
  nearest  - each pixel from its single nearest beam (classic scan conversion)
  incoh    - incoherent (envelope) sum of all 73

Streak metric: after removing the sector envelope AND the tissue structure (median filter in
range), the residual angular ripple that is COHERENT ACROSS DEPTH. Speckle averages down with
depth; a radial streak does not - so this isolates exactly the artefact in question.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

OUT = os.path.dirname(os.path.abspath(__file__))
per_tx = np.load(os.path.join(OUT, "buf3_per_tx.npy"))     # (73, nz, nx, 2)
coords = np.load(os.path.join(OUT, "buf3_coords.npy"))     # (nz, nx, 3)
az = np.load(os.path.join(OUT, "buf3_az.npy"))             # (73,) radians
n_tx = per_tx.shape[0]
cplx = per_tx[..., 0] + 1j * per_tx[..., 1]                # (73, nz, nx)

x = coords[..., 0]
z = coords[..., -1]
th_pix = np.arctan2(x, np.maximum(z, 1e-6))                # (nz, nx) azimuth of each pixel
d_az = float(np.median(np.diff(np.sort(az))))              # beam spacing, rad
print(f"beam spacing {np.degrees(d_az):.4f} deg, {n_tx} beams")

# angular distance pixel <-> each beam
D = th_pix[None, :, :] - az[:, None, None]                 # (73, nz, nx)


def composite(rule):
    if rule == "all":
        return np.abs(cplx.sum(axis=0))
    if rule == "incoh":
        return np.abs(cplx).sum(axis=0)
    if rule == "nearest":
        k = np.argmin(np.abs(D), axis=0)
        return np.abs(np.take_along_axis(cplx, k[None], axis=0)[0])
    if rule.startswith("ang"):
        n_beams = float(rule[3:])
        sigma = n_beams * d_az
        w = np.exp(-0.5 * (D / sigma) ** 2).astype(np.float32)
        w /= (w.sum(axis=0, keepdims=True) + 1e-12)
        return np.abs((cplx * w).sum(axis=0))
    raise ValueError(rule)


def polar(env, n_th=1201, n_r=320, r_lo=0.03, r_hi=0.14):
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-38, 38, n_th))
    r = np.linspace(r_lo, r_hi, n_r)
    R, TH = np.meshgrid(r, th, indexing="ij")
    j = (R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)
    i = (R * np.cos(TH) - z0) / (z1 - z0) * (nz - 1)
    return th, r, env[np.clip(i.astype(int), 0, nz - 1), np.clip(j.astype(int), 0, nx - 1)]


def streak_metric(env, label):
    """Ripple that survives depth-averaging = radial streaks (speckle averages down)."""
    th, r, pol = polar(env)
    pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)     # per-depth gain
    dth = np.degrees(th[1] - th[0])
    w = int(round(4.0 / dth)) | 1
    prof = pol.mean(axis=0)                                          # average over depth
    ripple = (prof - uniform_filter1d(prof, w, mode="nearest"))
    ripple /= (uniform_filter1d(prof, w, mode="nearest") + 1e-12)
    # per-depth ripple, for the "does it average down?" comparison
    single = pol[pol.shape[0] // 2]
    s_rip = (single - uniform_filter1d(single, w, mode="nearest"))
    s_rip /= (uniform_filter1d(single, w, mode="nearest") + 1e-12)
    n_dep = pol.shape[0]
    expected = s_rip.std() / np.sqrt(n_dep)        # if it were pure speckle
    print(f"  {label:10s} depth-avg ripple {ripple.std()*100:6.2f}%   "
          f"single-depth {s_rip.std()*100:6.2f}%   "
          f"speckle-only expectation {expected*100:5.2f}%   "
          f"excess x{ripple.std()/max(expected,1e-9):5.2f}")
    return th, ripple, ripple.std()


rules = ["all", "ang0.5", "ang1", "ang2", "nearest", "incoh"]
print("\nstreak metric (excess = how much more angular ripple survives depth-averaging\n"
      "than pure speckle would; 1.0 = no radial streaks):")
res, envs = {}, {}
for rule in rules:
    e = composite(rule)
    envs[rule] = e
    res[rule] = streak_metric(e, rule)

fig, axes = plt.subplots(2, 3, figsize=(19, 11))
for ax, rule in zip(axes.ravel(), rules):
    e = envs[rule]
    d = 20 * np.log10(e / e.max() + 1e-12)
    ax.imshow(d, cmap="gray", vmin=-50, vmax=0, aspect="auto")
    ax.set_title(f"{rule}   excess x{res[rule][2] / 1:.3g}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("buffer 3 frame 0 - per-transmit composite rules (-50..0 dB)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "buf3_composite.png"), dpi=95)
print("\nwrote buf3_composite.png")
