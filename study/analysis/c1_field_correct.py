"""Apply the phantom's two field-normalisation maps to C000000001 buffer 3, in vivo.

The phantom (README §4a) produced two candidate correction maps for the focused buffer:

* **zea ``compute_pfield``**, compounded as ``sum_tx |A_tx|``. Magnitudes only, so the compound is
  smooth by construction (0.02% ripple against the image's 2.02%). On the phantom this was a
  **no-op**. Included here so the in-vivo answer is measured rather than assumed.
* **A synthesised COMPLEX field** at the 2nd harmonic, compounded as ``|sum_tx A_tx|``. On the
  phantom this carried ripple at 1.260 deg against the image's 1.284 deg - the right periodicity -
  but it was **anti-correlated** with the image (-0.68), so dividing by it *increased* the ripple.
  What it did do, unexpectedly, was sharpen: lateral PSF 1.40 -> 1.07 mm and CNR 13.8 -> 18.2 dB
  at gamma 0.5-1.0, behaving as a crude deconvolution of the transmit field.

That sharpening is the only reason to run this in vivo. **It cannot be confirmed here**: in vivo
there are no point targets, so there is no PSF to measure, and the lateral-correlation length is
an anatomy-scale proxy - the exact proxy that produced the wrong REFoCUS verdict earlier in this
study. So the numbers below are reported as what they are (ripple, speckle SNR, correlation
length, dynamic range) and the deliverable is the **visual** comparison; the phantom remains the
only place the resolution claim is actually measured.

Writes, next to the standard output:
    CombinedData_buffer3_pfieldnorm_iq.hdf5 / .gif      (divided by the pfield compound)
    CombinedData_buffer3_deconv-g<NN>_iq.hdf5 / .gif    (divided by the harmonic field, per gamma)
and a side-by-side montage in the working folder's montages/.
"""
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, r"D:/Luuk van Knippenberg/Github/shearWaveProcessing/src")

import h5py
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from zea import File, init_device
from zea.beamform.pfield import compute_pfield

from swp.acquisition.beamform import DEFAULT_COMPRESSION, _save_beamformed, apply_grid, find_mat
from swp.acquisition.sequence import SPEC_BY_INDEX, read_swi_meta

DEFAULT_FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54"
# Second positional-free arg: any measurement folder. The cached field maps are per-folder,
# because the grid (and therefore the map) differs between the in-vivo and phantom sequences.
FOLDER = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
TAG = sys.argv[2] if len(sys.argv) > 2 else "c1"
OUT = os.path.dirname(os.path.abspath(__file__))
GAMMAS = (0.25, 0.5, 0.75, 1.0)
CLIP_DB = 12.0            # cap the correction so a field null cannot manufacture signal
LINE_SPACING_DEG = 1.1111
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PIX_CHUNK = 20000


# --------------------------------------------------------------------------- metrics
def angular_ripple(env, co, r_lo=0.045, r_hi=0.085, n_th=1501, th_max=34.0):
    """Frame-averaged angular ripple in a speckle band: (% at line spacing, RMS %, peak deg).

    Frame-averaged is the point - a single frame is speckle-dominated, which is how a genuine
    20x enrichment was missed earlier in this study.
    """
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
    sel = (freq > 0.3) & (freq < 2.0)          # fit the peak; never trust the window alone
    return spec[band].sum() / tot * 100, rip.std() * 100, 1.0 / freq[sel][np.argmax(spec[sel])]


def lateral_corr_mm(env, co):
    """Lateral autocorrelation -6 dB half-width. An ANATOMY-scale proxy, not a PSF."""
    x = co[0, :, 0] * 1e3
    dx = abs(x[1] - x[0])
    band = env[env.shape[0] // 4: env.shape[0] * 3 // 4]
    band = band - band.mean(axis=1, keepdims=True)
    ac = np.zeros(60)
    for lag in range(60):
        a = band[:, : band.shape[1] - lag]
        b = band[:, lag:]
        ac[lag] = (a * b).mean()
    ac /= ac[0] + 1e-20
    below = np.nonzero(ac < 0.5)[0]
    return (below[0] * dx) if below.size else np.nan


def speckle_snr(env):
    """Envelope mean/std in the mid-field. ~1.91 for fully developed speckle."""
    b = env[env.shape[0] // 4: env.shape[0] * 3 // 4]
    b = b[b > 0]
    return float(b.mean() / (b.std() + 1e-20))


def dyn_range_db(env):
    ins = env[env > 0]
    return float(20 * np.log10(np.percentile(ins, 99.9) / np.percentile(ins, 5)))


# --------------------------------------------------------------------------- field synthesis
def transmit_field(grid, probe_xyz, delays, apod, f_hz, c, harmonic=False):
    """Coherent compound |sum_tx A_tx(p)| on the grid -> (nz, nx) float32."""
    nz, nx = grid.shape[:2]
    pts = torch.as_tensor(grid.reshape(-1, 3), dtype=torch.float32, device=DEV)
    el = torch.as_tensor(probe_xyz, dtype=torch.float32, device=DEV)
    tau = torch.as_tensor(delays, dtype=torch.float32, device=DEV)
    ap = torch.as_tensor(apod, dtype=torch.float32, device=DEV)
    out = torch.zeros(pts.shape[0], dtype=torch.complex64, device=DEV)
    w = 2.0 * np.pi * f_hz
    for s in range(0, pts.shape[0], PIX_CHUNK):
        p = pts[s:s + PIX_CHUNK]
        d = torch.cdist(p, el).clamp_min(1e-4)
        ph = -w * (tau[:, None, :] + d[None] / c)
        amp = ap[:, None, :] / d[None]
        field = (amp * torch.exp(1j * ph)) ** 2 if harmonic else amp * torch.exp(1j * ph)
        out[s:s + PIX_CHUNK] = field.sum(dim=-1).sum(dim=0)
    return out.abs().reshape(nz, nx).cpu().numpy().astype(np.float32)


def normalised(S, inside):
    """Median-normalise inside the sector and clip, so nulls cannot blow up the division."""
    S = S / np.median(S[inside])
    lo, hi = 10 ** (-CLIP_DB / 20), 10 ** (CLIP_DB / 20)
    return np.clip(S, lo, hi)


# --------------------------------------------------------------------------- main
init_device(verbose=True)
try:
    mat = find_mat(FOLDER)
except Exception as exc:
    # The phantom has no campaign-matched base config on Z: (it needs the same session's
    # elasticity-phantom CombinedData.mat, passed explicitly at beamform time). Its
    # CombinedData.mat is already written and correct, so read it directly here.
    mat = Path(FOLDER) / "CombinedData.mat"
    if not mat.is_file():
        raise
    print(f"base-config validation skipped ({type(exc).__name__}); using {mat}")
out_dir = os.path.join(FOLDER, "output")
spec = SPEC_BY_INDEX[2]
meta = read_swi_meta(mat)
fps = meta.fps.get(2)

with File(os.path.join(out_dir, "converted", f"{mat.stem}_buffer3.hdf5")) as fh:
    params = fh.load_parameters()
apply_grid(params, meta.grids[2])

with h5py.File(os.path.join(out_dir, f"{mat.stem}_buffer3_iq.hdf5"), "r") as f:
    g = f["tracks/track_0/data/beamformed_data"]
    iq = np.asarray(g["values"][:])
    co = np.asarray(g["coordinates"])
env_frames = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2).astype(np.float32)
env_mean = env_frames.mean(axis=0)
inside = env_mean > 0
print(f"{TAG} buffer 3: {env_frames.shape[0]} frames, grid {env_mean.shape}, {fps:.1f} FPS")

grid = np.asarray(params.grid, np.float32)
assert grid.shape[:2] == env_mean.shape, f"grid {grid.shape} vs image {env_mean.shape}"
probe = np.asarray(params.probe_geometry, np.float32)
delays = np.asarray(params.t0_delays, np.float32)
apod = np.asarray(params.tx_apodizations, np.float32)
c = float(params.sound_speed)
f_tx = float(np.asarray(params.center_frequency).reshape(-1)[0])

# --- map 1: zea pfield, compounded incoherently
pf_path = os.path.join(OUT, f"{TAG}_pfield_sensitivity.npy")
if os.path.isfile(pf_path):
    S_pf = np.load(pf_path)
else:
    bw = getattr(params, "probe_bandwidth_percent", None)
    t0 = time.perf_counter()
    pf = compute_pfield(sound_speed=c, center_frequency=f_tx,
                        probe_bandwidth_percent=float(bw) if bw is not None else 60.0,
                        n_el=int(params.n_el), probe_geometry=probe, tx_apodizations=apod,
                        grid=grid, t0_delays=delays,
                        downsample=1,      # must resolve the ~1.5 mm scalloping
                        norm=False)
    pf = np.asarray(pf.cpu() if hasattr(pf, "cpu") else pf)   # returns a device tensor
    assert pf.shape[1:] == grid.shape[:2], f"pfield {pf.shape} vs grid {grid.shape}"
    S_pf = np.abs(pf).sum(axis=0)                              # transmit axis is FIRST
    np.save(pf_path, S_pf)
    print(f"  compute_pfield: {time.perf_counter() - t0:.0f}s")

# --- map 2: synthesised complex field, 2nd harmonic
hm_path = os.path.join(OUT, f"{TAG}_txfield_harm.npy")
if os.path.isfile(hm_path):
    S_hm = np.load(hm_path)
else:
    t0 = time.perf_counter()
    S_hm = transmit_field(grid, probe, delays, apod, f_tx, c, harmonic=True)
    np.save(hm_path, S_hm)
    print(f"  synthesised harmonic field: {time.perf_counter() - t0:.0f}s")

S_pf = normalised(S_pf, inside)
S_hm = normalised(S_hm, inside)
for lab, S in (("pfield compound", S_pf), ("synth harmonic", S_hm)):
    p, rms, pk = angular_ripple(S, co)
    r = np.corrcoef(S[inside], env_mean[inside])[0, 1]
    print(f"  {lab:16s}: span {20 * np.log10(S[inside].max() / S[inside].min()):5.1f} dB  "
          f"ripple {np.sqrt(p / 100) * rms:5.2f}% @ {pk:.3f} deg  corr with image {r:+.3f}")

# --------------------------------------------------------------------------- evaluate
print()
print(f"{'reconstruction':30s} {'ripple':>8s} {'peak':>9s} {'lat.corr':>9s} {'spk SNR':>8s} "
      f"{'dyn.rng':>8s}")
print("-" * 78)
variants = [("standard (no correction)", None)]
variants += [("/ pfield compound", ("pfield", 1.0))]
variants += [(f"/ synth harmonic, gamma {g:.2f}", ("harm", g)) for g in GAMMAS]

rows = {}
for lab, spec_v in variants:
    if spec_v is None:
        e = env_mean
    else:
        S = S_pf if spec_v[0] == "pfield" else S_hm
        e = env_mean / (S ** spec_v[1])
    p, rms, pk = angular_ripple(e, co)
    amp = np.sqrt(p / 100) * rms
    rows[lab] = (amp, lateral_corr_mm(e, co), speckle_snr(e), dyn_range_db(e))
    print(f"{lab:30s} {amp:7.2f}% {pk:8.3f}deg {rows[lab][1]:8.2f}mm {rows[lab][2]:8.2f} "
          f"{rows[lab][3]:7.1f}dB")

# --------------------------------------------------------------------------- write images
def write(env3d, suffix, description):
    out = np.zeros(env3d.shape + (2,), np.float32)
    out[..., 0] = env3d
    path = Path(out_dir) / f"{mat.stem}_buffer3_{suffix}_iq.hdf5"
    _save_beamformed(path, out, np.asarray(params.grid, np.float32), fps=fps,
                     description=description, compression=DEFAULT_COMPRESSION)
    from swp.acquisition.gifs import gif_for_file
    gif_for_file(path)
    print(f"  wrote {path.name} + .gif")
    return path


print()
write(env_frames / S_pf, "pfieldnorm",
      f"{spec.name}: divided by the compounded pfield transmit sensitivity")
for g in (0.5, 1.0):
    write(env_frames / (S_hm ** g),
          f"deconv-g{int(g * 100):03d}",
          f"{spec.name}: divided by the synthesised 2nd-harmonic transmit field, gamma {g}")
np.save(os.path.join(OUT, f"{TAG}_field_correct_metrics.npy"), rows, allow_pickle=True)
print(f"\nsaved {TAG}_field_correct_metrics.npy")
