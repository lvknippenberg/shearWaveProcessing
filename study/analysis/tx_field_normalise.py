"""Synthesise the COMPLEX transmit field per transmit, compound it coherently, normalise by it.

zea's ``compute_pfield`` returns float32 pressure-field MAGNITUDES, so the only compound it can
build is ``sum_tx |A_tx|`` - incoherent, and smooth by construction (measured: 0.018% ripple at
the line spacing, against the image's 2.02%). Dividing by that is a no-op, which is what we saw.

The image's scalloping comes from the COHERENT compound. For a point scatterer at p, transmit i
deposits ``A_i(p)`` and receive beamforming contributes a factor common to all transmits, so the
reconstructed image goes as ``|sum_i A_i(p)|``. That is the map we need, and it requires phase.

So compute it directly from the transmit geometry - a monochromatic Rayleigh-Sommerfeld sum over
elements, using the delays and apodisation the sequence actually used::

    A_i(p) = sum_e  a_ie * exp(-j 2 pi f (tau_ie + |p - r_e| / c)) / |p - r_e|

Two frequencies are worth trying and the data decides between them: the transmit fundamental
(1.95 MHz), and the 2nd harmonic the images are actually formed at (3.9 MHz) - harmonic
generation roughly squares the fundamental amplitude and doubles its phase. The map that carries
ripple at the measured 1.284 deg is the relevant one; if neither does, the scalloping hypothesis
is wrong and this closes it out.
"""
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, r"D:/Luuk van Knippenberg/Github/shearWaveProcessing/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from zea import File

from swp.acquisition.beamform import apply_grid
from swp.acquisition.sequence import read_swi_meta

ROOT = r"D:\swp_res\Resolution phantom\DefaultPatient_SW_data_18-June-2026_13-52-51"
OUT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PIX_CHUNK = 20000


def transmit_field(grid, probe_xyz, delays, apod, f_hz, c, harmonic=False):
    """Coherent compound |sum_tx A_tx(p)| on the grid. Returns (nz, nx) float32."""
    nz, nx = grid.shape[:2]
    pts = torch.as_tensor(grid.reshape(-1, 3), dtype=torch.float32, device=DEV)
    el = torch.as_tensor(probe_xyz, dtype=torch.float32, device=DEV)          # (n_el, 3)
    tau = torch.as_tensor(delays, dtype=torch.float32, device=DEV)            # (n_tx, n_el)
    ap = torch.as_tensor(apod, dtype=torch.float32, device=DEV)               # (n_tx, n_el)
    out = torch.zeros(pts.shape[0], dtype=torch.complex64, device=DEV)
    w = 2.0 * np.pi * f_hz
    for s in range(0, pts.shape[0], PIX_CHUNK):
        p = pts[s:s + PIX_CHUNK]                                             # (P, 3)
        d = torch.cdist(p, el).clamp_min(1e-4)                               # (P, n_el)
        # phase from element e to pixel p, plus that element's transmit delay
        ph = -w * (tau[:, None, :] + d[None] / c)                            # (n_tx, P, n_el)
        amp = ap[:, None, :] / d[None]
        if harmonic:
            # 2nd harmonic: amplitude ~ fundamental^2, phase doubled
            field = (amp * torch.exp(1j * ph)) ** 2
        else:
            field = amp * torch.exp(1j * ph)
        out[s:s + PIX_CHUNK] = field.sum(dim=-1).sum(dim=0)                   # sum el, then tx
    return out.abs().reshape(nz, nx).cpu().numpy().astype(np.float32)


meta = read_swi_meta(os.path.join(ROOT, "CombinedData.mat"))
with File(os.path.join(ROOT, "output", "converted", "CombinedData_buffer3.hdf5")) as f:
    params = f.load_parameters()
apply_grid(params, meta.grids[2])

grid = np.asarray(params.grid, np.float32)
probe = np.asarray(params.probe_geometry, np.float32)
delays = np.asarray(params.t0_delays, np.float32)
apod = np.asarray(params.tx_apodizations, np.float32)
c = float(params.sound_speed)
f_tx = float(np.asarray(params.center_frequency).reshape(-1)[0])
print(f"device {DEV}; grid {grid.shape}, {delays.shape[0]} transmits x {probe.shape[0]} elements")
print(f"transmit frequency {f_tx / 1e6:.3f} MHz, c = {c} m/s")

for lab, f_hz, harm in (("fundamental", f_tx, False), ("2nd harmonic", f_tx, True)):
    t0 = time.perf_counter()
    S = transmit_field(grid, probe, delays, apod, f_hz, c, harmonic=harm)
    S = S / np.median(S[S > 0])
    np.save(os.path.join(OUT, f"txfield_{'harm' if harm else 'fund'}.npy"), S)
    print(f"  {lab:12s}: {time.perf_counter() - t0:5.0f}s  "
          f"span {20 * np.log10(S.max() / max(S[S > 0].min(), 1e-9)):5.1f} dB  "
          f"-> txfield_{'harm' if harm else 'fund'}.npy")
