"""Shared library for the buffer-1 (widebeam) reconstruction experiments.

Buffer 1 of the SWI Widebeam sequence is 21 wide beams, 4.0 deg apart over +/-40 deg,
each from a virtual source 123.2 mm BEHIND the array with the full 80-element
(20.07 mm) Hann-tapered aperture.  The geometric opening of one beam is therefore only
~9.3 deg: at any pixel only about 2.3 of the 21 transmits actually insonified it, yet the
pipeline compounds all 21.  The other ~19 contribute noise, clutter and off-axis
sidelobe energy with no signal.

Everything here works off a PER-TRANSMIT COMPLEX STACK: each transmit is beamformed
separately onto the full PData grid and cached, after which ANY per-pixel/per-transmit
weight map can be evaluated for free (compounding is a linear sum over transmits, so
compositing the cached stack is algebraically identical to running zea's
``Beamform(enable_aligned_apodization=True)`` with the same mask).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO = Path("D:/Luuk van Knippenberg/Github/shearWaveProcessing")
sys.path.insert(0, str(_REPO / "src"))

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
FIGS = HERE / "figs"
CACHE.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

DATASETS = {
    "invivo": dict(
        root=Path("Z:/raw_data/C000000001/SWE_01_SW_data_21-April-2026_12-12-54"),
        frames="spread",
    ),
    "invivo2": dict(
        root=Path("Z:/raw_data/C000000002/SWE02_SW_data_28-April-2026_10-09-45"),
        frames="spread",
    ),
    "invivo3": dict(
        root=Path("Z:/raw_data/C000000044/VIS-034_SW_data_28-July-2026_11-02-22"),
        frames="spread",
    ),
    "phantom": dict(
        root=Path("D:/swp_res/Resolution phantom/DefaultPatient_SW_data_18-June-2026_13-52-51"),
        frames="all",
    ),
}
BUFFER = 1


# ----------------------------------------------------------------- geometry
class TxGeometry:
    """Per-transmit virtual-source geometry of the widebeam buffer."""

    def __init__(self, params, coords):
        d = dict(params)
        self.theta = np.asarray(d["polar_angles"], np.float64)          # (n_tx,) rad
        self.origin = np.asarray(d["transmit_origins"], np.float64)     # (n_tx, 3) m
        self.focus = np.asarray(d["focus_distances"], np.float64)       # (n_tx,) m, <0 = behind
        self.apod = np.asarray(d["tx_apodizations"], np.float64)        # (n_tx, n_el)
        self.probe = np.asarray(d["probe_geometry"], np.float64)        # (n_el, 3) m
        self.n_tx = self.theta.size

        # Virtual source of every transmit: origin + focus * beam direction.
        u = np.stack([np.sin(self.theta), np.cos(self.theta)], axis=1)  # (n_tx,2) beam dir
        self.dirn = u
        self.vs = self.origin[:, [0, 2]] + self.focus[:, None] * u      # (n_tx,2) [x,z] m

        self.px = np.asarray(coords[..., 0], np.float64)                # (nz,nx)
        self.pz = np.asarray(coords[..., -1], np.float64)
        self.shape = self.px.shape

        self._phi = None

    @property
    def phi(self):
        """(n_tx, nz, nx) signed angle of every pixel off each beam axis, radians."""
        if self._phi is None:
            out = np.empty((self.n_tx,) + self.shape, np.float32)
            for i in range(self.n_tx):
                dx = self.px - self.vs[i, 0]
                dz = self.pz - self.vs[i, 1]
                ang = np.arctan2(dx, dz) - np.arctan2(self.dirn[i, 0], self.dirn[i, 1])
                out[i] = np.arctan2(np.sin(ang), np.cos(ang))
            self._phi = out
        return self._phi

    def phi_max(self, apod_thresh=0.0):
        """Geometric half-opening angle of each beam, from the aperture edges.

        ``apod_thresh`` selects which elements count as 'the aperture': 0 gives the
        full 80-element geometric cone, 0.5 the -6 dB effective aperture of the
        Hann-tapered apodization.
        """
        out = np.zeros(self.n_tx)
        for i in range(self.n_tx):
            sel = self.apod[i] > apod_thresh
            ex, ez = self.probe[sel, 0], self.probe[sel, 2]
            dx, dz = ex - self.vs[i, 0], ez - self.vs[i, 1]
            ang = np.arctan2(dx, dz) - np.arctan2(self.dirn[i, 0], self.dirn[i, 1])
            ang = np.arctan2(np.sin(ang), np.cos(ang))
            out[i] = np.abs(ang).max()
        return out

    def summary(self):
        pm = np.degrees(self.phi_max(0.0))
        p6 = np.degrees(self.phi_max(0.5))
        pitch = np.degrees(np.median(np.diff(self.theta)))
        # How many transmits actually reach a pixel: count the geometric cones that
        # contain it, at the sector centre, 100 mm deep.  (An 'overlap' computed as
        # 2*phi_max/pitch is WRONG: phi_max is an angle about the transmit's own
        # virtual source 123 mm behind the array, while the pitch is an angle about
        # the array, and at 100 mm depth the two differ by ~2.2x.)
        iz = np.argmin(np.abs(self.pz[:, 0] - 0.100))
        ix = np.argmin(np.abs(self.px[0, :] - 0.0))
        inside = (np.abs(self.phi[:, iz, ix]) <=
                  self.phi_max(0.0)).sum()
        return (f"{self.n_tx} transmits, pitch {pitch:.3f} deg, virtual source "
                f"{self.focus[0] * 1e3:.1f} mm; geometric half-opening "
                f"{pm.min():.2f}-{pm.max():.2f} deg (full aperture), "
                f"{p6.min():.2f}-{p6.max():.2f} deg (-6 dB aperture); "
                f"{inside} of {self.n_tx} transmits insonify (0 mm, 100 mm)")


# ------------------------------------------------------------ stack builder
def rx_tag(rx):
    """Short cache key for a receive-aperture setting."""
    if not rx:
        return ""
    return "_rx-" + "-".join(f"{k}{rx[k]}" for k in sorted(rx))


def _taper(u, kind):
    """Window on |u| in [0, 1]; 0 outside."""
    a = np.abs(u)
    if kind == "hann":
        return np.where(a <= 1, 0.5 * (1 + np.cos(np.pi * np.clip(a, 0, 1))), 0.0)
    if kind == "hamming":
        return np.where(a <= 1, 0.54 + 0.46 * np.cos(np.pi * np.clip(a, 0, 1)), 0.0)
    if kind.startswith("tukey"):
        f = float(kind[5:]) / 100.0
        ramp = np.clip((a - (1 - f)) / max(f, 1e-9), 0.0, 1.0)
        return np.where(a <= 1, 0.5 * (1 + np.cos(np.pi * ramp)), 0.0)
    if kind == "rect":
        return (a <= 1).astype(np.float64)
    raise ValueError(f"unknown taper {kind!r}")


def receive_apodization(coords, probe_geometry, kind, mode="fixed", f_number=1.0):
    """Per-pixel, per-element receive weight ``(n_pix, n_el)`` for zea.

    zea's built-in ``fnum_window_fn`` tapers across the f-number ACCEPTANCE cone. At the
    pipeline's f/1 that cone (+/-26.6 deg) is far wider than the 20 mm array subtends at any
    useful depth (+/-5.7 deg at 100 mm), so the receive aperture is effectively RECTANGULAR -
    -13 dB sidelobes, and those sidelobes are an off-axis clutter path.

    ``mode="fixed"``   taper across the physical aperture, same at every depth.
    ``mode="dynamic"`` taper across the f-number acceptance cone, i.e. a growing aperture
                       (classical dynamic receive apodization); identical to ``fixed`` once
                       the acceptance cone exceeds the array.
    """
    px = np.asarray(coords[..., 0], np.float64).reshape(-1)
    pz = np.asarray(coords[..., -1], np.float64).reshape(-1)
    ex = np.asarray(probe_geometry, np.float64)[:, 0]
    ez = np.asarray(probe_geometry, np.float64)[:, 2]
    if mode == "fixed":
        half = 0.5 * (ex.max() - ex.min())
        u = (ex - 0.5 * (ex.max() + ex.min())) / half
        return np.broadcast_to(_taper(u, kind)[None, :],
                               (px.size, ex.size)).astype(np.float32).copy()
    # dynamic: normalised receive angle, as zea's fnumber_mask defines it
    alpha = np.arctan2(np.abs(ex[None, :] - px[:, None]), np.abs(pz[:, None] - ez[None, :]))
    alpha_max = np.arctan(1.0 / (2.0 * f_number))
    return _taper(alpha / alpha_max, kind).astype(np.float32)


def build_stack(tag, n_frames=16, overwrite=False, rx=None):
    """Beamform buffer 1 one transmit at a time; cache (n_tx, n_frames, nz, nx) complex64.

    ``rx`` optionally changes the RECEIVE aperture, which cannot be composited offline
    (it acts before the element sum), so each setting needs its own stack:
    ``{"f": <f_number>, "w": <fnum_window_fn: rect|hann|tukey>, "apod": <fixed taper>}``.
    """
    cfg = DATASETS[tag]
    root = cfg["root"]
    npy = CACHE / f"{tag}_b{BUFFER}_pertx{rx_tag(rx)}.npy"
    meta_npz = CACHE / f"{tag}_b{BUFFER}_meta.npz"
    if npy.is_file() and meta_npz.is_file() and not overwrite:
        return np.load(npy, mmap_mode="r"), np.load(meta_npz, allow_pickle=True)

    import zea
    from zea import File, init_device
    from zea.ops import Beamform, Cast, Demodulate
    from swp.acquisition.beamform import _ensure_cpu_t_peak, apply_grid, plan_patches_and_chunk
    from swp.acquisition.sequence import read_swi_meta

    init_device(verbose=False)
    meta = read_swi_meta(root / "CombinedData.mat")
    grid = meta.grids[BUFFER - 1]
    conv = root / "output" / "converted" / f"CombinedData_buffer{BUFFER}.hdf5"
    # The RF lives on a network share and the read dominates (~2 min for 919 MB), so the
    # selected frames are cached locally: a receive-aperture sweep re-beamforms the same RF
    # many times and must not re-read it each time.
    raw_npy = CACHE / f"{tag}_b{BUFFER}_raw{n_frames}.npy"
    with File(str(conv)) as fh:
        n_all = fh.data.raw_data.shape[0]
        if cfg["frames"] == "all" or n_frames >= n_all:
            sel = np.arange(n_all)
        else:
            sel = np.unique(np.linspace(0, n_all - 1, n_frames).astype(int))
        if raw_npy.is_file():
            raw = np.load(raw_npy, mmap_mode="r")
        else:
            raw = np.asarray(fh.data.raw_data[sel])
            np.save(raw_npy, raw)
        params = fh.load_parameters()
    print(f"[{tag}] raw {raw.shape} from {n_all} frames, using {len(sel)}: {sel}")

    apply_grid(params, grid)
    _ensure_cpu_t_peak(params)
    coords = np.asarray(params.grid, np.float32)
    n_tx = raw.shape[1]
    # Capture the FULL transmit geometry now: params.set_transmits() below narrows
    # these arrays to the single selected transmit.
    d0 = dict(params)
    geom = {k: np.asarray(d0[k]) for k in
            ("polar_angles", "transmit_origins", "focus_distances",
             "tx_apodizations", "probe_geometry")}
    num_patches, _ = plan_patches_and_chunk(params, 1, raw.shape[3])
    rx = rx or {}
    if rx.get("bp"):
        # Receive band-pass on the RF, before demodulation.
        #
        # MOTIVATION, AND WHY IT TURNED OUT NOT TO MATTER. This buffer transmits at
        # 1.953 MHz and demodulates at 3.906 MHz - second-harmonic imaging - and zea's
        # ``Demodulate`` applies NO filter (it takes the analytic signal and shifts the
        # spectrum). Any surviving fundamental would therefore sit at -1.953 MHz in the
        # "IQ", where the beamformer's phase rotation is wrong by a factor of two, and land
        # as defocused haze in the echo-free regions. Measured instead (``figs/rf_spectrum.png``):
        # the received band runs 2-5 MHz peaking at ~3.5 MHz, and at 1.953 MHz the spectrum
        # is already ~34 dB down, i.e. at the noise floor. The Verasonics receive filter and
        # the probe response have removed the fundamental before the data is stored, so there
        # is nothing here to reject. Kept as a knob, with the negative result recorded.
        from scipy.signal import butter, sosfiltfilt
        fs = float(np.asarray(dict(params)["sampling_frequency"]).ravel()[0])
        lo, hi = rx["bp"]
        sos = butter(4, [lo * 1e6 / (fs / 2), hi * 1e6 / (fs / 2)], btype="band", output="sos")
        raw = sosfiltfilt(sos, np.asarray(raw, np.float32), axis=2).astype(np.float32)
        print(f"[{tag}] receive band-pass {lo}-{hi} MHz (fs {fs / 1e6:.3f} MHz)")
    if "f" in rx:
        params.f_number = float(rx["f"])
    bf_kwargs = {}
    flat_rx = None
    if rx.get("apod"):
        flat_rx = receive_apodization(coords, geom["probe_geometry"], rx["apod"],
                                      mode=rx.get("mode", "fixed"),
                                      f_number=float(rx.get("f", 1.0)))
        bf_kwargs["enable_receive_apodization"] = True
    pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                         Beamform(beamformer="delay_and_sum", num_patches=num_patches,
                                  enable_pfield=False, **bf_kwargs)],
                        with_batch_dim=True, jit_options=None)
    out = None
    t0 = time.perf_counter()
    for i in range(n_tx):
        params.set_transmits([i])
        bf_in = pipe.prepare_parameters(params)
        if flat_rx is not None:
            bf_in["flat_receive_apodization"] = flat_rx
        iq = pipe(data=np.asarray(raw[:, i:i + 1]), **bf_in)[pipe.output_key]
        iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)
        c = (iq[..., 0] + 1j * iq[..., 1]).astype(np.complex64)
        if out is None:
            out = np.lib.format.open_memmap(npy, mode="w+", dtype=np.complex64,
                                            shape=(n_tx,) + c.shape)
        out[i] = c
        print(f"  tx {i + 1}/{n_tx} ({time.perf_counter() - t0:.0f}s)", end="\r")
    out.flush()
    print(f"\n[{tag}] per-transmit beamform: {time.perf_counter() - t0:.0f}s -> {out.shape}")

    np.savez(meta_npz, coords=coords, frames=sel, fps=meta.fps.get(BUFFER - 1), **geom)
    return np.load(npy, mmap_mode="r"), np.load(meta_npz, allow_pickle=True)


def build_image(tag, beamformer="delay_and_sum", bf_kwargs=None, tx_window=None,
                rx=None, n_frames=16, overwrite=False):
    """Beamform ALL transmits in one pass, optionally with an adaptive beamformer.

    The coherence-factor beamformers weight and compound across the RECEIVE ELEMENT axis and
    return an already-compounded image, so they cannot be studied from the per-transmit stack
    the way a transmit weight map can - each setting needs its own run. ``tx_window`` (an
    ``(n_tx, nz, nx)`` map) is applied inside the same pipeline via ``AlignedApodization``,
    which is exactly how the production script does it.

    Returns ``(n_frames, nz, nx)`` float32 envelope.
    """
    key = (f"{tag}_b{BUFFER}_img_{beamformer}"
           + ("".join(f"-{k}{v}" for k, v in sorted((bf_kwargs or {}).items())))
           + rx_tag(rx) + ("_txwin" if tx_window is not None else "") + ".npy")
    npy = CACHE / key
    if npy.is_file() and not overwrite:
        return np.load(npy)

    import zea
    from zea import File, init_device
    from zea.ops import Beamform, Cast, Demodulate
    from swp.acquisition.beamform import _ensure_cpu_t_peak, apply_grid, plan_patches_and_chunk
    from swp.acquisition.sequence import read_swi_meta

    init_device(verbose=False)
    root = DATASETS[tag]["root"]
    meta = read_swi_meta(root / "CombinedData.mat")
    conv = root / "output" / "converted" / f"CombinedData_buffer{BUFFER}.hdf5"
    raw_npy = CACHE / f"{tag}_b{BUFFER}_raw{n_frames}.npy"
    with File(str(conv)) as fh:
        n_all = fh.data.raw_data.shape[0]
        sel = (np.arange(n_all) if DATASETS[tag]["frames"] == "all" or n_frames >= n_all
               else np.unique(np.linspace(0, n_all - 1, n_frames).astype(int)))
        raw = (np.load(raw_npy) if raw_npy.is_file()
               else np.asarray(fh.data.raw_data[sel]))
        params = fh.load_parameters()
    if not raw_npy.is_file():
        np.save(raw_npy, raw)
    apply_grid(params, meta.grids[BUFFER - 1])
    _ensure_cpu_t_peak(params)
    coords = np.asarray(params.grid, np.float32)
    n_tx, n_el = raw.shape[1], raw.shape[3]
    rx = rx or {}
    if "f" in rx:
        params.f_number = float(rx["f"])
    kw = dict(bf_kwargs or {})
    flat_rx = None
    if rx.get("apod"):
        flat_rx = receive_apodization(coords, np.asarray(dict(params)["probe_geometry"]),
                                      rx["apod"], mode=rx.get("mode", "fixed"),
                                      f_number=float(rx.get("f", 1.0)))
        kw["enable_receive_apodization"] = True
    if tx_window is not None:
        kw["enable_aligned_apodization"] = True
    num_patches, _ = plan_patches_and_chunk(params, n_tx, n_el)
    pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                         Beamform(beamformer=beamformer, num_patches=num_patches, **kw)],
                        with_batch_dim=True, jit_options=None)
    bf_in = pipe.prepare_parameters(params)
    if flat_rx is not None:
        bf_in["flat_receive_apodization"] = flat_rx
    if tx_window is not None:
        nz, nx = coords.shape[:2]
        bf_in["flat_aligned_apodization"] = np.moveaxis(
            tx_window, 0, -1).reshape(nz * nx, n_tx).astype(np.float32)

    out, t0 = [], time.perf_counter()
    for s in range(0, raw.shape[0], 4):
        iq = pipe(data=raw[s:s + 4], **bf_in)[pipe.output_key]
        iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, np.float32)
        out.append(np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2))
    env = np.concatenate(out).astype(np.float32)
    print(f"[{tag}] {beamformer}{kw} -> {env.shape} in {time.perf_counter() - t0:.0f}s")
    np.save(npy, env)
    return env


def geometry_from_meta(m):
    keys = ("polar_angles", "transmit_origins", "focus_distances",
            "tx_apodizations", "probe_geometry")
    return TxGeometry({k: m[k] for k in keys}, m["coords"])
