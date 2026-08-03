"""SWI Widebeam sequence: static buffer description, .mat metadata, and frame assembly.

The single place that knows *what each RF buffer of the SWI Widebeam sequence contains* and
how to decode the application-level metadata the generic Verasonics->zea converter does not
carry. Three cohesive parts (formerly ``swi_config`` / ``swi_meta`` / ``swi_frames``):

* **Buffer description** - ``BufferSpec`` + ``SEQUENCE`` / ``SPEC_BY_INDEX`` (mirrors
  ``SetUp_SWI_Widebeam.m``): what each of the 6 buffers is and how it is processed.
* **Workspace metadata** - ``read_swi_meta`` reads per-buffer ``ActualFPS``, the ``PData``
  beamforming grids, the shear-wave (buffer 2) geometry and ECG straight from the ``.mat``
  (v7.3 / HDF5) with ``h5py``; ``sw_roi_grid`` builds the buffer-2 reconstruction window.
* **Frame assembly** - ``assemble_tracking_frames`` reassembles the active-tracking buffer
  (buffer 2), whose transmits are a TIME SEQUENCE, into pulse-inversion-recombined frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# ============================== buffer description ==============================

from dataclasses import dataclass


@dataclass(frozen=True)
class BufferSpec:
    """How one RF buffer of this sequence is processed."""

    matlab: int                 # 1-based RcvData{k}
    name: str
    kind: str                   # "bmode" | "active_track"
    struct: str                 # MATLAB struct describing it (for swi_meta / FPS)
    role: str = ""              # human-readable note
    # Active-tracking only: default pulse-inversion recombination.
    #   "sliding"    -> overlapping +/- pairs, ~2x PRF (default; best temporal sampling)
    #   "accumulate" -> sum consecutive +/- pairs, ~1x PRF (matches hardware buffers)
    pi_mode: str = "sliding"
    # Marks the ultrafast buffer whose B-mode frame stream doubles as the passive
    # shear-wave tracking sequence.
    passive_source: bool = False

    @property
    def index(self) -> int:
        """0-based (Python) buffer index."""
        return self.matlab - 1


SEQUENCE: list[BufferSpec] = [
    BufferSpec(1, "bmode_widebeam", "bmode", "Bmode_WB",
               role="widebeam B-mode (orientation), ~88 FPS"),
    BufferSpec(2, "active_sw", "active_track", "SW",
               role="active shear-wave: reference + ARF push + tracking",
               pi_mode="sliding"),
    BufferSpec(3, "bmode_focused", "bmode", "Bmode_FC",
               role="focused B-mode, ~25 FPS"),
    BufferSpec(4, "passive_sw", "bmode", "Bmode_DW",
               role="ultrafast diverging-wave B-mode, ~925 FPS (passive elastography source)",
               passive_source=True),
    BufferSpec(5, "bmode_during_sw", "bmode", "Bmode_WB",
               role="widebeam B-mode, one frame per SW measurement"),
    BufferSpec(6, "bmode_strain", "bmode", "Bmode_strain",
               role="widebeam B-mode, long recording (~3 beats) for strain"),
]

SPEC_BY_INDEX = {s.index: s for s in SEQUENCE}
# ============================== workspace metadata ==============================

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# 0-based buffer index -> name of the MATLAB struct describing that buffer.
BUFFER_STRUCT = {
    0: "Bmode_WB",
    1: "SW",
    2: "Bmode_FC",
    3: "Bmode_DW",
    4: "Bmode_WB",
    5: "Bmode_strain",
}


@dataclass
class BufferGrid:
    """A PData beamforming grid, in metres."""

    xlims: tuple[float, float]
    zlims: tuple[float, float]
    n_x: int
    n_z: int
    pixel_size_m: float


@dataclass
class SWGeometry:
    """Active shear-wave (buffer 2) geometry and timing, in metres / seconds."""

    n_reference: int          # detect positions before the push (Ndetect_pre)
    n_tracking: int           # detect positions after the push (Ndetect)
    na: int                   # angles per detect position
    harmonic: bool            # pulse-inversion (pos/neg stored separately)
    pri: float                # detect PRI, seconds
    focus_x: float            # push focal point, metres
    focus_z: float
    roi_xlims: tuple[float, float]
    roi_zlims: tuple[float, float]

    def prf(self, pi_mode: str = "sliding") -> float:
        """Effective tracking frame rate for a pulse-inversion recombination mode.

        Each detect position costs ``na * (harmonic + 1)`` transmits of one PRI.
        ``accumulate`` sums the +/- polarity pair into one frame; ``sliding``
        reuses each polarity in two overlapping pairs, doubling the frame rate.
        """
        base = 1.0 / (self.na * (2 if self.harmonic else 1) * self.pri)
        if pi_mode == "sliding" and self.harmonic:
            return 2.0 * base
        return base


@dataclass
class VerasonicsMeta:
    """Sequence metadata read from the raw ``.mat`` workspace."""

    wavelength_m: float
    sound_speed: float
    fps: dict[int, float] = field(default_factory=dict)      # buffer idx -> ActualFPS
    grids: dict[int, BufferGrid] = field(default_factory=dict)
    sw: SWGeometry | None = None
    ecg: dict | None = None


# ---------------------------------------------------------------------------
def _scalar(group, name, default=None):
    """Read a scalar struct field, or ``default`` if absent."""
    if name not in group:
        return default
    return float(np.asarray(group[name]).reshape(-1)[0])


def _deref_pdata_field(f, field_name, index):
    """Dereference ``PData(index+1).<field>`` into a 1-D float array.

    MATLAB stores the PData struct array as a column of HDF5 object references
    (one per buffer), exactly like ``RcvData``; each reference points at the
    numeric vector for that buffer.
    """
    dataset = f["PData"][field_name]
    refs = np.asarray(dataset).reshape(-1)
    return np.asarray(f[refs[index]]).reshape(-1)


def _pdata_grid(f, index, wavelength_m):
    """Build a :class:`BufferGrid` (metres) from ``PData(index+1)``.

    Verasonics convention: ``Origin = [x, y, z]``, ``Size = [nz, nx, ny]``,
    ``PDelta = [dx, dy, dz]`` - all in wavelengths. z runs from the origin
    downward; x is centred on the origin.
    """
    origin = _deref_pdata_field(f, "Origin", index)
    size = _deref_pdata_field(f, "Size", index)
    pdelta = _deref_pdata_field(f, "PDelta", index)

    n_z, n_x = int(size[0]), int(size[1])
    dx, dz = float(pdelta[0]), float(pdelta[2])
    x0, z0 = float(origin[0]), float(origin[2])

    xlims = ((x0) * wavelength_m, (x0 + n_x * dx) * wavelength_m)
    zlims = ((z0) * wavelength_m, (z0 + n_z * dz) * wavelength_m)
    return BufferGrid(
        xlims=xlims, zlims=zlims, n_x=n_x, n_z=n_z,
        pixel_size_m=dz * wavelength_m,
    )


def _sw_geometry(f, wavelength_m):
    """Read the active shear-wave geometry from the ``SW`` struct."""
    if "SW" not in f:
        return None
    sw = f["SW"]
    focus_x = _scalar(sw, "FocusX_wavelength", 0.0) * wavelength_m
    focus_z = _scalar(sw, "FocusZ_wavelength", 0.0) * wavelength_m
    roi_w = _scalar(sw, "ROI_width", 0.0) * wavelength_m
    roi_h = _scalar(sw, "ROI_height", 0.0) * wavelength_m
    return SWGeometry(
        n_reference=int(_scalar(sw, "Ndetect_pre", 0)),
        n_tracking=int(_scalar(sw, "Ndetect", 0)),
        na=int(_scalar(sw, "na", 1)),
        harmonic=bool(_scalar(sw, "HarmonicImaging", 1)),
        pri=_scalar(sw, "PRI_us", 270.0) * 1e-6,
        focus_x=focus_x,
        focus_z=focus_z,
        roi_xlims=(focus_x - roi_w / 2, focus_x + roi_w / 2),
        roi_zlims=(focus_z - roi_h / 2, focus_z + roi_h / 2),
    )


def sw_roi_grid(sw: SWGeometry, base_grid: BufferGrid, sw_roi=None) -> BufferGrid:
    """Beamforming grid for the active shear-wave buffer (buffer 2).

    The default ROI comes from the ``SW`` struct (``ROI_width`` x ``ROI_height``
    centred on the push focus). That window is *physically* motivated: the shear
    wave is induced by the focused extended push transmit and attenuates quickly
    with lateral distance from it, so pixels far outside the focus carry little
    trackable wave - even though the weakly diverging tracking beam does insonify
    a much wider sector.

    ``sw_roi`` lets that assumption be evaluated (see Stage B):

    * ``None`` (default) - the ``SW`` struct ROI, unchanged.
    * a number, e.g. ``2.0`` - scale the ROI width and height by this factor about
      the push focus (``1.0`` reproduces the default).
    * ``"full"`` - use the full imaging FOV (``base_grid``, i.e. the B-mode grid).
    * ``((xmin, xmax), (zmin, zmax))`` - an explicit window in **metres**.

    The result is always clipped to ``base_grid`` and sampled at its pixel pitch.
    """
    pix = base_grid.pixel_size_m

    if isinstance(sw_roi, str) and sw_roi.lower() == "full":
        xlims, zlims = base_grid.xlims, base_grid.zlims
    elif isinstance(sw_roi, (int, float)) and not isinstance(sw_roi, bool):
        half_w = (sw.roi_xlims[1] - sw.roi_xlims[0]) / 2 * float(sw_roi)
        half_h = (sw.roi_zlims[1] - sw.roi_zlims[0]) / 2 * float(sw_roi)
        xlims = (sw.focus_x - half_w, sw.focus_x + half_w)
        zlims = (sw.focus_z - half_h, sw.focus_z + half_h)
    elif sw_roi is not None:
        xlims, zlims = sw_roi
    else:
        xlims, zlims = sw.roi_xlims, sw.roi_zlims

    # Never reconstruct outside the imaged sector.
    xlims = (max(xlims[0], base_grid.xlims[0]), min(xlims[1], base_grid.xlims[1]))
    zlims = (max(zlims[0], base_grid.zlims[0]), min(zlims[1], base_grid.zlims[1]))

    n_x = max(1, round((xlims[1] - xlims[0]) / pix))
    n_z = max(1, round((zlims[1] - zlims[0]) / pix))
    return BufferGrid(xlims=xlims, zlims=zlims, n_x=n_x, n_z=n_z, pixel_size_m=pix)


def _read_ecg(f):
    """Read ECG arrays if the acquisition recorded them, else ``None``."""
    if not bool(_scalar(f, "ECG_connected", 0)):
        return None
    ecg = {}
    for key in ("ECG_data_raw", "ECG_data_lines", "ECG_data_values"):
        if key in f:
            ecg[key] = np.asarray(f[key]).squeeze()
    return ecg or None


def read_swi_meta(mat_path, sw_roi=None) -> VerasonicsMeta:
    """Read SWI sequence metadata from a Verasonics ``.mat`` workspace.

    Args:
        mat_path: Path to the ``.mat`` (v7.3 / HDF5) workspace.
        sw_roi: Optional override for the active shear-wave (buffer 2)
            reconstruction window - see :func:`sw_roi_grid`. ``None`` (default)
            uses the ``SW`` struct ROI around the push focus.

    Returns:
        A :class:`VerasonicsMeta`. Missing structs are skipped rather than
        raising, so a reduced workspace still yields whatever is present.
    """
    mat_path = Path(mat_path)
    with h5py.File(str(mat_path), "r") as f:
        sound_speed = _scalar(f["Resource"]["Parameters"], "speedOfSound", 1540.0)
        f_c = _scalar(f["Trans"], "frequency", None)
        wavelength_m = sound_speed / (f_c * 1e6) if f_c else _scalar(f, "lambda_mm", 0.5) * 1e-3

        fps, grids = {}, {}
        n_pdata = int(np.asarray(f["PData"]["Size"]).reshape(-1).size) if "PData" in f else 0

        for idx, struct_name in BUFFER_STRUCT.items():
            if struct_name in f and "ActualFPS" in f[struct_name]:
                fps[idx] = _scalar(f[struct_name], "ActualFPS")
            if "PData" in f and idx < n_pdata:
                grids[idx] = _pdata_grid(f, idx, wavelength_m)

        sw = _sw_geometry(f, wavelength_m)
        # Buffer 2 is reconstructed on the shear-wave ROI around the push focus,
        # not the full sector PData(2) may span: the wave is induced by the focused
        # push and attenuates quickly away from it, so this is where trackable wave
        # actually is (and it keeps the per-tracking-frame cost small).
        # ``sw_roi`` can widen/replace it for evaluation - see :func:`sw_roi_grid`.
        if sw is not None and 1 in grids and 0 in grids:
            grids[1] = sw_roi_grid(sw, grids[0], sw_roi)

        ecg = _read_ecg(f)

    return VerasonicsMeta(
        wavelength_m=wavelength_m, sound_speed=sound_speed,
        fps=fps, grids=grids, sw=sw, ecg=ecg,
    )
# =============================== frame assembly =================================

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackingFrames:
    """Recombined reference and post-push tracking frames for one buffer.

    Arrays are ``(n_meas, n_frames, n_ax, n_el, 1)``. Times are in seconds with
    ``t = 0`` at the first post-push tracking frame; reference times are negative.
    """

    reference: np.ndarray
    tracking: np.ndarray
    prf: float
    pi_mode: str
    t_reference: np.ndarray
    t_tracking: np.ndarray

    @property
    def n_meas(self) -> int:
        return self.tracking.shape[0]


def _recombine(block: np.ndarray, na: int, harmonic: bool, pi_mode: str) -> np.ndarray:
    """Pulse-inversion recombine one ``(n_meas, n_tx, n_ax, n_el, 1)`` block.

    Returns ``(n_meas, n_frames, n_ax, n_el, 1)`` float32 harmonic frames.
    """
    block = block.astype(np.float32)          # int16 sums overflow; promote first

    if not harmonic:
        # No pulse inversion: each position is na angles to compound.
        if na == 1:
            return block
        n_meas, n_tx = block.shape[0], block.shape[1]
        block = block.reshape((n_meas, n_tx // na, na) + block.shape[2:])
        return block.mean(axis=2)

    if na == 1:
        # Consecutive transmits alternate +/-, so every adjacent pair is a valid
        # harmonic sum. Non-overlapping pairs = accumulate; overlapping = sliding.
        if pi_mode == "sliding":
            return block[:, :-1] + block[:, 1:]
        return block[:, 0::2] + block[:, 1::2]

    # na > 1 harmonic: each position stores na (+) angles then na (-) angles.
    # Harmonic-compound within the position (sum polarity per angle, average
    # angles). Sliding across positions is not well defined here, so accumulate.
    if pi_mode == "sliding":
        raise NotImplementedError(
            "sliding pulse-inversion is only implemented for na == 1; "
            "use pi_mode='accumulate' for na > 1."
        )
    n_meas, n_tx = block.shape[0], block.shape[1]
    per_pos = 2 * na
    block = block.reshape((n_meas, n_tx // per_pos, per_pos) + block.shape[2:])
    pos, neg = block[:, :, :na], block[:, :, na:]
    return (pos + neg).mean(axis=2)


def assemble_tracking_frames(
    raw: np.ndarray,
    n_reference: int,
    n_tracking: int,
    na: int = 1,
    harmonic: bool = True,
    pri: float = 270e-6,
    pi_mode: str = "sliding",
) -> TrackingFrames:
    """Split a raw SW buffer into recombined reference and tracking frames.

    Args:
        raw: ``(n_meas, n_tx, n_ax, n_el, 1)`` converted SW buffer, where a
            "frame" (n_meas axis) is one shear-wave measurement.
        n_reference: detect positions before the push (``SW.Ndetect_pre``).
        n_tracking: detect positions after the push (``SW.Ndetect``).
        na: angles per detect position (``SW.na``).
        harmonic: pulse-inversion harmonic imaging (polarities stored separately).
        pri: detect PRI in seconds.
        pi_mode: ``"sliding"`` (default, ~2x PRF) or ``"accumulate"``.

    Returns:
        A :class:`TrackingFrames`.
    """
    per_pos = na * (2 if harmonic else 1)
    n_ref_tx = n_reference * per_pos
    n_trk_tx = n_tracking * per_pos
    expected = n_ref_tx + n_trk_tx
    n_tx = raw.shape[1]
    if n_tx != expected:
        raise ValueError(
            f"SW buffer has {n_tx} transmits but the sequence description implies "
            f"{expected} ({n_reference}+{n_tracking} positions x {per_pos} transmits). "
            "Check SW.Ndetect_pre / SW.Ndetect / SW.na / SW.HarmonicImaging."
        )

    ref_block = raw[:, :n_ref_tx]
    trk_block = raw[:, n_ref_tx:]

    reference = _recombine(ref_block, na, harmonic, pi_mode)
    tracking = _recombine(trk_block, na, harmonic, pi_mode)

    # Effective frame rate: sliding halves the sample spacing (na==1, harmonic).
    base_prf = 1.0 / (per_pos * pri)
    prf = 2.0 * base_prf if (pi_mode == "sliding" and harmonic and na == 1) else base_prf

    n_ref, n_trk = reference.shape[1], tracking.shape[1]
    # t = 0 at the first tracking frame; reference frames are before the push.
    t_tracking = np.arange(n_trk, dtype=np.float32) / prf
    t_reference = (np.arange(n_ref, dtype=np.float32) - n_ref) / prf

    return TrackingFrames(
        reference=reference, tracking=tracking, prf=prf, pi_mode=pi_mode,
        t_reference=t_reference, t_tracking=t_tracking,
    )
