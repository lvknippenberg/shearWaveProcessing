"""Make a beamformed shear-wave IQ file self-contained for downstream processing.

Ported from ``SWI/Zea/swi_dsp.py``. After beamforming, the per-measurement active IQ
(buffer 2) and the passive stream (buffer 4) get the scan parameters the visualization
stage needs (``demodulation_frequency`` = velocity scaling, ``transmit_frequency``,
``probe_center_frequency``, ``sound_speed``, ``wavelength``, ``prf``, ``dz``, ``dx``)
written as custom elements - read from the sibling converted RF file + derived from the
grid/timestamps. Existing custom elements (``reference_iq``, ``t_reference``, and the
``push_focus_x/z`` this repo adds) are preserved.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from zea import File
from zea.data.file import CustomElement

# Fallbacks used only when the sibling converted file is missing. For the SWI Widebeam
# sequence (S5-1, 2nd-harmonic imaging) the beamformed IQ is demodulated at the 2nd harmonic.
DEFAULT_FC = 3.90625e6                 # demodulation frequency (Hz)
DEFAULT_SOUND_SPEED = 1540.0
DEFAULT_CENTER_FREQUENCY = 3.125e6     # probe centre frequency Trans.frequency (Hz)


def probe_center_frequency(path, default=DEFAULT_CENTER_FREQUENCY):
    """Probe centre frequency ``Trans.frequency`` [Hz] from the sibling ``CombinedData.mat``.

    Verasonics defines the acoustic wavelength as ``sound_speed / Trans.frequency`` (the probe
    centre frequency) - not the ``center_frequency`` the converter stores (demod/2 = 1.953 MHz).
    Walks up ``path`` to find ``CombinedData.mat``; falls back to :data:`DEFAULT_CENTER_FREQUENCY`.
    """
    import h5py

    p = Path(path)
    for parent in [p, *p.parents]:
        mat = parent / "CombinedData.mat"
        if mat.is_file():
            try:
                with h5py.File(mat, "r") as f:
                    return float(np.asarray(f["Trans"]["frequency"]).reshape(-1)[0]) * 1e6
            except Exception:
                break
    return default


def _converted_path_for(iq_path: Path) -> Path:
    """Sibling converted RF file Stage A wrote, e.g. ``<out>/converted/<stem>.hdf5``.

    The active buffer splits per measurement (``..._buffer2_meas<m>_iq.hdf5``) but shares
    one converted RF (``..._buffer2.hdf5``), so the ``_meas<m>`` tag is stripped for lookup.
    """
    name = iq_path.name.replace("_iq.hdf5", ".hdf5")
    name = re.sub(r"_meas\d+", "", name)
    return iq_path.parent / "converted" / name


def append_scan_params_to_iq(iq_path, converted_path=None):
    """Re-save a beamformed IQ file with the scan parameters added as custom elements.

    Adds ``probe_center_frequency`` / ``transmit_frequency`` / ``demodulation_frequency`` /
    ``sound_speed`` / ``wavelength`` / ``prf`` / ``dz`` / ``dx`` (from the sibling converted RF
    file + derived from the grid/timestamps), preserving existing data and custom elements
    (``reference_iq``, ``t_reference``, ``push_focus_x``, ``push_focus_z``). No re-beamforming.
    """
    iq_path = Path(iq_path)
    with File(str(iq_path)) as f:
        bd = f.data.beamformed_data
        values = np.asarray(bd.values[:], np.float32)
        coords = np.asarray(bd.coordinates[:], np.float32)
        ts = np.asarray(bd.timestamps[:], np.float32) if hasattr(bd, "timestamps") else None
        sto = float(np.asarray(bd.start_time_offset)) if hasattr(bd, "start_time_offset") else 0.0
        customs = list(f.custom)

    conv = Path(converted_path) if converted_path else _converted_path_for(iq_path)
    fc = f_tx = c = None
    if conv.is_file():
        with File(str(conv)) as f:
            p = f.load_parameters()
        fc = float(p.demodulation_frequency)
        c = float(p.sound_speed)
        cf = np.asarray(p.center_frequency).reshape(-1)      # converter's "center_freq" = tx fundamental
        f_tx = float(cf[0]) if cf.size and np.isfinite(cf[0]) else None
    fc = fc or DEFAULT_FC                                    # demodulation (2nd harmonic) - velocity
    c = c or DEFAULT_SOUND_SPEED
    f_tx = f_tx or fc / 2.0                                  # transmit fundamental (harmonic imaging)
    f_probe = probe_center_frequency(iq_path)                # Trans.frequency - drives wavelength
    lam = c / f_probe
    prf = float(1.0 / np.median(np.diff(ts))) if (ts is not None and ts.size > 1) else None
    dz = float(np.median(np.abs(np.diff(coords[:, 0, 2]))))
    dx = float(np.median(np.abs(np.diff(coords[0, :, 0]))))

    added = {"demodulation_frequency", "transmit_frequency", "probe_center_frequency",
             "center_frequency", "sound_speed", "wavelength", "prf", "dz", "dx"}
    keep = [CustomElement(name=cu.name, data=np.asarray(cu.data),
                          description=getattr(cu, "description", ""), unit=getattr(cu, "unit", ""))
            for cu in customs if cu.name not in added]
    new = [
        CustomElement(name="probe_center_frequency", data=np.float32(f_probe),
                      description="probe centre frequency (Trans.frequency) - defines wavelength",
                      unit="Hz"),
        CustomElement(name="transmit_frequency", data=np.float32(f_tx),
                      description="transmit fundamental frequency (2nd-harmonic imaging: demod/2)",
                      unit="Hz"),
        CustomElement(name="demodulation_frequency", data=np.float32(fc),
                      description="IQ demodulation frequency (2nd harmonic) - velocity scaling",
                      unit="Hz"),
        CustomElement(name="sound_speed", data=np.float32(c), description="speed of sound", unit="m/s"),
        CustomElement(name="wavelength", data=np.float32(lam),
                      description="wavelength (sound_speed / probe_center_frequency)", unit="m"),
        CustomElement(name="dz", data=np.float32(dz), description="axial pixel spacing", unit="m"),
        CustomElement(name="dx", data=np.float32(dx), description="lateral pixel spacing", unit="m"),
    ]
    if prf:
        new.append(CustomElement(name="prf", data=np.float32(prf),
                                 description="tracking PRF (slow-time frame rate)", unit="Hz"))

    bdata = {"values": values, "coordinates": coords, "labels": np.array(["I", "Q"], dtype=np.str_)}
    if ts is not None:
        bdata["timestamps"] = ts
        bdata["start_time_offset"] = np.float32(sto)
    tmp = iq_path.with_suffix(".tmp.hdf5")
    if tmp.exists():
        tmp.unlink()
    File.create(str(tmp), data={"beamformed_data": bdata}, custom=keep + new,
                description="beamformed shear-wave IQ + scan parameters",
                compression="lzf", overwrite=True, ignore_warnings=True)
    iq_path.unlink()
    tmp.rename(iq_path)
    return iq_path
