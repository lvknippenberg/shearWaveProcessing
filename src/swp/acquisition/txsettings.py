"""Extract the push + tracking transmit settings needed to *simulate the transmit field*.

Reads the Verasonics ``CombinedData.mat`` (Trans / TX / TW / SW / Resource structs) and writes a
single self-documented HDF5 (``output/transmit_settings.hdf5``) with everything a transmit-field
simulator (Field II / k-Wave / an in-house solver) needs: probe geometry, medium, and the
**per-element apodization + delays**, focal point, aperture, frequency and waveform for both the
ARF **push** and the **tracking** (detect) transmits.

The push transmit is identified as the TX whose focus matches ``SW.FocusX/Z`` (long focused ARF);
the tracking transmits are the TX indices actually used by the buffer-2 receive events.
"""

from pathlib import Path

import h5py
import numpy as np


def _ref_vec(f, group, field, i):
    """Dereference entry ``i`` of a MATLAB cell/struct array field -> flat numpy vector."""
    refs = np.asarray(f[group][field]).reshape(-1)
    return np.asarray(f[refs[i]]).reshape(-1)


def _scalar(f, path):
    return float(np.asarray(f[path]).reshape(-1)[0])


def _matstr(f, path):
    try:
        a = np.asarray(f[path]).reshape(-1).astype(np.uint8)
        return bytes(a).decode(errors="ignore")
    except Exception:
        return ""


def _tracking_tx_indices(mat_path):
    """TX indices used by the buffer-2 (active) receive events = the detect/tracking beams."""
    from zea.data.convert.verasonics import VerasonicsFile

    with VerasonicsFile(str(mat_path)) as vf:
        tx_order, _, _ = vf.read_transmit_events("all", True, 1)
    return sorted({int(i) for i in np.asarray(tx_order).reshape(-1)})


def save_transmit_settings(mat_path, out_path=None):
    """Extract push+tracking transmit settings from ``mat_path`` -> HDF5 ``out_path``.

    Returns the output path. All lengths are in metres, times in seconds, frequencies in Hz.
    """
    mat_path = Path(mat_path)
    if out_path is None:
        out_path = mat_path.parent / "output" / "transmit_settings.hdf5"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    track_idx = _tracking_tx_indices(mat_path)

    with h5py.File(mat_path, "r") as f:
        # ---- medium / probe ----------------------------------------------------------
        c = _scalar(f, "Resource/Parameters/speedOfSound")
        trans_freq = _scalar(f, "Trans/frequency") * 1e6          # MHz -> Hz
        lam = c / trans_freq
        n_el = int(_scalar(f, "Trans/numelements"))
        elpos_mm = np.asarray(f["Trans/ElementPos"])              # (4, n_el): x,y,z,az (mm, deg)
        if elpos_mm.shape[0] != 4:
            elpos_mm = elpos_mm.T
        elpos_m = elpos_mm.copy()
        elpos_m[:3] *= 1e-3                                       # x,y,z mm -> m (col 4 = angle)
        elem_width = _scalar(f, "Trans/elementWidth") * lam       # Verasonics: elementWidth in wl
        try:
            pitch_m = _scalar(f, "Trans/spacingMm") * 1e-3
        except Exception:
            pitch_m = _scalar(f, "Trans/spacing") * lam
        bandwidth = np.asarray(f["Trans/Bandwidth"]).reshape(-1) * 1e6
        try:
            lens_corr = _scalar(f, "Trans/lensCorrection") * lam
        except Exception:
            lens_corr = 0.0
        probe_name = _matstr(f, "Trans/name")

        # ---- SW push / sequence parameters -------------------------------------------
        def sw(name, default=np.nan):
            try:
                return _scalar(f, f"SW/{name}")
            except Exception:
                return default

        demod_freq = sw("demodFrequency") * 1e6 if np.isfinite(sw("demodFrequency")) else np.nan
        push_fx = sw("FocusX_wavelength") * lam
        push_fz = sw("FocusZ_wavelength") * lam
        n_push_el = int(sw("nb_push_elmts"))
        push_freq = sw("PushFrequency") * 1e6
        push_cycles = sw("pushCycle")
        f_number = float(push_fz / (n_push_el * pitch_m)) if n_push_el * pitch_m else np.nan

        # ---- identify the push TX (focus matches SW focus) ---------------------------
        n_tx = np.asarray(f["TX/focus"]).reshape(-1).size
        foc = np.array([_ref_vec(f, "TX", "focus", i)[0] for i in range(n_tx)])
        focx = np.array([_ref_vec(f, "TX", "focusX", i)[0] for i in range(n_tx)])
        push_tx = int(np.argmin(np.abs(foc - sw("FocusZ_wavelength")) + np.abs(focx - sw("FocusX_wavelength"))))

        def tx_record(i):
            apod = _ref_vec(f, "TX", "Apod", i).astype(np.float32)
            delay_wl = _ref_vec(f, "TX", "Delay", i).astype(np.float64)
            rec = {
                "apodization": apod,                              # per-element weight [0..1]
                "delays_s": (delay_wl / trans_freq).astype(np.float32),
                "delays_wavelengths": delay_wl.astype(np.float32),
                "focus_wl": float(foc[i]),
                "focus_x_wl": float(focx[i]),
                "focus_z_m": float(foc[i] * lam),
                "focus_x_m": float(focx[i] * lam),
                "waveform_index": int(_ref_vec(f, "TX", "waveform", i)[0]),
                "tx_index": i,
            }
            for k in ("Origin", "Steer"):
                try:
                    rec[k.lower()] = _ref_vec(f, "TX", k, i).astype(np.float32)
                except Exception:
                    pass
            return rec

        push = tx_record(push_tx)
        tracking = {i: tx_record(i) for i in track_idx}

        # ---- transmit waveforms (TW) -------------------------------------------------
        n_tw = np.asarray(f["TW/envFrequency"]).reshape(-1).size
        tw = {}
        for k in ("envFrequency", "envNumCycles", "envPulseWidth", "Numpulses", "peak",
                  "estimatedAvgFreq"):
            try:
                tw[k] = np.array([_ref_vec(f, "TW", k, i)[0] for i in range(n_tw)], np.float64)
            except Exception:
                pass
        tw["envFrequency"] = tw.get("envFrequency", np.zeros(n_tw)) * 1e6  # -> Hz
        # tri-level waveform samples (per-waveform, variable length)
        tri = {}
        for i in range(n_tw):
            try:
                tri[i] = _ref_vec(f, "TW", "TriLvlWvfm_Sim", i).astype(np.float32)
            except Exception:
                pass

    # ------------------------------------------------------------------ write ---------
    with h5py.File(out_path, "w") as o:
        o.attrs["description"] = ("Push + tracking transmit settings to simulate the transmit "
                                  "field. Lengths [m], times [s], frequencies [Hz], angles [rad] "
                                  "unless noted. Source: " + mat_path.name)
        o.attrs["source_mat"] = mat_path.name

        g = o.create_group("medium")
        g.create_dataset("sound_speed", data=np.float64(c)); g["sound_speed"].attrs["unit"] = "m/s"
        g.create_dataset("wavelength", data=np.float64(lam)); g["wavelength"].attrs["unit"] = "m"
        g["wavelength"].attrs["desc"] = "sound_speed / probe_center_frequency (Verasonics convention)"
        g.create_dataset("probe_center_frequency", data=np.float64(trans_freq))
        g["probe_center_frequency"].attrs["unit"] = "Hz"
        g["probe_center_frequency"].attrs["desc"] = "Trans.frequency - defines the wavelength"
        # transmit fundamental (2nd-harmonic imaging demodulates at 2x the transmit)
        g.create_dataset("transmit_frequency", data=np.float64(demod_freq / 2.0))
        g["transmit_frequency"].attrs["unit"] = "Hz"
        g["transmit_frequency"].attrs["desc"] = "imaging transmit fundamental (demod/2)"
        g.create_dataset("demod_frequency", data=np.float64(demod_freq))
        g["demod_frequency"].attrs["unit"] = "Hz"
        g["demod_frequency"].attrs["desc"] = "2nd-harmonic receive demodulation - velocity scaling"

        g = o.create_group("probe")
        g.attrs["name"] = probe_name
        d = g.create_dataset("element_positions", data=elpos_m.astype(np.float64))
        d.attrs["layout"] = "rows = [x, y, z, azimuth]; cols = elements"
        d.attrs["unit"] = "x,y,z in m; azimuth in deg"
        g.create_dataset("pitch", data=np.float64(pitch_m)); g["pitch"].attrs["unit"] = "m"
        g.create_dataset("element_width", data=np.float64(elem_width))
        g["element_width"].attrs["unit"] = "m"
        g.create_dataset("bandwidth", data=bandwidth.astype(np.float64))
        g["bandwidth"].attrs["unit"] = "Hz"
        g.create_dataset("lens_correction", data=np.float64(lens_corr))
        g.create_dataset("n_elements", data=np.int64(n_el))

        g = o.create_group("push")
        g.attrs["description"] = ("ARF push transmit (TX #%d): long focused push that displaces "
                                  "tissue axially." % push_tx)
        g.create_dataset("focus_x", data=np.float64(push_fx)); g["focus_x"].attrs["unit"] = "m"
        g.create_dataset("focus_z", data=np.float64(push_fz)); g["focus_z"].attrs["unit"] = "m"
        g.create_dataset("frequency", data=np.float64(push_freq))
        g["frequency"].attrs["unit"] = "Hz"
        g.create_dataset("n_cycles", data=np.float64(push_cycles))
        g.create_dataset("n_active_elements", data=np.int64(n_push_el))
        g.create_dataset("f_number", data=np.float64(f_number))
        for k, v in push.items():
            g.create_dataset(k, data=v)
        g["apodization"].attrs["desc"] = "per-element transmit weight [0..1]"
        g["delays_s"].attrs["unit"] = "s"

        gt = o.create_group("tracking")
        gt.attrs["description"] = ("Detect/tracking transmits used by the buffer-2 events "
                                   "(TX indices %s); the diverging beams that image the shear "
                                   "wave." % track_idx)
        gt.attrs["tx_indices"] = np.array(track_idx, np.int64)
        for i, rec in tracking.items():
            sub = gt.create_group(f"tx{i}")
            for k, v in rec.items():
                sub.create_dataset(k, data=v)
            sub["delays_s"].attrs["unit"] = "s"
            sub["apodization"].attrs["desc"] = "per-element transmit weight [0..1]"

        g = o.create_group("waveforms")
        g.attrs["description"] = ("Transmit waveforms (TW struct). waveform_index in push/tracking "
                                  "is 1-based into these.")
        for k, v in tw.items():
            g.create_dataset(k, data=v)
        for i, samp in tri.items():
            g.create_dataset(f"trilevel_tw{i + 1}", data=samp)

    return out_path


if __name__ == "__main__":
    import sys

    for folder in sys.argv[1:]:
        mat = Path(folder) / "CombinedData.mat"
        p = save_transmit_settings(mat)
        print("wrote", p)
