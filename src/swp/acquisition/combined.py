"""Generate ``CombinedData.mat`` from the runtime dynamic parameters.

The Verasonics acquisition saves only the *dynamic* parameters at runtime
(``AcquisitionParametersAndECG.mat``); the *constant* parameters live in a base
config runtime ``.mat``. The beamformer needs the merged workspace
(``CombinedData.mat``, MATLAB v7.3). This module produces it.

For now the merge is delegated to MATLAB (``matlab -batch``) running the ported
``matlab/make_combined_data.m`` -- the base config is a v7.3 workspace with deep
Verasonics struct arrays, and letting MATLAB write it guarantees a file the
``zea`` reader (an h5py-based v7.3 reader) accepts. A pure-Python v7.3 writer is
planned to remove the MATLAB dependency later.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# Default location of the base config runtime .mat files and NonzeroRFcolumns.mat
# (overridable with the SWP_BASE_CONFIG_DIR environment variable).
DEFAULT_BASE_CONFIG_DIR = r"D:\Luuk van Knippenberg\SWI\Base config files"

_MATLAB_DIR = Path(__file__).resolve().parent / "matlab"


def base_config_dir() -> str:
    return os.environ.get("SWP_BASE_CONFIG_DIR", DEFAULT_BASE_CONFIG_DIR)


def _find_matlab() -> str:
    """Locate a MATLAB executable (``SWP_MATLAB`` env var, PATH, then common installs)."""
    override = os.environ.get("SWP_MATLAB")
    if override:
        return override
    exe = shutil.which("matlab")
    if exe:
        return exe
    for root in (r"C:\Program Files\MATLAB", r"C:\Program Files (x86)\MATLAB"):
        p = Path(root)
        if p.is_dir():
            for ver in sorted(p.iterdir(), reverse=True):   # newest release first
                cand = ver / "bin" / "matlab.exe"
                if cand.is_file():
                    return str(cand)
    raise FileNotFoundError(
        "MATLAB executable not found. Set SWP_MATLAB to matlab.exe, or add it to PATH."
    )


def ensure_combined_data(folder, base_config_dir_=None, overwrite=False) -> Path:
    """Ensure ``<folder>/CombinedData.mat`` exists, building it via MATLAB if needed.

    If ``CombinedData.mat`` is already present it is returned unchanged (unless
    ``overwrite``). Otherwise ``AcquisitionParametersAndECG.mat`` must exist in the
    folder and is merged with the base config into ``CombinedData.mat``.

    Returns the path to ``CombinedData.mat``.
    """
    folder = Path(folder)
    combined = folder / "CombinedData.mat"
    if combined.is_file() and not overwrite:
        repair_buffer2_receive(folder)
        return combined

    dynamic = folder / "AcquisitionParametersAndECG.mat"
    if not dynamic.is_file():
        raise FileNotFoundError(
            f"cannot build CombinedData.mat: {dynamic.name} not found in {folder}"
        )
    base_dir = base_config_dir_ or base_config_dir()
    if not Path(base_dir).is_dir():
        raise FileNotFoundError(
            f"base config directory not found: {base_dir!r} "
            "(set SWP_BASE_CONFIG_DIR)"
        )

    matlab = _find_matlab()
    # -batch runs non-interactively and returns a non-zero exit code on error.
    cmd = [
        matlab, "-batch",
        f"make_combined_data('{folder}','{base_dir}')",
    ]
    print(f"  [CombinedData] building via MATLAB -> {combined.name}")
    print(f"    {matlab} -batch make_combined_data('{folder}', '{base_dir}')")
    proc = subprocess.run(
        cmd, cwd=str(_MATLAB_DIR), capture_output=True, text=True,
    )
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(
            f"MATLAB failed to build CombinedData.mat (exit {proc.returncode}):\n"
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    if not combined.is_file():
        raise RuntimeError(
            "MATLAB reported success but CombinedData.mat was not created "
            f"in {folder}"
        )
    repair_buffer2_receive(folder)
    print(f"  [CombinedData] done -> {combined}")
    return combined


# ---------------------------------------------------------------------------------------------
# Buffer-2 receive-layout repair
# ---------------------------------------------------------------------------------------------
# The base config supplies the constant `Receive` struct, i.e. the map from raw RF samples to
# acquisitions. The buffer-2 (shear-wave tracking) receive length follows `SW.endDepth`, which is
# a *runtime* setting -- so a base config recorded at one endDepth silently mis-slices data
# acquired at another. This bit the 2026-08-17/18 campaigns: the `PhantomSweep/BaseConfig_10frames_*`
# files were recorded at endDepth 200 wl (2688 samples) while the data was acquired at 300 wl
# (3968 samples), and every tracking frame was read from a progressively shifted region of the
# buffer. The B-modes looked normal (their buffers are fixed full-depth) while the tracking
# speckle correlation collapsed from 0.98 to 0.40 and no shear wave survived.
#
# `repair_buffer2_receive` recomputes the layout from the runtime parameters the way Verasonics
# does and rewrites it, and is called by `ensure_combined_data` on every build.

def buffer2_receive_samples(max_acq_length, samples_per_wave):
    """Verasonics buffer-2 receive length: 2 * maxAcqLength * samplesPerWave, rounded up to 128."""
    import math
    return int(math.ceil(2.0 * float(max_acq_length) * float(samples_per_wave) / 128.0) * 128)


def repair_buffer2_receive(folder, verbose=True):
    """Make the buffer-2 ``Receive`` layout in ``CombinedData.mat`` match the acquisition.

    Returns ``(status, have, want)`` with status ``"ok"`` (already correct) or ``"fixed"``.
    Raises if the corrected layout would not fit in the buffer's ``rowsPerFrame``.
    """
    import h5py
    import numpy as np
    import scipy.io as sio

    folder = Path(folder)
    combined = folder / "CombinedData.mat"
    dynamic = folder / "AcquisitionParametersAndECG.mat"
    if not dynamic.is_file():
        # Rebuilt-from-workspace folders carry their own consistent Receive; nothing to check.
        return "skipped", None, None
    max_acq = float(sio.loadmat(dynamic, squeeze_me=True,
                                struct_as_record=False)["SW"].maxAcqLength)

    def _refs(group, field):
        return np.atleast_1d(np.array(group[field]).squeeze())

    with h5py.File(combined, "r+") as f:
        R = f["Receive"]
        vals = lambda k: np.array([float(np.array(f[r]).squeeze()) for r in _refs(R, k)])  # noqa: E731
        bufnum, acqnum = vals("bufnum"), vals("acqNum")
        start, end, spw = vals("startSample"), vals("endSample"), vals("samplesPerWave")
        idx = np.where(bufnum == 2)[0]
        if idx.size == 0:
            return "skipped", None, None
        want = buffer2_receive_samples(max_acq, float(np.unique(spw[idx])[0]))
        have = int(np.unique(end[idx] - start[idx] + 1)[0])
        if have == want:
            return "ok", have, want

        rpf = [float(np.array(f[r]).squeeze())
               for r in _refs(f["Resource"]["RcvBuffer"], "rowsPerFrame")]
        n_per_frame = int(acqnum[idx].max())
        if n_per_frame * want > rpf[1]:
            raise RuntimeError(
                f"{folder}: corrected buffer-2 layout ({n_per_frame} x {want} samples) exceeds "
                f"rowsPerFrame {rpf[1]:.0f}"
            )
        r_ss, r_es, r_ed = _refs(R, "startSample"), _refs(R, "endSample"), _refs(R, "endDepth")
        end_depth = want / (2.0 * float(np.unique(spw[idx])[0]))
        for i in idx:
            s = 1.0 + (acqnum[i] - 1.0) * want
            for ref, value in ((r_ss[i], s), (r_es[i], s + want - 1), (r_ed[i], end_depth)):
                f[ref][...] = np.array(value, dtype=f[ref].dtype).reshape(f[ref].shape)
    if verbose:
        print(f"  [CombinedData] buffer-2 receive layout repaired: {have} -> {want} samples "
              f"(base config endDepth did not match the acquisition)")
    return "fixed", have, want
