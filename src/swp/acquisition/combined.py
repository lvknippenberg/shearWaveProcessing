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
    print(f"  [CombinedData] done -> {combined}")
    return combined
