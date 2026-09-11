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


def base_config_file() -> str:
    """Explicit base config chosen with ``SWP_BASE_CONFIG`` (``""`` = auto-select).

    The in-vivo constant parameters differ per acquisition campaign
    (``S5_1_SWI_PulseInversion_P1-6`` / ``P11-14`` / ``P15-xx``) and only the
    campaign's own config has a ``Receive``/``RcvBuffer`` layout matching the data,
    so ``make_combined_data.m``'s default (newest campaign) is wrong for an older
    folder. Set ``SWP_BASE_CONFIG`` to an absolute path or to a file name inside
    :func:`base_config_dir` to pin it.
    """
    return os.environ.get("SWP_BASE_CONFIG", "")


# =====================================================================================
# Buffer-layout check: does a base config actually describe this acquisition?
# =====================================================================================
# The base config supplies the constant `Resource.RcvBuffer` -- how many frames each RF
# buffer holds and how many rows each frame occupies. The acquisition independently
# records what it actually wrote, as `RF_frames` / `RF_rows`. If a base config from the
# wrong campaign is used, those disagree and *every* buffer is sliced out of the wrong
# region of the RF: the 2026-04 campaign wrote [90, 20, 26, 926, 20, 268] frames while
# the newest config (which make_combined_data.m auto-selects for any in-vivo folder)
# describes [106, 24, 32, 1112, 24, 318].
#
# Nothing downstream notices -- beamforming succeeds and produces plausible-looking
# B-modes -- so this is checked explicitly, both when picking a base config and on the
# merged CombinedData.mat before it is ever read.

class BaseConfigMismatch(RuntimeError):
    """A base config's RcvBuffer layout does not match the acquisition's RF_frames/RF_rows."""


def _h5_ref_values(f, group, field):
    """Read a MATLAB struct-array field stored as HDF5 object references -> list of floats."""
    import numpy as np
    refs = np.atleast_1d(np.array(group[field]).squeeze())
    return [float(np.array(f[r]).squeeze()) for r in refs]


def rcvbuffer_layout(mat_path):
    """``(numFrames, rowsPerFrame)`` from a v7.3 workspace's ``Resource.RcvBuffer``."""
    import h5py
    with h5py.File(str(mat_path), "r") as f:
        rb = f["Resource"]["RcvBuffer"]
        return ([int(v) for v in _h5_ref_values(f, rb, "numFrames")],
                [int(v) for v in _h5_ref_values(f, rb, "rowsPerFrame")])


def acquisition_layout(folder):
    """``(RF_frames, RF_rows)`` the acquisition actually wrote, from the runtime .mat.

    Reads ``AcquisitionParametersAndECG.mat`` (v7 / MATLAB 5.0) if present, else falls
    back to the ``RF_frames``/``RF_rows`` copied into a built ``CombinedData.mat``.
    """
    import numpy as np
    folder = Path(folder)
    dynamic = folder / "AcquisitionParametersAndECG.mat"
    if dynamic.is_file():
        import scipy.io as sio
        d = sio.loadmat(str(dynamic), squeeze_me=True, struct_as_record=False)
        return ([int(v) for v in np.atleast_1d(d["RF_frames"]).reshape(-1)],
                [int(v) for v in np.atleast_1d(d["RF_rows"]).reshape(-1)])
    combined = folder / "CombinedData.mat"
    if combined.is_file():
        import h5py
        with h5py.File(str(combined), "r") as f:
            if "RF_frames" in f:
                return ([int(v) for v in np.array(f["RF_frames"]).reshape(-1)],
                        [int(v) for v in np.array(f["RF_rows"]).reshape(-1)])
    raise FileNotFoundError(f"no RF_frames/RF_rows available in {folder}")


def base_config_matches(mat_path, folder, raise_on_mismatch=False):
    """True when ``mat_path``'s RcvBuffer layout matches the acquisition in ``folder``.

    Compares only the buffers the acquisition actually wrote, so a config describing
    extra buffers is fine as long as the shared ones agree.
    """
    frames, rows = acquisition_layout(folder)
    try:
        cfg_frames, cfg_rows = rcvbuffer_layout(mat_path)
    except (KeyError, OSError) as exc:
        if raise_on_mismatch:
            raise BaseConfigMismatch(f"{mat_path}: no readable Resource.RcvBuffer ({exc})")
        return False
    n = min(len(frames), len(cfg_frames))
    ok = (n == len(frames)
          and cfg_frames[:n] == frames[:n] and cfg_rows[:n] == rows[:n])
    if not ok and raise_on_mismatch:
        raise BaseConfigMismatch(
            f"base config does not match this acquisition:\n"
            f"  base config {Path(mat_path).name}\n"
            f"    numFrames    = {cfg_frames}\n"
            f"    rowsPerFrame = {cfg_rows}\n"
            f"  acquisition {Path(folder).name}\n"
            f"    RF_frames    = {frames}\n"
            f"    RF_rows      = {rows}\n"
            "Using it would slice every RF buffer out of the wrong region. Pick the base "
            "config recorded for this acquisition campaign (--base-config / SWP_BASE_CONFIG), "
            "or let it be auto-selected by omitting both."
        )
    return ok


def candidate_base_configs(base_dir):
    """Base config .mat files in ``base_dir`` (+ ``PhantomSweep/``), newest-looking first.

    Skips the non-config files that live alongside them (``NonzeroRFcolumns.mat``, and
    any ``CombinedData.mat`` left in the directory).
    """
    base_dir = Path(base_dir)
    skip = {"nonzerorfcolumns.mat", "combineddata.mat"}
    found = [p for p in sorted(base_dir.glob("*.mat")) if p.name.lower() not in skip]
    found += sorted((base_dir / "PhantomSweep").glob("*.mat"))
    return found


def matching_base_configs(folder, base_dir):
    """All base configs in ``base_dir`` whose RcvBuffer layout matches ``folder``.

    Raises :class:`BaseConfigMismatch` when none matches - that is the case that would
    otherwise silently mis-slice every buffer.
    """
    cands = candidate_base_configs(base_dir)
    if not cands:
        raise BaseConfigMismatch(f"no base config .mat files found in {base_dir!r}")
    matches = [p for p in cands if base_config_matches(p, folder)]
    if not matches:
        frames, rows = acquisition_layout(folder)
        detail = "\n".join(f"    {p.name:46s} numFrames={_safe_layout(p)}" for p in cands)
        raise BaseConfigMismatch(
            f"no base config in {base_dir!r} matches this acquisition.\n"
            f"  acquisition RF_frames = {frames}\n"
            f"  candidates:\n{detail}\n"
            "The acquisition needs the base config recorded for its own campaign; add it "
            "to the base config directory."
        )
    return matches


def find_matching_base_config(folder, base_dir):
    """The single base config in ``base_dir`` matching ``folder``, or ``None`` if ambiguous.

    ``None`` means "several configs share this buffer layout, so the layout cannot
    choose between them" - which is the normal case for the **phantom sweep**, where
    every ``BaseConfig_10frames_*`` has the same ``RcvBuffer`` and the discriminator is
    the push setting encoded in the file *name* (cycles / elements / PRI). There,
    ``make_combined_data.m``'s own naming rule is the authority, so we defer to it
    rather than guessing; the merged file is validated afterwards either way.

    Raises :class:`BaseConfigMismatch` if nothing matches at all.
    """
    matches = matching_base_configs(folder, base_dir)
    return matches[0] if len(matches) == 1 else None


def _safe_layout(path):
    try:
        return rcvbuffer_layout(path)[0]
    except Exception:  # noqa: BLE001 - only for the error message
        return "<unreadable>"


def validate_combined_data(combined, folder=None):
    """Assert a built ``CombinedData.mat`` describes the acquisition it sits next to.

    This is the backstop that catches a wrong base config no matter how the file got
    there - freshly merged, left over from an earlier run with the old auto-selection,
    or copied in by hand. ``CombinedData.mat`` carries both halves (the base config's
    ``Resource.RcvBuffer`` and the acquisition's ``RF_frames``/``RF_rows``), so it can
    be checked on its own.
    """
    combined = Path(combined)
    folder = Path(folder) if folder is not None else combined.parent
    base_config_matches(combined, folder, raise_on_mismatch=True)
    return combined


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


def ensure_combined_data(folder, base_config_dir_=None, overwrite=False,
                         base_config=None) -> Path:
    """Ensure ``<folder>/CombinedData.mat`` exists, building it via MATLAB if needed.

    If ``CombinedData.mat`` is already present it is returned unchanged (unless
    ``overwrite``). Otherwise ``AcquisitionParametersAndECG.mat`` must exist in the
    folder and is merged with the base config into ``CombinedData.mat``.

    The base config is **auto-selected** by matching each candidate's
    ``Resource.RcvBuffer`` layout against the acquisition's ``RF_frames``/``RF_rows``,
    so the right campaign is picked without being named. Every path out of this
    function is validated (:func:`validate_combined_data`), including an already
    existing ``CombinedData.mat`` - a file built by an earlier run with the old
    newest-campaign default is rejected rather than silently reused.

    Args:
        base_config: explicit base config (absolute path, or a file name inside
            ``base_config_dir_``), skipping auto-selection. It is still checked
            against the acquisition. Defaults to :func:`base_config_file`
            (``SWP_BASE_CONFIG``); empty = auto-select.

    Returns the path to ``CombinedData.mat``.

    Raises:
        BaseConfigMismatch: the chosen (or existing) layout does not describe this
            acquisition.
    """
    folder = Path(folder)
    combined = folder / "CombinedData.mat"
    if combined.is_file() and not overwrite:
        validate_combined_data(combined, folder)
        print(f"  [CombinedData] reusing existing {combined.name} "
              f"(RcvBuffer layout matches the acquisition)")
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

    base_file = base_config if base_config is not None else base_config_file()
    if base_file:
        resolved = Path(base_file) if Path(base_file).is_file() else Path(base_dir) / base_file
        if not resolved.is_file():
            raise FileNotFoundError(
                f"base config not found: {base_file!r} (neither an absolute path nor a "
                f"file in {base_dir!r})"
            )
        # An explicitly named config is still checked: naming one by hand is exactly
        # where the wrong campaign gets picked.
        base_config_matches(resolved, folder, raise_on_mismatch=True)
        print(f"  [CombinedData] base config (given): {resolved.name}")
    else:
        resolved = find_matching_base_config(folder, base_dir)
        if resolved is None:
            # Several configs share this layout (the phantom sweep): let the MATLAB
            # rule pick by push settings, then validate what it produced.
            print("  [CombinedData] base config: several share this RcvBuffer layout "
                  "(phantom sweep) - deferring to make_combined_data.m's naming rule")
        else:
            print(f"  [CombinedData] base config (auto-selected by RcvBuffer layout): "
                  f"{resolved.name}")
    base_file = str(resolved) if resolved is not None else ""

    matlab = _find_matlab()
    # -batch runs non-interactively and returns a non-zero exit code on error.
    call = f"make_combined_data('{folder}','{base_dir}','{base_file}')"
    cmd = [matlab, "-batch", call]
    print(f"  [CombinedData] building via MATLAB -> {combined.name}")
    print(f"    {matlab} -batch {call}")
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
    # Backstop: confirm what MATLAB actually wrote matches the acquisition, so a bad
    # merge can never reach the beamformer.
    validate_combined_data(combined, folder)
    repair_buffer2_receive(folder)
    print(f"  [CombinedData] done -> {combined} (RcvBuffer layout verified)")
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
