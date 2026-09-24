"""Correct ``custom/t_reference`` in already-beamformed buffer-2 IQ files.

Files written before 2026-09-24 place the last reference frame one frame interval before the
first tracking frame, as if no push happened in between. The real interval is PRI + the push burst
rounded up to 100 us (`swp.acquisition.sequence.SWGeometry.push_gap_s`; 0.97 ms at 1500 cycles,
1.17 ms at 1900 cycles, against the 0.27 ms that was stored). New beamforms get it right; this
fixes old ones **in place** without re-beamforming:

* writes the corrected ``custom/t_reference`` and a ``custom/push_gap_s`` scalar;
* keeps the original array as ``custom/t_reference_v0`` (so the change is reversible);
* is idempotent - a file that already has ``push_gap_s`` is skipped.

Only the reference *timestamps* change; no IQ is touched. The push parameters come from the
folder's ``AcquisitionParametersAndECG.mat`` (or ``CombinedData.mat``).

    python scripts/retrofit_push_gap.py <folder> [<folder> ...]            # dry run: report only
    python scripts/retrofit_push_gap.py <folder> [<folder> ...] --apply
    python scripts/retrofit_push_gap.py --root "Z:/raw_data" --apply        # every folder below
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from swp.acquisition.sequence import SWGeometry            # noqa: E402


def sw_geometry(folder) -> SWGeometry:
    runtime = os.path.join(folder, "AcquisitionParametersAndECG.mat")
    if os.path.exists(runtime):
        import scipy.io as sio
        sw = sio.loadmat(runtime, squeeze_me=True, struct_as_record=False)["SW"]
        return SWGeometry(
            n_reference=int(sw.Ndetect_pre), n_tracking=int(sw.Ndetect), na=int(sw.na),
            harmonic=bool(sw.HarmonicImaging), pri=float(sw.PRI_us) * 1e-6,
            focus_x=0.0, focus_z=0.0, roi_xlims=(0.0, 0.0), roi_zlims=(0.0, 0.0),
            push_cycles=float(sw.pushCycle), push_freq_hz=float(sw.PushFrequency) * 1e6,
            switch_tpc=bool(getattr(sw, "SwitchTPCprofile", 0)))
    from swp.acquisition.sequence import read_swi_meta
    return read_swi_meta(os.path.join(folder, "CombinedData.mat")).sw


def corrected_t_reference(n_ref: int, g: SWGeometry, pi_mode: str = "sliding") -> np.ndarray:
    """Same formula as `assemble_tracking_frames` (kept in one place there; mirrored here for
    files whose raw data is not re-read)."""
    per_pos = g.na * (2 if g.harmonic else 1)
    step_tx = 1 if (pi_mode == "sliding" and g.harmonic and g.na == 1) else per_pos
    n_ref_tx = g.n_reference * per_pos
    first_tx = np.arange(n_ref) * step_tx
    return ((first_tx - (n_ref_tx - 1)) * g.pri - g.push_gap_s()).astype(np.float32)


def retrofit(folder, apply: bool) -> int:
    import h5py
    files = sorted(glob.glob(os.path.join(folder, "output", "*_buffer2_meas*_iq.hdf5")))
    if not files:
        return 0
    g = sw_geometry(folder)
    n_done = 0
    for p in files:
        with h5py.File(p, "r+" if apply else "r") as f:
            if "custom/t_reference" not in f:
                continue
            if "custom/push_gap_s" in f:
                continue
            old = np.asarray(f["custom/t_reference"])
            new = corrected_t_reference(old.shape[0], g)
            if n_done == 0:
                print(f"  {os.path.basename(folder)}: gap {g.push_gap_s() * 1e6:.0f} us; last reference "
                      f"frame {old[-1] * 1e6:.0f} -> {new[-1] * 1e6:.0f} us ({len(files)} files)")
            if apply:
                f.create_dataset("custom/t_reference_v0", data=old)
                del f["custom/t_reference"]
                f.create_dataset("custom/t_reference", data=new)
                f.create_dataset("custom/push_gap_s", data=np.float32(g.push_gap_s()))
            n_done += 1
    return n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--root", help="retrofit every measurement folder below this root")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    a = ap.parse_args()
    folders = list(a.folders)
    if a.root:
        folders += sorted({os.path.dirname(os.path.dirname(p)) for p in
                           glob.glob(os.path.join(a.root, "**", "output", "*_buffer2_meas0_iq.hdf5"),
                                     recursive=True)})
    total = 0
    for fo in folders:
        total += retrofit(fo, a.apply)
    print(f"{'corrected' if a.apply else 'would correct'} {total} file(s) in {len(folders)} folder(s)"
          + ("" if a.apply else "  (dry run - add --apply)"))


if __name__ == "__main__":
    main()
