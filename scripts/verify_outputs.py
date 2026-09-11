"""Verify (and optionally repair) the stage-1/2 outputs of a processed raw-data tree.

Checks each measurement folder's ``output/`` for what the downstream stages actually need,
and reports anything missing. With ``--repair`` it fixes what can be fixed in place without
re-beamforming.

Checked per folder:
  * the expected IQ files exist for every buffer present in the acquisition
  * every shear-wave IQ (buffer 2 ``*_meas*``, and the buffer-4 passive stream) carries the
    scan parameters the viz/passive stages read (prf / dz / dx / sound_speed / wavelength /
    demodulation_frequency ...). This is the one that intermittently goes missing: the
    append step writes via a temp file + os.replace, and on a network share that
    occasionally loses to a sharing violation (see swp.acquisition.scanparams).
  * a GIF exists next to every IQ file
  * no leftover ``*.tmp.hdf5`` / ``.tmp-*`` files from an interrupted write

``--repair`` re-appends the scan parameters (``append_scan_params_to_iq``) and removes stale
temp files. Missing IQ or GIF files are reported but not rebuilt - rerun
``process_raw_data.py --folder <f> --redo`` for those.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# Scan parameters the viz / passive loaders rely on.
REQUIRED_SCAN_PARAMS = {"prf", "dz", "dx", "sound_speed", "wavelength",
                        "demodulation_frequency"}


def custom_keys(path):
    import h5py
    try:
        with h5py.File(str(path), "r") as f:
            return set(f["custom"].keys()) if "custom" in f else set()
    except OSError:
        return None            # unreadable / truncated


def check_folder(folder, repair=False):
    """Return (n_problems, list of message strings) for one measurement folder."""
    folder = Path(folder)
    out = folder / "output"
    msgs = []
    if not out.is_dir():
        return 1, [f"no output/ directory"]

    iq_files = sorted(out.glob("*_iq.hdf5"))
    if not iq_files:
        return 1, ["no *_iq.hdf5 files"]

    # stale temp files from an interrupted write
    stale = sorted(out.glob("*.tmp.hdf5")) + sorted(out.glob(".*tmp-*.hdf5"))
    for s in stale:
        if repair:
            try:
                s.unlink()
                msgs.append(f"removed stale temp {s.name}")
            except OSError as exc:
                msgs.append(f"could not remove stale temp {s.name}: {exc}")
        else:
            msgs.append(f"stale temp file {s.name}")

    n_bad = 0
    for iq in iq_files:
        # every IQ file should have a GIF beside it
        if not iq.with_suffix(".gif").exists():
            msgs.append(f"missing GIF for {iq.name}")
            n_bad += 1

        # shear-wave IQ (buffer 2 per-measurement, buffer 4 passive) needs scan params
        needs_params = "_meas" in iq.stem or "buffer4" in iq.stem
        if not needs_params:
            continue
        keys = custom_keys(iq)
        if keys is None:
            msgs.append(f"UNREADABLE {iq.name}")
            n_bad += 1
            continue
        missing = REQUIRED_SCAN_PARAMS - keys
        if not missing:
            continue
        n_bad += 1
        if not repair:
            msgs.append(f"{iq.name}: missing scan params {sorted(missing)}")
            continue
        from swp.acquisition.scanparams import append_scan_params_to_iq
        try:
            append_scan_params_to_iq(iq)
            still = REQUIRED_SCAN_PARAMS - (custom_keys(iq) or set())
            if still:
                msgs.append(f"{iq.name}: REPAIR INCOMPLETE, still missing {sorted(still)}")
            else:
                msgs.append(f"{iq.name}: repaired scan params")
                n_bad -= 1
        except Exception as exc:                     # noqa: BLE001
            msgs.append(f"{iq.name}: REPAIR FAILED {type(exc).__name__}: {exc}")
    return n_bad, msgs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help=r'study root, e.g. "Z:\raw_data"')
    p.add_argument("--folder", action="append", default=[], help="a measurement folder (repeatable)")
    p.add_argument("--repair", action="store_true",
                   help="re-append missing scan parameters and delete stale temp files")
    a = p.parse_args()
    if not a.root and not a.folder:
        p.error("give --root and/or --folder")

    sys.path.insert(0, str(_ROOT / "scripts"))
    from process_raw_data import find_measurement_folders
    folders = find_measurement_folders(a.root, a.folder, None)
    if not folders:
        raise SystemExit("no measurement folders found")

    print(f"=== verifying {len(folders)} folder(s){' (repair on)' if a.repair else ''} ===")
    total_bad, bad_folders = 0, []
    for f in folders:
        n_bad, msgs = check_folder(f, repair=a.repair)
        name = f"{f.parent.name}/{f.name}"
        if msgs:
            print(f"\n  {name}")
            for m in msgs:
                print(f"      {m}")
        if n_bad:
            total_bad += n_bad
            bad_folders.append(name)
    print(f"\n{len(folders) - len(bad_folders)}/{len(folders)} folder(s) clean; "
          f"{total_bad} outstanding problem(s)")
    if bad_folders:
        print("folders needing attention:")
        for b in bad_folders:
            print(f"  {b}")
    return 1 if bad_folders else 0


if __name__ == "__main__":
    sys.exit(main())
