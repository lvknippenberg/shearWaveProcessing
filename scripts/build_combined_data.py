"""Build ``CombinedData.mat`` for every measurement folder under a study root (MATLAB, Windows only).

The Verasonics acquisition saves only the dynamic parameters
(``AcquisitionParametersAndECG.mat``); the beamformer needs them merged with the campaign's constant
base config into ``CombinedData.mat``. That merge runs in MATLAB (``make_combined_data.m``), so it has
to happen on a machine with MATLAB - the Linux GPU server cannot do it (see ``docs/linux_server.md``).
This front-loads the merge for a whole tree so stages 1-2 can then run anywhere.

Per folder: the base config is auto-selected by matching ``Resource.RcvBuffer`` against the
acquisition's ``RF_frames``/``RF_rows``, MATLAB writes the file, and the result is validated and
buffer-2-repaired (:func:`swp.acquisition.combined.ensure_combined_data`).

Cost: ~2 min and ~500 MB per folder (the file is a copy of the base config plus the runtime
parameters), dominated by copying/writing over the network - hence ``--jobs`` for parallel MATLAB
sessions. Resumable: folders that already have a valid ``CombinedData.mat`` are skipped, and a
failure never stops the run.

Usage:
    python scripts/build_combined_data.py --root "Z:\\raw_data" --dry-run
    python scripts/build_combined_data.py --root "Z:\\raw_data" --jobs 4
    python scripts/build_combined_data.py --folder "<folder>" --overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

COMBINED_MAT = "CombinedData.mat"


def build_one(folder, base_config_dir=None, overwrite=False, quiet=True):
    """-> (folder, status, seconds, note); never raises."""
    import contextlib
    import io

    from swp.acquisition.combined import ensure_combined_data

    t0 = time.perf_counter()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf if quiet else sys.stdout):
            ensure_combined_data(folder, base_config_dir_=base_config_dir, overwrite=overwrite)
        return folder, "ok", time.perf_counter() - t0, ""
    except Exception as exc:                              # noqa: BLE001
        note = f"{type(exc).__name__}: {exc}".splitlines()[0]
        return folder, "FAILED", time.perf_counter() - t0, note


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help=r'study root, e.g. "Z:\raw_data"')
    p.add_argument("--subject", default=None)
    p.add_argument("--folder", action="append", default=[])
    p.add_argument("--base-config-dir", default=None,
                   help="directory of base config .mat files (default: $SWP_BASE_CONFIG_DIR)")
    p.add_argument("--jobs", type=int, default=1, help="parallel MATLAB sessions (default 1)")
    p.add_argument("--limit", type=int, default=None, help="only the first N folders (for a trial)")
    p.add_argument("--overwrite", action="store_true", help="rebuild even if the file exists")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not a.root and not a.folder:
        p.error("give --root and/or --folder")
    if a.base_config_dir:
        os.environ["SWP_BASE_CONFIG_DIR"] = a.base_config_dir

    from process_raw_data import find_measurement_folders
    folders = find_measurement_folders(a.root, a.folder, a.subject)
    todo = [f for f in folders if a.overwrite or not (f / COMBINED_MAT).is_file()]
    done = len(folders) - len(todo)
    if a.limit:
        todo = todo[:a.limit]
    print(f"=== CombinedData.mat: {len(folders)} folder(s), {done} already built, "
          f"{len(todo)} to build, {a.jobs} job(s) ===", flush=True)
    if a.dry_run:
        for f in todo:
            print(f"  would build: {f}")
        return 0

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futures = {pool.submit(build_one, str(f), a.base_config_dir, a.overwrite): f for f in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            folder, status, secs, note = fut.result()
            results.append((folder, status, secs, note))
            mins = (time.perf_counter() - t0) / 60
            rate = mins / k
            print(f"  [{k}/{len(todo)}] {status:6s} {secs / 60:4.1f} min  {folder}"
                  + (f"\n        {note}" if note else "")
                  + f"\n        ({mins:.0f} min elapsed, {rate:.2f} min/folder, "
                    f"~{rate * (len(todo) - k):.0f} min left)", flush=True)

    failed = [r for r in results if r[1] != "ok"]
    print(f"\n=== {len(results) - len(failed)} ok, {len(failed)} failed, {done} skipped, "
          f"{(time.perf_counter() - t0) / 60:.1f} min ===")
    for folder, _, _, note in failed:
        print(f"  FAILED {folder}\n         {note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
