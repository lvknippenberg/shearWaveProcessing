"""Batch: ensure CombinedData.mat -> repair the buffer-2 Receive layout -> beamform.

Runs the three preparation steps for every measurement folder under one or more roots:

  1. `swp.acquisition.ensure_combined_data` (MATLAB merge of the runtime .mat + base config),
  2. `scripts/fix_buffer2_receive.patch` -- corrects the buffer-2 receive length when the base
     config's SW.endDepth does not match the acquisition (see that module's docstring),
  3. `run.py beamform --no-gifs --no-converted` (forced with --overwrite when step 2 changed
     anything, since any existing IQ was reconstructed from the wrong sample windows).

    python scripts/batch_prepare.py --roots D:/swp_ph17 D:/swp_tue [--jobs-note ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
os.environ.setdefault("KERAS_BACKEND", "torch")

from fix_buffer2_receive import patch                       # noqa: E402


def has_iq(folder):
    out = os.path.join(folder, "output")
    return os.path.isdir(out) and any(
        n.startswith("CombinedData_buffer2_meas") for n in os.listdir(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()

    from swp.acquisition import ensure_combined_data, process_folder

    folders = []
    for root in a.roots:
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if os.path.isdir(p) and (os.path.isfile(os.path.join(p, "AcquisitionParametersAndECG.mat"))
                                     or os.path.isfile(os.path.join(p, "CombinedData.mat"))):
                if not a.prefix or d.startswith(a.prefix):
                    folders.append(p)
    print(f"{len(folders)} folders", flush=True)

    t0 = time.time()
    for i, folder in enumerate(folders, 1):
        name = os.path.basename(folder)[-8:]
        try:
            ensure_combined_data(folder)
            status, have, want = patch(folder)
            need = status == "fixed" or not has_iq(folder)
            print(f"[{i}/{len(folders)}] {name} receive={status}({have}->{want}) "
                  f"beamform={'yes' if need else 'skip'}  t={time.time()-t0:.0f}s", flush=True)
            if need:
                process_folder(folder, make_gifs=False, overwrite=True, save_converted=False)
        except Exception:                                    # noqa: BLE001
            print(f"[{i}/{len(folders)}] {name} FAILED", flush=True)
            traceback.print_exc()
    print(f"DONE {len(folders)} folders in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
