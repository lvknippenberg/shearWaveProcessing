"""Put buffer 3 (focused live loop, circular buffer) of already-processed folders in chronological order.

New conversions do this automatically (``beamform.process_folder`` calls
``swp.acquisition.unwrap.unwrap_buffer3``); this script retrofits folders beamformed before
2026-09-25. It is idempotent: folders whose buffer-3 files carry a flag of the current unwrap VERSION
are skipped; older-version flags (v1, before 2026-09-28) are re-estimated and re-rotated, so it can simply be re-run (e.g. after the server batch has finished more folders).

    python scripts/unwrap_buffer3.py --root "Z:/raw_data" --dry-run      # estimate only, write nothing
    python scripts/unwrap_buffer3.py --root "Z:/raw_data"                # estimate + rewrite + GIF
    python scripts/unwrap_buffer3.py --folder "<folder>"

A folder is processed only once its beamforming is complete (the buffer-4 GIF exists: the batch
writes it after buffers 1-4). Per folder one row goes to study/logs/buffer3_unwrap_<date>.csv.
Background and validation: docs/buffer3_unwrap.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--folder", action="append", default=[])
    ap.add_argument("--subject", default=None)
    ap.add_argument("--dry-run", action="store_true", help="estimate only; write nothing to the data")
    ap.add_argument("--no-gif", action="store_true", help="do not re-render the buffer-3 GIF")
    ap.add_argument("--log", default=None, help="CSV path (default study/logs/buffer3_unwrap_<date>.csv)")
    a = ap.parse_args()
    if not a.root and not a.folder:
        ap.error("give --root and/or --folder")

    from process_raw_data import find_measurement_folders
    from swp.acquisition.unwrap import buffer_files, is_current, read_flag, summary, unwrap_buffer3

    folders = find_measurement_folders(a.root, a.folder, a.subject)
    out = Path(a.log) if a.log else _REPO / "study" / "logs" / (
        f"buffer3_unwrap_{'dryrun_' if a.dry_run else ''}{time.strftime('%Y%m%d')}.csv")
    rows, t_start = [], time.perf_counter()
    counts = {}
    for k, f in enumerate(folders):
        f = Path(f)
        name = f"{f.parent.name}/{f.name}"
        outdir = f / "output"
        conv, iq = buffer_files(outdir)
        if iq is None or not (outdir / "CombinedData_buffer4_iq.gif").exists():
            counts["not ready"] = counts.get("not ready", 0) + 1
            continue
        variants = [v for v in outdir.glob("*_buffer3_*_iq.hdf5") if not is_current(read_flag(v))]
        if is_current(read_flag(iq)) and (conv is None or is_current(read_flag(conv))) and not variants and not a.dry_run:
            counts["already done"] = counts.get("already done", 0) + 1
            continue
        t0 = time.perf_counter()
        print(f"[{k + 1}/{len(folders)}] {name}", flush=True)
        try:
            est = unwrap_buffer3(str(f), outdir, dry_run=a.dry_run, gif=not a.no_gif)
            row = dict(folder=name, **summary(est))
        except Exception as exc:                                    # noqa: BLE001
            traceback.print_exc()
            row = dict(folder=name, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = round(time.perf_counter() - t0, 1)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        rows.append(row)
        keys = list(dict.fromkeys(key for r in rows for key in r))
        with open(out, "w", newline="") as fh:                       # rewritten after every folder
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"\n{(time.perf_counter() - t_start) / 60:.1f} min: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + (f"  -> {out}" if rows else ""))


if __name__ == "__main__":
    main()
