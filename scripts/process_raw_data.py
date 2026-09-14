"""Batch stages 1-2 over a raw-data tree: RF -> zea HDF5 -> IQ -> real-time B-mode GIFs.

One command to (re)process every in-vivo measurement folder under a study root, e.g.

    python scripts/process_raw_data.py --root "Z:\\raw_data"
    python scripts/process_raw_data.py --root "Z:\\raw_data" --subject C000000001
    python scripts/process_raw_data.py --folder "Z:\\raw_data\\C000000001\\SWE_01_..."

A *measurement folder* is any directory holding the runtime
``AcquisitionParametersAndECG.mat`` (or an already-merged ``CombinedData.mat``); the usual
layout is ``<root>/<subject>/SWE_*``. Each folder is independent, so a failure is reported
and the run continues with the next one - a bad folder never blocks the batch.

The base config is auto-selected by matching ``Resource.RcvBuffer`` against the folder's
``RF_frames``/``RF_rows``, and the merged ``CombinedData.mat`` is validated before it is
read, so the campaign-mismatch failure mode cannot silently occur (see
``swp.acquisition.combined``). Nothing here needs to name a base config.

Already-processed folders are skipped unless ``--overwrite``/``--redo`` is given, so
re-running over the whole tree only picks up what is new.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

RUNTIME_MAT = "AcquisitionParametersAndECG.mat"
COMBINED_MAT = "CombinedData.mat"


def find_measurement_folders(root=None, folders=(), subject=None):
    """Measurement folders to process, sorted and de-duplicated."""
    found = [Path(f) for f in folders]
    if root:
        root = Path(root)
        pattern = f"{subject}/*/" if subject else "*/*/"
        # <root>/<subject>/<measurement>, plus <root>/<measurement> for a flat root.
        cands = list(root.glob(pattern.rstrip("/"))) + list(root.glob("*"))
        for p in cands:
            if p.is_dir() and ((p / RUNTIME_MAT).is_file() or (p / COMBINED_MAT).is_file()):
                found.append(p)
    seen, out = set(), []
    for p in found:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return sorted(out, key=lambda p: str(p).lower())


def is_processed(folder):
    """True when stage 2 already wrote IQ + GIFs for this folder."""
    out = Path(folder) / "output"
    return bool(list(out.glob("*_iq.hdf5"))) and bool(list(out.glob("*_iq.gif")))


def check_base_configs(folders):
    """Audit which base config each folder resolves to. Reads no RF; writes nothing.

    Answers "will this batch run cleanly?" in seconds, and is the quickest way to spot a
    campaign whose base config is missing from the base config directory.
    """
    from swp.acquisition import (BaseConfigMismatch, acquisition_layout,
                                 matching_base_configs, validate_combined_data)
    from swp.acquisition.combined import base_config_dir

    bdir = base_config_dir()
    print(f"=== base config audit ({len(folders)} folder(s)) against {bdir!r} ===")
    bad = 0
    for folder in folders:
        name = f"{folder.parent.name}/{folder.name}"
        try:
            frames, _ = acquisition_layout(folder)
        except Exception as exc:                      # noqa: BLE001
            print(f"  ?? {name}\n       cannot read layout: {exc}")
            bad += 1
            continue
        try:
            matches = matching_base_configs(folder, bdir)
            pick = (matches[0].name if len(matches) == 1
                    else f"{len(matches)} candidates (MATLAB naming rule decides)")
            status = "ok"
        except BaseConfigMismatch as exc:
            pick = str(exc).splitlines()[0]
            status = "NO MATCH"
            bad += 1
        combined = folder / COMBINED_MAT
        note = ""
        if combined.is_file():
            try:
                validate_combined_data(combined, folder)
                note = "  [existing CombinedData.mat verified]"
            except BaseConfigMismatch:
                note = "  [existing CombinedData.mat MISMATCHED - rebuild with --overwrite]"
                status = "STALE"
                bad += 1
        print(f"  {status:8s} {name}")
        print(f"       RF_frames={frames} -> {pick}{note}")
    print(f"\n{len(folders) - bad} ok, {bad} needing attention")
    return 1 if bad else 0


def process_one(folder, make_gifs=True, gif_stretch=None, overwrite=False,
                save_converted=True, buffers_matlab=None):
    from swp.acquisition import process_folder
    return process_folder(folder, make_gifs=make_gifs, gif_stretch=gif_stretch,
                          overwrite=overwrite, save_converted=save_converted,
                          buffers_matlab=buffers_matlab)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None,
                   help=r'study root to scan, e.g. "Z:\raw_data"')
    p.add_argument("--subject", default=None,
                   help="only this subject folder under --root (e.g. C000000001)")
    p.add_argument("--folder", action="append", default=[],
                   help="process this measurement folder (repeatable; may replace --root)")
    p.add_argument("--base-config-dir", default=None,
                   help="directory of base config .mat files (default: $SWP_BASE_CONFIG_DIR, "
                        "else the packaged default). The config itself is auto-selected.")
    p.add_argument("--buffers", default=None,
                   help="only these MATLAB buffer numbers, comma-separated (e.g. 3). "
                        "Default: every buffer present. Implies --redo, since the point of "
                        "naming buffers is to redo them.")
    p.add_argument("--no-gifs", action="store_true", help="skip GIF rendering")
    p.add_argument("--no-converted", action="store_true",
                   help="skip writing the converted RF zea database copies")
    p.add_argument("--gif-stretch", type=float, default=None,
                   help="GIF playback duration / acquisition duration "
                        "(default 1 = real time; 3 = 3x slow motion)")
    p.add_argument("--redo", action="store_true",
                   help="reprocess folders that already have IQ + GIFs")
    p.add_argument("--overwrite", action="store_true",
                   help="also re-read/re-convert the RF from the .mat (implies --redo)")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be processed and exit")
    p.add_argument("--check", action="store_true",
                   help="audit only: per folder, print RF_frames and the base config that "
                        "matches it (and validate any existing CombinedData.mat). Reads no "
                        "RF and writes nothing; exits non-zero if any folder has no match.")
    a = p.parse_args()

    if a.base_config_dir:
        os.environ["SWP_BASE_CONFIG_DIR"] = a.base_config_dir
    if not a.root and not a.folder:
        p.error("give --root and/or --folder")

    folders = find_measurement_folders(a.root, a.folder, a.subject)
    if not folders:
        raise SystemExit(f"no measurement folders found (looked for {RUNTIME_MAT} / "
                         f"{COMBINED_MAT} under {a.root!r})")

    if a.check:
        return check_base_configs(folders)

    buffers = [int(x) for x in a.buffers.split(",")] if a.buffers else None
    redo = a.redo or a.overwrite or bool(buffers)
    todo = [f for f in folders if redo or not is_processed(f)]
    skipped = [f for f in folders if f not in todo]

    print(f"=== swp batch: {len(folders)} measurement folder(s) found, "
          f"{len(todo)} to process, {len(skipped)} already done ===")
    for f in skipped:
        print(f"  skip (done)  {f}")
    for f in todo:
        print(f"  queued       {f}")
    if a.dry_run:
        return 0

    results = []
    for i, folder in enumerate(todo, 1):
        print(f"\n{'=' * 90}\n[{i}/{len(todo)}] {folder}\n{'=' * 90}")
        t0 = time.time()
        try:
            process_one(folder, make_gifs=not a.no_gifs, gif_stretch=a.gif_stretch,
                        overwrite=a.overwrite, save_converted=not a.no_converted,
                        buffers_matlab=buffers)
            results.append((folder, "ok", time.time() - t0, ""))
        except Exception as exc:                       # noqa: BLE001 - keep the batch going
            traceback.print_exc()
            results.append((folder, "FAILED", time.time() - t0,
                            f"{type(exc).__name__}: {exc}".splitlines()[0]))

    print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
    for folder, status, secs, note in results:
        print(f"  {status:7s} {secs / 60:5.1f} min  {folder}")
        if note:
            print(f"                        {note}")
    n_failed = sum(1 for r in results if r[1] != "ok")
    print(f"\n{len(results) - n_failed} ok, {n_failed} failed, {len(skipped)} skipped")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
