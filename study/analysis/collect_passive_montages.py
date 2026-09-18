"""Gather every folder's passive montages into one place, for scrolling through the study.

The montages live one per measurement folder, deep under ``Z:\\raw_data\\<subject>\\<folder>\\output\\
swp_passive``, which makes comparing subjects a matter of opening 36 directories. This copies them
into a flat tree named by subject, so each set can be paged through in one image viewer:

    <out>/main/         <subject>_<event lines>.png   the per-event montage (rows = windows, cols = views)
    <out>/split_full/   <subject>.png                 same windows, full M-line
    <out>/split_left/   <subject>.png                 left half
    <out>/split_right/  <subject>.png                 right half
    <out>/split_lines/  <subject>.png                 the three parts drawn on each event's B-mode
    <out>/bursts/       <subject>.png                 burst detection along the general line

Names start with the subject id so the viewer's own ordering walks the study in order. Only files
that exist are copied; a folder that was skipped or has no split simply does not appear.

Usage:
    python study/analysis/collect_passive_montages.py --root "Z:/raw_data" --out "<folder>"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

# (destination subfolder, path of the file inside output/swp_passive)
SETS = [
    ("main", "passive_windows_montage.png"),
    ("split_full", "split_full/passive_windows_montage.png"),
    ("split_left", "split_left/passive_windows_montage.png"),
    ("split_right", "split_right/passive_windows_montage.png"),
    ("split_lines", "split_lines.png"),
    ("bursts", "passive_bursts.png"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--out", required=True, help="destination folder (created if missing)")
    ap.add_argument("--config", default=str(_REPO / "configs" / "passive.yaml"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="limit to these destination sets (default: all)")
    a = ap.parse_args()

    from passive_study import status
    from process_raw_data import find_measurement_folders
    from swp.passive import _paths, read_windows

    sets = [s for s in SETS if a.only is None or s[0] in a.only]
    out = Path(a.out)
    counts = {name: 0 for name, _ in sets}
    rows = []
    for f in find_measurement_folders(a.root, [], None):
        _, p = _paths(str(f), a.config)
        outdir = Path(p["outdir"])
        if not outdir.is_dir():
            continue
        # A folder skipped at drawing can still hold a montage from an earlier round, drawn on a
        # line that no longer exists. Copying it would put a stale result in the middle of the
        # scroll with nothing to mark it as such.
        if status(f, a.config) != "processed":
            continue
        subject = f.parent.name
        st, windows = read_windows(p["windows_json"])
        st = st or {}
        wm = st.get("window_mlines") or {}
        n_sk = len(st.get("skipped") or [])
        # Name the main montage with what it was built from, so the difference is visible while
        # scrolling rather than buried in a json.
        if not wm:
            tag = "general-line"
        elif n_sk >= len(windows):
            tag = "general-line-all-events-skipped"      # nothing was reprocessed for this folder
        elif n_sk:
            tag = f"per-event-{len(windows) - n_sk}of{len(windows)}"
        else:
            tag = "per-event"
        for name, rel in sets:
            src = outdir / rel
            if not src.is_file():
                continue
            dst_dir = out / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / (f"{subject}_{tag}.png" if name == "main" else f"{subject}.png")
            shutil.copy2(src, dst)
            counts[name] += 1
        rows.append((subject, tag, len(windows), n_sk))

    print(f"{len(rows)} folder(s) -> {out}")
    for name, _ in sets:
        print(f"  {name:<12s} {counts[name]:>3d} file(s)")
    missing = [r[0] for r in rows if not (out / "split_full" / f"{r[0]}.png").exists()]
    if missing:
        print("  no split montage: " + ", ".join(missing))


if __name__ == "__main__":
    sys.exit(main())
