"""Passive SWE over a whole study: one M-line per folder on the focused B-mode, then unattended.

Most buffer-4 (diverging-wave) frames do not show the septum clearly, and it moves too much over
the cardiac cycle for a line drawn on a cine. So each folder gets ONE M-line, drawn on the
**first frame of the focused B-mode (buffer 3)**. The acquisitions are R-peak gated, so that frame
sits at the same cardiac phase as buffer 4's first frame, and the line (in metres) maps onto the
buffer-4 grid as is. It is used for burst detection and for every detected window.

Two phases, run as two separate processes:

``draw`` (interactive) - one still-frame prompt per folder, back to back (only one frame is read).
``process`` (unattended) - load buffer 4, detect bursts along the line, run every window through
    the 3 passive views -> ``output/swp_passive/passive_windows_montage.png`` +
    ``passive_speeds.json``. ``--watch`` keeps polling so it can run alongside ``draw``.

Resumable: a line is saved the moment it is drawn and drawn folders are skipped; processed folders
are skipped. Closing a window without clicking skips that folder (``--retry-skipped`` asks again).

Usage:
    python scripts/passive_study.py draw    --root "Z:/raw_data"
    python scripts/passive_study.py process --root "Z:/raw_data" --watch
    python scripts/passive_study.py status  --root "Z:/raw_data"
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
sys.path.insert(0, str(_ROOT / "scripts"))

CONFIG = str(_ROOT / "configs" / "passive.yaml")


def status(folder, config):
    """'missing' | 'todo' | 'skipped' | 'drawn' | 'no-bursts' | 'processed'."""
    from swp.passive import FOCUSED_BMODE, _paths, read_windows

    _, p = _paths(str(folder), config)
    if not (os.path.exists(p["iq"]) and os.path.exists(os.path.join(p["output"], FOCUSED_BMODE))):
        return "missing"
    st, windows = read_windows(p["windows_json"])
    if st is not None and st.get("skipped_general"):
        return "skipped"
    if not os.path.exists(p["general"]):
        return "todo"
    t_line = os.path.getmtime(p["general"])
    if st is not None and "key" in st and not windows and os.path.getmtime(p["windows_json"]) >= t_line:
        return "no-bursts"
    if os.path.exists(p["montage"]) and os.path.getmtime(p["montage"]) >= t_line:
        return "processed"
    return "drawn"


def _name(f):
    return f"{f.parent.name}/{f.name}"


def cmd_status(folders, a):
    counts = {}
    for f in folders:
        s = status(f, a.config)
        counts[s] = counts.get(s, 0) + 1
        print(f"  {s:10s} {_name(f)}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def cmd_draw(folders, a):
    from swp.passive import _paths, draw_focused_mline

    todo = []
    for f in folders:
        s = status(f, a.config)
        if s == "todo" or (a.redraw and s != "missing"):
            todo.append(f)
        elif a.retry_skipped and s == "skipped":
            os.remove(_paths(str(f), a.config)[1]["windows_json"])
            todo.append(f)
    print(f"=== passive M-lines (focused B-mode, frame 0): {len(todo)} of {len(folders)} to draw ===")
    print("Left-click points along the septum (any order), drag to adjust, right-click to delete,"
          "\nENTER to accept. Close the window without clicking to SKIP the folder. Ctrl-C stops;"
          "\neverything drawn so far is kept and the next run resumes.\n")
    t_start = time.perf_counter()
    for k, f in enumerate(todo):
        label = f"[{k + 1}/{len(todo)}] "
        print(f"\n{label}{_name(f)}", flush=True)
        try:
            draw_focused_mline(str(f), a.config, label=label)
        except KeyboardInterrupt:
            print("\ninterrupted - rerun to continue.")
            break
        except Exception:                                 # noqa: BLE001
            print(f"  FAILED:\n{traceback.format_exc()}")
    print(f"\n{(time.perf_counter() - t_start) / 60:.1f} min")


def cmd_process(folders, a):
    from swp.passive import process_single_line

    failed = set()
    while True:
        states = {f: status(f, a.config) for f in folders}
        ready = [f for f, s in states.items() if s == "drawn" and f not in failed]
        pending = [f for f, s in states.items() if s == "todo"]
        if not ready:
            if a.watch and pending:
                time.sleep(30)
                continue
            break
        f = ready[0]
        print(f"\n=== process {_name(f)} ({len(ready)} ready, {len(pending)} still to draw) ===",
              flush=True)
        t0 = time.perf_counter()
        try:
            process_single_line(str(f), a.config)
            print(f"  {time.perf_counter() - t0:.0f} s", flush=True)
        except Exception:                                 # noqa: BLE001
            failed.add(f)
            print(f"  FAILED:\n{traceback.format_exc()}", flush=True)
    states = [status(f, a.config) for f in folders]
    print(f"\nprocessed {states.count('processed')}/{len(folders)}, no bursts "
          f"{states.count('no-bursts')}, skipped {states.count('skipped')}; "
          f"failed this run: {len(failed)}")
    for f in failed:
        print(f"  FAILED {_name(f)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["draw", "process", "status"])
    p.add_argument("--root", default=None)
    p.add_argument("--folder", action="append", default=[])
    p.add_argument("--subject", default=None)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--redraw", action="store_true", help="draw: redo every line")
    p.add_argument("--retry-skipped", action="store_true", help="draw: ask again for skipped folders")
    p.add_argument("--watch", action="store_true",
                   help="process: keep polling for folders that finish drawing")
    a = p.parse_args()
    if not a.root and not a.folder:
        p.error("give --root and/or --folder")

    from process_raw_data import find_measurement_folders
    folders = find_measurement_folders(a.root, a.folder, a.subject)
    {"draw": cmd_draw, "process": cmd_process, "status": cmd_status}[a.phase](folders, a)


if __name__ == "__main__":
    sys.exit(main())
