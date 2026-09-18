"""Passive SWE over a whole study: one M-line per folder on a B-mode frame, then unattended.

Most buffer-4 (diverging-wave) frames do not show the septum clearly, and it moves too much over
the cardiac cycle for a line drawn on a cine. So each folder gets ONE M-line, drawn on the
**widebeam B-mode (buffer 1) frame nearest an R-peak**. Buffer 4 is R-peak triggered (its frame 0
is on an R-peak in all 44 folders) but buffer 1 is not, so the frame is picked from the trigger
log in the runtime .mat (``swp.acquisition.triggerlog``; always within ~5 ms of an R-peak here).
That frame shows the anatomy at buffer-4 frame 0's cardiac phase, and the line (in metres) maps
onto the buffer-4 grid as is. It is used for burst detection and for every detected window.
``--buffer`` / ``--frame`` override the source (e.g. ``--buffer 3 --frame 0``).

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
    python scripts/passive_study.py draw-events --folder <folder>   # one line per detected event
    python scripts/passive_study.py label   --root "Z:/raw_data"      # MVC / AVC / AK per window
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


def status(folder, config, buffer=1):
    """'missing' | 'todo' | 'skipped' | 'drawn' | 'no-bursts' | 'processed'."""
    from swp.passive import _paths, bmode_file, read_windows

    _, p = _paths(str(folder), config)
    if not (os.path.exists(p["iq"]) and os.path.exists(os.path.join(p["output"], bmode_file(buffer)))):
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
    from swp.passive import _paths, draw_bmode_mline

    todo = []
    for f in folders:
        s = status(f, a.config)
        if s == "todo" or (a.redraw and s != "missing"):
            todo.append(f)
        elif a.retry_skipped and s == "skipped":
            os.remove(_paths(str(f), a.config)[1]["windows_json"])
            todo.append(f)
    print(f"=== passive M-lines (buffer {a.buffer}, frame {a.frame}): {len(todo)} of {len(folders)} to draw ===")
    print("Left-click points along the septum (any order), drag to adjust, right-click to delete,"
          "\nENTER to accept. Close the window without clicking to SKIP the folder. Ctrl-C stops;"
          "\neverything drawn so far is kept and the next run resumes.\n")
    t_start = time.perf_counter()
    for k, f in enumerate(todo):
        label = f"[{k + 1}/{len(todo)}] "
        print(f"\n{label}{_name(f)}", flush=True)
        try:
            draw_bmode_mline(str(f), a.config, buffer=a.buffer,
                             frame=a.frame if a.frame == "rpeak" else int(a.frame), label=label)
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


def cmd_draw_events(folders, a):
    """Per-event M-lines on the phase-matched B-mode frames, then reprocess with them.

    ``--defer-process`` skips the reprocessing so a whole study can be drawn in one sitting - a
    folder takes ~2.5 min to reprocess, which otherwise stalls the prompts between folders. Run
    the ``reprocess`` phase afterwards.
    """
    from swp.passive import draw_event_mlines, process_single_line

    todo = [f for f in folders if status(f, a.config) == "processed"]
    print(f"=== per-event M-lines (buffer {a.buffer}): {len(todo)} processed folder(s) ===")
    if a.defer_process:
        print("  --defer-process: run `passive_study.py reprocess` when the drawing is done.")
    for k, f in enumerate(todo):
        label = f"[{k + 1}/{len(todo)}] "
        print(f"\n{label}{_name(f)}", flush=True)
        try:
            draw_event_mlines(str(f), a.config, buffer=a.buffer, label=label)
            if not a.defer_process:
                process_single_line(str(f), a.config)
        except KeyboardInterrupt:
            print("\ninterrupted - rerun to continue.")
            break
        except Exception:                                 # noqa: BLE001
            print(f"  FAILED:\n{traceback.format_exc()}")


def cmd_reprocess(folders, a):
    """Reprocess folders regardless of status - for per-event lines drawn with --defer-process.

    ``status()`` compares the montage against the *general* line, so it cannot see that
    ``draw-events`` has replaced the per-window lines; this forces the run.
    """
    from swp.passive import _paths, process_single_line, read_windows

    todo = []
    for f in folders:
        if status(f, a.config) not in ("processed", "drawn"):
            continue
        st, _ = read_windows(_paths(str(f), a.config)[1]["windows_json"])
        if a.only_event_lines and not (st or {}).get("window_mlines"):
            continue
        todo.append(f)
    print(f"=== reprocess {len(todo)} folder(s) ===")
    failed, t_start = [], time.perf_counter()
    for k, f in enumerate(todo):
        print(f"\n[{k + 1}/{len(todo)}] {_name(f)}", flush=True)
        t0 = time.perf_counter()
        try:
            process_single_line(str(f), a.config)
            print(f"  {time.perf_counter() - t0:.0f} s", flush=True)
        except Exception:                                 # noqa: BLE001
            failed.append(f)
            print(f"  FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"\nreprocessed {len(todo) - len(failed)}/{len(todo)} in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")
    for f in failed:
        print(f"  FAILED {_name(f)}")


def cmd_label(folders, a):
    """Label the detected windows of every folder (MVC / AVC / AK / other) from the trigger log.

    Updates ``passive_windows.json`` (labels + timing) and adds the label to the rows of
    ``passive_speeds.json``; montages pick the labels up the next time a folder is processed.
    Writes a study table ``passive_window_labels.csv`` into the logs folder.
    """
    import csv
    import json
    from swp.passive import _paths, _write_windows, label_windows, read_windows

    rows, counts = [], {}
    for f in folders:
        _, p = _paths(str(f), a.config)
        st, _ = read_windows(p["windows_json"])
        if st is None or not st.get("windows"):
            continue
        if label_windows(str(f), st):
            _write_windows(p["windows_json"], st)
        phases = st.get("window_phases") or [{}] * len(st["windows"])
        spath = os.path.join(p["outdir"], "passive_speeds.json")
        if os.path.exists(spath):
            with open(spath) as fh:
                speeds = json.load(fh)
            for r in speeds:
                r["label"] = st["windows"][r["window"]].get("label", "")
            with open(spath, "w") as fh:
                json.dump(speeds, fh, indent=1)
        tags = []
        for i, (w, ph) in enumerate(zip(st["windows"], phases)):
            lab = w.get("label") or "?"
            counts[lab] = counts.get(lab, 0) + 1
            tags.append(f"{lab}@{w['t_peak'] * 1e3:.0f}")
            rows.append(dict(folder=_name(f), window=i, t_peak_ms=round(w["t_peak"] * 1e3, 1),
                             label=lab, **{k: ph.get(k) for k in
                                           ("phase_ms", "to_next_r_ms", "rr_ms", "hr_bpm", "qs2_ms",
                                            "next_r_logged")}))
        print(f"  {_name(f)[:10]}  " + "  ".join(tags))
    out = _ROOT / "study" / "logs" / "passive_window_labels.csv"
    if rows:
        with open(out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print("\n  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + f"  -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase",
                   choices=["draw", "process", "status", "draw-events", "reprocess", "label"])
    p.add_argument("--root", default=None)
    p.add_argument("--folder", action="append", default=[])
    p.add_argument("--subject", default=None)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--buffer", type=int, default=1, help="draw: B-mode buffer to draw on (default 1)")
    p.add_argument("--frame", default="rpeak",
                   help="draw: frame index, or 'rpeak' (default) = the frame nearest an R-peak")
    p.add_argument("--redraw", action="store_true", help="draw: redo every line")
    p.add_argument("--retry-skipped", action="store_true", help="draw: ask again for skipped folders")
    p.add_argument("--watch", action="store_true",
                   help="process: keep polling for folders that finish drawing")
    p.add_argument("--defer-process", action="store_true",
                   help="draw-events: do not reprocess between folders (run `reprocess` after)")
    p.add_argument("--only-event-lines", action="store_true",
                   help="reprocess: only folders that have per-event M-lines")
    a = p.parse_args()
    if not a.root and not a.folder:
        p.error("give --root and/or --folder")

    from process_raw_data import find_measurement_folders
    folders = find_measurement_folders(a.root, a.folder, a.subject)
    {"draw": cmd_draw, "process": cmd_process, "status": cmd_status,
     "draw-events": cmd_draw_events, "reprocess": cmd_reprocess,
     "label": cmd_label}[a.phase](folders, a)


if __name__ == "__main__":
    sys.exit(main())
