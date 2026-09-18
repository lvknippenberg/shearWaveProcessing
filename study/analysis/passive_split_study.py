"""Full / left / right M-line analysis over a whole study.

``passive_mline_split.py`` runs the three M-line variants on ONE measurement folder. This drives it
over every folder that ``scripts/passive_study.py process`` has finished, then gathers the three
``split_<variant>/passive_speeds.json`` files of every folder into one table.

Each folder is run as its own subprocess: ``passive_mline_split`` monkey-patches ``swp.passive._paths``
while it works, and one folder failing must not take the study down with it.

Resumable: a folder whose ``split_speeds.txt`` is newer than its montage is skipped
(``--overwrite`` redoes it, ``--collect-only`` skips straight to the table).

Usage:
    python study/analysis/passive_split_study.py --root "Z:/raw_data"
    python study/analysis/passive_split_study.py --root "Z:/raw_data" --collect-only
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import numpy as np

VARIANTS = ("full", "left", "right")
SPLIT_SCRIPT = Path(__file__).with_name("passive_mline_split.py")
CMIN, CMAX = 1.0, 20.0                      # slant-stack search bounds (swp.passive.SPEED_CMAX)
CSV_OUT = _REPO / "study" / "logs" / "passive_split_speeds.csv"


def _name(f):
    return f"{f.parent.name}/{f.name}"


def _out(folder, config):
    from swp.passive import _paths

    return _paths(str(folder), config)[1]["outdir"]


def _split_state(folder, config):
    """'todo' | 'done' | 'not-processed' | 'no-line' — is the split up to date for this folder?

    Readiness is ``passive_study.status() == "processed"``, i.e. a montage at least as new as the
    general line: a folder that was processed in an earlier round and then SKIPPED at drawing still
    has that old montage, but its window lines were archived, and splitting it can only fail.

    ``no-line`` (nothing drawn, or nothing to draw) is kept apart from ``not-processed`` so that
    ``--watch`` waits only for folders actually on their way through ``process``: most of the tree
    is measurements that were never beamformed, and they must not hold the watch open.
    """
    from passive_study import status
    from swp.passive import _paths

    from swp.passive import read_windows

    _, p = _paths(str(folder), config)
    s = status(folder, config)
    if s != "processed":
        return "not-processed" if s == "drawn" else "no-line"
    # Per-event drawing can skip every event of a folder. Nothing is then processed, the montage is
    # left over from the general-line round, and an old split next to it would enter the table as if
    # it described the per-event lines.
    st, windows = read_windows(p["windows_json"])
    if (st or {}).get("window_mlines") and len(st.get("skipped") or []) >= len(windows):
        return "no-line"
    # Staleness is judged against the montage, the marker that `passive_study.py process` finished.
    # (Not passive_windows.json: the `label` pass rewrites that file without changing any window and
    # would otherwise force the whole study to be split again.)
    table = os.path.join(p["outdir"], "split_speeds.txt")
    if os.path.exists(table) and os.path.getmtime(table) >= os.path.getmtime(p["montage"]):
        return "done"
    return "todo"


def run_folder(folder, config):
    """Run the three variants on one folder in a subprocess. Returns True on success."""
    cmd = [sys.executable, "-u", str(SPLIT_SCRIPT), "--folder", str(folder), "--config", config]
    log = Path(_out(folder, config)) / "split_run.log"
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           env={**os.environ, "KERAS_BACKEND": "torch", "MPLBACKEND": "Agg"})
    return r.returncode == 0


def _timed(folder, config):
    t0 = time.perf_counter()
    return run_folder(folder, config), time.perf_counter() - t0


def collect(folders, config):
    """Gather every folder's three variants into one CSV. Returns the rows."""
    from swp.passive import _paths, read_windows

    rows = []
    for f in folders:
        _, p = _paths(str(f), config)
        st, _ = read_windows(p["windows_json"])
        for v in VARIANTS:
            path = os.path.join(p["outdir"], f"split_{v}", "passive_speeds.json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                speeds = json.load(fh)
            for r in speeds:
                rows.append(dict(
                    folder=_name(f), subject=f.parent.name, variant=v,
                    window=r["window"], label=r.get("label") or "",
                    t_peak_ms=round(r.get("t_peak_ms", float("nan")), 1),
                    view=r["view"], speed_m_s=round(r["speed_m_s"], 3),
                    abs_speed_m_s=round(abs(r["speed_m_s"]), 3),
                    semblance=round(r["semblance"], 4),
                    mline_length_mm=round(r.get("mline_length_mm", float("nan")), 1),
                    at_bound=int(abs(abs(r["speed_m_s"]) - CMIN) < 1e-6
                                 or abs(abs(r["speed_m_s"]) - CMAX) < 1e-6),
                    mline_source=(st or {}).get("key", {}).get("mline_source", ""),
                    per_event_lines=int(bool((st or {}).get("window_mlines"))),
                ))
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(CSV_OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def summarise(rows):
    """Per variant, and per variant x label: how many fits, how usable they are."""
    if not rows:
        print("no rows collected")
        return
    print(f"\n{len(rows)} fits from {len({r['folder'] for r in rows})} folder(s)"
          f", {len({(r['folder'], r['window']) for r in rows})} window(s)\n")

    def block(title, keyf):
        print(title)
        print(f"  {'group':<22s}{'n':>5s}{'median|c|':>11s}{'IQR':>16s}"
              f"{'at bound':>10s}{'med sem':>9s}{'len mm':>8s}")
        for k in sorted({keyf(r) for r in rows}):
            g = [r for r in rows if keyf(r) == k]
            ok = [r for r in g if not r["at_bound"]]
            c = np.array([r["abs_speed_m_s"] for r in ok]) if ok else np.array([np.nan])
            sem = np.array([r["semblance"] for r in g])
            ln = np.array([r["mline_length_mm"] for r in g])
            iqr = "%.1f-%.1f" % (np.percentile(c, 25), np.percentile(c, 75))
            bound = "%.0f%%" % (100 * sum(r["at_bound"] for r in g) / len(g))
            print(f"  {str(k):<22s}{len(g):>5d}{np.median(c):>11.2f}{iqr:>16s}"
                  f"{bound:>10s}{np.median(sem):>9.2f}{np.median(ln):>8.0f}")
        print()

    block("by M-line part", lambda r: r["variant"])
    block("by part x cardiac event", lambda r: f"{r['variant']} / {r['label'] or '?'}")

    # Do the three views agree? A real front gives the same speed AND the same direction in all
    # three recipes; noise does not. Spread is judged relatively as well as absolutely, because
    # 1 m/s means something very different at 2 m/s than at 8 m/s.
    print("view agreement per part  (over the 3 views of a window, non-bound fits only)")
    print(f"  {'part':<10s}{'windows':>9s}{'all 3 free':>12s}{'same sign':>11s}"
          f"{'spread<1 m/s':>14s}{'spread<25%':>12s}{'usable':>8s}")
    for v in VARIANTS:
        keys = {(r["folder"], r["window"]) for r in rows if r["variant"] == v}
        full3 = tight = rel = signed = usable = 0
        for k in keys:
            g = [r for r in rows if r["variant"] == v and (r["folder"], r["window"]) == k
                 and not r["at_bound"]]
            if len(g) < 3:
                continue
            full3 += 1
            c = np.array([r["abs_speed_m_s"] for r in g])
            s = np.array([np.sign(r["speed_m_s"]) for r in g])
            one_way = abs(s.sum()) == len(s)
            spread_rel = (c.max() - c.min()) / np.median(c)
            tight += int(c.max() - c.min() < 1.0)
            rel += int(spread_rel < 0.25)
            signed += int(one_way)
            usable += int(one_way and spread_rel < 0.25)      # one direction, consistent speed
        print(f"  {v:<10s}{len(keys):>9d}{full3:>12d}{signed:>11d}{tight:>14d}{rel:>12d}"
              f"{usable:>8d}")
    print("  'usable' = all 3 views free of the bounds, agreeing on direction, spread < 25%")
    print(f"\n-> {CSV_OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--folder", action="append", default=[])
    ap.add_argument("--subject", default=None)
    ap.add_argument("--config", default=str(_REPO / "configs" / "passive.yaml"))
    ap.add_argument("--overwrite", action="store_true", help="redo folders already split")
    ap.add_argument("--collect-only", action="store_true", help="only rebuild the table")
    ap.add_argument("--watch", action="store_true",
                    help="keep polling for folders that finish processing")
    ap.add_argument("--jobs", type=int, default=1,
                    help="folders to split in parallel (default 1; the work is single-threaded)")
    a = ap.parse_args()
    if not a.root and not a.folder:
        ap.error("give --root and/or --folder")

    from process_raw_data import find_measurement_folders
    folders = find_measurement_folders(a.root, a.folder, a.subject)

    done, failed = [], []
    if not a.collect_only:
        t_start = time.perf_counter()
        while True:
            states = {f: _split_state(f, a.config) for f in folders}
            todo = [f for f, s in states.items() if s == "todo" and f not in failed]
            if a.overwrite:
                todo = [f for f, s in states.items()
                        if s in ("todo", "done") and f not in failed and f not in done]
            if not todo:
                pending = [f for f, s in states.items() if s == "not-processed"]
                if a.watch and pending:
                    time.sleep(60)
                    continue
                break
            print(f"\n===== splitting {len(todo)} folder(s), {a.jobs} at a time =====", flush=True)
            # One worker process per folder. Each writes only inside its own output tree and the
            # work is single-threaded numpy, so folders parallelise cleanly; a worker holds ~1 GB
            # of buffer-4 IQ while it runs.
            with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
                futures = {ex.submit(_timed, f, a.config): f for f in todo}
                for k, fut in enumerate(cf.as_completed(futures), 1):
                    f = futures[fut]
                    try:
                        ok, dt = fut.result()
                    except Exception:                          # noqa: BLE001
                        ok, dt = False, 0.0
                        print(f"  FAILED {_name(f)}:\n{traceback.format_exc()}", flush=True)
                    (done if ok else failed).append(f)
                    print(f"  [{k}/{len(todo)}] {'ok  ' if ok else 'FAIL'} "
                          f"{_name(f)}  {dt:.0f} s", flush=True)
        print(f"\nsplit {len(done)} folder(s) in {(time.perf_counter() - t_start) / 60:.1f} min; "
              f"{len(failed)} failed")
        for f in failed:
            print(f"  FAILED {_name(f)}")

    have = [f for f in folders
            if os.path.exists(os.path.join(_out(f, a.config), "split_full", "passive_speeds.json"))]
    summarise(collect(have, a.config))


if __name__ == "__main__":
    sys.exit(main())
