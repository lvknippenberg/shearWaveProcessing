"""Undo an accidental re-detection of passive windows (2026-09-24, C000000001-C000000004).

What happened: ``passive_study.py reprocess`` called ``process_single_line``, whose window cache is
keyed on the detection settings. The stored windows predate ``detect_mode`` in that key (they were
detected in energy mode), so the cache was rejected: the windows were re-detected in phase mode,
the hand-drawn per-event lines were moved to ``mlines/archive_20260924_*`` and the general line
was written in their place. ``process_single_line`` now refuses to do that (StaleWindowsError) and
``reprocess`` no longer detects.

The repair, per folder:
1. Re-detect the ORIGINAL windows: energy mode with the v1 config (``configs/passive_v1.yaml``;
   detection overview on displacement), and label them from the trigger log.
2. Check them against the record written before the accident
   (``swp_passive/v1_displacement/passive_speeds.json``: t_peak and label of every window). Any
   mismatch > 1 ms or a different label -> the folder is NOT touched.
3. Rebuild ``window_mlines`` (the phase-matched buffer-1 frame of each window, as
   ``draw_event_mlines`` records it) and the old-style cache key.
4. ``--apply``: move the re-detected state aside (``*.redetected_20260924`` /
   ``mlines/redetected_20260924/``), move the archived hand-drawn lines back, write the restored
   ``passive_windows.json``.

    python scripts/restore_redetected_windows.py            # dry run: detect + compare only
    python scripts/restore_redetected_windows.py --apply
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import glob
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import yaml                                                      # noqa: E402

from swp import paths as P                                       # noqa: E402

CONFIG_V1 = str(_ROOT / "configs" / "passive_v1.yaml")
SUBJECTS = ("C000000001", "C000000002", "C000000003", "C000000004")
TAG = "redetected_20260924"


def folders():
    out = []
    for s in SUBJECTS:
        for arch in glob.glob(f"{P.RAW_DATA}/{s}/*/output/mlines/archive_20260924_*"):
            out.append((os.path.dirname(os.path.dirname(os.path.dirname(arch))), arch))
    return out


def v1_record(p):
    s = json.load(open(os.path.join(p["outdir"], "v1_displacement", "passive_speeds.json")))
    rows = s if isinstance(s, list) else s.get("speeds", [])
    rec = {}
    for r in rows:
        rec.setdefault(int(r["window"]), (r["label"], float(r["t_peak_ms"])))
    return rec


def rebuild(folder):
    import swp.passive as SP
    cfg_path = CONFIG_V1
    cfg, p = SP._paths(folder, cfg_path)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("detect", {})["mode"] = "energy"
    n = cfg["mline"].get("n_samples", 250)
    gen = SP._load_line(p["general"], n)
    acq = SP.load_acq(folder, cfg_path)
    windows = SP.detect_windows(acq, gen, cfg, p["outdir"], 100.0, 4, 2, folder=folder)
    key = SP._detect_key(gen, cfg, 100.0, 4, 2)
    key.pop("detect_mode")                               # the stored keys predate this field
    key["mline_source"] = SP._mline_source(p)
    st = dict(key=key, windows=[dataclasses.asdict(w) for w in windows])
    SP.label_windows(folder, st)
    return p, st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import swp.passive as SP
    for folder, arch in folders():
        name = os.path.relpath(folder, P.RAW_DATA)
        print(f"\n=== {name}", flush=True)
        p, st = rebuild(folder)
        rec = v1_record(p)
        ws = st["windows"]
        ok = len(ws) == len(rec)
        for i, w in enumerate(ws):
            lab, tp = rec.get(i, ("-", float("nan")))
            good = abs(w["t_peak"] * 1e3 - tp) <= 1.0 and w.get("label") == lab
            ok &= good
            print(f"  win{i}: re-detected {w.get('label')} @ {w['t_peak'] * 1e3:.1f} ms | v1 record {lab} @ {tp:.1f} ms "
                  f"{'OK' if good else 'MISMATCH'}")
        archived = sorted(glob.glob(os.path.join(arch, "passive_win*_mline.npz")))
        idx = sorted(int(os.path.basename(f)[len("passive_win"):].split("_")[0]) for f in archived)
        ok &= idx == list(range(len(ws)))
        print(f"  archived lines for windows {idx}")
        if not ok:
            print("  -> NOT restored (mismatch)")
            continue
        wins = [SP.BurstWindow(**w) for w in ws]
        frames = SP.event_bmode_frames(folder, wins, buffer=1)
        wm = {str(i): dict(buffer=1, **frames[i], from_general=False) for i in idx}
        st.update(window_mlines=wm, drawn=idx, skipped=[], from_general=[])
        if not a.apply:
            print("  dry run: would restore " + ", ".join(f"win{i} (b1 frame {frames[i]['frame']})" for i in idx))
            continue
        # move the re-detected state aside
        side = os.path.join(p["mlines"], TAG)
        os.makedirs(side, exist_ok=True)
        for f in glob.glob(os.path.join(p["mlines"], "passive_win*_mline.*")):
            shutil.move(f, os.path.join(side, os.path.basename(f)))
        for f in (p["windows_json"], p["windows_json"] + ".tmp"):
            if os.path.exists(f):
                shutil.move(f, f"{f}.{TAG}")
        # the hand-drawn lines back
        for f in glob.glob(os.path.join(arch, "*")):
            shutil.move(f, os.path.join(p["mlines"], os.path.basename(f)))
        os.rmdir(arch)
        SP._write_windows(p["windows_json"], st)
        print(f"  restored: {len(idx)} hand-drawn lines, windows.json rebuilt (re-detected state in {TAG})")


if __name__ == "__main__":
    main()
