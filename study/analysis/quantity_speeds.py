"""Passive speed by quantity (displacement / velocity / acceleration) on the 15 labelled windows.

Runs ``swp.passive._speeds_by_quantity`` - view A's recipe re-run with each quantity, same slant
stack - on every window of ``study/logs/labelled_panels.json``, and sets the result next to the
hand-drawn displacement and velocity speeds. The literature reports velocity (Keijzer 2019/2020)
or acceleration (Petrescu, Santos, Espeland), never cumulative displacement, and the wave is
dispersive, so the three differ systematically.

    python study/analysis/quantity_speeds.py  -> study/logs/quantity_speeds.csv + stdout
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))
os.environ.setdefault("KERAS_BACKEND", "torch")

from swp import paths as P                                     # noqa: E402

CACHE = _REPO / "study" / "analysis" / "panel_cache"


def main():
    import swp.passive as SP
    from swp.viz.mline import mline_from_points
    from swp.provenance import stamp_text
    from passive_mline_split import split_line
    panels = json.load(open(_REPO / "study/logs/labelled_panels.json"))
    config = str(_REPO / "configs" / "passive.yaml")
    rows = []
    for c in panels:
        folder = f"{P.RAW_DATA}/{c['folder']}"
        print(f"{c['subject']} win{c['window']} {c['part']}", flush=True)
        try:
            cfg, p = SP._paths(folder, config)
            _, ws = SP.read_windows(p["windows_json"])
            w = ws[c["window"]]
            acq = SP.load_acq(folder, config)
            n = cfg["mline"].get("n_samples", 250)
            ml_full = SP._load_line(SP._window_npz(p["mlines"], c["window"]), n)
            ml = ml_full if c["part"] == "full" else mline_from_points(split_line(ml_full, n)[c["part"]], n)
            i0 = SP._frame_at_time(acq.t, w.t0 - 0.02)
            i1 = SP._frame_at_time(acq.t, w.t1 + 0.02) + 1
            acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
            sp = {d["quantity"]: d["speed_m_s"] for d in
                  SP._speeds_by_quantity(acq_w, ml, SP._build_views(cfg, acq)[0], c["window"], w)}
            del acq, acq_w
        except Exception as e:                                       # noqa: BLE001
            print(f"   FAILED {type(e).__name__}: {e}")
            continue
        hand = {q: float(np.load(CACHE / f"{c['subject']}_win{c['window']}_{c['part']}_{q}.npz")["hand"])
                for q in ("displacement", "velocity")
                if (CACHE / f"{c['subject']}_win{c['window']}_{c['part']}_{q}.npz").exists()}
        rows.append(dict(subject=c["subject"], window=c["window"], part=c["part"], label=c["label"],
                         disp=sp.get("displacement", np.nan), vel=sp.get("velocity", np.nan),
                         acc=sp.get("acceleration", np.nan),
                         hand_disp=hand.get("displacement", np.nan), hand_vel=hand.get("velocity", np.nan)))
    out = _REPO / "study/logs/quantity_speeds.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(stamp_text(config={"recipe": "passive view A, quantity swapped", "speed": "slant stack 1-20"}))
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("wrote", out)
    a = {k: np.abs(np.array([r[k] for r in rows], float)) for k in ("disp", "vel", "acc", "hand_disp", "hand_vel")}
    ok = np.all([np.isfinite(a[k]) & (a[k] < 19.8) for k in ("disp", "vel", "acc")], axis=0)
    print(f"\n{ok.sum()} of {len(rows)} windows with no railed fit")
    print(f"median |speed|: displacement {np.median(a['disp'][ok]):.2f}, velocity {np.median(a['vel'][ok]):.2f}, "
          f"acceleration {np.median(a['acc'][ok]):.2f} m/s")
    print(f"ordering disp <= vel <= acc holds in {np.mean(((a['disp'] <= a['vel']) & (a['vel'] <= a['acc']))[ok]):.0%}")
    print(f"hand medians: displacement {np.nanmedian(a['hand_disp']):.2f}, velocity {np.nanmedian(a['hand_vel']):.2f} m/s")


if __name__ == "__main__":
    main()
