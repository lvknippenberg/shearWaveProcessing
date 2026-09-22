"""Stage 4: test the phase-gradient estimator against the 34 hand-drawn wavefronts.

The comparison is a **projection**, not a direct one. A hand-drawn slope on an M-line measures
the *apparent* speed, which is the true speed divided by the cosine of the angle between the wave
and the line:

    predicted apparent speed = c_field / cos(theta_field - theta_line)

so the test exercises ``c`` and ``theta`` jointly. An estimate that gets the speed right and the
direction wrong fails it, which a speed-only comparison would not catch.

Two further checks come for free and are worth as much as the projection:

* **V2, displacement vs velocity.** Velocity is the time derivative of displacement, so at a fixed
  frequency it is multiplied by ``i*omega`` - a constant. ``grad U / U`` is therefore *unchanged*,
  and the phase estimator should return the same ``k`` from either field. Any disagreement is
  noise or a broken assumption, not physiology. The 1-D path could not make this check because its
  answer depended on which feature the eye or the tracker followed.
* **Direction relative to the wall.** A septal shear wave should travel roughly along the M-line
  the operator drew. A direction perpendicular to it is the signature of the failure mode that
  killed the structure tensor.

Nothing here is tuned: every estimator parameter was fixed on synthetic data before this ran
(V4's intent - with n = 15 a held-out split buys less than simply not tuning).

    python study/analysis/field_stage4.py [--freq 16] [--out study/logs/field_stage4.csv]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))

import numpy as np

VIEWS = {"disp bp10-150 gauss mean3": "displacement",
         "velocity bp15-90 gauss1.0 mean5": "velocity"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", default=str(_REPO / "study" / "logs" / "labelled_panels.json"))
    ap.add_argument("--out", default=str(_REPO / "study" / "logs" / "field_stage4.csv"))
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--config", default=str(_REPO / "configs" / "passive.yaml"))
    ap.add_argument("--freqs", default="13,16,20")
    ap.add_argument("--roi-mm", type=float, default=12.0)
    a = ap.parse_args()

    import swp.passive as P
    from swp.field.phase import aggregate_phase, dominant_frequency, phase_gradient_speed
    from swp.field.roi import roi_from_mline
    from swp.field.structure import Grid
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline

    panels = json.load(open(a.panels))
    freqs = [float(x) for x in a.freqs.split(",")]
    rows = []

    for i, c in enumerate(panels, 1):
        folder = f"{a.root}/{c['folder']}"
        print(f"[{i:2d}/{len(panels)}] {c['subject']} win{c['window']} {c['label']} "
              f"({c['part']})", flush=True)
        try:
            cfg, p = P._paths(folder, a.config)
            st, ws = P.read_windows(p["windows_json"])
            w = ws[c["window"]]
            acq = P.load_acq(folder, a.config)
            picks = json.load(open(Path(folder) / "output/swp_passive/manual_slopes.json"))
            ml_full = P._load_line(P._window_npz(p["mlines"], c["window"]), 250)
        except Exception as e:                                        # noqa: BLE001
            print(f"     skipped: {type(e).__name__} {e}", flush=True)
            continue

        # the M-line part the hand pick was drawn on: that is what we project onto
        if c["part"] == "full":
            ml = ml_full
        else:
            from passive_mline_split import split_line
            ml = mline_from_points(split_line(ml_full, 250)[c["part"]], 250)
        th_line = float(np.degrees(np.arctan2(ml.z[-1] - ml.z[0], ml.x[-1] - ml.x[0])))
        th_line = (th_line + 90) % 180 - 90

        roi = roi_from_mline(ml, a.roi_mm)
        mask = roi.mask(acq.x, acq.z)
        grid = Grid(dz=float(acq.dz), dx=float(acq.dx), dt=float(1.0 / acq.prf))
        i0 = P._frame_at_time(acq.t, w.t0 - 0.02)
        i1 = P._frame_at_time(acq.t, w.t1 + 0.02) + 1
        acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
        views = P._build_views(cfg, acq)

        for vname, quantity in VIEWS.items():
            vv = [v for v in views if vname in v[0]]
            if not vv:
                continue
            res = run_pipeline(acq_w, ml_full, vv[0][1], focus=None)
            fld = np.asarray(res.field, float)
            pk = picks.get(f"win{c['window']}|{c['part']}|{vname}")
            hand = abs(pk["speed_m_s"]) if pk else np.nan
            f_dom = dominant_frequency(fld, grid, (8.0, 40.0))
            for fh in list(freqs) + [f_dom]:
                est = phase_gradient_speed(fld, grid, f_hz=fh, window_space_mm=2.0)
                s = aggregate_phase(est, mask=mask, min_coherence=0.2, min_weight_pct=70)
                if not np.isfinite(s["speed"]):
                    continue
                d = (s["theta_deg"] - th_line + 90) % 180 - 90
                proj = s["speed"] / max(abs(np.cos(np.radians(d))), 1.0 / 8.0)
                rows.append(dict(
                    subject=c["subject"], window=c["window"], label=c["label"],
                    part=c["part"], quantity=quantity,
                    f_hz=round(float(fh), 2), is_dominant=abs(fh - f_dom) < 1e-6,
                    c_field=round(float(s["speed"]), 3),
                    theta_deg=round(float(s["theta_deg"]), 1),
                    mline_theta_deg=round(th_line, 1), dtheta_deg=round(float(d), 1),
                    projected=round(float(proj), 3), hand=round(float(hand), 3),
                    err_pct=(round(100.0 * (proj / hand - 1.0), 1)
                             if np.isfinite(hand) and hand > 0 else np.nan),
                    coherence=round(float(s["coherence"]), 3),
                    spread_deg=round(float(s["direction_spread_deg"]), 1),
                    frac_kept=round(float(s["frac_kept"]), 4)))
        if rows:
            with open(a.out, "w", newline="") as fh:
                wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wtr.writeheader()
                wtr.writerows(rows)
    print(f"\n-> {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
