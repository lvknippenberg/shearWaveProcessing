"""Clutter filter wave imaging (CFWI) on the 15 hand-labelled passive windows.

CFWI (Salles 2019; Espeland 2024, the PLAX-view clinical study) is the literature method built for
exactly this problem - a valve-closure wave on a moving wall. For every window of
``study/logs/labelled_panels.json`` this builds the CFWI panel on the same M-line and window as the
cached displacement / velocity panels (``score_panels.py --prepare``), fits the same signed slant
stack as the passive path, and compares with the hand-drawn speeds.

CFWI recipe: IQ slow-time high-pass at 2 cm/s (Espeland) -> envelope -> temporal derivative
(``estimators.cfwi``), then Gaussian 1.0 x 2.0 mm + moving mean 3 (Salles' spatio-temporal
smoothing step), no directional filter, 5 offsets x 0.5 mm - the passive view-A geometry.

    python study/analysis/cfwi_benchmark.py            # builds study/analysis/panel_cache/*_cfwi.npz
    python study/analysis/cfwi_benchmark.py --report   # table + montage from the cache only
-> study/logs/cfwi_benchmark.csv, study/montages/cfwi_benchmark.png
"""
from __future__ import annotations

import argparse
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

from swp import paths as P                                    # noqa: E402

CACHE = _REPO / "study" / "analysis" / "panel_cache"
CUTOFF_V = 0.02
EDGE = 10   # frames (~11 ms) dropped at both window ends: the slow-time high-pass leaves an
            # end transient in the envelope derivative that otherwise dominates the fit and scale


def cfwi_cfg(base):
    from swp.viz.pipeline import Step
    return dataclasses.replace(
        base, estimator="cfwi", estimator_params={"cutoff_velocity": CUTOFF_V}, quantity="velocity",
        mode="frame_to_frame", iq_filters=[], directional=False, mline_offsets=5,
        mline_offset_step_m=0.5e-3,
        field_filters=[Step("spatial_smooth", {"sigma_z_m": 1.0e-3, "sigma_x_m": 2.0e-3}),
                       Step("temporal_moving_mean", {"window": 3})])


def build(panels, config):
    import swp.passive as SP
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline
    from passive_mline_split import split_line
    CACHE.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(panels, 1):
        out = CACHE / f"{c['subject']}_win{c['window']}_{c['part']}_cfwi.npz"
        if out.exists():
            continue
        folder = f"{P.RAW_DATA}/{c['folder']}"
        print(f"[{i}/{len(panels)}] {c['subject']} win{c['window']} {c['part']}", flush=True)
        try:
            cfg, p = SP._paths(folder, config)
            st, ws = SP.read_windows(p["windows_json"])
            w = ws[c["window"]]
            acq = SP.load_acq(folder, config)
            n = cfg["mline"].get("n_samples", 250)
            ml_full = SP._load_line(SP._window_npz(p["mlines"], c["window"]), n)
            ml = (ml_full if c["part"] == "full"
                  else mline_from_points(split_line(ml_full, n)[c["part"]], n))
            i0 = SP._frame_at_time(acq.t, w.t0 - 0.02)
            i1 = SP._frame_at_time(acq.t, w.t1 + 0.02) + 1
            acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
            base = SP._build_views(cfg, acq)[0][1]
            res = run_pipeline(acq_w, ml, cfwi_cfg(base), focus=None)
            np.savez_compressed(out, data=res.st.data, r=res.st.r, t=res.st.t, label=str(w.label or "?"))
            del acq, acq_w
        except Exception as e:                                    # noqa: BLE001
            print(f"     FAILED: {type(e).__name__} {e}", flush=True)


def report(panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from swp.provenance import stamp_text
    from swp.viz.metrics import slant_stack_speed
    from swp.viz.speed.spacetime import SpaceTime
    conf = {}
    cp = _REPO / "study/logs/panel_confidence.csv"
    if cp.exists():
        conf = {(r["subject"], r["window"], r["part"], r["quantity"]): r["meaning"]
                for r in csv.DictReader(open(cp))}
    rows, figrows = [], []
    for c in panels:
        key = f"{c['subject']}_win{c['window']}_{c['part']}"
        fc = CACHE / f"{key}_cfwi.npz"
        if not fc.exists():
            continue
        z = np.load(fc, allow_pickle=True)
        st = SpaceTime(z["data"][EDGE:-EDGE], z["r"], z["t"][EDGE:-EDGE], "velocity")
        sem, cc = slant_stack_speed(st, None, cmin=1.0, cmax=20.0, remove_flat=False)
        hand = {}
        for q in ("displacement", "velocity"):
            f = CACHE / f"{key}_{q}.npz"
            if f.exists():
                hand[q] = float(np.load(f, allow_pickle=True)["hand"])
        rows.append(dict(subject=c["subject"], window=c["window"], part=c["part"], label=c["label"],
                         cfwi_speed=cc, cfwi_semblance=sem,
                         hand_disp=hand.get("displacement", np.nan), hand_vel=hand.get("velocity", np.nan),
                         conf_disp=conf.get((c["subject"], str(c["window"]), c["part"], "displacement"), "?"),
                         conf_vel=conf.get((c["subject"], str(c["window"]), c["part"], "velocity"), "?")))
        figrows.append((key, c, st))
    out = _REPO / "study/logs/cfwi_benchmark.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(stamp_text(config=dict(cutoff_velocity=CUTOFF_V, smoothing="gauss1x2mm+mean3")))
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)
    print(f"\n{'window':<28}{'label':>6}{'CFWI':>8}{'hand disp':>11}{'hand vel':>10}   confidence d/v")
    for r in rows:
        print(f"{r['subject']+' w'+str(r['window'])+' '+r['part']:<28}{r['label']:>6}{r['cfwi_speed']:8.2f}"
              f"{r['hand_disp']:11.2f}{r['hand_vel']:10.2f}   {r['conf_disp']}/{r['conf_vel']}")
    for q, conf_key in (("hand_disp", "conf_disp"), ("hand_vel", "conf_vel")):
        for lev in ("all", "clear"):
            sel = [r for r in rows if np.isfinite(r[q]) and (lev == "all" or r[conf_key] == lev)]
            if not sel:
                continue
            e = np.array([r["cfwi_speed"] for r in sel]); h = np.array([r[q] for r in sel])
            ok = (np.sign(e) == np.sign(h)) & (np.abs(np.abs(e) / np.abs(h) - 1) <= 0.25)
            print(f"CFWI vs {q:<9} ({lev:<5} n={len(sel):2d}): direction agrees {np.mean(np.sign(e) == np.sign(h)):4.0%}, "
                  f"within 25% {ok.mean():4.0%}, median |c|/hand {np.median(np.abs(e) / np.abs(h)):.2f}, "
                  f"railed {np.mean(np.abs(e) >= 19.8):3.0%}")

    # montage: displacement | velocity | CFWI, M-mode orientation (x = time, y = along line)
    fig, axes = plt.subplots(len(figrows), 3, figsize=(11, 2.1 * len(figrows)), squeeze=False)
    for i, (key, c, st) in enumerate(figrows):
        for j, q in enumerate(("displacement", "velocity", "cfwi")):
            f = CACHE / f"{key}_{q}.npz"
            if not f.exists():
                axes[i, j].axis("off"); continue
            z = np.load(f, allow_pickle=True)
            e = EDGE if q == "cfwi" else 0
            d = z["data"][e:len(z["data"]) - e]; tt = z["t"][e:len(z["t"]) - e]
            clim = np.percentile(np.abs(d), 99) or 1
            axes[i, j].imshow(d.T, aspect="auto", cmap="RdBu_r", vmin=-clim, vmax=clim, origin="lower",
                              extent=(tt[0] * 1e3, tt[-1] * 1e3, z["r"][0] * 1e3, z["r"][-1] * 1e3))
            if q != "cfwi" and "points" in z.files and np.isfinite(float(z["hand"])):
                (t1, r1), (t2, r2) = np.asarray(z["points"], float)
                axes[i, j].plot([t1, t2], [r1, r2], "o-", color="lime", ms=3, lw=1.2)
            axes[i, j].set_title(f"{c['subject'][-3:]} w{c['window']} {c['label']} {c['part']} - {q}", fontsize=7)
            axes[i, j].tick_params(labelsize=6)
    fig.suptitle("Hand-labelled passive windows: displacement | velocity (hand line in green) | CFWI "
                 "(2 cm/s clutter filter, envelope derivative)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fp = _REPO / "study/montages/cfwi_benchmark.png"
    fig.savefig(fp, dpi=100)
    print("wrote", fp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(_REPO / "configs" / "passive.yaml"))
    ap.add_argument("--report", action="store_true", help="skip building; report from the cache")
    a = ap.parse_args()
    panels = json.load(open(_REPO / "study/logs/labelled_panels.json"))
    if not a.report:
        build(panels, a.config)
    report(panels)


if __name__ == "__main__":
    main()
