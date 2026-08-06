"""Render the two summary montages from an existing passive search:

  per_window_best.png  -- each burst window with ITS OWN top recipe (honest best-achievable view)
  consensus_perdir.png -- one global recipe (velocity/bp10_150/gauss1.0/mean3/off5x0.5mm) applied to
                          every window with the propagation DIRECTION chosen per window (r0 at the
                          better end), the physically correct passive consensus.

Fast: reloads the stream once and estimates only rank-0 Loupas per window (no full sweep).
Reads each window's own best recipe from results_win*.csv.
"""
from __future__ import annotations
import csv, glob, os, sys
from dataclasses import replace
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from swp.viz import runconfig as rc
from swp.viz.io import load_acquisition
from swp.viz.estimators import ESTIMATORS
from swp.viz.filters import svd_clutter
from swp.viz.speed.spacetime import SpaceTime
from swp.viz.viz import spacetime_montage
from swp.viz.pipeline import Step, run_pipeline
from swp.mline.select import detect_line_bursts
from swp.passive import _stride_acq, _frame_at_time, _middle_frame, _ensure_mline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search_passive as S


def best_recipe(csv_path):
    with open(csv_path) as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: -float(r["coherence"]))
    b = rows[0]
    return dict(quantity=b["quantity"], motion=b["motion"], spatial=b["spatial"],
                temporal=b["temporal"], direction=b["direction"],
                offsets=int(b["offsets"]), step=float(b["step"]),
                coherence=float(b["coherence"]), speed=float(b["speed"]))


def base_for_rank(acq_w, ranks):
    est = ESTIMATORS["loupas"]
    ekw = dict(dz=acq_w.dz, dx=acq_w.dx, c=acq_w.c, f_demod=acq_w.f_demod, prf=acq_w.prf,
               mode="frame_to_frame")
    out = {}
    for r in ranks:
        iqf = svd_clutter(acq_w.iq, n_remove=r) if r > 0 else acq_w.iq
        out[r] = est(iqf, **ekw)
    return out


def main():
    folder = r"D:\Luuk van Knippenberg\Claude\invivo_sw"
    cfg = rc.load_config(os.path.join(_ROOT, "configs", "passive.yaml"))
    output_dir = os.path.join(folder, "output")
    cfg["data"]["root"] = output_dir
    iq_path = rc.hdf5_path(cfg, 0)
    bmode_path = os.path.join(output_dir, os.path.basename(iq_path))
    mlines_dir = os.path.join(output_dir, "mlines")
    outdir = os.path.join(rc.outdir(cfg, output_dir), "search")

    acq = load_acquisition(iq_path)
    prf, dz, dx = acq.prf, acq.dz, acq.dx
    base_cfg = rc.build_pipeline_config(cfg, acq=acq)

    gen = _ensure_mline(os.path.join(mlines_dir, "passive_general_mline.npz"), bmode_path,
                        _middle_frame(bmode_path), "general", n_samples=cfg["mline"]["n_samples"])
    band = cfg["detect"]["band"]
    ov_cfg = replace(base_cfg, directional=False,
                     field_filters=[Step("temporal_bandpass", dict(f_lo=band[0], f_hi=band[1]))]
                     + [s for s in base_cfg.field_filters if s.name != "temporal_bandpass"])
    ov = run_pipeline(_stride_acq(acq, 2), gen, ov_cfg, focus=None)
    windows, _ = detect_line_bursts(np.asarray(ov.st.data).T, np.asarray(ov.st.t),
                                    window_ms=100.0, max_events=4)

    CONS = dict(quantity="velocity", motion="bp10_150", spatial="gauss1.0", temporal="mean3",
                offsets=5, step=0.5)
    best_panels, best_titles, cons_panels, cons_titles = [], [], [], []
    pad_s = 0.02
    for i, w in enumerate(windows):
        frame = _frame_at_time(acq.t, w.t_peak)
        mline = _ensure_mline(os.path.join(mlines_dir, f"passive_win{i}_mline.npz"), bmode_path,
                              frame, f"win{i}", n_samples=cfg["mline"]["n_samples"])
        i0 = _frame_at_time(acq.t, w.t0 - pad_s); i1 = _frame_at_time(acq.t, w.t1 + pad_s) + 1
        acq_w = replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
        tf = np.asarray(acq_w.t, float)

        rec = best_recipe(os.path.join(outdir, f"results_win{i}.csv"))
        rank = int(rec["motion"][3:]) if rec["motion"].startswith("svd") else 0
        base = base_for_rank(acq_w, {0, rank})

        st, r0 = S.build_spacetime_for(acq_w, mline, prf, dz, dx, tf, base, rec)
        best_panels.append(S._Panel(st, r0, f"c={rec['speed']:.1f} m/s"))
        best_titles.append(f"win{i} @ {w.t_peak*1e3:.0f}ms  pc={rec['coherence']:.2f}\n"
                           f"{S.recipe_str(rec)}")

        # consensus recipe, direction chosen per window
        cbest = None
        for direction in ("leftward", "rightward"):
            srec = dict(CONS, direction=direction)
            st_c, r0_c = S.build_spacetime_for(acq_w, mline, prf, dz, dx, tf, base, srec)
            pc, c = S.pc_fast(st_c.data, mline.r, float(st_c.t[1] - st_c.t[0]), r0_c)
            if cbest is None or pc > cbest[0]:
                cbest = (pc, c, st_c, r0_c, direction)
        pc, c, st_c, r0_c, direction = cbest
        cons_panels.append(S._Panel(st_c, r0_c, f"c={c:.1f} m/s"))
        cons_titles.append(f"win{i} @ {w.t_peak*1e3:.0f}ms  pc={pc:.2f}  ({direction})")
        print(f"win{i}: best pc={rec['coherence']:.3f} [{S.recipe_str(rec)}]  | "
              f"consensus pc={pc:.3f} dir={direction} c={c:.1f}")

    spacetime_montage(best_panels, os.path.join(outdir, "per_window_best.png"),
                      ncols=min(4, len(best_panels)),
                      suptitle="Passive SWE -- each window with ITS OWN best recipe (max passive_coherence)",
                      panel_titles=best_titles)
    spacetime_montage(cons_panels, os.path.join(outdir, "consensus_perdir.png"),
                      ncols=min(4, len(cons_panels)),
                      suptitle="Passive SWE -- CONSENSUS velocity/bp10_150/gauss1.0/mean3/off5x0.5mm "
                               "(direction chosen per window)",
                      panel_titles=cons_titles)
    print("wrote per_window_best.png, consensus_perdir.png ->", outdir)


if __name__ == "__main__":
    main()
