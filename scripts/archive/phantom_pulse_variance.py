"""Task 2 -- quantify the push-to-push variance across the 10 pushes of each phantom measurement.

For every measurement folder in the sweep (via the short junction root; see phantom_voltage_sweep.md
Windows MAX_PATH note) this loads the 10 beamformed pushes and, per push, computes:

  * focal push displacement (um) -- max |disp| in a +/-2 mm box at the ARF focus vs the pre-push
    reference (sweep_params.focal_disp_um). Direct push-strength readout. (Caveat: at strong pushes
    the focal speckle decorrelates and Loupas saturates ~this value, so treat it as a floor.)
  * the phantom VELOCITY+median space-time (same recipe as phantom_safety_grid / sweep_params):
    its RMS = wave energy, and its flattened image feeds a 10x10 pairwise-correlation matrix whose
    mean off-diagonal = how reproducible the space-time pattern is pulse-to-pulse.

Across the 10 pushes: mean, std, coefficient of variation (CV = std/mean). The HV supply-sag signature
(Readme: 0.6 A load > 0.5 A supply) would appear as rising CV / falling space-time correlation at high
voltage + element count.

Outputs: <root>/pulse_variance.csv  and  <root>/pulse_variance.png (CV & reproducibility vs voltage).

Usage:  python scripts/archive/phantom_pulse_variance.py --root D:\\swp_ph
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))           # archived: siblings + scripts/
for _p in (_HERE, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in ("src", "swp_gui", "scripts"):
    sys.path.insert(0, os.path.join(_ROOT, _p))
import core                                                   # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing             # noqa: E402
import sweep_extract as sw                                    # noqa: E402
from sweep_params import focal_disp_um                        # noqa: E402

REC = {"iq": "none",
       "motion": [("temporal_bandpass", {"f_lo": 80, "f_hi": 500, "order": 2})],
       "spatial": [("spatial_median", {"size_z_m": 0.90e-3, "size_x_m": 2.64e-3})],
       "temporal": [("temporal_moving_median", {"window": 5})],
       "offsets": 9, "step_m": 1.09e-3, "sm": "median", "tm": "median", "f_lo": 80, "f_hi": 500}
_REC0 = core.Recipe(mline_source="horizontal_push")


def config_of(folder):
    m = sio.loadmat(os.path.join(folder, "AcquisitionParametersAndECG.mat"),
                    squeeze_me=True, struct_as_record=False)
    SW, TPC = m["SW"], m["TPC"]
    return dict(el=int(SW.nb_push_elmts), cyc=int(SW.pushCycle),
                V=int(round(float(TPC[4].hv))), npush=int(SW.Nframes))


def process_folder(folder):
    if not all(os.path.isfile(os.path.join(folder, "output", f"CombinedData_buffer2_meas{i}_iq.hdf5"))
               for i in range(10)):
        return None
    foc, ene, vecs = [], [], []
    for m in range(10):
        try:
            acq = core.load_acq(folder, m, _REC0)
            ml = core.load_mline_for(folder, m, acq, _REC0)
            r0 = _r0_lateral_crossing(ml, float(acq.push_x))
            est = sw.estimator_for_iq(acq, "none")
            foc.append(focal_disp_um(acq, est))
            st = sw.spacetime_for(est, acq, ml, r0, REC, "velocity")
            ene.append(float(np.sqrt(np.mean(st.data ** 2))))
            vecs.append(st.data.ravel())
        except Exception as exc:  # noqa: BLE001
            print(f"    {os.path.basename(folder)[-8:]} push{m} failed: {exc}")
    if len(vecs) < 2:
        return None
    foc, ene = np.array(foc), np.array(ene)
    C = np.corrcoef(np.vstack(vecs))
    mean_corr = float((C.sum() - np.trace(C)) / (C.size - len(C)))
    cv = lambda x: float(np.std(x) / np.mean(x)) if np.mean(x) else np.nan
    cfg = config_of(folder)
    return dict(cfg, focal=foc, energy=ene, focal_mean=float(foc.mean()), focal_cv=cv(foc),
                energy_mean=float(ene.mean()), energy_cv=cv(ene), img_corr=mean_corr, n=len(vecs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()

    subs = sorted(d for d in os.listdir(a.root) if d.startswith("DefaultPatient"))
    rows = []
    for d in subs:
        r = process_folder(os.path.join(a.root, d))
        if r is None:
            print(f"skip (not beamformed): {d}")
            continue
        rows.append(r)
        print(f"el={r['el']:2d} cyc={r['cyc']} V={r['V']:2d} | focal={r['focal_mean']:5.1f}um "
              f"CV={r['focal_cv']*100:4.1f}% | energy CV={r['energy_cv']*100:4.1f}% | "
              f"space-time corr={r['img_corr']:.2f}")
    if not rows:
        raise SystemExit("no beamformed folders")

    csv_path = os.path.join(a.root, "pulse_variance.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["el", "cyc", "V", "focal_mean_um", "focal_cv", "energy_cv", "img_corr",
                    "focal_per_pulse_um"])
        for r in rows:
            w.writerow([r["el"], r["cyc"], r["V"], f"{r['focal_mean']:.2f}", f"{r['focal_cv']:.4f}",
                        f"{r['energy_cv']:.4f}", f"{r['img_corr']:.4f}",
                        ";".join(f"{x:.1f}" for x in r["focal"])])
    print(f"\nwrote {csv_path}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for (el, cyc) in sorted({(r["el"], r["cyc"]) for r in rows}):
        sel = sorted((r for r in rows if r["el"] == el and r["cyc"] == cyc), key=lambda r: r["V"])
        if not sel:
            continue
        Vs = [r["V"] for r in sel]
        ax[0].plot(Vs, [r["focal_cv"] * 100 for r in sel], "o-", label=f"{el}el {cyc}c")
        ax[1].plot(Vs, [r["energy_cv"] * 100 for r in sel], "o-", label=f"{el}el {cyc}c")
        ax[2].plot(Vs, [r["img_corr"] for r in sel], "o-", label=f"{el}el {cyc}c")
    ax[0].set(xlabel="commanded TX voltage (V)", ylabel="focal-displacement CV (%)",
              title="push-strength variance across 10 pushes")
    ax[1].set(xlabel="commanded TX voltage (V)", ylabel="space-time energy CV (%)",
              title="wave-energy variance")
    ax[2].set(xlabel="commanded TX voltage (V)", ylabel="mean pairwise space-time corr",
              title="space-time reproducibility (1 = identical)")
    for x in ax:
        x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(a.root, "pulse_variance.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
