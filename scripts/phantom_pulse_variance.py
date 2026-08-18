"""Task 2 -- quantify the push-to-push variance across the 10 pushes of each phantom measurement.

For every measurement folder in the sweep (via the short junction root, see phantom_voltage_sweep.md
Windows MAX_PATH note) this reads the 10 per-pulse space-time HDF5s written by
`run.py viz --phantom` (swp_meas0..9.hdf5) and, per pulse, computes push-strength / wave proxies:

  * focal amplitude  = max_t |disp| within |r - r0| < FOCAL_MM of the push origin (initial ARF push);
  * wave energy      = RMS of the whole space-time displacement map (shear-wave content);
  * image, as a vector, feeds a 10x10 pairwise-correlation matrix -> mean off-diagonal correlation
    = how reproducible the space-time pattern is pulse-to-pulse.

Across the 10 pulses it reports mean, std and coefficient of variation (CV = std/mean). High CV /
low image-correlation at high TX voltage + high element count is the signature of the HV supply sag
(0.6 A load > 0.5 A supply) making the delivered push inconsistent firing to firing.

Outputs: <root>/pulse_variance.csv  and  <root>/pulse_variance.png (CV vs voltage, one line/config).

Usage:  python scripts/phantom_pulse_variance.py --root D:\\swp_ph
        (add --band "disp bp120-700 gauss mean3" to pick the recipe band; that is the default)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio

FOCAL_MM = 3.0
DEFAULT_BAND = "disp bp120-700 gauss mean3"


def config_of(folder):
    m = sio.loadmat(os.path.join(folder, "AcquisitionParametersAndECG.mat"),
                    squeeze_me=True, struct_as_record=False)
    SW, TPC = m["SW"], m["TPC"]
    return dict(el=int(SW.nb_push_elmts), cyc=int(SW.pushCycle),
                V=int(round(float(TPC[4].hv))), npush=int(SW.Nframes))


def pulse_metrics(h5path, band):
    with h5py.File(h5path, "r") as f:
        key = band if band in f["displacement"] else list(f["displacement"])[0]
        D = np.asarray(f["displacement"][key], dtype=np.float64)   # (t, r)
        r = np.asarray(f["r_m"]) * 1e3                             # mm
        r0 = float(f.attrs["r0_m"]) * 1e3
    foc = np.abs(D[:, np.abs(r - r0) < FOCAL_MM])
    focal = float(foc.max()) if foc.size else np.nan
    energy = float(np.sqrt(np.mean(D**2)))
    return focal, energy, D.ravel()


def process_folder(folder, band):
    paths = [os.path.join(folder, "output", "swp_active", f"swp_meas{i}.hdf5") for i in range(10)]
    if not all(os.path.isfile(p) for p in paths):
        return None
    foc, ene, vecs = [], [], []
    for p in paths:
        a, b, v = pulse_metrics(p, band)
        foc.append(a); ene.append(b); vecs.append(v)
    foc, ene = np.array(foc), np.array(ene)
    V = np.vstack(vecs)
    C = np.corrcoef(V)
    mean_corr = float((C.sum() - np.trace(C)) / (C.size - len(C)))
    cv = lambda x: float(np.std(x) / np.mean(x)) if np.mean(x) else np.nan
    cfg = config_of(folder)
    return dict(cfg, focal=foc, energy=ene,
                focal_mean=float(foc.mean()), focal_cv=cv(foc),
                energy_mean=float(ene.mean()), energy_cv=cv(ene),
                img_corr=mean_corr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="short junction root of the sweep")
    ap.add_argument("--band", default=DEFAULT_BAND)
    a = ap.parse_args()

    subs = sorted(d for d in os.listdir(a.root) if d.startswith("DefaultPatient"))
    rows = []
    for d in subs:
        r = process_folder(os.path.join(a.root, d), a.band)
        if r is None:
            print(f"skip (not beamformed yet): {d}")
            continue
        rows.append(r)
        print(f"el={r['el']:2d} cyc={r['cyc']} V={r['V']:2d} | "
              f"focal={r['focal_mean']:.2e} CV={r['focal_cv']*100:4.1f}% | "
              f"energy CV={r['energy_cv']*100:4.1f}% | img-corr={r['img_corr']:.2f}")

    if not rows:
        raise SystemExit("no beamformed folders yet -- run run.py beamform+viz --phantom first")

    csv_path = os.path.join(a.root, "pulse_variance.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["el", "cyc", "V", "focal_mean", "focal_cv", "energy_mean", "energy_cv",
                    "img_corr", "focal_per_pulse"])
        for r in rows:
            w.writerow([r["el"], r["cyc"], r["V"], f"{r['focal_mean']:.4e}", f"{r['focal_cv']:.4f}",
                        f"{r['energy_mean']:.4e}", f"{r['energy_cv']:.4f}", f"{r['img_corr']:.4f}",
                        ";".join(f"{x:.4e}" for x in r["focal"])])
    print(f"\nwrote {csv_path}")

    # CV-of-focal-amplitude vs voltage, one line per (el, cyc)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for (el, cyc) in sorted({(r["el"], r["cyc"]) for r in rows}):
        sel = sorted((r for r in rows if r["el"] == el and r["cyc"] == cyc), key=lambda r: r["V"])
        if not sel:
            continue
        Vs = [r["V"] for r in sel]
        ax[0].plot(Vs, [r["focal_cv"] * 100 for r in sel], "o-", label=f"{el}el {cyc}c")
        ax[1].plot(Vs, [r["img_corr"] for r in sel], "o-", label=f"{el}el {cyc}c")
    ax[0].set(xlabel="commanded TX voltage (V)", ylabel="focal-amplitude CV across 10 pushes (%)",
              title="push-to-push variance (higher = worse)")
    ax[1].set(xlabel="commanded TX voltage (V)", ylabel="mean pairwise space-time correlation",
              title="space-time reproducibility (1 = identical)")
    for x in ax:
        x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(a.root, "pulse_variance.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
