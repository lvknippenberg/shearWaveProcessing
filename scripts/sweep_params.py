"""Phantom parameter-sweep analysis: does pulse length / aperture / PRF change the space-time?

Data: 2026_08_07 'Phantom parameter sweep' - 11 configs x {20,30} V, 10 pushes each (interleaved).
Per push we compute:
  - FOCAL DISPLACEMENT (peak |disp| at the ARF focus vs the pre-push reference) = a direct push-strength
    proxy, isolating the push levers (pulse length, aperture, voltage) from the tracking lever (PRF);
  - wavefront ROI-contrast (locked 50 V V-template) + mirror-symmetry + best-fit speed.
Aggregate the median over the 10 pushes, then compare each lever against the base config at 20/30 V.
No MI data available -> quality is reported per config, not per MI.

    <zea-python> scripts/sweep_params.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "swp_gui"), os.path.join(_ROOT, "scripts")):
    sys.path.insert(0, p)

import core                                              # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing        # noqa: E402
import sweep_extract as sw                                # noqa: E402
from detect_v import roi_contrast, symmetric_v_score      # noqa: E402

SWEEP = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom parameter sweep"
OUT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
REC = {"iq": "none",
       "motion": [("temporal_bandpass", {"f_lo": 80, "f_hi": 500, "order": 2})],
       "spatial": [("spatial_median", {"size_z_m": 0.9e-3, "size_x_m": 2.6e-3})],
       "temporal": [("temporal_moving_median", {"window": 5})],
       "offsets": 9, "step_m": 1.0e-3}
SPEEDS = np.arange(1.0, 4.01, 0.25)

# label each config by which single lever it changes vs base (1500/41/270)
def lever(cyc, el, pri):
    base = (cyc == 1500, el == 41, pri == 270)
    if all(base): return "base"
    if not base[0] and base[1] and base[2]: return f"pulse {cyc}c"
    if base[0] and not base[1] and base[2]: return f"aperture {el}el"
    if base[0] and base[1] and not base[2]: return f"PRI {pri}us"
    return f"combo {cyc}c/{el}el/{pri}us"


def config_of(folder):
    m = sio.loadmat(os.path.join(folder, "AcquisitionParametersAndECG.mat"),
                    squeeze_me=True, struct_as_record=False)
    SW, TPC = m["SW"], m["TPC"]
    return dict(cyc=int(SW.pushCycle), el=int(SW.nb_push_elmts), pri=int(SW.PRI_us),
                V=int(round(float(TPC[4].hv))), npush=int(SW.Nframes))


def focal_disp_um(acq, est):
    """Peak focal displacement [um]: max over early tracking frames of the median |disp| in a +/-2 mm
    box at the ARF focus (relative-to-reference). Direct push-strength readout."""
    iz = int(np.argmin(np.abs(acq.z - acq.push_z)))
    ix = int(np.argmin(np.abs(acq.x - acq.push_x)))
    dz = max(1, int(round(2e-3 / acq.dz))); dx = max(1, int(round(2e-3 / acq.dx)))
    d = np.abs(est.displacement)                          # (n, nz, nx)
    box = d[:, max(0, iz - dz):iz + dz + 1, max(0, ix - dx):ix + dx + 1]
    nf = box.shape[0]
    per_frame = np.median(box.reshape(nf, -1), axis=1)    # median |disp| in the box per frame
    return float(np.max(per_frame[:min(6, nf)]) * 1e6)    # peak in the immediate post-push transient


def best_speed(st, r0, tmpl):
    best_c, best_v = np.nan, -np.inf
    for c in SPEEDS:
        v = roi_contrast(st, {**tmpl, "c_mps": float(c)}, r0=r0)
        if np.isfinite(v) and v > best_v:
            best_v, best_c = v, float(c)
    return best_c, (best_v if np.isfinite(best_v) else np.nan)


def process(folder, tmpl):
    fd, roi, sym, spd = [], [], [], []
    rec0 = core.Recipe(mline_source="horizontal_push")
    for meas in range(10):
        try:
            acq = core.load_acq(folder, meas, rec0)
            ml = core.load_mline_for(folder, meas, acq, rec0)
            r0 = _r0_lateral_crossing(ml, float(acq.push_x))
            est = sw.estimator_for_iq(acq, "none")
            fd.append(focal_disp_um(acq, est))
            best = None
            for q in ("velocity", "displacement"):
                st = sw.spacetime_for(est, acq, ml, r0, REC, q)
                c, v = best_speed(st, r0, tmpl)
                s = symmetric_v_score(st, r0)[0]
                if best is None or v > best[0]:
                    best = (v, s, c)
            roi.append(best[0]); sym.append(best[1]); spd.append(best[2])
        except Exception as exc:  # noqa: BLE001
            print(f"    push {meas} failed: {exc}")
    med = lambda a: float(np.nanmedian(a)) if a else np.nan
    return dict(focal=med(fd), roi=med(roi), sym=med(sym), speed=med(spd),
                focal_iqr=(float(np.nanpercentile(fd, 25)), float(np.nanpercentile(fd, 75))) if fd else (0, 0),
                n=len(roi))


def main():
    tmpl = json.load(open(os.path.join(OUT, "v_roi_template.json"), encoding="utf-8"))
    folders = sorted(os.path.join(SWEEP, d) for d in os.listdir(SWEEP)
                     if d.startswith("DefaultPatient_"))
    rows = []
    for folder in folders:
        n = len([f for f in os.listdir(os.path.join(folder, "output"))
                 if "buffer2_meas" in f and f.endswith("_iq.hdf5")]) if os.path.isdir(os.path.join(folder, "output")) else 0
        if n < 10:
            print(f"skip {os.path.basename(folder)[-8:]} ({n} pushes beamformed)"); continue
        cfg = config_of(folder)
        r = process(folder, tmpl)
        r.update(cfg, lever=lever(cfg["cyc"], cfg["el"], cfg["pri"]), t=os.path.basename(folder)[-8:])
        rows.append(r)
        print(f"  {r['t']} {r['lever']:>16} {cfg['V']}V: focal={r['focal']:.2f}um "
              f"roi={r['roi']:.3f} sym={r['sym']:.2f} c={r['speed']:.2f}m/s (n={r['n']})")

    fields = ["t", "lever", "cyc", "el", "pri", "V", "focal", "roi", "sym", "speed", "n"]
    with open(os.path.join(OUT, "sweep_params.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    _plots(rows)
    print(f"\nwrote {os.path.join(OUT, 'sweep_params.csv')} + sweep_params.png")


def _plots(rows):
    order = ["base", "pulse 1000c", "pulse 1900c", "aperture 21el", "aperture 61el", "aperture 79el",
             "PRI 500us", "PRI 200us", "PRI 168us", "combo 1900c/79el/270us", "combo 1900c/79el/168us"]
    labels = {"base": "base", "pulse 1000c": "pulse-", "pulse 1900c": "pulse+", "aperture 21el": "ap 21",
              "aperture 61el": "ap 61", "aperture 79el": "ap 79", "PRI 500us": "PRF 2k", "PRI 200us": "PRF 5k",
              "PRI 168us": "PRF 6k", "combo 1900c/79el/270us": "max ap+pulse",
              "combo 1900c/79el/168us": "max+6kPRF"}
    metrics = [("focal", "focal displacement [um]"), ("roi", "wavefront ROI-contrast"),
               ("sym", "mirror-symmetry"), ("speed", "best-fit speed [m/s]")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    x = np.arange(len(order)); w = 0.38
    for ax, (mk, ttl) in zip(axs.ravel(), metrics):
        for i, V in enumerate((20, 30)):
            vals = [next((r[mk] for r in rows if r["lever"] == lv and r["V"] == V), np.nan) for lv in order]
            ax.bar(x + (i - 0.5) * w, vals, w, label=f"{V} V")
        ax.set_xticks(x); ax.set_xticklabels([labels[o] for o in order], rotation=45, ha="right", fontsize=8)
        ax.set_title(ttl); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
        ax.axvline(0.5, color="0.7", ls=":", lw=1)   # separate base from variations
    fig.suptitle("Phantom parameter sweep — effect of pulse / aperture / PRF (median over 10 pushes)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "sweep_params.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
