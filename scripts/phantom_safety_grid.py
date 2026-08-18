"""Task 3 -- per-pulse 5x3 space-time grids (TX voltage x element count), annotated with safety indices.

Two figure sets (1500 and 1900 push cycles); within each, ONE figure per push pulse (10) so the
push-to-push variance is visible by flipping through them. Each figure is a 5x3 grid:
  rows  = TX voltage 20,25,30,35,40 V   (commanded)
  cols  = 41, 61, 79 push elements
Each cell shows that measurement's phantom shear-wave space-time for the given pulse.

RECIPE: the phantom-tuned VELOCITY + median recipe from the parameter-sweep analysis
(docs/phantom_parameter_sweep.md, scripts/sweep_params.py / sweep_top_montage.py): temporal band-pass
80-500 Hz -> spatial MEDIAN 0.90 x 2.64 mm -> temporal moving-MEDIAN (win 5) -> 9 M-line offsets ->
outward directional filter, quantity = velocity. This is NOT the run.py viz --phantom output, which
uses the in-vivo displacement consensus (bp120-700 / gaussian) and looks much noisier on the phantom.
Space-times are built directly from the beamformed buffer-2 IQ via sweep_extract.spacetime_for.

SAFETY INDICES (titles): MEASURED for all three apertures from the DIRECT no-preamp, peak-optimised
("_opt") hydrophone push captures at these commanded voltages (MI.3 and I_sppa.3 derated 0.3
dB/cm/MHz at depth 36.3 mm; see CorrectedMaxVoltage.m). Worst-case peak values that set the limits
(41el max 32 V, 61el 23 V, 79el 20 V, all I_sppa.3-limited). I_spta.3 = I_sppa.3 * PD * PRF_eff
(PD ~ 0.854 ms at 1900 cyc, scales with cyc/1900; burst duty PRF_eff = 0.769 Hz).

Usage:  python scripts/phantom_safety_grid.py --root D:\\swp_ph [--pulses 0-9] [--cycles 1500,1900]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "swp_gui", "scripts"):
    sys.path.insert(0, os.path.join(_ROOT, _p))
import core                                                   # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing             # noqa: E402
import sweep_extract as sw                                    # noqa: E402
from swp.viz.core.geometry import robust_clim                 # noqa: E402

TX = [20, 25, 30, 35, 40]
EL = [41, 61, 79]

# Phantom-tuned velocity+median recipe (sweep_top_montage FIXED_REC / sweep_params REC):
REC = {"iq": "none",
       "motion": [("temporal_bandpass", {"f_lo": 80, "f_hi": 500, "order": 2})],
       "spatial": [("spatial_median", {"size_z_m": 0.90e-3, "size_x_m": 2.64e-3})],
       "temporal": [("temporal_moving_median", {"window": 5})],
       "offsets": 9, "step_m": 1.09e-3, "sm": "median", "tm": "median", "f_lo": 80, "f_hi": 500}
QUANTITY = "velocity"

# MEASURED (no-preamp _opt captures, derated) at V = 20,25,30,35,40 -- CorrectedMaxVoltage.m:
MI_MEAS = {41: [0.93, 1.15, 1.37, 1.64, 1.94],
           61: [1.33, 1.56, 2.15, 2.66, 2.86],
           79: [1.61, 2.08, 2.65, 2.86, 2.86]}
ISPPA_MEAS = {41: [68, 108, 163, 230, 291],
              61: [143, 220, 327, 471, 586],
              79: [189, 299, 436, 481, 489]}       # W/cm^2
VGRID = [20, 25, 30, 35, 40]
PRF_EFF = 0.769
PD_1900_S = 0.854e-3

_REC0 = core.Recipe(mline_source="horizontal_push")


def safety(el, V, cyc):
    k = VGRID.index(V)
    mi = MI_MEAS[el][k]
    isppa = ISPPA_MEAS[el][k]
    ispta = isppa * PD_1900_S * (cyc / 1900.0) * PRF_EFF * 1000.0
    return mi, isppa, ispta


def build_index(root):
    idx = {}
    for d in sorted(os.listdir(root)):
        if not d.startswith("DefaultPatient"):
            continue
        folder = os.path.join(root, d)
        try:
            m = sio.loadmat(os.path.join(folder, "AcquisitionParametersAndECG.mat"),
                            squeeze_me=True, struct_as_record=False)
        except Exception:
            continue
        SW, TPC = m["SW"], m["TPC"]
        idx[(int(SW.nb_push_elmts), int(SW.pushCycle), int(round(float(TPC[4].hv))))] = folder
    return idx


def load_st(folder, pulse):
    """Velocity+median space-time for one push -> (data[t,r], t_ms, r_mm, r0_mm) or None."""
    if not os.path.isfile(os.path.join(folder, "output", f"CombinedData_buffer2_meas{pulse}_iq.hdf5")):
        return None
    try:
        acq = core.load_acq(folder, pulse, _REC0)
        ml = core.load_mline_for(folder, pulse, acq, _REC0)
        r0 = _r0_lateral_crossing(ml, float(acq.push_x))
        est = sw.estimator_for_iq(acq, "none")
        st = sw.spacetime_for(est, acq, ml, r0, REC, QUANTITY)
    except Exception as exc:  # noqa: BLE001
        print(f"    {os.path.basename(folder)[-8:]} push{pulse} failed: {exc}")
        return None
    return st.data * 1e3, st.t * 1e3, st.r * 1e3, r0 * 1e3


def make_figure(idx, cyc, pulse, outdir):
    fig, axes = plt.subplots(len(TX), len(EL), figsize=(11, 14), squeeze=False)
    fig.suptitle(f"Phantom shear-wave space-time (velocity, median recipe)  |  {cyc} cycles  |  "
                 f"push pulse {pulse}/9\nrows = TX voltage, cols = elements  "
                 f"(title: MI / Isppa.3 W/cm2 / Ispta.3 mW/cm2, measured no-preamp worst-case)",
                 fontsize=11)
    any_data = False
    for i, V in enumerate(TX):
        for j, el in enumerate(EL):
            ax = axes[i][j]
            folder = idx.get((el, cyc, V))
            st = load_st(folder, pulse) if folder else None
            mi, isppa, ispta = safety(el, V, cyc)
            if st is None:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.5")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                any_data = True
                D, t, r, r0 = st
                rc = (r > 0.1 * r[-1]) & (r < 0.9 * r[-1])
                lim = robust_clim(D, rc, 97) or np.percentile(np.abs(D), 99) or 1.0
                ax.imshow(D, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                          extent=[r[0], r[-1], t[-1], t[0]])
                ax.axvline(r0, ls="--", color="0.3", lw=0.8)
                if i == len(TX) - 1:
                    ax.set_xlabel("r [mm]")
                if j == 0:
                    ax.set_ylabel(f"{V} V\nt [ms]")
            ax.set_title(f"{el}el {V}V  MI{mi:.2f}  Is{isppa:.0f}  It{ispta:.0f}", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"grid_{cyc}c_pulse{pulse}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, any_data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--pulses", default="0-9")
    ap.add_argument("--cycles", default="1500,1900")
    a = ap.parse_args()

    lo, hi = (a.pulses.split("-") + [a.pulses])[:2] if "-" in a.pulses else (a.pulses, a.pulses)
    pulses = range(int(lo), int(hi) + 1)
    cycles = [int(c) for c in a.cycles.split(",")]

    idx = build_index(a.root)
    outdir = os.path.join(a.root, "safety_grids")
    n = 0
    for cyc in cycles:
        for pulse in pulses:
            out, has = make_figure(idx, cyc, pulse, outdir)
            n += 1
            print(f"{'ok ' if has else 'EMPTY '}{out}")
    print(f"\nwrote {n} figures to {outdir}")


if __name__ == "__main__":
    main()
