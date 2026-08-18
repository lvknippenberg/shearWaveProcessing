"""Task 3 -- per-pulse 5x3 space-time grids (TX voltage x element count), annotated with safety indices.

Two figure sets (1500 and 1900 push cycles); within each, ONE figure per push pulse (10) so the
push-to-push variance is visible by flipping through them. Each figure is a 5x3 grid:
  rows  = TX voltage 20,25,30,35,40 V   (commanded)
  cols  = 41, 61, 79 push elements
Each cell shows that measurement's phantom space-time displacement (recipe band bp120-700) for the
given pulse, titled with the hydrophone-derived safety indices MI / I_sppa.3 / I_spta.3.

SAFETY INDICES: taken from the DIRECT no-preamp hydrophone push measurements at these commanded
voltages (79 el, table below -- MI and I_sppa.3 derated, from FreqDomainSensitivity.m fixed-f0).
61 el / 41 el are ESTIMATED from 79 el by focal-gain scaling pr ~ (N/79)^0.7 (I_sppa ~ (N/79)^1.4);
those cells are marked "~". I_spta.3 = I_sppa.3 * PD * PRF_eff with the push burst duty
(PRF_eff = 0.77 Hz) and PD ~ cycles (1900 cyc -> ~0.85 ms; scales with cyc/1900).

NB: these no-preamp values supersede the earlier with-preamp-extrapolated table (which read pressure
~4x high because the AH-2010 preamp saturated above ~15 V); reconcile before quoting a single number.

Usage:  python scripts/phantom_safety_grid.py --root D:\\swp_ph [--pulses 0-9] [--cycles 1500,1900]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio

BAND = "disp bp120-700 gauss mean3"
TX = [20, 25, 30, 35, 40]
EL = [41, 61, 79]

# 79 el, no-preamp, derated -- MEASURED (FreqDomainSensitivity.m, fixed-f0):
MI79   = {20: 0.99, 25: 1.27, 30: 1.42, 35: 1.54, 40: 1.84}
ISPPA79 = {20: 47.0, 25: 67.1, 30: 91.2, 35: 108.8, 40: 128.8}   # W/cm^2
PRF_EFF = 0.77          # Hz, push burst duty (20 Hz for 1.2 s then >=30 s off)
PD_1900_S = 0.85e-3     # 10-90% pulse duration at 1900 cycles


def safety(el, V, cyc):
    """(MI, I_sppa.3 [W/cm2], I_spta.3 [mW/cm2], measured_flag)."""
    scale_pr = (el / 79.0) ** 0.7
    mi = MI79[V] * scale_pr
    isppa = ISPPA79[V] * scale_pr ** 2
    pd = PD_1900_S * (cyc / 1900.0)
    ispta = isppa * pd * PRF_EFF * 1000.0        # W/cm2 -> mW/cm2
    return mi, isppa, ispta, (el == 79)


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
    p = os.path.join(folder, "output", "swp_active", f"swp_meas{pulse}.hdf5")
    if not os.path.isfile(p):
        return None
    with h5py.File(p, "r") as f:
        disp = f["displacement"]
        key = BAND if BAND in disp else [k for k in disp if k != "t_s"][0]
        D = np.asarray(disp[key], dtype=np.float64)
        t = np.asarray(disp["t_s"]) * 1e3
        r = np.asarray(f["r_m"]) * 1e3
        r0 = float(f.attrs["r0_m"]) * 1e3
    return D, t, r, r0


def make_figure(root, idx, cyc, pulse, outdir):
    fig, axes = plt.subplots(len(TX), len(EL), figsize=(11, 14), squeeze=False)
    fig.suptitle(f"Phantom space-time  |  {cyc} cycles  |  push pulse {pulse}/9\n"
                 f"rows = TX voltage, cols = elements  (title: MI / Isppa.3 W/cm2 / Ispta.3 mW/cm2; "
                 f"~ = element-scaled estimate)", fontsize=11)
    any_data = False
    for i, V in enumerate(TX):
        for j, el in enumerate(EL):
            ax = axes[i][j]
            folder = idx.get((el, cyc, V))
            st = load_st(folder, pulse) if folder else None
            mi, isppa, ispta, meas = safety(el, V, cyc)
            tag = "" if meas else "~"
            if st is None:
                ax.text(0.5, 0.5, "(pending\nbeamform)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.5")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                any_data = True
                D, t, r, r0 = st
                lim = np.percentile(np.abs(D), 99) or 1.0
                ax.imshow(D, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                          extent=[r[0], r[-1], t[-1], t[0]])
                ax.axvline(r0, ls="--", color="0.3", lw=0.8)
                if i == len(TX) - 1:
                    ax.set_xlabel("r [mm]")
                if j == 0:
                    ax.set_ylabel(f"{V} V\nt [ms]")
            ax.set_title(f"{el}el {V}V  MI{tag}{mi:.2f}  Is{tag}{isppa:.0f}  It{tag}{ispta:.0f}",
                         fontsize=8)
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
            out, has = make_figure(a.root, idx, cyc, pulse, outdir)
            n += 1
            print(f"{'ok ' if has else 'EMPTY '}{out}")
    print(f"\nwrote {n} figures to {outdir}")


if __name__ == "__main__":
    main()
