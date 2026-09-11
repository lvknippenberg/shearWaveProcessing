"""Task 5 -- can pushes from two different heartbeats, at the same cardiac phase, be combined?

The 2026-08-18 in-vivo acquisitions are R-peak triggered and fire 24 pushes on a 50 ms grid, which
at ~68 bpm spans 1.33 cardiac cycles. The last six pushes therefore repeat the cardiac phase of the
first six, in the *next* beat. That is a free, in-data test of whether combining beats helps -- no
new acquisition, no breath-hold.

Two questions, and they have opposite implications:

1. **Is the cardiac motion reproducible from beat to beat at matched phase?** Measured as the
   correlation between the two pushes' *no-push controls* (the pre-push reference, i.e. cardiac
   motion alone), against the null level from phase-*mismatched* control pairs in the same
   acquisition.
     - reproducible  -> averaging beats will NOT suppress it (it adds coherently), but an
       interleaved push / no-push design could SUBTRACT it;
     - not reproducible -> averaging suppresses it as sqrt(N), subtraction does not work.

2. **Does averaging the two beats actually improve the push-minus-control contrast?** Computed by
   averaging the pair's space-times (and their controls) on the common r-axis anchored at r0, and
   scoring the average the same way as the singles.

    python scripts/task5_cross_beat.py --folder <in-vivo folder> --outdir <dir>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swe_lib as L                                              # noqa: E402
from swp_ecg import push_phases, cross_beat_pairs, ecg_path                # noqa: E402
from task4_invivo_compare import REF_SPLIT, SCORE_KW             # noqa: E402


def ncc(a, b):
    """Normalised cross-correlation of two space-time images (both demeaned)."""
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def collect(folder, recipe, quantity):
    """Per push: the PUSH and NO-PUSH space-times on a common r-axis relative to r0."""
    out = {}
    n = L.n_pushes(folder)
    for m in range(n):
        rec = {}
        for tag, nopush in (("push", False), ("ctrl", True)):
            st, r0 = L.st_for(folder, m, recipe, quantity=quantity, phantom=False,
                              nopush=nopush, ref_split=REF_SPLIT)
            rec[tag] = (st.data, st.r - r0, st.t)
        rec["r0"] = r0
        out[m] = rec
        print(f"    push {m:2d} done", flush=True)
    return out


def common_grid(a, b):
    """Resample two (data, d, t) triples onto a shared distance-from-r0 / time grid."""
    (da, ra, ta), (db, rb, tb) = a, b
    nt = min(da.shape[0], db.shape[0])
    lo, hi = max(ra[0], rb[0]), min(ra[-1], rb[-1])
    rr = np.linspace(lo, hi, min(len(ra), len(rb)))
    A = np.stack([np.interp(rr, ra, da[i]) for i in range(nt)])
    B = np.stack([np.interp(rr, rb, db[i]) for i in range(nt)])
    return A, B, rr, ta[:nt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--quantity", default="displacement")
    ap.add_argument("--tol", type=float, default=40.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rec = L.REC_INVIVO

    n = L.n_pushes(a.folder)
    ph = push_phases(ecg_path(a.folder), n)
    pairs = cross_beat_pairs(ph, a.tol)
    print(f"HR {60000/np.nanmedian(ph['rr']):.1f} bpm, {len(pairs)} cross-beat pairs\n")
    print("  computing space-times")
    st = collect(a.folder, rec, a.quantity)

    # ---- null level: control-vs-control for phase-MISMATCHED pairs in the same beat
    null = []
    for i in range(n):
        for j in range(i + 1, n):
            if ph["beat"][i] != ph["beat"][j]:
                continue
            if abs(ph["phase"][i] - ph["phase"][j]) < 150:
                continue                                     # only genuinely different phases
            A, B, _, _ = common_grid(st[i]["ctrl"], st[j]["ctrl"])
            null.append(abs(ncc(A, B)))
    null = np.array(null)

    rows = []
    print(f"\n{'pair':>9} {'dphase':>7} | {'ctrl-ctrl':>10} {'push-push':>10} | "
          f"{'dOC single':>11} {'dOC avg':>9} {'gain':>7}")
    for i, j, d in pairs:
        Ac, Bc, rr, tt = common_grid(st[i]["ctrl"], st[j]["ctrl"])
        Ap, Bp, rp_, tp = common_grid(st[i]["push"], st[j]["push"])
        cc, cp = ncc(Ac, Bc), ncc(Ap, Bp)

        class S:                                             # minimal SpaceTime-like shim
            def __init__(self, data, r, t, q="displacement"):
                self.data, self.r, self.t, self.quantity = data, r, t, q

        def score(data, r, t, n_t):
            return L.scores(L.truncate(S(data, r + st[i]["r0"], t), n_t), st[i]["r0"], **SCORE_KW)

        n_ctrl = Ac.shape[0]
        oc_i = score(Ap, rp_, tp, n_ctrl)[0]
        oc_j = score(Bp, rp_, tp, n_ctrl)[0]
        oc_ci = score(Ac, rr, tt, n_ctrl)[0]
        oc_cj = score(Bc, rr, tt, n_ctrl)[0]
        d_single = np.mean([oc_i - oc_ci, oc_j - oc_cj])

        avg_p, avg_c = 0.5 * (Ap + Bp), 0.5 * (Ac + Bc)
        d_avg = score(avg_p, rp_, tp, n_ctrl)[0] - score(avg_c, rr, tt, n_ctrl)[0]

        rows.append(dict(i=i, j=j, dphase=d, cc_ctrl=cc, cc_push=cp,
                         d_single=d_single, d_avg=d_avg,
                         avg_p=avg_p, avg_c=avg_c, Ap=Ap, Bp=Bp, Ac=Ac, Bc=Bc,
                         r=rp_, t=tp, rc=rr, tc=tt))
        print(f"  {i:2d}<->{j:2d} {d:7.1f} | {cc:10.3f} {cp:10.3f} | "
              f"{d_single:11.3f} {d_avg:9.3f} {d_avg-d_single:+7.3f}")

    print(f"\nnull level (control vs control, phase-mismatched, n={len(null)}): "
          f"|ncc| median {np.median(null):.3f}, 90th pct {np.percentile(null,90):.3f}")
    ccs = np.array([r["cc_ctrl"] for r in rows])
    cps = np.array([r["cc_push"] for r in rows])
    print(f"cross-beat control-control ncc : median {np.median(ccs):+.3f}  "
          f"(range {ccs.min():+.3f} to {ccs.max():+.3f})")
    print(f"cross-beat push-push       ncc : median {np.median(cps):+.3f}")
    ds = np.array([r["d_single"] for r in rows])
    da = np.array([r["d_avg"] for r in rows])
    print(f"\npush-minus-control, single pushes : median {np.median(ds):+.3f}")
    print(f"push-minus-control, 2-beat average: median {np.median(da):+.3f}  "
          f"(change {np.median(da)-np.median(ds):+.3f})")

    # ---------------------------------------------------------------- figures
    unit = 1e3 if a.quantity == "velocity" else 1e6
    ncol = len(rows)
    fig, axes = plt.subplots(3, ncol, figsize=(2.7 * ncol, 9.6), squeeze=False)
    clim = float(np.median([np.percentile(np.abs(r["Ap"]), 99) for r in rows])) * unit
    for k, r in enumerate(rows):
        for row, (data, lbl) in enumerate(((r["Ap"], f"beat 1 - push {r['i']}"),
                                           (r["Bp"], f"beat 2 - push {r['j']}"),
                                           (r["avg_p"], "average of the two"))):
            ax = axes[row][k]
            ax.imshow(data * unit,
                      extent=(r["r"][0] * 1e3, r["r"][-1] * 1e3, r["t"][-1] * 1e3, r["t"][0] * 1e3),
                      cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto", origin="upper")
            ax.axvline(0, color="0.2", ls="--", lw=0.8, alpha=0.7)
            ax.tick_params(labelsize=6)
            ax.set_title(lbl, fontsize=8)
            if k == 0:
                ax.set_ylabel("t [ms]", fontsize=8)
            if row == 2:
                ax.set_xlabel("distance from $r_0$ [mm]", fontsize=8)
        axes[0][k].text(0.5, 1.30, f"phase {r['dphase']:.0f} ms apart\n"
                        f"ctrl-ctrl ncc {r['cc_ctrl']:+.2f}", transform=axes[0][k].transAxes,
                        ha="center", fontsize=8, color="0.35")
    fig.suptitle(f"Combining the same cardiac phase from two consecutive beats "
                 f"({os.path.basename(a.folder)[:34]}, {a.quantity})\n"
                 f"the acquisition spans 1.33 cardiac cycles, so pushes 18-23 repeat the phase of "
                 f"pushes 0-5", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(a.outdir, "cross_beat_pushes.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("\nwrote", out)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.hist(null, bins=20, color="0.75", label=f"phase-mismatched (null), n={len(null)}")
    for c in ccs:
        ax.axvline(abs(c), color="tab:red", lw=1.6)
    ax.axvline(np.median(null), color="0.35", ls="--", lw=1.2)
    ax.set_xlabel("|normalised cross-correlation| of two control space-times")
    ax.set_ylabel("count")
    ax.set_title("Is cardiac motion reproducible\nbeat-to-beat at matched phase?", fontsize=10)
    ax.legend(fontsize=7)
    ax.text(0.97, 0.85, "red = cross-beat,\nphase-matched", transform=ax.transAxes, ha="right",
            fontsize=8, color="tab:red")

    ax = axes[1]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, ds, 0.4, label="single pushes (mean of the pair)", color="tab:blue")
    ax.bar(x + 0.2, da, 0.4, label="2-beat average", color="tab:red")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['i']}/{r['j']}" for r in rows], fontsize=8)
    ax.set_xlabel("push pair")
    ax.set_ylabel("push minus control (origin coherence)")
    ax.set_title("Does averaging the two beats help?", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ecg = ph["ecg"]
    t0 = ph["t"][0]
    ax.plot(ecg["Time_ms"] - t0, ecg["Signal_V"], color="0.35", lw=1)
    for k, tt in enumerate(ph["t"]):
        ax.axvline(tt - t0, color="tab:blue" if ph["beat"][k] == 0 else "tab:red", lw=0.8,
                   alpha=0.75)
    ax.set_xlim(-200, 1400)
    ax.set_xlabel("time relative to push 0 [ms]")
    ax.set_ylabel("ECG [V]")
    ax.set_title("Push timing on the ECG\n(blue = beat 1, red = beat 2)", fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle("Cross-beat combination test, in-vivo R-peak-triggered acquisition", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(a.outdir, "cross_beat_summary.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)

    with open(os.path.join(a.outdir, "cross_beat.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["push_i", "push_j", "dphase_ms", "ncc_ctrl_ctrl", "ncc_push_push",
                    "dOC_single", "dOC_2beat_avg"])
        for r in rows:
            w.writerow([r["i"], r["j"], f"{r['dphase']:.1f}", f"{r['cc_ctrl']:.4f}",
                        f"{r['cc_push']:.4f}", f"{r['d_single']:.4f}", f"{r['d_avg']:.4f}"])
    print("wrote cross_beat.csv")


if __name__ == "__main__":
    main()
