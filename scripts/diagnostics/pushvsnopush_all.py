"""Per-push: is the post-push (tracking) space-time actually stronger than the pre-push
(reference) space-time? Run view A on both windows for all 24 pushes at 30 V and 40 V.

If push ~ no-push (ratio ~1) for most pushes, the plotted "wave" is cardiac motion, present
with or without the push. A real captured shear wave would give push >> no-push.
"""
from __future__ import annotations
import os, sys, dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = r"D:/Luuk van Knippenberg/Github/shearWaveProcessing"
sys.path.insert(0, os.path.join(_REPO, "src"))
from swp.viz import runconfig as rc
from swp.viz.pipeline import run_pipeline
from swp.viz.metrics import origin_coherence
from swp.viz.io import load_mline
from swp.viz.mline import mline_from_points

CFG = rc.load_config(os.path.join(_REPO, "configs", "active.yaml"))
VIEW = "disp bp120-700 gauss mean3"
IV_ROOT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Invivo"
FOLDERS = [("30 V", "Luuk30V_SW_data_04-August-2026_13-35-06"),
           ("40 V", "Luuk40V_SW_data_04-August-2026_13-36-07")]


def view_cfg(acq):
    for name, vc in rc.build_views(CFG, acq):
        if name == VIEW:
            return vc


def amp(res):
    return float(np.nanpercentile(np.abs(res.st.data * 1e6), 97))


def one(iqp, mlnpz):
    acq = rc.load_acquisition(iqp)
    pts, ns = load_mline(mlnpz)
    mline = mline_from_points(pts, ns)
    focus = rc.build_focus(CFG, acq)
    vc = view_cfg(acq)
    rp = run_pipeline(acq, mline, vc, focus=focus)
    n_ref = acq.ref_iq.shape[0]
    t_ref = acq.t_ref if acq.t_ref is not None else (np.arange(n_ref) / acq.prf)
    acq_np = dataclasses.replace(acq, iq=acq.ref_iq, t=np.asarray(t_ref, float))
    rr = run_pipeline(acq_np, mline, vc, focus=focus)
    return amp(rp), amp(rr)


def main():
    res = {}
    for lbl, fld in FOLDERS:
        push, nopush = [], []
        for m in range(24):
            iqp = os.path.join(IV_ROOT, fld, "output", f"CombinedData_buffer2_meas{m}_iq.hdf5")
            mlnpz = os.path.join(IV_ROOT, fld, "output", "mlines", f"active_meas{m}_mline.npz")
            if not (os.path.exists(iqp) and os.path.exists(mlnpz)):
                push.append(np.nan); nopush.append(np.nan); continue
            p, n = one(iqp, mlnpz)
            push.append(p); nopush.append(n)
        res[lbl] = (np.array(push), np.array(nopush))
        p, n = res[lbl]
        ratio = p / n
        print(f"{lbl}: push amp median {np.nanmedian(p):5.2f}  no-push median {np.nanmedian(n):5.2f} um | "
              f"ratio push/nopush median {np.nanmedian(ratio):.2f}  "
              f"# pushes with ratio>1.5: {int(np.nansum(ratio > 1.5))}/24")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"30 V": "tab:blue", "40 V": "tab:red"}
    ax = axs[0]
    for lbl, _ in FOLDERS:
        p, n = res[lbl]
        ax.scatter(n, p, c=colors[lbl], label=lbl, s=30, alpha=0.8)
    lim = np.nanmax([np.nanmax(res[l][0]) for l, _ in FOLDERS] +
                    [np.nanmax(res[l][1]) for l, _ in FOLDERS]) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="push = no-push")
    ax.plot([0, lim], [0, 1.5 * lim], color="0.6", ls=":", lw=1, label="push = 1.5x no-push")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("NO-PUSH (pre-push reference) amplitude (um)")
    ax.set_ylabel("PUSH (post-push tracking) amplitude (um)")
    ax.set_title("Per-push: push vs no-push band-passed amplitude\n(on the diagonal = no push contribution)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axs[1]
    for j, (lbl, _) in enumerate(FOLDERS):
        p, n = res[lbl]
        ratio = (p / n)
        ratio = ratio[np.isfinite(ratio)]
        bp = ax.boxplot([ratio], positions=[j], widths=0.6, patch_artist=True, showmeans=True)
        for b in bp["boxes"]:
            b.set(facecolor=colors[lbl], alpha=0.55)
        ax.plot(np.random.normal(j, 0.05, ratio.size), ratio, ".", color="0.2", ms=4)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="push = no-push")
    ax.axhline(1.5, color="0.6", ls=":", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels([l for l, _ in FOLDERS])
    ax.set_ylabel("push / no-push amplitude ratio")
    ax.set_title("Ratio distribution over 24 pushes")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("In-vivo: does the ARF push add signal over the pre-push cardiac motion? "
                 "(ratio ~1 = no, dominated by cardiac motion)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(IV_ROOT, "push_vs_nopush_all.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
