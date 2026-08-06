"""No-push control: run the active pipeline on the PRE-PUSH reference frames.

If the band-passed, directional space-time shows the same "outward wavefronts" during the
reference period (no push has fired) as during the tracking period (post-push), then the
pipeline is imaging ongoing cardiac motion / reverberation, not the ARF shear wave.

Rows: phantom 40 V (known-good) and in-vivo 30 V meas6 (one of the cleanest, oc=0.98).
Cols: PUSH window (tracking IQ) vs NO-PUSH window (reference IQ), same recipe & colour scale.
"""
from __future__ import annotations
import os, sys, dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = r"D:/Luuk van Knippenberg/Github/shearWaveProcessing"
sys.path.insert(0, os.path.join(_REPO, "src"))
from swp.viz import runconfig as rc                       # noqa: E402
from swp.viz.pipeline import run_pipeline                 # noqa: E402
from swp.viz.metrics import origin_coherence              # noqa: E402

CFG = rc.load_config(os.path.join(_REPO, "configs", "active.yaml"))
VIEW_NAME = "disp bp120-700 gauss mean3"

PH_ROOT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom"
IV_ROOT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Invivo"


def view_cfg(acq):
    for name, vc in rc.build_views(CFG, acq):
        if name == VIEW_NAME:
            return vc
    raise RuntimeError("view not found")


def spacetime_push_and_nopush(iq_path, mline_npz, phantom):
    acq = rc.load_acquisition(iq_path)
    # M-line
    if phantom:
        cfg = dict(CFG); cfg = {**CFG, "mline": {**CFG["mline"], "type": "horizontal_push"}}
        mline = rc.build_mline(cfg, acq, 0)
    else:
        from swp.viz.io import load_mline
        from swp.viz.mline import mline_from_points
        pts, ns = load_mline(mline_npz)
        mline = mline_from_points(pts, ns)
    focus = rc.build_focus(CFG, acq)
    vc = view_cfg(acq)

    # PUSH window: normal tracking IQ
    res_push = run_pipeline(acq, mline, vc, focus=focus)
    oc_push = origin_coherence(run_pipeline(acq, mline, dataclasses.replace(vc, directional=False),
                                            focus=focus).st, res_push.r0)

    # NO-PUSH window: feed the reference frames as the tracking ensemble (still relative to the
    # mean reference), identical recipe.
    n_ref = acq.ref_iq.shape[0]
    t_ref = acq.t_ref if acq.t_ref is not None else (np.arange(n_ref) / acq.prf)
    acq_nopush = dataclasses.replace(acq, iq=acq.ref_iq, t=np.asarray(t_ref, float))
    res_ref = run_pipeline(acq_nopush, mline, vc, focus=focus)
    oc_ref = origin_coherence(run_pipeline(acq_nopush, mline,
                                           dataclasses.replace(vc, directional=False),
                                           focus=focus).st, res_ref.r0)
    return (res_push, oc_push), (res_ref, oc_ref)


def draw(ax, res, title, clim):
    st = res.st
    r = st.r * 1e3; t = st.t * 1e3
    img = st.data * 1e6
    ax.imshow(img, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
              vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    ax.axvline(res.r0 * 1e3, color="0.2", ls="--", lw=0.8, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("r [mm]", fontsize=8); ax.set_ylabel("t [ms]", fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    from swp.acquisition import discover_measurements
    ph40 = [f for f, pv in discover_measurements(PH_ROOT) if abs(pv.hv - 40) < 1][0]
    cases = [
        ("phantom 40 V", os.path.join(ph40, "output", "CombinedData_buffer2_meas0_iq.hdf5"),
         None, True),
        ("in-vivo 30 V m6",
         os.path.join(IV_ROOT, "Luuk30V_SW_data_04-August-2026_13-35-06", "output",
                      "CombinedData_buffer2_meas6_iq.hdf5"),
         os.path.join(IV_ROOT, "Luuk30V_SW_data_04-August-2026_13-35-06", "output",
                      "mlines", "active_meas6_mline.npz"), False),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for i, (lbl, iqp, mlnpz, phantom) in enumerate(cases):
        (res_push, oc_push), (res_ref, oc_ref) = spacetime_push_and_nopush(iqp, mlnpz, phantom)
        # shared colour scale per row from the PUSH panel
        d = res_push.st.data * 1e6
        clim = float(np.nanpercentile(np.abs(d), 97)) or 1.0
        draw(axs[i][0], res_push, f"{lbl}\nPUSH window (tracking)   oc={oc_push:.2f}", clim)
        draw(axs[i][1], res_ref, f"{lbl}\nNO-PUSH window (reference)   oc={oc_ref:.2f}", clim)
        print(f"{lbl}: PUSH oc={oc_push:.3f}  peak={clim:.2f} um | "
              f"NO-PUSH oc={oc_ref:.3f}  peak={np.nanpercentile(np.abs(res_ref.st.data*1e6),97):.2f} um")
    fig.suptitle("No-push control: same recipe (bp120-700/gauss/mean3, outward-directional) on the\n"
                 "post-push tracking window vs the pre-push reference window (same colour scale per row)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(IV_ROOT, "nopush_control.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
