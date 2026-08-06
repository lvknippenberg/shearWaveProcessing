"""Is the in-vivo signal actually the ARF push-induced shear wave?

Decisive test: raw axial displacement at the push focus, relative to the pre-push reference.
A real ARF push displaces tissue at the focus by a few..tens of um right after the push, and
that peak scales with push intensity (~V^2). We compare:
  (1) PHANTOM (known-good): peak focal displacement vs delivered voltage 15..50 V.
  (2) IN-VIVO: peak focal displacement, 30 V vs 40 V (24 pushes each).
Plus the raw displacement-vs-time at the focus (should show a push-locked transient).
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = r"D:/Luuk van Knippenberg/Github/shearWaveProcessing"
sys.path.insert(0, os.path.join(_REPO, "src"))
from swp.viz.io.loader import load_acquisition          # noqa: E402
from swp.viz.estimators.loupas import loupas_displacement  # noqa: E402
from swp.acquisition import discover_measurements        # noqa: E402

PH_ROOT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom"
IV_ROOT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Invivo"
IV = [("30 V", "Luuk30V_SW_data_04-August-2026_13-35-06"),
      ("40 V", "Luuk40V_SW_data_04-August-2026_13-36-07")]


def raw_disp(path):
    """Return (disp (n_t,nz,nx) um vs reference, x, z, t_ms, push_x, push_z, focus_profile_um)."""
    acq = load_acquisition(path)
    res = loupas_displacement(acq.iq, dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod,
                              prf=acq.prf, mode="relative_to_reference", reference=acq.ref_iq)
    disp = res.displacement * 1e6                # um
    t_ms = (acq.t - acq.t[0]) * 1e3
    return disp, acq.x, acq.z, t_ms, acq.push_x, acq.push_z


def focus_metrics(path, drop=2, rx_mm=2.5, rz_mm=3.0):
    """Peak & profile of raw focal displacement. Returns (peak_um, prof_um[n_t], t_ms, disp,x,z,px,pz)."""
    disp, x, z, t_ms, px, pz = raw_disp(path)
    if px is None:
        px = 0.0
    if pz is None:
        pz = z[len(z) // 2]
    rx = (np.abs(x - px) <= rx_mm * 1e-3)
    rz = (np.abs(z - pz) <= rz_mm * 1e-3)
    roi = disp[:, rz][:, :, rx]                  # (n_t, nz_roi, nx_roi)
    prof = np.nanmedian(roi, axis=(1, 2))        # focal displacement vs time (median over ROI)
    # peak transient after the push, ignoring the first `drop` (wrap-prone) frames
    peak = float(np.nanmax(np.abs(prof[drop:]))) if prof.size > drop else np.nan
    return peak, prof, t_ms, disp, x, z, px, pz


def main():
    # ---------- PHANTOM: amplitude vs voltage ----------
    ph = []
    for folder, pv in discover_measurements(PH_ROOT):
        p = os.path.join(folder, "output", "CombinedData_buffer2_meas0_iq.hdf5")
        if not os.path.exists(p):
            continue
        peak, prof, t_ms, *_ = focus_metrics(p)
        ph.append((pv.hv, peak, prof, t_ms))
        print(f"PHANTOM {pv.hv:>4.0f} V : peak focal |disp| = {peak:6.2f} um")
    ph.sort(key=lambda r: r[0])
    phv = np.array([r[0] for r in ph]); php = np.array([r[1] for r in ph])

    # ---------- IN-VIVO: amplitude vs voltage (24 pushes each) ----------
    iv = {}
    for lbl, fld in IV:
        peaks = []
        for m in range(24):
            p = os.path.join(IV_ROOT, fld, "output", f"CombinedData_buffer2_meas{m}_iq.hdf5")
            if not os.path.exists(p):
                peaks.append(np.nan); continue
            peak, *_ = focus_metrics(p)
            peaks.append(peak)
        iv[lbl] = np.array(peaks)
        v = iv[lbl][np.isfinite(iv[lbl])]
        print(f"IN-VIVO {lbl}: peak focal |disp| median {np.median(v):5.2f}  "
              f"mean {np.mean(v):5.2f}  IQR [{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}] um")

    # expected V^2 scaling anchored at the lowest phantom voltage
    v2 = php[0] * (phv / phv[0]) ** 2

    # ---------- FIGURE ----------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axs[0]
    ax.plot(phv, php, "o-", color="tab:green", label="phantom (measured)")
    ax.plot(phv, v2, "k--", lw=1, label=r"$\propto V^2$ (anchored at 15 V)")
    ax.set_xlabel("delivered push voltage (V)"); ax.set_ylabel("peak focal |disp| (um)")
    ax.set_title("PHANTOM: focal push displacement vs voltage")
    ax.grid(alpha=0.3); ax.legend()

    ax = axs[1]
    colors = {"30 V": "tab:blue", "40 V": "tab:red"}
    for j, (lbl, _) in enumerate(IV):
        v = iv[lbl][np.isfinite(iv[lbl])]
        bp = ax.boxplot([v], positions=[j], widths=0.6, patch_artist=True, showmeans=True)
        for b in bp["boxes"]:
            b.set(facecolor=colors[lbl], alpha=0.55)
        ax.plot(np.random.normal(j, 0.05, v.size), v, ".", color="0.2", ms=4)
    # overlay the phantom 30 & 40 V points for scale
    for volt in (30, 40):
        if volt in phv:
            ax.plot(0 if volt == 30 else 1, php[list(phv).index(volt)], "*",
                    color="tab:green", ms=16, label="phantom same V" if volt == 30 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels([l for l, _ in IV])
    ax.set_ylabel("peak focal |disp| (um)")
    ax.set_title("IN-VIVO: focal displacement, 30 vs 40 V\n(green * = phantom at same V)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    ax = axs[2]
    # focal displacement-vs-time: phantom 50V, phantom 20V, in-vivo 30V m6, in-vivo 40V m5
    def prof_of(path):
        _, prof, t_ms, *_ = focus_metrics(path)
        return t_ms, prof
    examples = [
        (os.path.join([f for f, pv in discover_measurements(PH_ROOT) if abs(pv.hv-50) < 1][0],
                      "output", "CombinedData_buffer2_meas0_iq.hdf5"), "phantom 50 V", "tab:green"),
        (os.path.join([f for f, pv in discover_measurements(PH_ROOT) if abs(pv.hv-20) < 1][0],
                      "output", "CombinedData_buffer2_meas0_iq.hdf5"), "phantom 20 V", "darkolivegreen"),
        (os.path.join(IV_ROOT, IV[0][1], "output", "CombinedData_buffer2_meas6_iq.hdf5"),
         "in-vivo 30 V m6", "tab:blue"),
        (os.path.join(IV_ROOT, IV[1][1], "output", "CombinedData_buffer2_meas5_iq.hdf5"),
         "in-vivo 40 V m5", "tab:red"),
    ]
    for p, lbl, col in examples:
        if os.path.exists(p):
            t_ms, prof = prof_of(p)
            ax.plot(t_ms, prof, color=col, label=lbl)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("tracking time since 1st track frame (ms)")
    ax.set_ylabel("focal displacement vs reference (um)")
    ax.set_title("Raw focal displacement vs time\n(real push = transient locked to push onset)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("ARF push reality check: does focal displacement scale with voltage? "
                 "(phantom = known-good control)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(IV_ROOT, "arf_push_reality_check.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")

    # numeric summary of scaling
    print("\n--- SCALING SUMMARY ---")
    if len(phv) >= 2:
        print(f"phantom peak disp 15->50 V: {php[0]:.2f} -> {php[-1]:.2f} um  "
              f"(x{php[-1]/php[0]:.1f}); V^2 predicts x{(phv[-1]/phv[0])**2:.1f}")
        for volt in (30, 40):
            if volt in phv:
                print(f"phantom {volt} V: {php[list(phv).index(volt)]:.2f} um")
    m30 = np.nanmedian(iv['30 V']); m40 = np.nanmedian(iv['40 V'])
    print(f"in-vivo median 30 V {m30:.2f} -> 40 V {m40:.2f} um (x{m40/m30:.2f}); "
          f"V^2 predicts x{(40/30)**2:.2f}")


if __name__ == "__main__":
    main()
