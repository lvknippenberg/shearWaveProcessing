"""Draw the shear-wave "V" ROI once on the clear 50 V phantom space-time plot; it is fitted, saved,
and then LOCKED as the template for detecting the wave at lower voltages (same phantom -> same V
geometry, so no per-measurement speed fit).

Click points along the V wavefront ridges (either lobe; both sides are handled via |r - r0|), then
press Enter. A symmetric V ``t = t0 + |r - r0| / c`` is fit (r0 = the push crossing, fixed), drawn with
its +/- band, and saved to ``metric_experiment/v_roi_template.json``.

    KERAS_BACKEND=torch  <zea-python>  scripts/draw_v_roi.py [--voltage 50V] [--quantity velocity] [--band-ms 1.2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "TkAgg")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))

import numpy as np

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
PH = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom"
FOLDER = {"50V": "/DefaultPatient_SW_data_04-August-2026_13-21-52",
          "40V": "/DefaultPatient_SW_data_04-August-2026_13-25-06",
          "30V": "/DefaultPatient_SW_data_04-August-2026_13-27-38"}


def main():
    import core
    from swp.viz.core.geometry import robust_clim
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--voltage", default="50V")
    ap.add_argument("--quantity", default="velocity", help="draw on velocity (sharp V) by default")
    ap.add_argument("--band-ms", type=float, default=1.2)
    a = ap.parse_args()

    folder = PH + FOLDER[a.voltage]
    rec = core.Recipe(mline_source="horizontal_push",
                      motion_steps=[("temporal_bandpass", {"f_lo": 80, "f_hi": 600})],
                      spatial_steps=[("spatial_smooth", {"sigma_z_m": 700, "sigma_x_m": 1400})],
                      temporal_steps=[("temporal_moving_mean", {"window": 5})],
                      directional="outward", offsets=9, quantity=a.quantity)
    acq = core.load_acq(folder, 0, core.Recipe(mline_source="horizontal_push"))
    ml = core.load_mline_for(folder, 0, acq, core.Recipe(mline_source="horizontal_push"))
    import dataclasses
    res = core.run_recipe(acq, ml, core.to_config(dataclasses.replace(rec), acq))
    st = res.st; r0 = res.r0
    r = st.r * 1e3; t = st.t * 1e3
    unit = 1e3 if a.quantity == "velocity" else (1e6 if a.quantity == "displacement" else 1.0)
    rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
    cl = (robust_clim(st.data, rc, 97) * unit) or 1.0

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(st.data * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r", vmin=-cl, vmax=cl,
              aspect="auto", origin="upper")
    ax.axvline(r0 * 1e3, color="0.2", ls="--", lw=1)
    ax.set_xlabel("r [mm]"); ax.set_ylabel("t [ms]")
    ax.set_title(f"{a.voltage} {a.quantity}: click along the V wavefront (either/both lobes), then ENTER")
    print("\n>>> Click points ON the V wavefront ridges (either lobe; both are handled). "
          "Right-click removes the last; press ENTER when done.\n")
    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)
    if len(pts) < 2:
        raise SystemExit("need >= 2 points on the wavefront")

    rr = np.array([p[0] for p in pts]) * 1e-3          # m
    tt = np.array([p[1] for p in pts]) * 1e-3          # s
    dabs = np.abs(rr - r0)
    # linear fit t = t0 + slope*|r-r0|, slope = 1/c
    A = np.vstack([dabs, np.ones_like(dabs)]).T
    (slope, t0), *_ = np.linalg.lstsq(A, tt, rcond=None)
    c = 1.0 / slope if slope > 1e-9 else np.nan
    template = {"voltage": a.voltage, "quantity": a.quantity, "r0_m": float(r0),
                "c_mps": float(c), "t0_s": float(t0), "band_s": float(a.band_ms * 1e-3),
                "d_min_m": 2e-3, "d_max_m": 16e-3}
    with open(os.path.join(BASE, "v_roi_template.json"), "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    print(f"fitted V: c={c:.2f} m/s, t0={t0*1e3:.2f} ms, band=+/-{a.band_ms} ms  (r0={r0*1e3:.1f} mm)")

    # show the fitted V + band as a record
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt2
    fig2, ax2 = plt2.subplots(figsize=(9, 7))
    ax2.imshow(st.data * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r", vmin=-cl, vmax=cl,
               aspect="auto", origin="upper")
    line = (t0 + np.abs(st.r - r0) / c) * 1e3
    for band in (0, a.band_ms, -a.band_ms):
        ls = "-" if band == 0 else ":"
        ax2.plot(st.r * 1e3, line + band, "k", ls=ls, lw=1.2)
    ax2.plot(rr * 1e3, tt * 1e3, "yo", ms=5, mec="k")
    ax2.axvline(r0 * 1e3, color="0.2", ls="--", lw=1)
    ax2.set_xlabel("r [mm]"); ax2.set_ylabel("t [ms]")
    ax2.set_title(f"Locked V-ROI template  c={c:.2f} m/s  t0={t0*1e3:.1f} ms  band+/-{a.band_ms} ms")
    out = os.path.join(BASE, "v_roi_template.png")
    fig2.tight_layout(); fig2.savefig(out, dpi=140); plt2.close(fig2)
    print(f"saved template -> {os.path.join(BASE, 'v_roi_template.json')}  (+ {out})")


if __name__ == "__main__":
    main()
