"""Atlas of the passive-SWE processing methods: the same in-vivo windows under every recipe.

Figures for ``report/passive_methods/passive_methods.tex``. Space-time panels only - no automatic
speed is fitted or drawn (the automatic fit is not what we trust; see
``docs/passive_speed_estimation.md``).

Windows: the six labelled windows scored **clear** (3 aortic-valve closure, 3 mitral-valve
closure), each on the M-line and M-line part it was hand-labelled on, plus one window scored
**none** as a reference for what "no wave" looks like. The time window is the detected event
+/- 20 ms, as in the passive montages.

Every recipe starts from view A of ``configs/passive.yaml`` (Loupas frame-to-frame, displacement,
10-150 Hz band-pass, Gaussian 0.6 x 1.2 mm, moving mean 3, 5 M-lines x 0.5 mm, no directional
filter) and changes one thing, except the 'literature recipes' and 'production views' families.
The IQ is cropped to the M-line box + 8 mm before processing (speed; the SVD clutter filter is
therefore local to that box).

    python study/analysis/passive_methods_atlas.py            # build cache (network reads) + figures
    python study/analysis/passive_methods_atlas.py --figures  # figures only, from the cache
-> study/analysis/atlas_cache/*.npz (not tracked), report/passive_methods/figures/*.png
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))
os.environ.setdefault("KERAS_BACKEND", "torch")

from swp import paths as P                                     # noqa: E402

CACHE = _REPO / "study" / "analysis" / "atlas_cache"
FIGDIR = _REPO / "report" / "passive_methods" / "figures"
WINDOWS = [("C000000003", 1, "full"), ("C000000019", 1, "left"), ("C000000020", 1, "left"),
           ("C000000002", 0, "right"), ("C000000027", 0, "right"), ("C000000039", 0, "right"),
           ("C000000008", 2, "right")]            # last: scored "none" - the no-wave reference
PAD_S = 0.02
CROP_M = 8e-3
CFWI_EDGE = 10       # frames (~11 ms) dropped at each end of CFWI panels


def S(name, **p):
    from swp.viz.pipeline import Step
    return Step(name, p)


def bp(lo, hi, order=2):
    return S("temporal_bandpass", f_lo=lo, f_hi=hi, order=order)


GAUSS = S("spatial_smooth", sigma_z_m=0.6e-3, sigma_x_m=1.2e-3)
MEAN3 = S("temporal_moving_mean", window=3)


def families():
    """{family: [(row label, overrides of view A)]}. Overrides are PipelineConfig fields."""
    return {
        "quantity": [
            ("displacement", dict()),
            ("velocity", dict(quantity="velocity")),
            ("acceleration", dict(quantity="acceleration")),
        ],
        "band_displacement": [(f"{lo}-{hi} Hz", dict(field_filters=[bp(lo, hi), GAUSS, MEAN3]))
                              for lo, hi in ((5, 150), (10, 150), (15, 100), (15, 90), (8, 45), (30, 150))],
        "band_velocity": [(f"{lo}-{hi} Hz", dict(quantity="velocity", field_filters=[bp(lo, hi), GAUSS, MEAN3]))
                          for lo, hi in ((5, 150), (10, 150), (15, 100), (15, 90), (8, 45), (30, 150))],
        "directional": [
            ("displacement, none", dict()),
            ("displacement, keep toward low r", dict(directional=True, directional_mode="leftward")),
            ("displacement, keep toward high r", dict(directional=True, directional_mode="rightward")),
            ("velocity, none", dict(quantity="velocity")),
            ("velocity, keep toward low r", dict(quantity="velocity", directional=True, directional_mode="leftward")),
            ("velocity, keep toward high r", dict(quantity="velocity", directional=True, directional_mode="rightward")),
        ],
        "spatial": [
            ("none", dict(field_filters=[bp(10, 150), MEAN3])),
            ("Gaussian 0.6 x 1.2 mm (view A)", dict()),
            ("Gaussian 1.0 x 2.0 mm", dict(field_filters=[bp(10, 150), S("spatial_smooth", sigma_z_m=1e-3, sigma_x_m=2e-3), MEAN3])),
            ("Gaussian 2.0 x 4.0 mm", dict(field_filters=[bp(10, 150), S("spatial_smooth", sigma_z_m=2e-3, sigma_x_m=4e-3), MEAN3])),
            ("median 1.0 x 2.0 mm", dict(field_filters=[bp(10, 150), S("spatial_median", size_z_m=1e-3, size_x_m=2e-3), MEAN3])),
        ],
        "temporal": [
            ("none", dict(field_filters=[bp(10, 150), GAUSS])),
            ("moving mean 3 (view A)", dict()),
            ("moving mean 5", dict(field_filters=[bp(10, 150), GAUSS, S("temporal_moving_mean", window=5)])),
            ("moving median 5", dict(field_filters=[bp(10, 150), GAUSS, S("temporal_moving_median", window=5)])),
        ],
        "mline": [
            ("1 line", dict(mline_offsets=1)),
            ("5 lines x 0.5 mm (view A)", dict()),
            ("9 lines x 0.8 mm", dict(mline_offsets=9, mline_offset_step_m=0.8e-3)),
            ("15 lines x 0.8 mm", dict(mline_offsets=15, mline_offset_step_m=0.8e-3)),
        ],
        "motion": [
            ("band-pass 10-150 (view A)", dict()),
            ("+ IQ slow-time low-pass 250 Hz", dict(iq_filters=[S("iq_slowtime_lowpass", fc_hz=250.0, order=3)])),
            ("+ IQ SVD clutter (1 component)", dict(iq_filters=[S("svd_clutter", n_remove=1)])),
            ("polynomial detrend (order 3) + low-pass 150", dict(field_filters=[S("polynomial_drift", order=3), bp(0, 150), GAUSS, MEAN3])),
            ("high-pass 5 Hz only (bulk motion kept)", dict(field_filters=[bp(5, 400), GAUSS, MEAN3])),
        ],
        "literature": [
            ("Keijzer 2019/20: IQ low-pass 250, velocity, R1 Gauss 4 mm, 15-100 Hz",
             dict(quantity="velocity", iq_filters=[S("iq_slowtime_lowpass", fc_hz=250.0, order=3)],
                  estimator_params=dict(kernel_z_m=4e-3, kernel_x_m=3e-3, kernel_shape="gaussian"),
                  field_filters=[bp(15, 100, 3)])),
            ("Petrescu / Santos: acceleration, velocity mean 3",
             dict(quantity="acceleration", field_filters=[S("temporal_moving_mean", window=3), bp(10, 150), GAUSS])),
            ("Espeland 2024 CFWI: 2 cm/s clutter filter, envelope derivative",
             dict(estimator="cfwi", estimator_params=dict(cutoff_velocity=0.02), quantity="velocity",
                  field_filters=[S("spatial_smooth", sigma_z_m=1e-3, sigma_x_m=2e-3), MEAN3])),
        ],
        "views": [
            ("A: displacement, 10-150, Gauss 0.6, mean 3, 5 lines", dict()),
            ("B: displacement, 5-150, median 1 x 2, no temporal, 9 lines",
             dict(field_filters=[bp(5, 150), S("spatial_median", size_z_m=1e-3, size_x_m=2e-3)],
                  mline_offsets=9, mline_offset_step_m=0.5e-3)),
            ("C: velocity, 15-90, Gauss 1.0 x 2.0, mean 5, 9 lines",
             dict(quantity="velocity", field_filters=[bp(15, 90), S("spatial_smooth", sigma_z_m=1e-3, sigma_x_m=2e-3),
                                                     S("temporal_moving_mean", window=5)],
                  mline_offsets=9, mline_offset_step_m=0.8e-3)),
        ],
    }


def _crop(acq, ml):
    ix = np.where((acq.x >= ml.x.min() - CROP_M) & (acq.x <= ml.x.max() + CROP_M))[0]
    iz = np.where((acq.z >= ml.z.min() - CROP_M) & (acq.z <= ml.z.max() + CROP_M))[0]
    coords = acq.coords[iz][:, ix] if acq.coords is not None else None
    return dataclasses.replace(acq, x=acq.x[ix], z=acq.z[iz], iq=acq.iq[:, iz][:, :, ix], coords=coords)


def build(config, force=False):
    import swp.passive as SP
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline
    from passive_mline_split import split_line
    labelled = {(c["subject"], c["window"], c["part"]): c
                for c in json.load(open(_REPO / "study/logs/labelled_panels.json"))}
    CACHE.mkdir(parents=True, exist_ok=True)
    fams = families()
    for subj, win, part in WINDOWS:
        out = CACHE / f"{subj}_win{win}_{part}.npz"
        if out.exists() and not force:
            continue
        c = labelled[(subj, win, part)]
        folder = f"{P.RAW_DATA}/{c['folder']}"
        print(f"{subj} win{win} {part}", flush=True)
        cfg, p = SP._paths(folder, config)
        st, ws = SP.read_windows(p["windows_json"])
        w = ws[win]
        acq = SP.load_acq(folder, config)
        n = cfg["mline"].get("n_samples", 250)
        ml_full = SP._load_line(SP._window_npz(p["mlines"], win), n)
        ml = ml_full if part == "full" else mline_from_points(split_line(ml_full, n)[part], n)
        bm = SP._row_bmode(p, st, win, w, ml_full, acq)
        i0 = SP._frame_at_time(acq.t, w.t0 - PAD_S)
        i1 = SP._frame_at_time(acq.t, w.t1 + PAD_S) + 1
        acq_w = _crop(dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1]), ml)
        del acq
        base = SP._build_views(cfg, acq_w)[0][1]
        panels = {}
        for fam, rows in fams.items():
            for label, ov in rows:
                key = f"{fam}|{label}"
                try:
                    res = run_pipeline(acq_w, ml, dataclasses.replace(base, **ov), focus=None)
                    panels[key] = (res.st.data.astype(np.float32), res.st.t)
                except Exception as exc:                        # noqa: BLE001
                    print(f"   {key}: FAILED {type(exc).__name__}: {exc}", flush=True)
        np.savez_compressed(
            out, keys=np.array(list(panels)), r=ml.r, t_peak=w.t_peak, label=str(w.label or "?"),
            confidence=str(c.get("confidence", "")), mline_mm=float(ml.r[-1] * 1e3),
            bm_img=bm["img"] if bm else np.zeros((2, 2)), bm_extent=np.array(bm["extent"] if bm else [0, 1, 1, 0]),
            ml_x=ml_full.x * 1e3, ml_z=ml_full.z * 1e3, part_x=ml.x * 1e3, part_z=ml.z * 1e3,
            **{f"d{i}": v[0] for i, v in enumerate(panels.values())},
            **{f"t{i}": v[1] for i, v in enumerate(panels.values())})
        print(f"   {len(panels)} panels -> {out.name}", flush=True)


def _load():
    out = []
    for subj, win, part in WINDOWS:
        f = CACHE / f"{subj}_win{win}_{part}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        keys = list(z["keys"])
        out.append(dict(name=f"{subj[-3:]} {str(z['label'])} ({part})", z=z,
                        panels={k: (z[f"d{i}"], z[f"t{i}"]) for i, k in enumerate(keys)},
                        none=(subj, win, part) == WINDOWS[-1]))
    return out


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGDIR.mkdir(parents=True, exist_ok=True)
    wins = _load()
    ncol = len(wins)
    # anatomy: the B-mode each line was drawn on, with the analysed part of the line in yellow
    fig, axes = plt.subplots(1, ncol, figsize=(2.3 * ncol, 2.6), squeeze=False)
    for ax, W in zip(axes[0], wins):
        z = W["z"]
        ax.imshow(z["bm_img"], cmap="gray", extent=tuple(z["bm_extent"]), aspect="auto")
        ax.plot(z["ml_x"], z["ml_z"], "-", color="0.7", lw=0.8)
        ax.plot(z["part_x"], z["part_z"], "-", color="yellow", lw=1.6)
        cx, cz = np.mean(z["part_x"]), np.mean(z["part_z"])
        ax.set_xlim(cx - 35, cx + 35); ax.set_ylim(cz + 30, cz - 30)
        ax.set_title(W["name"] + ("\nno-wave reference" if W["none"] else f"\n{float(z['mline_mm']):.0f} mm line"),
                     fontsize=7)
        ax.tick_params(labelsize=5)
    fig.tight_layout(); fig.savefig(FIGDIR / "windows.png", dpi=120); plt.close(fig)

    for fam, rows in families().items():
        labels = [lab for lab, _ in rows]
        fig, axes = plt.subplots(len(labels), ncol, figsize=(2.3 * ncol, 1.55 * len(labels) + 0.5),
                                 squeeze=False)
        for j, W in enumerate(wins):
            tp = float(W["z"]["t_peak"])
            r = W["z"]["r"] * 1e3
            for i, lab in enumerate(labels):
                ax = axes[i, j]
                d, t = W["panels"].get(f"{fam}|{lab}", (None, None))
                if d is None:
                    ax.axis("off"); continue
                if "CFWI" in lab:        # drop the slow-time high-pass end transient (as cfwi_benchmark)
                    d, t = d[CFWI_EDGE:-CFWI_EDGE], t[CFWI_EDGE:-CFWI_EDGE]
                clim = np.percentile(np.abs(d), 99) or 1.0
                ax.imshow(d.T, aspect="auto", cmap="RdBu_r", vmin=-clim, vmax=clim, origin="lower",
                          extent=((t[0] - tp) * 1e3, (t[-1] - tp) * 1e3, r[0], r[-1]))
                ax.axvline(0, color="k", lw=0.4, ls=":")
                ax.tick_params(labelsize=5, length=2)
                if i == 0:
                    ax.set_title(W["name"] + (" [none]" if W["none"] else ""), fontsize=6.5)
                if j == 0:
                    ax.set_ylabel(lab if len(lab) < 30 else lab[:28] + "...", fontsize=5.5)
                if i == len(labels) - 1:
                    ax.set_xlabel("t - event [ms]", fontsize=5.5)
        fig.tight_layout(h_pad=0.3, w_pad=0.3)
        fig.savefig(FIGDIR / f"{fam}.png", dpi=120)
        plt.close(fig)
        print("wrote", FIGDIR / f"{fam}.png")


def _hand_centre_ms(subj, win, part):
    """Time [ms] of the hand-drawn wavefront's midpoint (panel_cache), or None."""
    f = _REPO / "study" / "analysis" / "panel_cache" / f"{subj}_win{win}_{part}_displacement.npz"
    if not f.exists():
        return None
    pts = np.load(f)["points"]
    return float(np.mean(pts[:, 0])) if np.any(pts) else None


def zoom_quantity(half_ms=25.0):
    """Displacement / velocity / acceleration zoomed to +/- half_ms around the hand-drawn wavefront
    (event peak for the no-wave window): on the full 130 ms axis a 3 m/s wave crosses a 20 mm line
    in ~7 ms and its tilt is hard to see."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wins = _load()
    rows = [lab for lab, _ in families()["quantity"]]
    fig, axes = plt.subplots(len(rows), len(wins), figsize=(2.3 * len(wins), 1.9 * len(rows) + 0.5), squeeze=False)
    for j, (W, key) in enumerate(zip(wins, WINDOWS)):
        tp = float(W["z"]["t_peak"]) * 1e3
        tc = _hand_centre_ms(*key)
        tc = tp if (tc is None or W["none"]) else tc
        r = W["z"]["r"] * 1e3
        for i, lab in enumerate(rows):
            ax = axes[i, j]
            d, t = W["panels"][f"quantity|{lab}"]
            tm = t * 1e3
            sel = (tm >= tc - half_ms) & (tm <= tc + half_ms)
            dd = d[sel]
            clim = np.percentile(np.abs(dd), 99) or 1.0
            ax.imshow(dd.T, aspect="auto", cmap="RdBu_r", vmin=-clim, vmax=clim, origin="lower",
                      extent=(tm[sel][0] - tc, tm[sel][-1] - tc, r[0], r[-1]))
            ax.tick_params(labelsize=5, length=2)
            if i == 0:
                ax.set_title(W["name"] + (" [none]" if W["none"] else ""), fontsize=6.5)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6)
            if i == len(rows) - 1:
                ax.set_xlabel("t - hand wavefront [ms]", fontsize=5.5)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fig.savefig(FIGDIR / "quantity_zoom.png", dpi=120)
    plt.close(fig)
    print("wrote", FIGDIR / "quantity_zoom.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(_REPO / "configs" / "passive.yaml"))
    ap.add_argument("--figures", action="store_true", help="figures from the cache only")
    ap.add_argument("--force", action="store_true", help="rebuild cached windows")
    a = ap.parse_args()
    if not a.figures:
        build(a.config, a.force)
    figures()
    zoom_quantity()


if __name__ == "__main__":
    main()
