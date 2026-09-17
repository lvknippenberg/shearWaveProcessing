"""Passive SWE with each per-event M-line in full, and only its left or right half.

For every detected window the saved ``passive_win<i>_mline.npz`` is resampled along arc length and
cut at its midpoint; the half with the smaller mean x is "left". Each variant gets its own folder
``output/swp_passive/split_<variant>/`` with the lines, montage and ``passive_speeds.json`` (the
same processing as ``swp.passive.process_passive_windows``, detection reused), plus
``split_lines.png`` showing the halves on each event's B-mode frame and a speed table.

Usage:
    python study/analysis/passive_mline_split.py --folder <measurement folder>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np

import swp.passive as P
from swp.viz.mline import mline_from_points

VARIANTS = ("full", "left", "right")
N_ANCHORS = 8                    # anchor points kept to describe a half (a smooth sub-curve)


def split_line(ml, n_samples):
    """-> {"full", "left", "right"}: point arrays (k, 2) in metres, orientation preserved."""
    half = len(ml.x) // 2
    parts = {"a": (0, half + 1), "b": (half, len(ml.x))}
    pts = {}
    for name, (i0, i1) in parts.items():
        idx = np.linspace(i0, i1 - 1, N_ANCHORS).round().astype(int)
        pts[name] = np.column_stack([ml.x[idx], ml.z[idx]])
    a_left = pts["a"][:, 0].mean() < pts["b"][:, 0].mean()
    return {"full": np.asarray(ml.points, float),
            "left": pts["a"] if a_left else pts["b"],
            "right": pts["b"] if a_left else pts["a"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--config", default=str(_REPO / "configs" / "passive.yaml"))
    a = ap.parse_args()

    cfg, p = P._paths(a.folder, a.config)
    n_samples = cfg["mline"].get("n_samples", 250)
    st, windows = P.read_windows(p["windows_json"])
    lines = {i: P._load_line(P._window_npz(p["mlines"], i), n_samples)
             for i in range(len(windows)) if os.path.exists(P._window_npz(p["mlines"], i))}
    splits = {i: split_line(ml, n_samples) for i, ml in lines.items()}

    acq = P.load_acq(a.folder, a.config)
    original_paths = P._paths
    for v in VARIANTS:
        vdir = os.path.join(p["outdir"], f"split_{v}")
        shutil.rmtree(vdir, ignore_errors=True)
        os.makedirs(os.path.join(vdir, "mlines"))
        for i, parts in splits.items():
            np.savez(os.path.join(vdir, "mlines", f"passive_win{i}_mline.npz"),
                     points=parts[v], n_samples=n_samples)
        shutil.copy(p["windows_json"], os.path.join(vdir, P.WINDOWS_JSON))

        def _paths(folder, config, _vdir=vdir):
            c, q = original_paths(folder, config)
            q.update(mlines=os.path.join(_vdir, "mlines"), outdir=_vdir,
                     windows_json=os.path.join(_vdir, P.WINDOWS_JSON),
                     montage=os.path.join(_vdir, P.MONTAGE_PNG))
            return c, q

        P._paths = _paths
        try:
            print(f"\n===== variant: {v} =====", flush=True)
            P.process_passive_windows(a.folder, a.config, acq=acq)
        finally:
            P._paths = original_paths

    # --- figure: the three variants on each event's B-mode frame ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from swp.mline.select import load_bmode_frame, _grid_axes

    wm = st.get("window_mlines", {})
    idx = sorted(splits)
    fig, axs = plt.subplots(1, len(idx), figsize=(5 * len(idx), 5.5), squeeze=False)
    colors = dict(full="white", left="cyan", right="orange")
    for ax, i in zip(axs[0], idx):
        info = wm.get(str(i), {})
        buf, frame = info.get("buffer", 1), info.get("frame", 0)
        img, coords, _ = load_bmode_frame(os.path.join(p["output"], P.bmode_file(buf)), frame)
        xs, zs = _grid_axes(coords)
        ax.imshow(img, cmap="gray", extent=[xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3],
                  aspect="auto")
        for v, lw, ls in (("full", 5, "-"), ("left", 2, "-"), ("right", 2, "-")):
            ml = mline_from_points(splits[i][v], n_samples)
            ax.plot(ml.x * 1e3, ml.z * 1e3, ls, color=colors[v], lw=lw, alpha=0.5 if v == "full" else 1,
                    label=f"{v} ({ml.r[-1] * 1e3:.0f} mm)")
        ax.plot(*(splits[i]["full"][0] * 1e3), "o", color="yellow", ms=6)     # r = 0
        w = windows[i]
        ax.set_title(f"win{i} @ {w.t_peak * 1e3:.0f} ms, buffer {buf} frame {frame}")
        ax.set_xlim(-50, 50); ax.set_ylim(90, 20)
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig_path = os.path.join(p["outdir"], "split_lines.png")
    fig.savefig(fig_path, dpi=110)

    # --- speed table ---
    rows = {}
    for v in VARIANTS:
        with open(os.path.join(p["outdir"], f"split_{v}", "passive_speeds.json")) as f:
            for r in json.load(f):
                rows.setdefault((r["window"], r["view"]), {})[v] = (r["speed_m_s"], r["semblance"])
    lines_out = ["window  view                                  " + "".join(f"{v:>18s}" for v in VARIANTS)]
    for (i, view), d in sorted(rows.items()):
        cells = "".join(f"{d[v][0]:+8.2f} ({d[v][1]:.2f})" if v in d else " " * 18 for v in VARIANTS)
        lines_out.append(f"win{i:<4d} {view:<38s}{cells}")
    table = "\n".join(lines_out)
    print("\nspeed m/s (semblance)\n" + table)
    with open(os.path.join(p["outdir"], "split_speeds.txt"), "w") as f:
        f.write("speed m/s (semblance); r = 0 at the first anchor of the full line (yellow dot)\n")
        f.write(table + "\n")
    print(f"\n-> {fig_path}")


if __name__ == "__main__":
    main()
