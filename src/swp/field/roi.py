"""Draw, store and apply the region of interest the field estimator works over.

This replaces the M-line as the user input. The difference is the point of the whole approach:
an M-line is an *oriented* line, so it fixes both where to look and which direction to measure
along - and the second of those is what makes every 1-D estimate an apparent speed `c/cos(theta)`.
An ROI fixes only where to look; direction becomes an output.

Practical consequences:

* one ROI per acquisition rather than one line per cardiac event;
* an ROI drawn 10 % too large changes the averaging, whereas an M-line drawn 10 degrees off
  changes the speed by 1/cos(10 deg);
* it is a segmentation problem, so it can be automated later - a direction never could be.

The ROI is a polygon in (x, z) on the same B-mode frame the M-line is drawn on, stored next to
the existing M-line artefacts as ``mlines/field_roi.npz``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

__all__ = ["ROI", "load_roi", "save_roi", "roi_from_mline", "select_roi", "ensure_roi"]

ROI_NPZ = "field_roi.npz"


@dataclass
class ROI:
    """A polygon in physical coordinates, plus the frame it was drawn on."""
    points: np.ndarray          # (n, 2) of (x, z) in metres, in order
    source: str = ""            # how it was made, for provenance
    frame: int = -1             # B-mode frame it was drawn on

    def mask(self, x, z):
        """Boolean (n_z, n_x) mask of the pixels inside the polygon."""
        from matplotlib.path import Path as MplPath

        X, Z = np.meshgrid(np.asarray(x, float), np.asarray(z, float))
        pts = np.column_stack([X.ravel(), Z.ravel()])
        inside = MplPath(np.asarray(self.points, float)).contains_points(pts)
        return inside.reshape(X.shape)

    def bbox_mm(self):
        p = np.asarray(self.points, float) * 1e3
        return (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())

    def extent_mm(self):
        """(length, thickness) of the ROI along its own principal axes, in mm.

        Reported because Stage 1 showed the estimator's oblique bias depends on the *thickness*:
        <= 0.5 % in a 24 mm ROI against +7.3 % at 75 deg in a 12 mm one.
        """
        p = np.asarray(self.points, float) * 1e3
        c = p - p.mean(axis=0)
        u, s, _ = np.linalg.svd(c - c.mean(axis=0), full_matrices=False)
        proj = c @ np.linalg.svd(c, full_matrices=False)[2].T
        return float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]))


def save_roi(path, roi):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, points=np.asarray(roi.points, float), source=roi.source, frame=roi.frame)
    return path


def load_roi(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    return ROI(points=d["points"], source=str(d["source"]) if "source" in d else "",
               frame=int(d["frame"]) if "frame" in d else -1)


def roi_from_mline(mline, thickness_mm=12.0):
    """A band of given thickness centred on an existing M-line.

    Not the intended long-term input - the point of the field approach is to stop supplying a
    direction - but it lets Stage 3 run on the 36 folders that already have per-event M-lines,
    without redrawing anything. The band is centred on the line and perpendicular to it, so the
    ROI contains the wall the operator already identified.
    """
    x, z = np.asarray(mline.x, float), np.asarray(mline.z, float)
    d = np.array([x[-1] - x[0], z[-1] - z[0]])
    n = np.linalg.norm(d)
    if n <= 0:
        raise ValueError("degenerate M-line")
    perp = np.array([-d[1], d[0]]) / n * (thickness_mm * 1e-3 / 2.0)
    side_a = np.column_stack([x + perp[0], z + perp[1]])
    side_b = np.column_stack([x - perp[0], z - perp[1]])[::-1]
    return ROI(points=np.vstack([side_a, side_b]), source=f"mline+{thickness_mm:.0f}mm")


def select_roi(bmode_u8, coords, title="", n_vertices=0):
    """Draw a polygon on a B-mode frame. Click vertices, ENTER to close, 'r' to restart."""
    import matplotlib.pyplot as plt

    from ..mline.select import _grid_axes

    xs, zs = _grid_axes(coords)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(bmode_u8, cmap="gray",
              extent=[xs[0] * 1e3, xs[-1] * 1e3, zs[-1] * 1e3, zs[0] * 1e3], aspect="equal")
    ax.set_title(f"{title}\nclick the wall outline; ENTER accepts, 'r' restarts", fontsize=9)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    pts, drawn = [], []

    def redraw():
        for h in drawn:
            h.remove()
        drawn.clear()
        if pts:
            a = np.array(pts)
            drawn.append(ax.plot(a[:, 0], a[:, 1], "o-", color="lime", ms=5, lw=1.5)[0])
            if len(pts) > 2:
                drawn.append(ax.plot([a[-1, 0], a[0, 0]], [a[-1, 1], a[0, 1]], "--",
                                     color="lime", lw=1.0)[0])
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is ax and ev.button == 1:
            pts.append((float(ev.xdata), float(ev.ydata)))
            redraw()

    state = {"done": False}

    def on_key(ev):
        if ev.key == "r":
            pts.clear()
            redraw()
        elif ev.key in ("enter", "return") and len(pts) >= 3:
            state["done"] = True
            fig.canvas.stop_event_loop()

    cids = [fig.canvas.mpl_connect("button_press_event", on_click),
            fig.canvas.mpl_connect("key_press_event", on_key),
            fig.canvas.mpl_connect("close_event", lambda _e: fig.canvas.stop_event_loop())]
    fig.tight_layout()
    plt.show(block=False)
    fig.canvas.start_event_loop(timeout=-1)
    for c in cids:
        fig.canvas.mpl_disconnect(c)
    plt.close(fig)
    if not state["done"] or len(pts) < 3:
        return None
    return ROI(points=np.array(pts, float) * 1e-3, source="drawn")


def ensure_roi(folder, config="configs/passive.yaml", thickness_mm=12.0, redraw=False):
    """Load the stored ROI, else derive one from the existing M-line, else ask for one.

    Deriving from the M-line is the pragmatic path for Stage 3: it reuses the wall the operator
    has already identified on all 36 processed folders. Whether that band is good enough, or a
    drawn ROI is materially better, is itself a question for Stage 3 to answer.
    """
    from ..passive import _load_line, _paths

    _, p = _paths(folder, config)
    path = os.path.join(p["mlines"], ROI_NPZ)
    if not redraw:
        roi = load_roi(path)
        if roi is not None:
            return roi, path
    npz = os.path.join(p["mlines"], "passive_mline.npz")
    if os.path.exists(npz):
        roi = roi_from_mline(_load_line(npz, 250), thickness_mm=thickness_mm)
        save_roi(path, roi)
        return roi, path
    return None, path
