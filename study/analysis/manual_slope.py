"""Draw the wavefront slope on a passive space-time plot by hand, and read the speed off it.

The automatic slant-stack fit disagrees between views on most windows (docs/passive_mlines.md), and
on a short segment it can lock onto a nearly-vertical band. For a figure - or whenever the wavefront
is obvious to the eye but not to the fit - this lets the slope be drawn directly on the M-mode panel.

The panel is the same one the montage shows: x = time (ms), y = along-line position (mm), so a
straight wavefront has slope dr/dt and the speed is simply |dr/dt| in mm/ms = m/s. Sign follows the
montage convention: + travels toward increasing r (away from the r=0 end of the M-line).

Click two points on the wavefront; the line, its speed, and the automatic fit for comparison are
drawn live. 'r' clears, ENTER accepts and moves to the next panel, closing the window skips it.
Picks are stored per (window, part, view) in ``output/swp_passive/manual_slopes.json`` and reused on
a later run unless ``--redraw`` is given, so the figure can be re-rendered without redrawing.

Usage:
    python study/analysis/manual_slope.py --folder "<folder>" --window 0 --part left
    python study/analysis/manual_slope.py --folder "<folder>" --window 0 --part left --figure out.png
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))

import numpy as np

import swp.passive as P
from swp.viz.core.geometry import robust_clim
from swp.viz.metrics import slant_stack_speed
from swp.viz.mline import mline_from_points
from swp.viz.pipeline import run_pipeline

PICKS_JSON = "manual_slopes.json"


def _load_picks(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _key(window, part, view):
    return f"win{window}|{part}|{view}"


def build_spacetime(folder, config, window, part, pad_ms=20.0):
    """-> (acq, window, mline, [(view name, PipelineResult, auto (sem, c))])."""
    from passive_mline_split import split_line

    cfg, p = P._paths(folder, config)
    n_samples = cfg["mline"].get("n_samples", 250)
    st, windows = P.read_windows(p["windows_json"])
    if st is None or window >= len(windows):
        raise SystemExit(f"window {window} not found in {p['windows_json']}")
    w = windows[window]
    npz = P._window_npz(p["mlines"], window)
    if not os.path.exists(npz):
        raise SystemExit(f"no M-line for window {window} (it was skipped): {npz}")
    ml_full = P._load_line(npz, n_samples)
    ml = ml_full if part == "full" else mline_from_points(
        split_line(ml_full, n_samples)[part], n_samples)

    acq = P.load_acq(folder, config)
    views = P._build_views(cfg, acq)
    pad_s = pad_ms * 1e-3
    i0 = P._frame_at_time(acq.t, w.t0 - pad_s)
    i1 = P._frame_at_time(acq.t, w.t1 + pad_s) + 1
    acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
    out = []
    for vname, vcfg in views:
        res = run_pipeline(acq_w, ml, vcfg, focus=None)
        sem, c = slant_stack_speed(res.st, res.r0, cmin=1.0, cmax=P.SPEED_CMAX, remove_flat=False)
        out.append((vname, res, (float(sem), float(c))))
    return acq, w, ml, out


def draw_panel(ax, st, title):
    """The montage's M-mode panel: x = time (ms), y = along-line (mm), r = 0 at the top."""
    unit = 1e3 if st.quantity == "velocity" else 1e6
    img = st.data * unit
    rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
    clim = robust_clim(st.data, rc, pct=97) * unit
    ax.imshow(img.T, extent=[st.t[0] * 1e3, st.t[-1] * 1e3, st.r[-1] * 1e3, st.r[0] * 1e3],
              cmap="RdBu_r", vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    ax.set_xlabel("t [ms]"); ax.set_ylabel("r along M-line [mm]")
    ax.set_title(title, fontsize=9)


class SlopePicker:
    """Two clicks -> a straight wavefront; 'r' clears, ENTER accepts, closing skips."""

    def __init__(self, ax, fig, auto=None, existing=None):
        self.ax, self.fig, self.auto = ax, fig, auto
        self.pts = list(existing) if existing else []
        self.line = None
        self.marks = []
        self.accepted = False
        self.skipped = False
        self.cids = [fig.canvas.mpl_connect("button_press_event", self.on_click),
                     fig.canvas.mpl_connect("key_press_event", self.on_key),
                     fig.canvas.mpl_connect("close_event", self.on_close)]
        self.redraw()

    # -- geometry ------------------------------------------------------------
    def speed(self):
        if len(self.pts) < 2:
            return float("nan")
        (t1, r1), (t2, r2) = self.pts[:2]
        dt = t2 - t1
        return float("nan") if abs(dt) < 1e-9 else (r2 - r1) / dt      # mm/ms = m/s

    def redraw(self):
        for m in self.marks:
            m.remove()
        self.marks = []
        if self.line is not None:
            self.line.remove(); self.line = None
        for t, r in self.pts:
            self.marks.append(self.ax.plot(t, r, "o", color="lime", ms=7, mec="k", zorder=5)[0])
        c = self.speed()
        if len(self.pts) >= 2:
            (t1, r1), (t2, r2) = self.pts[:2]
            # extend the drawn segment across the full panel height
            rl, rh = self.ax.get_ylim()
            rs = np.array([min(rl, rh), max(rl, rh)])
            ts = t1 + (rs - r1) * (t2 - t1) / (r2 - r1) if abs(r2 - r1) > 1e-9 else np.array([t1, t2])
            self.line = self.ax.plot(ts, rs, "-", color="lime", lw=2, zorder=4)[0]
        auto = f"   |   auto {self.auto[1]:+.2f} m/s (sem {self.auto[0]:.2f})" if self.auto else ""
        msg = "click 2 points on the wavefront" if len(self.pts) < 2 else f"manual {c:+.2f} m/s"
        self.ax.set_xlabel(f"t [ms]      [{msg}{auto}]   r = clear, ENTER = accept")
        self.fig.canvas.draw_idle()

    # -- events --------------------------------------------------------------
    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.button != 1:
            return
        if len(self.pts) >= 2:
            self.pts = []
        self.pts.append((float(ev.xdata), float(ev.ydata)))
        self.redraw()

    def on_key(self, ev):
        if ev.key == "r":
            self.pts = []; self.redraw()
        elif ev.key in ("enter", "return") and len(self.pts) >= 2:
            self.accepted = True
            self.fig.canvas.stop_event_loop()

    def on_close(self, _ev):
        self.skipped = not self.accepted
        try:
            self.fig.canvas.stop_event_loop()
        except Exception:                                          # noqa: BLE001
            pass


class SlopeSlider:
    """Anchor the line with ONE click, then set its slope with a slider.

    Why an alternative to two clicks. The two-click pick puts the whole measurement in the
    difference of two hand-placed points, so a 1 ms slip over a ~5 ms moveout is a 25 % error -
    which is the measured precision of the method. Anchoring on the single point the operator is
    most confident about and then *rotating* the line separates the two judgements: where the
    wavefront is, and how steep it is. The slider also makes the sensitivity visible - if a wide
    range of speeds looks equally good, that is information about the panel, not a failure to
    click accurately.

    The two-click picker is unchanged and remains the default; this is opt-in with
    ``--mode slider``. Picks from both are stored in the same format (two points on the line plus
    the speed), so everything downstream reads them identically.
    """

    def __init__(self, ax, fig, auto=None, cmax=12.0):
        from matplotlib.widgets import Slider

        self.ax, self.fig, self.auto = ax, fig, auto
        self.anchor = None
        self.speed = float(auto[1]) if auto and np.isfinite(auto[1]) else 3.0
        self.speed = float(np.clip(self.speed, -cmax, cmax))
        self.line = self.mark = None
        self.accepted = self.skipped = False

        fig.subplots_adjust(bottom=0.22)
        sax = fig.add_axes([0.13, 0.08, 0.72, 0.035])
        self.slider = Slider(sax, "speed [m/s]", -cmax, cmax, valinit=self.speed, valstep=0.01)
        self.slider.on_changed(self._on_slide)
        self.cids = [fig.canvas.mpl_connect("button_press_event", self.on_click),
                     fig.canvas.mpl_connect("key_press_event", self.on_key),
                     fig.canvas.mpl_connect("close_event", self.on_close)]
        self.redraw()

    # -- geometry ------------------------------------------------------------
    def points(self):
        """Two points on the drawn line, so the stored format matches the two-click picker."""
        if self.anchor is None:
            return []
        t0, r0 = self.anchor
        rl, rh = self.ax.get_ylim()
        rs = np.array([min(rl, rh), max(rl, rh)])
        ts = t0 + (rs - r0) / self.speed if abs(self.speed) > 1e-9 else np.array([t0, t0])
        return [(float(ts[0]), float(rs[0])), (float(ts[1]), float(rs[1]))]

    def _on_slide(self, val):
        self.speed = float(val)
        self.redraw()

    def redraw(self):
        for h in (self.line, self.mark):
            if h is not None:
                h.remove()
        self.line = self.mark = None
        if self.anchor is not None:
            t0, r0 = self.anchor
            self.mark = self.ax.plot(t0, r0, "o", color="lime", ms=8, mec="k", zorder=6)[0]
            pts = self.points()
            self.line = self.ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                                     "-", color="lime", lw=2, zorder=5)[0]
        auto = (f"   |   auto {self.auto[1]:+.2f} m/s (sem {self.auto[0]:.2f})"
                if self.auto else "")
        msg = ("click ONE point on the wavefront" if self.anchor is None
               else f"slope {self.speed:+.2f} m/s")
        self.ax.set_xlabel(f"t [ms]      [{msg}{auto}]   drag the slider, "
                           f"r = clear, ENTER = accept")
        self.fig.canvas.draw_idle()

    # -- events --------------------------------------------------------------
    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.button != 1:
            return
        self.anchor = (float(ev.xdata), float(ev.ydata))
        self.redraw()

    def on_key(self, ev):
        if ev.key == "r":
            self.anchor = None
            self.redraw()
        elif ev.key in ("left", "right"):          # nudge without grabbing the slider
            self.slider.set_val(np.clip(self.speed + (0.05 if ev.key == "right" else -0.05),
                                        self.slider.valmin, self.slider.valmax))
        elif ev.key in ("enter", "return") and self.anchor is not None:
            self.accepted = True
            self.fig.canvas.stop_event_loop()

    def on_close(self, _ev):
        self.skipped = not self.accepted
        try:
            self.fig.canvas.stop_event_loop()
        except Exception:                                          # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--part", default="left", choices=["full", "left", "right"])
    ap.add_argument("--views", default=None,
                    help="comma-separated substrings of view names (default: all)")
    ap.add_argument("--config", default=str(_REPO / "configs" / "passive.yaml"))
    ap.add_argument("--redraw", action="store_true", help="ignore stored picks")
    ap.add_argument("--mode", default="clicks", choices=["clicks", "slider"],
                    help="clicks (default): pick two points on the wavefront. "
                         "slider: pick ONE anchor point and rotate the line with a slider - "
                         "separates 'where is the wavefront' from 'how steep is it', and makes "
                         "the sensitivity of the answer visible.")
    ap.add_argument("--figure", default=None, help="also write a figure of the accepted panels")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("TkAgg", force=True)          # the picking is interactive; the figure re-imports Agg
    import matplotlib.pyplot as plt

    cfg, p = P._paths(a.folder, a.config)
    picks_path = os.path.join(p["outdir"], PICKS_JSON)
    picks = _load_picks(picks_path)

    _, w, ml, views = build_spacetime(a.folder, a.config, a.window, a.part)
    if a.views:
        want = [s.strip().lower() for s in a.views.split(",")]
        views = [v for v in views if any(s in v[0].lower() for s in want)]
    label = w.label or "?"
    print(f"window {a.window} ({label}, {w.t_peak * 1e3:.0f} ms), part {a.part}, "
          f"{ml.r[-1] * 1e3:.0f} mm, {len(views)} view(s)")

    for vname, res, auto in views:
        k = _key(a.window, a.part, vname)
        if k in picks and not a.redraw:
            print(f"  [{vname}] stored: {picks[k]['speed_m_s']:+.2f} m/s")
            continue
        fig, ax = plt.subplots(figsize=(8, 5.5))
        draw_panel(ax, res.st, f"{os.path.basename(a.folder)[:28]}  win{a.window} {label} "
                               f"{w.t_peak * 1e3:.0f} ms  [{vname}]  ({a.part}, "
                               f"{ml.r[-1] * 1e3:.0f} mm)")
        picker = (SlopeSlider(ax, fig, auto=auto) if a.mode == "slider"
                  else SlopePicker(ax, fig, auto=auto))
        fig.tight_layout()
        plt.show(block=False)
        fig.canvas.start_event_loop(timeout=-1)
        if picker.accepted:
            pts = picker.points() if a.mode == "slider" else picker.pts
            spd = picker.speed if a.mode == "slider" else picker.speed()
            picks[k] = dict(window=a.window, part=a.part, view=vname, label=label,
                            method=a.mode,
                            t_peak_ms=w.t_peak * 1e3, points=pts,
                            speed_m_s=spd, auto_speed_m_s=auto[1],
                            auto_semblance=auto[0], mline_length_mm=float(ml.r[-1] * 1e3))
            print(f"  [{vname}] manual {spd:+.2f} m/s "
                  f"(auto {auto[1]:+.2f}, sem {auto[0]:.2f})  [{a.mode}]")
        else:
            print(f"  [{vname}] skipped")
        plt.close(fig)

    with open(picks_path, "w") as f:
        json.dump(picks, f, indent=1)
    print(f"-> {picks_path}")

    if a.figure:
        render_figure(a.folder, a.config, a.window, a.part, views, picks, a.figure)


def render_figure(folder, config, window, part, views, picks, out_path):
    """One row of panels with the manual line drawn and the speed annotated."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    use = [(v, res, auto) for v, res, auto in views if _key(window, part, v) in picks]
    if not use:
        print("no accepted picks - no figure written")
        return
    fig, axs = plt.subplots(1, len(use), figsize=(5.2 * len(use), 4.6), squeeze=False)
    for ax, (vname, res, auto) in zip(axs[0], use):
        pk = picks[_key(window, part, vname)]
        draw_panel(ax, res.st, f"{vname}\nmanual {pk['speed_m_s']:+.2f} m/s   "
                               f"(auto {auto[1]:+.2f}, sem {auto[0]:.2f})")
        (t1, r1), (t2, r2) = pk["points"][:2]
        rl, rh = ax.get_ylim()
        rs = np.array([min(rl, rh), max(rl, rh)])
        ts = t1 + (rs - r1) * (t2 - t1) / (r2 - r1) if abs(r2 - r1) > 1e-9 else np.array([t1, t2])
        ax.plot(ts, rs, "-", color="lime", lw=2)
        ax.plot([t1, t2], [r1, r2], "o", color="lime", ms=6, mec="k")
        ax.set_ylim(rl, rh)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"-> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
