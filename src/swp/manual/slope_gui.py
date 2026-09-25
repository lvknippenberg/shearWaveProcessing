"""Hand slope on five passive space-time views at once, with the line mirrored on all of them.

Top row: displacement (Gaussian), velocity (median), velocity (Gaussian). Bottom row: the two
literature recipes (Keijzer velocity, Petrescu/Santos acceleration) and the buffer-4 B-mode with
the M-line (star = r = 0, the top of every space-time panel). Panels are the montage's M-mode
orientation: x = time, y = distance along the line, so a straight wavefront has slope dr/dt and
the speed is that slope in mm/ms = m/s. Positive = travelling away from the r = 0 end.

Click ONE point on the wavefront in any panel, then set the tilt with the slider or the arrow keys
(left/right 0.05 m/s, up/down 0.5 m/s, f flips the direction). The same line is drawn on every
panel, so you can check that it follows the wave in all of them.

Displacement and velocity weight different frequencies of a dispersive wave and can give different
slopes (by hand, velocity came out faster in 87 % of windows). ``u`` unlinks the displacement
panel: it gets its own line (click on it and tilt), stored separately; ``u`` again re-links it.

Accept by scoring how clearly a wavefront is visible - the score predicts estimator error better
than anything else measured (docs/passive_speed_estimation.md): 3 clear | 2 plausible | 1 guess |
0 none (no line needed for 0). x skips the window undecided, b goes back, q quits.
"""
from __future__ import annotations

import copy

import numpy as np

CONFIDENCE = {"3": "clear", "2": "plausible", "1": "guess", "0": "none"}
CMAX = 12.0
DISP = 0                 # index of the displacement view (can be unlinked)
HELP = ("click: anchor | slider / left-right: +-0.05 | up-down: +-0.5 m/s | f: flip | u: unlink "
        "displacement | r: clear anchor | 3 clear  2 plausible  1 guess  0 none = accept | "
        "x: skip | b: back | q: quit")


def _robust_clim(data, r):
    rc = (r > 0.1 * r[-1]) & (r < 0.9 * r[-1])
    v = np.abs(np.asarray(data)[:, rc] if rc.any() else np.asarray(data))
    c = float(np.nanpercentile(v, 97)) if v.size else 1.0
    return c if c > 0 else 1.0


class SlopeEditor:
    def __init__(self, data, title, preload=None, init_speed=3.0, maximize=True):
        """
        data      dict from the worker's st_win<i>.npz (see swp.manual.worker.load_spacetimes)
        preload   an earlier slopes.json record (redo): its lines are restored
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.data = data
        views = data["views"]
        self.fig, axs = plt.subplots(2, 3, figsize=(19, 10))
        self.axs = list(axs.ravel())
        self.st_axes = self.axs[:len(views)]
        self.fig.suptitle(title, fontsize=10)
        self.fig.text(0.5, 0.008, HELP, ha="center", fontsize=8, color="0.3")
        self.status = self.fig.text(0.5, 0.075, "", ha="center", fontsize=11,
                                    bbox=dict(facecolor="0.95", lw=0))
        for ax, v in zip(self.st_axes, views):
            unit, u = (1e6, "um") if v["quantity"] == "displacement" else (1e3, "mm/s") \
                if v["quantity"] == "velocity" else (1.0, "m/s2")
            c = _robust_clim(v["data"], v["r"]) * unit
            ax.imshow(np.asarray(v["data"]).T * unit,
                      extent=[v["t"][0] * 1e3, v["t"][-1] * 1e3, v["r"][-1] * 1e3, v["r"][0] * 1e3],
                      cmap="RdBu_r", vmin=-c, vmax=c, aspect="auto", origin="upper")
            ax.set_title(f"{v['name']}  (+-{c:.3g} {u})", fontsize=9)
            ax.set_xlabel("t [ms]"); ax.set_ylabel("r along M-line [mm]")
            ax.set_ylim(v["r"][-1] * 1e3, v["r"][0] * 1e3)
            ax.set_xlim(v["t"][0] * 1e3, v["t"][-1] * 1e3)
        self._draw_bmode(self.axs[len(views)] if len(views) < 6 else None)
        for ax in self.axs[len(views) + 1:]:
            ax.axis("off")
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.16, hspace=0.3, wspace=0.18)
        sax = self.fig.add_axes([0.2, 0.035, 0.6, 0.022])
        init = float(np.clip(init_speed if np.isfinite(init_speed) else 3.0, -CMAX, CMAX))
        if abs(init) < 0.3:
            init = 3.0
        self.slider = Slider(sax, "speed [m/s]", -CMAX, CMAX, valinit=init, valstep=0.01)
        self.slider.on_changed(self._on_slide)
        self.lines = {"shared": dict(anchor=None, speed=init, view=None), "disp": None}
        self.active = "shared"
        if preload:
            for g in ("shared", "disp"):
                rec = preload.get(g)
                if rec and rec.get("anchor_t_ms") is not None:
                    self.lines[g] = dict(anchor=(rec["anchor_t_ms"], rec["anchor_r_mm"]),
                                         speed=rec["speed_m_s"], view=rec.get("anchor_view"))
                elif g == "shared" and rec:
                    self.lines[g]["speed"] = rec.get("speed_m_s") or init
            self._set_slider(self.lines["shared"]["speed"])
        self.handles = []
        self.result = None
        self._muted = False
        c = self.fig.canvas
        self.cids = [c.mpl_connect("button_press_event", self.on_click),
                     c.mpl_connect("key_press_event", self.on_key),
                     c.mpl_connect("close_event", self.on_close)]
        if maximize:
            try:
                self.fig.canvas.manager.window.state("zoomed")
            except Exception:                                          # noqa: BLE001
                pass
        self.redraw()

    def _draw_bmode(self, ax):
        if ax is None or self.data.get("bmode_u8") is None:
            return
        ext = self.data["bmode_extent"]
        ax.imshow(self.data["bmode_u8"], cmap="gray", extent=ext, aspect="equal", vmin=0, vmax=255)
        x, z = self.data["line_x_mm"], self.data["line_z_mm"]
        ax.plot(x, z, "-", color="cyan", lw=2)
        ax.plot([x[0]], [z[0]], "*", color="yellow", ms=14, mec="k")
        ax.set_title(self.data.get("bmode_title", "buffer 4 at the event"), fontsize=9)
        ax.set_xlabel("x [mm]")

    # ------------------------------------------------------------ geometry
    def _group_of(self, j):
        return "disp" if (j == DISP and self.lines["disp"] is not None) else "shared"

    def _segment(self, ax, g):
        L = self.lines[g]
        if L is None or L["anchor"] is None:
            return None
        t0, r0 = L["anchor"]
        s = L["speed"] if abs(L["speed"]) >= 0.05 else 0.05 * np.sign(L["speed"] or 1)
        rl, rh = ax.get_ylim()
        rs = np.array([min(rl, rh), max(rl, rh)])
        return t0 + (rs - r0) / s, rs

    def redraw(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        for j, ax in enumerate(self.st_axes):
            g = self._group_of(j)
            seg = self._segment(ax, g)
            if seg is None:
                continue
            act = g == self.active
            xl = ax.get_xlim()
            self.handles += ax.plot(seg[0], seg[1], "-" if act else "--",
                                    color="lime" if act else "yellow", lw=2.2 if act else 1.6, zorder=5)
            t0, r0 = self.lines[g]["anchor"]
            self.handles += ax.plot([t0], [r0], "o", color="lime" if act else "yellow", ms=7,
                                    mec="k", zorder=6)
            ax.set_xlim(xl)
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _status_text(self):
        sh = self.lines["shared"]
        s = ("click a point on the wavefront (any panel)" if sh["anchor"] is None
             else f"line: {sh['speed']:+.2f} m/s")
        if self.lines["disp"] is not None:
            d = self.lines["disp"]
            ds = "click on the displacement panel" if d["anchor"] is None else f"{d['speed']:+.2f} m/s"
            s = (f"velocity/other panels {s}   |   displacement (unlinked): {ds}"
                 f"   [editing: {'displacement' if self.active == 'disp' else 'shared'} line]")
        return s + "   ->  score 3/2/1/0 to accept"

    def _set_slider(self, v):
        self._muted = True
        self.slider.set_val(float(np.clip(v, -CMAX, CMAX)))
        self._muted = False

    # ------------------------------------------------------------ events
    def _on_slide(self, val):
        if self._muted:
            return
        self.lines[self.active]["speed"] = float(val)
        self.redraw()

    def on_click(self, e):
        if e.inaxes not in self.st_axes or e.button != 1 or e.xdata is None:
            return
        tb = getattr(getattr(self.fig.canvas, "manager", None), "toolbar", None)
        if getattr(tb, "mode", ""):
            return
        j = self.st_axes.index(e.inaxes)
        g = self._group_of(j)
        self.active = g
        self.lines[g]["anchor"] = (float(e.xdata), float(e.ydata))
        self.lines[g]["view"] = self.data["views"][j]["name"]
        self._set_slider(self.lines[g]["speed"])
        self.redraw()

    def on_key(self, e):
        k = e.key
        L = self.lines[self.active]
        if k in ("left", "right", "up", "down"):
            step = {"left": -0.05, "right": 0.05, "down": -0.5, "up": 0.5}[k]
            v = float(np.clip(L["speed"] + step, -CMAX, CMAX))
            if abs(v) < 0.05:                    # step across zero instead of stalling on it
                v = 0.05 * np.sign(step)
            L["speed"] = v
            self._set_slider(v)
            self.redraw()
        elif k == "f":
            L["speed"] = -L["speed"]
            self._set_slider(L["speed"])
            self.redraw()
        elif k == "u":
            if self.lines["disp"] is None:
                self.lines["disp"] = copy.deepcopy(self.lines["shared"])
                self.active = "disp"
            else:
                self.lines["disp"] = None
                self.active = "shared"
            self._set_slider(self.lines[self.active]["speed"])
            self.redraw()
        elif k == "r":
            L["anchor"] = None
            self.redraw()
        elif k in CONFIDENCE:
            need = [g for g in ("shared", "disp") if self.lines[g] is not None
                    and self.lines[g]["anchor"] is None]
            if k != "0" and need:
                self.status.set_text("Place the line first (click the wavefront), or 0 = no wave.")
                self.fig.canvas.draw_idle()
                return
            self._finish("accept", int(k))
        elif k == "x":
            self._finish("skip")
        elif k == "b":
            self._finish("back")
        elif k == "q":
            self._finish("quit")

    def on_close(self, _e):
        if self.result is None:
            self.result = dict(action="quit")
        try:
            self.fig.canvas.stop_event_loop()
        except Exception:                                            # noqa: BLE001
            pass

    def _rec(self, g):
        L = self.lines[g]
        if L is None:
            return None
        if L["anchor"] is None:
            return dict(anchor_t_ms=None, anchor_r_mm=None, speed_m_s=None, anchor_view=None)
        return dict(anchor_t_ms=L["anchor"][0], anchor_r_mm=L["anchor"][1],
                    speed_m_s=float(L["speed"]), anchor_view=L["view"])

    def _finish(self, action, confidence=None):
        res = dict(action=action)
        if action == "accept":
            res.update(confidence=confidence, confidence_meaning=CONFIDENCE[str(confidence)],
                       shared=self._rec("shared"), disp=self._rec("disp"))
        self.result = res
        self.fig.canvas.stop_event_loop()

    def run(self, snapshot=None):
        import matplotlib.pyplot as plt
        plt.show(block=False)
        self.fig.canvas.start_event_loop(timeout=0)
        if self.result is None:
            self.result = dict(action="quit")
        if snapshot and self.result["action"] == "accept":
            try:
                self.status.set_text(self._status_text().replace("->  score 3/2/1/0 to accept",
                                                                 f"ACCEPTED, confidence "
                                                                 f"{self.result['confidence']}"))
                self.fig.savefig(snapshot, dpi=70)
            except Exception:                                          # noqa: BLE001
                pass
        plt.close(self.fig)
        return self.result
