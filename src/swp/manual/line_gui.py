"""Three-buffer M-line editor: buffers 1 | 3 | 4 side by side at the same cardiac phase.

Draw the line on whichever buffer shows the septum best - buffer 4 when it is readable (it is the
data the passive analysis runs on), otherwise buffer 1, otherwise buffer 3. The line is mirrored on
the other two panels:

* while drawing: the same coordinates, dashed white.

**Drawn on buffer 4** (the usual case since 2026-10-01): the saved line is exactly the one drawn, so
ONE ENTER accepts it - no registration and no review step (the buffer-1 / 3 panels only help to
read the anatomy).

**Drawn on buffer 1 or 3:** buffers 1 and 3 are recorded in other heartbeats than buffer 4, so the
first ENTER moves the line onto buffer 4 by the anatomy registration
(:func:`swp.mline.transfer.transfer_line`): solid orange on the other panels, the buffer-4 line -
the one that is saved and processed - in green, and the status line says whether the registration
is trusted (ensemble agreement + known-shift check). If it is not, nudge the green line with the
arrow keys, or press ``i`` to keep the uncorrected coordinates; the second ENTER accepts.

Mouse: left-click adds a point (any order), drag moves one, right-click deletes one.
Keys: ENTER accept (buffer 4) or review / accept (buffer 1 / 3) | c clear | backspace remove last point | arrows nudge the green
buffer-4 line 0.25 mm (shift: 1 mm) | i toggle motion correction | z zoom all panels on the line /
full sector | [ ] step the frame of the panel under the mouse | a toggle buffer-4 averaging |
x skip (no usable septum) | b back | q quit.
"""
from __future__ import annotations

import numpy as np

from . import frames as F
from ._light import order_points, transfer

HELP = ("click: add point | drag: move | right-click: delete | ENTER: accept (buffer 1/3: review first) | "
        "c: clear | arrows: nudge green line | i: motion correction on/off | z: zoom on line / full | "
        "[ ]: frame of panel under mouse | a: buffer-4 averaging | x: skip (no septum) | b: back | q: quit")
ZOOM_MM = 25.0


def _spline_mm(points_mm, n=250):
    from ..viz.mline import mline_from_points
    ml = mline_from_points(np.asarray(points_mm) * 1e-3, n)
    return ml.x * 1e3, ml.z * 1e3


class LineEditor:
    """One prompt. ``run()`` -> dict(action=accept|skip|back|quit, ...)."""

    def __init__(self, panels, title, reload_panel, preload=None, reference=None, maximize=True):
        """
        panels        {buffer: frames.Panel or None}
        reload_panel  callable(buffer, frame, n_avg) -> Panel (frame stepping / averaging)
        preload       dict(source=buffer, points_mm=(k, 2), what=str) - editable starting line
        reference     (label, points_mm on buffer 4) drawn dashed magenta on buffer 4 only
        """
        import matplotlib.pyplot as plt

        self.panels = panels
        self.reload_panel = reload_panel
        self.bufs = [b for b in F.BUFFERS if panels.get(b) is not None]
        self.fig, axs = plt.subplots(1, len(self.bufs), figsize=(19, 8.2))
        axs = np.atleast_1d(axs)
        self.ax = dict(zip(self.bufs, axs))
        self.fig.suptitle(title, fontsize=10)
        self.fig.text(0.5, 0.012, HELP, ha="center", fontsize=8, color="0.3")
        self.status = self.fig.text(0.5, 0.045, "", ha="center", fontsize=10,
                                    bbox=dict(facecolor="0.95", lw=0))
        self.im, self.art = {}, {}
        for b in self.bufs:
            self._draw_panel(b)
        self.reference = reference
        if reference is not None and 4 in self.ax:
            lbl, pts = reference
            xs, zs = _spline_mm(pts)
            self.ax[4].plot(xs, zs, "--", color="magenta", lw=1.2, alpha=0.8, zorder=3, label=lbl)
            self.ax[4].legend(loc="lower right", fontsize=8)
        self.source = None
        self.points = []                     # (x, z) mm on the source panel, click order
        self.mapped = {}                     # buffer -> (k, 2) mm, None when stale
        self.map_info = {}
        self.nudge = np.zeros(2)
        self.identity = False
        self.phase = "edit"
        self.drag = None
        self.result = None
        self.preload_what = None
        if preload is not None and preload.get("source") in self.ax:
            self.source = preload["source"]
            self.points = [tuple(p) for p in np.asarray(preload["points_mm"], float)]
            self.preload_what = preload.get("what", "pre-loaded line")
        self.zoomed = len(self.points) >= 2          # start zoomed on a pre-loaded line
        self._apply_zoom()
        c = self.fig.canvas
        self.cids = [c.mpl_connect("button_press_event", self.on_press),
                     c.mpl_connect("motion_notify_event", self.on_motion),
                     c.mpl_connect("button_release_event", self.on_release),
                     c.mpl_connect("key_press_event", self.on_key),
                     c.mpl_connect("close_event", self.on_close)]
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.12, wspace=0.12)
        if maximize:
            try:
                self.fig.canvas.manager.window.state("zoomed")        # TkAgg on Windows
            except Exception:                                          # noqa: BLE001
                pass
        self.redraw()

    # ------------------------------------------------------------ drawing
    def _draw_panel(self, b):
        pn = self.panels[b]
        ax = self.ax[b]
        if b in self.im:
            self.im[b].set_data(pn.u8())
            self.im[b].set_extent(pn.extent)
        else:
            self.im[b] = ax.imshow(pn.u8(), cmap="gray", extent=pn.extent, aspect="equal",
                                   vmin=0, vmax=255, interpolation="bilinear")
            ax.set_xlabel("x [mm]")
            if b == self.bufs[0]:
                ax.set_ylabel("z [mm]")
            a = dict(line=ax.plot([], [], "-", color="cyan", lw=2, zorder=5)[0],
                     pts=ax.plot([], [], "o", color="yellow", ms=7, mec="k", zorder=6)[0],
                     mirror=ax.plot([], [], "--", color="white", lw=1.4, zorder=4)[0],
                     raw=ax.plot([], [], "--", color="white", lw=1.2, alpha=0.8, zorder=3)[0],
                     start=ax.plot([], [], "*", color="yellow", ms=13, mec="k", zorder=7)[0])
            self.art[b] = a
        self._title(b)

    def _title(self, b):
        pn = self.panels[b]
        avg = f", mean of {pn.n_avg}" if pn.n_avg > 1 else ""
        t = f"buffer {b}: frame {pn.frame}/{pn.n_frames}{avg}\n{pn.note}"
        mi = self.map_info.get(b) if getattr(self, "phase", "edit") == "review" else None
        if mi:
            t += ("\nregistration FAILED" if "error" in mi else
                  f"\nregistration {'trusted' if mi['reliable'] else 'NOT TRUSTED'}: moved "
                  f"{mi['shift_mm']:.1f} mm, agreement {mi['agree']:.2f}")
        self.ax[b].set_title(t, fontsize=9, color="k" if not mi or mi.get("reliable") else "red")

    def _apply_zoom(self):
        """z: all panels on the same box around the line (+-ZOOM_MM), or the full sector."""
        if self.zoomed and len(self.points) >= 2:
            pts = np.asarray(self.points, float)
            lo, hi = pts.min(axis=0) - ZOOM_MM, pts.max(axis=0) + ZOOM_MM
            for ax in self.ax.values():
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(hi[1], lo[1])
        else:
            for b, ax in self.ax.items():
                e = self.panels[b].extent
                ax.set_xlim(e[0], e[1])
                ax.set_ylim(e[2], e[3])

    def _source_ordered(self):
        return order_points(np.asarray(self.points, float), anchor=self.points[0])

    def final_points(self):
        """The line on buffer 4 (mm) that will be saved, or None."""
        if self.source is None or len(self.points) < 2:
            return None
        pts = self._source_ordered()
        if self.source == 4 or self.identity or self.mapped.get(4) is None:
            return pts + (self.nudge if self.source != 4 else 0)
        return self.mapped[4] + self.nudge

    def redraw(self):
        have = self.source is not None and len(self.points) >= 2
        pts = self._source_ordered() if have else None
        for b in self.bufs:
            a = self.art[b]
            for h in a.values():
                h.set_data([], [])
            a["mirror"].set_linestyle("--"); a["mirror"].set_color("white"); a["mirror"].set_linewidth(1.4)
            if self.source is None:
                continue
            if b == self.source:
                a["pts"].set_data([p[0] for p in self.points], [p[1] for p in self.points])
                if have:
                    a["line"].set_data(*_spline_mm(pts))
                    a["start"].set_data([pts[0, 0]], [pts[0, 1]])
            elif have:
                shown = None
                if self.phase == "review":
                    # uncorrected coordinates, for comparison with the registered line
                    a["raw"].set_data(*_spline_mm(pts))
                if self.phase == "review" and b == 4:
                    shown = self.final_points()
                    a["mirror"].set_color("lime"); a["mirror"].set_linestyle("-"); a["mirror"].set_linewidth(2.2)
                elif self.phase == "review" and self.mapped.get(b) is not None:
                    shown = self.mapped[b]
                    ok = (self.map_info.get(b) or {}).get("reliable")
                    a["mirror"].set_color("orange" if ok else "red")
                    a["mirror"].set_linestyle("-" if ok else ":")
                    a["mirror"].set_linewidth(1.8 if ok else 2.2)
                else:
                    shown = pts
                    a["raw"].set_data([], [])
                a["mirror"].set_data(*_spline_mm(shown))
                a["start"].set_data([shown[0, 0]], [shown[0, 1]])
        for b in self.bufs:
            self._title(b)
        if self.phase == "review" and self.source == 4 and have:
            a = self.art[4]
            a["line"].set_color("lime")
        elif 4 in self.art:
            self.art[4]["line"].set_color("cyan")
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _status_text(self):
        if self.source is None:
            return "Draw the M-line on any panel (buffer 4 preferred when the septum is visible)."
        n = len(self.points)
        head = f"drawing on buffer {self.source} ({n} point{'s' if n != 1 else ''})"
        if self.preload_what and self.phase == "edit":
            head += f" - pre-loaded: {self.preload_what}"
        if self.phase == "edit":
            return head + (" - ENTER accepts (saved as drawn)" if self.source == 4 else
                           " - ENTER to map onto buffer 4 and review")
        if self.source == 4:
            return (head + " - REVIEW: the saved line is the one you drew. Buffers 1/3 show it "
                    "registered (orange = trusted, red dotted = not trusted) and uncorrected (dashed "
                    "white). ENTER accepts.")
        mi = self.map_info.get(4, {})
        if self.identity:
            corr = "motion correction OFF (i): buffer-4 line = drawn coordinates"
        elif mi:
            rel = "TRUSTED" if mi["reliable"] else "NOT TRUSTED - check the green line"
            corr = (f"moved {mi['shift_mm']:.1f} mm (dx {mi['dx']:+.1f}, dz {mi['dz']:+.1f}); "
                    f"agreement {mi['agree']:.2f}, check {mi['known_err_mm']:.1f} mm -> {rel}")
        else:
            corr = "registration failed - uncorrected coordinates"
        nud = f"; nudged {self.nudge[0]:+.2f}/{self.nudge[1]:+.2f} mm" if self.nudge.any() else ""
        return f"{head} - REVIEW buffer 4 (green): {corr}{nud}. ENTER accepts."

    # ------------------------------------------------------------ mapping
    def compute_mapping(self):
        tl = transfer()
        src = self.panels[self.source]
        pts = self._source_ordered()
        self.mapped, self.map_info = {}, {}
        self.status.set_text("registering ...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        for b in self.bufs:
            if b == self.source:
                continue
            dst = self.panels[b]
            try:
                # the known-shift check for every panel: a mirrored line must carry a verdict too
                r = tl.transfer_line(pts, (src.env, src.x_mm, src.z_mm), (dst.env, dst.x_mm, dst.z_mm),
                                     check=True)
                self.mapped[b] = r.points
                self.map_info[b] = dict(shift_mm=r.shift_mm, dx=r.transform.dx, dz=r.transform.dz,
                                        agree=r.agree, known_err_mm=r.known_err_mm,
                                        corr=r.transform.corr, corr0=r.transform.corr0,
                                        reliable=r.reliable())
            except Exception as exc:                                   # noqa: BLE001
                self.mapped[b] = None
                self.map_info[b] = dict(error=str(exc))

    def _stale(self):
        self.phase, self.mapped, self.map_info = "edit", {}, {}
        self.nudge = np.zeros(2)

    # ------------------------------------------------------------ events
    def _buf_of(self, ax):
        for b, a in self.ax.items():
            if a is ax:
                return b
        return None

    def _toolbar_active(self):
        tb = getattr(getattr(self.fig.canvas, "manager", None), "toolbar", None)
        return bool(getattr(tb, "mode", ""))

    def on_press(self, e):
        b = self._buf_of(e.inaxes)
        if b is None or e.xdata is None or self._toolbar_active():
            return
        if self.source is not None and self.points and b != self.source:
            self.status.set_text(f"The line is drawn on buffer {self.source}. Press c to clear it "
                                 f"and draw on buffer {b} instead.")
            self.fig.canvas.draw_idle()
            return
        self.source = b
        pr = 0.03 * abs(np.diff(e.inaxes.get_xlim())[0])
        d = [np.hypot(px - e.xdata, pz - e.ydata) for px, pz in self.points]
        near = int(np.argmin(d)) if d else None
        if e.button == 1:
            if near is not None and d[near] <= pr:
                self.drag = near
            else:
                self.points.append((e.xdata, e.ydata))
                self.drag = len(self.points) - 1
        elif e.button == 3 and near is not None and d[near] <= pr:
            self.points.pop(near)
            if not self.points:
                self.source = None
        self.preload_what = None
        self._stale()
        self.redraw()

    def on_motion(self, e):
        if self.drag is None or self._buf_of(e.inaxes) != self.source or e.xdata is None:
            return
        self.points[self.drag] = (e.xdata, e.ydata)
        self.redraw()

    def on_release(self, _e):
        self.drag = None

    def on_key(self, e):
        k = e.key
        if k in ("enter", "return"):
            if self.source is None or len(self.points) < 2:
                self.status.set_text("Need at least 2 points (or x to skip this line).")
                self.fig.canvas.draw_idle()
                return
            if self.source == 4:                       # saved as drawn: nothing to register
                self._finish("accept")
            elif self.phase == "edit":
                self.compute_mapping()
                self.phase = "review"
                self.redraw()
            else:
                self._finish("accept")
        elif k == "c":
            self.source, self.points = None, []
            self.identity = False
            self._stale()
            self.redraw()
        elif k == "backspace" and self.points:
            self.points.pop()
            if not self.points:
                self.source = None
            self._stale()
            self.redraw()
        elif k in ("left", "right", "up", "down", "shift+left", "shift+right", "shift+up", "shift+down"):
            if self.phase != "review" or self.source == 4:
                return
            step = 1.0 if k.startswith("shift+") else 0.25
            d = {"left": (-step, 0), "right": (step, 0), "up": (0, -step), "down": (0, step)}[k.split("+")[-1]]
            self.nudge = self.nudge + np.array(d)
            self.redraw()
        elif k == "i" and self.phase == "review" and self.source != 4:
            self.identity = not self.identity
            self.redraw()
        elif k in ("[", "]"):
            b = self._buf_of(e.inaxes)
            if b is None:
                return
            pn = self.panels[b]
            step = (5 if b == 4 else 1) * (1 if k == "]" else -1)
            self._reload(b, pn.frame + step, pn.n_avg)
        elif k == "z":
            self.zoomed = not self.zoomed
            self._apply_zoom()
            self.fig.canvas.draw_idle()
        elif k == "a" and 4 in self.panels:
            pn = self.panels[4]
            self._reload(4, pn.frame, 1 if pn.n_avg > 1 else F.AVG4)
        elif k == "x":
            self._finish("skip")
        elif k == "b":
            self._finish("back")
        elif k == "q":
            self._finish("quit")

    def _reload(self, b, frame, n_avg):
        self.status.set_text(f"loading buffer {b} ...")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.panels[b] = self.reload_panel(b, frame, n_avg)
        self._draw_panel(b)
        self._apply_zoom()
        if self.phase == "review":
            self._stale()
        self.redraw()

    def on_close(self, _e):
        if self.result is None:
            self.result = dict(action="quit")
        try:
            self.fig.canvas.stop_event_loop()
        except Exception:                                            # noqa: BLE001
            pass

    def _finish(self, action):
        res = dict(action=action)
        if action == "accept":
            pts4 = self.final_points()
            res.update(source_buffer=self.source,
                       points_src_mm=self._source_ordered().tolist(),
                       points4_mm=np.asarray(pts4).tolist(),
                       motion_correction=(self.source != 4 and not self.identity
                                          and self.mapped.get(4) is not None),
                       mapping=self.map_info.get(4), mapping_other={b: v for b, v in self.map_info.items() if b != 4},
                       nudge_mm=self.nudge.tolist(),
                       frames={b: dict(frame=p.frame, n_avg=p.n_avg, phase_ms=p.phase_ms, note=p.note)
                               for b, p in self.panels.items() if p is not None})
        self.result = res
        self.fig.canvas.stop_event_loop()

    def run(self, snapshot=None):
        """Block until accept / skip / back / quit. ``snapshot``: path to save the figure on accept."""
        import matplotlib.pyplot as plt
        plt.show(block=False)
        self.fig.canvas.start_event_loop(timeout=0)
        if self.result is None:
            self.result = dict(action="quit")
        if snapshot and self.result["action"] == "accept":
            try:
                self.status.set_text(self._status_text().replace(" ENTER accepts.", " ACCEPTED."))
                self.fig.savefig(snapshot, dpi=70)
            except Exception:                                            # noqa: BLE001
                pass
        plt.close(self.fig)
        return self.result
