"""Mark regions of interest (time intervals) on the whole-recording space-time of the general line.

The detector's 100 ms windows (energy pick, semblance screen) turned out to be a major source of
error, so here the windows are chosen by eye instead. Shown: the R-peaks, the expected valve-closure
search windows (MVC R+0-150 ms, AVC QS2 +-120 ms; as ``swp.mline.select.phase_targets``) and the
"velocity gauss" space-time along the general M-line over the whole buffer-4 recording, on one
colour scale. x = time on the buffer-4 clock, y = distance along the line.

Drag left-right to draw a horizontal line: its time span is one ROI. Draw as many as needed.

=====================  =====================================================================
mouse / key            action
=====================  =====================================================================
drag                   draw a new ROI (at least 5 ms)
click                  with --click-ms: a window of that length centred on the click
=====================  =====================================================================

Fixed-length mode (``fixed_ms``, the window review of the manual study): every window has that
length - a drag or a click places one centred on it, dragging an end or the middle moves it.

=====================  =====================================================================
drag an end / middle   move that end / the whole ROI
right-click            delete the ROI under the mouse
scroll                 zoom time around the mouse; ``0`` resets
up / down              contrast (the colour-scale percentile)
l                      cycle the label of the ROI under the mouse (MVC, AVC, AK, other)
c                      clear all ROIs
ENTER                  accept (needs at least one ROI)
n                      accept "nothing worth investigating" (no ROI)
x / b / q              skip undecided / back / quit
=====================  =====================================================================

zea-free (only numpy + matplotlib), so the editor opens in seconds.
"""
from __future__ import annotations

import numpy as np

from ..passive_valves import expected_windows  # noqa: F401 (re-exported)

LABELS = ("MVC", "AVC", "AK", "other")
COL = {"MVC": "#2a78d6", "AVC": "#eb6834", "AK": "#1f9e6e", "other": "#8a8983"}
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
MIN_ROI_MS = 5.0
EDGE_PX = 7
CLIM_PCTS = (90.0, 95.0, 98.0, 99.0, 99.5, 99.8, 99.95)
CLIM_DEFAULT = 4                              # 99.5: the overview figures' scale
HELP = ("drag: draw ROI | drag end/middle: move | right-click: delete | scroll: zoom time (0: reset) | "
        "up/down: contrast | l: label | c: clear | ENTER: accept | n: none | x: skip | b: back | q: quit")


def describe(t0, t1, r_peaks_s, expected):
    """Phase and expected-window overlap of a ROI [t0, t1] (s): the suggested label is the expected
    window covering most of it, else "other"."""
    best, frac = None, 0.0
    for name, lo, hi in expected:
        f = max(0.0, min(t1, hi) - max(t0, lo)) / max(t1 - t0, 1e-9)
        if f > frac:
            best, frac = name, f
    mid = 0.5 * (t0 + t1)
    rp = np.asarray(r_peaks_s, float)
    before = rp[rp <= mid]
    phase = float((mid - before[-1]) * 1e3) if before.size else None
    return dict(expected=best, expected_overlap=round(frac, 3), phase_ms=phase,
                suggested=best if best and frac > 0 else "other")


class RoiEditor:
    def __init__(self, data, title, preload=None, maximize=True, click_ms=None, fixed_ms=None,
                 hints=None, roi_name="your ROIs"):
        """
        data      dict: v (n_t, n_r) velocity, t_s, r_m, r_peaks_s, rr_s (see scripts/passive_roi.py)
        preload   an earlier rois.json record: its ROIs (and contrast) are restored
        click_ms  semi-automatic: a plain click (no drag) places a window of this length centred
                  on the click (shifted inside the recording); None = clicks do nothing
        fixed_ms  every window has this length (implies click_ms = fixed_ms)
        hints     [dict(t0, t1 [s], text)]: drawn dashed in their own lane, not editable (e.g. the
                  detected windows below the screen)
        """
        import matplotlib.pyplot as plt

        self.fixed_ms = fixed_ms
        self.click_ms = fixed_ms or click_ms
        self.hints = list(hints or [])
        self.roi_name = roi_name
        self.d = data
        self.t_ms = np.asarray(data["t_s"], float) * 1e3
        self.r_mm = np.asarray(data["r_m"], float) * 1e3
        self.rp = np.asarray(data["r_peaks_s"], float)
        rr = data.get("rr_s")
        self.expected = expected_windows(self.rp, rr) if self.rp.size else []
        self.v = np.asarray(data["v"], float)
        self.abs_v = np.abs(self.v[np.isfinite(self.v)])
        self.rois = []                     # dict(t0, t1 [ms], r [mm], label)
        self.clim_i = CLIM_DEFAULT
        if preload:
            for q in preload.get("rois", []):
                self.rois.append(dict(t0=q["t0"] * 1e3, t1=q["t1"] * 1e3,
                                      r=q.get("r_mm", float(self.r_mm[-1]) / 2), label=q["label"]))
            if preload.get("clim_pct") in CLIM_PCTS:
                self.clim_i = CLIM_PCTS.index(preload["clim_pct"])

        self.fig = plt.figure(figsize=(17, 9), facecolor=SURF)
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.06)
        self.strip = self.fig.add_subplot(gs[0])
        self.ax = self.fig.add_subplot(gs[1], sharex=self.strip)
        self.fig.suptitle(title, fontsize=10, x=0.01, ha="left")
        help_ = HELP if not self.fixed_ms else (
            f"click / drag: place a {self.fixed_ms:.0f} ms window | drag a window: move | right-click: delete | "
            "scroll: zoom time (0: reset) | up/down: contrast | l: label | c: clear | ENTER: accept | "
            "n: no window | x: skip | b: back | q: quit")
        self.fig.text(0.5, 0.012, help_, ha="center", fontsize=8, color=INK2)
        self.status = self.fig.text(0.5, 0.04, "", ha="center", fontsize=11,
                                    bbox=dict(facecolor="0.94", lw=0))
        ext = [self.t_ms[0], self.t_ms[-1], self.r_mm[-1], self.r_mm[0]]
        self.img = self.ax.imshow(self.v.T, aspect="auto", cmap="RdBu_r", extent=ext,
                                  interpolation="nearest")
        self.ax.set_ylabel("along the general line [mm]")
        self.ax.set_xlabel("time on the buffer-4 clock [ms]")
        self.full_xlim = (self.t_ms[0], self.t_ms[-1])
        self._draw_strip()
        for a in (self.strip, self.ax):
            for r in self.rp[(self.rp * 1e3 >= self.full_xlim[0]) & (self.rp * 1e3 <= self.full_xlim[1])]:
                a.axvline(r * 1e3, color=INK2, lw=0.9, ls=":")
            a.set_facecolor(SURF)
        self.ax.set_xlim(*self.full_xlim)
        self.fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.135)

        self.handles = []
        self.drag = None                   # dict(kind=new|t0|t1|move, i, x0, ...)
        self.sel = None
        self.result = None
        c = self.fig.canvas
        self.cids = [c.mpl_connect("button_press_event", self.on_press),
                     c.mpl_connect("motion_notify_event", self.on_motion),
                     c.mpl_connect("button_release_event", self.on_release),
                     c.mpl_connect("scroll_event", self.on_scroll),
                     c.mpl_connect("key_press_event", self.on_key),
                     c.mpl_connect("close_event", self.on_close)]
        if maximize:
            try:
                self.fig.canvas.manager.window.state("zoomed")
            except Exception:                                          # noqa: BLE001
                pass
        self._apply_clim()
        self.redraw()

    # ------------------------------------------------------------ drawing
    def _draw_strip(self):
        from matplotlib.patches import Rectangle
        s = self.strip
        for name, lo, hi in self.expected:
            s.add_patch(Rectangle((lo * 1e3, 1.65), (hi - lo) * 1e3, 0.7, facecolor="none",
                                  edgecolor=COL[name], lw=1.6))
            s.text((lo + hi) / 2 * 1e3, 2.0, name, ha="center", va="center", fontsize=8,
                   color=COL[name], fontweight="bold", clip_on=True)
        for h in self.hints:
            s.add_patch(Rectangle((h["t0"] * 1e3, 0.95), (h["t1"] - h["t0"]) * 1e3, 0.5, facecolor="none",
                                  edgecolor=INK2, lw=1.2, ls="--"))
            s.text((h["t0"] + h["t1"]) / 2 * 1e3, 1.2, h.get("text", ""), ha="center", va="center",
                   fontsize=7, color=INK2, clip_on=True)
        s.set_ylim(-0.5, 2.6)
        ticks = [(2, "expected window"), (0.5, self.roi_name)] + ([(1.2, "screened")] if self.hints else [])
        s.set_yticks([k for k, _ in ticks], [n for _, n in ticks], fontsize=8)
        s.tick_params(axis="x", labelbottom=False)
        if not self.expected:
            s.text(0.5, 2.0, "no usable R-peaks: no expected windows", transform=s.get_yaxis_transform(),
                   ha="center", va="center", fontsize=9, color=INK2)
        for sp in ("top", "right"):
            s.spines[sp].set_visible(False)

    def _apply_clim(self):
        pct = CLIM_PCTS[self.clim_i]
        lim = float(np.percentile(self.abs_v, pct)) if self.abs_v.size else 1.0
        lim = lim if lim > 0 else 1.0
        self.img.set_clim(-lim, lim)
        self.clim_text = f"colour scale ±{lim * 1e3:.3g} mm/s ({pct:g}th percentile)"

    def redraw(self):
        from matplotlib.patches import Rectangle
        import matplotlib.patheffects as pe
        for h in self.handles:
            h.remove()
        self.handles = []
        halo = [pe.withStroke(linewidth=4.5, foreground="white")]
        for i, q in enumerate(self.rois):
            sel = i == self.sel
            lw = 3.0 if sel else 2.0
            col = COL.get(q["label"], COL["other"])
            for t in (q["t0"], q["t1"]):
                self.handles.append(self.ax.axvline(t, color=INK, lw=1.0, ls="--", alpha=0.8))
            self.handles += self.ax.plot([q["t0"], q["t1"]], [q["r"], q["r"]], "-|", color=INK, lw=lw,
                                         ms=14, mew=lw, path_effects=halo, zorder=5)
            self.handles.append(self.ax.text(q["t0"], self.r_mm[0], f" {i + 1} {q['label']}", va="top",
                                             ha="left", fontsize=9, fontweight="bold", color="white",
                                             zorder=6, bbox=dict(facecolor=col, lw=2 if sel else 0,
                                                                 edgecolor=INK, pad=1.5)))
            self.handles.append(self.strip.add_patch(Rectangle(
                (q["t0"], 0.15), q["t1"] - q["t0"], 0.7, facecolor=col, edgecolor=INK, lw=2 if sel else 0)))
            self.handles.append(self.strip.text((q["t0"] + q["t1"]) / 2, 0.5, f"{i + 1}", ha="center",
                                                va="center", fontsize=8, color="white", fontweight="bold",
                                                clip_on=True))
        if self.drag and self.drag["kind"] == "new" and self.drag.get("x1") is not None:
            a, b = sorted((self.drag["x0"], self.drag["x1"]))
            self.handles += self.ax.plot([a, b], [self.drag["y"]] * 2, "-|", color="#e5d200", lw=2.5,
                                         ms=14, mew=2.5, path_effects=[pe.withStroke(linewidth=4.5,
                                                                                    foreground=INK)])
            self.handles.append(self.ax.axvspan(a, b, color="#e5d200", alpha=0.12, lw=0))
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _status_text(self):
        if not self.rois:
            s = "drag left-right on the space-time to mark a time window"
        else:
            s = "  |  ".join(f"{i + 1} {q['label']} {q['t0']:.0f}-{q['t1']:.0f} ms ({q['t1'] - q['t0']:.0f})"
                             for i, q in enumerate(self.rois))
        return f"{s}      [{self.clim_text}]"

    # ------------------------------------------------------------ hit tests
    def _toolbar_busy(self):
        tb = getattr(getattr(self.fig.canvas, "manager", None), "toolbar", None)
        return bool(getattr(tb, "mode", ""))

    def _px_per_ms(self, ax):
        x0, x1 = ax.get_xlim()
        return ax.bbox.width / max(abs(x1 - x0), 1e-9)

    def _hit(self, e):
        """(index, part) of the ROI under the mouse: part = t0 | t1 | move; or (None, None)."""
        if e.xdata is None:
            return None, None
        tol = EDGE_PX / self._px_per_ms(e.inaxes)
        best = (None, None, np.inf)
        for i, q in enumerate(self.rois):
            for part in ("t0", "t1"):
                dist = abs(e.xdata - q[part])
                if dist <= tol and dist < best[2]:
                    best = (i, part, dist)
        if best[0] is not None:
            return best[0], best[1]
        inside = [i for i, q in enumerate(self.rois) if q["t0"] <= e.xdata <= q["t1"]]
        if inside:                         # the narrowest ROI wins when they overlap
            i = min(inside, key=lambda k: self.rois[k]["t1"] - self.rois[k]["t0"])
            return i, "move"
        return None, None

    # ------------------------------------------------------------ events
    def on_press(self, e):
        if e.inaxes not in (self.ax, self.strip) or e.xdata is None or self._toolbar_busy():
            return
        i, part = self._hit(e)
        if e.button == 3:
            if i is not None:
                del self.rois[i]
                self.sel = None
                self.redraw()
            return
        if e.button != 1:
            return
        y = e.ydata if e.inaxes is self.ax else float(self.r_mm[-1]) / 2
        if i is None:
            self.drag = dict(kind="new", x0=float(e.xdata), x1=None, y=float(y))
            self.sel = None
        else:
            q = self.rois[i]
            self.drag = dict(kind="move" if self.fixed_ms else part, i=i, x0=float(e.xdata),
                             orig=(q["t0"], q["t1"], q["r"]),
                             y0=float(y), on_st=e.inaxes is self.ax)
            self.sel = i
        self.redraw()

    def on_motion(self, e):
        if self.drag is None or e.xdata is None or e.inaxes not in (self.ax, self.strip):
            return
        x = float(np.clip(e.xdata, *self.full_xlim))
        d = self.drag
        if d["kind"] == "new":
            d["x1"] = x
        else:
            q = self.rois[d["i"]]
            t0, t1, r = d["orig"]
            if d["kind"] == "t0":
                q["t0"] = min(x, t1 - MIN_ROI_MS)
            elif d["kind"] == "t1":
                q["t1"] = max(x, t0 + MIN_ROI_MS)
            else:
                dx = float(np.clip(x - d["x0"], self.full_xlim[0] - t0, self.full_xlim[1] - t1))
                q["t0"], q["t1"] = t0 + dx, t1 + dx
                if d["on_st"] and e.inaxes is self.ax:
                    q["r"] = float(np.clip(r + e.ydata - d["y0"], self.r_mm[0], self.r_mm[-1]))
        self.redraw()

    def on_release(self, e):
        d, self.drag = self.drag, None
        if d is None:
            return
        clicked = d["kind"] == "new" and (d["x1"] is None or abs(d["x1"] - d["x0"]) < MIN_ROI_MS)
        if d["kind"] == "new" and (self.fixed_ms or (clicked and self.click_ms)):
            centre = d["x0"] if clicked else 0.5 * (d["x0"] + d["x1"])
            t0 = float(np.clip(centre - self.click_ms / 2, self.full_xlim[0],
                               self.full_xlim[1] - self.click_ms))
            d = dict(d, x0=t0, x1=t0 + self.click_ms)
        if d["kind"] == "new" and d["x1"] is not None and abs(d["x1"] - d["x0"]) >= MIN_ROI_MS:
            t0, t1 = sorted((d["x0"], d["x1"]))
            self.rois.append(dict(t0=t0, t1=t1, r=d["y"], label=self._suggest(t0, t1)))
            self.rois.sort(key=lambda q: q["t0"])
            self.sel = next(k for k, q in enumerate(self.rois) if q["t0"] == t0 and q["t1"] == t1)
        elif d["kind"] in ("t0", "t1", "move"):
            q = self.rois[d["i"]]
            if q["label"] == self._suggest(*d["orig"][:2]):          # keep a hand-set label
                q["label"] = self._suggest(q["t0"], q["t1"])
            self.rois.sort(key=lambda r: r["t0"])
            self.sel = self.rois.index(q)
        self.redraw()

    def _suggest(self, t0_ms, t1_ms):
        return describe(t0_ms * 1e-3, t1_ms * 1e-3, self.rp, self.expected)["suggested"]

    def on_scroll(self, e):
        if e.inaxes not in (self.ax, self.strip) or e.xdata is None:
            return
        x0, x1 = self.ax.get_xlim()
        f = 0.8 if e.button == "up" else 1.25
        lo, hi = e.xdata - (e.xdata - x0) * f, e.xdata + (x1 - e.xdata) * f
        span = self.full_xlim[1] - self.full_xlim[0]
        if hi - lo >= span:
            lo, hi = self.full_xlim
        elif lo < self.full_xlim[0]:
            lo, hi = self.full_xlim[0], self.full_xlim[0] + (hi - lo)
        elif hi > self.full_xlim[1]:
            lo, hi = self.full_xlim[1] - (hi - lo), self.full_xlim[1]
        self.ax.set_xlim(lo, hi)
        self.fig.canvas.draw_idle()

    def on_key(self, e):
        k = e.key
        if k in ("up", "down"):
            self.clim_i = int(np.clip(self.clim_i + (1 if k == "up" else -1), 0, len(CLIM_PCTS) - 1))
            self._apply_clim()
            self.redraw()
        elif k == "0":
            self.ax.set_xlim(*self.full_xlim)
            self.fig.canvas.draw_idle()
        elif k == "l":
            i, _ = self._hit(e) if e.inaxes in (self.ax, self.strip) else (None, None)
            i = self.sel if i is None else i
            if i is not None:
                q = self.rois[i]
                q["label"] = LABELS[(LABELS.index(q["label"]) + 1) % len(LABELS)] \
                    if q["label"] in LABELS else LABELS[0]
                self.sel = i
                self.redraw()
        elif k == "c":
            self.rois, self.sel = [], None
            self.redraw()
        elif k == "enter":
            if not self.rois:
                self.status.set_text("No ROI drawn: drag one, or press n for 'nothing worth investigating'.")
                self.fig.canvas.draw_idle()
                return
            self._finish("accept")
        elif k == "n":
            if self.rois:
                self.status.set_text("n means no ROI: press c to clear the drawn ROIs first, or ENTER to accept them.")
                self.fig.canvas.draw_idle()
                return
            self._finish("none")
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

    # ------------------------------------------------------------ result
    def records(self):
        """The ROIs as saved: times in s on the buffer-4 clock, plus phase and expected overlap."""
        out = []
        for q in sorted(self.rois, key=lambda r: r["t0"]):
            t0, t1 = q["t0"] * 1e-3, q["t1"] * 1e-3
            info = describe(t0, t1, self.rp, self.expected)
            out.append(dict(t0=round(t0, 5), t1=round(t1, 5), t_mid=round(0.5 * (t0 + t1), 5),
                            duration_ms=round((t1 - t0) * 1e3, 2), r_mm=round(q["r"], 2), label=q["label"],
                            suggested_label=info["suggested"], expected=info["expected"],
                            expected_overlap=info["expected_overlap"],
                            phase_ms=None if info["phase_ms"] is None else round(info["phase_ms"], 1)))
        return out

    def _finish(self, action):
        res = dict(action=action)
        if action in ("accept", "none"):
            res.update(rois=self.records() if action == "accept" else [], clim_pct=CLIM_PCTS[self.clim_i])
        self.result = res
        self.fig.canvas.stop_event_loop()

    def run(self, snapshot=None):
        import matplotlib.pyplot as plt
        plt.show(block=False)
        self.fig.canvas.start_event_loop(timeout=0)
        if self.result is None:
            self.result = dict(action="quit")
        if snapshot and self.result["action"] in ("accept", "none"):
            try:
                self.sel = None
                self.ax.set_xlim(*self.full_xlim)
                self.redraw()
                self.status.set_text(("ACCEPTED: " if self.rois else "ACCEPTED: no ROI   ")
                                     + self._status_text())
                self.fig.savefig(snapshot, dpi=80, facecolor=SURF)
            except Exception:                                          # noqa: BLE001
                pass
        plt.close(self.fig)
        return self.result
