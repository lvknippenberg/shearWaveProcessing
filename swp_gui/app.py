"""shearWaveProcessing - interactive IQ/RF -> space-time method-exploration GUI.

Load a measurement folder, pick a push, and experiment with every stage of the pipeline from IQ/RF
to the space-time plot. Each filter stage is an **ordered, add/remove chain of methods** (e.g.
polynomial detrend -> band-pass), every method is tunable and its **source code is viewable**, the
B-mode shows the **spline M-line + offset lines**, the acquisition constants (f0/PRF/c/dz) are shown
read-only, per-step and global **reset** revert to defaults, and a built-in **no-push control** runs
the same recipe on the pre-push reference to reveal whether you image the ARF wave or cardiac motion.

Run:
    cd shearWaveProcessing
    KERAS_BACKEND=torch  <zea-python>  -m streamlit run swp_gui/app.py
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import subprocess
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml
import matplotlib.pyplot as plt
import streamlit as st

import core
import caenen
import registry as reg
import render
from swp.viz.core.geometry import robust_clim

st.set_page_config(page_title="SWE method explorer", layout="wide")
_SWEEP = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep"
DATA_ROOTS = {"In-vivo": _SWEEP + "/Invivo", "Phantom": _SWEEP + "/Phantom"}


def _folder_label(f):
    """Short, readable folder name (the long DefaultPatient_..._13-27-38 names don't fit the box)."""
    parts = os.path.basename(f).split("_")
    return f"{parts[0]} · {parts[-1]}" if len(parts) > 1 else os.path.basename(f)

# filter stages that are dynamic method-chains: (key, title, methods, default method names)
FILTER_STAGES = {
    "iq":       ("1 · IQ pre-filter (before displacement)", reg.IQ_METHODS, []),
    "motion":   ("3 · Cardiac-motion removal", reg.MOTION_METHODS, ["temporal_bandpass"]),
    "spatial":  ("4 · Spatial filtering", reg.SPATIAL_METHODS, ["spatial_smooth"]),
    "temporal": ("5 · Temporal filtering", reg.TEMPORAL_METHODS, ["temporal_moving_mean"]),
}


# --------------------------------------------------------------- cached heavy step
@st.cache_data(show_spinner="Loading / beamforming acquisition...")
def cached_acq(folder, meas, use_fine, ppw, margin_mm, is_rf, mline_pts):
    r = core.Recipe(use_fine=use_fine, ppw=ppw, margin_mm=margin_mm,
                    estimator="rf_ncc" if is_rf else "loupas")
    pts = np.array(mline_pts) if mline_pts else None
    return core.load_acq(folder, meas, r, mline_points=pts)


# --------------------------------------------------------------- widget helpers
def param_widget(p: reg.Param, kp: str):
    k = f"{kp}_{p.arg}"
    if p.kind == "bool":
        return bool(st.checkbox(p.label, value=bool(p.default), key=k, help=p.help))
    if p.kind == "select":
        return st.selectbox(p.label, p.options, index=p.options.index(p.default), key=k, help=p.help)
    if p.kind == "int":
        return int(st.number_input(p.label, min_value=int(p.lo), max_value=int(p.hi),
                                   value=int(p.default), step=int(p.step or 1), key=k, help=p.help))
    return float(st.number_input(p.label, min_value=float(p.lo), max_value=float(p.hi),
                                 value=float(p.default), step=float(p.step or 0.1), key=k, help=p.help))


def _method_block(methods, kp, code=True):
    """One method selectbox + its param widgets (+ code viewer). Returns (name, params)."""
    names = [m.name for m in methods]
    labels = {m.name: m.label for m in methods}
    sel = st.selectbox("method", names, format_func=lambda n: labels[n], key=f"{kp}_method",
                       label_visibility="collapsed")
    method = reg.by_name(methods, sel)
    if method.help:
        st.caption(method.help)
    params = {p.arg: param_widget(p, kp) for p in method.params}
    if code and method.fn is not None:
        with st.expander("view code"):
            try:
                st.code(inspect.getsource(method.fn), language="python")
            except (OSError, TypeError):
                st.write("(source unavailable)")
    return sel, params


def _reset_keys(prefix: str, keep=()):
    for k in [k for k in st.session_state if k.startswith(prefix) and k not in keep]:
        del st.session_state[k]


def dynamic_stage(stage_key, methods, default_methods, expanded=False):
    """A stage that is an ordered, add/remove chain of method blocks. Returns [(name, params), ...]."""
    ids_key, init_key = f"{stage_key}_ids", f"{stage_key}_init"
    if ids_key not in st.session_state:
        st.session_state[init_key] = {}
        st.session_state[ids_key] = []
        for dm in default_methods:
            i = st.session_state.get("_nid", 0)
            st.session_state["_nid"] = i + 1
            st.session_state[ids_key].append(i)
            st.session_state[init_key][i] = dm
    title = FILTER_STAGES[stage_key][0]
    n = len(st.session_state[ids_key])
    with st.expander(f"{title}  ·  {n} step(s)", expanded=expanded):
        steps, to_remove = [], None
        for pos, sid in enumerate(list(st.session_state[ids_key])):
            kp = f"{stage_key}_{sid}"
            # seed the method selectbox with this step's initial method the first time it renders
            if f"{kp}_method" not in st.session_state and sid in st.session_state.get(init_key, {}):
                st.session_state[f"{kp}_method"] = st.session_state[init_key][sid]
            with st.container(border=True):
                head = st.columns([3, 1, 1, 1, 1])
                enabled = head[0].toggle(f"step {pos + 1}", value=True, key=f"{kp}_on",
                                         help="enable / disable this step (kept in place, just skipped)")
                ids = st.session_state[ids_key]
                # reorder: move this step up / down (steps are keyed by sid, so state follows the move)
                if head[1].button("▲", key=f"{kp}_up", disabled=(pos == 0), help="move up"):
                    ids[pos - 1], ids[pos] = ids[pos], ids[pos - 1]
                    st.rerun()
                if head[2].button("▼", key=f"{kp}_dn", disabled=(pos == n - 1), help="move down"):
                    ids[pos + 1], ids[pos] = ids[pos], ids[pos + 1]
                    st.rerun()
                if head[3].button("↺", key=f"{kp}_rst", help="reset this step's params"):
                    _reset_keys(f"{kp}_", keep=(f"{kp}_method", f"{kp}_on"))   # keep method + on/off
                    st.rerun()
                if head[4].button("✕", key=f"{kp}_rm", help="remove this step"):
                    to_remove = sid
                step = _method_block(methods, kp)
                if enabled:
                    steps.append(step)
                else:
                    st.caption("⏸ disabled — not applied")
        if to_remove is not None:
            st.session_state[ids_key].remove(to_remove)
            _reset_keys(f"{stage_key}_{to_remove}_")
            st.rerun()
        bcols = st.columns(2)
        if bcols[0].button("➕ add step", key=f"{stage_key}_add"):
            i = st.session_state.get("_nid", 0)
            st.session_state["_nid"] = i + 1
            st.session_state[ids_key].append(i)
            st.rerun()
        if bcols[1].button("reset stage", key=f"{stage_key}_stagerst"):
            for sid in st.session_state[ids_key]:
                _reset_keys(f"{stage_key}_{sid}_")
            del st.session_state[ids_key]
            st.rerun()
    return steps


def single_stage(methods, kp, code=True):
    return _method_block(methods, kp, code=code)


# --------------------------------------------------------------- sidebar (global display + actions)
st.sidebar.title("Display")
fig_scale = st.sidebar.slider("Figure size", 0.4, 1.8, 0.9, 0.1, key="figscale",
                              help="Scale all result figures up or down.")
per_cell_clim = st.sidebar.checkbox("Colour scale per plot", value=False, key="percell",
                                    help="Each panel uses its own scale (default: shared per quantity "
                                         "column, so a large previous row won't squash the current).")
hide_prev = st.sidebar.checkbox("Hide previous-config row", value=False, key="hideprev")
st.sidebar.divider()
st.sidebar.subheader("Animation (all pushes)")
anim_fps = st.sidebar.slider("frames / sec", 1, 8, 2, 1)
animate = st.sidebar.button("▶ Animate all pushes")
if st.sidebar.button("Reset ALL to defaults"):
    for k in list(st.session_state):
        if k not in ("data_kind", "figscale", "percell", "hideprev"):
            del st.session_state[k]
    st.rerun()

# --------------------------------------------------------------- main: 3 columns (data | pipeline | results)
st.title("Shear-wave method explorer  ·  IQ/RF → space-time")
c_data, c_pipe, c_res = st.columns([0.75, 1.15, 1.5])

with c_data:
    st.subheader("1 · Data")
    kinds = list(DATA_ROOTS) + (["Caenen (pig)"] if caenen.available() else [])
    data_kind = st.selectbox("Data set", kinds, key="data_kind")
    is_caenen = data_kind == "Caenen (pig)"
    folder = "caenen" if is_caenen else None
    if is_caenen:
        measurements = caenen.pushes()
        if not measurements:
            st.error("No Caenen push_*.h5 found. Run export_push.m first.")
            st.stop()
        meas = st.selectbox("Push", measurements)
        mline_source = "manual"
        st.caption("Cartesian grid, M-line drawn on the wall (SWE_results/push_<p>/mline.npz).")
    else:
        root = DATA_ROOTS[data_kind]
        st.session_state["root"] = root
        folders = core.list_measurement_folders(root)
        if not folders:
            st.warning(f"No measurement folders under {root}.")
            st.stop()
        folder = st.selectbox("Measurement folder", folders, format_func=_folder_label)
        measurements = core.list_measurements(folder)
        if not measurements:
            st.error("No beamformed buffer-2 IQ. Run `python run.py beamform <folder>` first.")
            st.stop()
        meas = st.selectbox("Push / measurement", measurements)
        mline_source = st.selectbox("M-line source", ["auto", "manual", "horizontal_push"],
                                    help="auto: saved .npz if present, else horizontal line at push depth.")
        if st.button("Draw / redraw M-line (opens picker)"):
            try:
                subprocess.Popen([sys.executable, os.path.join(_ROOT, "swp_gui", "draw_mline.py"),
                                  folder, str(meas)], env={**os.environ, "MPLBACKEND": "TkAgg"})
                st.info("Picker launched in a separate window. Draw, press Enter, then re-run.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"could not launch picker: {exc}")
    st.divider()
    use_fine = False if is_caenen else st.checkbox(
        "Fine-grid re-beamform", value=False,
        help="Re-beamform buffer 2 on a fine axial grid around the M-line (auto for RF cross-correlation).")
    if use_fine:
        ppw = st.slider("pixels / wavelength", 2.0, 20.0, 8.0, 1.0)
        margin_mm = st.slider("ROI margin (mm)", 1.0, 10.0, 3.0, 0.5)
    else:
        ppw, margin_mm = 8.0, 3.0
    nopush = st.checkbox("Compute no-push control", value=True)
    auto = st.checkbox("Auto-run on change", value=not use_fine)
    run = st.button("Run recipe", type="primary")

with c_pipe:
    st.subheader("2 · Pipeline")
    with st.container(height=680):                         # scrollable → results stay on screen
        iq_steps = dynamic_stage("iq", reg.IQ_METHODS, FILTER_STAGES["iq"][2], expanded=False)
        with st.expander("2 · Displacement estimation", expanded=True):
            est_m, est_p = single_stage(reg.ESTIMATOR_METHODS, "est")
            mode = st.selectbox("reference mode", reg.MODES, index=0, key="mode")
            quantity = st.selectbox("primary quantity (drives the metric/speed; all 3 are displayed)",
                                    reg.QUANTITIES, index=0, key="quantity")
            drop_first = int(st.number_input("drop first frames", 0, 5, 1, 1, key="dropf"))
            if est_m == "rf_ncc" and not use_fine:
                st.caption("RF NCC forces the fine grid (enable it under Data).")
        motion_steps = dynamic_stage("motion", reg.MOTION_METHODS, FILTER_STAGES["motion"][2], expanded=True)
        spatial_steps = dynamic_stage("spatial", reg.SPATIAL_METHODS, FILTER_STAGES["spatial"][2])
        temporal_steps = dynamic_stage("temporal", reg.TEMPORAL_METHODS, FILTER_STAGES["temporal"][2])
        with st.expander("6 · M-line offset averaging", expanded=False):
            offsets = int(st.number_input("number of lines", 1, 21, 7, 2, key="noff"))
            step_mm = float(st.number_input("spacing (mm)", 0.1, 3.0, 0.8, 0.1, key="ostep"))
            agg = st.selectbox("combine", ["mean", "median"], key="magg")
        with st.expander("7 · Directional filter", expanded=False):
            direc, _ = single_stage(reg.DIRECTIONAL_METHODS, "dir")
        with st.expander("8 · Speed estimator", expanded=False):
            speed, _ = single_stage(reg.SPEED_METHODS_UI, "spd")

recipe = core.Recipe(
    iq_steps=iq_steps, estimator=est_m, est_params=est_p, mode=mode, quantity=quantity,
    drop_first=drop_first, use_fine=use_fine, ppw=ppw, margin_mm=margin_mm,
    motion_steps=motion_steps, spatial_steps=spatial_steps, temporal_steps=temporal_steps,
    offsets=offsets, offset_step_mm=step_mm, mline_agg=agg, mline_source=mline_source,
    directional=direc, speed=speed)
with c_pipe:
    st.download_button("Download recipe (YAML)", yaml.safe_dump(dataclasses.asdict(recipe)),
                       file_name="swe_recipe.yaml")


@st.cache_data(show_spinner="Loading Caenen push...")
def cached_caenen(push):
    return caenen.load(push)


def do_run():
    is_rf = recipe.estimator == "rf_ncc"
    if is_caenen:
        acq, ml = cached_caenen(meas)
    else:
        pts = core.mline_points_for(folder, meas)
        acq = cached_acq(folder, meas, use_fine or is_rf, ppw, margin_mm, is_rf,
                         tuple(map(tuple, pts)) if pts is not None else None)
        ml = core.load_mline_for(folder, meas, acq, recipe)
    # run all three quantities (displacement / velocity / acceleration) with the same recipe
    res_by_q = {q: core.run_recipe(acq, ml, core.to_config(dataclasses.replace(recipe, quantity=q), acq))
                for q in render.QORDER}
    res = res_by_q[recipe.quantity]                      # primary drives metrics / speed / B-mode
    out = {"acq": acq, "ml": ml, "res": res, "res_by_q": res_by_q,
           "met": core.compute_metrics(res), "dparams": core.data_params(acq),
           "svd": core.svd_spectrum(acq, recipe)}         # spectrum for the SVD-cutoff tuning plot
    if nopush:
        resn_by_q = {}
        for q in render.QORDER:
            rn = core.run_recipe(acq, ml, core.to_config(dataclasses.replace(recipe, quantity=q), acq),
                                 nopush=True)
            if rn is not None:
                resn_by_q[q] = rn
        if resn_by_q:
            out["resn_by_q"] = resn_by_q
            out["resn"] = resn_by_q.get(recipe.quantity, next(iter(resn_by_q.values())))
            out["metn"] = core.compute_metrics(out["resn"])
    return out


def _effective_dict(recipe):
    """asdict(recipe) with no-op 'none' steps stripped, so adding/removing a 'none' step is a TRUE no-op
    for the run/history keys (it produces identical field_filters) - no recompute, no spurious row."""
    d = dataclasses.asdict(recipe)
    for k in ("iq_steps", "motion_steps", "spatial_steps", "temporal_steps"):
        d[k] = [s for s in d[k] if s[0] != "none"]
    return d


def _snapshot(out):
    # `key` = the effective recipe (minus the display-only primary quantity) so we keep the PREVIOUS
    # DISTINCT config as row 3 - and a 'none' step doesn't count as a change.
    key = repr({k: v for k, v in _effective_dict(recipe).items() if k != "quantity"})
    return dict(cells={q: render.cell_of(out["res_by_q"][q]) for q in render.QORDER},
                label=core.recipe_label(recipe), oc=out["met"]["origin_coherence"], key=key)


# Only (re)compute the pipeline when the recipe/data actually change (or Run is pressed) - NOT on every
# rerun. Otherwise a click on the speed plot, a figure-size tweak, or a tab switch would recompute all
# six pipeline runs (3 quantities x push/no-push). Interactions that don't change this key are instant.
_run_key = (repr(_effective_dict(recipe)), data_kind, meas, use_fine, ppw, margin_mm, nopush)
if run or (auto and st.session_state.get("run_key") != _run_key):
    try:
        st.session_state["out"] = do_run()
        st.session_state["run_key"] = _run_key
        st.session_state["err"] = None
        snap = _snapshot(st.session_state["out"])
        prev = st.session_state.get("history", [])
        if not prev:                                     # first run
            hist = [snap]
        elif prev[0]["key"] == snap["key"]:              # same config re-run: refresh current, keep previous
            hist = [snap] + prev[1:2]
        else:                                            # config changed: old current -> previous (row 3)
            hist = [snap, prev[0]]
        st.session_state["history"] = hist
    except Exception as exc:  # noqa: BLE001
        st.session_state["err"] = f"{type(exc).__name__}: {exc}"

with c_res:
    st.subheader("3 · Results")
    if st.session_state.get("err"):
        st.error(st.session_state["err"])
    out = st.session_state.get("out")
    if not out:
        st.info("Set up a recipe and press **Run recipe** (or enable auto-run).")
    else:
        acq, ml, res, met = out["acq"], out["ml"], out["res"], out["met"]
        hist = st.session_state.get("history", [])
        if st.session_state.get("speed_ctx") != (data_kind, meas):   # reset the line when the push changes
            st.session_state["speed_ctx"] = (data_kind, meas)
            st.session_state.pop("speed_pts", None); st.session_state.pop("speed_line", None)
        m = st.columns(4)
        m[0].metric("★ origin coherence", f"{met['origin_coherence']:.3f}",
                    help="THE wave-clarity metric (validated vs human scoring across voltage + "
                         "in-vivo; symmetric outward-from-r0, early-onset). Higher = clearer wave.")
        m[1].metric("outward fraction", f"{met['outward_fraction']:.2f}")
        m[2].metric("slant semblance", f"{met['slant_semblance']:.2f}",
                    help=f"signed speed {met['slant_speed']:.2f} m/s")
        m[3].metric(f"amp p95 [{met['amp_unit']}]", f"{met['amp_p95']:.1f}")

        # one grid: rows = before push (no-push) · current · previous ; cols = disp / vel / acc
        rows, labels = [], []
        if "resn_by_q" in out:
            rows.append({q: render.cell_of(out["resn_by_q"][q]) for q in out["resn_by_q"]})
            labels.append("before push")
        rows.append({q: render.cell_of(out["res_by_q"][q]) for q in render.QORDER})
        labels.append(f"current config (oc={met['origin_coherence']:.2f})")
        if len(hist) >= 2 and not hide_prev:
            rows.append(hist[1]["cells"])
            labels.append(f"previous config (oc={hist[1]['oc']:.2f})")
        fig_g = render.fig_quantity_grid(rows, labels, scale=fig_scale, per_cell=per_cell_clim,
                                         speed_line=st.session_state.get("speed_line"))
        st.pyplot(fig_g, width='content'); plt.close(fig_g)
        if len(hist) < 2:
            st.caption("↳ change any pipeline setting to fill the **previous config** row.")

        if out.get("svd"):
            s = out["svd"]
            st.pyplot(render.fig_svd_spectrum(s["S"], s["nr"], s["nh"], s["domain"], scale=fig_scale),
                      width='content')
            st.caption("SVD cutoffs: put **remove-low** at the end of the steep clutter drop, "
                       "**remove-high** where the tail goes flat (noise).")

        cap = (f"**{recipe.quantity}** · r0={res.r0*1e3:.1f}mm · c={met['c_pos']:.2f}/{met['c_neg']:.2f} m/s "
               f"· oc L/R {met['oc_left']:.2f}/{met['oc_right']:.2f}")
        if "resn_by_q" in out:
            ratio = out["metn"]["amp_p95"] / (out["met"]["amp_p95"] + 1e-9)
            cap += (f" · before/after amp = **{ratio:.2f}** "
                    + ("⚠️ cardiac" if ratio > 0.7 else "✅ push"))
        st.caption(cap)

        # ---- manual 2-click speed measurement (Plotly click capture) ----
        with st.expander("📏 Manual speed (2-click)", expanded=bool(st.session_state.get("speed_line"))):
            dq = st.radio("draw on", render.QORDER, index=1, horizontal=True, key="speed_q",
                          format_func=lambda q: q[:3])
            cell = render.cell_of(out["res_by_q"][dq])
            rc = (cell["r"] > 0.1 * cell["r"][-1]) & (cell["r"] < 0.9 * cell["r"][-1])
            u = render.QUNITS[dq][0]
            clim = (robust_clim(cell["data"], rc, 97) * u) or float(np.nanpercentile(np.abs(cell["data"] * u), 97))
            ev = st.plotly_chart(render.fig_speed_plotly(cell, clim, st.session_state.get("speed_line"),
                                                         scale=fig_scale),
                                 key=f"spd_{data_kind}_{meas}_{dq}", on_select="rerun",
                                 selection_mode="points")
            def _sel_points(e):
                s = (e.get("selection") if hasattr(e, "get") else getattr(e, "selection", None)) if e else None
                if s is None:
                    return []
                return (s.get("points") if hasattr(s, "get") else getattr(s, "points", None)) or []

            newpt = None
            try:
                pp = _sel_points(ev)
                if pp:
                    p = pp[-1]
                    x = p["x"] if hasattr(p, "__getitem__") else p.x
                    y = p["y"] if hasattr(p, "__getitem__") else p.y
                    newpt = (round(float(x), 4), round(float(y), 4))
            except Exception:  # noqa: BLE001 - never break the run on an odd selection payload
                newpt = None
            pts = st.session_state.get("speed_pts", [])
            if newpt is not None and (not pts or pts[-1] != newpt):
                pts = [newpt] if len(pts) >= 2 else pts + [newpt]
                st.session_state["speed_pts"] = pts
                if len(pts) == 2:                             # line complete -> redraw grid above with it
                    st.session_state["speed_line"] = pts
                    st.rerun()
                else:
                    st.session_state.pop("speed_line", None)  # 1st point: no extra rerun
            cc = st.columns([3, 1])
            line = st.session_state.get("speed_line")
            if line:
                (r0m, t0m), (r1m, t1m) = line
                dr, dt = r1m - r0m, t1m - t0m
                spd = abs(dr / dt) if abs(dt) > 1e-6 else float("inf")
                side = "right of r0" if (r0m + r1m) / 2 >= res.r0 * 1e3 else "left of r0"
                cc[0].success(f"**speed = {spd:.2f} m/s**  ·  Δr={dr:+.1f} mm, Δt={dt:+.1f} ms  ·  {side}")
            else:
                cc[0].caption(f"click the wavefront **start** then **end** — {len(pts)}/2 picked")
            if cc[1].button("clear", key="speed_clear"):
                st.session_state.pop("speed_pts", None); st.session_state.pop("speed_line", None)
                st.rerun()

        tab_b, tab_c, tab_a = st.tabs(["B-mode + M-line", "recipe / history", "acquisition"])
        with tab_b:
            img, ext = caenen.bmode(acq) if is_caenen else core.bmode_for_display(folder, meas, acq)
            fig_b = render.fig_bmode_mline(img, ext, ml, push_xz=(acq.push_x, acq.push_z),
                                           n_offsets=recipe.offsets,
                                           offset_step_m=recipe.offset_step_mm * 1e-3, scale=fig_scale)
            st.pyplot(fig_b, width='content'); plt.close(fig_b)
        with tab_c:
            st.caption(f"current: {hist[0]['label']}" if hist else "")
            if len(hist) >= 2:
                st.caption(f"previous: {hist[1]['label']}")
                if st.button("clear comparison history"):
                    st.session_state["history"] = []
                    st.rerun()
        with tab_a:
            st.table({k: [f"{v:.3f}" if isinstance(v, float) else v]
                      for k, v in out["dparams"].items()})

# --------------------------------------------------------------- animation across all pushes
def _push_clim(res):
    st_ = res.st
    unit = 1e6 if st_.quantity == "displacement" else 1e3
    rc = (st_.r > 0.1 * st_.r[-1]) & (st_.r < 0.9 * st_.r[-1])
    return (robust_clim(st_.data, rc, pct=97) * unit) or float(np.nanpercentile(np.abs(st_.data * unit), 97))


if animate:
    try:
        is_rf = recipe.estimator == "rf_ncc"
        prog = st.progress(0.0, text="computing all pushes with the current recipe...")
        items = []
        for i, mz in enumerate(measurements):
            if is_caenen:
                acq, ml = cached_caenen(mz)
            else:
                pts = core.mline_points_for(folder, mz)
                acq = cached_acq(folder, mz, use_fine or is_rf, ppw, margin_mm, is_rf,
                                 tuple(map(tuple, pts)) if pts is not None else None)
                ml = core.load_mline_for(folder, mz, acq, recipe)
            res = core.run_recipe(acq, ml, core.to_config(recipe, acq))
            met = core.compute_metrics(res)
            img, ext = caenen.bmode(acq) if is_caenen else core.bmode_for_display(folder, mz, acq)
            items.append((mz, acq, ml, res, met, img, ext))
            prog.progress((i + 1) / len(measurements), text=f"push {i+1}/{len(measurements)}")
        clim = float(np.percentile([_push_clim(it[3]) for it in items], 75))  # shared amplitude scale
        frames = [render.frame_rgb(img, ext, ml, res, mz, met, push_xz=(acq.push_x, acq.push_z),
                                   n_offsets=recipe.offsets, offset_step_m=recipe.offset_step_mm * 1e-3,
                                   clim=clim)
                  for (mz, acq, ml, res, met, img, ext) in items]
        st.session_state["anim"] = render.build_gif(frames, fps=anim_fps)
        st.session_state["anim_err"] = None
        prog.empty()
    except Exception as exc:  # noqa: BLE001
        st.session_state["anim_err"] = f"{type(exc).__name__}: {exc}"

if st.session_state.get("anim_err"):
    st.error(f"animation failed: {st.session_state['anim_err']}")
if st.session_state.get("anim"):
    st.divider()
    st.subheader(f"Animation across all {len(measurements)} pushes (looping) — shared colour scale")
    st.image(st.session_state["anim"])
    st.download_button("Download animation GIF", st.session_state["anim"], file_name="all_pushes.gif")
