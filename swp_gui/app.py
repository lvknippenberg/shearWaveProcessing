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
import registry as reg
import render
from swp.viz.core.geometry import robust_clim

st.set_page_config(page_title="SWE method explorer", layout="wide")
DEFAULT_ROOT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Invivo"

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
                head = st.columns([6, 1, 1])
                head[0].markdown(f"**step {pos + 1}**")
                if head[1].button("↺", key=f"{kp}_rst", help="reset this step's params"):
                    _reset_keys(f"{kp}_", keep=(f"{kp}_method",))   # params -> defaults, keep method
                    st.rerun()
                if head[2].button("✕", key=f"{kp}_rm", help="remove this step"):
                    to_remove = sid
                steps.append(_method_block(methods, kp))
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


# --------------------------------------------------------------- sidebar
st.sidebar.title("Data")
root = st.sidebar.text_input("Data root folder", value=st.session_state.get("root", DEFAULT_ROOT))
st.session_state["root"] = root
folders = core.list_measurement_folders(root)
if not folders:
    st.sidebar.warning("No measurement folders found under this root.")
    st.stop()
folder = st.sidebar.selectbox("Measurement folder", folders, format_func=os.path.basename)
measurements = core.list_measurements(folder)
if not measurements:
    st.sidebar.error("No beamformed buffer-2 IQ. Run `python run.py beamform <folder>` first.")
    st.stop()
meas = st.sidebar.selectbox("Push / measurement", measurements)
mline_source = st.sidebar.selectbox("M-line source", ["auto", "manual", "horizontal_push"],
                                    help="auto: saved .npz if present, else horizontal line at push depth.")
if st.sidebar.button("Draw / redraw M-line (opens picker)"):
    try:
        subprocess.Popen([sys.executable, os.path.join(_ROOT, "swp_gui", "draw_mline.py"),
                          folder, str(meas)], env={**os.environ, "MPLBACKEND": "TkAgg"})
        st.sidebar.info("Picker launched in a separate window. Draw, press Enter, then re-run.")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"could not launch picker: {exc}")

st.sidebar.divider()
st.sidebar.subheader("Fine-grid re-beamform")
use_fine = st.sidebar.checkbox("Use fine grid (local RF re-beamform)", value=False,
                               help="Re-beamform buffer 2 on a fine axial grid around the M-line. "
                                    "Required (auto) for RF cross-correlation.")
ppw = st.sidebar.slider("pixels / wavelength", 2.0, 20.0, 8.0, 1.0)
margin_mm = st.sidebar.slider("ROI margin (mm)", 1.0, 10.0, 3.0, 0.5)

st.sidebar.divider()
nopush = st.sidebar.checkbox("Compute no-push control", value=True)
auto = st.sidebar.checkbox("Auto-run on change", value=not use_fine)
run = st.sidebar.button("Run recipe", type="primary")

st.sidebar.divider()
st.sidebar.subheader("Animation (all pushes)")
anim_fps = st.sidebar.slider("frames / sec", 1, 8, 2, 1)
animate = st.sidebar.button("▶ Animate all pushes")
if st.sidebar.button("Reset ALL to defaults"):
    for k in list(st.session_state):
        if k not in ("root",):
            del st.session_state[k]
    st.rerun()

# --------------------------------------------------------------- main: pipeline + output
st.title("Shear-wave method explorer  ·  IQ/RF → space-time")
left, right = st.columns([1.0, 1.35])

with left:
    st.subheader("Pipeline")
    iq_steps = dynamic_stage("iq", reg.IQ_METHODS, FILTER_STAGES["iq"][2], expanded=False)
    with st.expander("2 · Displacement estimation", expanded=True):
        est_m, est_p = single_stage(reg.ESTIMATOR_METHODS, "est")
        mode = st.selectbox("reference mode", reg.MODES, index=0, key="mode")
        quantity = st.selectbox("quantity", reg.QUANTITIES, index=0, key="quantity")
        drop_first = int(st.number_input("drop first frames", 0, 5, 1, 1, key="dropf"))
        if est_m == "rf_ncc" and not use_fine:
            st.caption("RF NCC forces the fine grid (enable it in the sidebar to tune density).")
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
with left:
    st.download_button("Download recipe (YAML)", yaml.safe_dump(dataclasses.asdict(recipe)),
                       file_name="swe_recipe.yaml")


def do_run():
    is_rf = recipe.estimator == "rf_ncc"
    pts = core.mline_points_for(folder, meas)
    acq = cached_acq(folder, meas, use_fine or is_rf, ppw, margin_mm, is_rf,
                     tuple(map(tuple, pts)) if pts is not None else None)
    ml = core.load_mline_for(folder, meas, acq, recipe)
    cfg = core.to_config(recipe, acq)
    res = core.run_recipe(acq, ml, cfg)
    out = {"acq": acq, "ml": ml, "res": res, "met": core.compute_metrics(res),
           "dparams": core.data_params(acq)}
    if nopush:
        resn = core.run_recipe(acq, ml, cfg, nopush=True)
        if resn is not None:
            out["resn"], out["metn"] = resn, core.compute_metrics(resn)
    return out


def _snapshot(out):
    res = out["res"]
    return dict(data=res.st.data.copy(), r=res.st.r.copy(), t=res.st.t.copy(),
                quantity=res.st.quantity, r0=res.r0, label=core.recipe_label(recipe),
                oc=out["met"]["origin_coherence"])


if run or auto:
    try:
        st.session_state["out"] = do_run()
        st.session_state["err"] = None
        hist = [_snapshot(st.session_state["out"])] + st.session_state.get("history", [])
        st.session_state["history"] = hist[:3]           # current + last 2
    except Exception as exc:  # noqa: BLE001
        st.session_state["err"] = f"{type(exc).__name__}: {exc}"

with right:
    st.subheader("Result")
    if st.session_state.get("err"):
        st.error(st.session_state["err"])
    out = st.session_state.get("out")
    if not out:
        st.info("Set up a recipe and press **Run recipe** (or enable auto-run).")
    else:
        acq, ml, res, met = out["acq"], out["ml"], out["res"], out["met"]
        c1, c2 = st.columns([1.0, 1.25])
        with c1:
            img, ext = core.bmode_for_display(folder, meas, acq)
            fig_b = render.fig_bmode_mline(img, ext, ml, push_xz=(acq.push_x, acq.push_z),
                                           n_offsets=recipe.offsets,
                                           offset_step_m=recipe.offset_step_mm * 1e-3)
            st.pyplot(fig_b); plt.close(fig_b)
        with c2:
            fig_s, _ = render.fig_spacetime(res, f"{recipe.estimator} · r0={res.r0*1e3:.1f}mm · "
                                                 f"{res.speed.label()}")
            st.pyplot(fig_s); plt.close(fig_s)
        m = st.columns(4)
        m[0].metric("★ origin coherence", f"{met['origin_coherence']:.3f}",
                    help="THE wave-clarity metric (validated against human scoring across voltage + "
                         "in-vivo; symmetric outward-from-r0, early-onset). Higher = clearer wave.")
        m[1].metric("outward fraction", f"{met['outward_fraction']:.2f}")
        m[2].metric("slant semblance", f"{met['slant_semblance']:.2f}",
                    help=f"signed speed {met['slant_speed']:.2f} m/s")
        m[3].metric(f"amp p95 [{met['amp_unit']}]", f"{met['amp_p95']:.1f}")
        st.caption(f"speed c_pos/c_neg = {met['c_pos']:.2f}/{met['c_neg']:.2f} m/s "
                   f"(q={met['speed_quality']:.2f}); symmetric oc left/right = "
                   f"{met['oc_left']:.2f}/{met['oc_right']:.2f}")

        hist = st.session_state.get("history", [])
        if len(hist) >= 2:
            st.markdown("**Compare with previous runs** (space-time; shared colour scale)")
            fig_h = render.fig_history_strip(hist)
            if fig_h is not None:
                st.pyplot(fig_h); plt.close(fig_h)
            if st.button("clear comparison history"):
                st.session_state["history"] = []
                st.rerun()
        with st.expander("Acquisition constants (from Verasonics data — fixed, not tunable)"):
            st.table({k: [f"{v:.3f}" if isinstance(v, float) else v]
                      for k, v in out["dparams"].items()})

        if "resn" in out:
            st.divider()
            st.subheader("No-push control (same recipe on the pre-push reference)")
            fig_n = render.fig_push_vs_nopush(res, out["resn"], met, out["metn"])
            st.pyplot(fig_n); plt.close(fig_n)
            ratio = out["metn"]["amp_p95"] / (met["amp_p95"] + 1e-9)
            verdict = ("⚠️ no-push is comparably strong → likely dominated by cardiac motion, not the push"
                       if ratio > 0.7 else
                       "✅ the push window is substantially stronger than the no-push window")
            st.markdown(f"no-push / push amplitude ratio = **{ratio:.2f}** — {verdict}")

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
            pts = core.mline_points_for(folder, mz)
            acq = cached_acq(folder, mz, use_fine or is_rf, ppw, margin_mm, is_rf,
                             tuple(map(tuple, pts)) if pts is not None else None)
            ml = core.load_mline_for(folder, mz, acq, recipe)
            res = core.run_recipe(acq, ml, core.to_config(recipe, acq))
            met = core.compute_metrics(res)
            img, ext = core.bmode_for_display(folder, mz, acq)
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
