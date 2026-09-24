"""Diagnostic: does remove_flat change the passive per-window speeds?

passive.py calls slant_stack_speed(..., remove_flat=False) on the grounds that "the band-pass
already removed the bulk band", while the function's own docstring calls remove_flat "the
decisive fix" against the spatially-uniform band that otherwise drives the fit to the flattest
(highest-c) trial. This reruns the exact per-window views both ways and prints them side by side.
Reads only the already-saved M-lines + the buffer-4 IQ; writes nothing.
"""
import dataclasses
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
from dataclasses import replace

from swp.viz import runconfig as rc
from swp.viz.io import load_acquisition, load_mline
from swp.viz.mline import mline_from_points
from swp.viz.pipeline import run_pipeline, Step
from swp.viz.metrics import slant_stack_speed
from swp.mline.select import detect_line_bursts
from swp.passive import _build_views, _stride_acq, _frame_at_time, SPEED_CMAX

FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54"
CFG = os.path.join(REPO, "configs", "passive.yaml")


def mline(path, default_ns=250):
    pts, ns = load_mline(path)
    return mline_from_points(pts, ns or default_ns)


cfg = rc.load_config(CFG)
out = os.path.join(FOLDER, "output")
cfg["data"]["root"] = out
iq_path = rc.hdf5_path(cfg, 0)
mlines_dir = os.path.join(out, "mlines")

acq = load_acquisition(iq_path)
base = rc.build_pipeline_config(cfg, acq=acq)
gen = mline(os.path.join(mlines_dir, "passive_general_mline.npz"))

# Reproduce the burst windows exactly as the workflow does.
detect_band = cfg.get("detect", {}).get("band", [5.0, 150.0])
smoothing = [s for s in base.field_filters if s.name != "temporal_bandpass"]
ov_filters = [Step("temporal_bandpass", dict(f_lo=detect_band[0], f_hi=detect_band[1]))] + smoothing
ov = run_pipeline(_stride_acq(acq, 2), gen,
                  replace(base, directional=False, field_filters=ov_filters), focus=None)
windows, _ = detect_line_bursts(np.asarray(ov.st.data).T, np.asarray(ov.st.t),
                                window_ms=100.0, max_events=4)
views = _build_views(cfg, acq)

print(f"\nslant-stack speed, cmin=1.0 cmax={SPEED_CMAX}  "
      f"(F = remove_flat False = what passive.py uses today; T = True)\n")
print(f"{'window':>6s}  {'view':34s} {'sem F':>6s} {'c F':>8s}   {'sem T':>6s} {'c T':>8s}")
print("-" * 76)
for i, w in enumerate(windows):
    ml = mline(os.path.join(mlines_dir, f"passive_win{i}_mline.npz"))
    i0 = _frame_at_time(acq.t, w.t0 - 0.020)
    i1 = _frame_at_time(acq.t, w.t1 + 0.020) + 1
    acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
    for vname, vcfg in views:
        res = run_pipeline(acq_w, ml, vcfg, focus=None)
        sF, cF = slant_stack_speed(res.st, res.r0, cmin=1.0, cmax=SPEED_CMAX, remove_flat=False)
        sT, cT = slant_stack_speed(res.st, res.r0, cmin=1.0, cmax=SPEED_CMAX, remove_flat=True)
        flag = "  <-- changes" if abs(abs(cT) - abs(cF)) > 0.5 else ""
        print(f"{('win' + str(i)):>6s}  {vname[:34]:34s} {sF:6.2f} {cF:8.2f}   "
              f"{sT:6.2f} {cT:8.2f}{flag}")
