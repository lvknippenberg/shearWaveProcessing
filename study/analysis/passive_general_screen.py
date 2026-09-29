"""General-line screen of the manual passive study - cache the whole-recording signals.

For every folder whose general line is drawn (scripts/passive_manual.py), recompute along that
line, over the WHOLE buffer-4 recording, exactly what the detector saw plus the default view:

* the detection overview (displacement 5-150 Hz, stride 2, as swp.passive.detect_windows) and its
  along-line energy -> reproduces the stored windows (a check that this is the detector's signal);
* the "velocity gauss" view of configs/passive_manual.yaml (the default recipe);
* the R-peaks on the buffer-4 clock and the RR interval.

Nothing is written to the folders; one npz per folder goes to study/analysis/general_screen_cache/
(a regenerable cache). Analysis: passive_general_screen_eval.py.

    python study/analysis/passive_general_screen.py [--part 0/3]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("KERAS_BACKEND", "torch")

from swp.manual import store as S          # noqa: E402
from swp.manual import worker as Wk         # noqa: E402

CACHE = os.path.join(REPO, "study", "analysis", "general_screen_cache")


def cache_path(folder):
    return os.path.join(CACHE, f"{os.path.basename(os.path.dirname(folder))}__{os.path.basename(folder)}.npz")


def compute(folder):
    from swp.passive import _build_views, _stride_acq
    from swp.viz import runconfig as rc
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline, Step
    from swp.mline.select import energy_along_line
    from swp.acquisition.triggerlog import buffer_timing, clean_r_peaks

    p = S.Paths(folder)
    pts = S.load_points(p.general_npz)
    cfg = Wk._cfg()
    cfg["data"]["root"] = p.output
    acq = Wk._load(p, [pts])
    ml = mline_from_points(pts, S.N_SAMPLES)

    # detection overview, as swp.passive.detect_windows
    base = rc.build_pipeline_config(cfg, acq=acq)
    band = cfg.get("detect", {}).get("band", [5.0, 150.0])
    smoothing = [s for s in base.field_filters if s.name != "temporal_bandpass"]
    ov_cfg = replace(base, directional=False,
                     field_filters=[Step("temporal_bandpass", dict(f_lo=band[0], f_hi=band[1]))] + smoothing,
                     quantity=str(cfg.get("detect", {}).get("quantity", "displacement")))
    ov = run_pipeline(_stride_acq(acq, Wk.OVERVIEW_STRIDE), ml, ov_cfg, focus=None)
    D_ov = np.asarray(ov.st.data).T                            # (n_s, n_t)
    e_raw, e_masked = energy_along_line(D_ov, 30)

    # the default view over the whole recording
    views = dict(_build_views(cfg, acq))
    vres = run_pipeline(acq, ml, views["velocity gauss"], focus=None)

    bt = buffer_timing(folder, 4)
    r_rel, rr_ms = np.array([]), np.nan
    if bt is not None:
        kept, rr_ms = clean_r_peaks(bt.r_peaks_ms)
        if kept is not None:
            r_rel = (np.asarray(kept, float) - bt.t0_ms) * 1e-3
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(
        cache_path(folder), folder=np.array(folder),
        ov_t=np.asarray(ov.st.t), ov_r=np.asarray(ov.st.r), ov_D=D_ov.astype(np.float32),
        e_raw=e_raw, e_masked=e_masked,
        v_t=np.asarray(vres.st.t), v_r=np.asarray(vres.st.r),
        v_data=np.asarray(vres.st.data, np.float32),         # (n_t, n_r)
        r_peaks_s=r_rel, rr_ms=np.array(float(rr_ms) if rr_ms is not None else np.nan),
        general_hash=np.array(S.read_json(p.general_json)["hash"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--part", default="0/1", help="i/N: this process takes every N-th folder")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    i, n = (int(x) for x in a.part.split("/"))
    folders = [f for f in S.find_folders(a.root) if S.state(f)["stage"] in ("done", "need-slopes", "need-events",
                                                                             "processing")]
    folders = folders[i::n]
    print(f"{len(folders)} folder(s) in part {a.part}", flush=True)
    for f in folders:
        if os.path.exists(cache_path(f)) and not a.force:
            print(f"  cached  {f}", flush=True)
            continue
        t = time.time()
        try:
            compute(f)
            print(f"  ok {time.time() - t:5.0f} s  {f}", flush=True)
        except Exception:                                      # noqa: BLE001
            print(f"  FAILED  {f}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
