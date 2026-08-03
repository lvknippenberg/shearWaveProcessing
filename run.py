"""shearWaveProcessing -- unified cardiac SWE pipeline driver.

Three modular stages (each runnable on its own if its inputs already exist):

  1. convert   Verasonics .mat -> zea .hdf5 (converted RF buffers). No GPU needed.
  2. beamform  RF -> IQ + B-mode GIFs + per-measurement shear-wave IQ (with the ARF push
               location and scan parameters). Reuses converted files from stage 1 if present.
  3. viz       IQ -> shear-wave space-time plots (+ speed), for ACTIVE (buffer 2 + buffer 5)
               or PASSIVE (buffer 4) SWE. Draws the M-line interactively if it is not saved yet.

Usage
-----
  python run.py convert  <folder>
  python run.py beamform <folder> [--no-gifs] [--overwrite]
  python run.py viz      <folder> --config configs/active.yaml  [--meas N]
  python run.py viz      <folder> --config configs/passive.yaml
  python run.py all      <folder> --config configs/active.yaml           # 1 -> 2 -> 3

Stages 1 and 2 run together in one RF read when you call `beamform` (the converted files are a
by-product), so `convert` is only needed if you want the zea database copies without beamforming.
`<folder>` is a measurement folder containing the Verasonics .mat (CombinedData.mat). Stage 3
reads the IQ written by stage 2 into `<folder>/output`.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))


# ======================================================================================
# Stage 1 + 2 -- acquisition
# ======================================================================================
def stage_convert(folder, overwrite=False):
    from swp.acquisition import convert_folder
    return convert_folder(folder, overwrite=overwrite)


def stage_beamform(folder, make_gifs=True, overwrite=False, save_converted=True):
    from swp.acquisition import process_folder
    return process_folder(folder, make_gifs=make_gifs, overwrite=overwrite,
                          save_converted=save_converted)


# ======================================================================================
# Stage 3 -- visualization (ported iq2sws core + Zea interactive M-line)
# ======================================================================================
# Motion-comp filters: the multi-band confirmation view replaces these with a per-band
# band-pass, keeping only the spatial/temporal-smoothing steps from the config.
MOTION_FILTERS = {"polynomial_drift", "temporal_highpass", "temporal_bandpass",
                  "reference_motion_comp", "adaptive_highpass"}


def _smoothing_steps(pcfg):
    return [s for s in pcfg.field_filters if s.name not in MOTION_FILTERS]


def _band_recipe(smooth, band):
    from swp.viz.pipeline import Step
    if band is None:
        return smooth
    f_lo, f_hi = band
    return [Step("temporal_bandpass", dict(f_lo=f_lo, f_hi=f_hi))] + smooth


def _ensure_mline(npz_path, bmode_path, frame, title, n_samples=250):
    """Load a saved M-line if present, else draw one interactively on a B-mode frame + save it.

    Uses the ported Zea selector: active -> buffer-5 frame `m`; passive -> a buffer-4 frame.
    The saved .npz (points (k,2)=(x,z) m + n_samples) is what swp.viz loads for sampling.
    """
    if os.path.exists(npz_path):
        print(f"  [M-line] reuse {os.path.basename(npz_path)}")
        return
    from swp.mline import load_bmode_frame, select_mline, save_mline, draw_mline_on_bmode
    if not os.path.exists(bmode_path):
        raise SystemExit(f"cannot draw M-line: B-mode file not found: {bmode_path}")
    bmode_u8, coords, n = load_bmode_frame(bmode_path, frame)
    frame = int(np.clip(frame, 0, n - 1))
    mline = select_mline(bmode_u8, coords, n_samples=n_samples, title=title)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    save_mline(npz_path, mline)
    draw_mline_on_bmode(bmode_u8, coords, mline, npz_path.replace(".npz", ".png"), title=title)
    print(f"  [M-line] drawn ({mline.length * 1e3:.1f} mm) -> {os.path.basename(npz_path)} (+ .png)")


def _write_hdf5(path, m, r0, bands, base, store):
    """Space-time maps behind the montage: /<quantity>/<band> = (n_t, n_r), + axes, r0, speeds."""
    import h5py
    smooth = "+".join(s.name for s in _smoothing_steps(base)) or "none"
    with h5py.File(path, "w") as f:
        f.attrs["measurement"] = int(m)
        f.attrs["r0_m"] = float(r0)
        f.attrs["estimator"] = base.estimator
        f.attrs["mode"] = base.mode
        f.attrs["smoothing"] = smooth
        f.attrs["directional"] = bool(base.directional)
        f.attrs["bands_hz"] = np.array(bands, float) if bands is not None else np.zeros((0, 2))
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        f.create_dataset("r_m", data=next(iter(store.values()))[0].st.r.astype(np.float64))
        for (quantity, tag), (res, oc, band) in store.items():
            g = f.require_group(quantity)
            if "t_s" not in g:
                g.create_dataset("t_s", data=res.st.t.astype(np.float64))
            d = g.create_dataset(tag, data=res.st.data.astype(np.float32))
            s = res.speed
            d.attrs["origin_coherence"] = float(oc)
            d.attrs["c_pos"] = float(s.c_pos); d.attrs["c_neg"] = float(s.c_neg)
            d.attrs["c_mean"] = float(s.c_mean); d.attrs["quality"] = float(s.quality)
            if band is not None:
                d.attrs["f_lo_hz"] = float(band[0]); d.attrs["f_hi_hz"] = float(band[1])


def _process_measurement(cfg, base_cfg, m, bands, outdir, bmode_path, bmode_frame):
    from swp.viz import runconfig as rc
    from swp.viz.pipeline import run_pipeline
    from swp.viz.metrics import origin_coherence
    from swp.viz.viz import spacetime_montage

    iq_path = rc.hdf5_path(cfg, m)
    if not os.path.exists(iq_path):
        print(f"  meas{m}: IQ not found ({os.path.basename(iq_path)}) - skipping")
        return
    # Only a manual (anatomical) M-line is drawn/loaded interactively; the horizontal
    # phantom line is generated from the axes + push focal depth, so no B-mode/drawing.
    if cfg["mline"]["type"] == "manual":
        npz_path = rc.mline_npz_path(cfg, m)
        _ensure_mline(npz_path, bmode_path, bmode_frame,
                      title=f"M-line for meas{m} ({os.path.basename(iq_path)})",
                      n_samples=cfg["mline"].get("n_samples", 250))

    acq = rc.load_acq(cfg, m)
    # rebuild base with this acquisition so a 'stored' focus can anchor r0 on the saved push x
    base = rc.build_pipeline_config(cfg, acq=acq)
    mline = rc.build_mline(cfg, acq, m)
    focus = rc.build_focus(cfg, acq)
    smooth = _smoothing_steps(base)
    cols = bands if bands is not None else [None]

    results, titles, store = [], [], {}
    for quantity in ("displacement", "velocity"):
        for band in cols:
            ff = _band_recipe(smooth, band)
            raw = run_pipeline(acq, mline, replace(base, quantity=quantity, field_filters=ff,
                                                   directional=False), focus=focus)
            oc = origin_coherence(raw.st, raw.r0)
            res = run_pipeline(acq, mline, replace(base, quantity=quantity, field_filters=ff,
                                                   directional=base.directional), focus=focus)
            results.append(res)
            tag = f"bp{int(band[0])}-{int(band[1])}" if band is not None else "recipe"
            titles.append(f"{quantity[:4]}  {tag}\noc={oc:.3f}")
            store[(quantity, tag)] = (res, oc, band)
    r0 = results[0].r0

    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"swp_meas{m}.png")
    spacetime_montage(results, png, ncols=len(cols),
                      suptitle=f"meas{m} [{acq.source}]  top: displacement, bottom: velocity; "
                               f"columns = bands; dashed r0 (={r0*1e3:.1f} mm)",
                      panel_titles=titles)
    h5 = os.path.join(outdir, f"swp_meas{m}.hdf5")
    _write_hdf5(h5, m, r0, bands, base, store)
    print(f"  meas{m}: wrote {os.path.basename(png)} + {os.path.basename(h5)} "
          f"(r0={r0*1e3:.1f} mm, push_x={acq.push_x})")


def _apply_phantom(cfg):
    """Adapt an active config for a phantom measurement (in place).

    Phantoms have no anatomical M-line and no natural (buffer-4) shear wave: the ARF
    push produces a symmetric wave about the focal point, so the M-line is simply a
    horizontal line through the push focal depth (``horizontal_push``), needing no
    interactive drawing and no buffer-5 B-mode. The processing recipe (bands, filters,
    stored-focus r0, outward-directional) is left exactly as the active config.
    """
    cfg["mline"] = {**cfg["mline"], "type": "horizontal_push"}
    cfg["data"] = {**cfg["data"], "bmode": None}      # no buffer-5 frame needed
    return cfg


def stage_viz(folder, config, meas=None, phantom=False):
    from swp.viz import runconfig as rc

    cfg = rc.load_config(config)
    if phantom:
        _apply_phantom(cfg)
    output_dir = os.path.join(folder, "output")
    cfg["data"]["root"] = output_dir

    base = rc.build_pipeline_config(cfg)            # provisional (per-meas rebuilt with acq)
    bands = rc.bands(cfg)

    if meas is not None:
        meas_list = [meas]
    else:
        meas_list = rc.discover_measurements(cfg) or rc.measurements(cfg)
    if not meas_list:
        raise SystemExit(f"no measurements found under {rc.hdf5_path(cfg, 0)!r}")

    outdir = rc.outdir(cfg, output_dir)
    bmode_tmpl = cfg["data"].get("bmode")
    print(f"swp viz [{os.path.basename(config)}]: {len(meas_list)} measurement(s) from {output_dir}")
    print(f"  bands: {bands}  |  mode: {base.mode}  |  smoothing: "
          f"{'+'.join(s.name for s in _smoothing_steps(base)) or 'none'}")
    for m in meas_list:
        bmode_path = os.path.join(output_dir, bmode_tmpl.format(m=m)) if bmode_tmpl else rc.hdf5_path(cfg, m)
        # active: buffer-5 has one frame per push -> frame m. passive: single cine -> middle frame.
        bmode_frame = m if (bmode_tmpl and "buffer5" in bmode_tmpl) else -1
        if bmode_frame == -1:
            bmode_frame = _middle_frame(bmode_path)
        _process_measurement(cfg, base, m, bands, outdir, bmode_path, bmode_frame)
    print(f"done -> {outdir}")


def _middle_frame(bmode_path):
    if not os.path.exists(bmode_path):
        return 0
    import h5py
    with h5py.File(bmode_path, "r") as f:
        n = f["tracks/track_0/data/beamformed_data/values"].shape[0]
    return n // 2


# ======================================================================================
# CLI
# ======================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    pc = sub.add_parser("convert", help="Stage 1: .mat -> converted zea RF .hdf5")
    pc.add_argument("folder")
    pc.add_argument("--overwrite", action="store_true")

    pb = sub.add_parser("beamform", help="Stage 2: RF -> IQ + GIFs + SW IQ (with push location)")
    pb.add_argument("folder")
    pb.add_argument("--no-gifs", action="store_true")
    pb.add_argument("--no-converted", action="store_true")
    pb.add_argument("--overwrite", action="store_true")

    pv = sub.add_parser("viz", help="Stage 3: IQ -> shear-wave space-time plots (active or passive)")
    pv.add_argument("folder")
    pv.add_argument("--config", required=True, help="configs/active.yaml or configs/passive.yaml")
    pv.add_argument("--meas", type=int, default=None, help="only this measurement (default: all)")
    pv.add_argument("--phantom", action="store_true",
                    help="phantom measurement: horizontal M-line through the push focal point "
                         "(no anatomical M-line / buffer-5 B-mode / buffer-4 shear wave)")

    pp = sub.add_parser("passive", help="Passive burst-window workflow: general M-line -> burst "
                                        "detection -> per-window M-line -> 100 ms space-time montage")
    pp.add_argument("folder")
    pp.add_argument("--config", default="configs/passive.yaml")
    pp.add_argument("--window-ms", type=float, default=100.0)
    pp.add_argument("--max-events", type=int, default=4)
    pp.add_argument("--pad-ms", type=float, default=20.0)
    pp.add_argument("--overview-stride", type=int, default=2)
    pp.add_argument("--redraw", action="store_true", help="re-draw every M-line even if saved")

    pa = sub.add_parser("all", help="Stages 1->2->3")
    pa.add_argument("folder")
    pa.add_argument("--config", required=True)
    pa.add_argument("--no-gifs", action="store_true")
    pa.add_argument("--overwrite", action="store_true")
    pa.add_argument("--meas", type=int, default=None)
    pa.add_argument("--phantom", action="store_true",
                    help="phantom measurement (see 'viz --phantom')")

    a = p.parse_args()
    if a.stage == "convert":
        stage_convert(a.folder, overwrite=a.overwrite)
    elif a.stage == "beamform":
        stage_beamform(a.folder, make_gifs=not a.no_gifs, overwrite=a.overwrite,
                       save_converted=not a.no_converted)
    elif a.stage == "viz":
        stage_viz(a.folder, a.config, meas=a.meas, phantom=a.phantom)
    elif a.stage == "passive":
        from swp.passive import process_passive
        process_passive(a.folder, a.config, window_ms=a.window_ms, max_events=a.max_events,
                        pad_ms=a.pad_ms, overview_stride=a.overview_stride, redraw=a.redraw)
    elif a.stage == "all":
        stage_beamform(a.folder, make_gifs=not a.no_gifs, overwrite=a.overwrite)
        stage_viz(a.folder, a.config, meas=a.meas, phantom=a.phantom)


if __name__ == "__main__":
    main()
