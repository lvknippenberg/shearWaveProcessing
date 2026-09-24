"""Screen every ARF push in the in-vivo study for a push effect above cardiac motion.

The study (``Z:/raw_data``, 71 processed acquisitions x 24 pushes) was only ever analysed for
passive (valve) waves. Each buffer-2 push is scored here exactly as in
``scripts/invivo_recipe_contrast.py``: push against its own length-matched no-push control
(split pre-push reference), RMS ratio in 2-14 mm from r0 plus the origin-coherence difference.

No hand-drawn active M-lines exist for the study, so the line is proposed automatically
(``scripts/auto_mline.py``: straight fit to the septal ridge around the push focus, re-anchored
on the push depth; validated against 24 hand-drawn lines) and **kept in memory** - nothing is
written into the study folders. A misplaced line lowers both push and control; it cannot create a
push effect, so this screen is conservative (it can miss waves, not invent them).

Also records per acquisition the delivered push settings (elements, cycles, TPC profile-5
voltage, R-peak gating) and per push its time after the R-peak (push k at k / 20 Hz when
``SW.WaitForRpeak``; NaN otherwise).

    python scripts/study_active_screen.py [--root Z:/raw_data] [--jobs 6] [--recipes caenen_dir current]

-> ``study/logs/study_active_screen.csv`` + ``study/montages/study_active_screen.png``
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "swp_gui", "scripts"):
    if os.path.join(_ROOT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _p))

from swp import paths as P                                     # noqa: E402

PUSH_RATE_HZ = 20.0


def folders(root):
    return sorted(os.path.dirname(os.path.dirname(p)) for p in
                  glob.glob(os.path.join(root, "C*", "*", "output", "CombinedData_buffer2_meas0_iq.hdf5")))


def acq_settings(folder):
    import scipy.io as sio
    out = dict(el=np.nan, cycles=np.nan, V=np.nan, rpeak=0)
    try:
        m = sio.loadmat(os.path.join(folder, "AcquisitionParametersAndECG.mat"),
                        squeeze_me=True, struct_as_record=False)
        sw, tpc = m["SW"], m["TPC"]
        out.update(el=int(sw.nb_push_elmts), cycles=int(sw.pushCycle), V=float(tpc[4].hv),
                   rpeak=int(getattr(sw, "WaitForRpeak", 0)))
    except Exception as exc:                                    # noqa: BLE001
        out["error"] = str(exc)[:80]
    return out


def screen_folder(folder, recipes):
    """All pushes of one acquisition -> list of row dicts (runs in a worker process)."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import invivo_recipe_contrast as H
    import auto_mline
    from swp.viz.io import load_acquisition
    from swp.viz.mline.mline import mline_from_points
    from swp.viz.pipeline import _r0_lateral_crossing
    s = acq_settings(folder)
    subject = os.path.basename(os.path.dirname(folder))
    rows = []
    files = glob.glob(os.path.join(folder, "output", "CombinedData_buffer2_meas*_iq.hdf5"))
    for m in sorted(int(f.split("meas")[-1].split("_")[0]) for f in files):
        try:
            acq = load_acquisition(os.path.join(folder, "output", f"CombinedData_buffer2_meas{m}_iq.hdf5"))
            px, pz = float(acq.push_x or 0.0), float(acq.push_z)
            pts = auto_mline.propose(folder, m, px * 1e3, pz * 1e3, order=1, x_span_mm=34.0)
            ml = mline_from_points(pts, 250)
            r0 = _r0_lateral_crossing(ml, px)
        except Exception as exc:                                # noqa: BLE001
            rows.append(dict(subject=subject, folder=os.path.basename(folder), meas=m, error=str(exc)[:80]))
            continue
        for rn in recipes:
            try:
                stp = H.spacetime(acq, ml, r0, H.RECIPES[rn], nopush=False)
                stn = H.spacetime(acq, ml, r0, H.RECIPES[rn], nopush=True)
                ocp, _, ap = H.score(stp, r0)
                ocn, _, an = H.score(stn, r0)
            except Exception as exc:                            # noqa: BLE001
                rows.append(dict(subject=subject, folder=os.path.basename(folder), meas=m,
                                 recipe=rn, error=str(exc)[:80]))
                continue
            rows.append(dict(subject=subject, folder=os.path.basename(folder), meas=m, recipe=rn,
                             t_after_R_ms=(1000.0 * m / PUSH_RATE_HZ) if s["rpeak"] else np.nan,
                             el=s["el"], cycles=s["cycles"], V=s["V"], rpeak=s["rpeak"],
                             amp_push=ap, amp_nopush=an, log2_amp=float(np.log2(ap / an)),
                             oc_push=ocp, oc_nopush=ocn, d_oc=ocp - ocn, error=""))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=P.RAW_DATA)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--recipes", nargs="+", default=["caenen_dir", "current"])
    ap.add_argument("--limit", type=int, default=0, help="first N folders only (testing)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "study", "logs", "study_active_screen.csv"))
    a = ap.parse_args()
    fs = folders(a.root)[: a.limit or None]
    print(f"{len(fs)} acquisitions under {a.root}", flush=True)
    rows, t0 = [], time.time()
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(screen_folder, f, a.recipes): f for f in fs}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as exc:                            # noqa: BLE001
                print(f"  FAILED {futs[fu]}: {exc}", flush=True)
                continue
            rows += r
            ok = [x for x in r if not x.get("error") and x.get("recipe") == a.recipes[0]]
            med = 2 ** np.median([x["log2_amp"] for x in ok]) if ok else float("nan")
            print(f"  [{i}/{len(fs)}] {os.path.basename(futs[fu])[:40]}: {len(ok)} pushes, "
                  f"median push/control {med:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    from swp.provenance import stamp_text
    import invivo_recipe_contrast as H
    keys = ["subject", "folder", "meas", "recipe", "t_after_R_ms", "el", "cycles", "V", "rpeak",
            "amp_push", "amp_nopush", "log2_amp", "oc_push", "oc_nopush", "d_oc", "error"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        f.write(stamp_text(config={"recipes": {k: H.RECIPES[k] for k in a.recipes},
                                   "REF_SPLIT": H.REF_SPLIT, "AMP_BAND": H.AMP_BAND,
                                   "mline": "auto_mline.propose(order=1, x_span_mm=34)"}))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
