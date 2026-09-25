"""Calibrate the image-based buffer-3 head estimators against the EXACT trigger count.

The trigger log holds one trigger per buffer-3 frame of the live loop. When the log still contains
the START of that run (143/341 ready folders on 2026-09-25; in the rest the circular 1500/2000-entry
log has overwritten it), the head is known without images: with L triggers in the run, the loop
wrote slots 1..N cyclically and the last trigger belongs to the frame being acquired when the loop
was left - normally aborted mid-frame and never transferred - so the oldest stored slot is
``c = (L - 1) mod N`` (0-based). If the loop happened to stop between frames, the last frame WAS
transferred and the head is ``c + 1``; the two differ only in whether slot c is the oldest or the
newest frame, so one local continuity comparison decides (the break is on the side of slot c with
the lower neighbour correlation).

For every complete-run folder this script records that truth and scores against it:
  iq      cyclic continuity of the beamformed IQ (swp.acquisition.unwrap.continuity_scores)
  b1      buffer-1 similarity with the trigger-to-frame shift FIXED at -1 (mid-frame abort)
  b1free  buffer-1 similarity with the shift free in {-1, 0, +1} (as in the first validation)
Accuracy per method as a function of its confidence margin sets the thresholds used for the
folders whose run start is lost. Read-only.

    python study/analysis/buffer3_unwrap_calibration.py [--limit N]
-> study/logs/buffer3_unwrap_calibration.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from swp.acquisition import unwrap as U                            # noqa: E402
from swp.manual import store as S                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    folders = [f for f in S.find_folders(a.root) if S.ready(S.Paths(f))]
    out = _REPO / "study" / "logs" / "buffer3_unwrap_calibration.csv"
    rows = []
    for k, f in enumerate(folders):
        tc = U.trigger_count_head(f)
        if tc is None:
            continue
        if a.limit and len(rows) >= a.limit:
            break
        t0 = time.perf_counter()
        try:
            outdir = Path(f) / "output"
            _, iq3 = U.buffer_files(outdir)
            _, iq1 = U.buffer_files(outdir, buffer=1)
            A3 = U.anatomy_stack(iq3)
            n = len(A3)
            truth, how, local = U.resolve_trigger_count(tc, A3)
            row = dict(folder=f"{Path(f).parent.name}/{Path(f).name}", n_frames=n, run_length=tc["run_length"],
                       truth=truth, stop=how, local_margin=round(local, 4))
            first, margin = U.best_and_margin(U.continuity_scores(A3), lower_is_better=True)
            row.update(iq_first=first, iq_margin=round(margin, 4), iq_err=U._cyc(first, truth, n))
            try:
                ecg_ok = bool(U._ecg_ok(f))
            except Exception:                                          # noqa: BLE001
                ecg_ok = False
            row["ecg_trustworthy"] = ecg_ok
            if ecg_ok and iq1 is not None:
                sc = U.buffer1_scores(f, A3, iq1)
                if sc is not None:
                    fb, mb = U.best_and_margin(sc[-1], lower_is_better=False)
                    row.update(b1_first=fb, b1_margin=round(mb, 4), b1_err=U._cyc(fb, truth, n))
                    shift = max(sc, key=lambda s: np.nanmax(sc[s]))
                    ff, mf = U.best_and_margin(sc[shift], lower_is_better=False)
                    row.update(b1free_first=ff, b1free_shift=shift, b1free_margin=round(mf, 4),
                               b1free_err=U._cyc(ff, truth, n))
            row["seconds"] = round(time.perf_counter() - t0, 1)
        except Exception:                                              # noqa: BLE001
            print(f"{f}: FAILED\n{traceback.format_exc()}", flush=True)
            continue
        rows.append(row)
        print(f"[{len(rows)}] {row['folder'][:40]:40s} N={n} truth {truth:2d} ({how}, local {local:.2f}) | "
              f"iq {row['iq_first']:2d} (err {row['iq_err']}, m {row['iq_margin']:.2f}) | "
              f"b1 {row.get('b1_first', '-')!s:>2} (err {row.get('b1_err', '-')}, m {row.get('b1_margin', float('nan')):.3f}) | "
              f"b1free {row.get('b1free_first', '-')!s:>2} (err {row.get('b1free_err', '-')})", flush=True)
        keys = list(dict.fromkeys(key for r in rows for key in r))
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
