"""Assess the R-peak record of every acquisition folder and flag the ones that cannot be trusted.

    python study/analysis/ecg_check_study.py
    python study/analysis/ecg_check_study.py --root Z:/raw_data --csv study/logs/ecg_check.csv

Trigger record only - see ``swp.acquisition.rrcheck`` and the "Do not re-detect R-peaks from the
``Signal`` trace" section of ``docs/ecg_timing.md`` for why the logged waveform is not used.

The same assessment runs automatically inside the passive workflow
(``swp.passive.check_ecg_quality``); this script is the batch view over a whole study tree.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np

from swp.acquisition.rrcheck import assess_rr

DEFAULT_ROOT = r"Z:/raw_data"


def acquisition_folders(root, beamformed_only=False):
    out = []
    for subj in sorted(Path(root).glob("C*")):
        if not subj.is_dir():
            continue
        for f in sorted(subj.iterdir()):
            if not (f.is_dir() and (f / "AcquisitionParametersAndECG.mat").is_file()):
                continue
            if beamformed_only and not (f / "output").is_dir():
                continue
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--csv", default=str(_REPO / "study" / "logs" / "ecg_check.csv"))
    ap.add_argument("--beamformed-only", action="store_true")
    a = ap.parse_args()

    folders = acquisition_folders(a.root, a.beamformed_only)
    print(f"{len(folders)} acquisition folder(s) under {a.root}\n")
    print(f"{'subject':<12}{'status':<11}{'quality':<17}{'RR':>6}{'HR':>5}{'SD':>6}{'CV':>7}"
          f"{'RMSSD':>7}{'n':>4}{'corr':>5}{'gate us':>9}")
    print("-" * 94)

    rows = []
    for f in folders:
        chk = assess_rr(str(f))
        chk.folder = f"{f.parent.name}/{f.name}"
        rows.append(chk)
        print(f"{f.parent.name:<12}{chk.status:<11}{chk.quality:<17}"
              f"{chk.rr_median_ms:>6.0f}{chk.hr_bpm:>5.0f}{chk.rr_sd_ms:>6.0f}"
              f"{chk.rr_cv:>7.3f}{chk.rmssd_ms:>7.0f}{chk.n_rr:>4d}{chk.n_corrected:>5d}"
              f"{chk.gating_error_us:>9.1f}")

    if rows:
        os.makedirs(os.path.dirname(a.csv), exist_ok=True)
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].as_row().keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r.as_row())
        print(f"\n-> {a.csv}")

    print("\n=== summary ===")
    print("  status : " + str(collections.Counter(r.status for r in rows)))
    print("  quality: " + str(collections.Counter(r.quality for r in rows)))
    trust = [r for r in rows if r.trustworthy]
    print(f"\n  cardiac phases trustworthy in {len(trust)}/{len(rows)} folders")
    gate = np.array([r.gating_error_us for r in rows if np.isfinite(r.gating_error_us)])
    if gate.size:
        print(f"  passive block vs closest R-peak: max |error| {np.abs(gate).max():.1f} us "
              f"over {gate.size} folders - the gating itself is exact")
    bad = [r for r in rows if not r.trustworthy]
    if bad:
        print(f"\n  folders whose cardiac phases must NOT be trusted ({len(bad)}):")
        for r in bad:
            # show the actual warning, not a "recoverable" note that happens to come first
            warn = next((m for m in r.messages if m.startswith("WARNING")), "")
            print(f"    {r.folder.split('/')[0]:<12} {r.status:<10} {r.quality:<16} "
                  + warn[len("WARNING: "):][:76])


if __name__ == "__main__":
    sys.exit(main())
