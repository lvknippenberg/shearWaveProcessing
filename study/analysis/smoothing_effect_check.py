"""Why do temporal smoothing and M-line averaging change the passive panels so little?

Suspicion raised on report part 2: the "temporal" and "mline" families look almost identical to the
default. This checks (1) that the options are actually applied, and (2) how large their effect is
expected to be given the signal, using the cached 15-window panels (``passive_atlas_all15.py``,
line source b1, recipe set v2).

1. Per recipe: relative RMS difference to the default panel, median over windows. Nonzero and
   monotonic in the filter strength = the option is applied.
2. The same filters applied to the default's own input would remove little if the signal has no
   energy where they act. Reported: the fraction of the default panel's energy (after the 15-150 Hz
   band-pass) above 50 / 100 Hz, and the amplitude response of the moving means at the signal's
   f50 / f90.
3. Across-line structure: the M-line offsets average perpendicular to the line; they only change
   the panel if the field varies across the line on the offset scale. Reported: the correlation
   between the 1-line and the 15 x 0.8 mm panel.

    python study/analysis/smoothing_effect_check.py
-> study/logs/smoothing_effect_check.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))

import passive_atlas_all15 as A15                               # noqa: E402

DEFAULT = {"temporal": "moving mean 3 (default)", "mline": "5 lines x 0.5 mm (default)",
           "spatial": "Gaussian 0.6 x 1.2 mm (default)"}
PRF = 926.0


def mm_response(n, f):
    x = np.pi * f / PRF
    return abs(np.sin(n * x) / (n * np.sin(x))) if f > 0 else 1.0


def main():
    from swp.provenance import stamp_text
    wins = A15._load("b1", "v2")
    rows = []
    for fam in ("temporal", "mline", "spatial"):
        labels = [lab for lab, _ in A15.A.SETS["v2"]()[fam]]
        for lab in labels:
            rel, corr = [], []
            for W in wins:
                d0, _ = W["panels"][f"{fam}|{DEFAULT[fam]}"]
                d, _ = W["panels"].get(f"{fam}|{lab}", (None, None))
                if d is None or d.shape != d0.shape:
                    continue
                rel.append(np.sqrt(np.mean((d - d0) ** 2) / np.mean(d0 ** 2)))
                corr.append(np.corrcoef(d.ravel(), d0.ravel())[0, 1])
            rows.append(dict(family=fam, recipe=lab, rel_rms_diff=float(np.median(rel)),
                             rel_rms_diff_max=float(np.max(rel)), corr_to_default=float(np.median(corr)), n=len(rel)))
            print(f"{fam:9s} {lab:36s} rel. RMS difference to default: median {rows[-1]['rel_rms_diff']:.3f} "
                  f"(max {rows[-1]['rel_rms_diff_max']:.3f}), correlation {rows[-1]['corr_to_default']:.3f}")
    # spectrum of the default panels (slow time), after band-pass + mean 3
    f50s, f90s, hi50, hi100 = [], [], [], []
    for W in wins:
        d, t = W["panels"]["temporal|none"]
        P = (np.abs(np.fft.rfft(d - d.mean(0), axis=0)) ** 2).sum(axis=1)
        f = np.fft.rfftfreq(d.shape[0], 1 / PRF)
        c = np.cumsum(P) / P.sum()
        f50s.append(f[np.searchsorted(c, 0.5)]); f90s.append(f[np.searchsorted(c, 0.9)])
        hi50.append(P[f > 50].sum() / P.sum()); hi100.append(P[f > 100].sum() / P.sum())
    f50, f90 = np.median(f50s), np.median(f90s)
    print(f"\nslow-time spectrum of the velocity 15-150 Hz panels without temporal smoothing (median over "
          f"{len(wins)} windows): f50 {f50:.0f} Hz, f90 {f90:.0f} Hz; energy above 50 Hz {np.median(hi50):.1%}, "
          f"above 100 Hz {np.median(hi100):.1%}")
    for n in (3, 5, 7, 9):
        print(f"  moving mean {n}: amplitude at f50 {mm_response(n, f50):.2f}, at f90 {mm_response(n, f90):.2f}, "
              f"at 100 Hz {mm_response(n, 100):.2f}, first zero {PRF / n:.0f} Hz")
    out = _REPO / "study/logs/smoothing_effect_check.csv"
    with open(out, "w", newline="") as fh:
        fh.write(stamp_text(config=dict(source="atlas15_cache/b1/v2", f50=float(f50), f90=float(f90))))
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
