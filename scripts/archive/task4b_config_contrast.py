"""Task 4b -- is there a push effect at all? Push-minus-control contrast per in-vivo configuration.

Pools every push of each in-vivo acquisition and compares the ARF space-time against its own
no-push control (split pre-push reference, scored over the same window). Includes the older
2026-08-04 free-running 40 V acquisition as the reference distribution, so the two new
2026-08-18 R-peak-triggered configurations can be judged against what we already had.

A configuration that images a real ARF wave should sit clearly above zero. Zero-centred means the
recipe is picking up cardiac motion equally with and without the push.

    python scripts/archive/task4b_config_contrast.py --folders <f1> <f2> ... --labels <l1> <l2> ...
                                             --out <fig.png>
"""
from __future__ import annotations

import argparse
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))           # archived: siblings + scripts/
for _p in (_HERE, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swe_lib as L                                              # noqa: E402
from task4_invivo_compare import REF_SPLIT, SCORE_KW             # noqa: E402


def contrasts(folder, recipe, quantity):
    out = []
    for m in range(L.n_pushes(folder)):
        try:
            stp, r0 = L.st_for(folder, m, recipe, quantity=quantity, phantom=False)
            stn, _ = L.st_for(folder, m, recipe, quantity=quantity, phantom=False,
                              nopush=True, ref_split=REF_SPLIT)
        except Exception as exc:                                  # noqa: BLE001
            print(f"    m{m}: {exc}")
            continue
        n = stn.data.shape[0]
        ocp, symp = L.scores(L.truncate(stp, n), r0, **SCORE_KW)
        ocn, symn = L.scores(stn, r0, **SCORE_KW)
        out.append((ocp - ocn, symp - symn, ocp, ocn))
        L._ACQ_CACHE.pop((folder, m, False), None)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantity", default="displacement")
    a = ap.parse_args()
    assert len(a.folders) == len(a.labels)

    res = {}
    for folder, lbl in zip(a.folders, a.labels):
        print(f"=== {lbl}", flush=True)
        res[lbl] = contrasts(folder, L.REC_INVIVO, a.quantity)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, i, name in ((axes[0], 0, "origin coherence"), (axes[1], 1, "mirror symmetry")):
        data = [res[l][:, i] for l in a.labels if len(res[l])]
        labs = [l for l in a.labels if len(res[l])]
        ax.boxplot(data, tick_labels=labs, showmeans=True)
        for k, d in enumerate(data, start=1):
            ax.plot(np.full(len(d), k) + np.random.uniform(-0.09, 0.09, len(d)), d, ".",
                    color="0.4", ms=5, alpha=0.7)
        ax.axhline(0, color="firebrick", lw=1.2, ls="--")
        ax.set_title(f"push - no-push contrast, {name}")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8, rotation=12)
    axes[0].set_ylabel("push minus its own control")
    fig.suptitle(f"In-vivo: does the ARF push add anything above cardiac motion?  ({a.quantity}, "
                 f"{L.REC_INVIVO['tag']})\nEach dot is one push, scored against its own split "
                 f"pre-push reference over the same window", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)

    print(f"\n{'configuration':<42} {'n':>3} {'median dOC':>11} {'frac>0':>7} {'median dSYM':>12}")
    for lbl in a.labels:
        d = res[lbl]
        if not len(d):
            continue
        print(f"{lbl:<42} {len(d):3d} {np.median(d[:,0]):11.3f} "
              f"{float(np.mean(d[:,0] > 0)):7.2f} {np.median(d[:,1]):12.3f}")


if __name__ == "__main__":
    main()
