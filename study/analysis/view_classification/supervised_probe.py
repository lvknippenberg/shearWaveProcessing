"""Leave-one-subject-out check: binary PLAX/PSAX head on frozen EchoPrime features, trained on the
manual review labels (for sorting future subjects).

Feature sets (all from extract_features.py, subject-centred as in label_free_sort.py):
  frame      mean of the per-frame features over the loop (no temporal information)
  frame+std  plus their temporal std over the cycle (how the features move)
  video      video-encoder embedding (16 frames)
  all        frame+std+video
Usage: python supervised_probe.py <features.npz>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from label_free_sort import centre, load, standardise   # noqa: E402

LOGS = Path(__file__).resolve().parents[2] / "logs" / "view_classification"


def main():
    d, fmean, fstd, video = load(sys.argv[1])
    man = pd.read_csv(LOGS / "sw_views_manual.csv").set_index("folder").label
    d["label"] = d.folder.map(man)
    subj = d.subject.values
    sets = {"frame": [fmean], "frame+std": [fmean, fstd], "video": [video], "all": [fmean, fstd, video]}
    keep = d.label.isin(["PLAX", "PSAX"]).values
    y = (d.label == "PLAX").astype(int).values
    lines = []
    for name, parts in sets.items():
        X = standardise(np.hstack([centre(p, subj) for p in parts]))
        p = np.full(len(d), np.nan)
        for s in np.unique(subj):
            tr, te = keep & (subj != s), subj == s
            clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        pk, yk = p[keep], y[keep]
        wrong = int(((pk > 0.5) != yk).sum())
        sure = np.abs(pk - 0.5) > 0.4                       # p < 0.1 or > 0.9
        lines.append(f"{name:<10} LOSO accuracy {1 - wrong / keep.sum():.3f} ({wrong} wrong of {keep.sum()}); "
                     f"p<0.1|>0.9: {sure.mean():.0%} of loops, {int(((pk[sure] > 0.5) != yk[sure]).sum())} wrong")
        if name == "all":
            d["p_plax_sup"] = p
    txt = "\n".join(lines)
    print(txt)
    with open(LOGS / "voter_scores.txt", "a") as f:
        f.write("\nsupervised head on frozen EchoPrime features, leave-one-subject-out on the manual labels:\n"
                + txt + "\n")
    bad = d[keep & ((d.p_plax_sup > 0.5) != (d.label == "PLAX"))]
    print(bad[["subject", "folder", "label", "p_plax_sup"]].to_string(index=False))


if __name__ == "__main__":
    main()
