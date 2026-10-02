"""Label-free PLAX/PSAX sorting of the SW buffer-3 loops: four independent voters + consensus.

Voters (no human labels anywhere):
  ep      EchoPrime view classifier, softmax averaged over the loop (the first pass).
  clu_f   within-subject 2-cluster split of EchoPrime frame features. Each loop is described by
          the mean AND the temporal std of its per-frame features (anatomy + how it moves over
          the cycle), centred on the subject mean so patient identity drops out.
  clu_v   the same on EchoPrime video-encoder embeddings (16-frame clips, motion seen directly).
  self    logistic regression trained on the CONFIDENT EchoPrime loops of all OTHER subjects
          (leave-subject-out pseudo-labels) on the subject-centred [frame mean|std|video]
          features: a binary head adapted to our image domain, without labels.
A cluster is named by the summed EchoPrime log-odds of its members, so one off loop cannot flip it.

consensus = all four agree -> auto label; otherwise 'review'. EchoPrime p_Apical > 0.5 or
p_Other > 0.5 -> 'review' regardless (binary voters cannot say "neither").

Usage: python label_free_sort.py <features.npz> <out.csv>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.linear_model import LogisticRegression

PLAX = ["Parasternal_Long", "Doppler_Parasternal_Long"]
PSAX = ["Parasternal_Short", "Doppler_Parasternal_Short"]
APICAL = ["A2C", "A3C", "A4C", "A5C", "Apical_Doppler"]
OTHER = ["SSN", "Subcostal"]
EPS = 1e-4


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def load(path):
    z = np.load(path)
    off, classes = z["frame_offsets"], list(z["classes"])
    ff = l2(z["frame_feat"].astype(np.float32))
    fp = z["frame_prob"]
    n = len(off) - 1
    fmean = np.stack([ff[off[i]:off[i + 1]].mean(0) for i in range(n)])
    fstd = np.stack([ff[off[i]:off[i + 1]].std(0) for i in range(n)])
    prob = np.stack([fp[off[i]:off[i + 1]].mean(0) for i in range(n)])
    grp = lambda names: prob[:, [classes.index(c) for c in names]].sum(1)
    d = pd.DataFrame({"subject": z["subject"], "folder": z["folder"],
                      "p_plax": grp(PLAX), "p_psax": grp(PSAX), "p_apical": grp(APICAL), "p_other": grp(OTHER)})
    d["ep_logodds"] = np.log(d.p_plax + EPS) - np.log(d.p_psax + EPS)
    video = l2(z["video_emb"].mean(1))
    return d, fmean, fstd, video


def centre(x, subj):
    out = np.empty_like(x)
    for s in np.unique(subj):
        m = subj == s
        out[m] = x[m] - x[m].mean(0)
    return out


def standardise(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def two_clusters(x, logodds):
    """2-way split of one subject's loops; returns +1 (PLAX) / -1 (PSAX) per loop and a margin."""
    xn = l2(x)
    lab = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(xn)
    if min(np.bincount(lab, minlength=2)) < 2:            # a singleton outlier split -> k-means instead
        lab = KMeans(2, n_init=20, random_state=0).fit_predict(xn)
    # every subject has both views (protocol), so the cluster with more PLAX evidence is PLAX
    score = np.array([logodds[lab == k].sum() for k in (0, 1)])
    name = np.where(np.arange(2) == np.argmax(score), 1, -1)
    c = np.stack([xn[lab == k].mean(0) for k in (0, 1)])
    sim = xn @ l2(c).T                                     # cosine to each centroid
    margin = np.abs(sim[:, 0] - sim[:, 1])
    return name[lab], margin


def main():
    feats, out = sys.argv[1], sys.argv[2]
    d, fmean, fstd, video = load(feats)
    subj = d.subject.values
    Xf = standardise(np.hstack([centre(fmean, subj), centre(fstd, subj)]))
    Xv = standardise(centre(video, subj))

    d["ep"] = np.where(d.ep_logodds > 0, "PLAX", "PSAX")
    for col, X in (("clu_f", Xf), ("clu_v", Xv)):
        lab = np.empty(len(d), dtype=object)
        mar = np.empty(len(d))
        for s in np.unique(subj):
            m = subj == s
            sign, mg = two_clusters(X[m], d.ep_logodds.values[m])
            lab[m] = np.where(sign > 0, "PLAX", "PSAX")
            mar[m] = mg
        d[col], d[col + "_margin"] = lab, mar

    # self-training on confident pseudo-labels, leave-subject-out
    conf = (np.maximum(d.p_plax, d.p_psax) >= 0.95).values
    y = (d.ep_logodds > 0).astype(int).values
    Xs = np.hstack([Xf, Xv])
    p_self = np.empty(len(d))
    for s in np.unique(subj):
        tr = conf & (subj != s)
        clf = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced").fit(Xs[tr], y[tr])
        p_self[subj == s] = clf.predict_proba(Xs[subj == s])[:, 1]
    d["p_self_plax"] = p_self
    d["self"] = np.where(p_self > 0.5, "PLAX", "PSAX")

    votes = d[["ep", "clu_f", "clu_v", "self"]]
    d["n_plax_votes"] = (votes == "PLAX").sum(1)
    unanimous = d.n_plax_votes.isin([0, 4])
    odd = (d.p_apical > 0.5) | (d.p_other > 0.5)
    d["consensus"] = np.where(unanimous & ~odd, votes.iloc[:, 0], "review")
    d["review_reason"] = np.where(odd, np.where(d.p_apical > 0.5, "apical?", "other?"),
                                  np.where(unanimous, "", "split vote"))
    d["time"] = d.folder.str.split("_").str[-1]
    d = d.sort_values(["subject", "time"]).drop(columns="time")
    d.to_csv(out, index=False)

    n = len(d)
    print(f"{n} loops, {d.subject.nunique()} subjects")
    for a in ("clu_f", "clu_v", "self"):
        print(f"  agreement ep vs {a:<6} {np.mean(d.ep == d[a]):.3f}")
    print(f"  clu_f vs clu_v {np.mean(d.clu_f == d.clu_v):.3f}   self vs clu_v {np.mean(d.self == d.clu_v):.3f}")
    print("consensus:", d.consensus.value_counts().to_dict())
    print("review reasons:", d.review_reason.value_counts().to_dict())

    def switches(v):
        v = [x for x in v if x != "review"]
        return sum(a != b for a, b in zip(v, v[1:]))
    sw = d.groupby("subject", sort=True)
    print("label switches per subject in acquisition order (protocol = 1):")
    for col in ("ep", "clu_f", "clu_v", "self", "consensus"):
        s = sw[col].apply(lambda v: switches(list(v)))
        print(f"  {col:<9} mean {s.mean():.2f}  subjects with >1: {(s > 1).sum()}")
    print("review per subject:", d[d.consensus == "review"].groupby("subject").size().to_dict())


if __name__ == "__main__":
    main()
