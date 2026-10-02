"""Four voters per loop, their consensus and the review proposal.

Voters (all on the cached EchoPrime features, ``swp.views.echoprime``):
  ep     EchoPrime's own call: PLAX vs PSAX log-odds of the frame softmax averaged over the loop.
  clu_f  per-subject 2-cluster split of [mean | temporal std] of the frame features (anatomy and how
         it changes over the cycle), centred on the subject mean so patient identity drops out.
  clu_v  the same on the video-encoder embeddings.
  sup    binary logistic-regression head on [frame mean | std | video], subject-centred, trained on
         the reviewed labels (``train_head``; stored in ``view_head.npz``).
A cluster is named by the summed EchoPrime log-odds of its members (every subject has both views).

Consensus: all four agree and EchoPrime sees no apical / other view (p < 0.5) -> safe auto label;
else flagged for review. Proposal (what the review window pre-fills): consensus, else the majority,
2-2 ties by the subject's protocol block fit (a PLAX block, then a PSAX block); EchoPrime apical ->
Apical. Every subject must be classified as a whole (centring and clustering are per subject).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import acq_time
from .echoprime import COARSE_VIEWS, GROUPS

EPS = 1e-4
VOTERS = ("ep", "clu_f", "clu_v", "sup")
HEAD_C = 0.05                                    # LR regularisation (study value, not tuned)


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def describe(feats: dict) -> dict:
    """Per-loop descriptors from the cached per-frame / per-clip features."""
    ff = _l2(feats["frame_feat"].astype(np.float32))
    prob = feats["frame_prob"].mean(0)
    grp = {k: float(sum(prob[COARSE_VIEWS.index(c)] for c in v)) for k, v in GROUPS.items()}
    return {"fmean": ff.mean(0), "fstd": ff.std(0), "video": _l2(feats["video_emb"].mean(0)),
            "p_plax": grp["PLAX"], "p_psax": grp["PSAX"], "p_apical": grp["Apical"], "p_other": grp["Other"]}


def table(items) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """items: iterable of (subject, folder, feats). Returns the loop table (sorted by subject and
    acquisition time) and the frame block [fmean|fstd] and video block, NOT yet centred."""
    rows, F, V = [], [], []
    for s, f, feats in sorted(items, key=lambda t: (t[0], acq_time(t[1]))):
        d = describe(feats)
        rows.append({"subject": s, "folder": f, **{k: d[k] for k in ("p_plax", "p_psax", "p_apical", "p_other")}})
        F.append(np.concatenate([d["fmean"], d["fstd"]]))
        V.append(d["video"])
    df = pd.DataFrame(rows)
    df["ep_logodds"] = np.log(df.p_plax + EPS) - np.log(df.p_psax + EPS)
    return df, np.stack(F), np.stack(V)


def centre(x: np.ndarray, subj: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for s in np.unique(subj):
        m = subj == s
        out[m] = x[m] - x[m].mean(0)
    return out


# ---------------------------------------------------------------------------------- the binary head
def fit_head(F, V, subj, y) -> dict:
    """Fit on subject-centred features; ``y`` = 1 PLAX / 0 PSAX / -1 excluded (Unclear, Apical, none).
    Standardisation statistics come from the training loops and are stored with the coefficients."""
    from sklearn.linear_model import LogisticRegression
    Fc, Vc = centre(F, subj), centre(V, subj)
    m = y >= 0
    head = {"mu_f": Fc[m].mean(0), "sd_f": Fc[m].std(0) + 1e-6, "mu_v": Vc[m].mean(0), "sd_v": Vc[m].std(0) + 1e-6}
    X = np.hstack([(Fc - head["mu_f"]) / head["sd_f"], (Vc - head["mu_v"]) / head["sd_v"]])
    clf = LogisticRegression(C=HEAD_C, max_iter=3000, class_weight="balanced").fit(X[m], y[m])
    head.update(coef=clf.coef_[0], intercept=np.array(clf.intercept_[0]), n_train=np.array(int(m.sum())),
                n_subjects=np.array(len(np.unique(subj[m]))), fitted=np.array(dt.date.today().isoformat()))
    return head


def save_head(head: dict, path):
    np.savez(path, **head)


def load_head(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _standardised(F, V, subj, head):
    Fc, Vc = centre(F, subj), centre(V, subj)
    return (Fc - head["mu_f"]) / head["sd_f"], (Vc - head["mu_v"]) / head["sd_v"]


# -------------------------------------------------------------------------------------- the voters
def two_clusters(x: np.ndarray, logodds: np.ndarray) -> np.ndarray:
    """2-way split of one subject's loops; +1 = PLAX cluster, -1 = PSAX cluster."""
    from sklearn.cluster import AgglomerativeClustering, KMeans
    xn = _l2(x)
    if len(xn) < 3:
        return np.where(logodds > 0, 1, -1)
    lab = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(xn)
    if min(np.bincount(lab, minlength=2)) < 2:            # singleton outlier split -> k-means instead
        lab = KMeans(2, n_init=20, random_state=0).fit_predict(xn)
    score = np.array([logodds[lab == k].sum() for k in (0, 1)])
    return np.where(np.arange(2) == np.argmax(score), 1, -1)[lab]


def block_fit(score) -> np.ndarray:
    """Protocol prior: one PLAX block, then one PSAX block (``score`` > 0 leans PLAX)."""
    s = np.asarray(score, dtype=float)
    k = int(np.argmax([s[:j].sum() - s[j:].sum() for j in range(len(s) + 1)]))
    return np.array(["PLAX"] * k + ["PSAX"] * (len(s) - k))


def vote(df: pd.DataFrame, F: np.ndarray, V: np.ndarray, head: dict) -> pd.DataFrame:
    """Add the four votes, consensus, review flag and proposal to the loop table from ``table``."""
    d = df.copy()
    subj = d.subject.to_numpy()
    Xf, Xv = _standardised(F, V, subj, head)
    d["ep"] = np.where(d.ep_logodds > 0, "PLAX", "PSAX")
    for col, X in (("clu_f", Xf), ("clu_v", Xv)):
        lab = np.empty(len(d), dtype=object)
        for s in np.unique(subj):
            m = subj == s
            lab[m] = np.where(two_clusters(X[m], d.ep_logodds.to_numpy()[m]) > 0, "PLAX", "PSAX")
        d[col] = lab
    z = np.hstack([Xf, Xv]) @ head["coef"] + float(head["intercept"])
    d["p_sup_plax"] = 1 / (1 + np.exp(-z))
    d["sup"] = np.where(d.p_sup_plax > 0.5, "PLAX", "PSAX")

    votes = d[list(VOTERS)]
    n_plax = (votes == "PLAX").sum(axis=1).to_numpy()
    odd = ((d.p_apical > 0.5) | (d.p_other > 0.5)).to_numpy()
    unanimous = np.isin(n_plax, (0, len(VOTERS)))
    d["needs_review"] = ~unanimous | odd
    d["reason"] = np.where(odd, np.where(d.p_apical > 0.5, "apical?", "other?"), np.where(unanimous, "", "split vote"))
    block = np.empty(len(d), dtype=object)
    for s in np.unique(subj):
        m = subj == s
        block[m] = block_fit((n_plax[m] - len(VOTERS) / 2) / (len(VOTERS) / 2))
    major = np.where(n_plax > 2, "PLAX", np.where(n_plax < 2, "PSAX", block))
    d["proposed"] = np.where(d.p_apical > 0.5, "Apical", major)
    d["votes"] = votes.apply(lambda r: " ".join("L" if x == "PLAX" else "S" for x in r), axis=1)
    return d


def evaluate(df, F, V, labels: pd.Series) -> str:
    """Leave-one-subject-out check: the head is refitted without each subject, then the full voter
    set runs on it. ``labels``: folder -> label. Returns a text report."""
    y_all = df.folder.map(labels)
    out = []
    for s in df.subject.unique():
        tr = (df.subject != s).to_numpy()
        y = np.where(y_all == "PLAX", 1, np.where(y_all == "PSAX", 0, -1))
        y[~tr] = -1
        head = fit_head(F[tr], V[tr], df.subject.to_numpy()[tr], y[tr])
        m = ~tr
        out.append(vote(df[m].reset_index(drop=True), F[m], V[m], head))
    r = pd.concat(out, ignore_index=True)
    r["label"] = r.folder.map(labels)
    b = r[r.label.isin(["PLAX", "PSAX"])]
    lines = [f"leave-one-subject-out on {len(b)} PLAX/PSAX-labelled loops of {b.subject.nunique()} subjects"]
    for v in VOTERS + ("proposed",):
        lines.append(f"  {v:<9} accuracy {np.mean(b[v] == b.label):.3f}  ({int((b[v] != b.label).sum())} wrong)")
    auto = b[~b.needs_review]
    lines.append(f"  auto (unanimous) {len(auto)}/{len(b)} = {len(auto) / len(b):.0%} of loops, accuracy "
                 f"{np.mean(auto.proposed == auto.label):.3f} ({int((auto.proposed != auto.label).sum())} wrong)")
    return "\n".join(lines)
