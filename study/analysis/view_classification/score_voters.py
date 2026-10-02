"""Score the label-free voters against the manual review labels.

Usage: python score_voters.py   (study defaults; writes study/logs/view_classification/voter_scores.txt)
"""
from pathlib import Path

import numpy as np
import pandas as pd

LOGS = Path(__file__).resolve().parents[2] / "logs" / "view_classification"


def main():
    man = pd.read_csv(LOGS / "sw_views_manual.csv")
    cons = pd.read_csv(LOGS / "sw_views_consensus.csv")
    first = pd.read_csv(LOGS / "all_sw_views.csv")[["folder", "p_view", "frame_agreement"]]
    d = man.merge(cons, on=["subject", "folder"]).merge(first, on="folder")
    lines = []
    out = lines.append
    out(f"{len(d)} loops, {d.subject.nunique()} subjects; manual labels {d.label.value_counts().to_dict()}")
    out(f"proposals changed by the reviewer: {int(d.changed.sum())} "
        f"(flagged {int(d.changed[d.needs_review].sum())}/{int(d.needs_review.sum())}, "
        f"unanimous {int(d.changed[~d.needs_review].sum())}/{int((~d.needs_review).sum())})")

    b = d[d.label.isin(["PLAX", "PSAX"])]
    out(f"\nbinary accuracy on the {len(b)} PLAX/PSAX-labelled loops (unclear/apical excluded):")
    for col, name in (("ep", "EchoPrime frame classifier"), ("clu_f", "cluster frame features"),
                      ("clu_v", "cluster video embeddings"), ("self", "self-trained head"),
                      ("proposed", "review proposal (majority + block fit)")):
        acc = np.mean(b[col] == b.label)
        out(f"  {name:<40} {acc:.3f}  ({int((b[col] != b.label).sum())} wrong)")
    u = b[b.consensus != "review"]
    out(f"  unanimous consensus (auto set)           {np.mean(u.consensus == u.label):.3f}  "
        f"({int((u.consensus != u.label).sum())} wrong of {len(u)})")
    conf = b[(b.p_view >= 0.8) & (b.frame_agreement >= 0.9)]
    out(f"  first pass 'confident' EchoPrime         {np.mean(conf.ep == conf.label):.3f}  "
        f"({int((conf.ep != conf.label).sum())} wrong of {len(conf)})")

    out("\nunclear / apical by the reviewer:")
    for _, r in d[~d.label.isin(["PLAX", "PSAX"])].iterrows():
        out(f"  {r.subject} {r.folder.split('_')[-1]}  {r.label:<8} proposed {r.proposed:<5} "
            f"flagged {bool(r.needs_review)}  votes {r.votes}  {r.review_reason if isinstance(r.review_reason, str) else ''}")
    out("\nchanged proposals:")
    for _, r in d[d.changed].iterrows():
        out(f"  {r.subject} {r.folder.split('_')[-1]}  {r.proposed:<5} -> {r.label:<8} "
            f"flagged {bool(r.needs_review)}  votes {r.votes}")
    txt = "\n".join(lines)
    print(txt)
    (LOGS / "voter_scores.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
