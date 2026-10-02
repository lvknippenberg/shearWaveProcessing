"""Sort SW acquisitions by echo view (PLAX / PSAX / Apical / Unclear): voters + manual review.

    python scripts/view_sort.py run      --root "Z:/raw_data"     # new data: features -> classify -> review
    python scripts/view_sort.py features --root "Z:/raw_data" [--subject C000000050]
    python scripts/view_sort.py classify --root "Z:/raw_data" [--subject ...]
    python scripts/view_sort.py review   [--subject ...] [--all] [--auto-accept-unanimous]
    python scripts/view_sort.py status   --root "Z:/raw_data"
    python scripts/view_sort.py train-head                        # refit the binary head on all labels
    python scripts/view_sort.py evaluate                          # leave-one-subject-out check vs the labels
    python scripts/view_sort.py download-weights                  # EchoPrime weights (once per machine)

Only reads Z: (the buffer-3 GIFs written by ``run.py beamform``). Feature cache, labels and votes
locations: ``swp.views`` (env SWP_VIEW_CACHE / SWP_VIEW_LABELS / SWP_VIEW_VOTES / SWP_ECHOPRIME_DIR).
The labels table ``study/logs/view_classification/sw_views_manual.csv`` is what other tools filter
on (column ``label``). Workflow, keys and numbers: docs/view_sorting.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from swp.paths import RAW_DATA                                           # noqa: E402
from swp.views import (CACHE_DIR, HEAD_FILE, LABELS_CSV, VOTES_CSV, WEIGHTS_DIR, acq_time,  # noqa: E402
                       find_loops)
from swp.views import echoprime, vote                                    # noqa: E402
from swp.views.review import read_labels, save_subject                   # noqa: E402


def cmd_download(a):
    print(echoprime.download_weights(Path(a.weights)))


def cmd_features(a):
    loops = find_loops(a.root, a.subject)
    todo = [(s, f, g) for s, f, g in loops if g is not None
            and (a.force or not echoprime.cache_path(s, f, a.cache).is_file())]
    missing = sum(g is None for _, _, g in loops)
    print(f"{len(loops)} SW folders, {missing} without a buffer-3 GIF (not beamformed), {len(todo)} to extract")
    if not todo:
        return
    ex = echoprime.Extractor(Path(a.weights))
    t0 = time.time()
    for i, (s, f, g) in enumerate(todo):
        echoprime.save_cached(s, f, ex(g), a.cache)
        if i % 25 == 0 or i == len(todo) - 1:
            print(f"  {i + 1}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)


def _cached_items(root, subjects, cache):
    items, gifs = [], {}
    for s, f, g in find_loops(root, subjects):
        feats = echoprime.load_cached(s, f, cache) if g is not None else None
        if feats is not None:
            items.append((s, f, feats))
            gifs[f] = str(g)
    return items, gifs


def cmd_classify(a):
    items, gifs = _cached_items(a.root, a.subject, a.cache)
    if not items:
        sys.exit("no cached features: run `features` first")
    df, F, V = vote.table(items)
    v = vote.vote(df, F, V, vote.load_head(a.head))
    v["gif"] = v.folder.map(gifs)
    old = pd.read_csv(a.votes) if Path(a.votes).is_file() else None
    if old is not None:
        old = old[~old.subject.isin(v.subject.unique())]
        v = pd.concat([old, v], ignore_index=True)
    v = v.sort_values(["subject", "folder"], key=lambda c: c if c.name == "subject" else c.map(acq_time))
    Path(a.votes).parent.mkdir(parents=True, exist_ok=True)
    v.to_csv(a.votes, index=False)
    lab = read_labels(a.labels)
    new = v[~v.folder.isin(lab.folder)]
    print(f"{len(df)} loops classified ({df.subject.nunique()} subjects) -> {a.votes}")
    print(f"  unlabelled: {len(new)} loops in {new.subject.nunique()} subjects, "
          f"{int(new.needs_review.sum())} flagged; proposals {new.proposed.value_counts().to_dict()}")


def _pending_subjects(v, lab):
    pend = v[~v.folder.isin(lab.folder)]
    return sorted(pend.subject.unique())


def cmd_review(a):
    from swp.views import review
    v = pd.read_csv(a.votes)
    v["reason"] = v.reason.fillna("")
    lab = read_labels(a.labels)
    subjects = a.subject or (sorted(v.subject.unique()) if a.all else _pending_subjects(v, lab))
    if a.auto_accept_unanimous and not a.subject:
        flagged = v.groupby("subject").needs_review.sum()
        for s in [s for s in subjects if flagged.get(s, 0) == 0]:
            sub = v[v.subject == s].reset_index(drop=True)
            save_subject(a.labels, sub, sub.proposed, source="auto")
            print(f"{s}: no flagged loop -> auto-labelled")
        subjects = [s for s in subjects if flagged.get(s, 0) > 0]
    if not subjects:
        print("nothing to review")
        return
    print(f"{len(subjects)} subject(s) to review")
    review.run(v, dict(zip(v.folder, v.gif)), Path(a.labels), subjects)
    print(f"labels: {a.labels}")


def cmd_status(a):
    loops = find_loops(a.root, a.subject)
    lab = read_labels(a.labels)
    d = pd.DataFrame(loops, columns=["subject", "folder", "gif"])
    d["gif_ok"] = d.gif.notna()
    d["cached"] = [echoprime.cache_path(s, f, a.cache).is_file() for s, f in zip(d.subject, d.folder)]
    d["labelled"] = d.folder.isin(lab.folder)
    print(f"{len(d)} SW folders in {d.subject.nunique()} subjects: buffer-3 GIF {int(d.gif_ok.sum())}, "
          f"features cached {int(d.cached.sum())}, labelled {int(d.labelled.sum())}")
    print("labels:", lab.label.value_counts().to_dict())
    todo = d[~d.labelled]
    if len(todo):
        print("not labelled yet:", todo.groupby("subject").size().to_dict())


def _labelled_arrays(a):
    lab = read_labels(a.labels).set_index("folder").label
    items, _ = _cached_items(a.root, sorted(read_labels(a.labels).subject.unique()), a.cache)
    items = [t for t in items if t[1] in lab.index]
    df, F, V = vote.table(items)
    return df, F, V, lab


def cmd_train_head(a):
    df, F, V, lab = _labelled_arrays(a)
    y = df.folder.map(lab).map({"PLAX": 1, "PSAX": 0}).fillna(-1).astype(int).to_numpy()
    head = vote.fit_head(F, V, df.subject.to_numpy(), y)
    vote.save_head(head, a.head)
    print(f"head fitted on {int(head['n_train'])} loops / {int(head['n_subjects'])} subjects -> {a.head}")


def cmd_evaluate(a):
    df, F, V, lab = _labelled_arrays(a)
    print(vote.evaluate(df, F, V, lab))


def cmd_run(a):
    cmd_features(a)
    cmd_classify(a)
    cmd_review(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=RAW_DATA)
    common.add_argument("--subject", action="append", default=[])
    common.add_argument("--cache", default=str(CACHE_DIR))
    common.add_argument("--labels", default=str(LABELS_CSV))
    common.add_argument("--votes", default=str(VOTES_CSV))
    common.add_argument("--head", default=str(HEAD_FILE))
    common.add_argument("--weights", default=str(WEIGHTS_DIR))
    for name, fn in (("download-weights", cmd_download), ("features", cmd_features), ("classify", cmd_classify),
                     ("review", cmd_review), ("status", cmd_status), ("train-head", cmd_train_head),
                     ("evaluate", cmd_evaluate), ("run", cmd_run)):
        p = sub.add_parser(name, parents=[common])
        if name in ("features", "run"):
            p.add_argument("--force", action="store_true", help="re-extract cached features")
        if name in ("review", "run"):
            p.add_argument("--all", action="store_true", help="also subjects that are already labelled")
            p.add_argument("--auto-accept-unanimous", action="store_true",
                           help="label subjects without any flagged loop automatically (source=auto)")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
