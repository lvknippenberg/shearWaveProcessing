"""Echo-view sorting of the SW acquisitions: PLAX / PSAX (/ Apical / Unclear) per measurement folder.

Passive SWE needs PLAX (and apical) views and active SWE also uses PSAX, so every acquisition gets a
view label. The labels live in ONE table, not in the folder layout (``Z:\\raw_data`` is the DataHub
mirror and is never restructured): ``study/logs/view_classification/sw_views_manual.csv``, columns
``subject, folder, label`` (+ provenance). Tools filter on ``label``.

Pipeline (``scripts/view_sort.py``; workflow and numbers in ``docs/view_sorting.md``):

1. ``features``  per buffer-3 GIF: EchoPrime frame-classifier features + 11-class probabilities per
                 frame, and EchoPrime video-encoder embeddings (``swp.views.echoprime``); cached
                 per folder.
2. ``classify``  four voters per loop (``swp.views.vote``): EchoPrime's own call, two per-subject
                 2-cluster splits (frame features incl. their change over the cycle; video
                 embeddings) and a binary head trained on the reviewed labels. Unanimous -> safe
                 to auto-label; anything else is flagged. Proposal = majority vote, 2-2 ties by
                 the protocol block fit (a PLAX block, then a PSAX block).
3. ``review``    the Tk review window (``swp.views.review``): one subject per screen, every loop
                 playing, flagged loops framed red; ENTER saves the subject to the labels table.
4. ``train-head`` refit the binary head on all labels (after a review round), ``evaluate`` =
                 leave-one-subject-out check of the whole voter set against the labels.

On the 2026-10-02 study (724 loops, 48 subjects) the unanimous consensus auto-labels ~81 % at
99.5 % accuracy; see the study record ``study/analysis/view_classification/README.md``.

EchoPrime (Vukadinovic et al. 2024, github.com/echonet/EchoPrime) is under the Cedars-Sinai
academic licence; its weights are downloaded, never committed.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..paths import DATA_ROOT, RAW_DATA, REPO_ROOT

LABELS = ("PLAX", "PSAX", "Apical", "Unclear")
LOGS = Path(REPO_ROOT) / "study" / "logs" / "view_classification"
LABELS_CSV = Path(os.environ.get("SWP_VIEW_LABELS") or LOGS / "sw_views_manual.csv")
VOTES_CSV = Path(os.environ.get("SWP_VIEW_VOTES") or LOGS / "sw_views_votes.csv")
CACHE_DIR = Path(os.environ.get("SWP_VIEW_CACHE") or Path(DATA_ROOT) / "view_classification" / "features")
WEIGHTS_DIR = Path(os.environ.get("SWP_ECHOPRIME_DIR")
                   or Path(DATA_ROOT) / "view_classification" / "EchoPrime" / "model_data" / "weights")
HEAD_FILE = Path(__file__).with_name("view_head.npz")
GIF_NAME = "CombinedData_buffer3_iq.gif"


def find_loops(root=RAW_DATA, subjects=None):
    """SW measurement folders under ``root`` (``<root>/<subject>/<*SW_data*>``): list of
    (subject, folder name, GIF path or None when buffer 3 is not beamformed yet)."""
    out = []
    root = Path(root)
    for sdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("C")):
        if subjects and sdir.name not in subjects:
            continue
        for f in sorted(p for p in sdir.iterdir() if p.is_dir() and "SW_data" in p.name):
            g = f / "output" / GIF_NAME
            out.append((sdir.name, f.name, g if g.is_file() else None))
    return out


def acq_time(folder: str) -> str:
    """Sort key within a subject: folder names end in <date>_<HH-MM-SS>."""
    return folder.split("_")[-1]
