"""Blind scoring app for the metric-validation experiment.

Shows each randomized 3-column space-time plot (displacement / velocity / acceleration) one at a
time; click 1-5 for how clearly a shear wave is visible and it **auto-advances** to the next plot.
Recipe and metric are HIDDEN (blind). Scores save to scores.csv after every click and resume where
you left off. When done, run scripts/metric_experiment_analyze.py.

    KERAS_BACKEND=torch  <zea-python>  -m streamlit run swp_gui/score_app.py
"""
from __future__ import annotations

import csv
import json
import os

import streamlit as st

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
RUBRIC = {1: "1 · noise", 2: "2 · unclear", 3: "3 · weak wave", 4: "4 · clear", 5: "5 · very clear"}

st.set_page_config(page_title="SWE metric scoring", layout="wide")


def available_rounds():
    return sorted(d for d in os.listdir(BASE)
                  if os.path.exists(os.path.join(BASE, d, "manifest.json")))


def load_scores(sp):
    d = {}
    if os.path.exists(sp):
        with open(sp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d[int(row["id"])] = int(row["score"])
    return d


def save_scores(sp, scores):
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "score"])
        for k in sorted(scores):
            w.writerow([k, scores[k]])


rounds = available_rounds()
if not rounds:
    st.error(f"No datasets under {BASE}. Run scripts/metric_experiment_generate.py first.")
    st.stop()

# ---- dataset selector: score each dataset independently (scores are relative within a dataset) ----
_cr = os.path.join(BASE, "current_round.txt")
default = open(_cr, encoding="utf-8").read().strip() if os.path.exists(_cr) else rounds[0]
default = default if default in rounds else rounds[0]
ROUND = st.sidebar.selectbox("Dataset", rounds, index=rounds.index(default),
                             help="Score each dataset on its own; a '4' on one voltage is not a '4' "
                                  "on another. Your scores are saved per dataset.")
open(_cr, "w", encoding="utf-8").write(ROUND)              # so analyze defaults to what you last viewed

OUT_ROOT = os.path.join(BASE, ROUND)
manifest = json.load(open(os.path.join(OUT_ROOT, "manifest.json"), encoding="utf-8"))
items = manifest["items"]
n = len(items)
SCORES = os.path.join(OUT_ROOT, "scores.csv")

# reset per-dataset state when the selected dataset changes
if st.session_state.get("cur_round") != ROUND:
    st.session_state["cur_round"] = ROUND
    st.session_state["scores"] = load_scores(SCORES)
    st.session_state.pop("pos", None)
scores = st.session_state["scores"]


def first_unscored():
    for i, it in enumerate(items):
        if it["id"] not in scores:
            return i
    return n            # all done

if "pos" not in st.session_state:
    st.session_state["pos"] = first_unscored()

pos = int(st.session_state["pos"])

# ---- sidebar: progress + minimal navigation ----
st.sidebar.title(f"Scoring · {ROUND}")
_ds = os.path.basename(manifest.get("folder", "")) or "?"
st.sidebar.caption(f"dataset: {_ds}  ·  meas {manifest.get('meas', 0)}  ·  mline {manifest.get('mline','?')}")
st.sidebar.progress(len(scores) / n, text=f"{len(scores)}/{n} scored")
c1, c2 = st.sidebar.columns(2)
if c1.button("◀ back", width="stretch"):
    st.session_state["pos"] = max(0, pos - 1); st.rerun()
if c2.button("skip ▶", width="stretch"):
    st.session_state["pos"] = min(n - 1, pos + 1); st.rerun()
if st.sidebar.button("jump to first unscored", width="stretch"):
    st.session_state["pos"] = first_unscored(); st.rerun()
st.sidebar.caption("Click 1-5 for how clearly a shear wave (an outward ∧ from the dashed origin) is "
                   "visible in ANY of the three columns; it auto-advances. Recipe & metric are hidden.")

if pos >= n:
    st.success(f"All {n} plots scored ✔  — run scripts/metric_experiment_analyze.py to compare.")
    if st.button("re-review from start"):
        st.session_state["pos"] = 0; st.rerun()
    st.stop()

item = items[pos]
prev = scores.get(item["id"])

st.markdown(f"### Plot {pos + 1} / {n}" + (f"  ·  _your score: {prev}_" if prev else ""))
st.image(os.path.join(OUT_ROOT, item["png"]), width="stretch")

cols = st.columns(5)
for k in range(1, 6):
    if cols[k - 1].button(RUBRIC[k], key=f"score_{k}", width="stretch", type="primary"):
        scores[item["id"]] = k
        save_scores(SCORES, scores)
        st.session_state["pos"] = pos + 1          # auto-advance to the next plot
        st.rerun()
