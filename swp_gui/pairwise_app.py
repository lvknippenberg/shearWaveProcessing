"""Pairwise (2-alternative) comparison scorer — far more sensitive than absolute 1-5 for near-ties.

Shows two space-time plots stacked top/bottom (so the same quantity - displacement / velocity /
acceleration - lines up vertically); you pick which shows the shear wave more clearly (or 'no
preference'). Comparisons save to pairs.csv and are later ranked (Bradley-Terry) and
correlated with the recipe parameters to pin the optimum (scripts/pairwise_analyze.py). Blind: recipes
and metric hidden. Reads the active dataset (current_round.txt) or pick one in the sidebar.

    KERAS_BACKEND=torch  <zea-python>  -m streamlit run swp_gui/pairwise_app.py
"""
from __future__ import annotations

import csv
import json
import os
import random

import streamlit as st

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
PASSES = 6          # each item appears in ~PASSES pairs

st.set_page_config(page_title="SWE pairwise", layout="wide")


def rounds_with_manifest():
    return sorted(d for d in os.listdir(BASE)
                  if os.path.exists(os.path.join(BASE, d, "manifest.json")))


rs = rounds_with_manifest()
if not rs:
    st.error("No datasets found."); st.stop()
_cr = os.path.join(BASE, "current_round.txt")
default = open(_cr, encoding="utf-8").read().strip() if os.path.exists(_cr) else rs[0]
default = default if default in rs else rs[0]
ROUND = st.sidebar.selectbox("Dataset", rs, index=rs.index(default))

OUT = os.path.join(BASE, ROUND)
items = json.load(open(os.path.join(OUT, "manifest.json"), encoding="utf-8"))["items"]
png_by_id = {it["id"]: os.path.join(OUT, it["png"]) for it in items}
ids = [it["id"] for it in items]
SCHED = os.path.join(OUT, "pairs_schedule.json")
PAIRS = os.path.join(OUT, "pairs.csv")

# build a fixed comparison schedule once (each item in ~PASSES pairs), shuffled
if not os.path.exists(SCHED):
    rng = random.Random(0)
    sched = []
    for _ in range(PASSES):
        shuf = ids[:]; rng.shuffle(shuf)
        for i in range(0, len(shuf) - 1, 2):
            sched.append([shuf[i], shuf[i + 1]])
    rng.shuffle(sched)
    json.dump(sched, open(SCHED, "w"))
sched = json.load(open(SCHED, encoding="utf-8"))


def load_done():
    if os.path.exists(PAIRS):
        return list(csv.reader(open(PAIRS, encoding="utf-8")))[1:]
    return []


def append_result(a, b, winner):
    new = not os.path.exists(PAIRS)
    with open(PAIRS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["a", "b", "winner"])
        w.writerow([a, b, winner])


if st.session_state.get("pw_round") != ROUND:
    st.session_state["pw_round"] = ROUND
    st.session_state["pw_pos"] = len(load_done())

pos = int(st.session_state["pw_pos"])
total = len(sched)

st.sidebar.title(f"Pairwise · {ROUND}")
st.sidebar.progress(min(pos, total) / total, text=f"{pos}/{total} comparisons")
img_w = st.sidebar.slider("image width (px)", 500, 1500, 780, 20,
                          help="Shrink until both plots + the buttons fit on screen without scrolling.")
st.sidebar.caption("Pick the plot where a symmetric shear wave (outward ∧ from the dashed origin) is "
                   "CLEARER across the three columns. 'No preference' is fine. You can stop anytime; "
                   "more comparisons = sharper ranking.")
if st.sidebar.button("undo last") and pos > 0:
    rows = load_done()[:-1]
    with open(PAIRS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["a", "b", "winner"]); w.writerows(rows)
    st.session_state["pw_pos"] = len(rows); st.rerun()

if pos >= total:
    st.success(f"All {total} comparisons done ✔ — run scripts/pairwise_analyze.py")
    st.stop()

a, b = sched[pos]
st.caption(f"Which is clearer?  ({pos + 1}/{total})")
# stacked top/bottom so the same quantity (displacement | velocity | acceleration) lines up vertically;
# width (and thus height) is controlled by the sidebar slider so it fits without scrolling.
st.image(png_by_id[a], width=img_w, caption="TOP")
st.image(png_by_id[b], width=img_w, caption="BOTTOM")

b1, b2, b3 = st.columns(3)


def choose(winner):
    append_result(a, b, winner)
    st.session_state["pw_pos"] = pos + 1
    st.rerun()


if b1.button("▲ TOP clearer", width="stretch", type="primary"):
    choose("a")
if b2.button("= no preference", width="stretch"):
    choose("tie")
if b3.button("▼ BOTTOM clearer", width="stretch", type="primary"):
    choose("b")
