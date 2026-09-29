"""Summary figure of the preliminary manual passive evaluation (after passive_manual_prelim.py).

Three panels from study/logs/passive_manual_prelim/windows.csv:
  a  confidence per detected window, by how the detector found it;
  b  AVC windows: detected peak time relative to the Weissler QS2, by confidence
     (the shaded band is the detector's +-120 ms search window);
  c  hand speed |c| of usable windows (confidence >= 2), MVC vs AVC; open markers = the wave
     crosses the line in < 5 frames (speed poorly resolved at ~926 Hz).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CONF = {0: "#cde2fb", 1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}      # ordinal blue ramp
MVC_C, AVC_C = "#2a78d6", "#eb6834"                                  # categorical slots 1, 2
MVC_MAX_MS, AVC_TOL_MS = 150.0, 120.0
FRAME_MS = 1.08


def main():
    w = pd.read_csv(os.path.join(D, "windows.csv"))
    s = w[w.confidence.notna()].copy()
    edge = (s.pos_in_search < 0.05) | (s.pos_in_search > 0.95)
    no_ecg = s.label.isna() | (s.label == "?")
    s["cat"] = np.select([s.expect.isna() & no_ecg, s.expect.isna(), edge],
                         ["no ECG:\nenergy only", "energy top-up\n(no phase match)",
                          "peak on search-\nwindow edge"], "")
    s.loc[s.cat == "", "cat"] = s.loc[s.cat == "", "expect"] + " phase\nwindow, interior"

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    fig.patch.set_facecolor(SURF)

    # a - stacked horizontal bars
    order = ["MVC phase\nwindow, interior", "AVC phase\nwindow, interior", "peak on search-\nwindow edge",
             "energy top-up\n(no phase match)", "no ECG:\nenergy only"]
    tab = pd.crosstab(s.cat, s.confidence).reindex(order).fillna(0)
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for c in (0, 1, 2, 3):
        v = tab.get(float(c), pd.Series(0, index=order)).to_numpy()
        ax[0].barh(y, v, left=left, height=0.62, color=CONF[c], edgecolor=SURF, linewidth=2,
                   label={0: "0 none", 1: "1 guess", 2: "2 plausible", 3: "3 clear"}[c])
        for yi, li, vi in zip(y, left, v):
            if vi >= 2:
                ax[0].text(li + vi / 2, yi, f"{int(vi)}", ha="center", va="center", fontsize=8,
                           color=INK if c < 2 else "white")
        left += v
    ax[0].set_yticks(y, order)
    ax[0].set_xlabel("windows")
    ax[0].set_title("a   Confidence by how the window was detected", loc="left", color=INK)
    ax[0].legend(title="confidence", frameon=False, fontsize=8, title_fontsize=8, loc="lower right")
    ax[0].grid(axis="x", color=GRID, lw=0.8); ax[0].set_axisbelow(True)

    # b - AVC timing
    a = s[s.expect == "AVC"].copy()
    a["d"] = a.phase_ms - a.qs2_ms
    ax[1].axhspan(-AVC_TOL_MS, AVC_TOL_MS, color=GRID, alpha=0.6, lw=0)
    rng = np.random.default_rng(0)
    for c in (0, 1, 2, 3):
        d = a[a.confidence == c].d
        ax[1].scatter(c + rng.uniform(-0.12, 0.12, len(d)), d, s=46, color=CONF[max(c, 2)] if c >= 2 else "#86b6ef",
                      edgecolor=SURF, linewidth=1.2, zorder=3)
    ax[1].axhline(0, color=INK2, lw=0.8, ls="--")
    ax[1].text(3.45, 0, "Weissler QS2", va="bottom", ha="right", fontsize=8, color=INK2)
    ax[1].text(3.45, AVC_TOL_MS, "search window", va="top", ha="right", fontsize=8, color=INK2)
    ax[1].set_xticks([0, 1, 2, 3], ["0\nnone", "1\nguess", "2\nplausible", "3\nclear"])
    ax[1].set_xlim(-0.5, 3.5)
    ax[1].set_ylabel("AVC peak - QS2 [ms]")
    ax[1].set_title("b   AVC windows: where the detector put them", loc="left", color=INK)
    ax[1].grid(axis="y", color=GRID, lw=0.8); ax[1].set_axisbelow(True)

    # c - speeds
    u = s[(s.confidence >= 2) & s.label.isin(["MVC", "AVC"])].copy()
    u["v"] = u.speed_m_s.abs()
    u["poor"] = u.mline_length_mm / u.v / FRAME_MS < 5
    for k, (lab, col) in enumerate((("MVC", MVC_C), ("AVC", AVC_C))):
        g = u[u.label == lab]
        x = k + rng.uniform(-0.13, 0.13, len(g))
        ok = ~g.poor.to_numpy()
        ax[2].scatter(x[ok], g.v[ok], s=46, color=col, edgecolor=SURF, linewidth=1.2, zorder=3)
        ax[2].scatter(x[~ok], g.v[~ok], s=46, facecolor="none", edgecolor=col, linewidth=1.6, zorder=3)
        m_all, m_ok = g.v.median(), g.v[ok].median()
        ax[2].plot([k - 0.3, k + 0.3], [m_ok, m_ok], color=INK, lw=2, zorder=4)
        ax[2].text(k + 0.33, m_ok, f"{m_ok:.1f}", va="center", fontsize=8, color=INK)
        ax[2].text(k, -0.02, f"n={ok.sum()} (+{(~ok).sum()} open)", transform=ax[2].get_xaxis_transform(),
                   ha="center", va="top", fontsize=8, color=INK2)
    ax[2].set_xticks([0, 1], ["MVC", "AVC"])
    ax[2].tick_params(axis="x", pad=14)
    ax[2].set_xlim(-0.6, 1.7)
    ax[2].set_ylabel("hand speed |c| [m/s]")
    ax[2].set_title("c   Usable windows (confidence >= 2)", loc="left", color=INK)
    ax[2].text(-0.58, ax[2].get_ylim()[1], "open = crosses the line\nin < 5 frames\nbar = median of filled",
               ha="left", va="top", fontsize=8, color=INK2)
    ax[2].grid(axis="y", color=GRID, lw=0.8); ax[2].set_axisbelow(True)

    for a_ in ax:
        a_.set_facecolor(SURF)
    fig.tight_layout()
    out = os.path.join(D, "prelim_summary.png")
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
