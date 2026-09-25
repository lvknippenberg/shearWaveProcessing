"""Read-only validation of three ways to find the chronological start of buffer 3 (circular live loop).

Buffer 3 (focused live loop) is stored in a circular receive buffer whose head is unknown: the saved
``Resource.RcvBuffer(3).lastFrame`` is always ``numFrames`` because ``SaveRFData`` copies the buffers
mid-sequence, while VSX only sets ``lastFrame`` when a script freezes or exits (Sequence
Programming Manual p57; docs/passive_manual.md). The frame triggers and R-peaks in the trigger log
are correct. Three estimators of ``first`` = the 0-based stored slot that was acquired FIRST
(``first = 0`` means the stored order is already chronological; slot q was acquired as chronological
frame ``(q - first) mod N``):

``rf``   frame continuity on the converted RF, no beamforming: per transmit, the channel envelopes
         summed without receive delays (an incoherent A-line), log-compressed, band-passed to
         anatomy scale (~1-6 mm). One cyclic neighbour pair is the jump from the newest back to the
         oldest frame; it is the least similar pair.
``iq``   the same continuity statistic on the beamformed buffer-3 IQ (anatomy on a common grid).
``b1``   similarity to buffer 1 (chronological, correctly timed): for each candidate ``first``, every
         stored buffer-3 frame gets its trigger time (frame centre = trigger + half a frame period;
         the trigger fires on the first line) and hence a cardiac phase, is paired with the buffer-1
         frame of the nearest phase (also at its frame centre), and the mean anatomy correlation of
         the pairs is the score. The trigger-to-frame assignment is allowed to be off by +-1 frame
         (a frame acquired but never transferred). Needs a trustworthy R-peak record.

Continuity score of a candidate break (between stored slots first-1 and first, cyclic): the mean of
the lag-1 and lag-2 correlations crossing it, each divided by the median of that lag. Confidence of
any method = the gap between its best candidate and the best candidate that is not adjacent to it
(adjacent candidates only move one frame across the boundary).

Nothing is written to the data. Outputs:
    study/logs/buffer3_unwrap_validation.csv
    study/montages/buffer3_unwrap/<subject>_<folder>.png   (continuity profiles + b1 score curve)

    python study/analysis/buffer3_unwrap_validation.py [--per-group 24] [--folder F ...]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))

import h5py                                                    # noqa: E402
import hdf5plugin                                              # noqa: E402,F401

from swp.manual import store as S                              # noqa: E402
from swp.manual._light import rrcheck, triggerlog              # noqa: E402
from buffer3_timing_fit import anatomy_stack, phase_of         # noqa: E402

_RAW = "tracks/track_0/data/raw_data"
CHECK_FOLDER = "Z:/2026_09_25 LastFrame check/testlastframe_SW_data_25-September-2026_14-12-02"


# ------------------------------------------------------------------ per-frame anatomy vectors
def rf_anatomy_stack(path, el_step=4, dec=8, skip=40):
    """Anatomy vectors from the converted RF (n_frames, n_tx, n_ax, n_el, 1), no beamforming."""
    from scipy.ndimage import gaussian_filter
    from scipy.signal import hilbert
    out = []
    with h5py.File(path, "r") as f:
        raw = f[_RAW]
        for k in range(raw.shape[0]):
            x = np.asarray(raw[k, :, :, ::el_step, 0], np.float32)          # (n_tx, n_ax, n_el/4)
            env = np.abs(hilbert(x, axis=1)).sum(-1)                         # incoherent A-lines
            n = (env.shape[1] // dec) * dec
            env = env[:, :n].reshape(env.shape[0], -1, dec).mean(-1)         # ~0.4 mm samples
            d = 20 * np.log10(env / env.max() + 1e-6)
            a = gaussian_filter(d, (0.7, 2.5)) - gaussian_filter(d, (3, 15))
            a = a[:, skip:]                                                  # drop transmit ring-down
            a = a - a.mean()
            out.append((a / (np.linalg.norm(a) + 1e-12)).ravel())
    return np.array(out)


# ------------------------------------------------------------------ method 1: continuity
def continuity(A):
    """-> (score per candidate first (N,), lag-1 cyclic pair corr (N,)). Lower score = more likely
    the break. Candidate ``first`` = the break sits between slots first-1 and first (cyclic)."""
    n = len(A)
    M = A @ A.T
    lag = {L: np.array([M[i, (i + L) % n] for i in range(n)]) for L in (1, 2)}   # pair (i, i+L)
    med = {L: np.median(v) for L, v in lag.items()}
    score = np.empty(n)
    for first in range(n):
        b = (first - 1) % n                                   # last slot before the break
        l1 = lag[1][b] / med[1]
        l2 = np.mean([lag[2][(b - 1) % n], lag[2][b]]) / med[2]   # pairs (b-1, b+1), (b, b+2)
        score[first] = 0.5 * (l1 + l2)
    return score, lag[1]


def best_and_margin(values, lower_is_better):
    """(best index, gap to the best non-adjacent candidate)."""
    v = np.asarray(values, float)
    n = len(v)
    s = v if lower_is_better else -v
    s = np.where(np.isfinite(s), s, np.inf)
    i = int(np.argmin(s))
    others = [j for j in range(n) if min((j - i) % n, (i - j) % n) >= 2]
    j = min(others, key=lambda k: s[k])
    return i, float(s[j] - s[i])


# ------------------------------------------------------------------ method 2: similarity to buffer 1
def b1_scores(folder, A3, A1):
    """Score (mean correlation) per candidate first, for assignment shifts -1, 0, +1 frame."""
    tl = triggerlog()
    b1, b3 = tl.buffer_timing(folder, 1), tl.buffer_timing(folder, 3)
    if b1 is None or b3 is None or len(A1) != b1.n_frames or len(A3) != b3.n_frames:
        return None
    rp = np.asarray(b3.r_peaks_ms)
    ph1 = phase_of(b1.frame_times() + b1.frame_ms / 2, rp)
    ok1 = np.isfinite(ph1)
    M = A3 @ A1.T
    n = len(A3)
    out = {}
    for shift in (-1, 0, 1):
        t_chrono = b3.frame_times() + b3.frame_ms / 2 + shift * b3.frame_ms   # chronological frames
        ph_chrono = phase_of(t_chrono, rp)
        sc = np.full(n, np.nan)
        for first in range(n):
            vals = []
            for q in range(n):
                ph = ph_chrono[(q - first) % n]
                if not np.isfinite(ph):
                    continue
                d = np.where(ok1, np.abs(ph1 - ph), np.inf)
                f = int(np.argmin(d))
                if d[f] < 10:
                    vals.append(M[q, f])
            if len(vals) >= n // 2:
                sc[first] = np.mean(vals)
        out[shift] = sc
    return out, float(np.median(np.diff(rp))) if rp.size > 1 else float("nan")


# ------------------------------------------------------------------ one folder
def analyse(folder, fig_dir):
    p = S.Paths(folder)
    row = dict(folder=f"{Path(folder).parent.name}/{Path(folder).name}")
    t0 = time.perf_counter()
    conv = Path(p.output) / "converted" / "CombinedData_buffer3.hdf5"
    A_rf = rf_anatomy_stack(str(conv))
    row["t_rf_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    A_iq = anatomy_stack(p.bmode(3))
    row["t_iq_s"] = round(time.perf_counter() - t0, 1)
    n = len(A_iq)
    row["n_frames"] = n
    res = {}
    for name, A in (("rf", A_rf), ("iq", A_iq)):
        sc, l1 = continuity(A)
        first, margin = best_and_margin(sc, lower_is_better=True)
        res[name] = (sc, l1)
        row[f"{name}_first"], row[f"{name}_margin"] = first, round(margin, 3)
        row[f"{name}_break_corr"] = round(float(l1[(first - 1) % n]), 3)
        row[f"{name}_median_corr"] = round(float(np.median(l1)), 3)
    try:
        chk = rrcheck().assess_rr(folder)
        row["ecg_trustworthy"], row["ecg"] = bool(chk.trustworthy), f"{chk.status}: {chk.quality}"
    except Exception as exc:                                           # noqa: BLE001
        row["ecg_trustworthy"], row["ecg"] = False, f"check failed: {exc}"
    b1 = None
    if row["ecg_trustworthy"] and Path(p.bmode(1)).exists():
        t0 = time.perf_counter()
        A1 = anatomy_stack(p.bmode(1))
        r = b1_scores(folder, A_iq, A1)
        row["t_b1_s"] = round(time.perf_counter() - t0, 1)
        if r is not None:
            scores, rr = r
            row["rr_ms"] = round(rr)
            row["span_over_rr"] = round(n * 39.42 / rr, 2) if np.isfinite(rr) else None
            best_shift = max(scores, key=lambda s: np.nanmax(scores[s]))
            first, margin = best_and_margin(scores[best_shift], lower_is_better=False)
            row["b1_first"], row["b1_margin"], row["b1_shift"] = first, round(margin, 3), best_shift
            row["b1_score"] = round(float(scores[best_shift][first]), 3)
            row["b1_score_first0"] = round(float(scores[0][0]), 3)
            b1 = scores
    for a, b in (("rf", "iq"), ("rf", "b1"), ("iq", "b1")):
        if f"{b}_first" in row:
            d = min((row[f"{a}_first"] - row[f"{b}_first"]) % n, (row[f"{b}_first"] - row[f"{a}_first"]) % n)
            row[f"{a}_vs_{b}"] = d
    _figure(row, res, b1, fig_dir)
    return row


def _figure(row, res, b1, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = row["n_frames"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    x = np.arange(n)
    for name, c in (("rf", "tab:blue"), ("iq", "tab:orange")):
        sc, l1 = res[name]
        # lag-1 pair (q-1 -> q) plotted at candidate first = q
        axs[0].plot(x, np.roll(l1, 1), "-o", ms=3, color=c,
                    label=f"{name}: first={row[f'{name}_first']} (margin {row[f'{name}_margin']:.2f})")
    axs[0].set_xlabel("candidate first slot q (pair q-1 -> q, cyclic; q=0 is the wrap pair)")
    axs[0].set_ylabel("neighbour correlation")
    axs[0].legend(fontsize=8)
    axs[0].set_title(row["folder"], fontsize=9)
    if b1 is not None:
        for s, sc in b1.items():
            axs[1].plot(x, sc, "-" if s == 0 else ":", label=f"shift {s:+d}")
        axs[1].axvline(row["b1_first"], color="k", lw=0.8)
        axs[1].set_title(f"buffer-1 similarity: first={row['b1_first']} shift {row['b1_shift']:+d} "
                         f"(margin {row['b1_margin']:.3f}); span/RR {row.get('span_over_rr')}", fontsize=9)
        axs[1].legend(fontsize=8)
    else:
        axs[1].text(0.5, 0.5, f"no buffer-1 method\nECG: {row.get('ecg')}", ha="center", va="center")
    axs[1].set_xlabel("candidate first slot")
    fig.tight_layout()
    fig.savefig(fig_dir / (row["folder"].replace("/", "_") + ".png"), dpi=80)
    plt.close(fig)


# ------------------------------------------------------------------ sample + main
def sample(per_group, root="Z:/raw_data"):
    """Stratified: per_group folders each with 26 and 32 buffer-3 frames (trustworthy ECG), plus a
    few with an untrustworthy ECG, spread evenly over the ready folders."""
    from swp.manual.frames import n_frames
    ready = [f for f in S.find_folders(root) if S.ready(S.Paths(f))
             and (Path(S.Paths(f).output) / "converted" / "CombinedData_buffer3.hdf5").exists()]
    groups = {26: [], 32: [], "bad_ecg": []}
    for f in ready:
        try:
            nf = n_frames(S.Paths(f).bmode(3))
            ok = rrcheck().assess_rr(f).trustworthy
        except Exception:                                              # noqa: BLE001
            continue
        if not ok:
            groups["bad_ecg"].append(f)
        elif nf in groups:
            groups[nf].append(f)
    out = []
    for key, k in ((26, per_group), (32, per_group), ("bad_ecg", max(4, per_group // 4))):
        g = groups[key]
        if g:
            idx = np.unique(np.linspace(0, len(g) - 1, min(k, len(g))).round().astype(int))
            out += [g[i] for i in idx]
    print(f"ready {len(ready)}: 26-frame {len(groups[26])}, 32-frame {len(groups[32])}, "
          f"untrustworthy ECG {len(groups['bad_ecg'])} -> sample {len(out)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=24)
    ap.add_argument("--folder", action="append", default=[])
    a = ap.parse_args()
    folders = a.folder or (sample(a.per_group) + [CHECK_FOLDER])
    fig_dir = _REPO / "study" / "montages" / "buffer3_unwrap"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    out = _REPO / "study" / "logs" / "buffer3_unwrap_validation.csv"
    for k, f in enumerate(folders):
        try:
            r = analyse(f, fig_dir)
        except Exception:                                              # noqa: BLE001
            print(f"[{k + 1}/{len(folders)}] {f}: FAILED\n{traceback.format_exc()}", flush=True)
            continue
        rows.append(r)
        print(f"[{k + 1}/{len(folders)}] {r['folder'][:40]:40s} N={r['n_frames']} "
              f"rf {r['rf_first']:2d} ({r['rf_margin']:.2f}) iq {r['iq_first']:2d} ({r['iq_margin']:.2f}) "
              f"b1 {r.get('b1_first', '-')!s:>2} ({r.get('b1_margin', float('nan')):.3f}, "
              f"shift {r.get('b1_shift', '-')}) span/RR {r.get('span_over_rr', '-')} | "
              f"t rf {r['t_rf_s']} iq {r['t_iq_s']} b1 {r.get('t_b1_s', '-')} s", flush=True)
        keys = list(dict.fromkeys(k2 for rr in rows for k2 in rr))
        with open(out, "w", newline="") as fh:                     # rewritten after every folder
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
