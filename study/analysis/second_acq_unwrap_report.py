"""Merge the 2026-09-28 buffer-3 unwrap of the 2nd SW acquisition and compare with the 1st.

Inputs: study/logs/buffer3_unwrap_second_acq_w*.csv (this run, 4 workers) and
study/logs/buffer3_unwrap_september_20260925.csv (the 1st acquisition of each subject).
Question: are the 2nd acquisitions resolved by the exact trigger count more often, because the live
focused loop before them is shorter and the circular trigger log still holds its start?

    python study/analysis/second_acq_unwrap_report.py
-> study/logs/buffer3_unwrap_second_acq_20260928.csv (merged) + first_vs_second table (stdout, .csv)
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
LOGS = _REPO / "study" / "logs"


def label(r):
    if r.status in ("unwrapped", "chronological"):
        return "trigger count" if r.method == "trigger-count" else "buffer-1 similarity"
    return f"ambiguous ({r.reason})" if isinstance(r.reason, str) and r.reason else r.status


def main():
    sec = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(LOGS / "buffer3_unwrap_second_acq_w*.csv")))])
    sec = sec.sort_values("folder").reset_index(drop=True)
    sec.to_csv(LOGS / "buffer3_unwrap_second_acq_20260928.csv", index=False)
    first = pd.read_csv(LOGS / "buffer3_unwrap_september_20260925.csv")
    for d in (sec, first):
        d["subject"] = d.folder.str.split("/").str[0]
        d["result"] = d.apply(label, axis=1)
    m = first.merge(sec, on="subject", how="outer", suffixes=("_1st", "_2nd"))
    cols = ["subject", "n_frames_2nd", "result_1st", "trigger_run_length_1st", "result_2nd",
            "trigger_run_length_2nd", "first_2nd", "buffer1_margin_2nd", "buffer1_agrees_2nd",
            "ecg_trustworthy_2nd"]
    m = m[[c for c in cols if c in m]].sort_values("subject")
    m.to_csv(LOGS / "buffer3_unwrap_first_vs_second_20260928.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60)
    print(m.to_string(index=False))

    def cat(s):
        return s.str.replace(r" \(.*", "", regex=True)
    print("\n1st acquisition:", cat(first.result).value_counts().to_dict())
    print("2nd acquisition:", cat(sec.result).value_counts().to_dict())
    both = m.dropna(subset=["result_1st", "result_2nd"])
    tc1 = cat(both.result_1st).eq("trigger count")
    tc2 = cat(both.result_2nd).eq("trigger count")
    print(f"paired subjects {len(both)}: trigger count 1st {tc1.sum()}, 2nd {tc2.sum()}; "
          f"gained {(tc2 & ~tc1).sum()}, lost {(tc1 & ~tc2).sum()}")
    for tag, d in (("1st", first), ("2nd", sec)):
        n = pd.to_numeric(d.get("trigger_run_length"), errors="coerce").dropna()
        print(f"{tag}: live-run triggers where countable: n={len(n)}, median {n.median():.0f}, max {n.max():.0f}")
    x = sec.dropna(subset=["buffer1_agrees"])
    print(f"2nd: buffer-1 cross-check where both ran: agrees {int(x.buffer1_agrees.astype(bool).sum())}/{len(x)}")


if __name__ == "__main__":
    main()
