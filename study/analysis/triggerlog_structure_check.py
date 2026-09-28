"""Check that every trigger log parses into the known acquisition sequence (2026-09-28). Read-only.

Sequence (SetUp_SWI_Widebeam.m), one frame trigger per frame / per SW event:
    [buffer-3 focused live loop, 1000/fps3 ms] -> buffer 1 (n1 @ 1000/fps1) -> wait R ->
    buffer 4 (n4 @ 1000/fps4) -> active SW (Nsw pushes x k triggers: reference, push, tracking, B-mode)
Parse backwards from the unmistakable buffer-4 block: buffer 1 = the n1 triggers before it, the live
loop = everything before buffer 1 that keeps the loop cadence. Compares the loop run with the old
period-based segmentation (triggerlog._runs) and reports where they differ.

    python study/analysis/triggerlog_structure_check.py -> study/logs/triggerlog_structure_20260928.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))
os.environ.setdefault("KERAS_BACKEND", "torch")


def check(folder):
    from swp.acquisition.triggerlog import _runs, read_log
    log = read_log(folder)
    if log is None:
        return dict(status="no log")
    r, trig, p = log
    m = sio.loadmat(str(Path(folder) / "AcquisitionParametersAndECG.mat"), squeeze_me=True,
                    struct_as_record=False, variable_names=["SW"])
    nsw = int(m["SW"].Nframes) if "SW" in m else -1
    n1, n3, n4 = p[1]["n"], p[3]["n"], p[4]["n"]
    f1, f3, f4 = (1000 / p[b]["fps"] for b in (1, 3, 4))
    d = np.diff(trig)
    # buffer 4: n4 triggers whose intervals are all far below the buffer-1 period (timestamps may be
    # whole ms in early campaigns -> 1/2 ms steps; occasional hiccups) and whose span matches n4
    fast = d < 0.25 * f1
    edges = np.flatnonzero(np.diff(np.r_[0, fast.astype(int), 0]))
    runs4 = [(s, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= n4 - 1]
    runs4 = [(s, s + n4 - 1) for s, e in runs4 if abs(trig[s + n4 - 1] - trig[s] - (n4 - 1) * f4) < 0.02 * n4 * f4]
    if len(runs4) != 1:
        return dict(status=f"buffer-4 blocks {len(runs4)}")
    b4s, b4e = runs4[0]
    b1s = b4s - n1
    if b1s < 1:
        return dict(status="buffer 1 not in log")
    d1 = d[b1s:b1s + n1 - 1]
    b1_ok = bool(np.all(np.abs(d1 - f1) < 0.15 * f1))
    after = len(trig) - 1 - b4e
    k_sw = after / nsw if nsw > 0 else np.nan
    # live loop: walk back from b1s - 1 while the cadence holds
    tol3 = max(1.5, 0.08 * f3)
    e = b1s - 1
    s = e
    while s > 0 and abs(d[s - 1] - f3) < tol3:
        s -= 1
    L = e - s + 1
    truncated = s == 0
    gap_loop_b1 = float(trig[b1s] - trig[e])
    # old segmentation, as used before 2026-09-28
    runs3 = [(i, j) for i, j in _runs(trig, f3, tol3) if j - i + 1 >= n3]
    oi, oj = runs3[-1] if runs3 else (-1, -1)
    return dict(status="ok" if b1_ok and L >= n3 else ("buffer-1 cadence bad" if not b1_ok else "loop < n3"),
                n_log=len(trig), n1=n1, n3=n3, n4=n4, nsw=nsw, sw_triggers=after, sw_trig_per_push=k_sw,
                loop_len=L, loop_truncated=truncated, loop_start=s, loop_end=e, b1_start=b1s,
                gap_loop_to_b1_ms=round(gap_loop_b1, 2),
                gap_before_loop_ms=round(float(trig[s] - trig[s - 1]), 1) if s > 0 else np.nan,
                old_end=oj, old_len=oj - oi + 1 if oj >= 0 else -1, old_truncated=oi == 0,
                old_end_minus_new=oj - e if oj >= 0 else np.nan,
                head_new=(L - 1) % n3 if not truncated else -1,
                head_old=((oj - oi) % n3) if oj >= 0 and oi > 0 else -1)


def main():
    from process_raw_data import find_measurement_folders
    rows = []
    for f in find_measurement_folders("Z:/raw_data"):
        if "_sw_data_" not in f.name.lower():
            continue
        try:
            row = check(str(f))
        except Exception as exc:                                     # noqa: BLE001
            row = dict(status=f"error {type(exc).__name__}: {exc}")
        rows.append(dict(folder=f"{f.parent.name}/{f.name}", **row))
    df = pd.DataFrame(rows)
    df.to_csv(_REPO / "study" / "logs" / "triggerlog_structure_20260928.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.status.value_counts().to_string())
    ok = df[df.status == "ok"]
    print("\nlog length:", ok.n_log.value_counts().to_dict())
    print("SW triggers per push:", ok.sw_trig_per_push.value_counts().to_dict())
    print("gap loop end -> buffer 1 (ms):", ok.gap_loop_to_b1_ms.describe().round(1).to_dict())
    print("old run end - new run end:", ok.old_end_minus_new.value_counts().to_dict())
    print("loop start in log (countable):", int((~ok.loop_truncated.astype(bool)).sum()), "of", len(ok),
          "| old:", int((~ok.old_truncated.astype(bool)).sum()))
    c = ok[~ok.loop_truncated.astype(bool)]
    print("gap before loop start (ms):", c.gap_before_loop_ms.describe().round(0).to_dict())
    print("head new vs old where both countable:",
          (c.head_new - c.head_old).value_counts().to_dict())


if __name__ == "__main__":
    main()
