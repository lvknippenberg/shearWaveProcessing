"""How much of the circular trigger log is left for the buffer-3 live run (2026-09-28).

Per folder: log length, triggers AFTER the live run (the SW acquisition itself), the live-run length
seen in the log, and whether the run start survived (countable). Countable iff the live run is
shorter than (log capacity - triggers after it).
    python study/analysis/trigger_log_budget.py -> study/logs/trigger_log_budget_20260928.csv
"""
import csv, sys
from pathlib import Path
import pandas as pd
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
from swp.acquisition.triggerlog import _runs, read_log  # noqa: E402

ROOT = Path("Z:/raw_data")
inv = list(csv.DictReader(open(_REPO / "study/logs/second_acq_inventory.csv")))
first = pd.read_csv(_REPO / "study/logs/buffer3_unwrap_september_20260925.csv").folder.tolist()
first.append("C000000001/SWE_01_SW_data_21-April-2026_12-12-54")
jobs = [("1st", ROOT / f) for f in first] + [("2nd", ROOT / r["subject"] / r["second"]) for r in inv if r.get("second")]
rows = []
for tag, f in jobs:
    log = read_log(str(f))
    if log is None or 3 not in log[2]:
        rows.append(dict(acq=tag, subject=f.parent.name, note="no log")); continue
    r, trig, p = log
    n, fms = p[3]["n"], 1000.0 / p[3]["fps"]
    runs = [(i, j) for i, j in _runs(trig, fms, max(1.5, 0.08 * fms)) if j - i + 1 >= n]
    i, j = runs[-1] if runs else (None, None)
    rows.append(dict(acq=tag, subject=f.parent.name, n_frames=n, log_len=len(trig),
                     run_seen=(j - i + 1) if runs else None, after_run=(len(trig) - 1 - j) if runs else None,
                     countable=bool(runs and i > 0), run_s=round((j - i + 1) * fms / 1000, 1) if runs else None))
d = pd.DataFrame(rows)
d.to_csv(_REPO / "study/logs/trigger_log_budget_20260928.csv", index=False)
pd.set_option("display.width", 200)
print(d.groupby(["acq", "n_frames"]).agg(n=("subject", "size"), countable=("countable", "sum"),
      log_len=("log_len", "median"), after_run=("after_run", "median"),
      run_seen_med=("run_seen", "median"), run_s_med=("run_s", "median")).to_string())
for tag in ("1st", "2nd"):
    x = d[(d.acq == tag) & d.countable]; y = d[(d.acq == tag) & ~d.countable.astype(bool)]
    print(f"{tag}: countable live runs {x.run_s.median()} s median (n={len(x)}); "
          f"truncated runs >= {y.run_s.median()} s median seen (n={len(y)})")
