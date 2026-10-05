"""Re-run the whole evaluation of the manual passive reading on a new (or given) snapshot.

    python run_all.py                       # new snapshot of Z:/raw_data, then every analysis
    python run_all.py --snapshot 20261005_1530    # re-run the analyses on an existing snapshot
    python run_all.py --skip 03b,04         # leave out slow steps (03b ~25 min, 04 ~5 min, 03 ~3 min)

Every step reads only the snapshot (and, for 03 / 03b / 04 / 07 --sheets, the B-mode / IQ files on
Z: read-only). Steps that cache their heavy part (04 measurements.csv, 03 raw.json, 06
estimates.csv) skip it when the cache exists: delete the cache to recompute.
Use the envs/zea_latest python.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [("00", ["00_snapshot.py"]), ("01", ["01_features.py"]), ("02", ["02_reproducibility.py", "--boot", "200"]),
         ("03", ["03_mline.py"]), ("03b", ["03b_mline_sensitivity.py"]), ("04", ["04_septum.py"]),
         ("05", ["05_events.py"]), ("06", ["06_slopes.py"]), ("07", ["07_quality.py", "--sheets"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None, help="existing snapshot: skip step 00")
    ap.add_argument("--skip", default="", help="comma-separated step ids to skip")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    if a.snapshot:
        skip.add("00")
    log = HERE / "results" / f"run_all_{time.strftime('%Y%m%d_%H%M')}.log"
    log.parent.mkdir(exist_ok=True)
    for sid, cmd in STEPS:
        if sid in skip:
            continue
        args = [sys.executable, "-W", "ignore", str(HERE / cmd[0])] + cmd[1:]
        if a.snapshot and sid != "00":
            args += ["--snapshot", a.snapshot]
        t0 = time.time()
        print(f"[{sid}] {' '.join(cmd)}", flush=True)
        with open(log, "a") as fh:
            fh.write(f"\n===== [{sid}] {' '.join(cmd)}  {time.ctime()}\n")
            fh.flush()
            r = subprocess.run(args, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        print(f"     exit {r.returncode}, {time.time() - t0:.0f} s", flush=True)
        if r.returncode:
            raise SystemExit(f"step {sid} failed, see {log}")
    print(f"done; log {log}")


if __name__ == "__main__":
    main()
