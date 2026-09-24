"""Re-run buffer 1 over the 44 processed study folders, N workers on N GPUs.

``scripts/process_raw_data.py`` is sequential, and buffer 1 costs ~7 min per folder (dominated
by the 919 MB read from the Z: share), so 44 folders is ~5 h. This splits the list round-robin
across workers pinned to separate GPUs with ``CUDA_VISIBLE_DEVICES``, each running the normal
batch script on a disjoint set - no shared state, so a worker failing costs only its own folders.

    python rerun44.py --workers 3 [--dry-run]

Per-worker logs go to ``logs/rerun_w<N>.log``; the summary at the end of each is the batch
script's own per-folder success/failure report.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
LIST = HERE / "folders44.txt"
LOGS = HERE / "logs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="1,2",
                   help="GPU indices to pin workers to, one worker each. This is a SHARED "
                        "machine - check `nvidia-smi` and pick the idle ones.")
    p.add_argument("--patch-budget", default="3.0e7",
                   help="ZEA_SWI_PATCH_BUDGET (gathered elements). The default sizes patches "
                        "to the GPU's TOTAL memory, which is wrong when someone else already "
                        "holds half of it; this caps our footprint. Lower = more, smaller "
                        "kernel launches.")
    p.add_argument("--buffers", default="1")
    p.add_argument("--skip", default="", help="comma-separated folder names already done")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    folders = [ln.strip() for ln in LIST.read_text().splitlines() if ln.strip()]
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    folders = [f for f in folders if not any(s and s in f for s in skip)]
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()]
    print(f"{len(folders)} folders, {len(gpus)} workers on GPU(s) {gpus}, "
          f"patch budget {a.patch_budget}")
    if not folders:
        return 0

    LOGS.mkdir(exist_ok=True)
    groups = [folders[i::len(gpus)] for i in range(len(gpus))]
    procs = []
    for w, (gpu, group) in enumerate(zip(gpus, groups)):
        if not group:
            continue
        cmd = [str(PYTHON), str(REPO / "scripts" / "process_raw_data.py"),
               "--buffers", a.buffers, "--redo"]
        for f in group:
            cmd += ["--folder", f]
        print(f"  worker {w} (GPU {gpu}): {len(group)} folders -> logs/rerun_w{w}.log")
        if a.dry_run:
            continue
        env = dict(os.environ, KERAS_BACKEND="torch", CUDA_VISIBLE_DEVICES=gpu,
                   ZEA_SWI_PATCH_BUDGET=a.patch_budget,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        log = open(LOGS / f"rerun_w{w}.log", "w", encoding="utf8", errors="replace")
        procs.append((w, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          cwd=str(REPO)), log))
    if a.dry_run:
        return 0

    t0 = time.perf_counter()
    rc = 0
    for w, proc, log in procs:
        code = proc.wait()
        log.close()
        rc |= code
        print(f"  worker {w} exited {code} after {(time.perf_counter() - t0) / 60:.0f} min")
    print(f"all workers done in {(time.perf_counter() - t0) / 60:.0f} min, rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
