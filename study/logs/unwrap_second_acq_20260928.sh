#!/bin/bash
# 2026-09-28: unwrap buffer 3 of the 2nd SW acquisition per subject, 4 workers (Windows, NAS-bound)
cd "/d/Luuk van Knippenberg/Github/shearWaveProcessing"
PY="/d/Luuk van Knippenberg/envs/zea_latest/python.exe"
for w in 0 1 2 3; do
  args=()
  while IFS= read -r f || [ -n "$f" ]; do
    f="${f%$'\r'}"                       # lists written by Windows python are CRLF
    args+=(--folder "$f")
  done < study/logs/_unwrap_second_w$w.txt
  "$PY" scripts/unwrap_buffer3.py "${args[@]}" --log study/logs/buffer3_unwrap_second_acq_w$w.csv \
     > study/logs/buffer3_unwrap_second_acq_w$w.log 2>&1 &
done
wait
echo ALL DONE
