#!/bin/bash
# 2026-09-28: buffer-3 unwrap VERSION 2 (parsed trigger log + combined estimator) on all SW folders,
# 4 workers from Windows (server unreachable: SSH host key). v1 folders are re-estimated and re-rotated.
cd "/d/Luuk van Knippenberg/Github/shearWaveProcessing"
PY="/d/Luuk van Knippenberg/envs/zea_latest/python.exe"
for w in 0 1 2 3; do
  args=()
  while IFS= read -r f || [ -n "$f" ]; do
    f="${f%$'\r'}"
    [ -n "$f" ] && args+=(--folder "$f")
  done < study/logs/_unwrap_v2_w$w.txt
  "$PY" -W ignore scripts/unwrap_buffer3.py "${args[@]}" --log study/logs/buffer3_unwrap_v2_w$w.csv \
     > study/logs/buffer3_unwrap_v2_w$w.log 2>&1 &
done
wait
echo ALL DONE
