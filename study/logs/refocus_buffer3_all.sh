#!/bin/bash
# REFoCUS adjoint on buffer 3 for every measurement folder under Z:\raw_data.
# Waits for the in-flight buffer-1 job so the two never share GPU 2.
S="D:/Luuk van Knippenberg/Claude/SWE_study_processing"
until grep -qE "ripple power|Traceback" "$S/logs/refocus_buffer1.log" 2>/dev/null; do sleep 20; done
echo "buffer-1 job clear; starting buffer-3 REFoCUS sweep"
cd "D:/Luuk van Knippenberg/Github/shearWaveProcessing"
n=0; ok=0; fail=0
for f in "Z:/raw_data"/C*/*/ ; do
  [ -f "$f/CombinedData.mat" ] || continue
  n=$((n+1))
  echo ""; echo "########## [$n] $f ##########"
  KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
    "D:/Luuk van Knippenberg/envs/zea_latest/python.exe" scripts/refocus_bmode.py \
    "$f" --buffer 3 --method adjoint 2>&1 | grep -E "buffer 3 \(|encoding matrix|REFoCUS\[|ripple power|wrote|Traceback|Error"
  if [ -f "$f/output/CombinedData_buffer3_refocus-adjoint_iq.gif" ]; then ok=$((ok+1)); else fail=$((fail+1)); echo "FAILED: $f"; fi
done
echo ""; echo "SWEEP DONE: $ok ok, $fail failed, $n folders"
echo "REFOCUS3_EXIT=0"
