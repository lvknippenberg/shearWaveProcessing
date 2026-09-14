#!/bin/bash
# Runs after the batch AND after the incoherent example, so nothing contends for GPU 2.
S="C:/Users/s126497/AppData/Local/Temp/4/claude/d--Luuk-van-Knippenberg-Onedrive-folder-OneDrive---TU-Eindhoven-Shear-wave-elastography/c55c275b-4f8c-4106-bb01-0c920318fbea/scratchpad"
until grep -q "BATCH_EXIT=" "$S/batch.log" 2>/dev/null; do sleep 60; done
until grep -q "INCOH_EXIT=" "$S/incoherent.log" 2>/dev/null; do sleep 30; done
echo "=== batch + incoherent done; starting REFoCUS sweep ==="
cd "D:/Luuk van Knippenberg/Github/shearWaveProcessing"
F="Z:/raw_data/C000000001/SWE_01_SW_data_21-April-2026_12-12-54"
for M in adjoint tikhonov tsvd; do
  echo ""; echo "########## REFoCUS method=$M ##########"
  KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
    "D:/Luuk van Knippenberg/envs/zea_latest/python.exe" scripts/refocus_bmode.py \
    "$F" --buffer 3 --method "$M"
  echo "method=$M exit=$?"
done
echo "REFOCUS_EXIT=0"
