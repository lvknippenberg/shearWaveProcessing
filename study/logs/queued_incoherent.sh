#!/bin/bash
# Wait for the batch to finish, then run the incoherent-compound example on C000000001 buffer 3.
S="C:/Users/s126497/AppData/Local/Temp/4/claude/d--Luuk-van-Knippenberg-Onedrive-folder-OneDrive---TU-Eindhoven-Shear-wave-elastography/c55c275b-4f8c-4106-bb01-0c920318fbea/scratchpad"
until grep -q "BATCH_EXIT=" "$S/batch.log" 2>/dev/null; do sleep 60; done
echo "batch finished; starting incoherent example"
cd "D:/Luuk van Knippenberg/Github/shearWaveProcessing"
KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
  "D:/Luuk van Knippenberg/envs/zea_latest/python.exe" scripts/incoherent_bmode.py \
  "Z:/raw_data/C000000001/SWE_01_SW_data_21-April-2026_12-12-54" --buffer 3
echo "INCOH_EXIT=$?"
