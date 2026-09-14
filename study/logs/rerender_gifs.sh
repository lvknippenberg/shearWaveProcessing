#!/bin/bash
# Re-render every GIF with the new display (adaptive levels + gamma2, +0 dB).
# GIF-only: reads the existing IQ, no re-beamforming, nothing else is touched.
cd "D:/Luuk van Knippenberg/Github/shearWaveProcessing"
PY="D:/Luuk van Knippenberg/envs/zea_latest/python.exe"
n=0
for f in "Z:/raw_data"/C*/*/output/ ; do
  [ -d "$f" ] || continue
  n=$((n+1))
  echo "########## [$n] $f"
  KERAS_BACKEND=torch PYTHONUNBUFFERED=1 "$PY" -m swp.acquisition.gifs "$f" 2>&1 \
    | grep -E "=== GIFs|B-mode GIF" | tail -6
done
echo "RERENDER DONE: $n folder(s)"
echo "RERENDER_EXIT=0"
