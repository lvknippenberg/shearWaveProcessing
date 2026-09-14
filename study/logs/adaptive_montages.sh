#!/bin/bash
cd "D:/Luuk van Knippenberg/Github/shearWaveProcessing"
M="D:/Luuk van Knippenberg/Claude/SWE_study_processing/montages"
C="D:/Luuk van Knippenberg/Claude/SWE_study_processing/analysis/tilecache"
PY="D:/Luuk van Knippenberg/envs/zea_latest/python.exe"
mkdir -p "$C"
"$PY" scripts/shared_norm_montage.py -o "$M/all_buffer1_adaptive.gif" --cols 8 --tile-width 170 \
  --auto --cache "$C/buffer1.npz" \
  --title "buffer 1 widebeam - ADAPTIVE display (per-set white point + range)" \
  --glob "Z:/raw_data/C*/*/output/CombinedData_buffer1_iq.hdf5"
"$PY" scripts/shared_norm_montage.py -o "$M/all_buffer3_standard_adaptive.gif" --cols 8 --tile-width 170 \
  --auto --cache "$C/buffer3_standard.npz" \
  --title "buffer 3 STANDARD - ADAPTIVE display (per-set white point + range)" \
  --glob "Z:/raw_data/C*/*/output/CombinedData_buffer3_iq.hdf5"
"$PY" scripts/shared_norm_montage.py -o "$M/all_buffer3_refocus_adaptive.gif" --cols 8 --tile-width 170 \
  --auto --cache "$C/buffer3_refocus.npz" \
  --title "buffer 3 REFoCUS adjoint - ADAPTIVE display (per-set white point + range)" \
  --glob "Z:/raw_data/C*/*/output/CombinedData_buffer3_refocus-adjoint_iq.hdf5"
echo "ADAPTIVE_EXIT=0"
