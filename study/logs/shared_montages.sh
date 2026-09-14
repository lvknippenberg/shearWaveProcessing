#!/bin/bash
cd "D:/Luuk van Knippenberg/Github/shearWaveProcessing"
M="D:/Luuk van Knippenberg/Claude/SWE_study_processing/montages"
PY="D:/Luuk van Knippenberg/envs/zea_latest/python.exe"
REF=43385            # median buffer-1 p99.9 envelope -> the common scale
REF_RF=5423125       # REF * 125: removes REFoCUS's +41.9 dB decode gain
DR=70                # dB display range, so buffer 3 (-28 dB) stays visible
echo "=== shared reference $REF (REFoCUS $REF_RF), dynamic range ${DR} dB ==="
"$PY" scripts/shared_norm_montage.py -o "$M/all_buffer1_shared.gif" --cols 8 --tile-width 170 \
  --reference $REF --dynamic-range $DR \
  --title "buffer 1 widebeam - SHARED normalisation (ref $REF, ${DR} dB)" \
  --glob "Z:/raw_data/C*/*/output/CombinedData_buffer1_iq.hdf5"
"$PY" scripts/shared_norm_montage.py -o "$M/all_buffer3_standard_shared.gif" --cols 8 --tile-width 170 \
  --reference $REF --dynamic-range $DR \
  --title "buffer 3 STANDARD - SHARED normalisation (same ref as buffer 1)" \
  --glob "Z:/raw_data/C*/*/output/CombinedData_buffer3_iq.hdf5"
"$PY" scripts/shared_norm_montage.py -o "$M/all_buffer3_refocus_shared.gif" --cols 8 --tile-width 170 \
  --reference $REF_RF --dynamic-range $DR \
  --title "buffer 3 REFoCUS adjoint - SHARED normalisation (decode gain removed)" \
  --glob "Z:/raw_data/C*/*/output/CombinedData_buffer3_refocus-adjoint_iq.hdf5"
echo "SHARED_EXIT=0"
