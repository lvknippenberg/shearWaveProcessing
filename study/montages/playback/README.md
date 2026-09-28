# Buffer 1 / buffer 3 playback montages (2026-09-28)

One tile per subject (48; C000000006 has no data), for the 1st (`acq1_*`) and 2nd (`acq2_*`) SW
acquisition of each subject (order = timestamp in the folder name). Made by
`study/analysis/buffer_playback_montage.py` and `buffer_playback_pair.py`.

**Plain playback, not synchronisation.** Every tile plays all of its frames once, in file order,
starting together; a shorter clip holds its last frame (grey label) until the GIF loops. Buffers 1
and 3 are not ECG-triggered, so tiles are deliberately not aligned to the R-peak or to each other.
The unwrap's purpose is continuous playback of buffer 3 without the wrap jump.

| file | content |
|---|---|
| `acqN_buffer1_stored.gif` | buffer 1 exactly as stored (never modified); 90-106 frames, 20 ms/frame (~0.57x real time) |
| `acqN_buffer3_stored.gif` | buffer 3 (REFoCUS adjoint) in the ORIGINAL stored slot order - before the unwrap; the label gives the slot after which the newest->oldest jump sits |
| `acqN_buffer3_unwrapped.gif` | buffer 3 after the VERSION-2 unwrap (chronological); label = method (trigger / combined / continuity); red = ambiguous, still stored order |
| `acqN_buffer3_stored_vs_unwrapped.gif` | the two paired per subject: [stored \| unwrapped] |
| `*_first.png` | frame 0 of every tile |

Buffer 3: 26 or 32 frames at 40 ms/frame (~real time). The stored order is rebuilt from the head the
unwrap recorded (`custom/unwrap_first_frame`).

Study-wide VERSION-2 result and the continuity check (the wrap jump leaves the inside of the playback
in 435 of 450 resolved folders that had one): `docs/buffer3_unwrap.md`, last section.

Not replaced: the older `study/montages/all_buffer3_{standard,refocus}_*` montages start every tile
at stored slot 0 (pre-unwrap, 2026-09-14) and are kept as a record of that state.
