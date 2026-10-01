# Manual passive reading with the old (energy) detector - reference, 2026-10-01

The 27 folders read 2026-09-25 to 09-29 (C000000001-31) with the energy detector
(`detect.picker: energy`, 100 ms windows, docs/passive_manual_prelim_2026-09-29.md). They are
re-read with the valves detector and the window review from 2026-10-01 on
(`passive_manual.py redetect`). This is the state just before that.

| what | where |
|---|---|
| slopes (97 windows, all with a slope), as `passive_manual.py export` | `passive_manual_slopes.csv` |
| folder states before the redetect | `status_before_redetect.txt` |
| the 27 folders | `folders.txt` |
| every JSON / JSONL record per folder (general line, windows, events, processed, slopes, rois, log) | `folders/<subject>/<folder>/` |
| full copy incl. line npz, space-times `st_win<i>.npz` and snapshot PNGs (712 files, 176 MB, SHA-1 verified) | `D:\Luuk van Knippenberg\Claude\passive_manual_reference_2026-10-01_energy\` (local disk) |
| the same files on the share | `<folder>/output/swp_passive_manual/archive_<time>_redetect_valves/` (moved there by `redetect`, never deleted) |

The evaluation of these results: docs/passive_manual_prelim_2026-09-29.md, tables in
`study/logs/passive_manual_prelim/`.
