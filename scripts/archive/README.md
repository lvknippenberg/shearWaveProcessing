# scripts/archive — finished campaigns

These scripts produced results that are recorded in `docs/` and `report/`. They are kept so those
results can be reproduced, but they are **not** part of the maintained toolset in `scripts/` and
nothing in `src/` depends on them. Moved here on 2026-09-24 (`review-followup` branch).

They still run from here: each one puts both `scripts/archive/` and `scripts/` on `sys.path`, so
imports between them (e.g. `sweep_invivo40 -> sweep_extract -> detect_v`) and of the shared
libraries left in `scripts/` (`swe_lib`, `detect_v`) keep working. Data locations come from
`swp.paths` (override with `SWP_DATA_ROOT` etc., see that module), not hard-coded strings — the
2026-08-04 voltage sweep has since moved under `Claude/MI estimation/`, which had broken every
one of them.

| campaign | scripts | written up in |
|---|---|---|
| metric validation (blind scoring, 2026-08-06) | `metric_experiment_{generate,handpicked,analyze,patterns}`, `metric_build`, `metric_crossdataset`, `pairwise_analyze`, `check_options`, `draw_v_roi` | `docs/HANDOFF.md` §0b |
| OAT ablation (scored by the retired `push_specificity`) | `ablation` | `docs/HANDOFF.md` §0b — **do not reuse its ranking** |
| low-SNR extraction sweep | `sweep_extract`, `sweep_analyze`, `sweep_top_montage`, `sweep_rfncc_probe` | `docs/low_snr_extraction_sweep.md` |
| phantom acquisition-parameter sweep | `sweep_params`, `phantom_pulse_variance`, `push_index_trend`, `aperture_trend` | `docs/phantom_parameter_sweep.md`, `report/acquisition_sweep_log.tex` |
| in-vivo 40 V / Caenen speed-scan sweeps | `sweep_invivo40`, `invivo40_analyze`, `sweep_caenen`, `caenen_analyze`, `caenen_methods` | `docs/HANDOFF.md` §0b |
| in-vivo motion removal (rel-ref displacement only) | `motion_removal` | `docs/HANDOFF.md` §0b; superseded by `scripts/invivo_recipe_contrast.py` |
| 2026-08-17/18 acquisition campaign (tasks 1-5) | `task1_repro_voltage`, `task2_element_cycle_grid`, `task2b_iso_safety`, `task3_probe_compare`, `task4_invivo_compare`, `task4b_config_contrast`, `task5_cross_beat` | `report/acquisition_sweep_log.tex` |
| passive parameter searches (tuned on `invivo_sw` only) | `search_passive`, `search_passive2`, `passive_best_montage`, `passive_compare_windows`, `passive_directional_test`, `passive_filter_variety`, `passive_slantstack` | `docs/passive_search.md` |
| 2026-08-06 no-push diagnostics | `diagnostics/{arf_diagnostic,nopush_control,pushvsnopush_all,invivo_voltage_compare}` | `docs/HANDOFF.md` §0b |
