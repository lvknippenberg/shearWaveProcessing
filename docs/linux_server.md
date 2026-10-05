# Running stages 1-2 on the Linux GPU server (bmdserver3)

Conversion + beamforming (`scripts/process_raw_data.py`, `run.py beamform`) run on the Linux server
inside a Docker container, detached from any laptop/VS Code session. Validated 2026-09-17 against the
Windows reference on C000000001: **torch bit-identical, JAX equal to float precision**, JAX 6.1 min vs
Windows 10.0 min per folder (details in [Validation result](#validation-result-2026-09-17)).

> **The zea version matters more than the backend.** The first server run used an older zea
> (`44208e0b`) and its IQ differed badly from Windows (buffer-4 displacement proxy correlation 0.21,
> depth-dependent phase errors up to 57°). With the same zea as Windows (`8c2699fd`) the difference
> vanished. Always build the image from the zea commit the reference was made with, and re-validate
> after any zea change.

## Paths

| | Windows | server host | inside the container |
|---|---|---|---|
| NAS share `VISUALIZE` | `Z:\` | `~/mounts/VISUALIZE` | `/mnt/z/VISUALIZE` |
| study data | `Z:\raw_data` | `~/mounts/VISUALIZE/raw_data` | `/mnt/z/VISUALIZE/raw_data` |
| this repo (clone on the NAS) | `Z:\shearWaveProcessing` | `~/mounts/VISUALIZE/shearWaveProcessing` | `/mnt/z/VISUALIZE/shearWaveProcessing` |
| zea source (fork, `8c2699fd`) | `D:\...\Github\zea` | `~/zea-8c2699fd` | `/zea` |

Keeping the repo on the NAS means logs and outputs are readable from Windows directly.

## What still needs Windows

`CombinedData.mat` is built by a MATLAB merge (`make_combined_data.m`); the server has no MATLAB.
For any folder that only has `AcquisitionParametersAndECG.mat`, build it on Windows first (no GPU,
no beamforming):

```
set KERAS_BACKEND=torch
python scripts/build_combined_data.py --root "Z:\raw_data" --dry-run      # what is missing
python scripts/build_combined_data.py --root "Z:\raw_data" --jobs 4       # build all missing (resumable)
python scripts/build_combined_data.py --folder "Z:\raw_data\C0000000xx\<folder>"
python scripts/process_raw_data.py --root "Z:\raw_data" --check      # audit (needs the Windows base-config dir)
```

`build_combined_data.py` skips folders that already have the file and keeps going past failures;
~1.3-1.8 min/folder with 4 jobs, ~535 MB written per folder. **With `--jobs > 1` its log is not
reliable:** worker threads redirect stdout (`contextlib.redirect_stdout` is process-wide, and
overlapping redirects restore in the wrong order), so per-folder progress lines and even the closing
summary can vanish. Judge a run by the process ending and by counting `CombinedData.mat` files;
find failures by listing folders that still lack one.

**Do not beamform a folder while its `CombinedData.mat` is still being built.** MATLAB copies the
base config first and appends the runtime parameters afterwards, so the file exists before it is
complete. Start the server batch only after the Windows build process has exited.

On the server an existing `CombinedData.mat` is only validated (pure Python; `repair_buffer2_receive`
may open it `r+` but only writes when a repair is needed).

## One-time setup (from scratch)

### 1. SSH and tmux

```bash
ssh luuk@bmdserver3
tmux new -s setup          # anything long runs in tmux; detach Ctrl-b d, reattach: tmux attach -t setup
```

### 2. zea source at the validated commit

```bash
git clone git@github.com:lvknippenberg/zea.git ~/zea-8c2699fd
cd ~/zea-8c2699fd && git checkout 8c2699fd      # branch verasonics-multibuffer-clean
git log --oneline -1                            # 8c2699fd Docs: document Verasonics multi-buffer ...
```

### 3. Build the image (zea's own Dockerfile; dependencies from its `uv.lock`)

```bash
docker build -t zea-swp:8c2699fd \
  --build-arg INSTALL_JAX=gpu --build-arg INSTALL_TORCH=gpu \
  . 2>&1 | tee ~/zea-build-8c2699fd.log
```

Add `--build-arg INSTALL_TF=gpu` only if TensorFlow is needed elsewhere. Needs ~15-20 GB on `/`.

### 4. Create the container (not VS Code-managed, so it keeps running)

```bash
docker run -d --name zea-swp \
  --gpus all -m 40g --cpus 7 \
  --env-file /home/luuk/projects/zea/.env \
  -v /home/luuk/mounts:/mnt/z \
  -v /home/luuk/zea-8c2699fd:/zea \
  -w /zea \
  zea-swp:8c2699fd sleep infinity
```

- `-m 40g --cpus 7 --env-file` mirror the group's dev-container settings
  (`~/projects/zea/.devcontainer/devcontainer.json`). The `.env` holds `WANDB_API_KEY`, `HF_TOKEN`, ...
- `/zea` is a bind mount: the editable zea install uses the host checkout, so source and image
  dependencies must be the same commit.
- `sleep infinity` keeps it alive (the image's default command is an interactive bash).

### 5. Verify

```bash
docker exec zea-swp bash -c 'git config --global --add safe.directory "*"; \
  git -C /zea log --oneline -1; \
  KERAS_BACKEND=jax python -c "import zea, jax; print(zea.__version__, zea.__file__, jax.devices())"; \
  KERAS_BACKEND=torch python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"'
```

Expect `8c2699fd`, zea `0.1.4` from `/zea/zea/__init__.py`, 8 CUDA devices, `True 8`.
(`safe.directory` stops git refusing a checkout owned by `luuk` inside the root container.)

### 6. This repo on the NAS

```bash
git clone https://github.com/lvknippenberg/shearWaveProcessing.git ~/mounts/VISUALIZE/shearWaveProcessing
# later: git -C ~/mounts/VISUALIZE/shearWaveProcessing pull
```

No install needed inside the container: the scripts put `src/` on `sys.path` themselves.

## Validate (after setup and after any zea / image change)

In tmux on the host:

```bash
tmux new -s swpval
REPO=/mnt/z/VISUALIZE/shearWaveProcessing
F=/mnt/z/VISUALIZE/raw_data/C000000001/SWE_01_SW_data_21-April-2026_12-12-54
docker exec -it -w $REPO zea-swp bash -c "rm -rf $F/output_linux $F/output_linux_torch && \
  KERAS_BACKEND=jax python scripts/linux_validation.py run --folder $F 2>&1 | tee validation_run_jax.log && \
  python scripts/linux_validation.py compare --folder $F 2>&1 | tee validation_compare_jax.log; \
  KERAS_BACKEND=torch python scripts/linux_validation.py run --folder $F --out-name output_linux_torch --no-converted 2>&1 | tee validation_run_torch.log && \
  python scripts/linux_validation.py compare --folder $F --out-name output_linux_torch 2>&1 | tee validation_compare_torch.log"
```

`linux_validation.py run` re-runs stages 1-2 for one folder into `<folder>/output_linux` (the reference
`output/` is untouched) and writes `run_info.json` (backend, GPUs, versions, commit, runtime).
`compare` checks every standard HDF5 dataset against `output/` (exact / close = relRMS < 1e-3 with
correlation and log-envelope correlation for IQ; GIFs by pixel difference) and writes
`compare_report.txt`; a description that differs only by an appended `[delay-and-sum]` tag is a `note`.

For a phase/displacement-level check beyond the report (constant vs time-varying phase offset,
frame-to-frame phase = displacement proxy), run from Windows or the container:

```
python study/analysis/iq_phase_check.py --folder <folder> output_linux output_linux_torch
```

### Validation result (2026-09-17)

| | server torch | server JAX | server JAX, **old zea `44208e0b`** |
|---|---|---|---|
| IQ, all buffers | bit-identical (402 datasets exact) | relRMS 4-7e-5, corr 1.000000 | corr 0.53-0.56, buffer 2 negative |
| converted RF | (not written) | exact | exact except missing `transmit_only` |
| displacement proxy, buffer 4 | identical | median 0.002°, p99 0.06° (~30 nm) | corr **0.21** |
| runtime C000000001 | 7.0 min | **6.1 min** | 7.0 min |

Windows (torch, zea `8c2699fd`) rerun on the same commit reproduced the reference bit-for-bit, which is
what separated "zea version" from "code drift".

## Batch processing on the server

Build any missing `CombinedData.mat` on Windows first (above). Then, in tmux on the host:

```bash
tmux new -s batch
docker exec -it -w /mnt/z/VISUALIZE/shearWaveProcessing zea-swp bash -c \
  "export KERAS_BACKEND=jax XLA_PYTHON_CLIENT_PREALLOCATE=false && \
   python scripts/process_raw_data.py --root /mnt/z/VISUALIZE/raw_data 2>&1 | tee study/logs/batch_server_\$(date +%Y%m%d).log"
```

- **Always set `KERAS_BACKEND` explicitly.** The `.env` defines it as an empty string, so a script's
  `os.environ.setdefault` does not apply and zea falls back to (uninstalled) TensorFlow.
- Folders that already have IQ + GIFs are skipped; rerun the same command after an interruption.
  `--folder <path>` (repeatable) processes specific folders; `--redo` forces a rebuild.
- JAX is fastest and matches to float precision; use `KERAS_BACKEND=torch` for bit-identical output.
- Fully detached alternative to tmux: `docker exec -d ... bash -c "... > log 2>&1"`, follow with
  `tail -f` on the log (it is on the NAS).

### Strain data (2026-10-02, beamforming started 2026-10-05)

`Z:\raw_data` holds 533 `*Strain_data*` folders next to the 724 SW folders. They carry buffers 3
and 6 only (`RF_data_3.bin` + `RF_data_6.bin`, ~5 GB per folder; buffer 6 = `Bmode_strain`, the long
widebeam recording) and no buffer 4, so the passive tools skip them. Their `CombinedData.mat` files
were built on Windows on 2026-10-02/03 (`build_combined_data.py --jobs 4`, 519 folders in 11.5 h,
~1.33 min/folder). **529 of 533 have one.** The four without are all `C000000012` and cannot be
built:

| folder | why |
|---|---|
| `VIS-014_Strain_data_07-July-2026_12-08-12` | RF bins complete, but `AcquisitionParametersAndECG.mat` (13.5 MB) is corrupt: MATLAB lists 5 small variables, then `load` fails ("File might be corrupt") |
| `..._12-11-03`, `..._12-15-43`, `..._12-17-19` | only a 0.5-4.6 kB runtime `.mat`, no RF (same at the DataHub source) |

The mirror matches DataHub by size, so this happened at acquisition, not in the download. The first
folder could only be rescued by reconstructing its runtime parameters from a sibling acquisition.
`process_raw_data.py` reports these four as failed; that is expected.

#### Beamforming on the server

`docker exec -d` runs the job inside the container, so it does not depend on the SSH session or
tmux: the call returns right away and the job survives logging out. Output goes to a log on the NAS.

```bash
ssh luuk@bmdserver3
docker ps --filter name=zea-swp --format '{{.Names}}  {{.Status}}'   # "Up ..."; else: docker start zea-swp
docker top zea-swp | grep process_raw_data                           # must print nothing before pulling
git -C ~/mounts/VISUALIZE/shearWaveProcessing pull --ff-only
```

1. **Pilot** (one folder, ~6.5 min). Check the log and the GIFs before starting the rest:

   ```bash
   docker exec -d -w /mnt/z/VISUALIZE/shearWaveProcessing zea-swp bash -c '
     export KERAS_BACKEND=jax XLA_PYTHON_CLIENT_PREALLOCATE=false
     F=$(ls -d /mnt/z/VISUALIZE/raw_data/C*/*Strain_data* | head -1)
     python scripts/process_raw_data.py --folder "$F" > study/logs/strain_pilot.log 2>&1'
   ```

2. **All strain folders** (the pilot folder is skipped as done):

   ```bash
   docker exec -d -w /mnt/z/VISUALIZE/shearWaveProcessing zea-swp bash -c '
     export KERAS_BACKEND=jax XLA_PYTHON_CLIENT_PREALLOCATE=false
     args=(); for d in /mnt/z/VISUALIZE/raw_data/C*/*Strain_data*; do args+=(--folder "$d"); done
     python scripts/process_raw_data.py "${args[@]}" > study/logs/batch_strain_server_$(date +%Y%m%d).log 2>&1'
   exit
   ```

- Never pull while a batch runs from this clone (its code must not change underneath it).
  Only Strain folders are passed (`--folder` per folder).
- Resumable: rerun step 2 after an interruption or failures; folders with IQ + GIFs are skipped.
  A buffer failure aborts its folder before any GIF is written, so failed folders are retried.
  The container has no restart policy: after a server reboot, `docker start zea-swp` first.
- Running? `docker top zea-swp | grep process_raw_data`. Stop: `docker exec zea-swp pkill -f process_raw_data`.
- Follow it in `Z:\shearWaveProcessing\study\logs\batch_strain_server_<date>.log` (mtime stays at the
  start time on Windows, contents are current).
- Buffer-3 unwrap runs automatically during beamforming but was validated on SW acquisitions only.
  When it cannot resolve the head, the folder keeps buffer 3 in stored order (not an error).

#### Status (2026-10-05)

- **Pilot** `C000000001/SWE_01B_Strain_data_21-April-2026_12-42-47`: ok in 6.4 min (JAX).
  Checked from Windows:

  | | buffer 3 (`bmode_focused`) | buffer 6 (`Bmode_strain`) |
  |---|---|---|
  | IQ | 26 x 382 x 529, REFoCUS adjoint | 268 x 382 x 509, delay-and-sum |
  | frame rate | 25.4 Hz | 88.2 Hz |
  | frame-mean envelope CV | 1.1 % | 1.6 % |
  | adjacent-frame complex corr. (median) | 0.72 | 0.92 |
  | GIF | 26 frames, real time | 152 frames @ 50 fps = 3.04 s, real time |

  Frame counts match the P1-6 base config; all values finite, no repeated or dropped frames. The
  buffer-3 grid is identical to the validated SW buffer 3 of the same subject (buffer 6 is slightly
  wider, like buffer 4); ~53 % zeros is the region outside the sector. Images match the anatomy of
  the scanner's `Strain_replay.avi`, and the 268-frame mean shows no fixed artefact. Buffer-3 unwrap
  was `ambiguous` (no trigger count, continuity margin 0.006) so it stays in stored order; the likely
  loop seam is between frames 2 and 3 (adjacent correlation 0.14).
- **Full batch** started 2026-10-05 10:37 (log `batch_strain_server_20261005.log`): 533 found, 532
  queued. At ~6.4 min/folder expect ~57 h, finishing around 2026-10-07 evening with 529 ok and the
  four `C000000012` folders above failed.

### Buffer-3 unwrap (2026-09-25)

Buffer 3 is a circular live loop, and its stored frames are rotated (`docs/buffer3_unwrap.md`).
New beamforms unwrap it automatically (`process_folder(..., unwrap=True)`). Folders beamformed
earlier are retrofitted with one idempotent command:

```bash
docker exec -it -w /mnt/z/VISUALIZE/shearWaveProcessing zea-swp bash -c \
  "export KERAS_BACKEND=torch && \
   python scripts/unwrap_buffer3.py --root /mnt/z/VISUALIZE/raw_data 2>&1 | tee study/logs/buffer3_unwrap_server_\$(date +%Y%m%d).log"
```

- **Pull first**, and only once no batch is running from this clone: a running process must not
  have its code replaced underneath it.
- Only folders whose beamforming is complete are touched (the buffer-4 GIF exists), and flagged
  files are skipped. Rerun it after later batches.
- `--dry-run` estimates without writing anything.
- It is CPU and NAS-bound (no GPU): ~6 s to estimate, plus a minute or two to rewrite a
  folder's ~1 GB converted buffer-3 file.
- **2026-09-28:** unwrap VERSION 2 was applied to all 724 SW folders from Windows (4 workers,
  2 h 20 min). Files flagged by an older version are redone automatically; files flagged by the
  current version are skipped. Pull this clone before the next beamforming batch so new folders
  get VERSION 2.

## Maintenance

- **Update this repo:** `git -C ~/mounts/VISUALIZE/shearWaveProcessing pull` (no rebuild needed).
- **Change zea version:** check out the new commit in a new folder, build a new image tag, create a
  new container, run the validation, then switch. Keep the previous container/image until it passes.
- **VS Code:** "Dev Containers: Attach to Running Container..." -> `zea-swp`. Attaching does not apply a
  `shutdownAction`, so closing VS Code leaves the container (and jobs) running.
- **Pitfall - VS Code-managed dev containers stop on close.** The group's
  `~/projects/zea/.devcontainer/devcontainer.json` sets `"shutdownAction": "stopContainer"`, so a job
  inside that container (`confident_lumiere`, image `vsc-zea-...`) dies when its VS Code window closes.
  Set `"shutdownAction": "none"` there, or use the standalone `zea-swp` container. Also avoid
  "Rebuild Container" on `~/projects/zea` unless its checkout is the commit you want.
- **Rollback:** the pre-switch dev container was saved as image `zea-backup:44208e0b`
  (`docker commit confident_lumiere zea-backup:44208e0b`). `docker stop zea-swp`, then
  `docker start confident_lumiere` (or run a container from the backup image with the same mounts).
