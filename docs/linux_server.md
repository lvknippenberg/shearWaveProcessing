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
python -c "import sys; from swp.acquisition.combined import ensure_combined_data; [ensure_combined_data(f) for f in sys.argv[1:]]" "Z:\raw_data\C0000000xx\<folder>"
python scripts/process_raw_data.py --root "Z:\raw_data" --check      # audit (needs the Windows base-config dir)
```

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
