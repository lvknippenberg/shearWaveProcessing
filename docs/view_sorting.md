# View sorting: PLAX / PSAX / Apical per SW acquisition

Passive SWE uses PLAX (and apical) views, where the septum is close to horizontal (or vertical). In
PSAX the natural wave runs mostly out of plane, so PSAX is used for active SWE only. Each subject's
SW acquisitions follow a protocol of **about 6 PLAX, then about 9 PSAX**. Apical views were recorded
in the `Strain_data` acquisitions (no buffer 4), and only one SW acquisition is apical (C9 10-02-09).

## The labels table

**`study/logs/view_classification/sw_views_manual.csv`** has one row per SW measurement folder:

| column | meaning |
|---|---|
| `subject`, `folder` | `Z:\raw_data\<subject>\<folder>` |
| `label` | **PLAX / PSAX / Apical / Unclear** (Unclear = off-axis, poor window, cannot tell) |
| `proposed`, `needs_review`, `votes` | what the classifier proposed, whether it was flagged, the four votes (ep clu_f clu_v sup; L = PLAX, S = PSAX) |
| `changed`, `source`, `reviewed_at` | label ≠ proposal; `review` (seen in the review window) or `auto`; timestamp |

Tools filter on `label`; the manual passive study reads only PLAX and Unclear (`a1927b6`).
**`Z:\raw_data` is not restructured** into view folders (decision 2026-10-02):
* it is the verified 1:1 DataHub mirror, so moving folders would make a refresh download them again;
* the swp tools assume `<subject>/<acquisition>`.

State on 2026-10-02: 724 folders, 48 subjects, all reviewed by hand (294 PLAX, 420 PSAX, 9 Unclear,
1 Apical).

## Running it on new data

Buffer 3 must be beamformed first (`run.py beamform` writes `output/CombinedData_buffer3_iq.gif`).
Then:

```
python scripts/view_sort.py run --root "Z:/raw_data"        # = features + classify + review
```

or step by step:

```
python scripts/view_sort.py status   --root "Z:/raw_data"   # GIFs / cached features / labelled, per subject
python scripts/view_sort.py features --root "Z:/raw_data"   # EchoPrime features of new loops (~1 s/loop on GPU)
python scripts/view_sort.py classify --root "Z:/raw_data"   # votes -> study/logs/view_classification/sw_views_votes.csv
python scripts/view_sort.py review                          # the window, for subjects with unlabelled loops
python scripts/view_sort.py train-head                      # after a review round: refit the head on all labels
python scripts/view_sort.py evaluate                        # leave-one-subject-out check against the labels
```

Options and setup:
* `review --auto-accept-unanimous` labels subjects that have no flagged loop without showing them
  (`source = auto`). `review --subject C… [--all]` re-opens a subject that is already labelled.
* **Classify a subject as a whole**: the features are centred per subject and the clustering voters
  split each subject's loops into two groups.
* **Once per machine**, `python scripts/view_sort.py download-weights` fetches EchoPrime's weights
  (a 1.3 GB archive, of which two files, ~490 MB, are kept).
* Locations (env override): weights `SWP_ECHOPRIME_DIR`, feature cache `SWP_VIEW_CACHE` (default
  `<SWP_DATA_ROOT>/view_classification/…`), labels `SWP_VIEW_LABELS`, votes `SWP_VIEW_VOTES`.

### The review window

One subject per screen. Every loop plays in acquisition order, flagged loops have a thick red frame,
and each title shows the label (`*` = differs from the proposal) and the votes.

| mouse / key | action |
|---|---|
| click a loop | cycle its label PLAX → PSAX → Apical → Unclear |
| 1 / 2 / 3 / 4 | set the loop under the mouse to PLAX / PSAX / Apical / Unclear |
| ENTER | accept the subject (saved at once), next |
| b / space / q | back / pause / quit |

It is plain Tk: frames are pre-made once per subject and a tick only swaps images. Matplotlib needed
~220 ms per frame on the loaded server, which made the window unusable over remote desktop.

## How it classifies (`src/swp/views/`)

**Features** (`echoprime.py`). EchoPrime (Vukadinovic et al. 2024, github.com/echonet/EchoPrime,
**Cedars-Sinai academic licence**; weights are never committed) provides two models:
* an 11-class single-frame view classifier (ConvNeXt-base);
* a video encoder (MViT-v2-S, 16 frames → 512-d).

Both are run with EchoPrime's own preprocessing on every buffer-3 GIF frame: 26 or 32 frames at
25.4 Hz, about one cardiac cycle, not ECG-gated. The cached features per loop are:
* frame features and class probabilities, per frame;
* video embeddings of 4 clips that span the loop with different start offsets.

**Voters** (`vote.py`):

| voter | what |
|---|---|
| `ep` | EchoPrime's PLAX-vs-PSAX call, softmax averaged over the loop |
| `clu_f` | per-subject 2-cluster split of [mean, temporal std] of the frame features, subject-centred |
| `clu_v` | the same on the video embeddings |
| `sup` | logistic-regression head on [frame mean, std, video], subject-centred, trained on the reviewed labels (`view_head.npz`, refitted by `train-head`) |

A cluster is named by its members' summed EchoPrime log-odds. **Unanimous** (and EchoPrime sees no
apical/other view) → safe auto label. Otherwise the loop is flagged. Proposal = majority vote, with
2–2 ties going to the subject's protocol block fit (one PLAX block, then one PSAX block).

**Performance** (`evaluate`, leave-one-subject-out on the 714 PLAX/PSAX-labelled loops; the head is
refitted without each subject):

| | accuracy |
|---|---|
| ep | 0.966 |
| clu_f | 0.931 |
| clu_v | 0.916 |
| sup | 0.987 |
| **proposal** | **0.990** |

**Unanimous auto labels: 82 % of loops, 0 errors.** The individual voters are mediocre, but their
agreement is a reliable signal. The errors are poor or atypical windows, and the temporal
descriptors add only about one loop with these frozen features. The study record (first EchoPrime
pass, the label-free consensus, the review, why the plan for a supervised video-model comparison was
cut down) is in `study/analysis/view_classification/README.md`.

## Confirming labels with a colleague

`study/analysis/view_classification/review_montages.py` writes one GIF per patient: the review
screen as a movie, with the reviewed label and the model's confidence in it per loop. The GIFs go
to `<DATA_ROOT>/view_classification/review_montages/`, at 4–8 MB each, outside git.
* **Ranked lowest confidence first** (`01_<subject>_conf….gif`, `index.csv`; a copy of the index is
  in `study/logs/view_classification/review_montages_index.csv`).
* A **red frame** marks a loop where the leave-one-subject-out head disagrees with the label.
* The confidence is per loop, the probability the head (trained without that patient) gives the
  label; Unclear counts as 0.5. The per-patient score is the mean over its loops.
* The ranking points at disagreement between the model and the reviewer, not at image quality.

On 2026-10-02 the lowest-ranked were C38 (7 Unclear), C7 (3 loops called PSAX by the model; the
user confirmed all 8 as attempts at PLAX) and C35.
