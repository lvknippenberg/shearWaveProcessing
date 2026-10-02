# Echo view classification (PLAX / PSAX / Apical) from the buffer-3 GIF

2026-10-02. Question: can the view of each acquisition be determined automatically from the
buffer-3 GIF, so subjects can be split into `PLAX/`, `PSAX/` and `Apical/` subfolders? (PLAX and apical are
for passive SWE. PSAX is for active SWE only: the natural wave there runs out of plane.)
Nothing on Z: was moved or restructured. The GIFs were only read.

## Method

[EchoPrime](https://github.com/echonet/EchoPrime) (Vukadinovic et al. 2024, Cedars-Sinai; **academic
licence**) ships a single-frame ConvNeXt-base **view classifier** with 11 classes (A2C, A3C, A4C, A5C,
Apical_Doppler, Doppler_PLAX/PSAX, Parasternal_Long, Parasternal_Short, SSN, Subcostal). It was trained on
clinical scan-converted DICOMs. `echoprime_views.py` feeds it the buffer-3 GIF frames with EchoPrime's own
preprocessing: centre square crop, 10 % zoom crop, 224 px, and its mean/std. It classifies **every frame**,
averages the softmax over the loop, and pools the classes to PLAX / PSAX / Apical / Other (SSN, Subcostal).
There was no training and no tuning on our data.
Weights (350 MB, not in git): `D:\Luuk van Knippenberg\Claude\view_classification\EchoPrime\model_data\weights\view_classifier.pt`
(from the v1.0.0 release `model_data.zip`).

`block_prior.py` adds the acquisition protocol: within a subject, a PLAX block is followed by a PSAX
block. It fits the single best change point and marks a folder for review when the call is not
confident (p < 0.8 or < 90 % frame agreement) or disagrees with that block fit.

## Results

**C000000049** (`study/logs/view_classification/C000000049_views.csv`, `sheets/C000000049_sw_views.png`)

* SW 08-46-35 … 08-49-57 → **PLAX** (6, p ≥ 0.97).
* SW 08-51-53 … 08-58-00 → **PSAX** (9). The last 5 are probably a more basal level, at p 0.76–0.92.
  Checked by eye: round LV with no vertical septum, so they are not apical.
  08-55-37 is flagged (p = 0.76), but it is PSAX by eye.
* **Strain_data**: these have no beamformed buffer 3 yet. 09-00-18 was beamformed locally as a test:
  **A4C, p = 1.00** (`sheets/C000000049_strain_09-00-18_frames.png`, a clean 4-chamber view). The other 13
  are *pending* until the Linux-server beamform (CombinedData.mat built on Z: 2026-10-02).
* **There is no apical view among the SW acquisitions.** The apical recordings are in the Strain_data
  folders, which have no buffer 4. So passive SWE on apical views is not possible for this subject with
  the current protocol, unless buffer 6 of the strain acquisition can be used.

**All 724 SW folders / 48 subjects** (`study/logs/view_classification/all_sw_views.csv`), as a check of feasibility:

* 427 PSAX, 278 PLAX, 18 Other (SSN/Subcostal), 1 Apical (C9 10-02-09, a real A4C by eye).
  Nearly every subject follows **about 6 PLAX, then about 9 PSAX**, including C1–C31. So the earlier
  passive work on "PLAX" relied on picking the right folders of mixed-view subjects.
* **Passive manual study cross-check** (`sheets/passive_read_but_PSAX.png`): 46 folders have
  `output/swp_passive_manual/`, and 40 of them are PLAX. Three that are classified PSAX carry a
  **completed reading (slopes.json)**:
  - C1 12-18-31 is a textbook mid-papillary PSAX (p = 1.00).
  - C33 08-43-33 (p = 0.82) and C35 10-54-57 (p = 0.79) show a round cavity with a horizontal echo
    below it: PSAX or off-axis, not a clean PLAX.

  C1 12-20-24 has only a general line. Those readings should be checked before they are pooled with
  the PLAX results.
* 587/724 (81 %) are confident. Only 4 confident calls disagree with the block prior, and all 4 are
  correct by eye: C9 apical, C12 two late PLAX, C33 a first PSAX.
* The errors are in the non-confident group. In C20, 3 of 17 were flipped at p 0.54–0.71, and the block
  fit corrects all of them. C3 has two p ≈ 0.5 "PLAX" calls inside the PSAX block. "Other" (SSN) means
  poor-window frames (C35, C38, C41), mostly PSAX-like.

## Verdict (first pass) and the user's assessment

My first-pass reading was **classify → flag → review**:
* Accept confident calls (about 80 %). No error was found among them in the sheets checked.
* Send the rest (about 140 folders, concentrated in C3–C8, C20, C33–C44) to a quick eye check. Show the
  block-prior suggestion, which fixes most of them.
* The model is off-domain (focused Verasonics B-mode, not clinical DICOM). The confidences are useful as
  a ranking but are not calibrated.

**User, 2026-10-02: not convinced by the sheets.** EchoPrime is a single-frame, 11-class, off-domain
model, and it is *not* the answer. Next step:
* Treat SW data as a **binary PLAX-vs-PSAX** problem. Apical views sit in Strain_data, and the only
  apical SW acquisition is C9 10-02-09.
* **Use the temporal content.** Each buffer-3 loop covers about one cardiac cycle: 32 frames at
  25.4 Hz = 1.22 s, unwrapped into chronological order.
* The user's plan, `D:\Luuk van Knippenberg\Claude\view_classification\Plan.txt`, was a supervised
  ResNet-18 / CNN+pooling / CNN-LSTM / R(2+1)D comparison with patient-grouped CV. After review it
  was **not implemented**:
  - the goal is sorting, not a methods comparison;
  - it assumed labels that did not exist.

  The user asked for as much as possible label-free, with labels only through a review UI.
  That is the second pass below.
* The EchoPrime output is not ground truth. At most it is a pre-label to correct by hand, and a
  zero-training baseline to beat.

## Second pass: label-free consensus + review (2026-10-02)

Goal (user): **sorting the data**, as far as possible without labels. Temporal content is used.
Each loop is about one cardiac cycle (26 or 32 frames at 25.4 Hz) and is not ECG-gated.

`extract_features.py` caches the following per loop (locally, `features_sw_v1.npz`, about 13 min):
* per-frame EchoPrime frame-classifier features (1024-d) and its 11-class probabilities;
* EchoPrime **video-encoder** embeddings: MViT-v2-S, 16 frames spanning the loop, 4 start offsets.

`label_free_sort.py` runs **four voters** with no human labels anywhere:

| voter | what | agreement with `ep` |
|---|---|---|
| `ep` | EchoPrime frame classifier, averaged over the loop (first pass) | – |
| `clu_f` | per-subject 2-cluster split of [mean, temporal std] of frame features, centred on the subject | 0.91 |
| `clu_v` | the same on video embeddings | 0.88 |
| `self` | logistic regression trained on the *other* subjects' confident (≥ 0.95) EchoPrime loops (pseudo-labels, leave-subject-out), on the subject-centred features | 0.97 |

A cluster is named PLAX or PSAX by its members' summed EchoPrime log-odds.

**Unanimous → auto: 584/724 (238 PLAX, 346 PSAX).** The other 140 need review:
* 128 split votes, 11 "other?" and 1 "apical?";
* spread over 31 subjects (C5, C32, C33, C37, C42 and C43 have the most).

Protocol plausibility, without labels: extra view switches beyond the single PLAX→PSAX switch occur
in 6–13 subjects per voter, but in 1 subject for the consensus.

**What label-free methods cannot do:** say which side of a split vote is right, or prove that the
unanimous 81 % is correct. Clustering alone is weaker than hoped, because PSAX at different levels
does not always form one tight group within a subject. So labels are still needed, but only as a
review.

`review_views.py` was the review UI, one subject per screen. It is now `swp.views.review`, run with
`scripts/view_sort.py review`; the study script was removed after the move.
* every loop **plays**;
* flagged loops have a red frame and show their votes;
* proposed labels = the unanimous label, else the majority vote, with 2–2 ties going to the
  subject's PLAX-then-PSAX block fit;
* click, or press 1–4, to set PLAX / PSAX / Apical / Unclear; ENTER accepts the subject.

The 31 flagged subjects come first, then the 17 unanimous ones (a glance validates the auto set).
Labels go to `study/logs/view_classification/sw_views_manual.csv`, and the tool resumes where it
stopped. Example screen: `sheets/review_ui_C000000020.png`, where the proposals match by eye
(8 PLAX, then 9 PSAX).

## Review result (2026-10-02): ground truth for all 724 SW loops

The user reviewed all 48 subjects in the review window. **`study/logs/view_classification/sw_views_manual.csv`
is the view label of every SW acquisition** (column `label`): 294 PLAX, 420 PSAX, 9 Unclear, 1 Apical.
It is the table to sort and filter on. Scores: `score_voters.py` → `voter_scores.txt`.

Changes to the proposals:
* 15 changes in total: 9 of the 140 flagged loops and 6 of the 584 unanimous loops.
* All 6 unanimous changes are in **C38**: 3 PLAX that every voter called PSAX, and 3 marked Unclear.
* Unclear: C5 09-27-12, C38 (7 loops), C39 12-13-36.
* Apical: C9 10-02-09.

Accuracy on the 714 PLAX/PSAX loops:

| | accuracy |
|---|---|
| EchoPrime frame classifier | 0.966 |
| clustering, frame features | 0.926 |
| clustering, video embeddings | 0.909 |
| self-trained head | 0.979 |
| **unanimous consensus (the 81 % auto set)** | **0.995** (3 wrong, all C38) |
| review proposals | 0.992 |
| first-pass "confident" EchoPrime (p ≥ 0.8) | 0.997 (2 wrong of 587) |

So the voters are individually mediocre (the clustering most of all), but their **agreement is a
reliable "safe to auto-sort" signal**. Its errors came as a whole subject with poor windows (C38),
not as scattered loops.

**Supervised head on frozen features** (`supervised_probe.py`, leave-one-subject-out on these labels):

| features | accuracy |
|---|---|
| frame mean | 0.982 |
| + temporal std | 0.983 |
| video embedding | 0.968 |
| all | 0.987 |

With frozen EchoPrime features, the **temporal information adds about 1 loop**: the anatomy in a
single frame already carries the view, and the errors are poor or atypical windows. Three C7 loops
labelled PLAX (11-47-50, 11-53-29, 11-54-10) are confidently called PSAX (p ≤ 0.05). The user
re-reviewed C7: all 8 are PLAX, as "not always clear, but all attempts at PLAX". Labels are
unchanged.

**For future subjects: now in the repo** as `src/swp/views/` + `scripts/view_sort.py`
(runbook: `docs/view_sorting.md`). The fourth voter is a supervised head trained on these labels
(`view_head.npz`), replacing the self-trained head.

Leave-one-subject-out through the packaged pipeline (`view_sort.py evaluate`):

| | accuracy |
|---|---|
| ep | 0.966 |
| clu_f | 0.931 |
| clu_v | 0.916 |
| sup | 0.987 |
| proposal | 0.990 |

**Unanimous: 82 % of loops, 0 errors.** C38's three all-wrong loops are now flagged instead of
auto-labelled.

Features moved to a per-folder cache: `<DATA_ROOT>/view_classification/features/<subject>/<folder>.npz`,
split from `features_sw_v1.npz` and bit-identical to a fresh extraction.

**Passive manual study:**
* All 42 folders with slopes are PLAX by these labels. C33 08-43-33 and C35 10-54-57, flagged as
  doubtful in the first pass, are PLAX.
* The 4 C1 PSAX folders no longer carry slopes.

## Decision: labels stay a CSV, `Z:\raw_data` is not restructured (user, 2026-10-02)

`study/logs/view_classification/sw_views_manual.csv` is the sorting manifest, and tools filter on its
`label`. The manual passive study already does this (`a1927b6`). Reasons:
* `Z:\raw_data` is the verified 1:1 mirror of DataHub P000000569. Moving folders into view subfolders
  would make a future `mdr_webdav --refresh` download them again.
* The swp tools assume `<subject>/<acquisition>`.

## Review montages: confirming the labels with a colleague

`review_montages.py` writes one GIF per patient to
`D:\Luuk van Knippenberg\Claude\view_classification\review_montages\`. The GIFs are 4–8 MB each and are
not in git.
* Each GIF is the review screen as a movie: every SW acquisition in order, its buffer-3 loop playing in
  real time, with the reviewed label.
* Files are **ranked by confidence, lowest first** (`01_<subject>_conf….gif`), and
  `index.csv` lists the ranking (copy: `study/logs/view_classification/review_montages_index.csv`).

How the confidence is computed:
* **Per loop:** the probability that the leave-one-subject-out supervised head, trained *without*
  that patient, gives the reviewed label. Unclear counts as 0.5, and Apical uses EchoPrime's apical
  probability.
* **Per patient:** the mean over its loops; the lowest loop is shown as well.
* A red frame marks a loop where the model disagrees with the label (< 0.5).
* The ranking therefore shows where the model and the reviewer differ. It is not a measure of image
  quality.

## Files

| file | what |
|---|---|
| `echoprime_views.py` | classifier → CSV (per-folder 11-class + pooled probabilities, frame agreement) |
| `block_prior.py` | protocol block fit, `suggested` + `review` columns |
| `contact_sheet.py` | labelled first-frame sheets (from a CSV) or unlabelled per subject |
| `big_frames.py` | 3 frames per acquisition at full size, for checking by eye |
| `local_strain_buffer3.py` | buffer-3 beamform of Strain_data on a LOCAL copy (only 09-00-18 was run; the beamform moved to the server) |
| `study/logs/view_classification/` | `all_sw_views.csv` (724 SW folders), `C000000049_views.csv` (with the proposed paths, not applied) |
| `study/logs/view_classification_strain_combined_C49.log` | CombinedData.mat build for the 14 C49 Strain_data folders on Z: (14/14 validated afterwards) |
| `extract_features.py` | EchoPrime frame + video-encoder features per loop (cached locally) |
| `label_free_sort.py` | four label-free voters + consensus → `study/logs/view_classification/sw_views_consensus.csv` |
| (moved) | review UI → `swp.views.review` / `scripts/view_sort.py`; labels in `study/logs/view_classification/sw_views_manual.csv` |
| `score_voters.py`, `supervised_probe.py` | voters vs labels; supervised head LOSO → `voter_scores.txt` |
| `review_montages.py` | one ranked review-montage GIF per patient (outside git) + `review_montages_index.csv` |
| `sheets/` | contact sheets of C49 and of the uncertain subjects, plus confident-vs-block cases |
