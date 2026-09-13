# Ball and Door Ours reference inference

Use the completed Ball job113231 step5500 model AND step5500 C table.
Door job113230 timed out after reaching approximately step3782; only the fully
published step3715 model AND step3715 C table are used. Never substitute a later
phase-only C table. Preparation pins the model by hard link and copies the
matching table, frozen training config, and training-only normalization.

## Preparation and support

All metadata analysis and video decoding run in a low-priority CPU allocation.
Freeze the v2 training manifests; do not change the dataset or train/test split.

Ball uses one train support per environment. Prefer a full-episode peak x in
[150,220] in original 640-wide camera coordinates. Require at least 25 pixels
of post-action rolling and at least 25 pixels of motion in the selected window,
95% usable tracking, no observed right-edge clipping, and neither full-episode
nor selected-window peak in the target [280,330]. If no candidate satisfies
the preferred interval, choose a qualifying fallback and record that fact.
Use existing calibrated tracking as PROPOSAL data, not certified ground truth.
Select a training-compatible precise window containing the peak when possible.
If no support meets the eligibility constraints, record the unavailable
environment explicitly; never silently prune or fabricate support.

Door uses available train episodes at minimum-success-level minus 1 and plus 2.
Missing requested levels are recorded, not clamped or synthesized. For example,
door-0 requests level0 (missing) and level3, so it uses the available level3 only;
door-12u-Half requests 8 and11 and uses 8 when11 is unavailable.
The unreachable door-12u-Half-11d-Half environment uses levels9 and10 and is
prediction-only, excluded from action-selection success. Accepted action levels
for reachable environments are the user-specified minimum and the next level;
door-4d therefore accepts 3 or4.

## Queries and inference

Select two support-disjoint train episodes plus all held-out test episodes in
each eligible environment. Pick the midpoint precise window from action-event
annotations; do not optimize query selection using its tracking outcomes.
Ball: 61 frames, 320x160, joint_target. Door:49 frames, 256x192,
joint_target_export. Both stride1, one observed frame, aligned14D target
actions, same letterboxing and tail padding as training. Playback is20fps,
not the Soft/Stick stride3 rate of20/3fps.

Stage1 uses the paired training table entry. Stage2 starts at the table mean,
optimizes only FP32 Z with all selected supports, and uses the existing
0.3x10,0.15x10,0.05x10,0.015x10 schedule. No query GT enters adaptation.
No ROI or inference light augmentation. Generate50 diffusion steps and use
matching per-query seeds for Stage1 and Stage2.

Save raw GT/S1/S2, per-query stacked videos, environment grids, support loss
and Z trajectories, and training-table-fitted PCA with circles for training
and black-bordered triangles for inference.

## Separate action-template extension

For one test initial frame per environment, reuse the same learned Z and
sweep original recorded train target-action templates at every available
level, excluding support episodes. Do not interpolate or retime actions.
Keep the initial image and diffusion seed fixed across action levels.
Record the normalized first-target difference from the anchor action so
initial-command mismatch is visible rather than hidden.
These outputs have NO paired future GT; save them under extensions/ with
explicit labels and never include them in factual prediction errors.
They are exploratory counterfactuals, not certified action-selection scores.

## Metric boundaries and safety

Full-episode target labels and selected prediction windows are not always
equivalent: some61-frame Ball windows end before the full-episode peak; some
49-frame Door windows end before the final closure. Record these coverage flags.
Do not treat a truncated rollout as proof of the full-episode maximum or stable
closure. Object-centric and action metrics remain pending tracking review.
Formal results are not published to results/ yet.

GPU jobs use one low-priority H100/H200/B200, with afterok CPU-preparation
dependencies. Reuse locked node-local Wan/model/data caches. Separate
job-specific output directories and locks protect concurrent/resumed runs;
context and per-variant completion files support requeue without starting over.
Old reference scripts/configs and existing results are unchanged.
