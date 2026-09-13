# Real97 support/query metric preparation

This is an opt-in evaluation branch. Existing training, data, splits and
simulation evaluators are unchanged. No model inference is launched here.

## Frozen split inventory

| Task | Eligible train | ID test | Additional OOD episodes |
|---|---:|---:|---:|
| Ball, 8 environments | 240 | 80 | 0 |
| Door, 10 environments | 438 | 100 | 0 |
| Stick, 8 environments | 214 | 24 | 0 |
| Soft, 15 ID + 4 OOD environments | 243 | 30 | 81 |

Ball and Door each retain exactly one test episode per environment/level 1-10.
Train counts per level are not necessarily equal. Stick's three test episodes
per environment deliberately represent moderate off-balance placements, not a
guaranteed reachable balanced-action set. Soft's two ID test episodes represent
visible opposite motion directions.

Support comes only from eligible train episodes, query only from whole held-out
test episodes. Environment IDs and source/episode IDs remain explicit. Soft OOD
episodes, including those originally labelled train before the environment
holdout, are not eligible first-batch ID supports.

## Measurement semantics

Ball target is absolute original-camera x in [280,330], not displacement.
The cropped image offset is (+80,+156). The target is x in [200,250] in the
560x248 crop. Use maximum observed x, not terminal x. A clipped or disappearing
ball does not imply a stationary endpoint. Missing/uncertain tracks require
review, and a truncated window cannot establish an unobserved eventual maximum.

Soft tracks the silver terminal block, not the rope or gripper. It has no
action-selection metric. A seed proposal and optical-flow success alone do not
certify correct identity; boxes and trajectories require review.

Door level correctness is based on the user-approved accepted level set.
door-4d levels 3 and 4 both count as correct selections, even if a particular
critical level-3 recording rebounds. Physical stable closure is a separate
diagnostic. door-12u-Half-11d-Half is retained for prediction and excluded from
the final action-selection denominator, not assigned a zero action score.

Stick tracks both blue endpoint masses plus the yellow support. Report angle
change relative to the rest configuration, endpoint/center trajectories and
actual lift. A level rod resting on the table is not a successful lift.
Re-estimate the empirical balance point using TRAIN outcomes only; the balance
estimate used to curate the test split may have used held-out observations and
must not be reused to select supports.

## Support quality, assessed on the actual selected chunk

- Ball: visible post-strike roll, no right-edge exit, not itself a target-reaching
  support. Initial screening uses >=25 px post-action travel, original peak
  <=580 px, and excludes [275,335] as a target-boundary guard. These are
  conservative proposal thresholds, not a modification of the [280,330] target.
  K is not yet frozen. A full-episode proposal must pass the same chunk-level
  checks before selection.
- Soft: four distinct train episodes/chunks, preferably two left and two right.
  Require >=25 px directional excursion and >=15 px sustained translation to
  reject jitter/rotation-only examples. At least one genuinely translated
  example in each direction is mandatory. No duplication to conceal shortages.
- Door: prefer level 2; a train recording just below the minimum closing level
  is another informative option. For the unreachable environment use levels
  9 and 10. The level reference is used only for support protocol/scoring,
  never as the model's query outcome.
- Stick: two nearby, visibly informative support positions bracketing a
  train-only empirical balance estimate. Avoid almost indistinguishable flat
  examples and extreme tipping. Use chunks covering the lift response.

Do not select supports by looking at test trajectory outcomes. The exact support
episode IDs and candidate counts will be frozen after measurement audit.

## Accuracy and artifacts

The new task-specific extractors are measurement PROPOSALS, not certified gold.
The same code runs on train/test and subsequently predictions. Preserve raw
frame tracks, detector confidence, missing reasons, source identities and
start/middle/end overlays. Ball additionally saves maximum-position frames and
separate train/test per-level tables. Human accuracy audit must include target
boundaries, exits, blur, occlusion, door rebound, and both soft motion directions.
Auto-seeded Soft results must not be used for final scores before seed review.

All model frames must be mapped back through the recorded resize/letterbox/crop
transform; never run a tracker on stacked comparison grids. Use the same native
frame indices for video and action. Exclude input frames and repeated padding.
Report missing-track rate rather than silently discarding failures.

## Reproduction

The isolated environment does not modify the training environment:

~~~bash
uv venv --python .venv/bin/python --system-site-packages .venv-real97-eval-20260909
uv pip install --python .venv-real97-eval-20260909/bin/python --no-deps \
  --default-index https://pypi.org/simple \
  opencv-python-headless==4.11.0.86 numpy==1.26.4
.venv-real97-eval-20260909/bin/python scripts/evaluation/prepare_real97_eval_inventory.py
.venv-real97-eval-20260909/bin/python scripts/evaluation/audit_real97_tracking.py \
  --inventory outputs/evaluation_real97_dataset_audit_20260909/episodes.jsonl \
  --output outputs/evaluation_real97_dataset_audit_20260909/tracking_v1 \
  --workers 2 --other-per-env-split 1
~~~

The first audit measures all 320 Ball episodes and one train/one test episode
per environment for other tasks. This is not a claim that all other-task tracks
have been validated. Use --other-per-env-split 0 for full extraction after
auditing the initial proposals. --seeds accepts reviewed per-episode Soft
initial boxes keyed by task/environment/episode_index.
