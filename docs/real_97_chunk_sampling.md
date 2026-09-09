# September 7 real datasets: proposed chunk sampling contract

Status: discussion and annotation only. No training launch and no loader changes.
The annotations are sidecars; source video, parquet, episodes metadata, and splits
are not modified. This proposal supersedes the older short-episode exclusions.

## Common conventions

- All four datasets use native 20 Hz video. Frame indices start at zero.
- `K` includes the conditioning first frame. There are `K-1` time intervals.
- Sample `index[k] = min(start + stride*k, N-1)` for `k=0,...,K-1`.
- Video, action, and state use exactly these same indices and last-row padding.
- Never discard an episode just because it is short. Do not loop its frames or
  temporally interpolate them. Padding repeats the terminal observation/row; it
  is not an extra observation of the physical object's future behavior.
- Training samples only episodes explicitly assigned to `train` by the existing
  episode-level split, including root split manifests where the episode metadata
  does not carry the assignment. Exclude every test episode from both model and
  C updates, including new-C-only curriculum phases. Missing or conflicting
  assignments are errors, never an implicit `train` default. Do not split
  overlapping chunks into different partitions. Evaluation window manifests
  should have fixed random seeds and be saved.
- No ROI weighting. Proposed online lighting augmentation probability is 0.70,
  with shared clip parameters and gx/gy each uniform in [-0.08, 0.08]. It does
  not change any video/action/state timestamps or stored original videos.
- Ball, door, and stick per GPU: three distinct active environments and eight
  sampled episodes/chunks per environment. Two GPUs give 48 micro-examples per update, assuming this
  entire group is accumulated before one optimizer update. Microbatch size is
  independent of this logical group size.
- All four tasks use the same hierarchy: select environments from the current
  eligible curriculum pool, sample episodes uniformly from each selected
  environment's `train` pool, then sample a chunk within each episode using its
  task-specific temporal rule. Request eight episodes per environment for ball,
  door, and stick, and six for soft. Use episode replacement only if the
  environment's train pool has fewer episodes than requested. Environments on
  separate GPUs may overlap.
- Do not sample levels, balance levels, align levels/actions across environments,
  or construct an environment-by-level Cartesian product. Level is metadata
  only. Consequently a level with more training episodes has proportionally
  greater probability of appearing; this is intentional episode-uniform sampling.
- The 50/50 and 30/70 branches below are per-chunk probabilities, not a forced
  exact count in every per-environment chunk group. Resample starts online, not a fixed
  list of ten chunks per episode.

## Spatial input sizes

Keep model-input pixel counts close to the old `224 x 224 = 50,176` budget,
while using rectangular inputs. All dimensions below are width x height and
individually divisible by 32. Existing training configurations are unchanged;
these are the settings for the new four experiments.

| Task | Dataset image | Model input | Pixels | Change from old budget |
| --- | --- | --- | --- | --- |
| Ball friction | 560 x 248 | 320 x 160 | 51,200 | +2.04% |
| Door close | 640 x 480 | 256 x 192 | 49,152 | -2.04% |
| Stick balance | 640 x 480 | 256 x 192 | 49,152 | -2.04% |
| Soft pull | 512 x 256 | 320 x 160 | 51,200 | +2.04% |

Use `resize_mode: letterbox`: preserve aspect ratio, then pad only as necessary.
Door, stick, and soft match their target aspect ratios exactly, with no padding.
Ball resizes to 320 x 142 and adds nine rows above and below; do not distort its
aspect ratio or use the earlier proposed 352 x 160 canvas.

Training and model inference must use the same spatial transform and dimensions.
Only at video export remove the recorded padding and resize to the dataset's
original image size; apply consistent geometry to GT comparisons. Restoring
output dimensions does not recover detail lost during downsampling. Spatial
transforms do not modify temporal sampling or action/state indices. This is a
configuration plan, not a completed GPU execution check.

## Action sources after the continuity audit

The full audit covered 1,450 episodes and 135,384 frames across all four
exports, including held-out data for diagnostics only. Reports are in
`outputs/action_audit_real_97_20260909T044554Z/`; the reproducible audit command
is `scripts/audit_real_97_target_actions.py`. No dataset values were modified
and no audit statistics were installed as training normalization statistics.

| Task | Source selected for this experiment | Reason |
| --- | --- | --- |
| Stick balance | `action`, the recorded 8D target EEF/gripper | All 238 episodes have a fixed target quaternion, with no sign flips. |
| Ball friction | `raw.commanded_target.joint_rad` | Ten episodes use the opposite target-quaternion sign, including eight train episodes; recorded joint targets are continuous. |
| Soft pull | `raw.commanded_target.joint_rad` | 607 native-adjacent target-quaternion sign flips occur in 314 of 354 episodes; recorded joint targets are continuous. |
| Door close | Pending export of actual target fields | Neither EEF targets nor joint targets are present in this export. Never substitute measured `action` silently. |

For ball and soft, every audited row has `raw.commanded_target.arm_sent=true`.
Recorded commanded joints and mapped candidate joints are identical in this
export, but select the commanded field to preserve the actual-sent provenance.
Maximum native-adjacent change of any joint is 0.067954 rad for ball and
0.041894 rad for soft, with no nonfinite target rows or pi-scale wrapping jumps.
If a later export contains unsent/invalid command rows, handle them explicitly;
do not substitute the measured action or an unsent candidate automatically.

Quaternion sign flips represent equivalent rotations, not physical flips of
the robot. Consistent quaternion canonicalization could retain EEF targets,
but this run takes the user's joint-target fallback rather than feeding
uncanonicalized EEF quaternions into the model. Ball's sign-reversed episodes
are `ball-4_lerobot/episode_000030` through `episode_000039`. Soft's main action
is exactly the float32 raw EEF target, so selecting its raw/calibrated EEF
field alone would not eliminate the sign flips.

These are source-selection decisions, not completed loader changes. Encoder
packing and train-only normalization must explicitly use the selected target
source. Preserve native time alignment and apply the same stride as video;
do not construct a future measured-state input merely to reuse an old action
loader. If gripper control is included, read its target field, not measured
gripper readback. Existing configurations and datasets remain unchanged.

## Ball friction

Dataset: `ball_friction_9_7_crop`. Eight environments, 320 episodes.
Use `K=61`, stride 1, native span 61 frames, elapsed span 3.0 seconds.

Target-based event definitions:

- `A`: beginning of the initial left swing, from existing skill annotations.
- `R`: beginning of the subsequent right swing, from existing skill annotations.
- `Z`: first right-swing frame whose target orientation reaches/crosses the
  episode's initial orientation. Compute from raw target wxyz quaternions.
- `B = Z-3`: leave three native frames (0.15 seconds) before that crossing.

Choose an integer start uniformly from `[R,B]` with probability 0.5, or from
`[0,B]` with probability 0.5. An explicit coverage guard intersects either range
with `start >= Z-60`, ensuring that the window reaches the target return event
instead of containing preparation only. The sidecar keeps both unguarded
documented ranges and guarded ranges, so this extra condition is auditable.

The precise branch starts at the beginning of the RIGHT swing, not the earlier
left preparation. If `R > B` for a fast action, the requested precise range is
empty. Mark the episode `needs_review`; do not silently exclude it, fall back
to the wide branch, or change the three-frame pre-return margin.

Do not cap the upper bound at `N-61`: end padding is allowed, including long
episodes with a late chosen start. This selection does not imply a precisely
known visual contact time or a known time at which the ball comes to rest.

## Door close

Dataset: `door_close_9_7`. Ten environments, 538 episodes.
Use `K=49`, stride 1, native span 49 frames, elapsed span 2.4 seconds.

The intended target-based policy is:

- `L`: first left-swing moving frame after the right preparation/pause.
- `Z`: last frame before that left swing crosses the initial orientation.
- Probability 0.5: uniform integer start in `[L,Z]`.
- Probability 0.5: uniform integer start in `[0,Z]`.
- Require `start+48 >= Z+1`; permit last-row padding.

Important data limitation: this export has `observation.eef_state`, measured
joint state and next measured joint action, but NO per-frame target trajectory.
Current sidecars therefore use **measured EEF proxies**, not target events. A
left or right phase starts at the first of two consecutive signed angular
increments above 0.05 degrees/frame about the base negative-X axis relative
to the first quaternion. `Z` is the frame just before the measured zero crossing.
The threshold rejects small stationary jitter; it is not a target-speed label.

These proxy annotations carry `requires_target_policy_approval=true`. Do not
silently pass them off as exact target timing. Either explicitly adopt this
measured-motion sampling variant or supply raw target data before implementing
the literal target-based policy. The scalar level labels remain unchanged.

## Stick balance

Dataset: `stick_balance_9_7`. Eight environments, 238 episodes.
Use `K=41`, stride 3, native span 121 frames, elapsed span 6.0 seconds.

Let `M = max(0,N-121)`. Existing `meta/action_segments.jsonl` defines target
lift segments. Let `[a,b)` be their full envelope, with exclusive end `b`.

- Probability 0.30: uniform integer start in `[0,M]`.
- Probability 0.70: uniform integer start in
  `[max(0,b-1-120), min(a,M)]`.

The second branch brackets the full annotated lifting interval in the native
timeline. It does not pretend that stride-3 sampling observes every native
frame. An annotation reaching the recording end is explicitly marked as such;
it does not establish that lifting or object motion stopped at the last frame.
If the interval is impossible, keep the episode as `needs_review`, never prune
it silently. Short episodes use start zero and aligned last-row padding.

## Soft pull

Dataset: `soft_pull_9_7_crop_512x256`. Nineteen environments, 354 episodes.

Soft-specific logical batch per GPU: five distinct eligible environments and
six train episodes/chunks per environment (`5 x 6 = 30`). Two GPUs contribute
60 samples per optimizer update when the full logical batch is accumulated.
This overrides the other three tasks' `3 x 8` setting; it does not require a
simultaneously resident microbatch of 30. Episode-uniform sampling and all
curriculum, split, and temporal sampling rules remain unchanged.

Select five environments uniformly without replacement from the eligible
curriculum pool. When the initial active pool or a new-C-only pool contains
exactly five environments, select all five, without repeating environment
slots. Never include held-out environments or old environments in a new-C-only
phase. Environments on separate GPUs may overlap.

Reserve the complete environments `soft-3l`, `soft-6m`, `soft-4l`, and `soft-7r`
for held-out-environment evaluation. NONE of their episodes participate in
training, including episodes labeled `train` by the original episode split.
Do not fit training normalization statistics or train environment C entries on
these four environments. Test-time C adaptation is an evaluation operation,
not part of this Stage 1 training pool.

The other 15 environments enter training in three rounds of five environments.
Keep the initial 300 model-only steps and the existing 1000-step curriculum
cycle for each round. Round completion steps are 1300, 2300, and 3300. After
step 3300, introduce no more environments and alternate 200 C-only steps with
200 model-only steps over the full 15-environment training pool. Exact round
membership is not yet assigned; the sorted environment list is not a curriculum
ordering. The protected step-2300 checkpoint policy remains unchanged.

The additive `ENVIRONMENT_HOLDOUT_SPLIT.json` records this environment holdout
without rewriting the existing `TRAIN_TEST_SPLIT.json`. Training eligibility
is the INTERSECTION of the 15 permitted environments and their original train
episodes. Original test episodes in those 15 environments remain in-domain
held-out-episode evaluation data. All episodes in the four excluded environments
are reserved for held-out-environment evaluation; keep the two evaluation pools
distinct. Environment holdout takes precedence over an episode's old train label.

Use `K=33`, stride 3, native span 97 frames, elapsed span 4.8 seconds.

Select an episode uniformly within the selected environment, then select an
integer start uniformly in `[0,max(0,N-97)]`. There is no common fixed key-motion
phase across all environments, so no artificial precise/wide branch is added.

Retain the 62-frame episode: formal `soft-3r_lerobot/episode_000000`, original
source episode 1. With start zero, the sample reaches native frame 60 and then
repeats the terminal frame 61 for indices that exceed the recording. Video,
action, and state are clipped identically. The older document's episode 1 name
uses source numbering and must not be confused with formal LeRobot episode 1.

## Annotation artifacts and limits

Run `scripts/annotate_real_97_chunk_events.py --datasets-root /path/to/datasets_real`
to create these files at each ball/door/stick dataset root:

- `chunk_events_v1.jsonl`: one row per episode, provenance, split, event frames
  and seconds, start ranges, padding counts, and the reference motion signal.
- `chunk_events_v1_summary.json`: counts, frame ranges, and review cases.
- `chunk_events_v1_examples.svg`: representative reference trajectories.
- `CHUNK_EVENTS_V1.md`: local schema notes.

The command refuses to overwrite previous outputs. Frame-order/range failures
are explicit rows, not silent episode removal. These annotations only describe
robot commands or measured motion. Visual object contact, ball stopping, and
door closure need separate image-based annotation and remain unknown here.
