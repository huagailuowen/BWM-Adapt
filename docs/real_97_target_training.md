# September real-data target-command training

These are new configs; legacy loaders, datasets and ROI experiments remain usable.
The launcher runs two independent 2-GPU process groups in one high-priority,
24-hour, 4-GPU H200/B200 allocation. A Python/DDP failure in one group does not
cancel its peer. Node failure, allocation timeout and Slurm cancellation still
affect the whole allocation; this is not hardware fault isolation.

The same launcher also accepts a single config for one two-GPU experiment.
In that mode submit with `--gres=gpu:2 --cpus-per-task=32 --mem=250G` to override
the four-GPU batch defaults. It does not retain a four-GPU allocation merely
to run one two-GPU child.

Before staging weights, `check_real97_cuda.py` requires the exact allocated
device count and executes a tiny CUDA operation on each visible device. Each
two-GPU child repeats the check after its visibility mask is set. The trainer
also requires a CUDA accelerator with exactly two distributed ranks when
`BWM_REQUIRE_CUDA=1`; legacy CPU-capable runs without this flag are unchanged.
Failed GPU preflight exits immediately, without starting CPU training.

On 2026-09-09, job 112827 received node haic-hgx-7 device minors 3,4,5,6.
Device minor 3, PCI 0000:13:00.0, UUID
GPU-40f85327-9a9e-df78-e1c1-720c0b07d78a returned NVML Unknown Error.
CUDA initialization failed even when masking that allocation to the UUIDs of
minors 5 and 6. This establishes an unusable CUDA allocation, not a definitive
diagnosis of the underlying hardware failure. Job 112804 on minors 0,1,2,7
continued normally and must not be reset or cancelled while repairing Door.
No node-wide exclusion or GPU reset is imposed by this launcher.

## Recorded conditions

- Ball and Soft pull: `joint_target` reads
  `raw.commanded_target.joint_rad`, and requires `arm_sent=true`.
  Seven normalized target joints occupy channels 0..6; 7..13 are exactly zero.
- Refreshed Door: `joint_target_export` reads `target_joint_action[:7]`,
  with the same seven-joint packing. The export also has target EEF in
  `target_action`; this is used to locate the strike, not as a model condition.
- Stick: `eef_target` reads the eight-channel target EEF `action` export:
  xyz, quaternion wxyz, gripper. Channels 8..13 are exactly zero.
- No measured-action fallback, inferred target, additional measured state,
  quaternion-to-joint inversion, or future observed-state concatenation.
- Normalization uses min/max from original TRAIN episodes only. Constant
  channels become zero. Raw files are never changed.
- Door's refreshed export supplies both target fields for all 538 episodes;
  438 train and 100 test episodes retain their original split.

## Windows and sampling

Each GPU independently samples environments without replacement, then distinct
uniform TRAIN episodes within each environment, then a random valid start.
There is no level balancing or matching action identifier across environments.

| Task | Per GPU | Frames / stride | Width x height | Curriculum |
| --- | --- | --- | --- | --- |
| Ball | 3 x 8 | 61 / 1 | 320 x 160 | 4 + 4 |
| Door | 3 x 8 | 49 / 1 | 256 x 192 | 5 + 5 |
| Stick | 3 x 8 | 41 / 3 | 256 x 192 | 4 + 4 |
| Soft | 5 x 6 | 33 / 3 | 320 x 160 | 5 + 5 + 5 |

Use letterboxing, never anisotropic stretching. Video and targets share native
indices `min(start + stride*k, episode_length-1)` including tail padding.
All short episodes are retained. Source train/test splits remain frozen.
Soft environments 3l, 6m, 4l, 7r are entirely excluded from training/statistics.
See `real_97_chunk_sampling.md` for the original temporal branch definitions.
The confirmed Ball revision is B = first target crossing of the initial/middle
angle during the right strike, minus TWO native frames (previously three).
Sample 50% uniformly from [right_start, B] and 50% from [0, B]. All 320 episodes
now have nonempty intervals, including the 64 formerly empty fast episodes.
No fallback or episode pruning is needed. Door's analogous L and B are now
derived from recorded `target_action`, with B = zero crossing minus one frame.
The exact revised events/ranges are frozen in each run's `chunk_events.jsonl`;
original dataset event sidecars and the already submitted Stick/Soft manifests
are left unchanged.

## Schedule, light augmentation and storage

Start from official `ckpt/BLM/step-12000.safetensors`. C32 entries are independent
uniform [-1,1] random vectors, not physical labels. Model LR is 1e-5 with a
100-step warmup; the first 300 steps update model only. Each 1000-step round is
new-C 200 at .15, all-C 200 at .03, model 200, all-C 200, model 200.
After the final curriculum, alternate C 200 / model 200 up to 5500 updates.
Model weight decay is .01; C decay is zero. No displacement projection,
optimizer-state reset or model-phase rewarmup. Fixed training diagnostic loss
is logged every 100 steps without light augmentation; it is not held-out loss.

No ROI weighting. Online light augmentation probability is .70. Gain .88..1.12,
contrast/gamma .90..1.10, tint .97..1.03, offset +/-.015, spatial gx/gy +/-.08,
noise std .004. Parameters and noise are fixed over the sampled episode chunk;
the remaining .30 uses the original input. No offline video copies or VAE-cache
invalidation pass is needed; augmentation precedes the normal VAE encoding.

At every model-phase end save a paired model/C checkpoint. Keep the newest two
ordinary pairs. Step 2300 is protected separately. At allocation start + 84600
seconds (23:30), request a complete, permanently protected model/current-C pair.
Ranks make one synchronized decision at an optimizer boundary: a running CUDA
step or checkpoint write is not interrupted. Therefore the deadline is derived
from Slurm allocation time exactly, but durable completion is after the current
step and disk publication, not guaranteed to the second. Only a scalar DDP
decision is added per update. Model/data caches are reused on node-local /tmp.
