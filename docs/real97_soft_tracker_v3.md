# Real97 Soft whole-object tracker v3

This is an opt-in evaluation-only change. No training or model inference starts.

## Corrections

The search starts at y=0, not y=70. Boundary flags use actual image borders.
The complete silver terminal body is the target, not its holes, rope or gripper.
Eleven visually reviewed ID TRAIN source views cover front, rear and oblique
appearance. Mirroring, mild rotation and scale variants improve pose coverage.
The detector rejects strongly colored patches to reduce gripper/tape confusion.

Between detections, object-interior corners undergo forward/backward optical flow
and RANSAC rigid motion fitting. Scale, rotation, displacement, inlier fraction
and appearance against the last independent image anchor must agree. Detection
is attempted every four frames and following tracking failure. An anchor can
support at most twelve frames without a new independent detection. Missing or
disagreeing measurements remain missing; positions are not forward-filled.

## Quality semantics

Template acceptance remains NCC >=0.5; the threshold is not reduced to inflate
coverage. Optical-flow confidence is a geometric inlier fraction, not NCC and
not a calibrated correctness probability. Separate appearance, forward/backward
error, detector source, anchor age and boundary fields are retained.

The metrics layer honors an explicit measurement_valid flag. Boundary-truncated
objects are retained as censored/review cases, not silently counted as fully
observed centers. Older records without this flag keep their previous behavior.

The eleven coarse visual landmarks check both isolated-frame detection and
sequential tracking. Yellow marks show visual labels, green shows detection,
magenta shows sequential tracking. These approximate development annotations
are not final human-certified gold. The high-image-location probe uses a held-out
environment, but no template is taken from that environment.

## Execution

~~~bash
sbatch scripts/evaluation/run_real97_soft_calibration_v3_cpu.sh
~~~

The audit covers all 354 Soft episodes on four CPUs, zero GPUs. It keeps
per-frame tracks, source IDs, split/domain labels, overlays and review queues.
Do not infer localization accuracy from detection coverage. The v2 full-audit
outputs and configuration remain available for comparison.

