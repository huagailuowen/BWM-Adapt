# Real97 calibration and future evaluation, v2

## Current scope

Only calibration, object tracking, and measurement preparation are running.
No world-model inference, training, or support-latent optimization is started.
All batch video decoding runs in a CPU Slurm allocation, never on the login node.
Existing training configurations, dataset splits, and v1 evaluators are unchanged.

Configuration:
- `configs/evaluation/real97_object_calibration_v2.json`
- `configs/evaluation/real97_support_query_protocol_v2.json`

## Recorded future inference rules

Support is selected from eligible TRAIN episodes of the same environment.
Adapt latent Z once with the agreed support set, then freeze it across all queries.
Query ground truth is never used for latent updates or action selection.

All four tasks first support factual reconstruction: take a train or test
query chunk, its own first frame, and its original recorded action sequence.
Train query episodes must be distinct from support episodes. Report train-query
and held-out test-query metrics separately; do not pool them as generalization.
Stick and Soft prioritize this mode. Soft has no action-selection score.

Ball and Door additionally support a fixed TEST-initial-image action sweep:
keep image and adapted Z fixed, vary the subsequent target-action level.
Check the robot's initial state, target continuity, action units/channel order,
sampling stride, and coverage of the actual strike/push before allowing a sweep.
Blindly inserting an unrelated episode's target sequence can create an artificial
jump and is not an acceptable counterfactual. No action editing is launched now.

A different episode's real video is NOT frame-paired ground truth for a
counterfactual with a different initial state. Such a sweep can be inspected
and used for action selection, but trajectory error needs a genuinely matched
recording. Door's approved level sets remain an independent selection reference.
Ball reachability from the current recorded test candidates must not be
misrepresented as reachability under every newly constructed initial state.

## Native coordinate and temporal contract

Invert model letterboxing/resizing before tracking. Ball adds (+80,+156), Soft
adds (+32,+224) to restore original 640x480 camera coordinates. Door/Stick already
use original coordinates. Center errors are original pixels; normalized errors
divide by the original-image diagonal, 800 px. No change to Ball's target:
maximum x in [280,330], not displacement, not final position.

Video/action indices must exactly match training stride and representation.
Exclude conditioning frame(s) and repeated padding. Never align predictions to
GT with dynamic time warping. Never infer an unobserved episode maximum from a
truncated generation window. Missing objects are missing, not frozen at their
last visible position.

## Tracker corrections and remaining calibration

Soft v1 could initialize on a screw-hole pair plus background and accumulate
optical-flow drift. V2 independently localizes a reviewed terminal-metal-body
template every frame. The template bank contains TRAIN images with several
orientations, scales and rotations. It does not track the rope or gripper.
Ambiguous and missing matches are retained explicitly. Boundary cases require
review. Its center is a projected visible-body center, not a claim of recovering
a physical 3-D center of mass. Templates and development landmarks are annotated
visually by the assistant with approximate pixel coordinates, not certified
human gold. Independent frame annotations remain necessary for final accuracy.

Door v1 compared the outer door edge to the cabinet front-panel seam and ignored
door-panel thickness. A visually closed TRAIN frame shows about 14 px offset.
V2 records that offset, has an explicit uncertain band, and keeps physical closure
separate from accepted action-level correctness. This is provisional calibration,
not proof that all episodes close/open correctly. Validate thresholds on independent
open/closed/rebound frames; never use a gear label as a visual closing label.

Ball and Stick retain v1 detection while adding missing/jump/boundary diagnostics.
Ball target-boundary cases within 3 px require pixel-level review. Stick success
must require BOTH endpoint lift and acceptable tilt, not simply a horizontal bar
resting on the table. Lift/tilt success tolerances are not certified yet.

## Measurement outputs

`object_measurements.trajectory_metrics` computes frame-paired center ADE/FDE,
coverage, missing counts, normalized ADE, Ball observed peak-x error, and Stick
angle MAE. Final-frame error is absent if the actual last frame is not detected.
An error on only visible frames is explicitly not a full-horizon certificate.
The caller must supply the real, non-conditioning, non-padding frame mask.

The CPU audit saves every frame's measurements, per-episode outcomes, review
images, an HTML review index, a review queue, and a frozen calibration snapshot.
No failed episode is silently dropped. Soft support flags remain proposals until
identity and actual selected-chunk translation are reviewed. Train/test/OOD
identities remain in records; OOD Soft data are not used as ID support.

Next accuracy gates:
- Independently annotate object positions and Door open/closed/uncertain frames.
- Evaluate localization error, coverage and false detections, not coverage alone.
- Audit movement extrema, occlusion, rebound, out-of-frame and target boundaries.
- Check whole-episode versus actual training-compatible chunk horizon.
- Freeze threshold/calibration version before comparing model methods.
- Keep tracker validation on generated videos as a separate later gate.

## CPU execution

~~~bash
sbatch scripts/evaluation/run_real97_calibration_cpu.sh
~~~

This requests 4 CPUs / 16 GB / zero GPUs on `yejin-lo`, up to two hours.
Long pending/running jobs can be checked manually every 10 minutes.

