# World-model evaluation

This directory contains additive evaluation entry points. It does not modify
generation, training, or legacy inference code.

Both evaluators consume one frozen JSONL manifest. Global evaluation reads GT
and predicted RGB videos. Object-centric evaluation reads precomputed GT and
prediction masks; segmentation and tracking are deliberately separate so every
method is measured with the same frozen evaluator.

Action, planning, and task-success evaluation is intentionally not implemented
until its protocol is finalized.

`evaluate_event80_legacy_infer.py` is a smoke adapter for the historical
Event80 run 89097. It evaluates the main camera, wrist camera, combined RGB,
and a frozen color/trajectory block tracker. The segmentation mask is used only
to extract the object centroid and is not itself scored. The primary object
metric is the per-frame Euclidean distance between GT and predicted centroids.
Off-screen blocks are represented by the fixed bottom-center sentinel
`(x=0.5, y=1.0)`. The historical run reused
the adapted episode for generation, so its outputs are explicitly labeled
`legacy_same_episode_smoke` and must not be reported as the final disjoint
support/query benchmark.
