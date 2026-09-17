# Soft main result: shared six environments

12 held-out queries, two per environment.
Environments: soft-1l, soft-2l, soft-1r, soft-5r, soft-2m, soft-8.

| Method | ADE (px) | FDE (px) |
|---|---:|---:|
| Standard | 32.78 | 39.41 |
| DINOv2 concat-MLP | 21.19 | 29.17 |
| Ours Stage2 | 18.10 | 22.38 |
| Ours Stage1 (reference) | 17.69 | 22.66 |

Only ADE/FDE are primary metrics. No action-selection metric.
Same 28 future timestamps and same 12 queries for every method. Coordinates: 512x256 crop.
ADE uses a common validity mask including DINO; FDE requires the actual final frame.
This cohort was defined by Standard training coverage; promotion to main followed inspection of scores.
The full nine-environment result remains available and favors DINO in aggregate.
Standard has five repeated initial frames; Ours has one frame and a known-family initialization prior.
Tracking remains provisional. A [six-query visual spot check](visual_spot_check.md)
covered all shared environments and 120 GT/prediction panels, with no obvious
wrong-object detections. This is not an exhaustive frame-by-frame audit.

[Full nine-environment comparison](../../real97_soft_dino_matched_20260916_v1/README.md).
[Selected-query metrics](per_query.json).
