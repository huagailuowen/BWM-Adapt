# Support-number ablation: detailed metrics

| Task | Setting | K | PSNR higher | SSIM higher | LPIPS lower | Object mean lower | Object final lower | Physical error lower | Action success higher |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Event80 | K1 informative | 1 | 31.976 | 0.9513 | 0.0495 | 3.854 | 7.593 | -- | 72% |
| Event80 | K2 mean loss | 2 | 31.839 | 0.9513 | 0.0507 | 4.554 | 9.904 | -- | 60% |
| Event80 | K4 mean loss | 4 | 31.895 | 0.9513 | 0.0500 | 4.215 | 9.113 | -- | 64% |
| Event80 | K2 sum loss | 2 | 31.973 | 0.9515 | 0.0492 | 3.650 | 7.397 | -- | 64% |
| Event80 | K4 sum loss | 4 | 31.738 | 0.9510 | 0.0505 | 4.184 | 8.339 | -- | 68% |
| Light Switch | K1 red-only | 1 | 32.089 | 0.9403 | 0.0196 | 0.083 | 0.146 | 10.500 | 50% |
| Light Switch | K2 red1+blue1 | 2 | 33.517 | 0.9442 | 0.0150 | 0.034 | 0.047 | 2.317 | 87.5% |
| Light Switch | K4 color-state coverage | 4 | 33.532 | 0.9443 | 0.0150 | 0.034 | 0.047 | 2.317 | 87.5% |
| Light Switch | K8 original formal | 8 | 33.454 | 0.9443 | 0.0150 | 0.036 | 0.050 | 2.300 | 87.5% |
| Mass Balance | K1 nearest-unbalanced | 1 | 32.662 | 0.9593 | 0.0246 | 1.199 | 3.135 | 0.739 | 70% |
| Mass Balance | K2 unbalanced bracket | 2 | 32.814 | 0.9601 | 0.0240 | 1.164 | 3.242 | 0.640 | 80% |
| Mass Balance | K2 unbalanced+balanced | 2 | 32.692 | 0.9594 | 0.0244 | 1.256 | 2.925 | 0.698 | 80% |
| Mass Balance | K4 mixed | 4 | 32.921 | 0.9609 | 0.0229 | 1.124 | 3.065 | 0.494 | 90% |

- Event80 object: pushed-block centroid ADE/FDE (px); no separate physical-error column.
- Light Switch object: yellow-light score MAE/final absolute error; physical: transition-time absolute error (frames).
- Mass Balance object: bar-centroid ADE/FDE (px); physical: beam-tilt MAE (deg).
- Event80 K=2/K=4 mean/sum rows retain support-query overlap and are diagnostic, not formal disjoint evaluations.
