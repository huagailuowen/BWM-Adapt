# Simulation benchmark: detailed metrics

| Task | Method | PSNR (up) | SSIM (up) | LPIPS (down) | Object / Physical Metric | Mean Error (down) | Final Error (down) | Action Score (up) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Push-box Friction | Standard Pooled WM | 31.113 | 0.9513 | 0.0667 | Object Centroid (px) | 11.0626 | 24.4116 | 32.0% |
| Push-box Friction | LoRA TTA | 32.031 | 0.9535 | 0.0582 | Object Centroid (px) | 8.1902 | 19.1902 | 40.0% |
| Push-box Friction | DINOv2 Context | 30.697 | 0.9495 | 0.0681 | Object Centroid (px) | 11.8142 | 26.1703 | 48.0% |
| Push-box Friction | TTT-KQV | 30.779 | 0.9488 | 0.0649 | Object Centroid (px) | 10.8137 | 23.8675 | 32.0% |
| Push-box Friction | Ours | 31.976 | 0.9513 | 0.0495 | Object Centroid (px) | 3.8540 | 7.5931 | 68.0% |
| Gravity | Standard Pooled WM | 32.509 | 0.9523 | 0.0345 | Object Centroid (px) | 21.4246 | 34.8779 | 23.5% |
| Gravity | LoRA TTA | 33.328 | 0.9541 | 0.0231 | Object Centroid (px) | 7.9005 | 13.7137 | 47.1% |
| Gravity | DINOv2 Context | 34.390 | 0.9568 | 0.0164 | Object Centroid (px) | 3.8973 | 5.8150 | 88.2% |
| Gravity | TTT-KQV | 33.308 | 0.9549 | 0.0291 | Object Centroid (px) | 18.9572 | 30.9355 | 35.3% |
| Gravity | Ours | 33.389 | 0.9504 | 0.0202 | Object Centroid (px) | 7.3408 | 11.0653 | 82.4% |
| Mass Collision | Standard Pooled WM | 29.805 | 0.9623 | 0.0904 | Object Centroid (px) | 21.9329 | 25.7364 | 38.9% |
| Mass Collision | LoRA TTA† | 30.049 | 0.9619 | 0.0691 | Object Centroid (px) | 15.8704 | 24.5877 | 33.3% |
| Mass Collision | DINOv2 Context | 31.039 | 0.9668 | 0.0540 | Object Centroid (px) | 8.5264 | 10.2101 | 38.9% |
| Mass Collision | TTT-KQV | 29.385 | 0.9600 | 0.0842 | Object Centroid (px) | 20.9753 | 29.2930 | 55.6% |
| Mass Collision | Ours | 31.150 | 0.9625 | 0.0538 | Object Centroid (px) | 6.4522 | 8.1321 | 61.1% |
| Light Switch | Standard Pooled WM | 32.619 | 0.9480 | 0.0187 | Lamp Intensity (score) | 0.1035 | 0.1882 | 50.0% |
| Light Switch | LoRA TTA | 34.381 | 0.9522 | 0.0134 | Lamp Intensity (score) | 0.0381 | 0.0559 | 87.5% |
| Light Switch | DINOv2 Context | 33.074 | 0.9523 | 0.0160 | Lamp Intensity (score) | 0.1000 | 0.1837 | 50.0% |
| Light Switch | TTT-KQV | 32.696 | 0.9523 | 0.0161 | Lamp Intensity (score) | 0.1046 | 0.1885 | 50.0% |
| Light Switch | Ours | 33.454 | 0.9443 | 0.0150 | Lamp Intensity (score) | 0.0358 | 0.0496 | 87.5% |
| Mass Balance | Standard Pooled WM | 32.921 | 0.9605 | 0.0248 | Bar Tilt (deg) | 1.2441 | 4.7822 | 40.0% |
| Mass Balance | LoRA TTA | 25.017 | 0.8865 | 0.0628 | Bar Tilt (deg) | 1.9245 | 7.7828 | 40.0% |
| Mass Balance | DINOv2 Context | 33.252 | 0.9621 | 0.0240 | Bar Tilt (deg) | 1.2733 | 4.9727 | 40.0% |
| Mass Balance | TTT-KQV | 32.167 | 0.9568 | 0.0305 | Bar Tilt (deg) | 2.3579 | 9.5113 | 30.0% |
| Mass Balance | Ours* | 33.509 | 0.9639 | 0.0218 | Bar Tilt (deg) | 0.7617 | 2.4564 | 70.0% |
| Mass x Friction | Standard Pooled WM | 31.077 | 0.9648 | 0.0610 | Object Centroid (px) | 16.0219 | 29.2454 | 35.0% |
| Mass x Friction | LoRA TTA | 29.865 | 0.9611 | 0.0701 | Object Centroid (px) | 16.8282 | 26.9302 | 26.7% |
| Mass x Friction | DINOv2 Context | 31.367 | 0.9664 | 0.0565 | Object Centroid (px) | 11.8116 | 19.0756 | 60.8% |
| Mass x Friction | TTT-KQV | -- | -- | -- | Object Centroid (px) | -- | -- | -- |
| Mass x Friction | Ours | 30.673 | 0.9631 | 0.0500 | Object Centroid (px) | 8.1481 | 13.7222 | 69.7% |

## Scope and provenance

- Lower is better for LPIPS and the task-specific object/physical metric; higher is better for Action Success.
- DINOv2 uses Transformer fusion for Push-box Friction and concat-MLP fusion for Gravity, Mass Collision, Light Switch, and Mass Balance; the completed Mass x Friction DINOv2 evaluation is included.
- Mass Balance Ours values marked with an asterisk use the completed fixed-pose 5-ID/5-OOD nearest-unbalanced-support test, while baseline rows use workspace-random data; available values participate in column ranking.
- Mass Collision LoRA values marked with a dagger come from the earlier compatible no-leak balanced-support run rather than the high-mass-2x result root.
- A double dash denotes an unfinished or unavailable evaluation, not zero performance.

- Object/physical errors use each task's named metric; Mean covers the full sequence and Final the last frame.
- Action Score uses each task's recorded protocol; Mass x Friction uses first-crossing +/-1 level match.
- * Mass Balance Ours: fixed-pose; baselines: workspace-random. Not a matched-dataset comparison.
- Dagger: Mass Collision LoRA uses the earlier recorded protocol. Missing results are --, not zero.
- Historical runs and their recorded settings are retained. See protocol.json for provenance and caveats.

Detailed numeric data: `sim_all_methods_main_table_detailed.csv`.
Full recorded configuration and sources: `protocol.json`.
