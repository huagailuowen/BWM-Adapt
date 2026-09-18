# Simulation benchmark: detailed metrics

| Task | Method | PSNR (up) | SSIM (up) | LPIPS (down) | Object metric (down) | Action Score (up) |
| --- | --- | --- | --- | --- | --- | --- |
| Push-box Friction | Standard Pooled WM | 31.113 | 0.9513 | 0.0667 | 11.0626 px | 32.0% |
| Push-box Friction | LoRA TTA | 32.031 | 0.9535 | 0.0582 | 8.1902 px | 40.0% |
| Push-box Friction | DINOv2 Context | 30.697 | 0.9495 | 0.0681 | 11.8142 px | 48.0% |
| Push-box Friction | TTT-KQV | 30.779 | 0.9488 | 0.0649 | 10.8137 px | 32.0% |
| Push-box Friction | Ours | 31.976 | 0.9513 | 0.0495 | 3.8540 px | 68.0% |
| Gravity | Standard Pooled WM | 32.509 | 0.9523 | 0.0345 | 21.4246 px | 23.5% |
| Gravity | LoRA TTA | 33.328 | 0.9541 | 0.0231 | 7.9005 px | 47.1% |
| Gravity | DINOv2 Context | 34.390 | 0.9568 | 0.0164 | 3.8973 px | 88.2% |
| Gravity | TTT-KQV | 33.308 | 0.9549 | 0.0291 | 18.9572 px | 35.3% |
| Gravity | Ours | 33.389 | 0.9504 | 0.0202 | 7.3408 px | 82.4% |
| Mass Collision | Standard Pooled WM | 29.805 | 0.9623 | 0.0904 | 21.9329 px | 38.9% |
| Mass Collision | LoRA TTA† | 30.049 | 0.9619 | 0.0691 | 15.8704 px | 33.3% |
| Mass Collision | DINOv2 Context | 31.039 | 0.9668 | 0.0540 | 8.5264 px | 38.9% |
| Mass Collision | TTT-KQV | 29.385 | 0.9600 | 0.0842 | 20.9753 px | 55.6% |
| Mass Collision | Ours | 31.150 | 0.9625 | 0.0538 | 6.4522 px | 61.1% |
| Light Switch | Standard Pooled WM | 32.619 | 0.9480 | 0.0187 | 0.1035 score | 50.0% |
| Light Switch | LoRA TTA | 34.381 | 0.9522 | 0.0134 | 0.0381 score | 87.5% |
| Light Switch | DINOv2 Context | 33.074 | 0.9523 | 0.0160 | 0.1000 score | 50.0% |
| Light Switch | TTT-KQV | 32.696 | 0.9523 | 0.0161 | 0.1046 score | 50.0% |
| Light Switch | Ours | 33.454 | 0.9443 | 0.0150 | 0.0358 score | 87.5% |
| Mass Balance | Standard Pooled WM | 32.921 | 0.9605 | 0.0248 | 1.2441 deg | 40.0% |
| Mass Balance | LoRA TTA | 25.017 | 0.8865 | 0.0628 | 1.9245 deg | 40.0% |
| Mass Balance | DINOv2 Context | 33.252 | 0.9621 | 0.0240 | 1.2733 deg | 40.0% |
| Mass Balance | TTT-KQV | 32.167 | 0.9568 | 0.0305 | 2.3579 deg | 30.0% |
| Mass Balance | Ours* | 33.509 | 0.9639 | 0.0218 | 0.7617 deg | 70.0% |
| Mass x Friction | Standard Pooled WM | 31.077 | 0.9648 | 0.0610 | 16.0219 px | 35.0% |
| Mass x Friction | LoRA TTA | 29.865 | 0.9611 | 0.0701 | 16.8282 px | 26.7% |
| Mass x Friction | DINOv2 Context | 31.367 | 0.9664 | 0.0565 | 11.8116 px | 60.8% |
| Mass x Friction | TTT-KQV | -- | -- | -- | -- px | -- |
| Mass x Friction | Ours | 30.673 | 0.9631 | 0.0500 | 8.1481 px | 69.7% |

## Scope and provenance

- Lower is better for LPIPS and the task-specific object/physical metric; higher is better for Action Success.
- DINOv2 uses Transformer fusion for Push-box Friction and concat-MLP fusion for Gravity, Mass Collision, Light Switch, and Mass Balance; the completed Mass x Friction DINOv2 evaluation is included.
- Mass Balance Ours values marked with an asterisk use the completed fixed-pose 5-ID/5-OOD nearest-unbalanced-support test, while baseline rows use workspace-random data; available values participate in column ranking.
- Mass Collision LoRA values marked with a dagger come from the earlier compatible no-leak balanced-support run rather than the high-mass-2x result root.
- A double dash denotes an unfinished or unavailable evaluation, not zero performance.

- Object metric: centroid ADE (px); Light Switch: lamp MAE; Mass Balance: bar-tilt MAE (deg).
- Action Score uses each task's recorded protocol; Mass x Friction uses first-crossing +/-1 level match.
- * Mass Balance Ours: fixed-pose; baselines: workspace-random. Not a matched-dataset comparison.
- Dagger: Mass Collision LoRA uses the earlier recorded protocol. Missing results are --, not zero.
- Historical runs and their recorded settings are retained. See protocol.json for provenance and caveats.

Detailed numeric data: `sim_all_methods_main_table_detailed.csv`.
Full recorded configuration and sources: `protocol.json`.
