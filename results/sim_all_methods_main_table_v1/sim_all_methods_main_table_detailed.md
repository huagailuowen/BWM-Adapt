# Simulation benchmark: detailed metrics

| Task | Method | PSNR (up) | SSIM (up) | LPIPS (down) | Object / Physical Metric | Mean Error (down) | Final Error (down) | Action Score (up) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Push-box Friction | Standard Pooled WM | 31.113 | 0.9513 | 0.0667 | Object Centroid (px) | 11.0626 | 24.4116 | 32.0% |
| Push-box Friction | LoRA TTA | 32.031 | 0.9535 | 0.0582 | Object Centroid (px) | 8.1902 | 19.1902 | 48.0% |
| Push-box Friction | DINOv2 Context | 30.697 | 0.9495 | 0.0681 | Object Centroid (px) | 11.8142 | 26.1703 | 48.0% |
| Push-box Friction | TTT-KQV | 30.779 | 0.9488 | 0.0649 | Object Centroid (px) | 10.8137 | 23.8675 | 32.0% |
| Push-box Friction | Ours | 31.976 | 0.9513 | 0.0495 | Object Centroid (px) | 3.8540 | 7.5931 | 72.0% |
| Gravity | Standard Pooled WM | 32.509 | 0.9523 | 0.0345 | Object Centroid (px) | 21.4246 | 34.8779 | 23.5% |
| Gravity | LoRA TTA | 33.328 | 0.9541 | 0.0231 | Object Centroid (px) | 7.9005 | 13.7137 | 47.1% |
| Gravity | DINOv2 Context | 34.095 | 0.9564 | 0.0195 | Object Centroid (px) | 5.9300 | 8.8290 | 76.5% |
| Gravity | TTT-KQV | 33.308 | 0.9549 | 0.0291 | Object Centroid (px) | 18.9572 | 30.9355 | 35.3% |
| Gravity | Ours | 33.389 | 0.9504 | 0.0202 | Object Centroid (px) | 7.3408 | 11.0653 | 82.4% |
| Mass Collision | Standard Pooled WM | 29.805 | 0.9623 | 0.0904 | Object Centroid (px) | 21.9329 | 25.7364 | 38.9% |
| Mass Collision | LoRA TTA† | 30.049 | 0.9619 | 0.0691 | Object Centroid (px) | 15.8704 | 24.5877 | 33.3% |
| Mass Collision | DINOv2 Context | 31.039 | 0.9668 | 0.0540 | Object Centroid (px) | 8.5264 | 10.2101 | 38.9% |
| Mass Collision | TTT-KQV | 29.385 | 0.9600 | 0.0842 | Object Centroid (px) | 20.9753 | 29.2930 | 55.6% |
| Mass Collision | Ours | 31.150 | 0.9625 | 0.0538 | Object Centroid (px) | 6.4522 | 8.1321 | 61.1% |
| Light Switch | Standard Pooled WM | 32.619 | 0.9480 | 0.0187 | Lamp Intensity (score) | 0.1035 | 0.1882 | 50.0% |
| Light Switch | LoRA TTA | 33.123 | 0.9485 | 0.0175 | Lamp Intensity (score) | 0.0828 | 0.1477 | 50.0% |
| Light Switch | DINOv2 Context | 33.074 | 0.9523 | 0.0160 | Lamp Intensity (score) | 0.1000 | 0.1837 | 50.0% |
| Light Switch | TTT-KQV | 32.692 | 0.9523 | 0.0161 | Lamp Intensity (score) | 0.1048 | 0.1887 | 50.0% |
| Light Switch | Ours | 33.517 | 0.9442 | 0.0150 | Lamp Intensity (score) | 0.0343 | 0.0474 | 87.5% |
| Mass Balance | Standard Pooled WM | 32.969 | 0.9608 | 0.0244 | Bar Tilt (deg) | 1.1844 | 4.5571 | 40.0% |
| Mass Balance | LoRA TTA | 24.895 | 0.8858 | 0.0631 | Bar Tilt (deg) | 1.7975 | 7.3105 | 50.0% |
| Mass Balance | DINOv2 Context | 33.263 | 0.9621 | 0.0239 | Bar Tilt (deg) | 1.2636 | 4.9480 | 30.0% |
| Mass Balance | TTT-KQV | 32.114 | 0.9565 | 0.0312 | Bar Tilt (deg) | 2.5075 | 10.1759 | 30.0% |
| Mass Balance | Ours | 32.814 | 0.9601 | 0.0240 | Bar Tilt (deg) | 0.6403 | 2.0842 | 80.0% |
| Mass x Friction | Standard Pooled WM | 31.077 | 0.9648 | 0.0610 | Object Centroid (px) | 16.0219 | 29.2454 | 35.0% |
| Mass x Friction | LoRA TTA | 29.865 | 0.9611 | 0.0701 | Object Centroid (px) | 16.8282 | 26.9302 | 26.7% |
| Mass x Friction | DINOv2 Context | 31.367 | 0.9664 | 0.0565 | Object Centroid (px) | 11.8116 | 19.0756 | 60.8% |
| Mass x Friction | TTT-KQV | 29.739 | 0.9614 | 0.0726 | Object Centroid (px) | 18.9511 | 29.1411 | 33.3% |
| Mass x Friction | Ours | 30.673 | 0.9631 | 0.0500 | Object Centroid (px) | 8.1481 | 13.7222 | 69.7% |
| Multi-background | Standard Pooled WM | 28.201 | 0.8046 | 0.1059 | Object Centroid (px) | 8.0260 | 16.1831 | 45.8% |
| Multi-background | LoRA TTA | 27.986 | 0.8007 | 0.1074 | Object Centroid (px) | 6.5805 | 16.2328 | 58.3% |
| Multi-background | DINOv2 Context | 28.000 | 0.8101 | 0.1064 | Object Centroid (px) | 7.7367 | 16.0065 | 58.3% |
| Multi-background | TTT-KQV | 27.691 | 0.7902 | 0.1116 | Object Centroid (px) | 7.7783 | 15.5869 | 58.3% |
| Multi-background | Ours* | 28.804 | 0.8204 | 0.0888 | Object Centroid (px) | 3.7145 | 6.3403 | 66.7% |

## Scope and provenance

- Multi-background methods are evaluated on the same original 5-background data with 5 ID and 5 OOD environments, K=1, and 90 disjoint queries. Baseline checkpoints were trained on matched-physics 30-background data with 20 active environments, whereas Ours* was trained on the original 5-background data; training data are not matched, so exclude this block from controlled-method ranking. Video metrics are multiview, object errors are centroid distances in pixels, and Action Success averages 24 GT-reachable complete-candidate decisions.
- Push-box Friction short-range success uses normalized image-y 0.595-0.68; if no predicted action reaches the long-range target, selection maximizes predicted image-y rounded to three decimals and breaks ties toward the higher action level.
- Lower is better for LPIPS and the task-specific object/physical metric; higher is better for Action Success.
- DINOv2 uses Transformer fusion for Push-box Friction and concat-MLP fusion for Gravity, Mass Collision, Light Switch, and Mass Balance; the completed Mass x Friction DINOv2 evaluation is included.
- Mass Balance uses workspace-random no-leak data for every method, with 5 ID and 5 OOD environments, shared K=2 bidirectional near-balance supports and 13 disjoint queries per environment; Standard pooled does not consume support. Ours uses step 4300.
- Mass Collision LoRA values marked with a dagger come from the earlier compatible no-leak balanced-support run rather than the high-mass-2x result root.
- A double dash denotes an unfinished or unavailable evaluation, not zero performance.
- Light Switch uses the fixed red-one plus blue-one K=2 support protocol for Ours, DINOv2, LoRA-TTA, and TTT-KQV; Standard pooled does not consume support.

- Object/physical errors use each task's named metric; Mean covers the full sequence and Final the last frame.
- Action Score uses each task's recorded protocol; Mass x Friction uses first-crossing +/-1 level match.
- * Mass Balance Ours: fixed-pose; baselines: workspace-random. Not a matched-dataset comparison.
- Dagger: Mass Collision LoRA uses the earlier recorded protocol. Missing results are --, not zero.
- Historical runs and their recorded settings are retained. See protocol.json for provenance and caveats.

Detailed numeric data: `sim_all_methods_main_table_detailed.csv`.
Full recorded configuration and sources: `protocol.json`.
