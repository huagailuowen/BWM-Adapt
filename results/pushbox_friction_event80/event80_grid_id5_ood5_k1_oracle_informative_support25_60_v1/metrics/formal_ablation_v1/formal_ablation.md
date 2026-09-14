# Event80 Formal Ablation Study

User selected candidates 1 (700-step cycle only), 3, 4, 6, 8, 9, and 10; include Ours as the reference.

| Method | Object ADE (px) lower | Object FDE (px) lower | PSNR (MV) higher | SSIM (MV) higher | LPIPS (MV) lower | Action success higher |
| --- | --- | --- | --- | --- | --- | --- |
| Ours (random C32) | 3.85 | 7.59 | 31.976 | 0.9513 | 0.0495 | 68% |
| Joint model-Z (700-step cycle) | 7.03 | 15.66 | 31.203 | 0.9465 | 0.0657 | 64% |
| No curriculum + joint | 5.68 | 12.28 | 31.697 | 0.9507 | 0.0547 | 60% |
| No curriculum + iterative | 6.17 | 12.81 | 31.799 | 0.9514 | 0.0549 | 56% |
| C=4 + MLP |  |  |  |  |  |  |
| Direct token (3072-D) |  |  |  |  |  |  |
| C32 shared initialization | 7.94 | 15.98 | 30.638 | 0.9466 | 0.0631 | 56% |
| C32 random [-0.05, 0.05] | 4.81 | 9.73 | 32.172 | 0.9509 | 0.0495 | 60% |

## Protocol and scope

Event80; 5 ID + 5 OOD environments; K=1 informative support; nine disjoint queries per environment.
Metrics are copied from the recorded formal scoreboard. No new rollout or metric computation is performed.

- Existing reported metrics are reused; no video generation or metric recomputation.
- Blank metric cells mean unavailable or pending, never zero.
- Historical initialization runs retain their recorded training schedules and are not claimed to be strict single-variable controls.
- C4 and direct-token jobs use two GPUs, 4 environments x 4 actions per rank, the original active35 pool, and the 1000-step alternating curriculum.
- The 1000-step pure-joint variant, C1, C128, new-C200/joint800, shuffled groups, and per-trajectory codes are not selected for this formal table.

## Run provenance

| Method | Checkpoint step | Source job | Status | Setting |
| --- | --- | --- | --- | --- |
| Ours (random C32) | 7272 | 88823 | scored | Original random-C32 alternating curriculum; 1000 steps per wave. |
| Joint model-Z (700-step cycle) | 4200 |  | scored | Progressive curriculum; model and environment codes update jointly; 700 steps per wave. |
| No curriculum + joint | 5200 |  | scored | All 35 training environments active from the beginning; joint model/code updates. |
| No curriculum + iterative | 7000 |  | scored | All 35 training environments active from the beginning; alternating 200-step model/code blocks. |
| C=4 + MLP |  | 115984 | training_incomplete | Independent U(0,1) 4-D codes; standard 1000-step alternating curriculum; active35. |
| Direct token (3072-D) |  | 115985 | training_incomplete | One 3072-D code per environment; identity projection; Gaussian initialization with mean 0 and std 0.02; standard alternating curriculum. |
| C32 shared initialization | 6814 | 88822 | scored | Historical shared-initialization C32 run; retain the recorded training configuration. |
| C32 random [-0.05, 0.05] | 7000 | 89030 | scored | Historical small-range random-initialization C32 run; retain the recorded training configuration. |

Source scores: `results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1/metrics/complete_v1/scoreboard.csv`.
Support/query identities: `results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1/protocol/support_query_manifest.json`.
Machine-readable selected runs and the source protocol snapshot are in `protocol.json`.
