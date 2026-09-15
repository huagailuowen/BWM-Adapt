# Real97 main results

Compiled on 2026-09-14 from completed evaluations. This is a table-only
compilation, not a new inference or scoring run. Source outputs are unchanged.
Only held-out test-query metrics are shown; Support episodes come from train.

[Download the wide CSV table](real97_all_methods_main_table.csv).

## Methods and missing entries

- `Ours` means Stage2 inference, not direct Stage1 table lookup.
- Door and Ball use the accepted historical Ours reference runs, not the newer
  dual-latent experiments.
- Soft uses the accepted final family-mean initialization. Only Standard and
  Ours are included for Soft. Soft has no action-selection metric.
- Stick balance is intentionally left blank for every method.
- Blank entries mean unavailable or intentionally omitted, never zero.
- PSNR and SSIM are higher-is-better. LPIPS, ADE, and FDE are lower-is-better.
  ADE and FDE are object-center errors in pixels. Cross-task pixel errors
  should not be averaged into a single score.
- Displayed numbers retain the precision of the previously reported summaries.
  The source JSON files below contain the detailed evaluation records.

## Door close

100 held-out queries across 10 environments for image and object metrics.
Action overlap is macro-averaged over the 9 closable environments.

| Method | PSNR | SSIM | LPIPS | ADE (px) | FDE (px) | Action pair overlap (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 29.96 | 0.9115 | 0.0570 | 21.82 | 41.22 | 38.89 |
| LoRA TTA | 30.24 | 0.9150 | 0.04946 | 17.35 | 32.14 | 44.44 |
| DINOv2 concat-MLP | 30.28 | 0.9137 | 0.0549 | 21.12 | 38.90 | 44.44 |
| TTT | 30.95 | 0.9178 | 0.05164 | 19.87 | 35.93 | 38.89 |
| Ours | 30.15 | 0.9117 | 0.0476 | 14.07 | 23.08 | 77.78 |

Action selection scans levels in ascending order and chooses the first
consecutive pair for which both predicted rollouts close the door. The score
is the intersection size with the environment's accepted GT pair, divided by
two: 0, 0.5, or 1. This is an overlap score, not a binary success rate.
`door-12u-Half-11d-Half` is excluded only from action scoring.

## Ball friction

80 held-out queries across 8 environments. Action overlap is macro-averaged
over the 8 environments.

| Method | PSNR | SSIM | LPIPS | ADE (px) | FDE (px) | Action pair overlap (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 36.04 | 0.9605 | 0.03153 | 28.36 | 64.42 | 37.50 |
| LoRA TTA | 35.58 | 0.9591 | 0.03697 | 36.82 | 70.02 | 56.25 |
| DINOv2 concat-MLP | 36.31 | 0.9613 | 0.02942 | 25.52 | 54.33 | 37.50 |
| TTT | 36.29 | 0.9613 | 0.03238 | 29.66 | 67.32 | 50.00 |
| Ours | 34.16 | 0.9529 | 0.03282 | 21.74 | 40.55 | 56.25 |

For both GT and predictions, scan levels in ascending order and find the first
level whose farthest visible ball-center x reaches the right blue marker minus
30 original-image pixels. That level and its successor form a pair; pair
intersection size divided by two gives the score. Confirmed right-screen exits
retain the last visible center. A first-reaching level of 10 has the formal
successor label 11; this does not imply that a level-11 rollout was generated.

## Stick balance

Intentionally unfilled pending the selected evaluation results.

| Method | PSNR | SSIM | LPIPS | Object-centric metric | Action metric |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | | | | | |
| LoRA TTA | | | | | |
| DINOv2 concat-MLP | | | | | |
| TTT | | | | | |
| Ours | | | | | |

## Soft pull

18 held-out queries, two per environment across 9 environments. There is no
action metric for this task. LPIPS was computed using AlexNet v0.1 on RGB
frames resized to the native crop size of 512 x 256. It uses every aligned
future frame, independently of the object-tracking validity mask. The entries were filled after the compute-node scoring job completed.

| Method | PSNR | SSIM | LPIPS | ADE (px) | FDE (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 33.10 | 0.9644 | 0.07745 | 30.28 | 48.75 |
| Ours | 32.47 | 0.9608 | 0.07921 | 22.69 | 40.35 |

Ours starts from the mean of the trained latent rows in the known L, R, or 8
family; `1R` belongs to R. Both trained replicas of each environment are
included in that family's mean. No initialization noise is used. Z adaptation
uses four ten-step learning-rate stages: 0.3, 0.15, 0.05, and 0.015. It uses
training Supports only, with no hard Z bounds and no ROI weighting.

Standard uses five repeated initial frames and Ours uses one initial frame.
Metrics align the common 28 future timestamps, native frames 3, 6, ..., 84,
rather than aligning raw output-video indices. ADE retains the existing
common-valid-frame mask across Standard, Stage1, global-mean Stage2, and
family-mean Stage2. Only Standard and the final family-mean Ours are displayed;
the metric mask was not recomputed when hiding the other columns. FDE uses
the eligible final frame, not an earlier last-detected frame.

### Training-coverage caveat

Ours trained on all nine evaluated environments, but Standard did not train on
`soft-4l`, `soft-7r`, or `soft-6m`. The main Soft table therefore does not have
equal training-environment coverage. The same completed evaluation also reports
the six jointly seen environments, with 12 held-out queries:

| Method | PSNR | SSIM | LPIPS | ADE (px) | FDE (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 32.90 | 0.9642 | 0.07953 | 32.78 | 39.41 |
| Ours | 32.61 | 0.9615 | 0.07549 | 18.10 | 22.38 |

## Interpretation and limitations

- These are held-out episodes, not a common unknown-environment evaluation.
  Historical Door/Ball Ours uses environment-informed initialization, and final
  Soft Ours uses a known-family prior. The methods do not all receive equal
  environment information.
- Training budgets differ. Standard, DINO, and the Standard base for LoRA use
  step 5500. Ours uses Door step 3715, Ball step 5500, and Soft step 5500.
  TTT uses Door step 3656 and Ball step 4144.
- Tracking-derived scores remain provisional. Soft has sampled visual audits
  covering all nine environments, not exhaustive frame-by-frame human labels.
  Ball/Door retain their source calibration and audit caveats, including the
  nominal Door-4d GT preference pair [3, 4].
- A better image metric does not necessarily indicate better object motion.
  Small Soft Z updates also mean these results alone do not establish that
  support adaptation discovered the environment family.

## Metric sources

- [Door/Ball completed six-method comparison](metrics/ball_door_comparison.json)
- [Soft final family-mean comparison summary](metrics/soft_family_mean_comparison.json)
- [Soft LPIPS summary](soft_lpips/summary.json)
- [Soft per-query LPIPS](soft_lpips/per_query.json)
- [Door accepted reference protocol](../door_close/real97_train_support_train_test_query_v1/README.md)
- [Ball accepted reference protocol](../ball_friction/real97_train_support_train_test_query_v1/README.md)

The two comparison JSON files are unchanged snapshots of
`outputs/eval_real97_lora_reference_scores_20260914_v1/comparison_with_baselines.json`
and `outputs/eval_real97_soft_family_mean_comparison_20260914_v1/initialization_comparison_summary.json`.
Their embedded absolute paths record local provenance; referenced raw videos and
full object-tracking records remain local. No videos, model checkpoints, or
datasets are copied into this table release.
