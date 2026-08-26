# Simulation Transfer Metrics

`evaluate_sim_transfer_metrics.py` evaluates the same frozen transfer-plan
queries used by action selection. It does not run model inference and does not
change training or legacy inference behavior.

## Metrics

- Global appearance: PSNR, SSIM, and optional official LPIPS.
- Object-centric: per-object centroid ADE/FDE in pixels and normalized image
  coordinates, plus missing-track rate.
- Task-specific: target-object kinematics, mass-balance bar angle, or
  LightSwitch lamp state and transition timing.
- Action selection remains in `evaluate_sim_action_selection.py` and reports
  task success, oracle agreement, and oracle regret.

For collision and joint mass-friction, both the red target and blue driver are
tracked. The headline `centroid_*` metrics aggregate only object 0, the red
struck or pushed target. Driver metrics are retained as named diagnostics.

Mass-balance metadata with `frame_stride: 2` is preserved in the frozen
manifest. GT and generated frames therefore remain time-aligned.

## Usage

CPU-safe PSNR, SSIM, and physical metrics:

```bash
uv run --no-sync python scripts/evaluation/evaluate_sim_transfer_metrics.py \
  --config configs/evaluation/action_tasks/gravity80_ours_89519.yaml
```

LPIPS is explicitly restricted to a scheduled compute node:

```bash
srun -A yejin -p yejin-lo --gres=gpu:h100:1 --pty bash
uv run --no-sync python scripts/evaluation/evaluate_sim_transfer_metrics.py \
  --config CONFIG.yaml --lpips --lpips-device cuda
```

## Output layout

For an action evaluation directory `METHOD/action_evaluation`, the default
comparison output is its sibling `METHOD/video_metrics`:

```text
video_metrics/
  manifest.jsonl
  protocol.json
  skipped.jsonl
  global/global_per_query.jsonl
  global/global_summary.json
  object_centric/object_summary.json
  task_specific/task_per_query.jsonl
  task_specific/task_summary.json
  states/sim_rgb_v1/ground_truth/*.npz
  states/sim_rgb_v1/prediction/*.npz
```

The global, object-centric, task-specific, and action-selection stages all use
the same transfer plan and query identities.
