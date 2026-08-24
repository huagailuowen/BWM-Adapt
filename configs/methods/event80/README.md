# Event80 method matrix

This directory defines the first controlled baseline and ablation matrix on the
PushBox Friction Event80 dataset. No file in this directory submits a job.

## Resource protocol

Every independently trained method receives the same resource envelope:

- two H200 GPUs;
- 24 training hours, or 48 H200 GPU-hours;
- a timer starting when the training process launches after common staging;
- the same pretrained Wan initialization, progressive environment stream,
  training split, and frozen evaluation protocol.

Optimizer steps, generator-gradient evaluations, clip exposure, and FLOPs are
not artificially matched. They are recorded as outcomes. GPU utilization is a
reported engineering diagnostic, not the equality metric. A method that cannot
saturate the allocated GPUs after reasonable tuning retains that systems cost.

Queue and common checkpoint/data staging time are excluded. The scheduler is
the authoritative deadline. The latest complete checkpoint written before the
deadline is evaluated; checkpoint retention remains two.

## Workflow

Build the non-executing matrix plan:

```bash
python scripts/methods/plan_matrix.py \
  --matrix configs/methods/event80/matrix.yaml
```

Validate one composed method config without loading Wan:

```bash
python scripts/methods/train_method.py \
  --config configs/methods/event80/ours/random_c32.yaml \
  --dry-run
```

Resolve an implementation-ready command without running it:

```bash
python scripts/methods/train_method.py \
  --config configs/methods/event80/ours/random_c32.yaml \
  --plan
```

The matrix must retain `do_not_submit: true` while any readiness gate is open.
Methods with missing Wan adapters are deliberately marked blocked rather than
being routed through an incorrect legacy experiment.

## Environment-code dimension ablation

The reference method uses `C=32`. Three additional runs isolate environment-code
capacity while preserving every other controlled variable:

| Experiment | C dimension | Initialization | Training budget |
| --- | ---: | --- | --- |
| `ours_context_dim_4` | 4 | independent `Uniform(0, 1)` per environment | 48 H200 GPU-hours |
| `ours_random_c32` | 32 | independent `Uniform(0, 1)` per environment | 48 H200 GPU-hours |
| `ours_context_dim_128` | 128 | independent `Uniform(0, 1)` per environment | 48 H200 GPU-hours |
| `ours_direct_environment_token` | 3072 (Wan2.2-TI2V-5B width) | independent `N(0, 0.02)` per environment | 48 H200 GPU-hours |

All four runs use the same Event80 windows, grouped batch, progressive stream,
alternating schedule, Wan initialization, optimizer settings, support/query
protocol, and evaluation manifest. The 4-D and 128-D runs change only
`physical_context_dim` relative to the projected 32-D reference. The direct-token
run instead uses `physical_context_dim=3072` and
`physical_context_projection=direct`, removing the projection MLP while keeping
the remaining controlled variables fixed. The three ablations are planned and
have not started training; they write to separate checkpoint directories.

## Evaluation

All methods use the same frozen, cross-action, disjoint support/query manifest.
K=1 and K=2 are separate adaptation episodes, and final metrics are averaged
over the fixed multi-action query set. New artifacts belong under:

```text
results/pushbox_friction_event80/event80_cross_action_k1_k2_v1/
```

Primary metrics are object-mask mean/final IoU, object-centroid ADE/FDE, and the
friction-specific stopping-distance error. Global PSNR and LPIPS are secondary.
