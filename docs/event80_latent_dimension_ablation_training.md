# Event80 Latent Dimension Ablation: Standard Training Launch

## Purpose

This protocol trains the Event80 latent-dimension ablation with `C=4`, `C=8`,
`C=64`, and `C=128`. All runs use the same data, active environments,
curriculum, optimization settings, compute allocation, and random seed. The
only intended experimental variable is `physical_context_dim`.

## Configurations

| Variant | Configuration | Output family |
|---|---|---|
| `C=4` | `configs/train/train_push_box_event80_iterative_c4_2gpu_24h.yaml` | `ours_context_dim_4` |
| `C=8` | `configs/train/train_push_box_event80_iterative_c8_2gpu_24h.yaml` | `ours_context_dim_8` |
| `C=64` | `configs/train/train_push_box_event80_iterative_c64_2gpu_24h.yaml` | `ours_context_dim_64` |
| `C=128` | `configs/train/train_push_box_event80_iterative_c128_2gpu_24h.yaml` | `ours_context_dim_128` |

Shared launcher:

`jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh`

## Fixed Training Protocol

| Setting | Value |
|---|---|
| Dataset | Event80, fixed frames `65-105` |
| Active environments | 35, matched to reference run `88823` |
| GPUs | 2 |
| Per-rank grouped batch | 4 environments x 4 actions |
| Effective clips per update | 32 |
| Environment-code initialization | Independent `U(0, 1)` vectors |
| Environment-code LR | `0.03` |
| New-environment code LR | `0.15` |
| Wan model LR | `1e-5` |
| Model LR warmup | 100 updates |
| Initial model-only phase | 300 updates |
| New-code phase | 200 updates |
| All-code phase | 200 updates |
| Model phase | 200 updates |
| Post-curriculum alternation | 200 code / 200 model updates |
| Projection | MLP, hidden width 128 |
| Seed | `20260708` |
| Wall time | 24 hours |
| Checkpoint retention | Latest 2 |

The launcher verifies the generated 35-environment pool against:

`configs/methods/event80/manifests/ours_88823_step7272_active35.yaml`

It aborts rather than silently training on a different active pool.

## Standard Launch Commands

Run from the repository root:

```bash
cd /afs/ir/users/c/y/cyzhou05/TTT-Physics/repos/BWM-Adapt
```

Launch an individual variant:

```bash
sbatch --no-requeue --job-name=event80-c4 \
  jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh c4

sbatch --no-requeue --job-name=event80-c8 \
  jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh c8

sbatch --no-requeue --job-name=event80-c64 \
  jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh c64

sbatch --no-requeue --job-name=event80-c128 \
  jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh c128
```

Launch all four and retain their job IDs:

```bash
jid_c4=$(sbatch --parsable --no-requeue --job-name=event80-c4 \
  jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh c4)
jid_c8=$(sbatch --parsable --no-requeue --job-name=event80-c8 \
  jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh c8)
jid_c64=$(sbatch --parsable --no-requeue --job-name=event80-c64 \
  jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh c64)
jid_c128=$(sbatch --parsable --no-requeue --job-name=event80-c128 \
  jobs/run_train_event80_active35_representation_iterative_2gpu_high_24h.sh c128)
printf 'c4=%s c8=%s c64=%s c128=%s\n' \
  "$jid_c4" "$jid_c8" "$jid_c64" "$jid_c128"
```

## Outputs and Logs

The launcher appends the Slurm job ID to each run directory. For example, the
`C=8` run with job ID `<JOB_ID>` writes to:

```text
outputs/method_benchmarks/pushbox_friction_event80/
  ours_context_dim_8/seed_20260708_job_<JOB_ID>/checkpoints/
```

Each output directory also contains:

| File | Meaning |
|---|---|
| `launch_config.yaml` | Exact resolved training configuration |
| `active_pool_manifest.json` | Active 35 environments and curriculum order |
| `step-*.safetensors` | Trainable model weights |
| `step-*.context_table.json` | Environment-code table paired with the checkpoint |

Slurm logs are written to:

```text
logs/methods/<job-name>-<job-id>.out
logs/methods/<job-name>-<job-id>.err
```

## Monitoring

```bash
squeue -j "$jid_c4,$jid_c8,$jid_c64,$jid_c128" \
  -o '%.10i %.20j %.9T %.10M %.20R %.16b'
```

Inspect one run without scanning unrelated logs:

```bash
tail -n 50 "logs/methods/event80-c8-${jid_c8}.out"
tail -n 20 "logs/methods/event80-c8-${jid_c8}.err"
```

A valid startup must report the reference-matched active-35 curriculum order,
two visible GPUs, and increasing `[train] step=...` records.

## Interruption and Resume Policy

Always submit these fresh runs with `--no-requeue`. The launcher creates a new
job-specific output directory and does not implement automatic optimizer or
context-table restoration. Direct Slurm requeue is therefore not a valid
resume mechanism.

For an interrupted run, preserve a matched checkpoint pair from the same step:

```text
step-N.safetensors
step-N.context_table.json
```

Create a dedicated resume configuration that loads both files, sets the
logical resume step to `N`, and writes back to a protected run directory. Do
not resume from the model checkpoint alone: doing so loses the learned
environment-code table and invalidates the ablation.

## Fair-Comparison Requirements

Do not change the active pool, curriculum order, seed, batch structure,
training duration, projection type, or learning rates for only one dimension.
Evaluation must use the same support/query manifest, inference schedule, and
checkpoint-selection rule for all four variants.
