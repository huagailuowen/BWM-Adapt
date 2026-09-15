# Evaluation results

All new world-model evaluation artifacts live under this directory. Training
checkpoints remain under outputs; historical output directories are not moved.

The canonical layout is:

    results/<benchmark>/<evaluation_id>/
      protocol/
        execution_plan.json
        frozen_query_manifest.jsonl
      methods/<method>/<checkpoint_tag>/seed_<seed>/
        run_manifest.json
        predictions/
        masks/
        metrics/global/
        metrics/object_centric/
        visualizations/
      comparisons/
        metrics/
        tables/
        plots/
        videos/

Names must use letters, digits, dots, underscores, or hyphens. Evaluation IDs
should describe the frozen protocol, for example
event80_cross_action_k1_k2_v1, rather than using only a timestamp.

Generated videos, masks, and metric files are ignored by Git. The directory
schema itself remains version controlled.

Selected text-only metric releases are explicitly version controlled despite
the default ignore rule. Media, datasets, logs, and model checkpoints remain
excluded. The real97 Door and Ball releases are indexed at:

- [Door close](door_close/real97_train_support_train_test_query_v1/README.md)
- [Ball friction](ball_friction/real97_train_support_train_test_query_v1/README.md)
- [Real97 main results table](real97_all_methods_main_table_v1/README.md)

Training resource telemetry remains next to the corresponding method checkpoint
under `outputs/method_benchmarks/`. It records the declared fixed hardware-time
budget, actual elapsed time and GPU-hours, sampled GPU utilization, and peak
memory. Evaluation protocol files under `results/` reference those immutable
training records rather than duplicating them.
