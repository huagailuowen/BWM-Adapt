# Additive method entry points

These entry points do not import, move, or replace the legacy experiment
scripts. Every config can be validated with dry-run immediately. Real execution
requires an approved method-local factory under the runtime section.

Example:

    PYTHONPATH=. .venv/bin/python scripts/methods/train_method.py \
      --config configs/methods/event80/baselines/standard_pooled_wm.yaml --dry-run

The factory boundary is intentional. It prevents an unfinished baseline from
silently changing the current Wan training or inference behavior.
