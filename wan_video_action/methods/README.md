# Method layer

This package is an additive home for method-level contracts and definitions.
It does not replace, move, or dispatch the existing scripts. Until an explicit
migration is approved, the current training and inference scripts remain the
canonical executable implementations.

The package separates three concepts:

- `ours/`: the persistent environment-memory method.
- `baselines/`: independently meaningful comparison methods.
- `ablations/`: controlled variants of Ours that remove or alter one claim.

`protocol.py` defines the grouped-training and disjoint support/query contracts.
`registry.py` exposes a read-only catalog of the built-in method definitions.
Future runners should implement `MethodRunner` and preserve each definition's
query-state policy. In particular, a `read_only` query may consume a state
written by support trajectories, but it must not update that state.

Ablations that differ only by configuration should stay configuration-only.
A Python implementation belongs here only when the training or adaptation
control flow differs from the parent method.

Method comparisons use a fixed hardware-time envelope rather than forcing equal
steps, clip exposure, or FLOPs. `configs/methods/<benchmark>/common.yaml` fixes
the exact GPU SKU, count, timer boundary, and wall-clock duration. Actual steps,
clips, throughput, utilization, and peak memory are reported. Configuration-only
planning is available through `scripts/methods/plan_matrix.py`; it never submits
a scheduler job.
