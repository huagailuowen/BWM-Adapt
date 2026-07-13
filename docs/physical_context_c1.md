# One-dimensional physical context for 6-friction push-box

This experiment uses a scalar physical context `C` for friction.

Stage1 is an oracle pretraining setting:

```text
C = normalize_push_box_friction_mu(friction_mu)
```

Stage2 is the realistic TTT setting:

```text
C starts from physical_context_encoder.default_context
inner loop updates C using support/show loss
query uses the adapted C
```

The fixed normalization is linear over the expected push-box friction range:

```text
mu_min = 0.0
mu_max = 0.25
C = mu / 0.25
```

This maps the six training frictions approximately to:

```text
0.005 -> 0.020000
0.010 -> 0.040000
0.020 -> 0.080000
0.050 -> 0.200000
0.100 -> 0.400000
0.150 -> 0.600000
0.200 -> 0.800000
0.250 -> 1.000000
```

Stage2 should initialize the learned `default_context` at `0.5`, the middle of
this range, unless a checkpoint already provides a trained default context.

For one-dimensional C, the physical-context MLP intentionally skips input
`LayerNorm`; `LayerNorm(1)` would erase the scalar value.

The implementation is `normalize_push_box_friction_mu` in:

```text
wan_video_action/models/physical_context.py
```

The default Stage1 oracle metadata uses this normalization:

```text
data/push_box_bwm_6fric_50pairs_straight_jitter_35_35_hidden_straight_8chunk_overlap/train_oracle_mu025_c1.jsonl
```

The realistic Stage2 metadata intentionally does not contain `physical_context`:

```text
data/push_box_bwm_6fric_50pairs_straight_jitter_35_35_hidden_straight_8chunk_overlap/train.jsonl
```
