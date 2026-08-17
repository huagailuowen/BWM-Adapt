# TTT-KVB baseline

The canonical name is **TTT with key-value binding (TTT-KVB)**. `TTT-KQV` is
kept only as a compatibility alias. Q is a read view; it is not part of the
inner write target.

## Algorithm

For support tokens `x`, the outer-loop projections produce `k = theta_K(x)`
and `v = theta_V(x)`. The per-environment fast state `W` is updated with the
MSE binding objective

```text
L_inner(W; x) = 0.5 * ||k + LN(MLP_W(k)) - v||^2.
```

For a disjoint query, `q = theta_Q(x_query)` reads the fixed adapted state:

```text
memory(query) = q + LN(MLP_W_adapted(q)).
```

The query flow-matching loss backpropagates through support-time writes. Query
tokens never update `W`. A fresh environment resets every replaced block to
the shared learned initialization `W0`.

## Wan integration

- Existing Wan self-attention remains unchanged.
- TTT is an independently projected gated parallel branch.
- Support is `write_only` and cannot change backbone activations.
- Query is `read_only`; the same state can serve a fixed set of query actions.
- Fast-state arithmetic is FP32; the Wan backbone can remain BF16.
- Activation checkpointing is disabled only during support writes. Query
  checkpointing is safe because the state is read-only and remains active
  through backward recomputation.

The Event80 default uniformly selects 512 tokens over the flattened 41-frame
Wan token sequence and applies 64-token inner mini-batches. This bounded token
stream is an explicit video-scale engineering choice, not a claim that the
language-model implementation processed only 512 tokens.

## Source alignment

- Original TTT formulation and TTT-MLP:
  <https://arxiv.org/abs/2407.04620>
- Official PyTorch reference:
  <https://github.com/test-time-training/ttt-lm-pytorch>
- Support-prefill/frozen-query-read precedent for N-dimensional data:
  <https://arxiv.org/abs/2505.23884>
- Updated interpretation and canonical TTT-KVB terminology:
  <https://arxiv.org/abs/2602.21204>

The 2026 analysis shows that KVB variants can often be rewritten as learned
linear attention. Therefore evaluation must report downstream query metrics;
inner binding-loss convergence alone is not evidence that physical state was
memorized.
