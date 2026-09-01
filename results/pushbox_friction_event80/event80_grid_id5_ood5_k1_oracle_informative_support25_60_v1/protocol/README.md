# Event80 oracle informative-support protocol

This benchmark preserves the fixed 5-ID/5-OOD environment set but chooses one method-independent support from all ten ground-truth actions in each environment. It is therefore an oracle demonstration-selection benchmark, not a strict online K=1 benchmark. The selected support is excluded from the nine read-only query trajectories, and every method consumes the same manifest.

## Legacy implementation labels

The existing scores remain valid for the exact checkpoints and implementations evaluated here. They must not be presented as results from the revised baselines introduced after this benchmark run.

| Existing result | Implementation evaluated | Revised method not yet represented by this result |
| --- | --- | --- |
| `ttt_kqv/step_12400` | Prequential read-then-write TTT, eight selected blocks, 512 sampled tokens per chunk | Forward-only One-Minute-style write-then-predict TTT on all blocks and all global tokens |
| `dinov2_amortized_context/step_11500` | Eight-frame action-conditioned transition encoder with temporal mean pooling | Eleven-anchor Wan-aligned temporal-attention DINO encoder |
| `history_conditioned_wm/step_26200` | Flamingo Perceiver Resampler and gated cross-attention memory | Native Prefix-History WM with an explicit reset latent group |
