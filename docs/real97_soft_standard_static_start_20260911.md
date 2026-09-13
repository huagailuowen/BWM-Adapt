# Soft Standard: stationary episode-start history ablation

Use the same step-5500 Standard Soft checkpoint and the same 60 episodes
(30 train, 30 test; fifteen environments) as the frozen reference inference.
This opt-in run resets every query to episode frame zero. The source dataset's
initial stationarity is supplied by the user, not inferred from detector
thresholds. No dataset is modified and no replacement episode is sampled.

Native frame indices for the 33-frame model input/target:

```text
observed:   0, 0, 0, 0, 0
predicted:  3, 6, 9, ..., 84
```

Read video and target EEF through the same training operators and frozen
training-only normalization. `HistoryPrefixDataset` prepends four copies of
frame zero to the first 29 strided video/action samples. Thus the entire
observed prefix is one fixed image and one fixed target EEF command, not zero
action channels and not a glimpse of recorded motion. Subsequent actions are
unchanged recorded targets at native frames 3,6,...,84. Short tails retain the
existing last-frame padding policy and are excluded from prediction metrics.

Raw GT and Standard videos contain 33 frames, including the four artificial
history repeats. Comparisons and per-environment grids omit those repeats:
they display 29 frames, native anchor zero plus 28 predictions, at 20/3 fps.
Per-query comparison metric indices exclude the anchor and padded tail.

History conditioning stays clean during all denoising steps. The old
one-frame and five-frame inference protocols remain the defaults when the
new command-line flag is absent. No support adaptation or latent fitting is
performed by this Standard baseline. The new run uses a separate output
directory, low-priority single GPU, and the shared node-local model cache.

Because anchors differ from the earlier middle-episode queries, this is not
a same-anchor causal estimate of removing history. It is the requested test
of predictive performance when every clip starts from the recorded resting
state. All future-frame comparisons use the matching factual GT.

```bash
sbatch --job-name=std97-soft-static0 scripts/evaluation/run_real97_soft_standard_static_start_gpu.sh
```
