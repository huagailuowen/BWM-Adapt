# Event80 C32 iteration-frequency ablations

These runs change the C/model alternation frequency, not curriculum wave length
or the number of model/C optimizer updates per wave. They do not use joint
training. The original default branch is unchanged unless the new frequency
argument is explicitly supplied.

| Setting | 100/100 | 400/400 |
|---|---|---|
| Initial phase | Model-only 300 updates, including 100-update model LR warmup | Same |
| Each curriculum wave | New-C 200, then (all-C 100, model 100) x4 | New-C 200, then all-C 400, model 400 |
| Wave duration | 1000 updates | 1000 updates |
| Wave C/model budgets | New-C 200 + all-C 400 + model 400 | Same |
| Activation | Initial 5, then add 5 per wave; same 35 as job 88823 | Same |
| After step 7300 | Alternate all-C 100 / model 100 | Alternate all-C 400 / model 400 |
| Latent | C32, independent U(0,1), MLP hidden width 128 | Same |
| Learning rates | Model 1e-5; all-C 0.03; new-C 0.15 | Same |
| Batch | 2 GPUs x 4 environments x 4 clips = 32 | Same |
| Data | Event80, frames 65-105, independent-window sampling | Same |
| Compute | Two GPUs, 24-hour allocation | Same |
| Checkpoints | Existing paired model/C saving, every 200 updates or 60 minutes, keep-last-two policy | Same |

The underlying `all_context_steps=200` and `model_steps=200` remain unchanged:
they specify the legacy two-block budgets. The opt-in
`grouped_context_curriculum_alternation_steps` repartitions those budgets into
100- or 400-update blocks. Changing those two legacy fields alone would
incorrectly change the duration of the curriculum wave.

Frozen parameters retain the existing requires-grad and optimizer-state
handling. Optimizer moments are not reset at phase boundaries. Model
initialization, latent initialization, data order seed, activation membership,
and support/query evaluation protocol remain unchanged. Shared node-local Wan
and BLM caches are reused, with immutable per-job launch configs and separate
output directories. H200 and B200 remain eligible through the existing generic
two-GPU high-priority launcher; no individual node is pinned.

These are additional ablations, not replacements for the existing joint,
no-curriculum, dimension, or initialization experiments. Do not insert scores
into the formal table before training and matched-protocol inference finish.
