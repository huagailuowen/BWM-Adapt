# Simulation Task-State Extraction

Use `sim_rgb_v1` for the current gravity, mass-collision, joint
mass-friction, mass-balance, and LightSwitch rollouts. The legacy mask/luma
paths remain unchanged and are selected by default.

```bash
uv run --no-sync python scripts/evaluation/extract_task_state.py \
  --extractor sim_rgb_v1 \
  --task gravity \
  --video-path path/to/prediction.mp4 \
  --output path/to/task_state.npz \
  --audit-video path/to/task_state_audit.mp4
```

## State definitions

| Task | Primary state | Additional state |
|---|---|---|
| Gravity | Blue-object centroid | Exit side and time |
| Mass collision | Red struck-object centroid | Blue striker centroid; exits |
| Joint mass-friction | Red pushed-object centroid | Blue driver centroid; exits |
| Mass balance | Canonicalized bar-axis angle | Bar centroid |
| LightSwitch | Yellow-pixel fraction in lamp ROI | Binary lamp state |

The raw Wan rollout is a horizontal main/wrist concatenation. `sim_rgb_v1`
always measures the left `224x224` main view unless `--main-view-width` is
overridden.

Colour-component association rejects frame-to-frame jumps larger than 64
pixels by default. This prevents a small same-colour robot or background region
from taking over an object's identity after it exits the image. The threshold
is deliberately above the observed rollout maximum of 29 pixels and may be
overridden with `--max-tracking-jump-px` for a separately audited dataset.

When a tracked object leaves the image, its centroid is held continuously at
its final observed position. `event_offscreen` and exactly one of `event_exit_left`,
`event_exit_right`, `event_exit_top`, or `event_exit_bottom` are set. This keeps
distance finite without inventing post-exit motion. An unexplained interior
detector loss remains missing and receives the configured missing-track penalty.

LightSwitch uses the default main-view ROI `(98,108)-(151,166)`. The score is
the fraction of yellow pixels, with a default on/off threshold of `0.35`.
Use `--roi` or `--yellow-threshold` only for a separately calibrated camera.

Every formal metric run should retain a small sample of `--audit-video`
outputs so object identities, screen exits, bar axes, and lamp states remain
visually auditable.
