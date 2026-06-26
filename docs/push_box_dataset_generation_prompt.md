# Prompt: Generate A Larger Push-Box Friction Dataset

Use this prompt for a separate agent that can edit and run the FastWAM/Libero data generation pipeline.

```text
We need a larger calibrated push-box video dataset for BWM physical-property adaptation.

Goal:
Generate hidden push-box episodes where friction is the environment-level property. The dataset must support test-time adaptation: given 1-3 support rollouts from one friction value, the model should improve prediction on other held-out rollouts with the same friction but different poses, push lengths, and push directions.

Target dataset size:
- 8 friction levels.
- For each friction level:
  - 25 straight-push episodes.
  - 25 angled-push episodes.
- Total: 8 * (25 straight + 25 angled) = 400 episodes.

Friction levels:
Use 8 clearly separated but physically plausible table friction coefficients. Include the previous bins as anchors if possible:
- 0.005
- 0.01
- 0.02
- 0.035
- 0.05
- 0.08
- 0.12
- 0.2

Trajectory diversity requirements:
For every friction level and every split type (straight/angled), vary:
- initial object XY position;
- robot pre-push pose;
- push direction;
- push speed / impulse;
- push distance;
- contact point / lateral offset;
- object orientation if the simulator supports it.

Straight episodes:
- Push direction should be mostly aligned with the object/table reference axis.
- Still randomize lateral offset, start pose, speed, distance, and object position.

Angled episodes:
- Use multiple angles, not a single angle.
- Suggested angle buckets: -30, -20, -10, +10, +20, +30 degrees.
- Balance positive and negative angles as evenly as possible.

Push displacement mix:
Keep a short/mid/long distribution so the dataset contains both low-motion and high-motion outcomes.
Suggested per 25 episodes in each friction/split group:
- 7 short pushes: low speed or short distance, small box displacement.
- 11 medium pushes: nominal speed/distance.
- 7 long pushes: higher speed or longer distance, larger displacement.

Episode structure:
- Preserve the current phase structure:
  - approach
  - descend
  - push
  - settle / post-push
- Ensure the push phase is visible around frames 50-75 when possible.
- Avoid episodes where the robot occludes the object for most of the push.
- Avoid invalid runs where the object falls off the table, clips through geometry, or has no visible motion unless it is an intentional high-friction short-push case.

Video and action outputs:
Match the existing BWM/FastWAM data layout:
- videos/chunk-000/observation.images.image/episode_XXXXXX.mp4
- videos/chunk-000/observation.images.wrist_image/episode_XXXXXX.mp4
- data/chunk-000/episode_XXXXXX.parquet
- meta/push_box_episode_metadata.jsonl

Required metadata per episode:
- episode_index
- split: "straight" or "angled"
- friction_mu
- angle_deg
- pair_id / case_id
- steps
- phase_counts
- push_start / derivable push start
- push_steps
- push_end / derivable push end
- any randomized control parameters:
  - initial object pose
  - push distance bucket: short/mid/long
  - commanded push distance
  - push speed / impulse scale
  - contact offset
  - robot start pose

Train/test split requirement:
Do not create train/test by slicing sequential episodes. Randomize within each friction/split group so each group has both train and test episodes.
Recommended split:
- 50% train, 50% test per friction/split group.
- Maintain roughly the same short/mid/long ratio in both train and test.

Quality checks:
After generation, print a summary table:
- count by friction_mu and split;
- count by displacement bucket and split;
- min/mean/max episode length;
- phase count statistics;
- number of invalid or filtered episodes.

Also generate a lightweight visual contact sheet:
- 1-2 frames from each friction/split group;
- include straight and angled examples;
- include short, mid, and long examples.

Deliverables:
1. The generated dataset directory.
2. Updated generation script/config.
3. The metadata summary table.
4. A short note explaining any rejected or regenerated episodes.
```

