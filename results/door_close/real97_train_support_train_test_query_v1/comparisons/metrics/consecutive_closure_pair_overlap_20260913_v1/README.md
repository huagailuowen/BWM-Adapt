# door_close: consecutive_closure_pair_overlap_20260913_v1

This is a text-only snapshot of the existing predictions and computed metrics. No inference, video decoding, or model training is run by the exporter.

## Score definitions

Scan levels upward and choose the first adjacent pair for which both predictions close the door. Score the intersection with the fixed user-specified GT pair / 2. door-12u-Half-11d-Half is excluded from action selection, but remains in image and object metrics. The held-out door-4d level-3 video is visibly open; its measured GT pair is {4,5}, while the agreed nominal scoring pair remains {3,4}. This known sample/protocol discrepancy is retained, not silently relabeled.

Image/object metrics keep train and held-out test queries separate. The main scoreboard uses equal environment weights on held-out queries. Conditioning frames, repeated tail padding, and letterbox bars are excluded as recorded in the frozen evaluation manifest. Counterfactual action sweeps have no paired GT and are not included.

## Files

- `scoreboard.csv` / `scoreboard.json`: held-out image, object, and action results.
- `image_object_summary.json`: query means, environment means, and valid counts.
- `action_decisions.json`: per-environment action-selection decisions.
- `object_observations.jsonl`: observed targets, peak positions, and closure labels.
- `frozen_query_manifest.jsonl`: query identities, frame masks, and source references.
- `protocol.json` / `provenance.json`: conventions, caveats, source paths, hashes.
- Per-method per-query metrics are under the evaluation's `methods/` directory.

## Limitations

Source audit status: `train_gt_calibrated_pending_heldout_overlay_audit`. Archiving does not upgrade this status or imply an exhaustive manual audit of every held-out tracking result.

Ours Stage2 uses the existing customized supports and initializations, including near-environment-table starts. This is not an equal-information or equal-training-step benchmark. Door Ours uses step 3715, while Door baselines use step 5500; all Ball methods use step 5500.

Only text metrics and their reproducibility records are published. Videos, images, model weights, latent checkpoints, logs, and dataset files are excluded.
