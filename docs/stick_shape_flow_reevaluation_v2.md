# Reference-shape terminal reevaluation

This independent review branch reevaluates the existing 238 native GT episodes
and 40 GT/Stage1/Stage2 query triplets. No new model inference or training is
required. Old configs, evaluators, videos, and evaluation outputs stay intact.

## Criterion and reference

Only the final 0.3 source-clock seconds of the commanded lift are scored.
Conditioning frames and padding are not evaluated. Both ends must actually
clear the table with no large tilt. A 1:2 rise ratio is not required.
Three- and five-degree image-plane thresholds are reported as sensitivity
variants pending visual calibration, not as certified physical success labels.

For query triplets, the exact same GT conditioning frame is used as the resting
reference for every method. Generated pre-lift drift must not redefine the
tabletop baseline. Native GT episodes use their actual pre-lift resting frame.

## Shadow separation

The reference body is segmented with marker seeds and local foreground/background
appearance. Optical flow is restricted to material features inside that body.
Its shape is warped into each terminal frame. A narrow shape neighborhood
constrains another segmentation, while surrounding pixels, including shadows,
provide background evidence. Clearance uses the current lower silhouette rather
than just marker centers. Partial geometry, failed tracking, and bodies touching
the search neighborhood boundary cannot certify success.

The visible rod edges provide a separate angle measurement. Rise ratios are not
used to reject success. All results remain diagnostic until reviewed: the local
floor projection assumes a fixed camera, and image-plane measurements alone
cannot certify hidden corners or calibrated three-dimensional clearance.

Submit `scripts/evaluation/run_reevaluate_stick_shape_flow_cpu.sh` with `sbatch`.
Bulk video processing runs on a CPU compute node. Outputs include per-video
terminal overlays, clearance/error records, method counts, and paired GT versus
prediction outcomes under both sensitivity variants. Unknown outcomes remain
separate from success and failure, and no formal `results` are published yet.
