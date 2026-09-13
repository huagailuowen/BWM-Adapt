# Stick balance: phase-referenced interval evidence

This opt-in CPU experiment does not change training, previous detectors, or the
formal results directory. Its output is not yet an approved action metric.

## Identity and reference

Left and right are distinct identities, confined to their respective image
halves. The low-level marker detector takes integer IDs 0 and 1, not strings.
Reference sites, tracked sites, and projected contact proxies are checked.
The reference is the last available frame before commanded lift. A command
boundary is not itself a visually annotated contact boundary.

The previous reference segmentation can reject a fully visible box when a local
ROI touches the connected component. This branch retains actual clipping as
unknown, but can use a lower-marker footprint with an explicit 20-pixel reference
geometry uncertainty when that footprint remains inside the proper image half.
The fallback is labeled as a proxy, never as an exact physical contact contour.

## Error propagation and decisions

For a fixed reference point p with localization error e and fitted similarity A,
the displacement error is (A-I)e. Pure translation does not repeatedly incur the
same static contour-location error. Registration errors and chained-anchor
errors are still retained. A scalar upward-motion projection n therefore has
geometry error bounded by norm(n^T(A-I)) times the reference error radius.
These are engineering error bounds, not statistically calibrated confidence
intervals. Generated query videos have a higher registration-error floor.

The candidate gates remain 1 native pixel, rise ratio 0.5, and 0.3 seconds.
Positive evidence requires BOTH lower rise bounds to clear the absolute gate
and the smaller lower bound to exceed 0.5 times the larger upper bound.
Strong asymmetry can be identified even when the stationary end overlaps zero:
its upper bound must be below 0.5 times the rising end's lower bound.
Uncertain intervals are not silently counted as failures or removed from the
whole-video temporal decision. Tracking coverage is reported separately from
decision coverage and is not an accuracy estimate.

## Audit

Run `sbatch scripts/evaluation/run_stick_phase_interval_flow_cpu.sh` on CPU Slurm.
It evaluates the same 238 full GT episodes and 40 GT/Stage1/Stage2 query triplets
as job 114226. Detailed JSONL retains per-frame identities, geometry provenance,
registration and geometry error terms, and bounds on the rise and rise ratio.

Fixed-coordinate sheets include untouched crops alongside tracking overlays.
Their crop coordinates do not follow each object's motion. Consecutive-frame
sheets inspect a sustained positive/negative interval without time subsampling.
Image-plane movement can still include depth translation, camera motion, or
unmodeled 3D rotation. In particular, tabletop sliding must not be certified as
vertical lift based solely on marker flow. Visual audit remains mandatory;
borderline physical contact remains unresolved until supported by evidence.

While a job is pending/running, the assistant checks personally every ten
minutes. No monitoring script is used. Do not publish to `results` until the
tracking and action judgment have passed review.
