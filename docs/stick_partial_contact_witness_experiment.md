# Partial visible endpoint evidence, opt-in only

Job 114250 inspected reference frames for 109 episodes/query videos rejected by
the full-footprint branch. Every rejection involved the horizontal proxy margin;
81 endpoint observations had a detected blue marker touching the original image
edge. These are not all failures of color localization. Many left boxes really
are cropped by the original camera image.

This branch retains the observable marker span instead of discarding all optical
flow for a partially cropped box. It does not move an unseen foot into the frame,
invent missing pixels, or allow left/right identity swaps. Partial-reference
observations retain a 20-pixel geometry-error radius and their own provenance.

Partial geometry is asymmetric evidence: a reliable non-rising visible contact
proxy can oppose a positive lift, but cannot certify the entire box is lifted.
For a rise-ratio rejection, the larger-end lower bound must come from a fully
observed footprint. A partly observed endpoint cannot certify that lower bound.
Success requires both complete geometries; otherwise retain
`unknown_partial_geometry`. Unknown intervals still block an unsupported negative
whole-video conclusion. The absolute, ratio, and duration thresholds are unchanged.

This remains an image-motion proxy under bounded reference-geometry assumptions,
not a certified physical-contact detector. Depth motion, unseen corners, incorrect
blue-marker association, and out-of-plane rotation require continued audit.

Launch with `scripts/evaluation/run_stick_phase_partial_witness_cpu.sh`. It sets
`STICK_PHASE_FLOW_CONFIG` for this process only. Existing configurations and the
default interval-flow entrypoint keep the previous full-geometry-only behavior.
All debug outputs stay in `outputs`; no formal metric is approved by this change.
