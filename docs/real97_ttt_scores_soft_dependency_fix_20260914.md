# Real97 TTT scores and Soft inference dependency repair

TTT Door step3656/job116108 and Ball step4144/job116109 are compared on exactly
the 216 factual queries of the frozen v2 baseline scoreboard. Reuse the same GT,
native geometry, padding masks, marker detector, exit handling, and one-sided
Door closure policy. Reject query-set and temporal/action metadata mismatches.
Only TTT predictions are newly decoded/tracked/scored; baseline scores stay intact.

CPU compute node: PSNR, SSIM, native-coordinate object tracks, ADE/FDE, coverage,
Ball peak-x error, Door closure, and action decisions. GPU compute node: LPIPS.
Both stages use a separate output directory and resume completed query records.

Ball action score scans levels upward for first maximum-x reaching blue-marker-x
minus 30 pixels, independently for GT and prediction, then compares first-level
and successor pairs by intersection/2. A successor of level10 is the formal
level11 label, not an observed action. A confirmed screen exit retains last
visible center. Missing predictions and incomplete GT are exposed explicitly.

Door scans upward for the first adjacent pair both classified closed through
the final 0.3 seconds, then intersects the pair with the unchanged nominal GT
pair. Exclude the impossible door environment from action scoring only. Keep
the existing door-4d calibration caveat and held-out visual-audit status.

comparison_with_baselines.json contains all methods, train/test separately,
with query means, environment-macro means, coverage, action decisions, and the
actual checkpoint steps. Do not claim equal training duration or initialization
information; historical Ours includes known-environment starts. No formal results
or previous experiment directories are overwritten.

Soft retry: install opencv-python-headless==4.11.0.86 into .venv using uv with
--no-deps --offline. This is the version already used in the CPU evaluation venv;
do not upgrade Torch or NumPy. Relaunch only the failed Soft inference, using
its existing job116188 output directory and protected step5500 model/table.
No new source-model changes or inference-policy changes are made.
