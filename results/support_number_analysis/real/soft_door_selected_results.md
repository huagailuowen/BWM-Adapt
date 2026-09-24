# Selected Soft and Door support-number results

Frozen by user on 2026-09-22. No new inference is scheduled.

| Task | K | Action (%) | PSNR | SSIM | LPIPS | Object mean error (px) | Object final error (px) |
|---|---:|---:|---:|---:|---:|---:|---:|
| door | 1 | 44.44444 | 31.48633 | 0.92263 | 0.04022 | 12.70876 | 22.81000 |
| door | 2 | 72.22222 | 32.07019 | 0.92660 | 0.03361 | 9.01027 | 15.75000 |
| door | 4 | 72.22222 | 32.42222 | 0.92875 | 0.03096 | 7.80200 | 11.65000 |
| soft | 1 | 81.81818 | 32.52206 | 0.96132 | 0.07583 | 18.71006 | 23.58782 |
| soft | 2 | 81.81818 | 32.57435 | 0.96142 | 0.07535 | 18.36283 | 22.74540 |
| soft | 4 | 81.82 | 32.58 | 0.96143 | 0.07534 | 18.06 | 22.59 |

Soft K=4 values are the existing rounded formal-table values. Other entries use full-precision metric summaries.

Door K=4 uses the explicitly selected revised supports. This was a post-hoc support revision after examining failures, not a preselected support-count-only comparison. Its full image/object/action metrics are switched together. Original K=4 results remain in `door/k4/`; revised results are in `support_level_revision_20260922/door/k4/`.

Door uses the formal lowest-predicted-closing-level rule (exact=1, adjacent=0.5), not the legacy consecutive-pair selection. Soft keeps the shared6/test12 cohort and the unchanged 6-degree onset score on 11 positive queries.

`selected_runs.json` freezes the source choice for subsequent aggregation. No main-table K=2 Door or K=4 Soft baseline is changed.
