# Soft initialization comparison

Run six low-priority CPU tracking shards and a dependent CPU aggregate.
Reuse the existing GroundingDINO/filter/tracker without changing thresholds.
Reuse GT/Standard/Stage1 tracks only after row/configuration and source-video hash agreement.
Keep all old inference and evaluation outputs intact.
Compare identical 36 factual queries (18 train, 18 test), native frames 3..84.
Report natural coverage and errors, common-mask ADE across all methods, and pair-common Stage2 errors.
FDE means the actual final eligible future frame, never the last detected point.
Stage2 family initialization uses known L/R/8 prior with 1R in R. Standard's three unseen training environments remain a separate cohort.
Output five-row videos/grids and annotated contact sheets for manual inspection; scoring remains provisional.
