# Soft shared-six: visual spot-check audit

Reviewed six of the twelve test queries, one per environment, selected by first query index rather than performance. Each contact sheet has GT / Standard / Stage1 / Ours Stage2 / DINO at native frames 0,27,54,84: 120 panels in total.

## Findings

No obvious wrong-object detections were found in the sampled panels. The boxes and center crosses follow the silver endpoint rather than the robot, blue markers, string, or background. Reviewed final-frame detections are on the depicted object for all methods. Boundary-clipped objects use the visible-box center, per the accepted protocol.

- [q0002_soft-1l_test_ep000004](/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/eval_real97_soft_dino_matched_20260916_v1/cases/q0002_soft-1l_test_ep000004/all_methods_contact_sheet.jpg): All sampled boxes remain on the silver endpoint. Some lower-edge clipping is visible; center follows the visible box. Predicted endpoint differences remain visible independently of the detector.
- [q0006_soft-2l_test_ep000001](/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/eval_real97_soft_dino_matched_20260916_v1/cases/q0006_soft-2l_test_ep000001/all_methods_contact_sheet.jpg): All methods track the silver endpoint at the sampled times, including partly clipped lower-edge observations. No confusion with the blue markers.
- [q0014_soft-1r_test_ep000002](/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/eval_real97_soft_dino_matched_20260916_v1/cases/q0014_soft-1r_test_ep000002/all_methods_contact_sheet.jpg): The GT endpoint moves to the right and upward by frame84; the model endpoints remain lower. Sampled boxes follow the actual depicted objects, not the GT trajectory.
- [q0022_soft-5r_test_ep000005](/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/eval_real97_soft_dino_matched_20260916_v1/cases/q0022_soft-5r_test_ep000005/all_methods_contact_sheet.jpg): GT and predictions move right. Sampled detector boxes remain on the silver object; early lower-edge clipping does not produce a switch to the robot.
- [q0026_soft-2m_test_ep000014](/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/eval_real97_soft_dino_matched_20260916_v1/cases/q0026_soft-2m_test_ep000014/all_methods_contact_sheet.jpg): The small endpoint starts near the left blue marker but the boxes identify the endpoint, not the marker. At frame84 boxes follow the silver object, including the partly clipped GT/Stage1 views.
- [q0034_soft-8_test_ep000005](/hai/scratch/cyzhou05/projects/TTT-Physics/repos/BWM-Adapt/outputs/eval_real97_soft_dino_matched_20260916_v1/cases/q0034_soft-8_test_ep000005/all_methods_contact_sheet.jpg): All sampled boxes identify the endpoint through leftward motion and the return segment. Lower-edge clipping and generated blur are present; no obvious object-identity switch.

## Scope and metric status

No sample was removed; no tracking threshold or metric value was changed. This is a sparse visual audit, not exhaustive frame-by-frame verification or a manual localization-error benchmark. Motion between reviewed timestamps, detector misses, and small box-center offsets are not ruled out. The metrics remain provisional, with the sampled visual audit now complete.

